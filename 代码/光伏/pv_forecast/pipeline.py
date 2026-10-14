"""Training and inference orchestration for the independent PV package."""

from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Iterable, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from .baselines import BaselineForecaster
from .data import (
    InputSpec,
    RegularizedData,
    WindowedData,
    build_windows,
    load_and_regularize,
    split_boundaries,
    feature_frame,
)
from .errors import InsufficientHistoryError, ModelUnavailableError
from .metrics import (
    apply_prediction_interval,
    calibrate_prediction_interval,
    regression_metrics,
)
from .models import GRUConfig, GRUForecaster
from .weather import WeatherForecastArchive, solar_elevation
from .records import build_forecast_record, validate_forecast_record
from .visuals import (
    generate_actual_vs_forecast_visualization,
    generate_forecast_visualizations,
    generate_training_evaluation_visualizations,
)


@dataclass
class TrainingConfig:
    interval_minutes: int = 15
    history_days: int = 3
    horizon_steps: int = 96
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    nominal_coverage: float = 0.80
    min_gru_days: int = 60
    gru_epochs: int = 50
    gru_batch_size: int = 128
    gru_patience: int = 8
    gru_units: int = 32
    gru_learning_rate: float = 0.0003
    gru_dropout: float = 0.2
    reduce_lr: bool = True
    lr_patience: int = 2
    lr_factor: float = 0.5
    min_lr: float = 0.00001
    min_delta: float = 0.00001
    max_gru_samples: Optional[int] = None
    skip_gru: bool = False
    require_gru: bool = False
    seed: int = 42
    model_id: str = "pv-power-forecast"
    model_version: str = "0.2.0-candidate"
    interval_conditioning: str = "pooled"
    night_lower_bound: str = "empirical"


def _fit_scalers(train: WindowedData):
    target_scaler = MinMaxScaler()
    target_scaler.fit(train.target.reshape(-1, 1))
    feature_scaler = MinMaxScaler()
    feature_scaler.fit(train.future_features.reshape(-1, train.future_features.shape[-1]))
    return target_scaler, feature_scaler


def _fit_history_scaler(train):
    return MinMaxScaler().fit(train.gru_history.reshape(-1, train.gru_history.shape[-1]))


def _gru_config(config):
    return GRUConfig(units=config.gru_units, epochs=config.gru_epochs,
        batch_size=config.gru_batch_size, patience=config.gru_patience, seed=config.seed,
        learning_rate=config.gru_learning_rate, dropout_rate=config.gru_dropout,
        reduce_lr=config.reduce_lr, lr_patience=config.lr_patience, lr_factor=config.lr_factor,
        min_lr=config.min_lr, min_delta=config.min_delta)


def _scale_windows(data: WindowedData, target_scaler, feature_scaler, history_scaler=None) -> WindowedData:
    if data.size == 0:
        return data
    hist = target_scaler.transform(data.history_power.reshape(-1, 1)).reshape(data.history_power.shape)
    target = target_scaler.transform(data.target.reshape(-1, 1)).reshape(data.target.shape)
    future = feature_scaler.transform(
        data.future_features.reshape(-1, data.future_features.shape[-1])
    ).reshape(data.future_features.shape)
    past = None if history_scaler is None else history_scaler.transform(
        data.gru_history.reshape(-1, data.gru_history.shape[-1])).reshape(data.gru_history.shape)
    return WindowedData(hist, future, target, data.forecast_starts, data.feature_names,
                        past, data.history_feature_names, data.weather_issued_at)


def _inverse_target(values: np.ndarray, scaler) -> np.ndarray:
    values = scaler.inverse_transform(np.asarray(values).reshape(-1, 1)).reshape(np.asarray(values).shape)
    return np.maximum(values, 0.0)


def _candidate_baselines(train: WindowedData, val: WindowedData, horizon: int):
    candidates = []
    for strategy in ("persistence", "previous_day", "same_slot_mean"):
        model = BaselineForecaster(strategy, steps_per_day=96)
        prediction = model.predict(val.history_power, horizon)
        score = regression_metrics(val.target, prediction)["rmse"]
        candidates.append((float(score) if score is not None else float("inf"), model, prediction))
    return min(candidates, key=lambda item: item[0]), candidates


def _limit_windows(data: WindowedData, maximum: Optional[int]) -> WindowedData:
    """Deterministically cap GRU training windows without changing validation."""

    if maximum is None or maximum <= 0 or data.size <= maximum:
        return data
    indices = np.linspace(0, data.size - 1, maximum, dtype=int)
    return WindowedData(
        data.history_power[indices],
        data.future_features[indices],
        data.target[indices],
        [data.forecast_starts[i] for i in indices],
        data.feature_names,
        None if data.history_features is None else data.history_features[indices],
        data.history_feature_names,
        [data.weather_issued_at[i] for i in indices] if len(data.weather_issued_at) else [],
    )


def _daylight_mask(starts, horizon: int, interval_minutes: int, latitude=None, longitude=None) -> np.ndarray:
    mask = np.zeros((len(starts), horizon), dtype=bool)
    for row, start in enumerate(starts):
        timestamps = pd.date_range(
            start=pd.Timestamp(start),
            periods=horizon,
            freq=f"{interval_minutes}min",
        )
        mask[row] = ((timestamps.hour >= 5) & (timestamps.hour <= 20) if latitude is None
                     else solar_elevation(timestamps, latitude, longitude) > 0)
    return mask


def _json_dump(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def train_model(
    source,
    output_dir,
    input_spec: InputSpec = InputSpec(),
    config: TrainingConfig = TrainingConfig(),
    progress=None,
    future_weather_source=None,
    selected_model_override=None,
) -> dict:
    """Train candidates and save an auditable candidate bundle."""

    np.random.seed(config.seed)
    if config.interval_conditioning not in {"pooled", "daylight"}:
        raise ValueError("interval_conditioning must be 'pooled' or 'daylight'")
    if config.night_lower_bound not in {"empirical", "zero"}:
        raise ValueError("night_lower_bound must be 'empirical' or 'zero'")
    if config.night_lower_bound == "zero" and config.interval_conditioning != "daylight":
        raise ValueError("zero night lower bound requires daylight conditioning")
    if config.require_gru and config.skip_gru:
        raise ValueError("require_gru and skip_gru cannot both be enabled")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    def emit(message):
        with (output / "training.log").open("a", encoding="utf-8") as log:
            log.write(message + "\n")
        if progress:
            progress(message)

    emit("读取数据并按时间顺序划分训练/验证/测试集……")
    regular = load_and_regularize(
        source, input_spec, interval_minutes=config.interval_minutes
    )
    archive = (WeatherForecastArchive(future_weather_source, input_spec.weather_columns, input_spec.timezone)
               if future_weather_source is not None else None)
    window_args = dict(target_col=input_spec.target_col, weather_columns=input_spec.weather_columns,
        history_weather_columns=input_spec.history_weather_columns, latitude=input_spec.latitude,
        longitude=input_spec.longitude, weather_archive=archive)
    train_end, val_end = split_boundaries(len(regular.frame), config.train_ratio, config.val_ratio)
    history_steps = config.history_days * (24 * 60 // config.interval_minutes)
    train = build_windows(
        regular, history_steps, train_end, config.history_days, config.horizon_steps,
        **window_args
    )
    val = build_windows(
        regular, train_end, val_end, config.history_days, config.horizon_steps,
        **window_args
    )
    test = build_windows(
        regular, val_end, len(regular.frame), config.history_days, config.horizon_steps,
        **window_args
    )
    if train.size == 0 or val.size == 0 or test.size == 0:
        raise InsufficientHistoryError(
            "clean train/validation/test windows are required; check data coverage and gaps"
        )
    split_info = {}
    for name, start, end, windows in (("train", 0, train_end, train),
                                       ("validation", train_end, val_end, val),
                                       ("test", val_end, len(regular.frame), test)):
        split_info[name] = {
            "start": regular.frame.index[start].isoformat(),
            "end": regular.frame.index[end - 1].isoformat(),
            "rows": end - start, "days": (end - start) * config.interval_minutes / 1440,
            "windows": windows.size,
            "first_forecast_start": windows.forecast_starts[0].isoformat(),
            "last_forecast_end": (windows.forecast_starts[-1] + pd.Timedelta(
                minutes=(config.horizon_steps - 1) * config.interval_minutes)).isoformat(),
        }
        emit(f"{name}: {split_info[name]['days']:g} 天，{end - start} 点，{windows.size} 个窗口 "
             f"[{split_info[name]['start']} → {split_info[name]['end']}]")
    _json_dump(output / "data_split.json", split_info)
    _json_dump(output / "training_config.json", asdict(config))

    val_daylight = _daylight_mask(val.forecast_starts, config.horizon_steps, config.interval_minutes,
                                  input_spec.latitude, input_spec.longitude)
    test_daylight = _daylight_mask(test.forecast_starts, config.horizon_steps, config.interval_minutes,
                                   input_spec.latitude, input_spec.longitude)
    calibration_mask = val_daylight if config.interval_conditioning == "daylight" else None

    # Baselines operate in physical kW and are always available.
    best_baseline, baseline_candidates = _candidate_baselines(train, val, config.horizon_steps)
    baseline_score, baseline_model, baseline_val_pred = best_baseline
    fallback_calibration = calibrate_prediction_interval(val.target, baseline_val_pred, config.nominal_coverage,
                                                        daylight_mask=calibration_mask, night_lower_bound=config.night_lower_bound)
    if selected_model_override and selected_model_override.startswith("baseline:"):
        strategy = selected_model_override.split(":", 1)[1]
        baseline_model = BaselineForecaster(strategy, steps_per_day=24 * 60 // config.interval_minutes)
        baseline_val_pred = baseline_model.predict(val.history_power, config.horizon_steps)
        baseline_score = regression_metrics(val.target, baseline_val_pred)["rmse"]
        fallback_calibration = calibrate_prediction_interval(val.target, baseline_val_pred, config.nominal_coverage,
                                                            daylight_mask=calibration_mask, night_lower_bound=config.night_lower_bound)
    selected_name = f"baseline:{baseline_model.strategy}"
    selected_val_pred = baseline_val_pred
    baseline_test_pred = baseline_model.predict(test.history_power, config.horizon_steps)
    selected_test_pred = baseline_test_pred
    gru_result = {"status": "skipped", "reason": "skip_gru=True"}
    gru = None
    gru_test_pred = None
    emit(f"最佳基线 {baseline_model.strategy}：验证 RMSE={baseline_score:.3f} kW")

    # GRU is optional and only attempted when enough daily history exists.
    days = train_end * config.interval_minutes / (24 * 60)
    if not config.skip_gru and days < config.min_gru_days:
        reason = f"训练分区仅 {days:g} 天，GRU 要求至少 {config.min_gru_days} 天"
        if config.require_gru:
            raise InsufficientHistoryError(reason)
        gru_result = {"status": "skipped", "reason": reason}
        emit(f"GRU 未训练：{reason}")
    elif config.skip_gru:
        emit("GRU 未训练：显式设置了 skip_gru=True")
    if not config.skip_gru and days >= config.min_gru_days:
        try:
            target_scaler, feature_scaler = _fit_scalers(train)
            history_scaler = _fit_history_scaler(train)
            train_scaled = _limit_windows(
                _scale_windows(train, target_scaler, feature_scaler, history_scaler),
                config.max_gru_samples,
            )
            val_scaled = _scale_windows(val, target_scaler, feature_scaler, history_scaler)
            test_scaled = _scale_windows(test, target_scaler, feature_scaler, history_scaler)
            gru = GRUForecaster(config=_gru_config(config))
            joblib.dump({"target_scaler": target_scaler, "feature_scaler": feature_scaler,
                         "history_scaler": history_scaler}, output / "scalers.joblib")
            emit(f"开始双输入 GRU：历史 {history_steps} 点 → 未来 {config.horizon_steps} 点；"
                 f"使用 {train_scaled.size}/{train.size} 个训练窗口，units={config.gru_units}，"
                 f"batch={config.gru_batch_size}，最多 {config.gru_epochs} 轮，早停 patience={config.gru_patience}。")
            gru.fit(
                train_scaled.gru_history,
                train_scaled.future_features,
                train_scaled.target,
                validation_data=(val_scaled.gru_history, val_scaled.future_features, val_scaled.target),
                target_scaler=target_scaler, output_dir=output, progress=emit,
            )
            # Always retain trained GRU weights, even if a baseline wins selection.
            gru.save(str(output / "gru.keras"))
            emit("用最佳权重计算完整验证集与留出测试集指标……")
            gru_val_scaled = gru.predict(val_scaled.gru_history, val_scaled.future_features)
            gru_test_scaled = gru.predict(test_scaled.gru_history, test_scaled.future_features)
            gru_val_pred = _inverse_target(gru_val_scaled, target_scaler)
            gru_test_pred = _inverse_target(gru_test_scaled, target_scaler)
            gru_score = regression_metrics(val.target, gru_val_pred)["rmse"]
            gru_calibration = calibrate_prediction_interval(val.target, gru_val_pred, config.nominal_coverage,
                                                           daylight_mask=calibration_mask, night_lower_bound=config.night_lower_bound)
            gru_lo, gru_hi = apply_prediction_interval(gru_test_pred, gru_calibration, test_daylight)
            gru_result = {
                "status": "completed", "validation_rmse": gru_score,
                "validation": regression_metrics(val.target, gru_val_pred),
                "test": regression_metrics(test.target, gru_test_pred, gru_lo, gru_hi),
                "test_daylight": regression_metrics(test.target, gru_test_pred, gru_lo, gru_hi,
                    daylight_mask=_daylight_mask(test.forecast_starts, config.horizon_steps, config.interval_minutes,
                                                input_spec.latitude, input_spec.longitude)),
                "training": gru.training_summary,
                "model_file": "gru.keras", "scalers_file": "scalers.joblib",
                "calibration": gru_calibration,
            }
            if selected_model_override == "gru" or (selected_model_override is None and gru_score is not None and gru_score < baseline_score):
                selected_name = "gru"
                selected_val_pred = gru_val_pred
                selected_test_pred = gru_test_pred
        except ModelUnavailableError as exc:
            gru_result = {"status": "unavailable", "unavailable_reason": str(exc)}
            _json_dump(output / "training_summary.json", gru_result)
            emit(f"GRU 未训练成功：{exc}")
            if config.require_gru:
                raise

    calibration = calibrate_prediction_interval(
        val.target, selected_val_pred, nominal_coverage=config.nominal_coverage, daylight_mask=calibration_mask,
        night_lower_bound=config.night_lower_bound
    )
    test_lower, test_upper = apply_prediction_interval(selected_test_pred, calibration, test_daylight)
    validation_metrics = regression_metrics(val.target, selected_val_pred)
    validation_daylight_metrics = regression_metrics(
        val.target, selected_val_pred, daylight_mask=val_daylight
    )
    test_metrics = regression_metrics(
        test.target, selected_test_pred, test_lower, test_upper
    )
    test_daylight_metrics = regression_metrics(
        test.target, selected_test_pred, test_lower, test_upper, daylight_mask=test_daylight
    )
    metrics = {
        "selected_model": selected_name,
        "validation": validation_metrics,
        "validation_daylight": validation_daylight_metrics,
        "test": test_metrics,
        "test_daylight": test_daylight_metrics,
        "baseline_candidates": {
            model.strategy: regression_metrics(
                val.target, model.predict(val.history_power, config.horizon_steps)
            )
            for _, model, _ in baseline_candidates
        },
        "gru": gru_result if isinstance(gru_result, dict) else None,
        "evaluation_scope": "all_valid_rolling_windows; overlapping target timestamps count at each forecast origin",
        "visualizations": [], "visualization_errors": [],
    }
    if gru_test_pred is not None:
        refs, errors = generate_training_evaluation_visualizations(
            test, gru_test_pred, baseline_test_pred,
            output / "visualizations", config.interval_minutes,
        )
        metrics["visualizations"] = refs
        metrics["visualization_errors"] = errors
    target_start = regular.frame.index.min().isoformat()
    target_end = regular.frame.index.max().isoformat()
    metadata = {
        "schema_version": "pv-model-bundle-1",
        "model_id": config.model_id,
        "model_version": config.model_version,
        "selected_model": selected_name,
        "interval_minutes": config.interval_minutes,
        "history_days": config.history_days,
        "horizon_steps": config.horizon_steps,
        "feature_names": list(train.feature_names),
        "history_feature_names": list(train.history_feature_names),
        "target_column": input_spec.target_col,
        "weather_columns": list(input_spec.weather_columns),
        "history_weather_columns": list(input_spec.history_weather_columns),
        "latitude": input_spec.latitude, "longitude": input_spec.longitude,
        "coordinate_basis": input_spec.coordinate_basis,
        "future_weather_contract": "issued_at <= forecast_origin; coherent latest eligible vintage",
        "input_spec": asdict(input_spec),
        "baseline_fallback": {"strategy": baseline_model.strategy, "calibration": fallback_calibration,
                              "validation_rmse": regression_metrics(val.target, baseline_val_pred)["rmse"],
                              "audit_ref": "metrics.json"},
        "weather_issue_times_used": sorted({issue.isoformat() for data in (train, val, test)
                                             for issue in data.weather_issued_at if issue is not None}),
        "selection_source": "rolling_validation" if selected_model_override else "single_validation_partition",
        "test_usage": "retrospective_report_only; this period has been inspected before and is not a new independent acceptance set",
        "training_data_window": {"start": split_info["train"]["start"], "end": split_info["train"]["end"]},
        "input_data_window": {"start": target_start, "end": target_end},
        "data_splits": split_info,
        "trained_gru_available": gru_result.get("status") == "completed",
        "training_summary_ref": "training_summary.json" if not config.skip_gru and days >= config.min_gru_days else None,
        "input_quality": regular.quality.as_dict(),
        "publication_status": "candidate",
        "lifecycle_operation": "evaluation",
        "config": asdict(config),
        "calibration": calibration,
    }
    _json_dump(output / "model_metadata.json", metadata)
    _json_dump(output / "metrics.json", metrics)
    _json_dump(output / "interval_calibration.json", calibration)
    _json_dump(output / "lifecycle_record.json", {
        "lifecycle_record_id": str(uuid.uuid4()),
        "forecast_type": "pv_power",
        "asset_scope": "configured_pv_assets",
        "operation": "evaluation",
        "current_model_ref": {"model_id": config.model_id, "model_version": config.model_version},
        "candidate_model_ref": {"model_id": config.model_id, "model_version": config.model_version},
        "previous_model_ref": None,
        "data_quality_summary": regular.quality.as_dict(),
        "metrics_ref": "metrics.json",
        "publication_status": "candidate_only_until_real_data_review",
    })
    _json_dump(output / "README.json", {
        "message": "Candidate bundle only; no production publication was performed.",
        "selected_model": selected_name,
        "metrics_file": "metrics.json",
    })
    def metric_line(label, values):
        def fmt(key):
            value = values.get(key)
            return "N/A" if value is None else f"{value:.4f}"
        emit(f"{label}：RMSE={fmt('rmse')} kW | MAE={fmt('mae')} kW | R²={fmt('r2')} | "
             f"nRMSE={fmt('nrmse')} | 区间覆盖率={fmt('coverage')}")
    if gru_result.get("status") == "completed":
        metric_line("GRU 留出测试集", gru_result["test"])
    emit(f"根据{'滚动' if selected_model_override else ''}验证集 RMSE 选中模型：{selected_name}（已训练的 GRU 权重单独保留）"
         if gru_result.get("status") == "completed" else f"选中模型：{selected_name}")
    if selected_name != "gru":
        metric_line("选中模型留出测试集", test_metrics)
    return {"output_dir": str(output), "metadata": metadata, "metrics": metrics}


def _load_bundle(bundle_dir, allow_fallback=False) -> tuple[dict, Optional[GRUForecaster], Optional[dict]]:
    bundle = Path(bundle_dir)
    metadata_path = bundle / "model_metadata.json"
    if not metadata_path.exists():
        raise ModelUnavailableError(f"model metadata not found: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    model = None
    scalers = None
    if metadata.get("selected_model") == "gru":
        model_path = bundle / "gru.keras"
        scaler_path = bundle / "scalers.joblib"
        try:
            if not model_path.exists() or not scaler_path.exists():
                raise ModelUnavailableError("GRU bundle is incomplete")
            model = GRUForecaster.load(str(model_path))
            scalers = joblib.load(scaler_path)
            correction = metadata.get("point_correction")
            if correction:
                import hashlib
                correction_path = bundle / "point_correction.joblib"
                if (correction.get("path") != "point_correction.joblib" or correction.get("method") != "solar_weather_ridge"
                        or hashlib.sha256(correction_path.read_bytes()).hexdigest() != correction.get("sha256")):
                    raise ModelUnavailableError("point correction artifact is missing or differs from saved metadata")
                from .point_correction import SolarWeatherRidge
                model.point_correction = joblib.load(correction_path)
                if not isinstance(model.point_correction, SolarWeatherRidge):
                    raise ModelUnavailableError("invalid point correction artifact")
        except Exception as exc:
            if not allow_fallback:
                raise
            metadata["model_load_error"] = str(exc)
            model, scalers = None, None
    return metadata, model, scalers


def _prediction_spec(metadata, supplied):
    return replace(supplied,
        timezone=metadata.get("input_spec", {}).get("timezone", supplied.timezone),
        target_col=metadata.get("target_column", supplied.target_col),
        history_weather_columns=tuple(metadata.get("history_weather_columns", [])),
        weather_columns=tuple(metadata.get("weather_columns", [])),
        latitude=metadata.get("latitude"), longitude=metadata.get("longitude"),
        coordinate_basis=metadata.get("coordinate_basis", "not_configured"))


def predict_record(
    bundle_dir,
    source,
    output_path=None,
    input_spec: InputSpec = InputSpec(),
    forecast_start=None,
    asset_scope="configured_pv_assets",
    future_weather_source=None,
) -> dict:
    metadata, gru, scalers = _load_bundle(bundle_dir, allow_fallback=True)
    input_spec = _prediction_spec(metadata, input_spec)
    regular = load_and_regularize(source, input_spec, interval_minutes=metadata["interval_minutes"],
                                   available_before=forecast_start)
    horizon = int(metadata["horizon_steps"])
    steps_day = 24 * 60 // metadata["interval_minutes"]
    history_steps = int(metadata["history_days"] * steps_day)
    frame = regular.frame
    start = pd.Timestamp(forecast_start) if forecast_start is not None else frame.index[-1] + pd.Timedelta(minutes=metadata["interval_minutes"])
    if start.tzinfo is None:
        start = start.tz_localize(regular.timezone)
    start = start.tz_convert(regular.timezone)
    frame = frame.loc[frame.index < start]
    if len(frame) < history_steps:
        record = build_forecast_record(
            str(uuid.uuid4()), [], forecast_start or regular.frame.index[-1], metadata["interval_minutes"],
            asset_scope, metadata["model_id"], metadata["model_version"], None,
            metadata["training_data_window"], metadata.get("calibration"), regular.quality.as_dict(),
            status="unavailable", fallback_forecast_ref="insufficient_history",
            limitations=["历史数据不足，未生成预测"], audit_refs=["model_metadata.json"],
        )
        validate_forecast_record(record)
        if output_path:
            _json_dump(Path(output_path), record)
        return record
    history = frame[input_spec.target_col].to_numpy(dtype=float)[-history_steps:]
    if not np.isfinite(history).all() or frame.index[-1] + pd.Timedelta(minutes=metadata["interval_minutes"]) != start:
        record = build_forecast_record(
            str(uuid.uuid4()), [], forecast_start or frame.index[-1], metadata["interval_minutes"],
            asset_scope, metadata["model_id"], metadata["model_version"], None,
            metadata["training_data_window"], metadata.get("calibration"), regular.quality.as_dict(),
            status="unavailable", fallback_forecast_ref="history_gap",
            limitations=["历史功率存在缺口或未衔接预测起点，未生成预测"], audit_refs=["model_metadata.json"],
        )
        validate_forecast_record(record)
        if output_path:
            _json_dump(Path(output_path), record)
        return record
    future_index = pd.date_range(start=start, periods=horizon, freq=f"{metadata['interval_minutes']}min", tz=regular.timezone)
    future_frame = pd.DataFrame(index=future_index)
    issue = None
    weather_columns = metadata.get("weather_columns", [])
    if weather_columns and future_weather_source is not None:
        archive = WeatherForecastArchive(future_weather_source, weather_columns, regular.timezone)
        weather, issue = archive.at(start, future_index)
        if weather is not None:
            future_frame = weather
    feature_values, _ = feature_frame(future_frame, input_spec.target_col, weather_columns,
                                     input_spec.latitude, input_spec.longitude)
    selected_status = "available"
    limitations = []
    calibration = metadata["calibration"]
    used_model = metadata["selected_model"]
    if metadata["selected_model"] == "gru":
        required = list(metadata.get("feature_names", []))
        past_names = metadata.get("history_feature_names", [input_spec.target_col])
        missing_history = any(name not in frame for name in past_names)
        past = None if missing_history else frame[past_names].iloc[-history_steps:].to_numpy(dtype=float)
        if gru is None or scalers is None:
            limitations.append(f"GRU 模型不可用：{metadata.get('model_load_error', 'missing model')}")
        if missing_history or not np.isfinite(past).all():
            limitations.append("GRU 历史气象输入缺失或包含缺口")
        if any(col not in feature_values.columns for col in required):
            limitations.append("缺少在预测起点前发布且完整覆盖预测窗口的未来天气预报")
        if not limitations:
            cov = feature_values[required].to_numpy(dtype=float)
            hist_scaler = scalers.get("history_scaler", scalers["target_scaler"])
            hist_scaled = hist_scaler.transform(past)[None, ...]
            cov_scaled = scalers["feature_scaler"].transform(cov)[None, ...]
            prediction = _inverse_target(gru.predict(hist_scaled, cov_scaled)[0], scalers["target_scaler"])
            if getattr(gru, "point_correction", None) is not None:
                prediction = gru.point_correction.predict(past[None], cov[None], prediction[None],
                    past_names, required)[0]
        else:
            fallback = metadata.get("baseline_fallback")
            if not fallback:
                record = build_forecast_record(str(uuid.uuid4()), [], start, metadata["interval_minutes"],
                    asset_scope, metadata["model_id"], metadata["model_version"], None,
                    metadata["training_data_window"], None, regular.quality.as_dict(), status="unavailable",
                    limitations=limitations + ["缺少已验证基线及其区间校准，未生成预测"], audit_refs=["model_metadata.json"])
                validate_forecast_record(record)
                if output_path:
                    _json_dump(Path(output_path), record)
                return record
            selected_status = "degraded"
            used_model = "baseline:" + fallback["strategy"]
            limitations.append(f"已回退 {used_model}，使用该基线独立校准的预测区间")
            calibration = fallback["calibration"]
            prediction = BaselineForecaster(fallback["strategy"], steps_per_day=steps_day).predict(history[None, :], horizon)[0]
    else:
        baseline = BaselineForecaster(metadata["selected_model"].split(":", 1)[-1], steps_per_day=steps_day)
        prediction = baseline.predict(history[None, :], horizon)[0]
    daylight = _daylight_mask([start], horizon, metadata["interval_minutes"],
                             input_spec.latitude, input_spec.longitude)
    lower, upper = apply_prediction_interval(prediction, calibration, daylight)
    forecast_id = str(uuid.uuid4())
    record = build_forecast_record(
        forecast_id, prediction, start, metadata["interval_minutes"], asset_scope,
        metadata["model_id"], metadata["model_version"], None,
        metadata["training_data_window"], calibration, regular.quality.as_dict(),
        status=selected_status, fallback_forecast_ref=(used_model if selected_status == "degraded" else None),
        limitations=limitations, lower=lower[0], upper=upper[0], audit_refs=["model_metadata.json"],
    )
    record["confidence"]["model_used"] = used_model
    record["confidence"]["weather_forecast_issued_at"] = issue.isoformat() if issue is not None and not limitations else None
    record["confidence"]["coordinate_basis"] = input_spec.coordinate_basis
    record["confidence"]["night_lower_bound"] = calibration.get("night_lower_bound", "empirical")
    record["confidence"]["point_correction"] = (metadata.get("point_correction", {}).get("method")
        if metadata.get("point_correction") and used_model == "gru" else None)
    if record["confidence"]["point_correction"]:
        record["confidence"]["limitations"].append("太阳位置与历史气象线性修正使用夜间零功率假设；未增加未来气象信息")
    if output_path:
        visualizations, visualization_errors = generate_forecast_visualizations(
            frame.index[-history_steps:],
            history,
            record,
            Path(output_path).parent / "visualizations",
        )
        record["visualizations"] = visualizations
        record["visualization_errors"] = visualization_errors
        if visualization_errors:
            record["confidence"]["limitations"].append("部分参考图片生成失败")
        validate_forecast_record(record)
        _json_dump(Path(output_path), record)
    else:
        validate_forecast_record(record)
    return record


def evaluate_bundle(
    bundle_dir,
    source,
    input_spec: InputSpec = InputSpec(),
    visualizations_dir=None,
    future_weather_source=None,
) -> dict:
    """Evaluate an existing bundle on a supplied labelled time series."""
    metadata, _, _ = _load_bundle(bundle_dir, allow_fallback=True)
    input_spec = _prediction_spec(metadata, input_spec)
    regular = load_and_regularize(
        source, input_spec, interval_minutes=metadata["interval_minutes"]
    )
    horizon = int(metadata["horizon_steps"])
    if len(regular.frame) <= horizon:
        return {
            "status": "unavailable",
            "metrics": None,
            "reason": "labelled series is shorter than one horizon",
            "visualizations": [],
            "visualization_errors": [],
        }
    # Hold out the last complete horizon.  The predictor receives only the
    # preceding history so it cannot see the values being evaluated.
    split = len(regular.frame) - horizon
    history_frame = regular.frame.iloc[:split].copy()
    history_frame.insert(0, "timestamp", history_frame.index)
    forecast_start = regular.frame.index[split]
    prediction = predict_record(
        bundle_dir,
        history_frame.reset_index(drop=True),
        input_spec=input_spec,
        forecast_start=forecast_start,
        future_weather_source=future_weather_source,
    )
    if not prediction["forecast_values"]:
        return {
            "status": prediction["status"],
            "metrics": None,
            "visualizations": [],
            "visualization_errors": [],
        }
    truth = regular.frame[input_spec.target_col].iloc[split : split + horizon].to_numpy(dtype=float)
    pred = np.asarray([item["value"] for item in prediction["forecast_values"]])
    visualizations, visualization_errors = [], []
    if visualizations_dir is not None:
        ref, errors = generate_actual_vs_forecast_visualization(
            regular.frame.index[split : split + horizon],
            truth,
            prediction,
            Path(visualizations_dir),
        )
        if ref is not None:
            visualizations.append(ref)
        visualization_errors.extend(errors)
    return {
        "status": prediction["status"],
        "metrics": regression_metrics(truth, pred),
        "visualizations": visualizations,
        "visualization_errors": visualization_errors,
    }
