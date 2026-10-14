"""Finite GRU comparisons with expanding, strictly chronological validation."""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .baselines import BaselineForecaster
from .data import InputSpec, build_windows, load_and_regularize, split_boundaries
from .errors import InsufficientHistoryError
from .metrics import regression_metrics
from .models import GRUForecaster
from .pipeline import (TrainingConfig, _daylight_mask, _fit_history_scaler, _fit_scalers,
                       _gru_config, _inverse_target, _json_dump, _limit_windows,
                       _scale_windows, train_model)
from .weather import WeatherForecastArchive


@dataclass(frozen=True)
class RollingFold:
    name: str
    train_end: str  # exclusive; validation begins here
    validation_end: str  # exclusive


def default_folds(index):
    """Jan-Apr -> May, Jan-Jun -> July, Jan-Aug -> September."""
    year = index[0].year
    return [RollingFold(f"month_{month:02d}", f"{year}-{month:02d}-01",
                        f"{year}-{month + 1:02d}-01") for month in (5, 7, 9)]


def fold_bounds(index, folds, development_end):
    """Reject folds that access the reserved test period or overlap in validation."""
    bounds, previous_end, names = [], 0, set()
    for fold in folds:
        if not fold.name or fold.name in names or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in fold.name):
            raise ValueError("fold names must be unique safe directory names")
        names.add(fold.name)
        stamps = []
        for value in (fold.train_end, fold.validation_end):
            stamp = pd.Timestamp(value)
            stamp = stamp.tz_localize(index.tz) if stamp.tzinfo is None else stamp.tz_convert(index.tz)
            stamps.append(stamp)
        boundary = index[development_end]
        if stamps[1] > boundary or stamps[0] <= index[0] or stamps[0] >= stamps[1]:
            raise ValueError("rolling folds must precede the reserved test period and have positive train/validation ranges")
        lo, hi = [int(index.searchsorted(stamp)) for stamp in stamps]
        if lo < previous_end:
            raise ValueError("rolling validation periods must be chronological and disjoint")
        if lo >= development_end or hi > development_end:
            raise ValueError("rolling validation cannot access the test partition")
        bounds.append((fold, lo, hi))
        previous_end = hi
    if not bounds:
        raise ValueError("at least one rolling fold is required")
    return bounds


def candidate_configs(config):
    """Hold features/batch/seed fixed; compare original rate, schedule, capacity."""
    return {
        "gru32_control": replace(config, gru_units=32, gru_learning_rate=.001,
            reduce_lr=False, gru_epochs=min(config.gru_epochs, 30), gru_patience=5, min_delta=0.),
        "gru32_scheduled": replace(config, gru_units=32, reduce_lr=True),
        "gru16_scheduled": replace(config, gru_units=16, reduce_lr=True),
    }


class MetricAccumulator:
    """Aggregate point-wise sums, not averages of fold R² or RMSE."""
    def __init__(self):
        self.n = 0
        self.sse = self.sae = self.total = self.square = 0.

    def add(self, true, pred, mask=None):
        true, pred = np.asarray(true, dtype=float), np.asarray(pred, dtype=float)
        if mask is not None:
            true, pred = true[mask], pred[mask]
        error = pred - true
        self.n += true.size
        self.sse += float(np.sum(error ** 2))
        self.sae += float(np.sum(np.abs(error)))
        self.total += float(np.sum(true))
        self.square += float(np.sum(true ** 2))

    def result(self):
        if not self.n:
            return {"count": 0, "rmse": None, "mae": None, "r2": None, "nrmse": None}
        variance = self.square - self.total ** 2 / self.n
        rmse = float(np.sqrt(self.sse / self.n))
        return {"count": self.n, "rmse": rmse, "mae": self.sae / self.n,
                "r2": 1 - self.sse / variance if variance > 1e-12 else None,
                "nrmse": rmse / (self.total / self.n) if self.total > 1e-12 else None}


def train_experiment(source, output_dir, input_spec=InputSpec(), config=TrainingConfig(),
                     *, folds=None, future_weather_source=None, progress=None):
    """Search only rolling development folds, then fit selected GRU and report test.

    Even when a baseline wins, the best GRU is trained/saved as an auditable
    candidate. The already-inspected original test set is explicitly retrospective.
    """
    if config.skip_gru:
        raise ValueError("rolling comparison requires GRU; use train_model for baseline-only runs")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    def emit(message):
        with (output / "training.log").open("a", encoding="utf-8") as handle:
            handle.write(message + "\n")
        if progress:
            progress(message)
    regular = load_and_regularize(source, input_spec, config.interval_minutes)
    _, development_end = split_boundaries(len(regular.frame), config.train_ratio, config.val_ratio)
    fold_list = default_folds(regular.frame.index) if folds is None else list(folds)
    bounds = fold_bounds(regular.frame.index, fold_list, development_end)
    archive = (WeatherForecastArchive(future_weather_source, input_spec.weather_columns, input_spec.timezone)
               if future_weather_source is not None else None)
    candidates = candidate_configs(config)
    baseline_names = ["baseline:" + s for s in ("persistence", "previous_day", "same_slot_mean")]
    accumulators = {name: (MetricAccumulator(), MetricAccumulator()) for name in [*candidates, *baseline_names]}
    report = {"status": "running", "folds": [], "candidates": {k: asdict(v) for k, v in candidates.items()},
              "selection_metric": "pooled rolling validation RMSE (kW)",
              "feature_policy": "identical historical weather and solar/calendar features for all GRU candidates",
              "reserved_test_start": regular.frame.index[development_end].isoformat(),
              "test_used_for_selection": False,
              "scope": "full_windows" if not config.max_gru_samples else "explicitly_capped_training_windows",
              "history_weather_columns": list(input_spec.history_weather_columns),
              "input_spec": asdict(input_spec)}
    _json_dump(output / "rolling_validation.json", report)
    emit(f"滚动验证：{len(bounds)} 个时间分区 × 3 组 GRU；测试时段从 {report['reserved_test_start']} 起，不参与选模。")

    for fold, train_end, val_end in bounds:
        train_days = train_end * config.interval_minutes / 1440
        if train_days < config.min_gru_days:
            raise InsufficientHistoryError(f"{fold.name}: {train_days:g} training days < {config.min_gru_days}")
        args = dict(history_days=config.history_days, horizon_steps=config.horizon_steps,
            target_col=input_spec.target_col, weather_columns=input_spec.weather_columns,
            history_weather_columns=input_spec.history_weather_columns,
            latitude=input_spec.latitude, longitude=input_spec.longitude, weather_archive=archive)
        train = build_windows(regular, 0, train_end, **args)
        val = build_windows(regular, train_end, val_end, **args)
        if not train.size or not val.size:
            raise InsufficientHistoryError(f"{fold.name}: no complete clean train/validation windows")
        target_scaler, feature_scaler = _fit_scalers(train)
        history_scaler = _fit_history_scaler(train)
        scaled_train = _limit_windows(_scale_windows(train, target_scaler, feature_scaler, history_scaler), config.max_gru_samples)
        scaled_val = _scale_windows(val, target_scaler, feature_scaler, history_scaler)
        day = _daylight_mask(val.forecast_starts, config.horizon_steps, config.interval_minutes,
                             input_spec.latitude, input_spec.longitude)
        fold_report = {"name": fold.name, "train_start": regular.frame.index[0].isoformat(),
            "train_end": regular.frame.index[train_end - 1].isoformat(),
            "validation_start": regular.frame.index[train_end].isoformat(),
            "validation_end": regular.frame.index[val_end - 1].isoformat(),
            "training_days": train_days, "training_windows_available": train.size,
            "training_windows_used": scaled_train.size, "validation_windows": val.size,
            "scalers_fit_on": "this fold training windows only", "results": {}}
        report["folds"].append(fold_report)
        emit(f"[{fold.name}] 训练 {train_days:g} 天，{scaled_train.size}/{train.size} 窗口；"
             f"验证 {fold_report['validation_start']} → {fold_report['validation_end']}，{val.size} 窗口。")

        def record_result(name, prediction):
            values = {"all_day": regression_metrics(val.target, prediction),
                      "daylight": regression_metrics(val.target, prediction, daylight_mask=day)}
            fold_report["results"][name] = values
            accumulators[name][0].add(val.target, prediction)
            accumulators[name][1].add(val.target, prediction, day)
            for period in ("all_day", "daylight"):
                metric = values[period]
                r2 = "N/A" if metric['r2'] is None else f"{metric['r2']:.4f}"
                emit(f"[{fold.name}/{name}/{period}] RMSE={metric['rmse']:.3f} kW "
                     f"MAE={metric['mae']:.3f} kW R²={r2}")
            return values

        for name in baseline_names:
            model = BaselineForecaster(name.split(":")[1], 1440 // config.interval_minutes)
            record_result(name, model.predict(val.history_power, config.horizon_steps))

        for name, candidate in candidates.items():
            directory = output / "experiments" / fold.name / name
            directory.mkdir(parents=True, exist_ok=True)
            _json_dump(directory / "training_config.json", asdict(candidate))
            _json_dump(directory / "data_split.json", {k: v for k, v in fold_report.items() if k != "results"})
            joblib.dump({"target_scaler": target_scaler, "feature_scaler": feature_scaler,
                         "history_scaler": history_scaler}, directory / "scalers.joblib")
            _json_dump(directory / "features.json", {
                "history": list(train.history_feature_names), "future": list(train.feature_names),
                "training_weather_issue_times": sorted({v.isoformat() for v in train.weather_issued_at if v is not None}),
                "validation_weather_issue_times": sorted({v.isoformat() for v in val.weather_issued_at if v is not None})})
            emit(f"[{fold.name}/{name}] units={candidate.gru_units} LR={candidate.gru_learning_rate:g} "
                 f"调度={candidate.reduce_lr} epochs≤{candidate.gru_epochs} patience={candidate.gru_patience}")
            def fold_progress(message):
                with (directory / "training.log").open("a", encoding="utf-8") as handle:
                    handle.write(message + "\n")
                emit(f"[{fold.name}/{name}] {message}")
            gru = GRUForecaster(config=_gru_config(candidate))
            gru.fit(scaled_train.gru_history, scaled_train.future_features, scaled_train.target,
                validation_data=(scaled_val.gru_history, scaled_val.future_features, scaled_val.target),
                target_scaler=target_scaler, output_dir=directory, progress=fold_progress)
            gru.save(str(directory / "gru.keras"))
            prediction = _inverse_target(gru.predict(scaled_val.gru_history, scaled_val.future_features), target_scaler)
            values = record_result(name, prediction)
            values["training_summary_ref"] = (directory / "training_summary.json").relative_to(output).as_posix()
            _json_dump(directory / "metrics.json", values)
            _json_dump(output / "rolling_validation.json", report)
            del gru

    report["aggregate"] = {name: {"all_day": pair[0].result(), "daylight": pair[1].result(),
        "worst_fold_rmse": max(f["results"][name]["all_day"]["rmse"] for f in report["folds"])}
        for name, pair in accumulators.items()}
    best_gru = min(candidates, key=lambda name: report["aggregate"][name]["all_day"]["rmse"])
    selected = min(report["aggregate"], key=lambda name: report["aggregate"][name]["all_day"]["rmse"])
    report.update(status="selection_complete", best_gru_candidate=best_gru, selected_candidate=selected,
                  selected_config=asdict(candidates[best_gru]))
    _json_dump(output / "rolling_validation.json", report)
    emit(f"滚动验证选中：{selected}；最终 GRU 配置：{best_gru}。开始最终训练及回顾性测试报告。")
    result = train_model(source, output, input_spec, candidates[best_gru], progress=progress,
        future_weather_source=future_weather_source,
        selected_model_override="gru" if selected in candidates else selected)
    report["status"] = "completed"
    result["metadata"]["rolling_validation_ref"] = "rolling_validation.json"
    result["metadata"]["selected_gru_candidate"] = best_gru
    result["metrics"]["rolling_validation_ref"] = "rolling_validation.json"
    _json_dump(output / "rolling_validation.json", report)
    _json_dump(output / "model_metadata.json", result["metadata"])
    _json_dump(output / "metrics.json", result["metrics"])
    result["rolling_validation"] = report
    return result
