"""Fit and validate a causal correction while preserving all parent GRU weights."""
from __future__ import annotations

import copy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import shutil

import joblib
import numpy as np
import pandas as pd

from .baselines import BaselineForecaster
from .data import (InputSpec, _apply_column_map, _normalise_timestamp, _read_source, load_and_regularize)
from .experiments import MetricAccumulator
from .metrics import calibrate_prediction_interval
from .models import GRUForecaster
from .pipeline import _daylight_mask, _json_dump, _load_bundle, _prediction_spec
from .point_correction import SolarWeatherRidge
from .recalibration import _partition_windows, _predict_windows, _scores, _sha256
from .weather import WeatherForecastArchive


def _batched_prediction(windows, meta, gru, scalers, batch_windows):
    # Slice directly; _limit_windows is deliberately not used for inference.
    from .backtesting import _take
    chunks = []
    for lo in range(0, windows.size, batch_windows):
        part = _take(windows, np.arange(lo, min(windows.size, lo+batch_windows)))
        chunks.append(_predict_windows(part, meta, gru, scalers))
    return np.concatenate(chunks)


def fit_point_correction_bundle(bundle_dir, source, output_dir, input_spec=InputSpec(), *,
                                alphas=(0.001, 0.01, 0.1), batch_windows=1024,
                                future_weather_source=None, progress=None):
    """Select a correction on frozen parent rolling validation, never the test.

    All nine parent GRU artifacts must exist and have used all available
    training windows. Parent model configuration stays frozen: the correction
    uses its selected GRU type in each of three expanding folds. Raw forecasts
    are also candidates, so a correction with no validated benefit is rejected.
    """
    parent, output = Path(bundle_dir).resolve(), Path(output_dir).resolve()
    if output.exists() or output.is_relative_to(parent):
        raise ValueError("point correction requires a new output directory outside the parent bundle")
    if not isinstance(batch_windows, int) or isinstance(batch_windows, bool) or batch_windows < 1:
        raise ValueError("batch_windows must be a positive integer")
    alphas = tuple(SolarWeatherRidge(a).alpha for a in alphas)
    if not alphas or len(set(alphas)) != len(alphas):
        raise ValueError("provide distinct positive correction penalties")
    meta, gru, scalers = _load_bundle(parent, allow_fallback=False)
    if meta.get("publication_status") != "candidate" or meta["selected_model"] != "gru" or meta.get("point_correction"):
        raise ValueError("requires an uncorrected candidate GRU parent")
    if meta["interval_minutes"] != 15 or meta["horizon_steps"] != 96 or meta.get("latitude") is None:
        raise ValueError("requires 15-minute/96-step data with configured solar geometry")
    rolling_path = parent / meta.get("rolling_validation_ref", "rolling_validation.json")
    rolling = json.loads(rolling_path.read_text(encoding="utf-8"))
    if rolling["status"] != "completed" or rolling["scope"] != "full_windows" or len(rolling["folds"]) != 3:
        raise ValueError("requires the completed full three-fold parent comparison")
    selected_gru = meta["selected_gru_candidate"]
    parent_hashes = {name: _sha256(parent/name) for name in
                     ("model_metadata.json", "gru.keras", "scalers.joblib", "rolling_validation.json")}
    fold_hashes = {}
    for fold in rolling["folds"]:
        for candidate in rolling["candidates"]:
            folder = parent/"experiments"/fold["name"]/candidate
            split = json.loads((folder/"data_split.json").read_text(encoding="utf-8"))
            summary = json.loads((folder/"training_summary.json").read_text(encoding="utf-8"))
            if split["training_windows_used"] != split["training_windows_available"] or summary["epochs_completed"] < 1:
                raise ValueError("all parent GRU candidates must retain complete training evidence")
            for name in ("gru.keras", "scalers.joblib", "data_split.json", "training_summary.json", "features.json"):
                path = folder/name
                fold_hashes[path.relative_to(parent).as_posix()] = _sha256(path)
    spec = _prediction_spec(meta, input_spec)
    raw = _apply_column_map(_read_source(source), spec)
    canonical_spec = replace(spec, column_map={})
    timestamps = _normalise_timestamp(raw[spec.timestamp_col], spec.timezone)
    development = raw.loc[timestamps <= pd.Timestamp(meta["data_splits"]["validation"]["end"])].copy()
    regular = load_and_regularize(development, canonical_spec, interval_minutes=15)
    archive = WeatherForecastArchive(future_weather_source, spec.weather_columns, spec.timezone) if future_weather_source is not None else None
    # The runner creates only a new directory and writes a protocol before
    # fitting. Test windows are built only after selection and calibration.
    output.mkdir(parents=True, exist_ok=False)
    names = ["raw_gru", *[f"ridge_{alpha:g}" for alpha in alphas]]
    accum = {name: {period: MetricAccumulator() for period in ("all_day", "daylight")} for name in names}
    report = {"status": "running", "operation": "frozen_gru_point_correction",
        "gru_weights_retrained": False, "point_correction_fitted": True,
        "parent_bundle": str(parent), "parent_bundle_sha256": parent_hashes,
        "parent_full_gru_artifact_count": 9, "parent_rolling_artifact_sha256": fold_hashes,
        "frozen_gru_candidate": selected_gru, "alphas": alphas,
        "selection_metric": "pooled rolling validation RMSE; require daylight and worst-fold improvement too",
        "test_used_for_selection": False, "test_used_for_correction_fit": False,
        "reserved_test_start": meta["data_splits"]["test"]["start"], "folds": [],
        "feature_policy": "raw GRU, prior history summaries and solar/calendar; no target weather labels",
        "evaluation_scope": "all recorded windows; overlapping target timestamps count per origin"}
    _json_dump(output/"point_correction_report.json", report)

    def emit(message):
        with (output/"point_correction.log").open("a", encoding="utf-8") as handle:
            handle.write(message+"\n")
        if progress:
            progress(message)

    def fit(windows, raw_pred):
        return SolarWeatherRidge(alphas[0]).fit(windows.gru_history, windows.future_features, raw_pred,
            windows.target, windows.history_feature_names, windows.feature_names, batch_windows=batch_windows)

    def corrected(model, windows, prediction):
        return model.predict(windows.gru_history, windows.future_features, prediction,
                             windows.history_feature_names, windows.feature_names)

    for fold in rolling["folds"]:
        folder = parent/"experiments"/fold["name"]/selected_gru
        split = json.loads((folder/"data_split.json").read_text(encoding="utf-8"))
        fold_meta = copy.deepcopy(meta)
        fold_meta["data_splits"] = {
            "train": {"start": split["train_start"], "end": split["train_end"],
                "rows": int((pd.Timestamp(split["train_end"])-pd.Timestamp(split["train_start"])).total_seconds()/900)+1,
                "windows": split["training_windows_available"]},
            "validation": {"start": split["validation_start"], "end": split["validation_end"],
                "rows": int((pd.Timestamp(split["validation_end"])-pd.Timestamp(split["validation_start"])).total_seconds()/900)+1,
                "windows": split["validation_windows"]}}
        features = json.loads((folder/"features.json").read_text(encoding="utf-8"))
        if features["history"] != meta["history_feature_names"] or features["future"] != meta["feature_names"]:
            raise ValueError("parent fold feature order does not match the frozen primary contract")
        train = _partition_windows(regular, fold_meta, spec, "train", archive)
        val = _partition_windows(regular, fold_meta, spec, "validation", archive)
        fold_gru, fold_scalers = GRUForecaster.load(str(folder/"gru.keras")), joblib.load(folder/"scalers.joblib")
        emit(f"[{fold['name']}] 冻结 {selected_gru}：全部训练{train.size}窗口、验证{val.size}窗口；开始推理。")
        train_pred = _batched_prediction(train, fold_meta, fold_gru, fold_scalers, batch_windows)
        val_pred = _batched_prediction(val, fold_meta, fold_gru, fold_scalers, batch_windows)
        head = fit(train, train_pred)
        day = _daylight_mask(val.forecast_starts, 96, 15, spec.latitude, spec.longitude)
        results = {}
        for name in names:
            prediction = val_pred if name == "raw_gru" else corrected(head.with_alpha(float(name[6:])), val, val_pred)
            results[name] = {}
            for period, mask in (("all_day", None), ("daylight", day)):
                meter = MetricAccumulator()
                meter.add(val.target, prediction, mask)
                results[name][period] = meter.result()
                accum[name][period].add(val.target, prediction, mask)
            emit(f"[{fold['name']}/{name}] 全天RMSE={results[name]['all_day']['rmse']:.3f}，白天RMSE={results[name]['daylight']['rmse']:.3f} kW")
        report["folds"].append({"name": fold["name"], "data_splits": fold_meta["data_splits"],
            "correction_training_windows": train.size, "daylight_counts_per_lead": head.daylight_counts.tolist(), "results": results})
        _json_dump(output/"point_correction_report.json", report)
        del train, val, train_pred, val_pred, fold_gru, fold_scalers, head
    aggregate = {name: {period: meter.result() for period, meter in pair.items()} for name, pair in accum.items()}
    for name in names:
        aggregate[name]["worst_fold_rmse"] = max(f["results"][name]["all_day"]["rmse"] for f in report["folds"])
    reference = aggregate["raw_gru"]
    eligible = [name for name in names[1:] if aggregate[name]["all_day"]["rmse"] < reference["all_day"]["rmse"]
        and aggregate[name]["daylight"]["rmse"] < reference["daylight"]["rmse"]
        and aggregate[name]["worst_fold_rmse"] <= reference["worst_fold_rmse"]]
    selected = min(eligible, key=lambda name: aggregate[name]["all_day"]["rmse"]) if eligible else "raw_gru"
    report.update(status="selection_complete", aggregate=aggregate, selected_correction=selected,
                  eligible_corrections=eligible)
    _json_dump(output/"point_correction_report.json", report)
    emit(f"修正层选中 {selected}；原GRU配置不变。开始最终训练窗口修正拟合和验证区间校准。")
    final_train = _partition_windows(regular, meta, spec, "train", archive)
    final_val = _partition_windows(regular, meta, spec, "validation", archive)
    final_head = None
    if selected != "raw_gru":
        final_train_pred = _batched_prediction(final_train, meta, gru, scalers, batch_windows)
        final_head = fit(final_train, final_train_pred).with_alpha(float(selected[6:]))
    final_val_raw = _batched_prediction(final_val, meta, gru, scalers, batch_windows)
    final_val_pred = final_val_raw if final_head is None else corrected(final_head, final_val, final_val_raw)
    day_val = _daylight_mask(final_val.forecast_starts, 96, 15, spec.latitude, spec.longitude)
    calibration = calibrate_prediction_interval(final_val.target, final_val_pred,
        meta["calibration"]["nominal_coverage"], daylight_mask=day_val, night_lower_bound="zero")
    fallback = copy.deepcopy(meta["baseline_fallback"])
    baseline_val = BaselineForecaster(fallback["strategy"]).predict(final_val.history_power, 96)
    fallback["calibration"] = calibrate_prediction_interval(final_val.target, baseline_val,
        meta["calibration"]["nominal_coverage"], daylight_mask=day_val, night_lower_bound="zero")
    fallback["audit_ref"] = "point_correction_report.json"
    report.update(status="calibration_frozen", final_correction_training_windows=final_train.size,
                  calibration_partition=copy.deepcopy(meta["data_splits"]["validation"]),
                  calibration=calibration, final_validation_scores={
                      "before": _scores(final_val, final_val_raw, meta["calibration"], day_val),
                      "after": _scores(final_val, final_val_pred, calibration, day_val)})
    _json_dump(output/"selection_and_calibration_frozen.json", report)
    # Build and inspect test labels only after all parameters above are frozen.
    all_regular = load_and_regularize(raw, canonical_spec, interval_minutes=15)
    test = _partition_windows(all_regular, meta, spec, "test", archive)
    test_raw = _batched_prediction(test, meta, gru, scalers, batch_windows)
    test_pred = test_raw if final_head is None else corrected(final_head, test, test_raw)
    day_test = _daylight_mask(test.forecast_starts, 96, 15, spec.latitude, spec.longitude)
    scores = {"before": _scores(test, test_raw, meta["calibration"], day_test),
              "after": _scores(test, test_pred, calibration, day_test)}
    derived = copy.deepcopy(meta)
    for name in ("gru.keras", "scalers.joblib"):
        shutil.copy2(parent/name, output/name)
        if _sha256(output/name) != parent_hashes[name]:
            raise RuntimeError("frozen parent copy changed")
    correction_ref = None
    if final_head is not None:
        joblib.dump(final_head, output/"point_correction.joblib")
        correction_ref = {"method": "solar_weather_ridge", "path": "point_correction.joblib",
                          "sha256": _sha256(output/"point_correction.joblib"), "alpha": final_head.alpha,
                          "zero_night_point": True, "training_windows": final_head.window_count,
                          "history_names": list(final_head.history_names), "future_names": list(final_head.future_names)}
    derived.update(point_correction=correction_ref, calibration=calibration, baseline_fallback=fallback,
        model_version=meta["model_version"]+"+point-correction-v1", lifecycle_operation="point_correction_fit",
        parent_bundle_ref=str(parent), point_correction_report_ref="point_correction_report.json",
        training_summary_ref=None, test_usage="retrospective_report_only; not new independent acceptance")
    derived["config"].update(interval_conditioning="daylight", night_lower_bound="zero")
    report.update(status="completed", test_scores=scores, test_partition=copy.deepcopy(meta["data_splits"]["test"]),
        point_correction=correction_ref, source_data_sha256=_sha256(source) if isinstance(source, (str, Path)) else
            hashlib.sha256(raw.to_csv(index=False).encode("utf-8")).hexdigest(),
        limitations=["Parent GRUs and all 2025 labels were previously inspected; retrospective development only.",
                     "Validation also served parent GRU early stopping and overlaps some rolling correction selection.",
                     "Night-zero point assumption needs real-site validation; future weather forecast information is not added."])
    metrics = {"selected_model": "gru", "point_correction": selected,
        "validation": report["final_validation_scores"]["after"]["all_day"],
        "test": scores["after"]["all_day"], "test_daylight": scores["after"]["daylight"],
        "test_night": scores["after"]["night"], "point_correction_report_ref": "point_correction_report.json"}
    for name, value in (("model_metadata.json", derived), ("interval_calibration.json", calibration),
                        ("metrics.json", metrics), ("point_correction_report.json", report)):
        _json_dump(output/name, value)
    if any(_sha256(parent/name) != digest for name, digest in {**parent_hashes, **fold_hashes}.items()):
        raise RuntimeError("parent artifacts changed during correction fit")
    emit(f"完整回顾测试{test.size}窗口：修正前RMSE={scores['before']['all_day']['rmse']:.3f}，修正后={scores['after']['all_day']['rmse']:.3f} kW")
    return {"output_dir": str(output), "metadata": derived, "metrics": metrics, "report": report}
