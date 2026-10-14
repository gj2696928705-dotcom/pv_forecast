"""Recalibrate a frozen candidate using its recorded validation partition."""

from __future__ import annotations

import copy
import hashlib
import shutil
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from .baselines import BaselineForecaster
from .data import (DataValidationError, InputSpec, _apply_column_map, _normalise_timestamp,
                   _read_source, build_windows, load_and_regularize)
from .errors import ModelUnavailableError
from .metrics import apply_prediction_interval, calibrate_prediction_interval, regression_metrics
from .pipeline import _daylight_mask, _inverse_target, _json_dump, _load_bundle, _prediction_spec, _scale_windows


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _partition_windows(regular, metadata, spec, name, archive):
    split = metadata["data_splits"][name]
    start, end = pd.Timestamp(split["start"]), pd.Timestamp(split["end"])
    index = regular.frame.index
    if start not in index or end not in index:
        raise DataValidationError(f"input does not cover recorded {name} partition")
    lo, hi = index.get_loc(start), index.get_loc(end) + 1
    if hi - lo != split["rows"]:
        raise DataValidationError(f"recorded {name} row count does not match input")
    windows = build_windows(regular, lo, hi, metadata["history_days"], metadata["horizon_steps"],
        target_col=spec.target_col, weather_columns=spec.weather_columns,
        history_weather_columns=spec.history_weather_columns, latitude=spec.latitude,
        longitude=spec.longitude, weather_archive=archive)
    if windows.size != split["windows"] or windows.size == 0:
        raise DataValidationError(f"input cannot reproduce all recorded {name} windows")
    return windows


def _predict_windows(windows, metadata, gru, scalers):
    if metadata["selected_model"] == "gru":
        if gru is None or scalers is None:
            raise ModelUnavailableError("recalibration requires the frozen primary GRU and scalers")
        scaled = _scale_windows(windows, **scalers)
        prediction = _inverse_target(gru.predict(scaled.gru_history, scaled.future_features), scalers["target_scaler"])
        if getattr(gru, "point_correction", None) is not None:
            prediction = gru.point_correction.predict(windows.gru_history, windows.future_features, prediction,
                windows.history_feature_names, windows.feature_names)
        return prediction
    return BaselineForecaster(metadata["selected_model"].split(":", 1)[-1],
        steps_per_day=24 * 60 // metadata["interval_minutes"]).predict(windows.history_power, metadata["horizon_steps"])


def _scores(windows, prediction, calibration, daylight):
    lower, upper = apply_prediction_interval(prediction, calibration, daylight)
    result = {}
    for name, mask in (("all_day", np.ones_like(daylight)), ("daylight", daylight), ("night", ~daylight)):
        metrics = regression_metrics(windows.target, prediction, lower, upper, daylight_mask=mask)
        metrics["mean_interval_width_kw"] = float(np.mean((upper - lower)[mask])) if mask.any() else None
        result[name] = metrics
    return result


def recalibrate_bundle(
    bundle_dir, source, output_dir, input_spec: InputSpec = InputSpec(),
    nominal_coverage=None, min_step_samples=30, future_weather_source=None, progress=None,
    night_lower_bound=None,
) -> dict:
    """Create a new daylight-conditioned bundle without fitting any weights.

    Partition boundaries and model selection are frozen in parent metadata.
    Calibration is finalized before test windows are built. The original
    validation also served early stopping, so this is a retrospective empirical
    experiment, not independent calibration/production acceptance.
    """
    parent, output = Path(bundle_dir).resolve(), Path(output_dir).resolve()
    if output.exists() or output == parent or output.is_relative_to(parent):
        raise ValueError("recalibration requires a new output directory outside the parent bundle")
    metadata, gru, scalers = _load_bundle(parent, allow_fallback=False)
    if metadata.get("publication_status") != "candidate":
        raise ValueError("only candidate bundles may be recalibrated")
    spec = _prediction_spec(metadata, input_spec)
    coverage = metadata["calibration"]["nominal_coverage"] if nominal_coverage is None else nominal_coverage
    night_lower_bound = metadata["calibration"].get("night_lower_bound", "empirical") if night_lower_bound is None else night_lower_bound
    interval = pd.Timedelta(minutes=metadata["interval_minutes"])
    splits = metadata["data_splits"]
    if (pd.Timestamp(splits["train"]["end"]) + interval != pd.Timestamp(splits["validation"]["start"])
            or pd.Timestamp(splits["validation"]["end"]) + interval != pd.Timestamp(splits["test"]["start"])):
        raise DataValidationError("recorded train/validation/test partitions must be chronological and contiguous")
    # Record parent identity before copying it to the child.
    hashes = {name: _sha256(parent / name) for name in
              ("model_metadata.json", "interval_calibration.json", "gru.keras", "scalers.joblib", "point_correction.joblib")
              if (parent / name).exists()}
    raw = _apply_column_map(_read_source(source), spec)
    canonical_spec = replace(spec, column_map={})
    timestamps = _normalise_timestamp(raw[spec.timestamp_col], spec.timezone)
    if future_weather_source is None:
        archive = None
    else:
        from .weather import WeatherForecastArchive
        archive = WeatherForecastArchive(future_weather_source, spec.weather_columns, spec.timezone)
    # Truncate before regularization as well as before window construction.
    # Gap filling cannot use test-era observations to decide validation quality.
    cutoff = pd.Timestamp(splits["validation"]["end"])
    validation_input = raw.loc[timestamps <= cutoff].copy()
    validation_regular = load_and_regularize(validation_input, canonical_spec,
                                             interval_minutes=metadata["interval_minutes"])
    val = _partition_windows(validation_regular, metadata, spec, "validation", archive)
    val_pred = _predict_windows(val, metadata, gru, scalers)
    val_daylight = _daylight_mask(val.forecast_starts, metadata["horizon_steps"], metadata["interval_minutes"],
                                  spec.latitude, spec.longitude)
    calibration = calibrate_prediction_interval(val.target, val_pred, coverage, min_step_samples, val_daylight,
                                                night_lower_bound=night_lower_bound)
    calibration["daylight_basis"] = "solar_elevation_gt_zero" if spec.latitude is not None else "calendar_05_to_21_approximation"
    fallback = metadata.get("baseline_fallback")
    new_fallback = None
    if fallback:
        baseline = BaselineForecaster(fallback["strategy"], steps_per_day=24 * 60 // metadata["interval_minutes"])
        fallback_pred = baseline.predict(val.history_power, metadata["horizon_steps"])
        new_fallback = copy.deepcopy(fallback)
        new_fallback["calibration"] = calibrate_prediction_interval(val.target, fallback_pred, coverage,
                                                                    min_step_samples, val_daylight,
                                                                    night_lower_bound=night_lower_bound)
        new_fallback["calibration"]["daylight_basis"] = calibration["daylight_basis"]
        new_fallback["audit_ref"] = "recalibration_report.json"
    if progress:
        progress(f"验证集 {val.size} 个完整窗口：已完成昼夜区间及独立基线校准；开始回顾性测试。")
    regular = load_and_regularize(raw, canonical_spec, interval_minutes=metadata["interval_minutes"])
    test = _partition_windows(regular, metadata, spec, "test", archive)
    test_pred = _predict_windows(test, metadata, gru, scalers)
    test_daylight = _daylight_mask(test.forecast_starts, metadata["horizon_steps"], metadata["interval_minutes"],
                                   spec.latitude, spec.longitude)
    reports = {
        "validation": {"before": _scores(val, val_pred, metadata["calibration"], val_daylight),
                       "after": _scores(val, val_pred, calibration, val_daylight)},
        "test": {"before": _scores(test, test_pred, metadata["calibration"], test_daylight),
                 "after": _scores(test, test_pred, calibration, test_daylight)},
    }
    source_hash = (_sha256(source) if isinstance(source, (str, Path)) else
                   hashlib.sha256(regular.frame.to_csv().encode("utf-8")).hexdigest())
    report = {
        "status": "completed", "operation": "daylight_interval_recalibration", "weights_retrained": False,
        "parent_bundle": str(parent), "parent_model_version": metadata["model_version"],
        "parent_bundle_sha256": hashes, "source_data_sha256": source_hash,
        "source_hash_basis": "raw_file" if isinstance(source, (str, Path)) else "regularized_frame_csv",
        "calibration_partition": copy.deepcopy(splits["validation"]),
        "test_partition": copy.deepcopy(splits["test"]), "test_used_for_calibration": False,
        "test_used_for_model_selection": False, "daylight_basis": calibration["daylight_basis"],
        "nominal_coverage": coverage, "night_lower_bound": night_lower_bound, "metrics": reports,
        "evaluation_scope": "all recorded windows; overlapping target timestamps counted per origin",
        "limitations": ["Retrospective empirical experiment; validation was also used for checkpoint selection.",
                        "Overlapping windows are dependent; no future conditional coverage guarantee.",
                        "Candidate only; real or unused data acceptance remains outstanding."],
    }
    # Write only after both complete recorded partitions have been reproduced.
    output.mkdir(parents=True, exist_ok=False)
    for name in ("gru.keras", "scalers.joblib", "point_correction.joblib"):
        if (parent / name).exists():
            shutil.copy2(parent / name, output / name)
            if _sha256(output / name) != hashes[name]:
                raise RuntimeError(f"copied {name} does not match frozen parent")
    derived = copy.deepcopy(metadata)
    derived.update(calibration=calibration, baseline_fallback=new_fallback,
        model_version=metadata["model_version"] + ("+daylight-zero-lower-v1" if night_lower_bound == "zero" else "+daylight-interval-v1"),
        lifecycle_operation="interval_recalibration", training_summary_ref=None,
        parent_bundle_ref=str(parent), recalibration_report_ref="recalibration_report.json",
        test_usage="retrospective_report_only; not a new independent acceptance set")
    derived["config"]["interval_conditioning"] = "daylight"
    derived["config"]["night_lower_bound"] = night_lower_bound
    metrics = {"selected_model": metadata["selected_model"],
        "validation": reports["validation"]["after"]["all_day"],
        "validation_daylight": reports["validation"]["after"]["daylight"],
        "validation_night": reports["validation"]["after"]["night"],
        "test": reports["test"]["after"]["all_day"],
        "test_daylight": reports["test"]["after"]["daylight"],
        "test_night": reports["test"]["after"]["night"],
        "evaluation_scope": report["evaluation_scope"], "recalibration_report_ref": "recalibration_report.json",
        "visualizations": [], "visualization_errors": []}
    _json_dump(output / "model_metadata.json", derived)
    _json_dump(output / "interval_calibration.json", calibration)
    _json_dump(output / "metrics.json", metrics)
    _json_dump(output / "recalibration_report.json", report)
    _json_dump(output / "README.json", {"message": "Candidate interval recalibration; no weight fitting or publication.",
        "parent_bundle": str(parent), "selected_model": metadata["selected_model"], "metrics_file": "metrics.json"})
    if any(_sha256(parent / name) != digest for name, digest in hashes.items()):
        raise RuntimeError("parent bundle changed during recalibration")
    return {"output_dir": str(output), "metadata": derived, "metrics": metrics, "report": report}
