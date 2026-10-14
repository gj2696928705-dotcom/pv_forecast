"""Metrics and empirical prediction-interval calibration."""

from __future__ import annotations

from typing import Optional

import numpy as np


def regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    lower: Optional[np.ndarray] = None,
    upper: Optional[np.ndarray] = None,
    daylight_mask: Optional[np.ndarray] = None,
) -> dict:
    true = np.asarray(y_true, dtype=float)
    pred = np.asarray(y_pred, dtype=float)
    if true.shape != pred.shape:
        raise ValueError("y_true and y_pred must have the same shape")
    mask = np.isfinite(true) & np.isfinite(pred)
    if daylight_mask is not None:
        mask &= np.asarray(daylight_mask, dtype=bool)
    a = true[mask]
    b = pred[mask]
    if a.size == 0:
        return {"count": 0, "mae": None, "rmse": None, "nrmse": None, "r2": None, "coverage": None}
    error = b - a
    mae = float(np.mean(np.abs(error)))
    rmse = float(np.sqrt(np.mean(error**2)))
    denominator = float(np.mean(np.abs(a)))
    nrmse = float(rmse / denominator) if denominator > 1e-12 else None
    total = float(np.sum((a - np.mean(a)) ** 2))
    r2 = float(1.0 - np.sum(error**2) / total) if total > 1e-12 else None
    coverage = None
    if lower is not None and upper is not None:
        lo = np.asarray(lower, dtype=float)[mask]
        hi = np.asarray(upper, dtype=float)[mask]
        coverage = float(np.mean((a >= lo) & (a <= hi)))
    return {
        "count": int(a.size),
        "mae": mae,
        "rmse": rmse,
        "nrmse": nrmse,
        "r2": r2,
        "coverage": coverage,
    }


def calibrate_prediction_interval(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    nominal_coverage: float = 0.80,
    min_step_samples: int = 30,
    daylight_mask: Optional[np.ndarray] = None,
    night_lower_bound: str = "empirical",
) -> dict:
    """Calibrate symmetric intervals, optionally by known target daylight.

    Conditioning uses the target timestamps' solar/calendar mask, never the
    observed power or future weather. Sparse steps pool within the same
    daylight group; an absent group uses the unconditioned fallback explicitly.
    Overlapping windows are dependent, so these are empirical intervals and
    carry no conformal or independent out-of-sample coverage guarantee.
    """

    if not 0 < nominal_coverage < 1:
        raise ValueError("nominal_coverage must be between 0 and 1")
    if night_lower_bound not in {"empirical", "zero"}:
        raise ValueError("night_lower_bound must be 'empirical' or 'zero'")
    if night_lower_bound == "zero" and daylight_mask is None:
        raise ValueError("zero night lower bound requires daylight conditioning")
    if not isinstance(min_step_samples, (int, np.integer)) or min_step_samples < 1:
        raise ValueError("min_step_samples must be a positive integer")
    true = np.asarray(y_true, dtype=float)
    pred = np.asarray(y_pred, dtype=float)
    if true.shape != pred.shape or true.ndim != 2:
        raise ValueError("interval calibration expects arrays shaped (samples, horizon)")
    residual = np.abs(true - pred)
    alpha = 1.0 - nominal_coverage
    global_values = residual[np.isfinite(residual)]
    if global_values.size == 0:
        raise ValueError("cannot calibrate interval with no finite residuals")
    global_q = float(np.quantile(global_values, 1.0 - alpha))
    per_step = []
    used_per_step = True
    for step in range(residual.shape[1]):
        values = residual[:, step]
        values = values[np.isfinite(values)]
        if values.size < min_step_samples:
            used_per_step = False
            per_step.append(global_q)
        else:
            per_step.append(float(np.quantile(values, 1.0 - alpha)))
    result = {
        "method": "empirical_absolute_residual",
        "nominal_coverage": nominal_coverage,
        "quantiles_kw": per_step,
        "global_quantile_kw": global_q,
        "step_specific": used_per_step,
        "calibration_samples": int(true.shape[0]),
    }
    if daylight_mask is None:
        return result
    daylight = _validated_daylight_mask(daylight_mask, true.shape)
    groups = {}
    for name, group in (("daylight", daylight), ("night", ~daylight)):
        finite = group & np.isfinite(residual)
        pooled = residual[finite]
        group_q = float(np.quantile(pooled, nominal_coverage)) if pooled.size else None
        quantiles, counts, fallback = [], [], []
        for step in range(residual.shape[1]):
            values = residual[:, step][finite[:, step]]
            counts.append(int(values.size))
            if values.size >= min_step_samples:
                quantiles.append(float(np.quantile(values, nominal_coverage)))
                fallback.append("step")
            elif group_q is not None:
                quantiles.append(group_q)
                fallback.append("group_pooled")
            else:
                quantiles.append(per_step[step])
                fallback.append("unconditioned")
        groups[name] = {
            "quantiles_kw": quantiles, "sample_counts": counts,
            "pooled_quantile_kw": group_q, "finite_residual_count": int(pooled.size),
            "fallback_sources": fallback,
        }
    result.update(method="empirical_daylight_absolute_residual", conditioning="target_daylight",
                  groups=groups, min_step_samples=min_step_samples,
                  night_lower_bound=night_lower_bound,
                  step_specific=all(s == "step" for g in groups.values() for s in g["fallback_sources"]))
    return result


def _validated_daylight_mask(mask, shape):
    daylight = np.asarray(mask)
    if daylight.shape != shape or daylight.dtype != np.dtype(bool):
        raise ValueError("daylight_mask must be boolean and match prediction/target shape")
    return daylight


def apply_prediction_interval(
    prediction: np.ndarray, calibration: dict, daylight_mask: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, np.ndarray]:
    pred = np.asarray(prediction, dtype=float)
    quantiles = np.asarray(calibration["quantiles_kw"], dtype=float)
    if pred.ndim == 1:
        pred = pred[None, :]
    if pred.ndim != 2 or quantiles.ndim != 1 or pred.shape[1] != quantiles.size:
        raise ValueError("prediction horizon does not match interval calibration")
    if not np.isfinite(pred).all():
        raise ValueError("prediction contains non-finite values")
    if not np.isfinite(quantiles).all() or (quantiles < 0).any():
        raise ValueError("interval quantiles must be finite and non-negative")
    widths = quantiles[None, :]
    night_lower_bound = calibration.get("night_lower_bound", "empirical")
    if night_lower_bound not in {"empirical", "zero"}:
        raise ValueError("unsupported night lower bound")
    mask = None
    if calibration.get("conditioning") == "target_daylight":
        if daylight_mask is None:
            raise ValueError("daylight-conditioned intervals require a target daylight_mask")
        mask = np.asarray(daylight_mask)
        if mask.ndim == 1 and pred.shape[0] == 1:
            mask = mask[None, :]
        mask = _validated_daylight_mask(mask, pred.shape)
        grouped = []
        for name in ("daylight", "night"):
            q = np.asarray(calibration["groups"][name]["quantiles_kw"], dtype=float)
            if q.shape != quantiles.shape or not np.isfinite(q).all() or (q < 0).any():
                raise ValueError("conditioned quantiles must match horizon and be finite and non-negative")
            grouped.append(q)
        widths = np.where(mask, grouped[0][None, :], grouped[1][None, :])
    elif calibration.get("conditioning") is not None:
        raise ValueError("unsupported interval conditioning")
    lower = np.maximum(pred - widths, 0.0)
    if night_lower_bound == "zero":
        if mask is None:
            raise ValueError("zero night lower bound requires daylight conditioning")
        # A support augmentation: widen only the night lower endpoint. This
        # never removes previously covered labels, and does not change points.
        lower = np.where(mask, lower, 0.0)
    return lower, np.maximum(pred + widths, 0.0)
