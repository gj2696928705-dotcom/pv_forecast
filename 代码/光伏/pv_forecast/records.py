"""Forecast record construction and schema validation."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np
import pandas as pd


def _iso(value) -> str:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("Asia/Shanghai")
    return timestamp.isoformat()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def build_forecast_record(
    forecast_id: str,
    forecast_values: Sequence[float],
    forecast_start,
    interval_minutes: int,
    asset_scope,
    model_id: str,
    model_version: str,
    model_published_at: Optional[str],
    training_data_window: dict,
    calibration: Optional[dict],
    input_quality: dict,
    status: str = "available",
    confidence_level: float = 0.8,
    fallback_forecast_ref: Optional[str] = None,
    audit_refs: Optional[Iterable[str]] = None,
    limitations: Optional[Sequence[str]] = None,
    lower: Optional[Sequence[float]] = None,
    upper: Optional[Sequence[float]] = None,
    generated_at: Optional[str] = None,
    visualizations: Optional[Sequence[dict]] = None,
    visualization_errors: Optional[Sequence[dict]] = None,
) -> dict:
    values = np.asarray(forecast_values, dtype=float)
    start = pd.Timestamp(forecast_start)
    if start.tzinfo is None:
        start = start.tz_localize("Asia/Shanghai")
    timestamps = [start + pd.Timedelta(minutes=interval_minutes * i) for i in range(values.size)]
    value_items = [
        {"timestamp": _iso(ts), "value": float(max(0.0, value)), "unit": "kW"}
        for ts, value in zip(timestamps, values)
    ]
    if lower is None or upper is None:
        lower_values = np.maximum(values, 0.0)
        upper_values = np.maximum(values, 0.0)
    else:
        lower_values = np.maximum(np.asarray(lower, dtype=float), 0.0)
        upper_values = np.maximum(np.asarray(upper, dtype=float), 0.0)
    interval_items = [
        {
            "timestamp": _iso(ts),
            "lower": float(lo),
            "upper": float(hi),
            "unit": "kW",
        }
        for ts, lo, hi in zip(timestamps, lower_values, upper_values)
    ]
    forecast_end = timestamps[-1] if timestamps else start
    limitations = list(limitations or [])
    if calibration and not calibration.get("step_specific", False):
        limitations.append("部分区间使用同昼夜组或未分组残差分位数，未达到逐预测步样本数门槛"
                           if calibration.get("conditioning") else
                           "区间使用全局残差分位数，未达到逐预测步样本数门槛")
    if calibration and calibration.get("conditioning"):
        limitations.append("经验区间按目标时刻昼夜分组；重叠窗口并非独立样本，不保证未来覆盖率")
    if calibration and calibration.get("night_lower_bound") == "zero":
        limitations.append("夜间区间下界包含零出力；昼夜判定沿用太阳位置或近似时段，覆盖率仍需验证")
    return {
        "forecast_id": forecast_id,
        "forecast_type": "pv_power",
        "asset_scope": asset_scope,
        "generated_at": generated_at or _utc_now(),
        "forecast_start_at": _iso(start),
        "forecast_end_at": _iso(forecast_end),
        "interval_minutes": interval_minutes,
        "forecast_values": value_items,
        "prediction_interval": {
            "method": (calibration or {}).get("method", "unavailable"),
            "nominal_coverage": (calibration or {}).get(
                "nominal_coverage", confidence_level
            ),
            "values": interval_items,
        },
        "confidence": {
            "level": (calibration or {}).get("nominal_coverage", confidence_level),
            "source": "validation_residual_calibration" if calibration else "unavailable",
            "limitations": limitations,
        },
        "input_data_quality_summary": input_quality,
        "model_id": model_id,
        "model_version": model_version,
        "model_published_at": model_published_at,
        "training_data_window": training_data_window,
        "fallback_forecast_ref": fallback_forecast_ref,
        "status": status,
        "audit_refs": list(audit_refs or []),
        "visualizations": list(visualizations or []),
        "visualization_errors": list(visualization_errors or []),
    }


def validate_forecast_record(record: dict) -> None:
    schema_path = Path(__file__).resolve().parent / "schema" / "forecast_record.schema.json"
    try:
        import jsonschema  # type: ignore
    except ImportError:  # pragma: no cover - optional in minimal runtime
        required = [
            "forecast_id", "forecast_type", "forecast_values", "prediction_interval",
            "model_id", "model_version", "status",
        ]
        missing = [field for field in required if field not in record]
        if missing:
            raise ValueError(f"forecast record missing fields: {missing}")
        return
    with schema_path.open("r", encoding="utf-8") as handle:
        schema = json.load(handle)
    jsonschema.validate(record, schema)
