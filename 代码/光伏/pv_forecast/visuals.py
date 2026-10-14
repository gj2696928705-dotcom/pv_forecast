"""PNG visualizations for human inspection of PV forecast records."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np
import pandas as pd


class VisualizationUnavailableError(RuntimeError):
    """Raised when the plotting backend cannot be loaded."""


def _pyplot():
    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.dates as mdates
        import matplotlib.pyplot as plt

        plt.rcParams["font.sans-serif"] = [
            "Microsoft YaHei",
            "SimHei",
            "DejaVu Sans",
        ]
        plt.rcParams["axes.unicode_minus"] = False
        return plt, mdates
    except Exception as exc:  # pragma: no cover - depends on runtime packages
        raise VisualizationUnavailableError(str(exc)) from exc


def _forecast_arrays(record: dict):
    values = record.get("forecast_values", [])
    intervals = record.get("prediction_interval", {}).get("values", [])
    timestamps = pd.DatetimeIndex(
        [pd.Timestamp(item["timestamp"]) for item in values]
    )
    prediction = np.asarray([float(item["value"]) for item in values], dtype=float)
    lower = np.asarray([float(item["lower"]) for item in intervals], dtype=float)
    upper = np.asarray([float(item["upper"]) for item in intervals], dtype=float)
    if not (len(timestamps) == len(prediction) == len(lower) == len(upper)):
        raise ValueError("forecast values and prediction interval lengths differ")
    return timestamps, prediction, lower, upper


def _ref(image_id: str, title: str, kind: str, path: Path, description: str) -> dict:
    return {
        "id": image_id,
        "kind": kind,
        "title": title,
        "path": path.as_posix(),
        "mime_type": "image/png",
        "description": description,
    }


def _save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fig.savefig(path, dpi=150, bbox_inches="tight")
    finally:
        import matplotlib.pyplot as plt
        plt.close(fig)


def _shade_daylight(ax, timestamps: pd.DatetimeIndex, ymax: float) -> None:
    if len(timestamps) == 0:
        return
    daylight = (timestamps.hour >= 5) & (timestamps.hour <= 20)
    if daylight.any():
        ax.fill_between(
            timestamps,
            0,
            max(ymax, 1.0),
            where=daylight,
            color="#f6c85f",
            alpha=0.12,
            label="参考白天时段（05:00–21:00）",
        )


def _plot_forecast_24h(record: dict, path: Path) -> None:
    plt, mdates = _pyplot()
    timestamps, prediction, lower, upper = _forecast_arrays(record)
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.fill_between(timestamps, lower, upper, color="#5dade2", alpha=0.25, label="预测区间")
    ax.plot(timestamps, prediction, color="#1f77b4", linewidth=2, label="预测功率")
    _shade_daylight(ax, timestamps, float(np.nanmax(upper)) if len(upper) else 1.0)
    ax.set_title("未来24小时光伏功率预测")
    ax.set_xlabel("时间")
    ax.set_ylabel("功率 (kW)")
    ax.grid(True, alpha=0.25)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M", tz=timestamps.tz))
    fig.autofmt_xdate()
    ax.legend(loc="upper right")
    _save(fig, path)


def _plot_history_forecast(
    history_index: Iterable,
    history_values: Sequence[float],
    record: dict,
    path: Path,
) -> None:
    plt, mdates = _pyplot()
    forecast_timestamps, prediction, lower, upper = _forecast_arrays(record)
    history_timestamps = pd.DatetimeIndex(list(history_index))
    history = np.asarray(history_values, dtype=float)
    if history_timestamps.size != history.size:
        raise ValueError("history timestamps and values lengths differ")
    keep = min(history.size, 3 * 96)
    history_timestamps = history_timestamps[-keep:]
    history = history[-keep:]
    fig, ax = plt.subplots(figsize=(12, 5))
    if keep:
        ax.plot(history_timestamps, history, color="#555555", linewidth=1.3, label="历史功率")
    ax.axvline(forecast_timestamps[0], color="#d62728", linestyle="--", label="预测起点")
    ax.fill_between(forecast_timestamps, lower, upper, color="#5dade2", alpha=0.25, label="预测区间")
    ax.plot(forecast_timestamps, prediction, color="#1f77b4", linewidth=2, label="预测功率")
    ax.set_title("历史功率与未来预测衔接")
    ax.set_xlabel("时间")
    ax.set_ylabel("功率 (kW)")
    ax.grid(True, alpha=0.25)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M", tz=forecast_timestamps.tz))
    fig.autofmt_xdate()
    ax.legend(loc="upper right")
    _save(fig, path)


def _plot_forecast_profile(record: dict, path: Path) -> None:
    plt, _ = _pyplot()
    _, prediction, lower, upper = _forecast_arrays(record)
    interval_minutes = int(record.get("interval_minutes", 15))
    relative_hours = np.arange(len(prediction), dtype=float) * interval_minutes / 60.0
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.fill_between(relative_hours, lower, upper, color="#5dade2", alpha=0.25, label="预测区间")
    ax.plot(relative_hours, prediction, color="#2ca02c", linewidth=2, label="预测功率")
    ax.set_title("预测窗口日内功率分布")
    ax.set_xlabel("预测起点后的小时")
    ax.set_ylabel("功率 (kW)")
    ax.set_xlim(0, max(24, relative_hours[-1] if len(relative_hours) else 24))
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right")
    _save(fig, path)


def generate_forecast_visualizations(
    history_index: Iterable,
    history_values: Sequence[float],
    record: dict,
    output_dir: Path,
    relative_dir: str = "visualizations",
) -> tuple[list[dict], list[dict]]:
    """Generate the three standard forecast figures and return refs/errors."""

    if not record.get("forecast_values"):
        return [], []
    output_dir = Path(output_dir)
    forecast_id = str(record["forecast_id"])
    jobs = [
        (
            "forecast_24h",
            "未来24小时光伏功率预测",
            "forecast_interval",
            "预测值及经验预测区间",
            lambda path: _plot_forecast_24h(record, path),
        ),
        (
            "history_forecast",
            "历史功率与未来预测衔接",
            "history_forecast",
            "最近历史功率与未来预测的衔接",
            lambda path: _plot_history_forecast(history_index, history_values, record, path),
        ),
        (
            "forecast_profile",
            "预测窗口日内功率分布",
            "forecast_profile",
            "预测窗口内的日内功率变化",
            lambda path: _plot_forecast_profile(record, path),
        ),
    ]
    refs, errors = [], []
    for image_id, title, kind, description, job in jobs:
        filename = f"{forecast_id}_{image_id}.png"
        path = output_dir / filename
        try:
            job(path)
            refs.append(_ref(image_id, title, kind, Path(relative_dir) / filename, description))
        except Exception as exc:
            errors.append({"id": image_id, "message": str(exc)})
    return refs, errors


def generate_actual_vs_forecast_visualization(
    actual_timestamps: Iterable,
    actual_values: Sequence[float],
    record: dict,
    output_dir: Path,
    relative_dir: str = "visualizations",
) -> tuple[Optional[dict], list[dict]]:
    """Generate an actual-vs-forecast plot for labelled evaluation data."""

    if not record.get("forecast_values"):
        return None, []
    try:
        plt, mdates = _pyplot()
        forecast_timestamps, prediction, lower, upper = _forecast_arrays(record)
        actual_index = pd.DatetimeIndex(list(actual_timestamps))
        actual = np.asarray(actual_values, dtype=float)
        if len(actual_index) != len(actual) or len(actual_index) != len(prediction):
            raise ValueError("actual and forecast lengths differ")
        output_dir = Path(output_dir)
        filename = f"{record['forecast_id']}_actual_vs_forecast.png"
        path = output_dir / filename
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.fill_between(forecast_timestamps, lower, upper, color="#5dade2", alpha=0.2, label="预测区间")
        ax.plot(forecast_timestamps, actual, color="#333333", linewidth=1.6, label="实际功率")
        ax.plot(forecast_timestamps, prediction, color="#d62728", linewidth=1.8, label="预测功率")
        ax.set_title("实际功率与预测功率对比")
        ax.set_xlabel("时间")
        ax.set_ylabel("功率 (kW)")
        ax.grid(True, alpha=0.25)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M", tz=forecast_timestamps.tz))
        fig.autofmt_xdate()
        ax.legend(loc="upper right")
        _save(fig, path)
        return (
            _ref(
                "actual_vs_forecast",
                "实际功率与预测功率对比",
                "actual_vs_forecast",
                Path(relative_dir) / filename,
                "评估窗口的实际值、预测值及预测区间",
            ),
            [],
        )
    except Exception as exc:
        return None, [{"id": "actual_vs_forecast", "message": str(exc)}]


def generate_training_history_visualizations(history, output_dir):
    """Updated each epoch; training metrics are online, validation is fixed-weight."""
    refs, errors = [], []
    if not history:
        return refs, errors
    epochs = [row["epoch"] for row in history]
    for image_id, title in (("training_loss", "GRU 训练与验证损失"),):
        try:
            plt, _ = _pyplot()
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(epochs, [r["loss"] for r in history], label="训练（在线统计）", color="#2471a3")
            if "val_loss" in history[0]:
                val = [r["val_loss"] for r in history]
                ax.plot(epochs, val, label="验证", color="#e67e22")
                best = int(np.nanargmin([np.nan if x is None else x for x in val]))
                ax.axvline(epochs[best], linestyle="--", color="#2e8b57", label=f"最佳轮次 {epochs[best]}")
            ax.set(xlabel="Epoch", ylabel="归一化 MSE", title=title)
            ax.grid(alpha=.25)
            ax.legend()
            filename = image_id + ".png"
            _save(fig, Path(output_dir) / filename)
            refs.append(_ref(image_id, title, "training", Path("visualizations") / filename,
                             "每轮更新；验证指标在整套验证窗口上计算；训练指标为更新权重过程中的累计统计"))
        except Exception as exc:
            errors.append({"id": image_id, "message": str(exc)})
    return refs, errors


def generate_training_evaluation_visualizations(test, prediction, baseline_prediction,
                                               output_dir, interval_minutes=15):
    """Independent test illustrations, never used to select the model."""
    refs, errors = [], []
    for image_id, title in (("gru_test_comparison", "GRU 留出测试集：前 7 个不重叠预测窗口"),):
        try:
            plt, mdates = _pyplot()
            fig, ax = plt.subplots(figsize=(12, 5))
            # Choose by timestamp so data gaps cannot cause overlapping illustrations.
            rows, next_start = [], None
            for i, start in enumerate(test.forecast_starts):
                if next_start is None or start >= next_start:
                    rows.append(i)
                    next_start = start + pd.Timedelta(minutes=interval_minutes * test.target.shape[1])
                if len(rows) == 7:
                    break
            for n, i in enumerate(rows):
                stamps = pd.date_range(test.forecast_starts[i], periods=test.target.shape[1], freq=f"{interval_minutes}min")
                ax.plot(stamps, test.target[i], color="#333333", label="实际功率" if n == 0 else None)
                ax.plot(stamps, prediction[i], color="#2471a3", label="GRU" if n == 0 else None)
                ax.plot(stamps, baseline_prediction[i], color="#e67e22", alpha=.7, linestyle="--",
                        label="最佳基线" if n == 0 else None)
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M", tz=stamps.tz))
            ax.set(xlabel="本地时间", ylabel="功率 (kW)")
            ax.legend()
            fig.autofmt_xdate()
            ax.set_title(title)
            ax.grid(axis="y", alpha=.25)
            filename = image_id + ".png"
            _save(fig, Path(output_dir) / filename)
            refs.append(_ref(image_id, title, "evaluation", Path("visualizations") / filename,
                             "完整指标见 metrics.json；功率曲线展示固定选取的测试窗口"))
        except Exception as exc:
            errors.append({"id": image_id, "message": str(exc)})
    return refs, errors
