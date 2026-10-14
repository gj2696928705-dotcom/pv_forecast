"""Input validation, time alignment and supervised-window construction."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd


class DataValidationError(ValueError):
    """Raised when input data cannot be safely used for forecasting."""


@dataclass(frozen=True)
class InputSpec:
    """Canonical input description.

    ``pv_power_kw`` is an instantaneous power series.  Energy columns are not
    guessed or silently converted; callers must map them explicitly before
    loading, or use ``input_kind='interval_energy'`` with a duration.
    """

    timestamp_col: str = "timestamp"
    target_col: str = "pv_power_kw"
    weather_columns: Tuple[str, ...] = ()
    history_weather_columns: Tuple[str, ...] = ()
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    coordinate_basis: str = "not_configured"
    timezone: str = "Asia/Shanghai"
    input_kind: str = "power"
    interval_minutes: Optional[int] = None
    column_map: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if (self.latitude is None) != (self.longitude is None):
            raise ValueError("latitude and longitude must be specified together")
        if self.latitude is not None and not (-90 <= self.latitude <= 90 and -180 <= self.longitude <= 180):
            raise ValueError("invalid latitude/longitude")
        if self.target_col in (*self.history_weather_columns, *self.weather_columns):
            raise ValueError("target column cannot be a weather feature")
        if self.input_kind not in {"power", "interval_energy"}:
            raise ValueError("input_kind must be 'power' or 'interval_energy'")
        if self.input_kind == "interval_energy" and (
            self.interval_minutes is None or self.interval_minutes <= 0
        ):
            raise ValueError(
                "interval_energy input requires a positive interval_minutes"
            )


@dataclass
class QualitySummary:
    source_rows: int
    regular_rows: int
    missing_target_rows: int
    missing_weather_rows: int
    interpolated_target_rows: int
    interpolated_weather_rows: int
    first_timestamp: Optional[str]
    last_timestamp: Optional[str]
    missing_history_weather_columns: Tuple[str, ...] = ()

    def as_dict(self) -> dict:
        return {
            "source_rows": self.source_rows,
            "regular_rows": self.regular_rows,
            "missing_target_rows": self.missing_target_rows,
            "missing_weather_rows": self.missing_weather_rows,
            "interpolated_target_rows": self.interpolated_target_rows,
            "interpolated_weather_rows": self.interpolated_weather_rows,
            "first_timestamp": self.first_timestamp,
            "last_timestamp": self.last_timestamp,
            "missing_history_weather_columns": list(self.missing_history_weather_columns),
        }


@dataclass
class RegularizedData:
    frame: pd.DataFrame
    quality: QualitySummary
    interval_minutes: int
    timezone: str


@dataclass
class WindowedData:
    history_power: np.ndarray
    future_features: np.ndarray
    target: np.ndarray
    forecast_starts: Sequence[pd.Timestamp]
    feature_names: Sequence[str]
    history_features: Optional[np.ndarray] = None
    history_feature_names: Sequence[str] = ()
    weather_issued_at: Sequence = ()

    @property
    def gru_history(self) -> np.ndarray:
        return self.history_power if self.history_features is None else self.history_features

    @property
    def size(self) -> int:
        return int(self.target.shape[0])


def _read_source(source: Union[str, Path, pd.DataFrame]) -> pd.DataFrame:
    if isinstance(source, pd.DataFrame):
        return source.copy()
    path = Path(source)
    if not path.exists():
        raise DataValidationError(f"input file does not exist: {path}")
    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    if path.suffix.lower() in {".csv", ".txt"}:
        return pd.read_csv(path)
    raise DataValidationError(
        f"unsupported input format {path.suffix!r}; use CSV or Excel"
    )


def _apply_column_map(frame: pd.DataFrame, spec: InputSpec) -> pd.DataFrame:
    if not spec.column_map:
        return frame
    # The mapping is source-name -> canonical-name, and is intentionally
    # explicit so a column named pv_kwh is never mistaken for kW.
    return frame.rename(columns=dict(spec.column_map))


def _normalise_timestamp(series: pd.Series, timezone: str) -> pd.DatetimeIndex:
    parsed = pd.to_datetime(series, errors="coerce")
    if parsed.isna().any():
        bad = int(parsed.isna().sum())
        raise DataValidationError(f"timestamp contains {bad} invalid value(s)")
    index = pd.DatetimeIndex(parsed)
    if index.tz is None:
        return index.tz_localize(timezone)
    return index.tz_convert(timezone)


def _validate_numeric(series: pd.Series, name: str, allow_missing: bool) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    if not allow_missing and values.isna().any():
        raise DataValidationError(
            f"{name} contains {int(values.isna().sum())} non-numeric or missing value(s)"
        )
    finite = np.isfinite(values.dropna().to_numpy(dtype=float))
    if not finite.all():
        raise DataValidationError(f"{name} contains non-finite values")
    return values.astype(float)


def load_and_regularize(
    source: Union[str, Path, pd.DataFrame],
    spec: InputSpec = InputSpec(),
    interval_minutes: int = 15,
    max_interpolation_gap_minutes: int = 30,
    available_before=None,
) -> RegularizedData:
    """Load, validate and align a PV series to a regular 15-minute grid.

    Only short gaps are filled from prior observations. A gap longer than
    ``max_interpolation_gap_minutes`` remains NaN and is later excluded from
    supervised windows or causes a degraded/unavailable forecast. When an
    origin is supplied, discard observations at/after it before numeric
    validation, gap filling and quality summaries.
    """

    if interval_minutes <= 0:
        raise ValueError("interval_minutes must be positive")
    frame = _apply_column_map(_read_source(source), spec)
    required = {spec.timestamp_col, spec.target_col}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise DataValidationError(f"missing required column(s): {', '.join(missing)}")

    timestamps = _normalise_timestamp(frame[spec.timestamp_col], spec.timezone)
    if available_before is not None:
        cutoff = pd.Timestamp(available_before)
        if cutoff.tzinfo is None:
            cutoff = cutoff.tz_localize(spec.timezone)
        cutoff = cutoff.tz_convert(spec.timezone)
        eligible = timestamps < cutoff
        frame = frame.loc[eligible].copy()
        timestamps = timestamps[eligible]
        if frame.empty:
            present = [c for c in spec.history_weather_columns if c in frame.columns]
            missing_weather = tuple(c for c in spec.history_weather_columns if c not in frame.columns)
            empty = pd.DataFrame(columns=[spec.target_col, *present],
                                 index=pd.DatetimeIndex([], tz=spec.timezone), dtype=float)
            return RegularizedData(empty, QualitySummary(0, 0, 0, 0, 0, 0, None, None, missing_weather),
                                   interval_minutes, spec.timezone)
    if frame.empty:
        raise DataValidationError("input contains no observations")
    if timestamps.has_duplicates:
        raise DataValidationError("timestamp contains duplicate values")
    if not timestamps.is_monotonic_increasing:
        raise DataValidationError("timestamp must be strictly increasing")

    target = _validate_numeric(frame[spec.target_col], spec.target_col, False)
    if (target < 0).any():
        raise DataValidationError(f"{spec.target_col} contains negative values")
    work = pd.DataFrame({spec.target_col: target.to_numpy()}, index=timestamps)

    weather_present = []
    # Future covariates come ONLY from an issued forecast archive. This table
    # may contain observations, which are eligible only as historical inputs.
    for column in spec.history_weather_columns:
        if column not in frame.columns:
            continue
        weather_present.append(column)
        work[column] = _validate_numeric(frame[column], column, True).to_numpy()

    # Explicit interval-energy conversion.  The source values are energy per
    # interval; kW is energy divided by hours.
    if spec.input_kind == "interval_energy":
        work[spec.target_col] = work[spec.target_col] / (spec.interval_minutes / 60.0)

    frequency = f"{interval_minutes}min"
    start = work.index.min().floor(frequency)
    end = work.index.max().ceil(frequency)
    regular_index = pd.date_range(start=start, end=end, freq=frequency, tz=spec.timezone)
    regular = work.reindex(regular_index)
    max_steps = max(0, int(max_interpolation_gap_minutes // interval_minutes) - 1)

    def causal_short_fill(series):
        missing = series.isna()
        groups = missing.ne(missing.shift()).cumsum()
        length = missing.groupby(groups).transform("sum")
        eligible = missing & (length <= max_steps)
        # No interpolation from observations after a forecast origin/split.
        return series.mask(eligible, series.ffill())

    target_before = regular[spec.target_col].isna()
    regular[spec.target_col] = causal_short_fill(regular[spec.target_col])
    target_after = regular[spec.target_col].isna()
    interpolated_target = int((target_before & ~target_after).sum())

    interpolated_weather = 0
    for column in weather_present:
        before = regular[column].isna()
        regular[column] = causal_short_fill(regular[column])
        after = regular[column].isna()
        interpolated_weather += int((before & ~after).sum())

    missing_history_weather = tuple(column for column in spec.history_weather_columns if column not in weather_present)
    missing_weather_rows = len(regular) if missing_history_weather else (
        int(regular[weather_present].isna().any(axis=1).sum()) if weather_present else 0
    )
    quality = QualitySummary(
        source_rows=len(work),
        regular_rows=len(regular),
        missing_target_rows=int(regular[spec.target_col].isna().sum()),
        missing_weather_rows=missing_weather_rows,
        interpolated_target_rows=interpolated_target,
        interpolated_weather_rows=interpolated_weather,
        first_timestamp=regular.index.min().isoformat(),
        last_timestamp=regular.index.max().isoformat(),
        missing_history_weather_columns=missing_history_weather,
    )
    return RegularizedData(
        frame=regular,
        quality=quality,
        interval_minutes=interval_minutes,
        timezone=spec.timezone,
    )


def feature_frame(
    frame: pd.DataFrame,
    target_col: str = "pv_power_kw",
    weather_columns: Iterable[str] = (),
    latitude=None,
    longitude=None,
) -> Tuple[pd.DataFrame, Sequence[str]]:
    """Build future-known calendar/weather features."""

    index = frame.index
    minute_of_day = index.hour * 60 + index.minute
    day_of_year = index.dayofyear
    result = pd.DataFrame(index=index)
    result["sin_time_of_day"] = np.sin(2 * np.pi * minute_of_day / 1440.0)
    result["cos_time_of_day"] = np.cos(2 * np.pi * minute_of_day / 1440.0)
    result["sin_day_of_year"] = np.sin(2 * np.pi * day_of_year / 365.25)
    result["cos_day_of_year"] = np.cos(2 * np.pi * day_of_year / 365.25)
    result["is_daylight"] = ((index.hour >= 5) & (index.hour <= 20)).astype(float)
    if latitude is not None and longitude is not None:
        from .weather import solar_elevation
        result["solar_elevation_deg"] = solar_elevation(index, latitude, longitude)
        result["is_daylight"] = (result["solar_elevation_deg"] > 0).astype(float)
    names = list(result.columns)
    for column in weather_columns:
        if column in frame.columns:
            result[column] = frame[column].astype(float)
            names.append(column)
    # The target is deliberately excluded from future features.
    if target_col in result.columns:
        result = result.drop(columns=[target_col])
    return result, names


def build_windows(
    regular: RegularizedData,
    target_start: int,
    target_end: int,
    history_days: int = 3,
    horizon_steps: int = 96,
    target_col: str = "pv_power_kw",
    weather_columns: Iterable[str] = (),
    history_weather_columns: Iterable[str] = (),
    latitude=None,
    longitude=None,
    weather_archive=None,
) -> WindowedData:
    """Create windows whose targets stay entirely inside one time split."""

    frame = regular.frame
    steps_per_day = int(round(24 * 60 / regular.interval_minutes))
    history_steps = history_days * steps_per_day
    if weather_columns and weather_archive is None:
        raise DataValidationError("future weather requires an issued forecast archive; observations are not forecasts")
    missing = set(history_weather_columns).difference(frame.columns)
    if missing:
        raise DataValidationError(f"missing historical weather columns: {sorted(missing)}")
    features, names = feature_frame(frame, target_col, (), latitude, longitude)
    names = [*names, *weather_columns]
    history_names = [target_col, *history_weather_columns]
    historical = frame[history_names].to_numpy(dtype=float)
    power = frame[target_col].to_numpy(dtype=float)
    future = features.to_numpy(dtype=float)
    histories, futures, targets, starts, past_features, issues = [], [], [], [], [], []
    lower = max(target_start, history_steps)
    # The last target, not merely its first point, must precede the boundary.
    upper = min(target_end, len(frame)) - horizon_steps + 1
    for start in range(lower, upper):
        hist = power[start - history_steps : start]
        cov = future[start : start + horizon_steps]
        issue = None
        if weather_columns:
            forecast, issue = weather_archive.at(frame.index[start], frame.index[start : start + horizon_steps])
            if forecast is None:
                continue
            cov = np.column_stack([cov, forecast[list(weather_columns)].to_numpy()])
        past = historical[start - history_steps : start]
        y = power[start : start + horizon_steps]
        if not (np.isfinite(past).all() and np.isfinite(cov).all() and np.isfinite(y).all()):
            continue
        histories.append(hist[:, None])
        futures.append(cov)
        targets.append(y)
        starts.append(frame.index[start])
        past_features.append(past)
        issues.append(issue)
    if not histories:
        return WindowedData(
            np.empty((0, history_steps, 1)),
            np.empty((0, horizon_steps, len(names))),
            np.empty((0, horizon_steps)),
            [],
            names,
            np.empty((0, history_steps, len(history_names))), history_names, [],
        )
    return WindowedData(
        np.asarray(histories, dtype=float),
        np.asarray(futures, dtype=float),
        np.asarray(targets, dtype=float),
        starts,
        names,
        np.asarray(past_features, dtype=float), history_names, issues,
    )


def split_boundaries(length: int, train_ratio: float = 0.70, val_ratio: float = 0.15) -> Tuple[int, int]:
    if length < 3:
        raise DataValidationError("at least three time partitions are required")
    if not (0 < train_ratio < 1 and 0 < val_ratio < 1 and train_ratio + val_ratio < 1):
        raise ValueError("invalid split ratios")
    train_end = int(length * train_ratio)
    val_end = int(length * (train_ratio + val_ratio))
    return train_end, val_end
