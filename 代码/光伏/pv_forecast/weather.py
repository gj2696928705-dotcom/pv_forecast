"""Weather forecasts selected by issue time, never by future observations."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data import DataValidationError, _normalise_timestamp, _read_source


class WeatherForecastArchive:
    """One site per input; columns: issued_at, timestamp, configured weather.

    Each issuance is a coherent forecast vintage. Missing points in the latest
    eligible vintage are unavailable; values from different vintages are not mixed.
    """
    def __init__(self, source, columns, timezone="Asia/Shanghai"):
        frame = _read_source(source)
        self.columns = tuple(columns)
        required = {"issued_at", "timestamp", *self.columns}
        if not self.columns or not required.issubset(frame.columns):
            raise DataValidationError(f"weather forecast requires columns: {sorted(required)}")
        frame["issued_at"] = _normalise_timestamp(frame["issued_at"], timezone)
        frame["timestamp"] = _normalise_timestamp(frame["timestamp"], timezone)
        if frame.duplicated(["issued_at", "timestamp"]).any():
            raise DataValidationError("duplicate weather issue/valid timestamp pair")
        for column in self.columns:
            values = pd.to_numeric(frame[column], errors="coerce")
            if not np.isfinite(values).all():
                raise DataValidationError(f"weather forecast {column} must be finite")
            frame[column] = values.astype(float)
        self.vintages = {
            issue: group.set_index("timestamp")[list(self.columns)].sort_index()
            for issue, group in frame.groupby("issued_at", sort=True)
        }
        self.issues = pd.DatetimeIndex(list(self.vintages))

    def at(self, origin, index):
        origin = pd.Timestamp(origin)
        position = self.issues.searchsorted(origin, side="right") - 1
        if position < 0:
            return None, None
        issue = self.issues[position]
        selected = self.vintages[issue].reindex(index)
        if not np.isfinite(selected.to_numpy()).all():
            return None, issue
        return selected, issue


def solar_elevation(index, latitude, longitude):
    """NOAA-style geometric solar elevation; coordinates are explicit inputs.

    Uses UTC to avoid confusing the configured display timezone with solar time.
    Atmospheric refraction and terrain shading are intentionally not modelled.
    """
    stamps = pd.DatetimeIndex(index)
    if stamps.tz is None:
        raise ValueError("solar position requires timezone-aware timestamps")
    utc = stamps.tz_convert("UTC")
    hour = utc.hour + utc.minute / 60 + utc.second / 3600
    gamma = 2 * np.pi / 365 * (utc.dayofyear - 1 + (hour - 12) / 24)
    equation = 229.18 * (0.000075 + 0.001868 * np.cos(gamma) - 0.032077 * np.sin(gamma)
                        - 0.014615 * np.cos(2 * gamma) - 0.040849 * np.sin(2 * gamma))
    declination = (0.006918 - 0.399912 * np.cos(gamma) + 0.070257 * np.sin(gamma)
                   - 0.006758 * np.cos(2 * gamma) + 0.000907 * np.sin(2 * gamma)
                   - 0.002697 * np.cos(3 * gamma) + 0.00148 * np.sin(3 * gamma))
    angle = np.deg2rad((hour * 60 + equation + 4 * longitude) % 1440 / 4 - 180)
    lat = np.deg2rad(latitude)
    sine = np.sin(lat) * np.sin(declination) + np.cos(lat) * np.cos(declination) * np.cos(angle)
    return np.rad2deg(np.arcsin(np.clip(sine, -1, 1)))
