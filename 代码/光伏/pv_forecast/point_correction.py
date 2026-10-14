"""Training-only linear correction of a frozen GRU using causal covariates."""
from __future__ import annotations

import numpy as np


class SolarWeatherRidge:
    """One standardized ridge per lead, fitted from all daytime train targets.

    No target weather is used. Future inputs are saved calendar/solar features;
    recent weather summaries come only from the historical branch. Night
    predictions are zero under the explicit photovoltaic daylight assumption.
    """
    def __init__(self, alpha=0.01):
        if not np.isfinite(alpha) or alpha <= 0:
            raise ValueError("alpha must be finite and positive")
        self.alpha = float(alpha)

    @staticmethod
    def features(history, future, raw_prediction, history_names, future_names):
        history, future, raw = [np.asarray(v, dtype=float) for v in (history, future, raw_prediction)]
        if history.ndim != 3 or future.ndim != 3 or raw.shape != future.shape[:2] or len(history) != len(future):
            raise ValueError("invalid point-correction input shapes")
        if history.shape[-1] != len(history_names) or future.shape[-1] != len(future_names):
            raise ValueError("point-correction feature names must match input axes")
        if history.shape[1] < 96 or raw.shape[1] != 96:
            raise ValueError("point correction requires 15-minute histories and a 96-step horizon")
        if "solar_elevation_deg" not in future_names or "is_daylight" not in future_names:
            raise ValueError("point correction requires configured solar geometry")
        elevation = future[..., list(future_names).index("solar_elevation_deg")]
        day = elevation > 0
        solar = np.maximum(np.sin(np.deg2rad(elevation)), 0)
        # Future observed meteorology is deliberately absent. Feature order is
        # a persisted contract, so newly added archive columns cannot slip in.
        calendar_names = ("sin_time_of_day", "cos_time_of_day", "sin_day_of_year", "cos_day_of_year")
        calendar = future[..., [list(future_names).index(name) for name in calendar_names]]
        context = np.concatenate([history[:, -1], history[:, -4:].mean(axis=1),
                                  history[:, -24:].mean(axis=1), history[:, -96:].mean(axis=1)], axis=1)
        context = np.broadcast_to(context[:, None, :], (*raw.shape, context.shape[-1]))
        previous_day = history[:, -96:, 0]
        slots = history.shape[1] // 96
        same_slot = history[:, -slots*96:, 0].reshape(len(history), slots, 96).mean(axis=1)
        x = np.concatenate([raw[..., None], previous_day[..., None], same_slot[..., None],
                            solar[..., None], (solar**2)[..., None], calendar,
                            context, context*solar[..., None]], axis=-1)
        if not np.isfinite(x).all() or not np.isfinite(raw).all():
            raise ValueError("point-correction inputs must be finite")
        return x, day

    def fit(self, history, future, raw_prediction, target, history_names, future_names, batch_windows=512):
        if not isinstance(batch_windows, int) or isinstance(batch_windows, bool) or batch_windows < 1:
            raise ValueError("batch_windows must be a positive integer")
        target = np.asarray(target, dtype=float)
        if target.shape != np.asarray(raw_prediction).shape or not np.isfinite(target).all() or (target < 0).any():
            raise ValueError("correction targets must be finite nonnegative powers with the forecast shape")
        self.history_names, self.future_names = tuple(history_names), tuple(future_names)
        self.history_steps = np.asarray(history).shape[1]
        self.window_count = len(target)
        stats = None
        for start in range(0, len(target), batch_windows):
            stop = min(len(target), start+batch_windows)
            x, mask = self.features(history[start:stop], future[start:stop], raw_prediction[start:stop],
                                    history_names, future_names)
            h, p = x.shape[1:]
            if stats is None:
                stats = [np.zeros(h), np.zeros((h, p)), np.zeros(h),
                         np.zeros((h, p, p)), np.zeros((h, p))]
            n, sx, sy, sxx, sxy = stats
            for lead in range(h):
                a, b = x[mask[:, lead], lead], target[start:stop, lead][mask[:, lead]]
                n[lead] += len(b)
                sx[lead] += a.sum(axis=0)
                sy[lead] += b.sum()
                sxx[lead] += a.T @ a
                sxy[lead] += a.T @ b
        if stats is None or (stats[0] < 2).any():
            raise ValueError("each lead requires at least two daytime training samples")
        n, sx, sy, sxx, sxy = stats
        self.daylight_counts = n.astype(int)
        self.mean = sx / n[:, None]
        self.target_mean = sy / n
        covariance = sxx/n[:, None, None] - self.mean[:, :, None]*self.mean[:, None, :]
        variance = np.maximum(np.diagonal(covariance, axis1=1, axis2=2), 0)
        self.scale = np.sqrt(variance)
        self.scale[self.scale < 1e-8] = 1
        normal = covariance / self.scale[:, :, None] / self.scale[:, None, :]
        rhs = (sxy/n[:, None]-self.mean*self.target_mean[:, None])/self.scale
        self.normal, self.rhs = normal, rhs
        self.coefficients = np.stack([np.linalg.solve(a+self.alpha*np.eye(len(a)), b)
                                      for a, b in zip(self.normal, self.rhs)])
        if not np.isfinite(self.coefficients).all():
            raise ValueError("non-finite correction fit")
        return self

    def with_alpha(self, alpha):
        """Reuse exactly the same training statistics for another fixed penalty."""
        import copy
        result = copy.deepcopy(self)
        result.alpha = SolarWeatherRidge(alpha).alpha
        result.coefficients = np.stack([np.linalg.solve(a+result.alpha*np.eye(len(a)), b)
                                        for a, b in zip(result.normal, result.rhs)])
        return result

    def predict(self, history, future, raw_prediction, history_names, future_names):
        if not hasattr(self, "coefficients"):
            raise ValueError("point correction has not been fitted")
        if (tuple(history_names) != self.history_names or tuple(future_names) != self.future_names
                or np.asarray(history).shape[1] != self.history_steps):
            raise ValueError("point-correction input contract does not match its training")
        x, day = self.features(history, future, raw_prediction, history_names, future_names)
        result = ((x-self.mean[None])/self.scale[None]*self.coefficients[None]).sum(axis=-1)+self.target_mean[None]
        if not np.isfinite(result).all():
            raise ValueError("non-finite point-correction output")
        return np.where(day, np.maximum(result, 0), 0)
