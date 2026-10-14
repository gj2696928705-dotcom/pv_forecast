"""Simple, transparent forecast baselines."""

from __future__ import annotations

import numpy as np


class BaselineForecaster:
    def __init__(self, strategy: str = "same_slot_mean", steps_per_day: int = 96):
        if strategy not in {"persistence", "previous_day", "same_slot_mean"}:
            raise ValueError(f"unsupported baseline strategy: {strategy}")
        self.strategy = strategy
        self.steps_per_day = steps_per_day

    def predict(self, history_power: np.ndarray, horizon_steps: int) -> np.ndarray:
        history = np.asarray(history_power, dtype=float)
        if history.ndim == 3:
            history = history[..., 0]
        if history.ndim == 1:
            history = history[None, :]
        if history.ndim != 2 or history.shape[1] == 0:
            raise ValueError("history_power must have shape (samples, history_steps)")
        outputs = []
        for row in history:
            if self.strategy == "persistence":
                prediction = np.repeat(row[-1], horizon_steps)
            elif self.strategy == "previous_day" and row.size >= self.steps_per_day:
                prediction = row[-self.steps_per_day :]
            else:
                chunks = row[-(3 * self.steps_per_day) :]
                usable = chunks[: (chunks.size // self.steps_per_day) * self.steps_per_day]
                if usable.size >= self.steps_per_day:
                    days = usable.reshape(-1, self.steps_per_day)
                    prediction = np.nanmean(days, axis=0)
                else:
                    prediction = np.repeat(row[-1], self.steps_per_day)
            if prediction.size != horizon_steps:
                prediction = np.resize(prediction, horizon_steps)
            outputs.append(np.maximum(prediction, 0.0))
        return np.asarray(outputs, dtype=float)
