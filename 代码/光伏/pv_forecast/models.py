"""Optional TensorFlow GRU model with a reference-inspired dual input shape."""

from __future__ import annotations

from dataclasses import dataclass
import csv
import json
from pathlib import Path
import tempfile
import time
from typing import Optional
import zipfile

import numpy as np

from .errors import ModelUnavailableError


def _tensorflow():
    try:
        import tensorflow as tf  # type: ignore

        return tf
    except Exception as exc:  # pragma: no cover - depends on local runtime
        raise ModelUnavailableError(f"TensorFlow is unavailable: {exc}") from exc


def build_gru_model(
    history_steps: int,
    horizon_steps: int,
    feature_count: int,
    units: int = 64,
    dropout_rate: float = 0.2,
    learning_rate: float = 0.001,
    history_feature_count: int = 1,
):
    tf = _tensorflow()
    hist_input = tf.keras.Input(shape=(history_steps, history_feature_count), name="history_power")
    hist = tf.keras.layers.GRU(units, return_sequences=False)(hist_input)
    hist = tf.keras.layers.Dropout(dropout_rate)(hist)
    hist = tf.keras.layers.RepeatVector(horizon_steps)(hist)

    future_input = tf.keras.Input(
        shape=(horizon_steps, feature_count), name="future_features"
    )
    future = tf.keras.layers.GRU(units, return_sequences=True)(future_input)
    future = tf.keras.layers.Dropout(dropout_rate)(future)
    merged = tf.keras.layers.Concatenate(axis=-1)([hist, future])
    decoded = tf.keras.layers.GRU(units, return_sequences=True)(merged)
    decoded = tf.keras.layers.TimeDistributed(
        tf.keras.layers.Dense(max(16, units // 2), activation="relu")
    )(decoded)
    output = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(1))(decoded)
    model = tf.keras.Model([hist_input, future_input], output)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss="mse"
    )
    return model


@dataclass
class GRUConfig:
    units: int = 64
    dropout_rate: float = 0.2
    learning_rate: float = 0.0003
    epochs: int = 50
    batch_size: int = 32
    patience: int = 8
    seed: int = 42
    reduce_lr: bool = True
    lr_patience: int = 2
    lr_factor: float = 0.5
    min_lr: float = 0.00001
    min_delta: float = 0.00001


class GRUForecaster:
    def __init__(self, model=None, config: Optional[GRUConfig] = None):
        self.model = model
        self.config = config or GRUConfig()
        self.history = []
        self.training_summary = {}

    def fit(self, x_hist, x_future, y, validation_data=None, *,
            target_scaler=None, output_dir=None, progress=None):
        tf = _tensorflow()
        tf.keras.utils.set_random_seed(self.config.seed)
        if self.config.epochs < 1 or self.config.batch_size < 1 or self.config.patience < 1:
            raise ValueError("epochs, batch_size and patience must be positive")
        if (self.config.learning_rate <= 0 or self.config.min_lr <= 0
                or self.config.min_lr > self.config.learning_rate
                or self.config.lr_patience < 1 or not 0 < self.config.lr_factor < 1
                or self.config.min_delta < 0):
            raise ValueError("invalid learning-rate schedule configuration")
        # Avoid retaining graphs from prior rolling folds/candidates.
        if self.model is None:
            tf.keras.backend.clear_session()
        tf.config.experimental.enable_op_determinism()
        if self.model is None:
            self.model = build_gru_model(
                x_hist.shape[1],
                x_future.shape[1],
                x_future.shape[2],
                units=self.config.units,
                dropout_rate=self.config.dropout_rate,
                learning_rate=self.config.learning_rate,
                history_feature_count=x_hist.shape[2],
            )
        # Streaming sums over ALL batches: R² is not an average of batch R².
        scale = 1.0 if target_scaler is None else float(1 / target_scaler.scale_[0])
        offset = 0.0 if target_scaler is None else float(-target_scaler.min_[0] * scale)
        metrics = power_metrics(tf, scale, offset)
        self.model.compile(optimizer=self.model.optimizer, loss="mse", metrics=metrics)
        monitor = "val_loss" if validation_data is not None else "loss"
        early_stop = tf.keras.callbacks.EarlyStopping(
            monitor=monitor, patience=self.config.patience, restore_best_weights=True,
            min_delta=self.config.min_delta,
        )
        output = Path(output_dir) if output_dir is not None else None
        if output:
            output.mkdir(parents=True, exist_ok=True)
        owner = self
        owner.history = []
        visualization_errors = []
        visualization_refs = []

        def emit(message):
            if progress:
                progress(message)

        class TrainingTrace(tf.keras.callbacks.Callback):
            best_value = float("inf")
            best_weights = None

            def on_epoch_begin(self, epoch, logs=None):
                self.started = self.last_update = time.perf_counter()
                self.epoch = epoch + 1
                self.learning_rate = float(tf.keras.backend.get_value(self.model.optimizer.learning_rate))

            def on_train_batch_end(self, batch, logs=None):
                now = time.perf_counter()
                if now - self.last_update >= 30:
                    emit(f"Epoch {self.epoch}/{owner.config.epochs} | "
                         f"batch {batch + 1}/{self.params['steps']} | 训练中")
                    self.last_update = now

            def on_epoch_end(self, epoch, logs=None):
                row = {"epoch": epoch + 1, "seconds": time.perf_counter() - self.started}
                row.update({key: float(value) if np.isfinite(value) else None
                            for key, value in (logs or {}).items()})
                row["learning_rate"] = self.learning_rate
                row["next_learning_rate"] = float(tf.keras.backend.get_value(self.model.optimizer.learning_rate))
                owner.history.append(row)
                if row.get(monitor) is None or row.get("loss") is None:
                    raise FloatingPointError("GRU loss became non-finite; training failed")
                if row[monitor] < self.best_value:
                    self.best_value = row[monitor]
                    self.best_weights = self.model.get_weights()
                def fmt(key, digits=3):
                    value = row.get(key)
                    return f"{value:.{digits}f}" if value is not None else "N/A"
                emit(f"Epoch {epoch + 1:02d}/{owner.config.epochs} | "
                     f"loss={fmt('loss', 6)} val_loss={fmt('val_loss', 6)} | "
                     f"训练 RMSE={fmt('rmse_kw')} kW R²={fmt('r2', 4)} | "
                     f"验证 RMSE={fmt('val_rmse_kw')} kW MAE={fmt('val_mae_kw')} kW "
                     f"R²={fmt('val_r2', 4)} | "
                     f"LR={self.learning_rate:.2g} → {row['next_learning_rate']:.2g} | "
                     f"早停 {early_stop.wait}/{owner.config.patience} | {row['seconds']:.1f}s")
                if output:
                    csv_path = output / "training_history.csv"
                    with csv_path.open("a" if epoch else "w", encoding="utf-8", newline="") as stream:
                        writer = csv.DictWriter(stream, fieldnames=list(row))
                        if not epoch:
                            writer.writeheader()
                        writer.writerow(row)
                    (output / "training_history.json").write_text(
                        json.dumps(owner.history, ensure_ascii=False, indent=2, allow_nan=False),
                        encoding="utf-8",
                    )
                    # Refresh the same files after every epoch for inspection during training.
                    from .visuals import generate_training_history_visualizations
                    refs, errors = generate_training_history_visualizations(owner.history, output / "visualizations")
                    visualization_refs[:] = refs
                    for error in errors:
                        if error not in visualization_errors:
                            visualization_errors.append(error)
                            emit(f"训练图片生成失败 [{error['id']}]：{error['message']}")

        callbacks = [tf.keras.callbacks.TerminateOnNaN()]
        if self.config.reduce_lr:
            callbacks.append(tf.keras.callbacks.ReduceLROnPlateau(
                monitor=monitor, factor=self.config.lr_factor, patience=self.config.lr_patience,
                min_lr=self.config.min_lr, min_delta=self.config.min_delta, verbose=0,
            ))
        callbacks.append(early_stop)
        if output:
            callbacks.append(tf.keras.callbacks.ModelCheckpoint(
                str(output / "gru.keras"), monitor=monitor, save_best_only=True, verbose=0,
            ))
        trace = TrainingTrace()
        callbacks.append(trace)
        y3 = np.asarray(y, dtype=np.float32)[..., None]
        kwargs = {
            "epochs": self.config.epochs,
            "batch_size": self.config.batch_size,
            "verbose": 0,
            "callbacks": callbacks,
            "shuffle": False,
        }
        if validation_data is not None:
            vx, vf, vy = validation_data
            kwargs["validation_data"] = ([vx, vf], np.asarray(vy)[..., None])
        self.model.fit([x_hist, x_future], y3, **kwargs)
        # EarlyStopping's min_delta governs patience. The saved candidate uses
        # the absolute lowest loss, including improvements smaller than min_delta.
        self.model.set_weights(trace.best_weights)
        best = min(owner.history, key=lambda row: row[monitor])
        self.training_summary = {
            "status": "completed", "epochs_completed": len(owner.history),
            "epochs_requested": self.config.epochs, "best_epoch": best["epoch"],
            "best_monitor_value": best[monitor], "monitor": monitor,
            "stopped_early": bool(early_stop.stopped_epoch),
            "stop_reason": "early_stopping" if early_stop.stopped_epoch else "epoch_limit",
            "shuffle": False, "training_windows_used": len(y),
            "history_feature_count": int(x_hist.shape[2]),
            "learning_rate_schedule": {
                "enabled": self.config.reduce_lr, "initial": self.config.learning_rate,
                "factor": self.config.lr_factor, "patience": self.config.lr_patience,
                "minimum": self.config.min_lr, "min_delta": self.config.min_delta,
                "reductions": sum(r["next_learning_rate"] < r["learning_rate"] for r in owner.history),
            },
            "metric_units": {"loss": "normalized MSE", "rmse_kw": "kW", "mae_kw": "kW", "r2": "dimensionless"},
            "visualizations": visualization_refs, "visualization_errors": visualization_errors,
        }
        if output:
            (output / "training_summary.json").write_text(
                json.dumps(self.training_summary, ensure_ascii=False, indent=2), encoding="utf-8",
            )
        emit(f"GRU 训练完成：{len(owner.history)} 轮，恢复第 {best['epoch']} 轮最佳权重；"
             f"{'触发早停' if early_stop.stopped_epoch else '达到轮数上限'}。")
        return self

    def predict(self, x_hist, x_future) -> np.ndarray:
        if self.model is None:
            raise ModelUnavailableError("GRU model has not been fitted")
        output = self.model.predict([x_hist, x_future], batch_size=self.config.batch_size, verbose=0)
        # Clip in physical units AFTER inverse scaling in the pipeline.
        return np.asarray(output)[..., 0]

    def save(self, path: str) -> None:
        if self.model is None:
            raise ModelUnavailableError("cannot save an unfitted GRU model")
        self.model.save(path)

    @classmethod
    def load(cls, path: str, config: Optional[GRUConfig] = None) -> "GRUForecaster":
        tf = _tensorflow()
        return cls(_load_keras_model_compat(tf, path), config=config)


def _load_keras_model_compat(tf, path: str):
    """Load a Keras model, handling one narrow cross-version initializer change.

    Some Keras versions write ``input_axes``/``output_axes`` as null fields in
    GlorotUniform configs.  Keras 3.14 rejects those *null* defaults although
    they carry no model information.  We first use the normal loader, then
    retry once with only those null fields removed from a temporary copy.  The
    original archive and weights are never modified, and any non-null or
    unrelated deserialization error is allowed to fail normally.
    """
    try:
        return tf.keras.models.load_model(path, compile=False)
    except Exception as first_error:
        message = str(first_error)
        if ("GlorotUniform" not in message or "unexpected keyword argument" not in message
                or not any(f"'{key}'" in message for key in ("input_axes", "output_axes"))
                or not zipfile.is_zipfile(path)):
            raise
        try:
            with zipfile.ZipFile(path, "r") as source:
                config_name = "config.json"
                if config_name not in source.namelist():
                    raise first_error
                config = json.loads(source.read(config_name).decode("utf-8"))
                removed = 0

                def strip_null_legacy_axes(value):
                    nonlocal removed
                    if isinstance(value, dict):
                        cleaned = {key: strip_null_legacy_axes(item) for key, item in value.items()}
                        if (value.get("module") == "keras.initializers"
                                and value.get("class_name") == "GlorotUniform"
                                and value.get("registered_name") is None
                                and isinstance(cleaned.get("config"), dict)):
                            for key in ("input_axes", "output_axes"):
                                if key in cleaned["config"] and cleaned["config"][key] is None:
                                    del cleaned["config"][key]
                                    removed += 1
                        return cleaned
                    if isinstance(value, list):
                        return [strip_null_legacy_axes(item) for item in value]
                    return value

                sanitized = strip_null_legacy_axes(config)
                if removed == 0:
                    raise first_error
                with tempfile.NamedTemporaryFile(suffix=".keras", delete=False) as temp:
                    temporary_path = temp.name
                try:
                    with zipfile.ZipFile(temporary_path, "w") as target:
                        for name in source.namelist():
                            payload = source.read(name)
                            if name == config_name:
                                payload = json.dumps(
                                    sanitized, ensure_ascii=False, separators=(",", ":")
                                ).encode("utf-8")
                            target.writestr(name, payload)
                    return tf.keras.models.load_model(temporary_path, compile=False)
                finally:
                    Path(temporary_path).unlink(missing_ok=True)
        except Exception as compatibility_error:
            raise first_error from compatibility_error


def power_metrics(tf, scale: float, offset: float):
    """Physical-kW metrics using global sufficient statistics, including R²."""
    class PowerMetric(tf.keras.metrics.Metric):
        def __init__(self, kind):
            super().__init__(name=kind, dtype="float64")
            self.kind = kind
            self.count = self.add_weight(name="count", initializer="zeros", dtype="float64")
            self.error = self.add_weight(name="error", initializer="zeros", dtype="float64")
            self.total = self.add_weight(name="total", initializer="zeros", dtype="float64")
            self.total_squared = self.add_weight(name="total_squared", initializer="zeros", dtype="float64")

        def update_state(self, y_true, y_pred, sample_weight=None):
            true = tf.cast(y_true, tf.float64) * scale + offset
            pred = tf.maximum(tf.cast(y_pred, tf.float64) * scale + offset, 0.0)
            delta = pred - true
            self.count.assign_add(tf.cast(tf.size(true), tf.float64))
            self.error.assign_add(tf.reduce_sum(tf.abs(delta) if self.kind == "mae_kw" else tf.square(delta)))
            self.total.assign_add(tf.reduce_sum(true))
            self.total_squared.assign_add(tf.reduce_sum(tf.square(true)))

        def result(self):
            if self.kind == "r2":
                variance = self.total_squared - tf.math.divide_no_nan(tf.square(self.total), self.count)
                return tf.where(variance > 1e-12, 1.0 - tf.math.divide_no_nan(self.error, variance),
                                tf.constant(float("nan"), tf.float64))
            value = tf.math.divide_no_nan(self.error, self.count)
            return tf.sqrt(value) if self.kind == "rmse_kw" else value

        def get_config(self):
            return {**super().get_config(), "kind": self.kind}

    return [PowerMetric(kind) for kind in ("rmse_kw", "mae_kw", "r2")]
