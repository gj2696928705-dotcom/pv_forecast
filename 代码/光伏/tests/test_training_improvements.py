from __future__ import annotations
import importlib.util
import json
import tempfile
import unittest
import zipfile
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd

from pv_forecast.data import InputSpec, build_windows, load_and_regularize, DataValidationError
from pv_forecast.experiments import (RollingFold, candidate_configs, default_folds, fold_bounds,
                                     MetricAccumulator, train_experiment)
from pv_forecast.pipeline import TrainingConfig, train_model, predict_record, _fit_history_scaler, _fit_scalers
from pv_forecast.weather import WeatherForecastArchive, solar_elevation
from pv_forecast.metrics import regression_metrics


def data(days=20):
    index = pd.date_range("2025-01-01", periods=days * 96, freq="15min", tz="Asia/Shanghai")
    daylight = np.maximum(0, np.sin(np.pi * (index.hour + index.minute / 60 - 6) / 12))
    return pd.DataFrame({"timestamp": index, "pv_power_kw": 100 * daylight,
        "SWGDN": 500 * daylight, "temperature_c": 10 + 10 * daylight, "cloud_cover": np.full(len(index), .2)})


HISTORY = ("SWGDN", "temperature_c", "cloud_cover")
SPEC = InputSpec(history_weather_columns=HISTORY, latitude=30.9, longitude=111.35,
                 coordinate_basis="test simulation")


class FeatureTests(unittest.TestCase):
    def test_missing_history_weather_is_reported_in_quality(self):
        source = data(5)
        for missing in (("cloud_cover",), HISTORY):
            regular = load_and_regularize(source.drop(columns=list(missing)), SPEC)
            self.assertEqual(regular.quality.missing_weather_rows, len(source))
            self.assertEqual(regular.quality.as_dict()["missing_history_weather_columns"], list(missing))
        complete = load_and_regularize(source, SPEC)
        self.assertEqual(complete.quality.missing_weather_rows, 0)
        self.assertEqual(complete.quality.as_dict()["missing_history_weather_columns"], [])

    def test_legacy_keras_null_initializer_axes_are_sanitized_only_in_temp_copy(self):
        from pv_forecast.models import _load_keras_model_compat

        with tempfile.TemporaryDirectory() as directory:
            archive = Path(directory) / "legacy.keras"
            config = {"kernel_initializer": {"module": "keras.initializers", "class_name": "GlorotUniform", "config": {
                "seed": None, "input_axes": None, "output_axes": None,
            }}, "unrelated": {"input_axes": [1, 2], "output_axes": None}}
            with zipfile.ZipFile(archive, "w") as stream:
                stream.writestr("config.json", json.dumps(config))
                stream.writestr("model.weights.h5", b"weights")

            class Loader:
                def __init__(self):
                    self.calls = []

                def load_model(self, path, compile=False):
                    self.calls.append(Path(path))
                    if len(self.calls) == 1:
                        raise TypeError("GlorotUniform.__init__() got an unexpected keyword argument 'input_axes'")
                    with zipfile.ZipFile(path) as stream:
                        sanitized = json.loads(stream.read("config.json"))
                        self.weights = stream.read("model.weights.h5")
                    self.sanitized = sanitized
                    return "loaded"

            loader = Loader()
            fake_tf = type("TF", (), {"keras": type("Keras", (), {
                "models": loader,
            })})
            self.assertEqual(_load_keras_model_compat(fake_tf, str(archive)), "loaded")
            self.assertEqual(len(loader.calls), 2)
            self.assertEqual(loader.sanitized["kernel_initializer"]["config"], {"seed": None})
            self.assertEqual(loader.sanitized["unrelated"]["input_axes"], [1, 2])
            self.assertIsNone(loader.sanitized["unrelated"]["output_axes"])
            self.assertEqual(loader.weights, b"weights")
            self.assertFalse(loader.calls[1].exists())
            with zipfile.ZipFile(archive) as stream:
                self.assertEqual(json.loads(stream.read("config.json")), config)

    def test_legacy_loader_preserves_nonnull_axes_and_unrelated_errors(self):
        from types import SimpleNamespace
        from unittest.mock import Mock
        from pv_forecast.models import _load_keras_model_compat

        with tempfile.TemporaryDirectory() as directory:
            archive = Path(directory) / "model.keras"
            config = {"module": "keras.initializers", "class_name": "GlorotUniform",
                      "config": {"input_axes": [0], "output_axes": [1]}}
            with zipfile.ZipFile(archive, "w") as stream:
                stream.writestr("config.json", json.dumps(config))
            for message in ("GlorotUniform got an unexpected keyword argument 'input_axes'",
                            "weights are corrupt"):
                loader = Mock(side_effect=TypeError(message))
                fake_tf = SimpleNamespace(keras=SimpleNamespace(models=SimpleNamespace(load_model=loader)))
                with self.assertRaisesRegex(TypeError, message):
                    _load_keras_model_compat(fake_tf, str(archive))
                loader.assert_called_once()

    def test_historical_inputs_exclude_future_weather_and_target(self):
        source = data()
        regular = load_and_regularize(source, SPEC)
        args = dict(history_weather_columns=HISTORY, latitude=SPEC.latitude, longitude=SPEC.longitude)
        first = build_windows(regular, 288, 384, **args)
        self.assertEqual(first.gru_history.shape, (1, 288, 4))
        self.assertEqual(first.future_features.shape, (1, 96, 6))
        self.assertIn("solar_elevation_deg", first.feature_names)
        changed = source.copy()
        changed.loc[288:, ["pv_power_kw", *HISTORY]] = 1000000.
        second = build_windows(load_and_regularize(changed, SPEC), 288, 384, **args)
        np.testing.assert_array_equal(first.gru_history, second.gru_history)
        np.testing.assert_array_equal(first.future_features, second.future_features)

    def test_future_weather_requires_issue_time_and_ignores_observed_columns(self):
        source = data()
        regular = load_and_regularize(source, SPEC)
        with self.assertRaisesRegex(DataValidationError, "issued forecast"):
            build_windows(regular, 288, 384, weather_columns=("SWGDN",))
        origin = regular.frame.index[288]
        future = regular.frame.index[288:384]
        issued = origin - pd.Timedelta(hours=1)
        archive_data = pd.DataFrame({"timestamp": future, "issued_at": issued, "SWGDN": 17.})
        newer = archive_data.assign(issued_at=origin + pd.Timedelta(minutes=1), SWGDN=999.)
        archive = WeatherForecastArchive(pd.concat([archive_data, newer]), ("SWGDN",))
        windows = build_windows(regular, 288, 384, weather_columns=("SWGDN",), weather_archive=archive)
        np.testing.assert_array_equal(windows.future_features[0, :, -1], 17.)
        self.assertEqual(windows.weather_issued_at[0], issued)
        with self.assertRaises(DataValidationError):
            WeatherForecastArchive(archive_data.drop(columns="issued_at"), ("SWGDN",))
        missing = WeatherForecastArchive(archive_data.iloc[:-1], ("SWGDN",))
        self.assertIsNone(missing.at(origin, future)[0])
        self.assertIsNone(archive.at(issued - pd.Timedelta(seconds=1), future)[0])

    def test_solar_geometry_is_timezone_invariant(self):
        index = pd.DatetimeIndex(["2025-03-20T12:00Z", "2025-03-20T00:00Z"])
        elevation = solar_elevation(index, 0., 0.)
        self.assertGreater(elevation[0], 85)
        self.assertLess(elevation[1], -85)
        np.testing.assert_allclose(elevation, solar_elevation(index.tz_convert("Asia/Shanghai"), 0., 0.))
        with self.assertRaises(ValueError):
            InputSpec(latitude=30.9)

    def test_long_gaps_remain_missing_and_scalers_use_only_train(self):
        source = data()
        source.loc[100:103, "cloud_cover"] = np.nan
        regular = load_and_regularize(source, SPEC)
        self.assertTrue(regular.frame.cloud_cover.iloc[100:104].isna().all())
        train = build_windows(regular, 0, 10 * 96, history_weather_columns=HISTORY)
        target, future = _fit_scalers(train)
        past = _fit_history_scaler(train)
        source.loc[10 * 96:, ["pv_power_kw", *HISTORY]] = 1e8
        other = build_windows(load_and_regularize(source, SPEC), 0, 10 * 96, history_weather_columns=HISTORY)
        t2, f2 = _fit_scalers(other)
        h2 = _fit_history_scaler(other)
        for a, b in ((target, t2), (future, f2), (past, h2)):
            np.testing.assert_array_equal(a.data_max_, b.data_max_)

    def test_fold_bounds_and_pooled_metrics(self):
        index = pd.date_range("2025-01-01", periods=365 * 96, freq="15min", tz="Asia/Shanghai")
        bounds = fold_bounds(index, default_folds(index), int(len(index) * .85))
        self.assertEqual(len(bounds), 3)
        self.assertEqual([index[b[1]].month for b in bounds], [5, 7, 9])
        for fold, lo, hi in bounds:
            self.assertLessEqual(hi, int(len(index) * .85))
        with self.assertRaisesRegex(ValueError, "test period"):
            fold_bounds(index, [RollingFold("bad", "2025-10-01", "2025-12-01")], int(len(index) * .85))
        truth, pred = np.arange(10.), np.arange(10.) * .8
        stats = MetricAccumulator()
        stats.add(truth[:2], pred[:2])
        stats.add(truth[2:], pred[2:])
        for key in ("rmse", "mae", "r2", "nrmse"):
            self.assertAlmostEqual(stats.result()[key], regression_metrics(truth, pred)[key])
        configs = candidate_configs(TrainingConfig())
        self.assertEqual([c.gru_units for c in configs.values()], [32, 32, 16])
        self.assertEqual([c.reduce_lr for c in configs.values()], [False, True, True])


@unittest.skipUnless(importlib.util.find_spec("tensorflow"), "TensorFlow not installed")
class TrainingTests(unittest.TestCase):
    def test_weather_archive_training_prediction_and_future_issuance_rejection(self):
        source = data(16)
        index = pd.date_range(source.timestamp.iloc[0], periods=17 * 96, freq="15min")
        issue = index[0] - pd.Timedelta(hours=1)
        archive = pd.DataFrame({"timestamp": index, "issued_at": issue, "SWGDN": 300.})
        spec = replace(SPEC, weather_columns=("SWGDN",))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train_model(source, root, spec, TrainingConfig(min_gru_days=0,
                gru_epochs=1, max_gru_samples=4, gru_units=4, gru_batch_size=64),
                future_weather_source=archive, selected_model_override="gru")
            good = predict_record(root, source, future_weather_source=archive)
            self.assertEqual(good["status"], "available")
            self.assertEqual(good["confidence"]["weather_forecast_issued_at"], issue.isoformat())
            missing = predict_record(root, source)
            self.assertEqual(missing["status"], "degraded")
            late_archive = archive.assign(issued_at=index[-1] + pd.Timedelta(days=1))
            late = predict_record(root, source, future_weather_source=late_archive)
            self.assertEqual(late["status"], "degraded")
            self.assertIsNone(late["confidence"]["weather_forecast_issued_at"])
            with self.assertRaises(DataValidationError):
                predict_record(root, source, future_weather_source=source)
            origin = source.timestamp.iloc[14 * 96]
            earlier = predict_record(root, source, forecast_start=origin, future_weather_source=archive)
            truncated = predict_record(root, source.iloc[:14 * 96], future_weather_source=archive)
            np.testing.assert_allclose([p['value'] for p in earlier['forecast_values']],
                                       [p['value'] for p in truncated['forecast_values']])

    def test_schedule_reduces_rate_before_early_stopping(self):
        from pv_forecast.models import GRUConfig, GRUForecaster
        hist, future, target = np.zeros((4, 6, 4)), np.zeros((4, 4, 5)), np.zeros((4, 4))
        model = GRUForecaster(config=GRUConfig(units=2, learning_rate=.001,
            epochs=8, patience=5, lr_patience=2, min_lr=.0001, batch_size=2))
        messages = []
        model.fit(hist, future, target, validation_data=(hist, future, target), progress=messages.append)
        self.assertEqual(model.training_summary["epochs_completed"], 6)
        self.assertEqual(model.training_summary["best_epoch"], 1)
        self.assertEqual(model.training_summary["learning_rate_schedule"]["reductions"], 2)
        self.assertAlmostEqual(model.history[-1]["learning_rate"], .00025)
        self.assertTrue(any("LR=" in s and "RMSE=" in s and "R²=" in s for s in messages))

    def test_multivariate_reload_and_validated_fallback(self):
        source = data(16)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result = train_model(source, root, SPEC, TrainingConfig(min_gru_days=0,
                gru_epochs=2, max_gru_samples=8, gru_units=4, gru_batch_size=32), selected_model_override="gru")
            from pv_forecast.models import GRUForecaster
            model = GRUForecaster.load(str(root / "gru.keras"))
            self.assertEqual(model.model.inputs[0].shape[-1], 4)
            self.assertEqual(model.model.inputs[1].shape[-1], 6)
            good = predict_record(root, source)
            self.assertEqual(good["status"], "available")
            bad = predict_record(root, source.drop(columns=list(HISTORY)))
            self.assertEqual(bad["status"], "degraded")
            self.assertTrue(bad["fallback_forecast_ref"].startswith("baseline:"))
            self.assertEqual(bad["confidence"]["model_used"], bad["fallback_forecast_ref"])
            fallback = result["metadata"]["baseline_fallback"]
            values = np.array([v["value"] for v in bad["forecast_values"]])
            expected_upper = values + np.array(fallback["calibration"]["quantiles_kw"])
            np.testing.assert_allclose([v["upper"] for v in bad["prediction_interval"]["values"]], expected_upper)
            (root / "gru.keras").unlink()
            self.assertEqual(predict_record(root, source)["status"], "degraded")
            stale = predict_record(root, source, forecast_start="2026-01-01")
            self.assertEqual(stale["status"], "unavailable")

    def test_rolling_end_to_end_and_no_test_access_during_selection(self):
        source = data(24)
        folds = [RollingFold("first", "2025-01-08", "2025-01-10"),
                 RollingFold("second", "2025-01-12", "2025-01-14"),
                 RollingFold("third", "2025-01-16", "2025-01-18")]
        test_start = source.timestamp.iloc[int(len(source) * .85)]
        from pv_forecast import experiments
        real_build = experiments.build_windows
        seen = []
        def checked(*args, **kwargs):
            windows = real_build(*args, **kwargs)
            for origin in windows.forecast_starts:
                self.assertLess(origin + pd.Timedelta(minutes=95 * 15), test_start)
            seen.append(windows)
            return windows
        with tempfile.TemporaryDirectory() as directory, patch.object(experiments, "build_windows", side_effect=checked):
            result = train_experiment(source, directory, SPEC, TrainingConfig(min_gru_days=0,
                gru_epochs=1, max_gru_samples=4, gru_batch_size=64, history_days=1), folds=folds)
            report = result["rolling_validation"]
            self.assertEqual(report["status"], "completed")
            self.assertEqual(len(report["folds"]), 3)
            self.assertEqual(len(report["aggregate"]), 6)
            self.assertFalse(report["test_used_for_selection"])
            self.assertEqual(report["scope"], "explicitly_capped_training_windows")
            for fold in report["folds"]:
                self.assertEqual(len(fold["results"]), 6)
                self.assertEqual(fold["training_windows_used"], 4)
            root = Path(directory)
            self.assertEqual(len(list(root.glob("experiments/*/*/gru.keras"))), 9)
            self.assertEqual(len(list(root.glob("experiments/*/*/scalers.joblib"))), 9)
            self.assertEqual(result["metadata"]["selection_source"], "rolling_validation")
            self.assertEqual(len(predict_record(root, source)["forecast_values"]), 96)
            self.assertFalse(list(root.rglob("training_metrics.png")))


if __name__ == "__main__":
    unittest.main()
