from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from pv_forecast.baselines import BaselineForecaster
from pv_forecast.data import (
    DataValidationError,
    InputSpec,
    build_windows,
    load_and_regularize,
    split_boundaries,
)
from pv_forecast.metrics import (
    apply_prediction_interval,
    calibrate_prediction_interval,
    regression_metrics,
)
from pv_forecast.records import build_forecast_record, validate_forecast_record
from pv_forecast.pipeline import TrainingConfig, evaluate_bundle, predict_record, train_model


def synthetic_frame(days: int = 16, frequency: str = "15min") -> pd.DataFrame:
    index = pd.date_range("2025-01-01", periods=days * 96, freq=frequency)
    minute = index.hour * 60 + index.minute
    daylight = np.maximum(0.0, np.sin(np.pi * (minute - 360) / 720.0))
    seasonal = 1.0 + 0.1 * np.sin(2 * np.pi * index.dayofyear / 365.25)
    power = 250.0 * daylight * seasonal
    return pd.DataFrame({"timestamp": index, "pv_power_kw": power})


class DataTests(unittest.TestCase):
    def test_load_regularizes_and_preserves_short_gap_only(self):
        frame = synthetic_frame(2).iloc[::4].reset_index(drop=True)
        loaded = load_and_regularize(frame, interval_minutes=15)
        self.assertEqual(loaded.interval_minutes, 15)
        self.assertEqual(loaded.quality.interpolated_target_rows, 0)
        self.assertGreater(loaded.quality.missing_target_rows, 0)
        # The source is sampled every hour and ends at 23:00 on day two, so
        # regularization covers the observed interval rather than inventing
        # the final unobserved 23:15/23:30/23:45 points.
        self.assertEqual(len(loaded.frame), 189)
        dense = synthetic_frame(2)
        missing = dense.drop(index=[40])
        short = load_and_regularize(missing)
        self.assertEqual(short.quality.interpolated_target_rows, 1)
        # Filling uses only an earlier observation, never a future label.
        self.assertEqual(short.frame.pv_power_kw.iloc[40], dense.pv_power_kw.iloc[39])

    def test_rejects_duplicate_and_negative_values(self):
        frame = synthetic_frame(1)
        duplicate = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
        with self.assertRaises(DataValidationError):
            load_and_regularize(duplicate)
        negative = frame.copy()
        negative.loc[0, "pv_power_kw"] = -1
        with self.assertRaises(DataValidationError):
            load_and_regularize(negative)

    def test_windows_do_not_cross_target_split(self):
        loaded = load_and_regularize(synthetic_frame(16))
        train_end, val_end = split_boundaries(len(loaded.frame))
        for lo, hi in ((0, train_end), (train_end, val_end), (val_end, len(loaded.frame))):
            windows = build_windows(loaded, lo, hi)
            self.assertGreater(windows.size, 0)
            self.assertEqual(windows.size, hi - 96 + 1 - max(lo, 288))
            for start in windows.forecast_starts:
                self.assertGreaterEqual(start, loaded.frame.index[lo])
                self.assertLessEqual(start + pd.Timedelta(minutes=95 * 15), loaded.frame.index[hi - 1])


class BaselineTests(unittest.TestCase):
    def test_baseline_is_nonnegative_and_has_96_points(self):
        history = np.tile(np.arange(96, dtype=float), (2, 3))
        pred = BaselineForecaster("same_slot_mean").predict(history, 96)
        self.assertEqual(pred.shape, (2, 96))
        self.assertTrue(np.all(pred >= 0))


class MetricsTests(unittest.TestCase):
    def test_rmse_and_interval(self):
        truth = np.ones((40, 4))
        pred = np.zeros((40, 4))
        calibration = calibrate_prediction_interval(truth, pred, nominal_coverage=0.8)
        lower, upper = apply_prediction_interval(pred, calibration)
        self.assertTrue(np.all(lower <= upper))
        metrics = regression_metrics(truth, pred, lower, upper)
        self.assertAlmostEqual(metrics["rmse"], 1.0)
        self.assertEqual(metrics["coverage"], 1.0)


class RecordTests(unittest.TestCase):
    def test_record_matches_schema(self):
        calibration = {
            "method": "empirical_absolute_residual",
            "nominal_coverage": 0.8,
            "quantiles_kw": [10.0, 10.0],
            "step_specific": True,
        }
        record = build_forecast_record(
            "forecast-1", [50, 60], "2025-01-01T00:00:00+08:00", 15,
            "pv-1", "pv-power-forecast", "0.1.0-candidate", None,
            {"start": "2025-01-01", "end": "2025-01-16"}, calibration,
            {"quality": "A"}, lower=[40, 50], upper=[60, 70], audit_refs=["test"],
        )
        validate_forecast_record(record)
        self.assertEqual(record["forecast_type"], "pv_power")
        self.assertEqual(len(record["forecast_values"]), 2)

    def test_evaluate_uses_last_labelled_horizon(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "data.csv"
            synthetic_frame(16).to_csv(source, index=False)
            bundle = root / "bundle"
            train_model(
                source,
                bundle,
                InputSpec(),
                TrainingConfig(min_gru_days=60),
            )
            result = evaluate_bundle(
                bundle,
                source,
                InputSpec(),
                visualizations_dir=root / "evaluation" / "visualizations",
            )
            self.assertEqual(result["status"], "available")
            self.assertIsNotNone(result["metrics"])
            self.assertEqual(result["metrics"]["count"], 96)
            self.assertEqual(len(result["visualizations"]), 1)
            image = root / "evaluation" / result["visualizations"][0]["path"]
            self.assertTrue(image.exists())

    @unittest.skipUnless(__import__("importlib").util.find_spec("matplotlib"), "Matplotlib not installed")
    def test_predict_writes_three_visualizations_and_json_refs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "data.csv"
            synthetic_frame(16).to_csv(source, index=False)
            bundle = root / "bundle"
            train_model(
                source,
                bundle,
                InputSpec(),
                TrainingConfig(min_gru_days=60, skip_gru=True),
            )
            output = root / "forecast.json"
            record = predict_record(bundle, source, output_path=output, input_spec=InputSpec())
            self.assertEqual(record["status"], "available")
            self.assertEqual(len(record["visualizations"]), 3)
            self.assertEqual(record["visualization_errors"], [])
            self.assertEqual(len(json.loads(output.read_text(encoding="utf-8"))["visualizations"]), 3)
            for reference in record["visualizations"]:
                image_path = output.parent / Path(reference["path"])
                self.assertTrue(image_path.exists())
                self.assertGreater(image_path.stat().st_size, 0)
                from PIL import Image

                with Image.open(image_path) as image:
                    image.verify()

    @unittest.skipUnless(__import__("importlib").util.find_spec("matplotlib"), "Matplotlib not installed")
    def test_unavailable_history_has_no_visualizations(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "data.csv"
            synthetic_frame(16).to_csv(source, index=False)
            bundle = root / "bundle"
            train_model(
                source,
                bundle,
                InputSpec(),
                TrainingConfig(min_gru_days=60, skip_gru=True),
            )
            short_source = root / "short.csv"
            synthetic_frame(2).to_csv(short_source, index=False)
            record = predict_record(
                bundle,
                short_source,
                output_path=root / "unavailable.json",
                input_spec=InputSpec(),
            )
            self.assertEqual(record["status"], "unavailable")
            self.assertEqual(record["visualizations"], [])
            self.assertEqual(record["visualization_errors"], [])

    def test_visualization_failure_keeps_structured_prediction(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "data.csv"
            synthetic_frame(16).to_csv(source, index=False)
            bundle = root / "bundle"
            train_model(
                source,
                bundle,
                InputSpec(),
                TrainingConfig(min_gru_days=60, skip_gru=True),
            )
            with patch("pv_forecast.visuals._pyplot", side_effect=RuntimeError("matplotlib missing")):
                record = predict_record(
                    bundle,
                    source,
                    output_path=root / "forecast.json",
                    input_spec=InputSpec(),
                )
            self.assertEqual(record["status"], "available")
            self.assertEqual(len(record["forecast_values"]), 96)
            self.assertEqual(record["visualizations"], [])
            self.assertEqual(len(record["visualization_errors"]), 3)


class GRUTest(unittest.TestCase):
    @unittest.skipUnless(__import__("importlib").util.find_spec("tensorflow"), "TensorFlow not installed")
    def test_gru_builds_and_predicts_one_batch(self):
        from pv_forecast.models import GRUConfig, GRUForecaster

        rng = np.random.default_rng(42)
        x_hist = rng.random((2, 12, 1))
        x_future = rng.random((2, 4, 5))
        y = rng.random((2, 4))
        model = GRUForecaster(config=GRUConfig(epochs=1, patience=1, units=8, batch_size=2))
        model.fit(x_hist, x_future, y)
        prediction = model.predict(x_hist, x_future)
        self.assertEqual(prediction.shape, (2, 4))
        self.assertTrue(np.isfinite(prediction).all())

    @unittest.skipUnless(__import__("importlib").util.find_spec("tensorflow"), "TensorFlow not installed")
    def test_physical_metrics_accumulate_across_batches(self):
        from pv_forecast.models import _tensorflow, power_metrics
        tf = _tensorflow()
        true = np.array([0., .1, .2, .9, 1.])
        pred = np.array([-.5, .3, .5, .8, .7])
        expected = regression_metrics(true * 300 + 20, np.maximum(pred * 300 + 20, 0))
        for metric, key in zip(power_metrics(tf, 300, 20), ("rmse", "mae", "r2")):
            metric.update_state(true[:2], pred[:2])
            metric.update_state(true[2:], pred[2:])
            self.assertAlmostEqual(float(metric.result()), expected[key], places=6)
            metric.reset_state()
            self.assertEqual(float(metric.count), 0.)

    @unittest.skipUnless(__import__("importlib").util.find_spec("tensorflow"), "TensorFlow not installed")
    def test_training_artifacts_and_reload_for_both_selection_paths(self):
        from pv_forecast.models import GRUForecaster
        from pv_forecast import pipeline
        from PIL import Image
        source = synthetic_frame(16)
        choose = pipeline._candidate_baselines
        for force_gru in (False, True):
            with self.subTest(force_gru=force_gru), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                messages = []
                def candidates(*args):
                    best, others = choose(*args)
                    return ((float("inf") if force_gru else -1., best[1], best[2]), others)
                with patch.object(pipeline, "_candidate_baselines", side_effect=candidates):
                    result = train_model(source, root, config=TrainingConfig(
                        min_gru_days=0, require_gru=True, gru_epochs=2,
                        gru_units=4, max_gru_samples=8, history_days=1, gru_batch_size=8,
                    ), progress=messages.append)
                gru = result["metrics"]["gru"]
                self.assertEqual(gru["status"], "completed")
                self.assertEqual(result["metadata"]["selected_model"] == "gru", force_gru)
                self.assertEqual(gru["training"]["epochs_completed"], 2)
                for name in ("gru.keras", "scalers.joblib", "training_history.csv", "training_summary.json"):
                    self.assertGreater((root / name).stat().st_size, 0)
                history = pd.read_csv(root / "training_history.csv")
                for key in ("loss", "val_loss", "rmse_kw", "val_rmse_kw", "r2", "val_r2", "val_mae_kw"):
                    self.assertTrue(np.isfinite(history[key]).all())
                best_row = history.iloc[gru["training"]["best_epoch"] - 1]
                self.assertAlmostEqual(best_row["val_rmse_kw"], gru["validation"]["rmse"], places=3)
                self.assertAlmostEqual(best_row["val_r2"], gru["validation"]["r2"], places=5)
                model = GRUForecaster.load(str(root / "gru.keras"))
                self.assertIsNotNone(model.model)
                forecast = predict_record(root, source)
                self.assertEqual(len(forecast["forecast_values"]), 96)
                self.assertTrue(all(v["value"] >= 0 for v in forecast["forecast_values"]))
                self.assertTrue(any("RMSE=" in msg and "R²=" in msg for msg in messages))
                self.assertLess(result["metadata"]["training_data_window"]["end"],
                                result["metadata"]["data_splits"]["validation"]["start"])
                for record in (gru["training"], result["metrics"]):
                    self.assertEqual(record["visualization_errors"], [])
                    self.assertEqual(len(record["visualizations"]), 1)
                    for ref in record["visualizations"]:
                        with Image.open(root / ref["path"]) as image:
                            image.verify()

    def test_required_gru_never_silently_falls_back(self):
        from pv_forecast.errors import ModelUnavailableError, InsufficientHistoryError
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(InsufficientHistoryError):
                train_model(synthetic_frame(16), directory, config=TrainingConfig(require_gru=True))
            with patch("pv_forecast.pipeline.GRUForecaster.fit", side_effect=ModelUnavailableError("missing tensorflow")):
                with self.assertRaisesRegex(ModelUnavailableError, "missing tensorflow"):
                    train_model(synthetic_frame(16), directory, config=TrainingConfig(require_gru=True, min_gru_days=0))

    @unittest.skipUnless(__import__("importlib").util.find_spec("tensorflow"), "TensorFlow not installed")
    def test_early_stopping_and_plot_failure_preserve_weights_and_history(self):
        from pv_forecast.models import GRUConfig, GRUForecaster, build_gru_model
        rng = np.random.default_rng(42)
        hist, future, truth = rng.random((4, 6, 1)), rng.random((4, 4, 5)), rng.random((4, 4))
        # Zero learning rate and no dropout deliberately produce a plateau.
        network = build_gru_model(6, 4, 5, units=2, dropout_rate=0., learning_rate=0.)
        model = GRUForecaster(network, GRUConfig(epochs=5, patience=1, batch_size=4))
        messages = []
        with tempfile.TemporaryDirectory() as directory:
            with patch("pv_forecast.visuals._pyplot", side_effect=RuntimeError("matplotlib unavailable")):
                model.fit(hist, future, truth, validation_data=(hist, future, truth),
                          output_dir=directory, progress=messages.append)
            summary = model.training_summary
            self.assertEqual(summary["epochs_completed"], 2)
            self.assertEqual(summary["best_epoch"], 1)
            self.assertEqual(summary["stop_reason"], "early_stopping")
            self.assertEqual(len(summary["visualization_errors"]), 1)
            self.assertEqual(summary["visualizations"], [])
            self.assertEqual(len(pd.read_csv(Path(directory) / "training_history.csv")), 2)
            loaded = GRUForecaster.load(str(Path(directory) / "gru.keras"))
            np.testing.assert_allclose(model.predict(hist, future), loaded.predict(hist, future), atol=1e-6)


if __name__ == "__main__":
    unittest.main()
