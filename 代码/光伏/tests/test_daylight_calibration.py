from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from pv_forecast.cli import build_parser
from pv_forecast.data import DataValidationError
from pv_forecast.metrics import apply_prediction_interval, calibrate_prediction_interval
from pv_forecast.pipeline import TrainingConfig, predict_record, train_model
from pv_forecast.recalibration import recalibrate_bundle
from pv_forecast.records import build_forecast_record, validate_forecast_record


def source_data():
    index = pd.date_range("2025-01-01", periods=16 * 96, freq="15min")
    phase = index.hour + index.minute / 60
    power = 250 * np.maximum(0, np.sin(np.pi * (phase - 6) / 12)) * (1 + index.dayofyear / 100)
    return pd.DataFrame({"timestamp": index, "pv_power_kw": power})


class ConditionalIntervalTests(unittest.TestCase):
    def test_night_residuals_do_not_shrink_daylight_interval(self):
        true, pred = np.zeros((100, 2)), np.zeros((100, 2))
        mask = np.zeros_like(true, dtype=bool)
        mask[81:] = True
        true[mask] = 100
        old = calibrate_prediction_interval(true, pred)
        self.assertEqual(old["quantiles_kw"], [0, 0])
        new = calibrate_prediction_interval(true, pred, daylight_mask=mask)
        lo, hi = apply_prediction_interval(pred, new, mask)
        self.assertTrue(np.all(hi[mask] == 100))
        self.assertTrue(np.all(hi[~mask] == 0))
        self.assertTrue(np.all(lo == 0))
        self.assertEqual(new["groups"]["daylight"]["fallback_sources"], ["group_pooled"] * 2)

    def test_sparse_step_pools_only_within_same_group(self):
        pred = np.zeros((40, 2))
        true = np.full((40, 2), 1000.)
        mask = np.zeros((40, 2), dtype=bool)
        mask[:, 0] = True
        true[:, 0] = 10
        new = calibrate_prediction_interval(true, pred, daylight_mask=mask)
        self.assertEqual(new["groups"]["daylight"]["quantiles_kw"], [10, 10])
        self.assertEqual(new["groups"]["night"]["quantiles_kw"], [1000, 1000])
        self.assertEqual(new["groups"]["daylight"]["sample_counts"], [40, 0])
        self.assertEqual(new["groups"]["daylight"]["fallback_sources"], ["step", "group_pooled"])

    def test_absent_group_is_auditable_and_uses_unconditioned_fallback(self):
        true, pred = np.full((40, 2), 5.), np.zeros((40, 2))
        new = calibrate_prediction_interval(true, pred, daylight_mask=np.zeros_like(true, dtype=bool))
        group = new["groups"]["daylight"]
        self.assertIsNone(group["pooled_quantile_kw"])
        self.assertEqual(group["finite_residual_count"], 0)
        self.assertEqual(group["fallback_sources"], ["unconditioned"] * 2)
        self.assertFalse(new["step_specific"])
        self.assertEqual(group["quantiles_kw"], [5, 5])

    def test_masks_and_saved_quantiles_are_validated(self):
        true, pred = np.ones((40, 2)), np.zeros((40, 2))
        for bad in (np.ones((40, 2)), np.ones((40, 1), dtype=bool)):
            with self.subTest(bad_shape=bad.shape), self.assertRaises(ValueError):
                calibrate_prediction_interval(true, pred, daylight_mask=bad)
        for minimum in (0, -1, 1.5):
            with self.assertRaises(ValueError):
                calibrate_prediction_interval(true, pred, min_step_samples=minimum)
        mask = np.ones_like(true, dtype=bool)
        new = calibrate_prediction_interval(true, pred, daylight_mask=mask)
        with self.assertRaisesRegex(ValueError, "require"):
            apply_prediction_interval(pred, new)
        for corrupt in ([float("nan"), 1], [-1, 1], [1]):
            new["groups"]["daylight"]["quantiles_kw"] = corrupt
            with self.assertRaises(ValueError):
                apply_prediction_interval(pred, new, mask)

    def test_legacy_intervals_and_single_origin_are_supported(self):
        legacy = calibrate_prediction_interval(np.ones((40, 2)), np.zeros((40, 2)))
        lo, hi = apply_prediction_interval(np.array([2., 3.]), legacy)
        np.testing.assert_equal(lo, [[1, 2]])
        np.testing.assert_equal(hi, [[3, 4]])
        new = calibrate_prediction_interval(np.ones((40, 2)), np.zeros((40, 2)),
                                           daylight_mask=np.ones((40, 2), dtype=bool))
        _, hi = apply_prediction_interval(np.array([2., 3.]), new, np.array([True, False]))
        np.testing.assert_equal(hi, [[3, 4]])

    def test_record_confidence_matches_requested_interval_level(self):
        calibration = calibrate_prediction_interval(np.ones((40, 2)), np.zeros((40, 2)), .9)
        record = build_forecast_record("id", [1, 2], "2025-01-01", 15, "pv", "model", "candidate", None,
                                       {}, calibration, {})
        validate_forecast_record(record)
        self.assertEqual(record["confidence"]["level"], .9)
        self.assertEqual(record["prediction_interval"]["nominal_coverage"], .9)


class RecalibrationTests(unittest.TestCase):
    def test_reuses_frozen_selection_and_calibrates_without_test_data(self):
        data = source_data()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            parent = root / "parent"
            original = train_model(data, parent, config=TrainingConfig(skip_gru=True))
            old_hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in parent.iterdir() if p.is_file()}
            changed = data.copy()
            test_start = pd.Timestamp(original["metadata"]["data_splits"]["test"]["start"]).tz_localize(None)
            changed.loc[changed.timestamp >= test_start, "pv_power_kw"] *= 20
            with patch("pv_forecast.models.GRUForecaster.fit", side_effect=AssertionError("must not train")):
                first = recalibrate_bundle(parent, data, root / "first")
                second = recalibrate_bundle(parent, changed, root / "second")
            self.assertEqual(first["metadata"]["calibration"], second["metadata"]["calibration"])
            self.assertEqual(first["metadata"]["baseline_fallback"]["calibration"],
                             second["metadata"]["baseline_fallback"]["calibration"])
            self.assertEqual(first["metadata"]["selected_model"], original["metadata"]["selected_model"])
            self.assertFalse(first["report"]["weights_retrained"])
            self.assertFalse(first["report"]["test_used_for_calibration"])
            self.assertEqual(old_hashes,
                {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in parent.iterdir() if p.is_file()})
            before, after = predict_record(parent, data), predict_record(root / "first", data)
            self.assertEqual(before["forecast_values"], after["forecast_values"])
            self.assertEqual(after["prediction_interval"]["method"], "empirical_daylight_absolute_residual")
            validate_forecast_record(after)
            with self.assertRaises(ValueError):
                recalibrate_bundle(parent, data, parent)
            with self.assertRaises(ValueError):
                recalibrate_bundle(parent, data, parent / "child")
            with self.assertRaises(ValueError):
                recalibrate_bundle(parent, data, root / "first")

    def test_truncated_partitions_cannot_produce_completed_bundle(self):
        data = source_data()
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory) / "parent"
            train_model(data, parent, config=TrainingConfig(skip_gru=True))
            output = Path(directory) / "bad"
            with self.assertRaises(DataValidationError):
                recalibrate_bundle(parent, data.iloc[:-96], output)
            self.assertFalse(output.exists())

    def test_opt_in_training_and_missing_gru_use_conditioned_baseline_intervals(self):
        data = source_data()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result = train_model(data, root, config=TrainingConfig(skip_gru=True, interval_conditioning="daylight"))
            self.assertEqual(result["metadata"]["calibration"]["conditioning"], "target_daylight")
            metadata = result["metadata"]
            metadata["selected_model"] = "gru"
            (root / "model_metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
            record = predict_record(root, data)
            self.assertEqual(record["status"], "degraded")
            times = pd.DatetimeIndex([v["timestamp"] for v in record["forecast_values"]])
            daylight = (times.hour >= 5) & (times.hour <= 20)
            groups = metadata["baseline_fallback"]["calibration"]["groups"]
            widths = np.where(daylight, groups["daylight"]["quantiles_kw"], groups["night"]["quantiles_kw"])
            pred = np.array([v["value"] for v in record["forecast_values"]])
            np.testing.assert_allclose([v["upper"] for v in record["prediction_interval"]["values"]], pred + widths)
            self.assertEqual(record["confidence"]["model_used"], record["fallback_forecast_ref"])

    def test_cli_exposes_recalibration_and_training_opt_in(self):
        parser = build_parser()
        args = parser.parse_args(["recalibrate", "--input", "data.csv", "--bundle-dir", "parent", "--output-dir", "new"])
        self.assertEqual(args.min_step_samples, 30)
        args = parser.parse_args(["train", "--input", "data.csv", "--output-dir", "new", "--interval-conditioning", "daylight"])
        self.assertEqual(args.interval_conditioning, "daylight")

    @unittest.skipUnless(__import__("importlib").util.find_spec("tensorflow"), "TensorFlow not installed")
    def test_gru_training_opt_in_retains_conditioned_primary_and_fallback(self):
        with tempfile.TemporaryDirectory() as directory:
            result = train_model(source_data(), directory, config=TrainingConfig(
                min_gru_days=0, require_gru=True, gru_epochs=1, gru_units=4,
                gru_batch_size=8, max_gru_samples=4, interval_conditioning="daylight", night_lower_bound="zero"),
                selected_model_override="gru")
            self.assertEqual(result["metrics"]["gru"]["status"], "completed")
            for calibration in (result["metadata"]["calibration"],
                                result["metadata"]["baseline_fallback"]["calibration"],
                                result["metrics"]["gru"]["calibration"]):
                self.assertEqual(calibration["conditioning"], "target_daylight")
                self.assertEqual(calibration["night_lower_bound"], "zero")
                self.assertEqual(len(calibration["groups"]["daylight"]["quantiles_kw"]), 96)
            record = predict_record(directory, source_data())
            self.assertEqual(record["status"], "available")
            self.assertEqual(record["prediction_interval"]["method"], "empirical_daylight_absolute_residual")
            validate_forecast_record(record)


if __name__ == "__main__":
    unittest.main()
