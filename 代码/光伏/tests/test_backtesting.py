import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from pv_forecast.backtesting import backtest_bundles
from pv_forecast.cli import build_parser
from pv_forecast.metrics import apply_prediction_interval, calibrate_prediction_interval, regression_metrics
from pv_forecast.pipeline import TrainingConfig, predict_record, train_model
from pv_forecast.recalibration import recalibrate_bundle


def data():
    index = pd.date_range("2025-01-28", periods=16*96, freq="15min")
    return pd.DataFrame({"timestamp": index,
        "pv_power_kw": 100*np.maximum(0, np.sin(np.pi*(index.hour+index.minute/60-6)/12)),
        "weather_type": np.where(index.dayofyear % 2, "clear", "rain")})


class NightSupportTests(unittest.TestCase):
    def test_zero_support_only_widens_night_lower_endpoint(self):
        true, pred = np.zeros((40, 2)), np.full((40, 2), 10.)
        mask = np.zeros_like(true, dtype=bool)
        mask[:, 1] = True
        old = calibrate_prediction_interval(true, pred, daylight_mask=mask)
        new = calibrate_prediction_interval(true, pred, daylight_mask=mask, night_lower_bound="zero")
        shifted = np.full((3, 2), 30.)
        future_mask = np.tile([False, True], (3, 1))
        old_lo, old_hi = apply_prediction_interval(shifted, old, future_mask)
        lo, hi = apply_prediction_interval(shifted, new, future_mask)
        np.testing.assert_array_equal(hi, old_hi)
        np.testing.assert_array_equal(lo[:, 1], old_lo[:, 1])
        np.testing.assert_array_equal(lo[:, 0], 0)
        self.assertTrue((lo <= old_lo).all())
        # Positive night generation is not assumed to be zero or guaranteed to
        # lie under the empirical upper bound; the support rule changes no point.
        self.assertLess(hi[0, 0], 100)

    def test_invalid_or_ungrouped_support_is_rejected(self):
        y = np.zeros((40, 2))
        with self.assertRaises(ValueError):
            calibrate_prediction_interval(y, y, night_lower_bound="zero")
        with self.assertRaises(ValueError):
            calibrate_prediction_interval(y, y, night_lower_bound="unexpected")
        cal = calibrate_prediction_interval(y, y)
        cal["night_lower_bound"] = "zero"
        with self.assertRaises(ValueError):
            apply_prediction_interval(y, cal)
        with tempfile.TemporaryDirectory() as root, self.assertRaises(ValueError):
            train_model(data(), root, config=TrainingConfig(skip_gru=True, night_lower_bound="zero"))

    def test_recalibration_preserves_and_reports_support_on_primary_and_fallback(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            source = data()
            train_model(source, root/"parent", config=TrainingConfig(skip_gru=True))
            result = recalibrate_bundle(root/"parent", source, root/"supported", night_lower_bound="zero")
            self.assertEqual(result["metadata"]["calibration"]["night_lower_bound"], "zero")
            self.assertEqual(result["metadata"]["baseline_fallback"]["calibration"]["night_lower_bound"], "zero")
            again = recalibrate_bundle(root/"supported", source, root/"inherited")
            self.assertEqual(again["metadata"]["calibration"]["night_lower_bound"], "zero")
            self.assertEqual(predict_record(root/"parent", source)["forecast_values"],
                             predict_record(root/"supported", source)["forecast_values"])
            meta = result["metadata"]
            meta["selected_model"] = "gru"
            (root/"supported"/"model_metadata.json").write_text(json.dumps(meta), encoding="utf-8")
            fallback = predict_record(root/"supported", source)
            self.assertEqual(fallback["status"], "degraded")
            self.assertEqual(fallback["confidence"]["night_lower_bound"], "zero")
            for point in fallback["prediction_interval"]["values"]:
                if pd.Timestamp(point["timestamp"]).hour < 5 or pd.Timestamp(point["timestamp"]).hour >= 21:
                    self.assertEqual(point["lower"], 0)


class BacktestTests(unittest.TestCase):
    def test_all_windows_batch_invariance_and_pooled_metrics(self):
        source = data()
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            train_model(source, root/"parent", config=TrainingConfig(skip_gru=True))
            recalibrate_bundle(root/"parent", source, root/"supported", night_lower_bound="zero")
            with patch("pv_forecast.models.GRUForecaster.fit", side_effect=AssertionError("no training")), \
                 patch("pv_forecast.metrics.calibrate_prediction_interval", side_effect=AssertionError("no calibration")):
                a = backtest_bundles([root/"parent", root/"supported"], source, root/"backtest.json", batch_windows=73)
                b = backtest_bundles([root/"parent", root/"supported"], source, batch_windows=151)
            self.assertEqual(a["evaluated_windows"], len(source)-288-96+1)
            self.assertEqual(a["skipped_origins"], 0)
            self.assertFalse(a["weights_retrained"])
            for name in ("parent", "supported"):
                for period in ("all_day", "daylight", "night"):
                    for metric in ("rmse", "mae", "r2", "coverage", "mean_interval_width_kw"):
                        if a["models"][name]["overall"][period][metric] is None:
                            self.assertIsNone(b["models"][name]["overall"][period][metric])
                            continue
                        np.testing.assert_allclose(a["models"][name]["overall"][period][metric],
                                                   b["models"][name]["overall"][period][metric], rtol=1e-10, atol=1e-10)
            origins = list(range(288, len(source)-95))
            predictions = [predict_record(root/"parent", source.iloc[:i], forecast_start=source.timestamp.iloc[i])
                           for i in origins[::100]]
            # Independently compare public prediction values for selected
            # origins with the streaming predictor rather than mirroring it.
            from pv_forecast import backtesting
            original_predict = backtesting._predict_windows
            seen = {}
            def capture(windows, *args):
                values = original_predict(windows, *args)
                for start, values_row in zip(windows.forecast_starts, values):
                    seen[start] = values_row
                return values
            with patch.object(backtesting, "_predict_windows", side_effect=capture):
                backtest_bundles([root/"parent"], source, batch_windows=151)
            for origin, record in zip(origins[::100], predictions):
                stamp = pd.Timestamp(source.timestamp.iloc[origin]).tz_localize("Asia/Shanghai")
                np.testing.assert_allclose(seen[stamp], [v["value"] for v in record["forecast_values"]])
            truth = np.array([source.pv_power_kw.iloc[i:i+96] for i in origins])
            pred = np.array([seen[pd.Timestamp(source.timestamp.iloc[i]).tz_localize("Asia/Shanghai")] for i in origins])
            full = regression_metrics(truth, pred)
            for metric in ("rmse", "mae", "r2"):
                np.testing.assert_allclose(a["models"]["parent"]["overall"]["all_day"][metric], full[metric], rtol=1e-10)
            self.assertEqual(sum(m["all_day"]["count"] for m in a["models"]["parent"]["months"].values()), truth.size)
            self.assertEqual(len(a["models"]["parent"]["months"]), 2)
            months = np.array([source.timestamp.iloc[i:i+96].dt.month for i in origins])
            self.assertEqual(a["models"]["parent"]["months"]["01"]["all_day"]["count"], int((months == 1).sum()))
            self.assertEqual(sum(m["count"] for m in a["models"]["parent"]["daylight_weather"].values()),
                             a["models"]["parent"]["overall"]["daylight"]["count"])

    def test_missing_labels_and_trailing_history_are_never_imputed_for_scoring(self):
        source = data()
        with tempfile.TemporaryDirectory() as root:
            parent = Path(root)/"parent"
            train_model(source, parent, config=TrainingConfig(skip_gru=True))
            missing = source.drop(index=999)
            result = backtest_bundles([parent], missing)
            # 96 origins whose label window contains point 999 plus origin
            # 1000 whose final historical observation is missing.
            self.assertEqual(result["skipped_origins"], 97)
            self.assertEqual(result["evaluated_windows"], len(source)-383-97)

    def test_cli_options_and_bad_batch_size(self):
        parser = build_parser()
        args = parser.parse_args(["backtest", "--input", "data.csv", "--bundle-dir", "one", "two", "--output", "report.json"])
        self.assertEqual(args.bundle_dir, ["one", "two"])
        args = parser.parse_args(["recalibrate", "--input", "data.csv", "--bundle-dir", "one", "--output-dir", "new", "--night-lower-bound", "zero"])
        self.assertEqual(args.night_lower_bound, "zero")
        for batch in (0, -1, 1.5, True):
            with self.assertRaises(ValueError):
                backtest_bundles([], data(), batch_windows=batch)

    def test_future_labels_and_weather_diagnostics_do_not_enter_first_prediction(self):
        from pv_forecast import backtesting
        source = data()
        with tempfile.TemporaryDirectory() as root:
            parent = Path(root)/"parent"
            train_model(source, parent, config=TrainingConfig(skip_gru=True))
            changed = source.copy()
            changed.loc[288:, "pv_power_kw"] = 9999
            changed.loc[288:, "weather_type"] = "arbitrary_diagnostic_label"
            real = backtesting._predict_windows
            first = []
            def capture(windows, *args):
                prediction = real(windows, *args)
                if not first:
                    first.append(prediction[0].copy())
                return prediction
            results = []
            for input_frame in (source, changed):
                first.clear()
                with patch.object(backtesting, "_predict_windows", side_effect=capture):
                    backtest_bundles([parent], input_frame, batch_windows=512)
                results.append(first[0])
            np.testing.assert_array_equal(results[0], results[1])

    def test_feature_contract_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            parent = Path(root)/"parent"
            result = train_model(data(), parent, config=TrainingConfig(skip_gru=True))
            meta = result["metadata"]
            meta["feature_names"] = list(reversed(meta["feature_names"]))
            (parent/"model_metadata.json").write_text(json.dumps(meta), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "feature order"):
                backtest_bundles([parent], data())


if __name__ == "__main__":
    unittest.main()
