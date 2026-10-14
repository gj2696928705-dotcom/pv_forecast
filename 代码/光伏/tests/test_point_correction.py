import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from pv_forecast.backtesting import backtest_bundles
from pv_forecast.cli import build_parser
from pv_forecast.data import InputSpec, build_windows, load_and_regularize
from pv_forecast.pipeline import (TrainingConfig, _fit_history_scaler, _fit_scalers,
                                  _prediction_spec, predict_record, train_model)
from pv_forecast.point_correction import SolarWeatherRidge
from pv_forecast.point_experiments import fit_point_correction_bundle
from pv_forecast.recalibration import recalibrate_bundle
from pv_forecast.weather import solar_elevation


def source_data():
    stamp = pd.date_range("2025-01-01", periods=40*96, freq="15min", tz="Asia/Shanghai")
    solar = np.maximum(np.sin(np.deg2rad(solar_elevation(stamp, 30.9, 111.35))), 0)
    return pd.DataFrame({"timestamp": stamp, "pv_power_kw": 1000*solar,
        "SWGDN": 1000*solar, "temperature_c": 20., "cloud_cover": .3, "weather_type": "clear"})


def make_parent(root, source):
    spec = InputSpec(latitude=30.9, longitude=111.35,
                     history_weather_columns=("SWGDN", "temperature_c", "cloud_cover"))
    parent = root/"parent"
    result = train_model(source, parent, spec, TrainingConfig(skip_gru=True))
    meta = result["metadata"]
    meta.update(selected_model="gru", selected_gru_candidate="gru32_scheduled", rolling_validation_ref="rolling_validation.json")
    regular = load_and_regularize(source, spec)
    rolling = {"status": "completed", "scope": "full_windows", "candidates": {
        name: {} for name in ("gru32_control", "gru32_scheduled", "gru16_scheduled")}, "folds": []}
    def windows(lo, hi):
        return build_windows(regular, lo, hi, latitude=spec.latitude, longitude=spec.longitude,
                             history_weather_columns=spec.history_weather_columns)
    for n, (train_days, val_days) in enumerate(((8, 12), (18, 22), (24, 28))):
        train, val = windows(0, train_days*96), windows(train_days*96, val_days*96)
        split = {"name": f"fold{n}", "train_start": regular.frame.index[0].isoformat(),
            "train_end": regular.frame.index[train_days*96-1].isoformat(),
            "validation_start": regular.frame.index[train_days*96].isoformat(),
            "validation_end": regular.frame.index[val_days*96-1].isoformat(),
            "training_windows_available": train.size, "training_windows_used": train.size,
            "validation_windows": val.size}
        rolling["folds"].append({"name": split["name"]})
        target, future = _fit_scalers(train)
        saved = {"target_scaler": target, "feature_scaler": future, "history_scaler": _fit_history_scaler(train)}
        for name in rolling["candidates"]:
            folder = parent/"experiments"/split["name"]/name
            folder.mkdir(parents=True)
            (folder/"gru.keras").write_bytes(b"fake_gru_for_unit_test_only")
            joblib.dump(saved, folder/"scalers.joblib")
            for filename, value in (("data_split.json", split), ("training_summary.json", {"epochs_completed": 1}),
                ("features.json", {"history": meta["history_feature_names"], "future": meta["feature_names"]})):
                (folder/filename).write_text(json.dumps(value), encoding="utf-8")
    final_train = windows(0, meta["data_splits"]["train"]["rows"])
    target, future = _fit_scalers(final_train)
    joblib.dump({"target_scaler": target, "feature_scaler": future, "history_scaler": _fit_history_scaler(final_train)}, parent/"scalers.joblib")
    (parent/"gru.keras").write_bytes(b"fake_gru_for_unit_test_only")
    (parent/"model_metadata.json").write_text(json.dumps(meta), encoding="utf-8")
    (parent/"rolling_validation.json").write_text(json.dumps(rolling), encoding="utf-8")
    return parent, meta, spec, regular


class FakeGRU:
    def predict(self, history, future):
        # A deliberately biased frozen predictor. This test double verifies
        # orchestration only; actual GRU evidence comes from the full runner.
        return np.full(future.shape[:2], .3)


class PointCorrectionTests(unittest.TestCase):
    def test_training_stats_match_independent_ridge_and_batching(self):
        source = source_data()
        spec = InputSpec(latitude=30.9, longitude=111.35, history_weather_columns=("SWGDN", "cloud_cover"))
        regular = load_and_regularize(source, spec)
        w = build_windows(regular, 0, 12*96, latitude=spec.latitude, longitude=spec.longitude,
                          history_weather_columns=spec.history_weather_columns)
        raw = np.full(w.target.shape, 300.)
        a = SolarWeatherRidge(.01).fit(w.gru_history, w.future_features, raw, w.target,
            w.history_feature_names, w.feature_names, batch_windows=73)
        b = SolarWeatherRidge(.01).fit(w.gru_history, w.future_features, raw, w.target,
            w.history_feature_names, w.feature_names, batch_windows=256)
        pa = a.predict(w.gru_history, w.future_features, raw, w.history_feature_names, w.feature_names)
        pb = b.predict(w.gru_history, w.future_features, raw, w.history_feature_names, w.feature_names)
        np.testing.assert_allclose(pa, pb, rtol=1e-8, atol=1e-5)
        x, day = a.features(w.gru_history, w.future_features, raw, w.history_feature_names, w.feature_names)
        lead = 40
        mask = day[:, lead]
        standard = (x[mask, lead]-a.mean[lead])/a.scale[lead]
        reference = Ridge(alpha=.01*mask.sum(), fit_intercept=True).fit(standard, w.target[mask, lead])
        np.testing.assert_allclose(pa[mask, lead], np.maximum(reference.predict(standard), 0), rtol=1e-8, atol=1e-5)
        self.assertEqual(a.window_count, w.size)
        np.testing.assert_array_equal(pa[~day], 0)

    def test_contract_validation_and_future_weather_labels_are_excluded(self):
        source = source_data()
        regular = load_and_regularize(source, InputSpec(latitude=30.9, longitude=111.35))
        w = build_windows(regular, 0, 10*96, latitude=30.9, longitude=111.35)
        raw = np.full(w.target.shape, 300.)
        model = SolarWeatherRidge(.1).fit(w.gru_history, w.future_features, raw, w.target,
            w.history_feature_names, w.feature_names)
        extra = np.concatenate([w.future_features, np.ones((*raw.shape, 1))*9999], axis=-1)
        a, _ = model.features(w.gru_history, extra, raw, w.history_feature_names, [*w.feature_names, "future_observed_cloud"])
        b, _ = model.features(w.gru_history, w.future_features, raw, w.history_feature_names, w.feature_names)
        np.testing.assert_array_equal(a, b)
        with self.assertRaisesRegex(ValueError, "contract"):
            model.predict(w.gru_history, w.future_features[..., ::-1], raw, w.history_feature_names, w.feature_names[::-1])
        for alpha in (0, -1, float("nan")):
            with self.assertRaises(ValueError):
                SolarWeatherRidge(alpha)
        with self.assertRaises(ValueError):
            model.fit(w.gru_history, w.future_features, raw, w.target, w.history_feature_names, w.feature_names, batch_windows=0)

    def test_complete_correction_workflow_reload_fallback_and_recalibration(self):
        source = source_data()
        with tempfile.TemporaryDirectory() as tmp, patch("pv_forecast.models.GRUForecaster.load", side_effect=lambda path: FakeGRU()), \
             patch("pv_forecast.models.GRUForecaster.fit", side_effect=AssertionError("GRU must stay frozen")):
            root = Path(tmp)
            parent, meta, spec, _ = make_parent(root, source)
            result = fit_point_correction_bundle(parent, source, root/"corrected", batch_windows=512)
            self.assertIsNotNone(result["metadata"]["point_correction"])
            self.assertEqual(result["report"]["parent_full_gru_artifact_count"], 9)
            self.assertFalse(result["report"]["test_used_for_selection"])
            record = predict_record(root/"corrected", source)
            self.assertEqual(record["status"], "available")
            self.assertEqual(record["confidence"]["point_correction"], "solar_weather_ridge")
            after = recalibrate_bundle(root/"corrected", source, root/"recalibrated")
            self.assertEqual(record["forecast_values"], predict_record(root/"recalibrated", source)["forecast_values"])
            self.assertTrue((root/"recalibrated"/"point_correction.joblib").exists())
            comparison = backtest_bundles([parent, root/"corrected"], source)
            self.assertLess(comparison["models"]["corrected"]["overall"]["daylight"]["rmse"],
                            comparison["models"]["parent"]["overall"]["daylight"]["rmse"])
            self.assertTrue((root/"corrected"/"selection_and_calibration_frozen.json").is_file())
            (root/"corrected"/"point_correction.joblib").write_bytes(b"corrupt")
            fallback = predict_record(root/"corrected", source)
            self.assertEqual(fallback["status"], "degraded")
            self.assertIsNone(fallback["confidence"]["point_correction"])
            self.assertEqual(fallback["confidence"]["night_lower_bound"], "zero")
            with self.assertRaises(Exception):
                backtest_bundles([root/"corrected"], source)

    def test_test_values_cannot_change_fitted_correction_or_calibration(self):
        source = source_data()
        with tempfile.TemporaryDirectory() as tmp, patch("pv_forecast.models.GRUForecaster.load", side_effect=lambda path: FakeGRU()):
            root = Path(tmp)
            parent, meta, _, _ = make_parent(root, source)
            changed = source.copy()
            changed.loc[changed.timestamp >= pd.Timestamp(meta["data_splits"]["test"]["start"]), "pv_power_kw"] = 99999
            a = fit_point_correction_bundle(parent, source, root/"a")
            b = fit_point_correction_bundle(parent, changed, root/"b")
            self.assertEqual(a["report"]["selected_correction"], b["report"]["selected_correction"])
            self.assertEqual(a["metadata"]["point_correction"]["sha256"], b["metadata"]["point_correction"]["sha256"])
            self.assertEqual(a["metadata"]["calibration"], b["metadata"]["calibration"])
            self.assertNotEqual(a["metrics"]["test"]["rmse"], b["metrics"]["test"]["rmse"])

    def test_invalid_parent_and_existing_output_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaises(ValueError):
                fit_point_correction_bundle(root/"missing", source_data(), root, batch_windows=512)
        args = build_parser().parse_args(["correct-point", "--input", "data.csv", "--bundle-dir", "parent", "--output-dir", "new"])
        self.assertEqual(args.batch_windows, 1024)


if __name__ == "__main__":
    unittest.main()
