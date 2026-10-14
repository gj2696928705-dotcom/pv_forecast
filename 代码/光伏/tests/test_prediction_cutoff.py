import tempfile
import unittest

import numpy as np
import pandas as pd

from pv_forecast.data import load_and_regularize
from pv_forecast.pipeline import TrainingConfig, predict_record, train_model


class PredictionCutoffTests(unittest.TestCase):
    def data(self):
        return pd.DataFrame({"timestamp": pd.date_range("2025-01-01", periods=16 * 96, freq="15min"),
                             "pv_power_kw": np.arange(16 * 96, dtype=float)})

    def test_future_rows_cannot_repair_trailing_history_gap(self):
        data = self.data()
        origin = data.timestamp.iloc[1000]
        with tempfile.TemporaryDirectory() as bundle:
            train_model(data, bundle, config=TrainingConfig(skip_gru=True))
            missing = data.drop(index=999)
            full = predict_record(bundle, missing, forecast_start=origin)
            past = predict_record(bundle, missing[missing.timestamp < origin], forecast_start=origin)
            self.assertEqual(full["status"], past["status"])
            self.assertEqual(full["status"], "unavailable")
            self.assertEqual(full["input_data_quality_summary"], past["input_data_quality_summary"])

    def test_future_values_do_not_participate_in_input_quality_or_validation(self):
        data = self.data()
        origin = data.timestamp.iloc[1000]
        with tempfile.TemporaryDirectory() as bundle:
            train_model(data, bundle, config=TrainingConfig(skip_gru=True))
            past = predict_record(bundle, data[data.timestamp < origin], forecast_start=origin)
            changed = data.copy()
            changed.loc[changed.timestamp >= origin, "pv_power_kw"] = -999
            full = predict_record(bundle, changed, forecast_start=origin)
            self.assertEqual(full["forecast_values"], past["forecast_values"])
            self.assertEqual(full["input_data_quality_summary"], past["input_data_quality_summary"])
            empty = predict_record(bundle, data, forecast_start="2024-01-01")
            self.assertEqual(empty["status"], "unavailable")
            self.assertEqual(empty["forecast_start_at"], "2024-01-01T00:00:00+08:00")


if __name__ == "__main__":
    unittest.main()
