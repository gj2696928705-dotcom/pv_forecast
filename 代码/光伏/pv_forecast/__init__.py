"""Independent photovoltaic power forecasting package.

The package deliberately remains separate from the station PoC.  It produces
forecast records that can be consumed by the energy coordination agent later,
but it never emits device commands or changes certified operating limits.
"""

from .data import DataValidationError, InputSpec, load_and_regularize
from .metrics import calibrate_prediction_interval, regression_metrics
from .records import build_forecast_record, validate_forecast_record

__all__ = [
    "DataValidationError",
    "InputSpec",
    "load_and_regularize",
    "calibrate_prediction_interval",
    "regression_metrics",
    "build_forecast_record",
    "validate_forecast_record",
]

__version__ = "0.1.0"
