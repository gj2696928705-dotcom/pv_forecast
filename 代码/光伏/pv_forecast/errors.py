class ForecastError(Exception):
    """Base error for the photovoltaic forecasting package."""


class ModelUnavailableError(ForecastError):
    """Raised when a requested model cannot be used safely."""


class InsufficientHistoryError(ForecastError):
    """Raised when there is not enough clean history for a forecast."""
