"""Custom exceptions for the anomaly detection platform."""


class AnomalyDetectionException(Exception):
    """Base exception for the anomaly detection platform."""
    pass


class DataIngestionError(AnomalyDetectionException):
    """Raised when data ingestion fails."""
    pass


class SchemaValidationError(AnomalyDetectionException):
    """Raised when data schema validation fails."""
    pass


class ModelTrainingError(AnomalyDetectionException):
    """Raised when model training fails."""
    pass


class ModelInferenceError(AnomalyDetectionException):
    """Raised when model inference fails."""
    pass


class FeatureStoreError(AnomalyDetectionException):
    """Raised when feature store operations fail."""
    pass


class AlertingError(AnomalyDetectionException):
    """Raised when alerting fails."""
    pass


class AuthenticationError(AnomalyDetectionException):
    """Raised when authentication fails."""
    pass


class AuthorizationError(AnomalyDetectionException):
    """Raised when authorization fails."""
    pass


class ConfigurationError(AnomalyDetectionException):
    """Raised when configuration is invalid."""
    pass
