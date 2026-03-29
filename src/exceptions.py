"""Project-wide exception hierarchy for B2."""


class ProjectBaseError(Exception):
    """Base class for all B2 project exceptions."""


class DataLoadError(ProjectBaseError):
    """Raised when raw data cannot be loaded or parsed."""


class DataValidationError(ProjectBaseError):
    """Raised when pandera schema validation fails."""


class ModelNotFoundError(ProjectBaseError):
    """Raised when a trained model file is missing."""


class PredictionError(ProjectBaseError):
    """Raised when inference fails (malformed input, shape mismatch, etc.)."""


class ConfigError(ProjectBaseError):
    """Raised when config/config.yaml is missing or has invalid keys."""
