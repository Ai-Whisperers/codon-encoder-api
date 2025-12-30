"""
Custom exception hierarchy for Codon Encoder API.

This module defines domain-specific exceptions for better error handling
and clearer error messages throughout the application.
"""


class CodonEncoderError(Exception):
    """Base exception for all Codon Encoder errors."""

    def __init__(self, message: str, details: dict | None = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}

    def __str__(self) -> str:
        if self.details:
            return f"{self.message} | Details: {self.details}"
        return self.message


class InvalidSequenceError(CodonEncoderError):
    """Raised when a DNA or protein sequence is invalid."""

    def __init__(
        self,
        message: str,
        sequence: str | None = None,
        position: int | None = None,
    ):
        details = {}
        if sequence is not None:
            details["sequence_preview"] = sequence[:50] + "..." if len(sequence) > 50 else sequence
        if position is not None:
            details["position"] = position
        super().__init__(message, details)
        self.sequence = sequence
        self.position = position


class InvalidCodonError(CodonEncoderError):
    """Raised when a codon is not in the standard genetic code."""

    def __init__(self, codon: str, message: str | None = None):
        msg = message or f"Invalid codon: '{codon}'"
        super().__init__(msg, {"codon": codon})
        self.codon = codon


class ModelLoadError(CodonEncoderError):
    """Raised when model loading fails."""

    def __init__(self, message: str, model_path: str | None = None):
        details = {}
        if model_path:
            details["model_path"] = model_path
        super().__init__(message, details)
        self.model_path = model_path


class ModelNotLoadedError(CodonEncoderError):
    """Raised when trying to use a model that hasn't been loaded."""

    def __init__(self, message: str = "Model not loaded. Call load() first."):
        super().__init__(message)


class CheckpointError(CodonEncoderError):
    """Raised when a model checkpoint is invalid or corrupted."""

    def __init__(self, message: str, missing_keys: list[str] | None = None):
        details = {}
        if missing_keys:
            details["missing_keys"] = missing_keys
        super().__init__(message, details)
        self.missing_keys = missing_keys or []


class ProjectionError(CodonEncoderError):
    """Raised when embedding projection fails."""

    def __init__(self, message: str, method: str | None = None):
        details = {}
        if method:
            details["projection_method"] = method
        super().__init__(message, details)
        self.method = method


class BatchProcessingError(CodonEncoderError):
    """Raised when batch processing encounters errors."""

    def __init__(
        self,
        message: str,
        total_sequences: int = 0,
        failed_count: int = 0,
        errors: list[str] | None = None,
    ):
        details = {
            "total_sequences": total_sequences,
            "failed_count": failed_count,
        }
        if errors:
            details["errors"] = errors[:10]  # Limit to first 10 errors
        super().__init__(message, details)
        self.total_sequences = total_sequences
        self.failed_count = failed_count
        self.errors = errors or []


class ConfigurationError(CodonEncoderError):
    """Raised when configuration is invalid."""

    def __init__(self, message: str, config_key: str | None = None):
        details = {}
        if config_key:
            details["config_key"] = config_key
        super().__init__(message, details)
        self.config_key = config_key
