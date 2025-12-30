"""
Logging configuration for Codon Encoder API.

This module provides a centralized logging setup with structured output
suitable for both development and production environments.
"""

import logging
import os
import sys
from typing import Optional


def setup_logging(
    name: str,
    level: Optional[str] = None,
    format_string: Optional[str] = None,
) -> logging.Logger:
    """
    Configure and return a logger instance.

    Args:
        name: Logger name (typically __name__ from calling module)
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
               Defaults to LOG_LEVEL env var or INFO
        format_string: Custom format string. Defaults to structured format.

    Returns:
        Configured logger instance

    Example:
        >>> from server.logging_config import setup_logging
        >>> logger = setup_logging(__name__)
        >>> logger.info("Model loaded successfully", extra={"model": "codon_encoder.pt"})
    """
    # Determine log level from arg, env var, or default
    if level is None:
        level = os.getenv("LOG_LEVEL", "INFO").upper()

    # Validate level
    numeric_level = getattr(logging, level, None)
    if not isinstance(numeric_level, int):
        numeric_level = logging.INFO

    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(numeric_level)

    # Avoid duplicate handlers if called multiple times
    if logger.handlers:
        return logger

    # Create console handler
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(numeric_level)

    # Default format: timestamp - name - level - message
    if format_string is None:
        format_string = "%(asctime)s | %(name)s | %(levelname)s | %(message)s"

    formatter = logging.Formatter(format_string, datefmt="%Y-%m-%d %H:%M:%S")
    handler.setFormatter(formatter)

    logger.addHandler(handler)

    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get or create a logger with default configuration.

    This is a convenience function for modules that just need a simple logger.

    Args:
        name: Logger name (typically __name__)

    Returns:
        Configured logger instance
    """
    return setup_logging(name)


# Pre-configured loggers for common modules
def get_model_logger() -> logging.Logger:
    """Get logger for model-related operations."""
    return setup_logging("codon.model")


def get_api_logger() -> logging.Logger:
    """Get logger for API operations."""
    return setup_logging("codon.api")


def get_inference_logger() -> logging.Logger:
    """Get logger for inference operations."""
    return setup_logging("codon.inference")


class LoggerAdapter(logging.LoggerAdapter):
    """
    Logger adapter that adds context to all log messages.

    Useful for adding request IDs, user info, or other contextual data.

    Example:
        >>> logger = get_logger(__name__)
        >>> adapter = LoggerAdapter(logger, {"request_id": "abc123"})
        >>> adapter.info("Processing request")
        # Output: 2024-01-15 10:30:00 | module | INFO | [request_id=abc123] Processing request
    """

    def process(self, msg: str, kwargs: dict) -> tuple[str, dict]:
        """Add extra context to log messages."""
        context_parts = [f"{k}={v}" for k, v in self.extra.items()]
        if context_parts:
            context_str = "[" + ", ".join(context_parts) + "] "
            msg = context_str + msg
        return msg, kwargs


# Module-level default logger
_default_logger: Optional[logging.Logger] = None


def get_default_logger() -> logging.Logger:
    """Get the default application logger."""
    global _default_logger
    if _default_logger is None:
        _default_logger = setup_logging("codon-encoder")
    return _default_logger
