"""
Logging configuration for the application.

This module sets up consistent logging across all components of the application,
making it easier to track events and debug issues.
"""

import logging
import sys


def setup_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    Configure a logger with consistent formatting.

    Args:
        name: Name of the logger
        level: Logging level (default: INFO)

    Returns:
        logging.Logger: Configured logger instance
    """
    logger = logging.getLogger(name)

    # Only configure if it hasn't been configured yet
    if not logger.handlers:
        logger.setLevel(level)

        # Check if RichHandler is already configured on root logger
        # This prevents duplicate logging when using the CLI
        root_logger = logging.getLogger()
        has_rich_handler = False

        try:
            from rich.logging import RichHandler

            has_rich_handler = any(
                isinstance(handler, RichHandler) for handler in root_logger.handlers
            )
        except ImportError:
            # RichHandler not available, proceed with standard handler
            pass

        # Only add StreamHandler if RichHandler is not present
        # This prevents duplicate log output in the CLI
        if not has_rich_handler:
            handler = logging.StreamHandler(sys.stdout)
            formatter = logging.Formatter(
                "[%(asctime)s] %(levelname)s - [%(name)s] - %(message)s"
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)

    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger with the given name.

    This function retrieves an existing logger or creates a new one.

    Args:
        name: Name of the logger

    Returns:
        logging.Logger: Logger instance
    """
    return setup_logger(name)
