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

    Handles two scenarios:
    1. CLI usage: ConsoleManager sets up RichHandler on root logger.
       We detect this and let logs propagate to root (single output).
    2. Direct usage: No RichHandler on root. We add our own StreamHandler
       and disable propagation to prevent duplicate output.

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
        # This indicates CLI mode where ConsoleManager is active
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

        if has_rich_handler:
            # CLI mode: Let logs propagate to root's RichHandler
            # No additional handler needed, propagation handles output
            pass
        else:
            # Direct usage (tests, scripts): Add our own handler
            # and disable propagation to prevent duplicate output
            handler = logging.StreamHandler(sys.stdout)
            formatter = logging.Formatter(
                "[%(asctime)s] %(levelname)s - [%(name)s] - %(message)s"
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            # CRITICAL: Disable propagation to prevent root logger's
            # basicConfig handler from also outputting the same message
            logger.propagate = False

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
