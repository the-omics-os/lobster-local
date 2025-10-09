"""
SSL/HTTPS utilities for handling certificate verification.

This module provides utilities for creating SSL contexts with proper
certificate handling, supporting both default system certificates and
custom certificate bundles. It also provides graceful error handling
for SSL verification issues common in different environments.
"""

import os
import ssl
from typing import Optional

from lobster.utils.logger import get_logger

logger = get_logger(__name__)


def create_ssl_context(
    verify: Optional[bool] = None, cert_path: Optional[str] = None
) -> Optional[ssl.SSLContext]:
    """
    Create an SSL context for HTTPS requests with proper certificate handling.

    This function creates an SSL context that:
    1. Uses certifi package for consistent certificate bundle (if available)
    2. Respects environment variables for SSL verification control
    3. Supports custom certificate bundles
    4. Provides security-first defaults

    Args:
        verify: Whether to verify SSL certificates. If None, reads from settings/env.
                Default is True (verify certificates).
        cert_path: Path to custom certificate bundle. If None, reads from settings/env.

    Returns:
        ssl.SSLContext configured appropriately, or None if verification is disabled.

    Environment Variables:
        LOBSTER_SSL_VERIFY: Set to 'false' to disable SSL verification (NOT RECOMMENDED)
        LOBSTER_SSL_CERT_PATH: Path to custom certificate bundle file

    Example:
        >>> context = create_ssl_context()
        >>> urllib.request.urlopen(url, context=context)
    """
    # Import settings here to avoid circular imports
    from lobster.config.settings import get_settings

    settings = get_settings()

    # Determine if we should verify SSL
    if verify is None:
        verify = settings.SSL_VERIFY

    # Determine cert path
    if cert_path is None:
        cert_path = settings.SSL_CERT_PATH

    # If verification is disabled, return None (urllib will use unverified context)
    if not verify:
        logger.warning(
            "SSL certificate verification is DISABLED. "
            "This is insecure and should only be used for testing. "
            "Set LOBSTER_SSL_VERIFY=true to re-enable verification."
        )
        return ssl._create_unverified_context()

    # Create default SSL context with certificate verification
    try:
        # Try to use certifi package for better certificate handling
        try:
            import certifi

            context = ssl.create_default_context(cafile=certifi.where())
            logger.debug("Using certifi certificate bundle")
        except ImportError:
            # Fall back to system certificates
            context = ssl.create_default_context()
            logger.debug("Using system certificate bundle (certifi not available)")

        # Load custom certificate bundle if provided
        if cert_path:
            if os.path.exists(cert_path):
                context.load_verify_locations(cafile=cert_path)
                logger.info(f"Loaded custom certificate bundle from: {cert_path}")
            else:
                logger.warning(f"Custom certificate path does not exist: {cert_path}")

        return context

    except Exception as e:
        logger.error(f"Error creating SSL context: {e}")
        # Return default context as fallback
        return ssl.create_default_context()


def get_ssl_error_help_message(error: Exception) -> str:
    """
    Generate a helpful error message for SSL certificate verification failures.

    Args:
        error: The SSL error exception

    Returns:
        Formatted error message with troubleshooting steps
    """
    help_msg = """
╔════════════════════════════════════════════════════════════════════════════╗
║                   SSL CERTIFICATE VERIFICATION FAILED                      ║
╚════════════════════════════════════════════════════════════════════════════╝

This usually occurs when:
  1. System is missing required root/intermediate certificates
  2. Running on a fresh/guest account with minimal certificate store
  3. Corporate proxy is using custom certificates
  4. Network configuration is blocking certificate validation

Recommended Solutions:

  1. Install/upgrade certifi package (recommended):
     pip install --upgrade certifi

  2. Use custom certificate bundle:
     export LOBSTER_SSL_CERT_PATH=/path/to/your/certificates.pem

  3. For testing ONLY (NOT RECOMMENDED for production):
     export LOBSTER_SSL_VERIFY=false

Technical Details:
  {error_details}

For more help, see: https://github.com/the-omics-os/lobster/wiki/SSL-Certificate-Issues
"""

    error_details = str(error)
    if "CERTIFICATE_VERIFY_FAILED" in error_details:
        error_details += "\n  → Certificate chain verification failed"
    if "self-signed certificate" in error_details:
        error_details += "\n  → Self-signed certificate detected in chain"

    return help_msg.format(error_details=error_details)


def handle_ssl_error(error: Exception, url: str, logger_instance=None) -> None:
    """
    Handle SSL errors with helpful logging and user guidance.

    Args:
        error: The SSL error exception
        url: The URL that caused the error
        logger_instance: Optional logger instance to use
    """
    log = logger_instance if logger_instance else logger

    log.error(f"SSL certificate verification failed for URL: {url}")
    log.error(get_ssl_error_help_message(error))
