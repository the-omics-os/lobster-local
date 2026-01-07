"""
SSL setup utilities for detecting and configuring SSL certificate handling.

This module provides utilities for testing SSL connectivity and guiding users
through SSL configuration during `lobster init`.

Environment Variables:
    LOBSTER_SSL_VERIFY: Set to 'false' to disable SSL verification (NOT RECOMMENDED)
    LOBSTER_SSL_CERT_PATH: Path to custom CA certificate bundle
"""

import logging
import ssl
import urllib.request
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

# Test endpoints for SSL connectivity
NCBI_TEST_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/einfo.fcgi"


def test_ssl_connectivity(
    test_url: str = NCBI_TEST_URL,
    timeout: int = 10,
) -> Tuple[bool, bool, Optional[str]]:
    """
    Test SSL connectivity to NCBI endpoint.

    Args:
        test_url: HTTPS endpoint to test (default: NCBI eutils)
        timeout: Request timeout in seconds

    Returns:
        Tuple of:
        - success (bool): Connection succeeded
        - is_ssl_error (bool): Error was SSL-related
        - error_message (Optional[str]): Error details if failed

    Example:
        >>> success, is_ssl, msg = test_ssl_connectivity()
        >>> if is_ssl:
        ...     print("SSL certificate error detected")
    """
    try:
        # Create default SSL context (with verification enabled)
        ssl_context = ssl.create_default_context()

        # Try to use certifi if available for better certificate handling
        try:
            import certifi

            ssl_context.load_verify_locations(cafile=certifi.where())
            logger.debug("Using certifi certificate bundle for SSL test")
        except ImportError:
            logger.debug("certifi not available, using system certificates")

        # Test connection
        with urllib.request.urlopen(
            test_url, context=ssl_context, timeout=timeout
        ) as response:
            response.read()  # Ensure full response
            logger.debug(f"SSL connectivity test succeeded: {test_url}")
            return (True, False, None)

    except ssl.SSLCertVerificationError as e:
        # Specific SSL certificate error
        logger.debug(f"SSL certificate verification failed: {e}")
        return (False, True, str(e))

    except ssl.SSLError as e:
        # Other SSL errors
        logger.debug(f"SSL error: {e}")
        return (False, True, str(e))

    except urllib.error.URLError as e:
        # Check if the underlying error is SSL-related
        error_str = str(e).lower()
        is_ssl = _is_ssl_error_message(error_str)
        logger.debug(f"URL error (SSL={is_ssl}): {e}")
        return (False, is_ssl, str(e))

    except Exception as e:
        # Check if error message indicates SSL issue
        error_str = str(e).lower()
        is_ssl = _is_ssl_error_message(error_str)
        logger.debug(f"Connectivity test failed (SSL={is_ssl}): {e}")
        return (False, is_ssl, str(e))


def _is_ssl_error_message(error_str: str) -> bool:
    """Check if error message indicates an SSL certificate issue."""
    error_lower = error_str.lower()
    ssl_patterns = [
        "certificate_verify_failed",
        "certificate verify failed",
        "ssl: certificate_verify_failed",
        "self-signed certificate",
        "ssl error",
        "ssl handshake",
        "unable to get local issuer certificate",
    ]
    return any(pattern in error_lower for pattern in ssl_patterns)


def get_ssl_error_guidance(error_message: Optional[str] = None) -> str:
    """
    Generate user-friendly guidance for SSL certificate errors.

    Args:
        error_message: Technical error message from SSL failure

    Returns:
        Formatted message for display
    """
    error_detail = ""
    if error_message:
        # Truncate long error messages
        short_msg = error_message[:80] + "..." if len(error_message) > 80 else error_message
        error_detail = f"\n  Technical details: {short_msg}\n"

    return f"""
[yellow bold]SSL Certificate Issue Detected[/yellow bold]

Lobster detected SSL certificate verification issues when connecting
to NCBI databases. This commonly occurs with:

  - Corporate networks with SSL inspection (proxy/firewall)
  - Networks using custom certificate authorities
  - Fresh systems with incomplete certificate stores
{error_detail}
[bold]Solutions (choose one):[/bold]

  [cyan]1. Upgrade certifi package (try this first):[/cyan]
     pip install --upgrade certifi

  [cyan]2. Use your corporate CA certificate:[/cyan]
     Add to your .env file:
     LOBSTER_SSL_CERT_PATH=/path/to/corporate-ca.pem

  [cyan]3. Disable SSL verification (TESTING ONLY - NOT SECURE):[/cyan]
     Add to your .env file:
     LOBSTER_SSL_VERIFY=false

     [red]Warning: This disables security checks and should only be[/red]
     [red]used in trusted corporate networks for testing.[/red]

For detailed troubleshooting:
  https://github.com/the-omics-os/lobster/wiki/SSL-Certificate-Issues
"""


def _prompt_fallback_ssl_fix(console) -> Optional[str]:
    """
    Prompt for fallback SSL fix options (when certifi upgrade didn't work).

    Args:
        console: Rich console for output

    Returns:
        Optional[str]: Environment variable line to add to .env, or None if skipped
    """
    from rich.prompt import Prompt

    console.print("[bold]Choose a fallback option:[/bold]")
    console.print("  [cyan]1[/cyan] - Use corporate CA certificate")
    console.print("  [cyan]2[/cyan] - Disable SSL verification (TESTING ONLY)")
    console.print("  [cyan]3[/cyan] - Skip (configure manually later)")
    console.print()

    fallback_choice = Prompt.ask(
        "[bold]Choose option[/bold]",
        choices=["1", "2", "3"],
        default="1",
    )

    if fallback_choice == "1":
        # Custom certificate path
        cert_path = Prompt.ask(
            "\n[bold]Enter path to your CA certificate file[/bold]",
            default="/etc/ssl/certs/ca-certificates.crt",
        )
        console.print(f"[green]Setting LOBSTER_SSL_CERT_PATH={cert_path}[/green]")
        return f"LOBSTER_SSL_CERT_PATH={cert_path}"

    elif fallback_choice == "2":
        # Disable SSL verification
        console.print(
            "\n[yellow]Warning: SSL verification will be disabled.[/yellow]"
        )
        console.print(
            "[yellow]This is NOT secure and should only be used for testing.[/yellow]\n"
        )
        return "LOBSTER_SSL_VERIFY=false"

    else:
        # Skip
        console.print(
            "\n[dim]Skipping SSL configuration. You may encounter errors later.[/dim]"
        )
        return None


def prompt_ssl_fix(console, error_message: Optional[str] = None) -> Optional[str]:
    """
    Prompt user to choose SSL fix option during init.

    Args:
        console: Rich console for output
        error_message: Technical error message from SSL failure

    Returns:
        Optional[str]: Environment variable line to add to .env, or None if skipped
    """
    from rich.panel import Panel
    from rich.prompt import Prompt

    # Show the error guidance
    console.print()
    console.print(
        Panel(
            get_ssl_error_guidance(error_message),
            border_style="yellow",
            title="SSL Configuration Required",
        )
    )
    console.print()

    # Prompt user
    choice = Prompt.ask(
        "[bold]Choose an option[/bold]",
        choices=["1", "2", "3", "skip"],
        default="1",
    )

    if choice == "1":
        # Try upgrading certifi automatically
        console.print("\n[cyan]Installing/upgrading certifi package...[/cyan]")

        import subprocess
        import sys

        try:
            # Run pip install in the current Python environment
            result = subprocess.run(
                [sys.executable, "-m", "pip", "install", "--upgrade", "certifi"],
                capture_output=True,
                text=True,
                timeout=60
            )

            if result.returncode == 0:
                console.print("[green]✓ certifi installed successfully[/green]")

                # Retest SSL connectivity
                console.print("[dim]Retesting SSL connectivity...[/dim]")
                success, is_ssl_error, new_error_msg = test_ssl_connectivity(timeout=10)

                if success:
                    console.print("[green]✓ SSL connectivity now working![/green]")
                    console.print("[dim]Continuing with setup...[/dim]\n")
                    return None  # SSL fixed, don't need to add env vars
                else:
                    console.print("[yellow]⚠️  SSL issue persists after certifi upgrade[/yellow]")
                    console.print("[yellow]This suggests a corporate proxy/firewall issue.[/yellow]\n")
                    # Fall through to prompt for other options
                    return _prompt_fallback_ssl_fix(console)
            else:
                console.print(f"[yellow]⚠️  certifi installation failed: {result.stderr[:100]}[/yellow]")
                console.print("[yellow]Falling back to manual configuration...[/yellow]\n")
                return _prompt_fallback_ssl_fix(console)

        except subprocess.TimeoutExpired:
            console.print("[yellow]⚠️  certifi installation timed out[/yellow]")
            return _prompt_fallback_ssl_fix(console)
        except Exception as e:
            console.print(f"[yellow]⚠️  Error installing certifi: {e}[/yellow]")
            return _prompt_fallback_ssl_fix(console)

    elif choice == "2":
        # User chose to provide CA cert directly (skipping certifi)
        cert_path = Prompt.ask(
            "\n[bold]Enter path to your CA certificate file[/bold]",
            default="/etc/ssl/certs/ca-certificates.crt",
        )
        console.print(f"[green]Setting LOBSTER_SSL_CERT_PATH={cert_path}[/green]")
        return f"LOBSTER_SSL_CERT_PATH={cert_path}"

    elif choice == "3":
        # User chose to disable SSL verification directly (skipping certifi)
        console.print(
            "\n[yellow]Warning: SSL verification will be disabled.[/yellow]"
        )
        console.print(
            "[yellow]This is NOT secure and should only be used for testing.[/yellow]\n"
        )
        return "LOBSTER_SSL_VERIFY=false"

    else:
        # Skip
        console.print(
            "\n[dim]Skipping SSL configuration. You may encounter errors later.[/dim]"
        )
        return None


def test_ssl_with_disabled_verification(
    test_url: str = NCBI_TEST_URL,
    timeout: int = 10,
) -> Tuple[bool, Optional[str]]:
    """
    Test if connection works with SSL verification disabled.

    This helps confirm the issue is SSL-related (not network/firewall).

    Args:
        test_url: HTTPS endpoint to test
        timeout: Request timeout in seconds

    Returns:
        Tuple of (success: bool, error_message: Optional[str])
    """
    try:
        # Create unverified SSL context
        ssl_context = ssl._create_unverified_context()

        with urllib.request.urlopen(
            test_url, context=ssl_context, timeout=timeout
        ) as response:
            response.read()
            return (True, None)

    except Exception as e:
        return (False, str(e))
