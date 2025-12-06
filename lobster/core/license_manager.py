"""
License manager for premium entitlement handling.

This module manages license/entitlement files that control access to
premium and custom features. Entitlements are stored in ~/.lobster/license.json
and are issued by the Omics-OS license service during activation.

Entitlement File Structure:
{
    "tier": "premium",
    "customer_id": "cust_abc123",
    "issued_at": "2024-12-01T00:00:00Z",
    "expires_at": "2025-12-01T00:00:00Z",
    "custom_packages": ["lobster-custom-databiomix"],
    "features": ["cloud_compute", "priority_support"],
    "signature": "base64_encoded_signature..."
}

Usage:
    from lobster.core.license_manager import (
        load_entitlement,
        get_current_tier,
        is_feature_enabled,
    )

    entitlement = load_entitlement()
    tier = get_current_tier()
    if is_feature_enabled("cloud_compute"):
        # Enable cloud features
        pass
"""

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# =============================================================================
# CONSTANTS
# =============================================================================

# Default location for license/entitlement file
DEFAULT_LICENSE_PATH = Path.home() / ".lobster" / "license.json"

# Environment variable to override license path (useful for testing)
LICENSE_PATH_ENV_VAR = "LOBSTER_LICENSE_PATH"

# Environment variable to set tier directly (for development/testing)
TIER_ENV_VAR = "LOBSTER_SUBSCRIPTION_TIER"

# Environment variable for cloud API key (implies premium tier)
CLOUD_KEY_ENV_VAR = "LOBSTER_CLOUD_KEY"

# Default entitlement for free tier users
DEFAULT_ENTITLEMENT: Dict[str, Any] = {
    "tier": "free",
    "customer_id": None,
    "issued_at": None,
    "expires_at": None,
    "custom_packages": [],
    "features": ["local_only", "community_support"],
    "valid": True,
    "source": "default",
}

# =============================================================================
# LICENSE FILE OPERATIONS
# =============================================================================


def get_license_path() -> Path:
    """
    Get the path to the license file.

    Checks environment variable first, then uses default location.

    Returns:
        Path to license.json file
    """
    env_path = os.environ.get(LICENSE_PATH_ENV_VAR)
    if env_path:
        return Path(env_path)
    return DEFAULT_LICENSE_PATH


def load_entitlement() -> Dict[str, Any]:
    """
    Load and validate entitlement from license file.

    Checks in order:
    1. LOBSTER_SUBSCRIPTION_TIER environment variable (dev override)
    2. LOBSTER_CLOUD_KEY environment variable (implies premium tier)
    3. License file at ~/.lobster/license.json
    4. Falls back to free tier defaults

    Returns:
        Entitlement dict with tier, features, custom_packages, etc.
    """
    # Check for environment variable override (development/testing)
    env_tier = os.environ.get(TIER_ENV_VAR)
    if env_tier:
        logger.debug(f"Using tier from environment variable: {env_tier}")
        return {
            **DEFAULT_ENTITLEMENT,
            "tier": env_tier.lower(),
            "source": "environment",
            "valid": True,
        }

    # Check for cloud API key (implies premium tier)
    cloud_key = os.environ.get(CLOUD_KEY_ENV_VAR)
    if cloud_key:
        logger.debug("Cloud API key detected, using premium tier")
        return {
            "tier": "premium",
            "customer_id": None,  # Will be resolved by cloud service
            "issued_at": None,
            "expires_at": None,
            "custom_packages": [],
            "features": [
                "local_only",
                "cloud_compute",
                "email_support",
                "priority_processing",
            ],
            "valid": True,
            "source": "cloud_key",
        }

    # Try to load from license file
    license_path = get_license_path()

    if not license_path.exists():
        logger.debug(f"No license file found at {license_path}, using free tier")
        return DEFAULT_ENTITLEMENT

    try:
        with open(license_path, "r") as f:
            data = json.load(f)

        # Validate required fields
        if "tier" not in data:
            logger.warning("License file missing 'tier' field, using free tier")
            return DEFAULT_ENTITLEMENT

        # Check expiration
        expires_at = data.get("expires_at")
        if expires_at:
            try:
                expiry_date = datetime.fromisoformat(expires_at.replace("Z", "+00:00"))
                if expiry_date < datetime.now(expiry_date.tzinfo):
                    logger.warning("License has expired, falling back to free tier")
                    return {
                        **DEFAULT_ENTITLEMENT,
                        "expired": True,
                        "expired_tier": data.get("tier"),
                        "source": "expired_license",
                    }
            except (ValueError, TypeError) as e:
                logger.warning(f"Could not parse expiry date: {e}")

        # Verify signature if present (placeholder for future cryptographic verification)
        if "signature" in data:
            if not _verify_signature(data):
                logger.warning("License signature verification failed, using free tier")
                return {
                    **DEFAULT_ENTITLEMENT,
                    "signature_invalid": True,
                    "source": "invalid_signature",
                }

        # Valid entitlement
        return {
            "tier": data.get("tier", "free").lower(),
            "customer_id": data.get("customer_id"),
            "issued_at": data.get("issued_at"),
            "expires_at": data.get("expires_at"),
            "custom_packages": data.get("custom_packages", []),
            "features": data.get("features", []),
            "valid": True,
            "source": "license_file",
        }

    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in license file: {e}")
        return {**DEFAULT_ENTITLEMENT, "parse_error": str(e)}
    except Exception as e:
        logger.error(f"Error reading license file: {e}")
        return {**DEFAULT_ENTITLEMENT, "read_error": str(e)}


def _verify_signature(data: Dict[str, Any]) -> bool:
    """
    Verify the cryptographic signature of an entitlement.

    Uses the JWKS endpoint to fetch the public key and verify
    that the entitlement was signed by the Omics-OS license service.

    Args:
        data: Entitlement data including signature and kid

    Returns:
        True if signature is valid
    """
    signature = data.get("signature")
    kid = data.get("kid")

    if not signature:
        logger.debug("No signature present - skipping verification")
        return True  # Allow unsigned for backward compatibility

    try:
        # Try to verify with cryptography library
        import base64
        import hashlib
        import json

        from cryptography.hazmat.backends import default_backend
        from cryptography.hazmat.primitives import hashes
        from cryptography.hazmat.primitives.asymmetric import padding

        # Fetch JWKS
        jwks_result = get_jwks()
        if not jwks_result.get("success"):
            logger.warning(f"Could not fetch JWKS: {jwks_result.get('error')}")
            # Fail open if offline (allow cached entitlement)
            return True

        jwks = jwks_result.get("jwks", {})
        keys = jwks.get("keys", [])

        # Find the matching key
        matching_key = None
        for key in keys:
            if kid and key.get("kid") == kid:
                matching_key = key
                break
            elif not kid:
                # Use first key if no kid specified
                matching_key = key
                break

        if not matching_key:
            logger.warning(f"No matching key found for kid={kid}")
            return False

        # Reconstruct the public key from JWKS
        n_b64 = matching_key.get("n")
        e_b64 = matching_key.get("e")

        if not n_b64 or not e_b64:
            logger.warning("JWKS key missing n or e parameters")
            return False

        # Decode base64url
        def base64url_decode(data: str) -> bytes:
            # Add padding if needed
            padding_needed = 4 - (len(data) % 4)
            if padding_needed != 4:
                data += "=" * padding_needed
            return base64.urlsafe_b64decode(data)

        n_bytes = base64url_decode(n_b64)
        e_bytes = base64url_decode(e_b64)

        n = int.from_bytes(n_bytes, byteorder="big")
        e = int.from_bytes(e_bytes, byteorder="big")

        from cryptography.hazmat.primitives.asymmetric.rsa import RSAPublicNumbers

        public_numbers = RSAPublicNumbers(e, n)
        public_key = public_numbers.public_key(default_backend())

        # Reconstruct the payload that was signed
        # The license service signs the entitlement payload (without signature/kid)
        payload_to_verify = {
            k: v
            for k, v in data.items()
            if k not in ("signature", "kid", "source", "valid")
        }
        canonical = json.dumps(payload_to_verify, sort_keys=True, separators=(",", ":"))
        message_bytes = canonical.encode("utf-8")
        message_digest = hashlib.sha256(message_bytes).digest()

        # Decode signature
        signature_bytes = base64.b64decode(signature)

        # Verify using PKCS1v15 with Prehashed (server signs a digest, not raw message)
        try:
            from cryptography.hazmat.primitives.asymmetric.utils import Prehashed

            public_key.verify(
                signature_bytes,
                message_digest,
                padding.PKCS1v15(),
                Prehashed(hashes.SHA256()),
            )
            logger.debug("Signature verification successful")
            return True
        except Exception as verify_error:
            logger.warning(f"Signature verification failed: {verify_error}")
            return False

    except ImportError:
        logger.debug(
            "cryptography library not available - skipping signature verification"
        )
        return True  # Fail open if crypto not available
    except Exception as e:
        logger.warning(f"Signature verification error: {e}")
        return True  # Fail open on unexpected errors (prefer availability)


def save_entitlement(entitlement: Dict[str, Any]) -> bool:
    """
    Save entitlement data to license file.

    This is called by the CLI during activation to persist
    the entitlement received from the license service.

    Args:
        entitlement: Entitlement data to save

    Returns:
        True if save was successful
    """
    license_path = get_license_path()

    try:
        # Ensure directory exists
        license_path.parent.mkdir(parents=True, exist_ok=True)

        # Write entitlement
        with open(license_path, "w") as f:
            json.dump(entitlement, f, indent=2, default=str)

        # Set restrictive permissions (owner read/write only)
        os.chmod(license_path, 0o600)

        logger.info(f"Saved entitlement to {license_path}")
        return True

    except Exception as e:
        logger.error(f"Failed to save entitlement: {e}")
        return False


def clear_entitlement() -> bool:
    """
    Remove the license file, reverting to free tier.

    Returns:
        True if file was removed or didn't exist
    """
    license_path = get_license_path()

    try:
        if license_path.exists():
            license_path.unlink()
            logger.info(f"Removed license file at {license_path}")
        return True
    except Exception as e:
        logger.error(f"Failed to remove license file: {e}")
        return False


# =============================================================================
# TIER AND FEATURE ACCESSORS
# =============================================================================


def get_current_tier() -> str:
    """
    Get the current subscription tier.

    Returns:
        Tier name: "free", "premium", or "enterprise"
    """
    entitlement = load_entitlement()
    return entitlement.get("tier", "free")


def get_custom_packages() -> List[str]:
    """
    Get list of authorized custom packages.

    Returns:
        List of package names (e.g., ["lobster-custom-databiomix"])
    """
    entitlement = load_entitlement()
    return entitlement.get("custom_packages", [])


def is_feature_enabled(feature: str) -> bool:
    """
    Check if a specific feature is enabled for current entitlement.

    Args:
        feature: Feature name to check (e.g., "cloud_compute")

    Returns:
        True if feature is enabled
    """
    entitlement = load_entitlement()
    features = entitlement.get("features", [])
    return feature in features


def is_premium() -> bool:
    """Check if current tier is premium or higher."""
    tier = get_current_tier()
    return tier in ("premium", "enterprise")


def is_enterprise() -> bool:
    """Check if current tier is enterprise."""
    return get_current_tier() == "enterprise"


def get_entitlement_status() -> Dict[str, Any]:
    """
    Get a summary of current entitlement status for display.

    Returns:
        Dict with tier, validity, expiry, and feature summary
    """
    entitlement = load_entitlement()

    status = {
        "tier": entitlement.get("tier", "free"),
        "tier_display": entitlement.get("tier", "free").title(),
        "valid": entitlement.get("valid", False),
        "source": entitlement.get("source", "unknown"),
    }

    # Add expiry info if present
    expires_at = entitlement.get("expires_at")
    if expires_at:
        try:
            expiry_date = datetime.fromisoformat(expires_at.replace("Z", "+00:00"))
            status["expires_at"] = expires_at
            status["days_until_expiry"] = (
                expiry_date - datetime.now(expiry_date.tzinfo)
            ).days
        except (ValueError, TypeError):
            status["expires_at"] = expires_at
            status["days_until_expiry"] = None

    # Add feature summary
    status["features"] = entitlement.get("features", [])
    status["custom_packages"] = entitlement.get("custom_packages", [])

    # Add any warnings
    warnings = []
    if entitlement.get("expired"):
        warnings.append(f"License expired (was {entitlement.get('expired_tier')})")
    if entitlement.get("signature_invalid"):
        warnings.append("License signature invalid")
    if entitlement.get("parse_error"):
        warnings.append(f"License file parse error: {entitlement.get('parse_error')}")

    if warnings:
        status["warnings"] = warnings

    return status


# =============================================================================
# ACTIVATION HELPERS (for CLI integration)
# =============================================================================


def _install_packages(packages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Install custom packages from presigned URLs.

    Uses `uv pip install` if available (faster), falls back to `pip install`.

    Args:
        packages: List of package dicts with {name, version, url, expires_at}

    Returns:
        List of installation results with {name, version, success, error}
    """
    import shutil
    import subprocess
    import sys

    results = []

    # Check if uv is available (preferred - much faster)
    uv_path = shutil.which("uv")
    use_uv = uv_path is not None

    for pkg in packages:
        name = pkg.get("name", "unknown")
        version = pkg.get("version", "unknown")
        url = pkg.get("url")

        if not url:
            results.append(
                {
                    "name": name,
                    "version": version,
                    "success": False,
                    "error": "No download URL provided",
                }
            )
            continue

        logger.info(f"Installing {name} v{version}...")

        try:
            if use_uv:
                # Use uv pip install (faster)
                cmd = [uv_path, "pip", "install", "--quiet", url]
            else:
                # Fall back to standard pip
                cmd = [sys.executable, "-m", "pip", "install", "--quiet", url]

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300,  # 5 minute timeout for large packages
            )

            if result.returncode == 0:
                logger.info(f"Successfully installed {name} v{version}")
                results.append(
                    {
                        "name": name,
                        "version": version,
                        "success": True,
                        "error": None,
                    }
                )
            else:
                error_msg = (
                    result.stderr.strip() or result.stdout.strip() or "Unknown error"
                )
                logger.error(f"Failed to install {name}: {error_msg}")
                results.append(
                    {
                        "name": name,
                        "version": version,
                        "success": False,
                        "error": error_msg[:200],  # Truncate long errors
                    }
                )

        except subprocess.TimeoutExpired:
            logger.error(f"Installation timeout for {name}")
            results.append(
                {
                    "name": name,
                    "version": version,
                    "success": False,
                    "error": "Installation timed out (>5 minutes)",
                }
            )
        except Exception as e:
            logger.error(f"Installation error for {name}: {e}")
            results.append(
                {
                    "name": name,
                    "version": version,
                    "success": False,
                    "error": str(e),
                }
            )

    return results


def activate_license(
    access_code: str, license_server_url: Optional[str] = None
) -> Dict[str, Any]:
    """
    Activate a license using an access code.

    This function contacts the license server to exchange an access
    code for an entitlement. The entitlement is then saved locally.

    Args:
        access_code: The activation code provided to the customer
        license_server_url: Optional override for license server URL

    Returns:
        Dict with activation result (success, entitlement, error)
    """
    # Default license server URL (can be overridden via env var or param)
    server_url = (
        license_server_url
        or os.environ.get("LOBSTER_LICENSE_SERVER_URL")
        or "https://x6gm9vfgl5.execute-api.us-east-1.amazonaws.com/v1"
    )

    try:
        import httpx

        # Call license server activation endpoint
        # Note: access_code is the LOBSTER_CLOUD_KEY (cloud_key)
        response = httpx.post(
            f"{server_url}/api/v1/activate",
            json={
                "cloud_key": access_code,
                "machine_fingerprint": _get_machine_fingerprint(),
            },
            timeout=30.0,
        )

        if response.status_code == 200:
            data = response.json()

            if not data.get("success"):
                return {
                    "success": False,
                    "error": data.get("message", "Activation failed"),
                }

            entitlement = data.get("entitlement", {})
            signature = data.get("signature")
            kid = data.get("kid")
            packages_to_install = data.get("packages_to_install", [])

            # Add signature and kid to entitlement for local storage
            if signature:
                entitlement["signature"] = signature
            if kid:
                entitlement["kid"] = kid

            # Save entitlement locally
            if not save_entitlement(entitlement):
                return {
                    "success": False,
                    "error": "Failed to save entitlement locally",
                }

            # Auto-install custom packages if any
            packages_installed = []
            packages_failed = []
            if packages_to_install:
                logger.info(
                    f"Installing {len(packages_to_install)} custom package(s)..."
                )
                install_results = _install_packages(packages_to_install)
                packages_installed = [r for r in install_results if r["success"]]
                packages_failed = [r for r in install_results if not r["success"]]

            return {
                "success": True,
                "entitlement": entitlement,
                "message": f"Successfully activated {entitlement.get('tier', 'premium')} license",
                "packages_installed": packages_installed,
                "packages_failed": packages_failed,
            }

        elif response.status_code == 401:
            return {
                "success": False,
                "error": "Invalid cloud key",
            }
        elif response.status_code == 403:
            return {
                "success": False,
                "error": "Cloud key already used or revoked",
            }
        else:
            error_detail = ""
            try:
                error_data = response.json()
                error_detail = error_data.get("message", "")
            except Exception:
                pass
            return {
                "success": False,
                "error": f"License server error: {response.status_code}"
                + (f" - {error_detail}" if error_detail else ""),
            }

    except ImportError:
        return {
            "success": False,
            "error": "httpx not installed - run 'pip install httpx' for license activation",
        }
    except Exception as e:
        logger.error(f"License activation failed: {e}")
        return {
            "success": False,
            "error": str(e),
        }


def _get_machine_fingerprint() -> str:
    """
    Generate a machine fingerprint for license binding.

    This creates a semi-stable identifier for the machine that
    can be used to bind licenses to specific installations.

    Returns:
        Machine fingerprint string
    """
    import hashlib
    import platform

    # Combine various system identifiers
    components = [
        platform.node(),  # Hostname
        platform.machine(),  # Architecture
        platform.system(),  # OS
    ]

    # Hash for privacy
    fingerprint = hashlib.sha256(":".join(components).encode()).hexdigest()[:32]
    return fingerprint


def refresh_entitlement(cloud_key: Optional[str] = None) -> Dict[str, Any]:
    """
    Refresh the current entitlement from the license server.

    This can be used to check for updates to the entitlement
    (e.g., tier upgrades, package additions) without re-activating.

    Args:
        cloud_key: The LOBSTER_CLOUD_KEY (required for refresh)

    Returns:
        Dict with refresh result
    """
    entitlement = load_entitlement()

    if entitlement.get("source") != "license_file":
        return {
            "success": False,
            "error": "No active license to refresh",
        }

    entitlement_id = entitlement.get("entitlement_id")
    if not entitlement_id:
        return {
            "success": False,
            "error": "No entitlement ID in current license",
        }

    # Get cloud key from parameter or environment
    cloud_key = cloud_key or os.environ.get(CLOUD_KEY_ENV_VAR)
    if not cloud_key:
        return {
            "success": False,
            "error": "Cloud key required for refresh (set LOBSTER_CLOUD_KEY)",
        }

    server_url = os.environ.get(
        "LOBSTER_LICENSE_SERVER_URL",
        "https://x6gm9vfgl5.execute-api.us-east-1.amazonaws.com/v1",
    )

    try:
        import httpx

        response = httpx.post(
            f"{server_url}/api/v1/refresh",
            json={
                "entitlement_id": entitlement_id,
                "cloud_key": cloud_key,
            },
            timeout=30.0,
        )

        if response.status_code == 200:
            data = response.json()

            if not data.get("success"):
                return {
                    "success": False,
                    "error": data.get("message", "Refresh failed"),
                }

            new_entitlement = data.get("entitlement", {})
            signature = data.get("signature")
            kid = data.get("kid")

            # Add signature and kid
            if signature:
                new_entitlement["signature"] = signature
            if kid:
                new_entitlement["kid"] = kid

            if save_entitlement(new_entitlement):
                return {
                    "success": True,
                    "entitlement": new_entitlement,
                    "message": "Entitlement refreshed successfully",
                }
            else:
                return {
                    "success": False,
                    "error": "Failed to save refreshed entitlement",
                }
        else:
            return {
                "success": False,
                "error": f"Refresh failed: {response.status_code}",
            }

    except ImportError:
        return {
            "success": False,
            "error": "httpx not installed",
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
        }


def check_revocation_status() -> Dict[str, Any]:
    """
    Check if the current entitlement has been revoked.

    This should be called periodically (e.g., every 24 hours) to ensure
    the local entitlement is still valid on the server.

    Returns:
        Dict with status check result
    """
    entitlement = load_entitlement()

    if entitlement.get("source") != "license_file":
        return {
            "success": True,
            "is_valid": True,
            "message": "No license file to check",
        }

    entitlement_id = entitlement.get("entitlement_id")
    if not entitlement_id:
        return {
            "success": True,
            "is_valid": True,
            "message": "No entitlement ID to check",
        }

    server_url = os.environ.get(
        "LOBSTER_LICENSE_SERVER_URL",
        "https://x6gm9vfgl5.execute-api.us-east-1.amazonaws.com/v1",
    )

    try:
        import httpx

        response = httpx.post(
            f"{server_url}/api/v1/status",
            json={"entitlement_id": entitlement_id},
            timeout=30.0,
        )

        if response.status_code == 200:
            data = response.json()
            is_valid = data.get("is_valid", False)

            if not is_valid:
                # Entitlement has been revoked or expired on server
                logger.warning(
                    f"Entitlement {entitlement_id} is no longer valid: {data.get('message')}"
                )

                # Clear local entitlement if revoked
                if data.get("status") == "revoked":
                    clear_entitlement()
                    return {
                        "success": True,
                        "is_valid": False,
                        "revoked": True,
                        "message": "License has been revoked",
                    }

            return {
                "success": True,
                "is_valid": is_valid,
                "status": data.get("status"),
                "tier": data.get("tier"),
                "expires_at": data.get("expires_at"),
                "message": data.get("message"),
            }
        else:
            return {
                "success": False,
                "error": f"Status check failed: {response.status_code}",
            }

    except ImportError:
        return {
            "success": False,
            "error": "httpx not installed",
        }
    except Exception as e:
        logger.debug(f"Status check failed (offline?): {e}")
        return {
            "success": False,
            "error": str(e),
        }


def get_jwks() -> Dict[str, Any]:
    """
    Fetch the JWKS (JSON Web Key Set) from the license server.

    This contains the public keys used to verify entitlement signatures.

    Returns:
        Dict with JWKS data or error
    """
    server_url = os.environ.get(
        "LOBSTER_LICENSE_SERVER_URL",
        "https://x6gm9vfgl5.execute-api.us-east-1.amazonaws.com/v1",
    )

    try:
        import httpx

        response = httpx.get(
            f"{server_url}/.well-known/jwks.json",
            timeout=30.0,
        )

        if response.status_code == 200:
            return {
                "success": True,
                "jwks": response.json(),
            }
        else:
            return {
                "success": False,
                "error": f"Failed to fetch JWKS: {response.status_code}",
            }

    except ImportError:
        return {
            "success": False,
            "error": "httpx not installed",
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
        }
