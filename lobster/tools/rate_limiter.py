"""
Redis-based rate limiter for NCBI API endpoints.

This module provides production-grade distributed rate limiting with graceful
degradation for NCBI E-utilities API calls. It ensures compliance with NCBI
rate limits (3 req/s without key, 10 req/s with key) across multiple users
and processes.
"""

import os
import time
from functools import wraps
from typing import Optional

import redis
from redis.exceptions import ConnectionError, RedisError

from lobster.utils.logger import get_logger

logger = get_logger(__name__)

# Global flag to ensure we only warn about Redis unavailability once
_REDIS_WARNING_SHOWN = False


def get_redis_client() -> Optional[redis.Redis]:
    """
    Get Redis client with health check and graceful degradation.

    Returns None if Redis is unavailable - allows system to continue
    operating (with warning) rather than failing completely.

    Returns:
        Redis client if available, None otherwise
    """
    global _REDIS_WARNING_SHOWN

    try:
        client = redis.Redis(
            host=os.getenv("REDIS_HOST", "localhost"),
            port=int(os.getenv("REDIS_PORT", "6379")),
            db=0,
            decode_responses=True,
            socket_connect_timeout=5,
            socket_timeout=5,
        )

        # Health check
        client.ping()
        logger.info("✓ Redis connection established for rate limiting")
        return client

    except ConnectionError as e:
        # Only show warning once during application startup
        if not _REDIS_WARNING_SHOWN:
            logger.warning(
                "⚠️  Redis unavailable - rate limiting disabled. "
                "For production use, start Redis with: docker-compose up -d redis"
            )
            _REDIS_WARNING_SHOWN = True
        return None
    except Exception as e:
        if not _REDIS_WARNING_SHOWN:
            logger.error(f"Unexpected error connecting to Redis: {e}")
            _REDIS_WARNING_SHOWN = True
        return None


class NCBIRateLimiter:
    """
    Rate limiter for NCBI API endpoints using Redis sliding window.

    Implements distributed rate limiting with automatic expiry and graceful
    degradation. If Redis is unavailable, the system logs a warning but
    continues operating (fail-open design).

    Attributes:
        redis_client: Redis client instance (None if unavailable)
        ncbi_api_key: NCBI API key from environment
        rate_limit: Requests per second limit (9 with key, 2 without)
        window_seconds: Time window for rate limiting (1 second)
    """

    def __init__(self, redis_client: Optional[redis.Redis] = None):
        """
        Initialize rate limiter with optional Redis client.

        Args:
            redis_client: Optional Redis client (creates new if not provided)
        """
        self.redis_client = (
            redis_client if redis_client is not None else get_redis_client()
        )

        # NCBI limits: 3 req/s without key, 10 req/s with key
        # Use conservative limits to avoid accidental bans
        self.ncbi_api_key = os.getenv("NCBI_API_KEY")
        self.rate_limit = 9 if self.ncbi_api_key else 2
        self.window_seconds = 1

    def check_rate_limit(self, api_name: str, user_id: str = "default") -> bool:
        """
        Check if request is within rate limit.

        Uses Redis sliding window algorithm with automatic expiry.
        If Redis is unavailable, returns True with warning (fail-open).

        Args:
            api_name: API endpoint name (e.g., "ncbi_esearch", "ncbi_efetch")
            user_id: User identifier for per-user rate limiting (default: "default")

        Returns:
            True if request should be allowed, False if rate limit exceeded
        """
        if self.redis_client is None:
            # Redis unavailable - allow request silently (warning already shown at startup)
            return True  # Fail open - risky but better than blocking all requests

        try:
            key = f"ratelimit:{api_name}:{user_id}"
            current = self.redis_client.get(key)

            if current is None:
                # First request in this 1-second window
                self.redis_client.setex(key, self.window_seconds, 1)
                return True

            current_count = int(current)
            if current_count >= self.rate_limit:
                logger.warning(
                    f"Rate limit exceeded for {api_name} by {user_id}: "
                    f"{current_count}/{self.rate_limit} requests"
                )
                return False

            # Increment counter
            self.redis_client.incr(key)
            return True

        except RedisError as e:
            # Redis error during operation - fail open with warning
            logger.error(f"Redis error during rate limiting: {e}")
            return True
        except Exception as e:
            # Unexpected error - fail open
            logger.error(f"Unexpected error in rate limiter: {e}")
            return True

    def wait_for_slot(
        self, api_name: str, user_id: str = "default", max_wait: float = 10.0
    ) -> bool:
        """
        Block until a rate limit slot is available.

        Polls the rate limiter every 100ms until a slot opens up or
        max_wait time is exceeded.

        Args:
            api_name: API endpoint name
            user_id: User identifier for per-user rate limiting
            max_wait: Maximum time to wait in seconds (default: 10.0)

        Returns:
            True if slot acquired, False if max_wait exceeded
        """
        start_time = time.time()
        wait_count = 0

        while not self.check_rate_limit(api_name, user_id):
            elapsed = time.time() - start_time
            if elapsed >= max_wait:
                logger.error(
                    f"Rate limit wait timeout for {api_name} after {elapsed:.1f}s"
                )
                return False

            wait_count += 1
            if wait_count % 10 == 0:  # Log every 1 second
                logger.debug(
                    f"Waiting for rate limit slot ({wait_count * 0.1:.1f}s elapsed)"
                )

            time.sleep(0.1)  # 100ms backoff

        return True

    def get_current_usage(self, api_name: str, user_id: str = "default") -> int:
        """
        Get current request count in the rate limit window.

        Args:
            api_name: API endpoint name
            user_id: User identifier

        Returns:
            Current request count (0 if Redis unavailable)
        """
        if self.redis_client is None:
            return 0

        try:
            key = f"ratelimit:{api_name}:{user_id}"
            current = self.redis_client.get(key)
            return int(current) if current else 0
        except Exception as e:
            logger.error(f"Error getting current usage: {e}")
            return 0

    def reset_limit(self, api_name: str, user_id: str = "default") -> bool:
        """
        Reset rate limit counter for testing purposes.

        Args:
            api_name: API endpoint name
            user_id: User identifier

        Returns:
            True if reset successful, False otherwise
        """
        if self.redis_client is None:
            return False

        try:
            key = f"ratelimit:{api_name}:{user_id}"
            self.redis_client.delete(key)
            logger.debug(f"Reset rate limit for {api_name}:{user_id}")
            return True
        except Exception as e:
            logger.error(f"Error resetting rate limit: {e}")
            return False


def rate_limited(api_name: str):
    """
    Decorator to enforce rate limiting on API calls.

    Automatically waits for a rate limit slot before executing the
    decorated function. Uses default user_id="default" for shared
    rate limiting across all users.

    Args:
        api_name: API endpoint name (e.g., "ncbi_esearch")

    Example:
        @rate_limited("ncbi_esearch")
        def search_pubmed(query: str):
            # API call here
            pass
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            rate_limiter = NCBIRateLimiter()
            if not rate_limiter.wait_for_slot(api_name):
                raise TimeoutError(
                    f"Rate limit wait timeout for {api_name}. "
                    "Too many concurrent requests."
                )
            return func(*args, **kwargs)

        return wrapper

    return decorator
