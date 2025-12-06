"""Utilities for normalizing publisher URLs for content extraction."""

from __future__ import annotations

import re
import urllib.parse
from typing import Dict

from lobster.utils.logger import get_logger

logger = get_logger(__name__)


PUBLISHER_URL_TRANSFORMS: Dict[str, Dict[str, str]] = {
    "link.springer.com": {
        "pattern": r"link\.springer\.com/content/pdf/(10\.\d+/[^/]+)\.pdf",
        "replacement": r"link.springer.com/article/\1",
        "description": "Springer PDF URL → HTML article page",
    },
    "api.wiley.com": {
        "pattern": r"api\.wiley\.com/onlinelibrary/tdm/v1/articles/(10\.\d+/[^?]+)",
        "replacement": r"onlinelibrary.wiley.com/doi/full/\1",
        "description": "Wiley TDM API → public DOI page",
    },
    "jstage.jst.go.jp": {
        "pattern": r"(jstage\.jst\.go\.jp/article/[^/]+/[^/]+/[^/]+/[^/_]+)/_article",
        "replacement": r"\1/_html/-char/en",
        "description": "J-STAGE article page → HTML full-text (English)",
    },
    # Add additional publisher transforms here as we discover them.
}


def transform_publisher_url(url: str) -> str:
    """Apply known publisher-specific URL transformations."""
    if not url:
        return url

    for domain, config in PUBLISHER_URL_TRANSFORMS.items():
        if domain in url:
            transformed = re.sub(config["pattern"], config["replacement"], url)
            transformed = urllib.parse.unquote(transformed)
            if transformed != url:
                logger.info(
                    "Transformed %s URL (%s): %s → %s",
                    domain,
                    config.get("description", "publisher transform"),
                    url,
                    transformed,
                )
                return transformed

    return url
