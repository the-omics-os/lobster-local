"""
Base Pydantic model for provider configuration with shared validation.

Uses abstract properties to define contracts that subclasses must fulfill,
avoiding brittle string-based programming patterns.
"""

import abc
import json
import logging
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, model_validator

from lobster.config.constants import VALID_PROVIDERS, VALID_PROFILES

logger = logging.getLogger(__name__)


class ProviderConfigBase(BaseModel, abc.ABC):
    """
    Abstract base class for provider configurations.

    Subclasses must implement:
    - provider_field_name: The field containing provider choice
    - model_field_suffix: Suffix for model fields ('_model' or '_default_model')
    """

    @property
    @abc.abstractmethod
    def provider_field_name(self) -> str:
        """Field name for provider (e.g., 'global_provider' or 'default_provider')."""
        pass

    @property
    @abc.abstractmethod
    def model_field_suffix(self) -> str:
        """Suffix for model fields (e.g., '_model' or '_default_model')."""
        pass

    @model_validator(mode="before")
    @classmethod
    def validate_providers_and_profiles(cls, data):
        """Validate provider and profile fields before model creation."""
        # Validate provider fields
        for field in ["global_provider", "default_provider"]:
            if field in data and data[field]:
                if data[field] not in VALID_PROVIDERS:
                    raise ValueError(
                        f"Invalid provider: '{data[field]}'. "
                        f"Must be one of: {', '.join(VALID_PROVIDERS)}"
                    )

        # Validate profile fields
        for field in ["profile", "default_profile"]:
            if field in data and data[field]:
                if data[field] not in VALID_PROFILES:
                    raise ValueError(
                        f"Invalid profile: '{data[field]}'. "
                        f"Must be one of: {', '.join(VALID_PROFILES)}"
                    )

        # Validate per-agent providers (workspace config only)
        if "per_agent_providers" in data:
            for agent, provider in data["per_agent_providers"].items():
                if provider not in VALID_PROVIDERS:
                    raise ValueError(
                        f"Invalid provider '{provider}' for agent '{agent}'. "
                        f"Must be one of: {', '.join(VALID_PROVIDERS)}"
                    )

        return data

    def get_model_for_provider(self, provider: str) -> Optional[str]:
        """Get the configured model for a specific provider."""
        if provider not in VALID_PROVIDERS:
            return None
        field_name = f"{provider}{self.model_field_suffix}"
        return getattr(self, field_name, None)

    def set_model_for_provider(self, provider: str, model: str) -> None:
        """Set the model for a specific provider."""
        if provider not in VALID_PROVIDERS:
            raise ValueError(f"Invalid provider: '{provider}'")
        field_name = f"{provider}{self.model_field_suffix}"
        if not hasattr(self, field_name):
            raise AttributeError(f"Config has no field for provider '{provider}'")
        setattr(self, field_name, model)
