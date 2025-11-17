"""
Provider Registry for capability-based routing.

This module provides centralized provider registration and capability-based
routing for the research agent refactoring (Phase 1). It enables dynamic
provider selection based on required capabilities and priority.
"""

from typing import Any, Dict, List, Optional, Tuple

from lobster.tools.providers.base_provider import DatasetType
from lobster.utils.logger import get_logger

logger = get_logger(__name__)


class ProviderRegistryError(Exception):
    """Base exception for ProviderRegistry errors."""

    pass


class ProviderNotFoundError(ProviderRegistryError):
    """No provider found for requested capability or dataset type."""

    pass


class ProviderRegistry:
    """
    Centralized registry for publication providers with capability-based routing.

    The registry maintains mappings between capabilities and providers,
    enabling dynamic provider selection based on required operations.
    Providers are sorted by priority (lower value = higher priority).

    Examples:
        >>> registry = ProviderRegistry()
        >>> registry.register_provider(pubmed_provider)
        >>> registry.register_provider(geo_provider)
        >>>
        >>> # Get providers for a capability
        >>> providers = registry.get_providers_for_capability(
        ...     ProviderCapability.SEARCH_LITERATURE
        ... )
        >>> print(f"Found {len(providers)} providers")
        >>>
        >>> # Get provider for dataset type
        >>> geo = registry.get_provider_for_dataset_type(DatasetType.GEO)
        >>> print(f"GEO provider: {geo.source}")
        >>>
        >>> # View capability matrix
        >>> matrix = registry.get_capability_matrix()
        >>> print(matrix)
    """

    def __init__(self):
        """
        Initialize empty provider registry.

        Sets up internal data structures for provider storage and
        capability mappings.
        """
        # Core provider storage
        self._providers: List[Any] = []  # List of registered providers

        # Capability-based routing maps
        # capability_name -> [(priority, provider), ...]
        self._capability_map: Dict[str, List[Tuple[int, Any]]] = {}

        # Dataset type routing map
        # dataset_type_value -> [provider, ...]
        self._dataset_type_map: Dict[str, List[Any]] = {}

        logger.debug("Initialized ProviderRegistry")

    def register_provider(self, provider: Any) -> None:
        """
        Register a provider and build capability mappings.

        This method:
        1. Checks for duplicate registration (warns and skips)
        2. Adds provider to internal registry
        3. Builds capability → provider mappings
        4. Builds dataset_type → provider mappings
        5. Sorts providers by priority for each capability

        Args:
            provider: Provider instance with get_supported_capabilities(),
                     priority, and supported_dataset_types properties

        Raises:
            ValueError: If provider missing required methods/properties
        """
        # Validate provider interface
        if not hasattr(provider, "get_supported_capabilities"):
            raise ValueError(
                f"Provider {type(provider).__name__} missing "
                "get_supported_capabilities() method"
            )
        if not hasattr(provider, "priority"):
            raise ValueError(
                f"Provider {type(provider).__name__} missing priority property"
            )
        if not hasattr(provider, "supported_dataset_types"):
            raise ValueError(
                f"Provider {type(provider).__name__} missing "
                "supported_dataset_types property"
            )

        # Check for duplicate registration
        provider_name = type(provider).__name__
        if any(type(p).__name__ == provider_name for p in self._providers):
            logger.warning(
                f"Provider {provider_name} already registered, skipping duplicate"
            )
            return

        # Add to registry
        self._providers.append(provider)
        logger.debug(
            f"Registered provider: {provider_name} " f"(priority {provider.priority})"
        )

        # Build capability mappings
        capabilities = provider.get_supported_capabilities()
        for capability, supported in capabilities.items():
            if supported:
                if capability not in self._capability_map:
                    self._capability_map[capability] = []

                self._capability_map[capability].append((provider.priority, provider))

        # Build dataset type mappings
        dataset_types = provider.supported_dataset_types
        for dataset_type in dataset_types:
            # Handle both DatasetType enum and string values
            if isinstance(dataset_type, DatasetType):
                key = dataset_type.value
            else:
                key = str(dataset_type)

            if key not in self._dataset_type_map:
                self._dataset_type_map[key] = []

            self._dataset_type_map[key].append(provider)

        # Sort capability providers by priority (lower = higher priority)
        for capability in self._capability_map:
            self._capability_map[capability].sort(key=lambda x: x[0])

        logger.debug(
            f"Provider {provider_name} registered with "
            f"{sum(capabilities.values())} capabilities"
        )

    def get_providers_for_capability(self, capability: str) -> List[Any]:
        """
        Get all providers supporting a capability, sorted by priority.

        Returns providers sorted by priority (lower value = higher priority).
        Ties are broken by registration order.

        Args:
            capability: Capability identifier (use ProviderCapability constants)

        Returns:
            List of providers supporting the capability, sorted by priority.
            Returns empty list if no providers support the capability.

        Examples:
            >>> providers = registry.get_providers_for_capability(
            ...     ProviderCapability.GET_FULL_CONTENT
            ... )
            >>> # Returns: [PMCProvider (priority 10), WebpageProvider (priority 50)]
        """
        if capability not in self._capability_map:
            logger.debug(f"No providers found for capability: {capability}")
            return []

        # Extract providers from (priority, provider) tuples
        providers = [provider for _, provider in self._capability_map[capability]]

        logger.debug(f"Found {len(providers)} providers for capability '{capability}'")
        return providers

    def get_provider_for_dataset_type(self, dataset_type: DatasetType) -> Optional[Any]:
        """
        Get provider for specific dataset type.

        Returns the first registered provider supporting the dataset type.
        For GEO datasets, this returns GEOProvider.

        Args:
            dataset_type: Dataset type enum value

        Returns:
            Provider instance if found, None otherwise

        Examples:
            >>> geo_provider = registry.get_provider_for_dataset_type(
            ...     DatasetType.GEO
            ... )
            >>> print(geo_provider.source)  # "geo"
        """
        # Handle both DatasetType enum and string values
        if isinstance(dataset_type, DatasetType):
            key = dataset_type.value
        else:
            key = str(dataset_type)

        providers = self._dataset_type_map.get(key, [])

        if not providers:
            logger.debug(f"No provider found for dataset type: {dataset_type}")
            return None

        logger.debug(f"Found provider for dataset type '{dataset_type}'")
        return providers[0]  # Return first (typically only one per type)

    def get_all_providers(self) -> List[Any]:
        """
        Get all registered providers.

        Returns:
            List of all registered provider instances
        """
        return self._providers.copy()

    def get_capability_matrix(self) -> str:
        """
        Generate capability matrix for debugging and documentation.

        Creates a formatted table showing which providers support which
        capabilities, along with priority values. Useful for debugging
        provider registration and understanding the capability landscape.

        Returns:
            str: Formatted capability matrix table

        Example output:
            ```
            Provider Capability Matrix
            ==========================

            Capability                  | PubMed (10) | GEO (10) | PMC (10) | Webpage (50)
            ----------------------------|-------------|----------|----------|-------------
            SEARCH_LITERATURE           | ✓           | -        | -        | -
            DISCOVER_DATASETS           | -           | ✓        | -        | -
            GET_FULL_CONTENT            | -           | -        | ✓        | ✓
            ...
            ```
        """
        if not self._providers:
            return "Provider Capability Matrix\n==========================\n\nNo providers registered."

        # Get all unique capabilities from all providers
        all_capabilities = set()
        for provider in self._providers:
            all_capabilities.update(provider.get_supported_capabilities().keys())

        # Sort capabilities for consistent output
        sorted_capabilities = sorted(all_capabilities)

        # Build header
        provider_names = [f"{type(p).__name__} ({p.priority})" for p in self._providers]
        header = (
            "Capability".ljust(30)
            + " | "
            + " | ".join(name.ljust(12) for name in provider_names)
        )
        separator = "-" * len(header)

        # Build rows
        rows = []
        for capability in sorted_capabilities:
            row_parts = [capability.ljust(30)]

            for provider in self._providers:
                capabilities = provider.get_supported_capabilities()
                supported = capabilities.get(capability, False)
                symbol = "✓" if supported else "-"
                row_parts.append(symbol.ljust(12))

            rows.append(" | ".join(row_parts))

        # Combine into final matrix
        matrix = "Provider Capability Matrix\n"
        matrix += "==========================\n\n"
        matrix += header + "\n"
        matrix += separator + "\n"
        matrix += "\n".join(rows) + "\n"

        return matrix

    def get_supported_dataset_types(self) -> List[DatasetType]:
        """
        Get all supported dataset types across all registered providers.

        Returns:
            List of DatasetType enum values supported by at least one provider
        """
        dataset_types = set()

        for provider in self._providers:
            for dt in provider.supported_dataset_types:
                if isinstance(dt, DatasetType):
                    dataset_types.add(dt)
                else:
                    # Try to convert string to DatasetType enum
                    try:
                        dataset_types.add(DatasetType(dt))
                    except ValueError:
                        logger.warning(
                            f"Unknown dataset type '{dt}' from "
                            f"{type(provider).__name__}"
                        )

        return sorted(list(dataset_types), key=lambda x: x.value)
