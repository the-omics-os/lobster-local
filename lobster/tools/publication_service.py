"""
Publication service orchestrator for modular literature and dataset discovery.

This service provides a unified interface to multiple publication providers
and implements comprehensive literature search and dataset discovery workflows.
"""

from typing import Any, Dict, List, Optional, Union
from enum import Enum

from pydantic import BaseModel, Field

from lobster.tools.providers.base_provider import (
    BasePublicationProvider,
    PublicationSource,
    DatasetType,
    PublicationMetadata,
    DatasetMetadata
)
from lobster.tools.providers.pubmed_provider import PubMedProvider, PubMedProviderConfig
from lobster.tools.providers.geo_provider import GEOProvider, GEOProviderConfig
from lobster.core.data_manager_v2 import DataManagerV2
from lobster.utils.logger import get_logger

logger = get_logger(__name__)

class OmicsDataType(Enum):
    """Enum for different omics data types - modular design for extensibility."""
    GEO = "geo"  # Gene Expression Omnibus
    SRA = "sra"  # Sequence Read Archive
    DBGAP = "dbgap"  # Database of Genotypes and Phenotypes
    BIOPROJECT = "bioproject"  # BioProject
    BIOSAMPLE = "biosample"  # BioSample
    PROTEOMICS = "proteomics"  # ProteomicsDB
    METABOLOMICS = "metabolomics"  # Metabolomics Workbench

class PublicationServiceConfig(BaseModel):
    """Configuration for the publication service."""
    
    # Default provider settings
    default_provider: PublicationSource = PublicationSource.PUBMED
    
    # Multi-provider search settings
    enable_multi_provider_search: bool = False
    max_results_per_provider: int = 3
    
    # Provider-specific configurations
    pubmed_config: PubMedProviderConfig = PubMedProviderConfig()
    geo_config: GEOProviderConfig = GEOProviderConfig()
    
    # Service behavior
    auto_register_providers: bool = True
    prefer_primary_provider: bool = True
    fallback_enabled: bool = True


class ProviderRegistry:
    """Registry for managing publication providers."""
    
    def __init__(self):
        self.providers: Dict[PublicationSource, BasePublicationProvider] = {}
        self.default_provider: Optional[PublicationSource] = None
    
    def register_provider(
        self, 
        provider: BasePublicationProvider,
        set_as_default: bool = False
    ) -> None:
        """
        Register a publication provider.
        
        Args:
            provider: Provider instance to register
            set_as_default: Whether to set as default provider
        """
        source = provider.source
        self.providers[source] = provider
        
        if set_as_default or self.default_provider is None:
            self.default_provider = source
        
        logger.info(f"Registered {source.value} provider")
    
    def get_provider(self, source: PublicationSource) -> Optional[BasePublicationProvider]:
        """Get provider by source."""
        return self.providers.get(source)
    
    def get_default_provider(self) -> Optional[BasePublicationProvider]:
        """Get the default provider."""
        if self.default_provider:
            return self.providers.get(self.default_provider)
        return None
    
    def list_providers(self) -> List[PublicationSource]:
        """List all registered providers."""
        return list(self.providers.keys())
    
    def get_providers_for_dataset_type(
        self, 
        dataset_type: DatasetType
    ) -> List[BasePublicationProvider]:
        """Get providers that support a specific dataset type."""
        supporting_providers = []
        for provider in self.providers.values():
            if dataset_type in provider.supported_dataset_types:
                supporting_providers.append(provider)
        return supporting_providers


class PublicationService:
    """
    Main orchestrator for publication search and dataset discovery.
    
    This service provides a unified interface to multiple publication providers
    and implements intelligent routing, fallback mechanisms, and result aggregation.
    """
    
    def __init__(
        self,
        data_manager: DataManagerV2,
        config: Optional[PublicationServiceConfig] = None,
        ncbi_api_key: Optional[str] = None
    ):
        """
        Initialize publication service.
        
        Args:
            data_manager: DataManagerV2 instance for provenance tracking
            config: Optional configuration, uses defaults if not provided
            ncbi_api_key: NCBI API key for PubMed provider
        """
        self.data_manager = data_manager
        self.config = config or PublicationServiceConfig()
        self.ncbi_api_key = ncbi_api_key
        self.registry = ProviderRegistry()
        
        # Auto-register providers if enabled
        if self.config.auto_register_providers:
            self._register_default_providers()
    
    def _register_default_providers(self) -> None:
        """Register default providers based on configuration."""
        try:
            # Create PubMed config with API key if available
            pubmed_config = self.config.pubmed_config
            if self.ncbi_api_key:
                # Create a new config with the API key
                pubmed_config = PubMedProviderConfig(
                    **pubmed_config.model_dump(),
                    api_key=self.ncbi_api_key
                )
            
            # Register PubMed provider
            pubmed_provider = PubMedProvider(
                data_manager=self.data_manager,
                config=pubmed_config
            )
            self.registry.register_provider(
                pubmed_provider,
                set_as_default=(self.config.default_provider == PublicationSource.PUBMED)
            )
        except Exception as e:
            logger.error(f"Failed to register PubMed provider: {e}")
        
        try:
            # Register GEO provider
            geo_provider = GEOProvider(
                data_manager=self.data_manager,
                config=self.config.geo_config
            )
            self.registry.register_provider(
                geo_provider,
                set_as_default=(self.config.default_provider == PublicationSource.GEO)
            )
        except Exception as e:
            logger.error(f"Failed to register GEO provider: {e}")
        
        # Future: Add other providers (bioRxiv, medRxiv) when implemented
        logger.info(f"Registered {len(self.registry.list_providers())} providers")
    
    def search_literature(
        self,
        query: str,
        max_results: int = 5,
        sources: Optional[List[PublicationSource]] = None,
        filters: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> str:
        """
        Search for literature across specified sources.
        
        Args:
            query: Search query string
            max_results: Maximum number of results to return
            sources: Specific sources to search (default: use default provider)
            filters: Optional search filters
            **kwargs: Additional parameters
            
        Returns:
            str: Formatted search results
        """
        logger.info(f"Literature search: {query[:50]}... across {len(sources or [self.config.default_provider])} sources")
        
        try:
            # Determine which providers to use
            if sources:
                providers = [self.registry.get_provider(src) for src in sources]
                providers = [p for p in providers if p is not None]
            else:
                default_provider = self.registry.get_default_provider()
                providers = [default_provider] if default_provider else []
            
            if not providers:
                return "No available providers for literature search."
            
            # Multi-provider search or single provider
            if len(providers) > 1 and self.config.enable_multi_provider_search:
                return self._search_across_providers(query, providers, max_results, filters, **kwargs)
            else:
                # Use single provider (first available)
                provider = providers[0]
                results = provider.search_publications(
                    query=query,
                    max_results=max_results,
                    filters=filters,
                    **kwargs
                )
                
                # Log the search
                self.data_manager.log_tool_usage(
                    tool_name="search_literature",
                    parameters={
                        "query": query[:100],
                        "max_results": max_results,
                        "provider": provider.source.value,
                        "filters": filters
                    },
                    description="Literature search via publication service"
                )
                
                return results
        
        except Exception as e:
            logger.error(f"Literature search error: {e}")
            return f"Literature search error: {str(e)}"
    
    def find_datasets_from_publication(
        self,
        identifier: str,
        dataset_types: Optional[List[DatasetType]] = None,
        source: Optional[PublicationSource] = None,
        **kwargs
    ) -> str:
        """
        Find datasets associated with a publication.
        
        Args:
            identifier: Publication identifier (DOI, PMID, etc.)
            dataset_types: Types of datasets to search for
            source: Specific source to use (default: auto-detect)
            **kwargs: Additional parameters
            
        Returns:
            str: Formatted dataset discovery results
        """
        logger.info(f"Finding datasets from publication: {identifier}")
        
        try:
            # Determine which provider to use
            if source:
                provider = self.registry.get_provider(source)
                if not provider:
                    return f"Provider {source.value} not available."
            else:
                # Auto-detect based on identifier format
                provider = self._select_provider_for_identifier(identifier)
                if not provider:
                    return f"No suitable provider found for identifier: {identifier}"
            
            # Validate identifier with provider
            if not provider.validate_identifier(identifier):
                return f"Invalid identifier format for {provider.source.value}: {identifier}"
            
            # Search for datasets
            results = provider.find_datasets_from_publication(
                identifier=identifier,
                dataset_types=dataset_types,
                **kwargs
            )
            
            # Log the search
            self.data_manager.log_tool_usage(
                tool_name="find_datasets_from_publication",
                parameters={
                    "identifier": identifier,
                    "dataset_types": [dt.value for dt in dataset_types] if dataset_types else None,
                    "provider": provider.source.value
                },
                description="Dataset discovery from publication"
            )
            
            return results
            
        except Exception as e:
            logger.error(f"Dataset discovery error: {e}")
            return f"Dataset discovery error: {str(e)}"
    
    def extract_publication_metadata(
        self,
        identifier: str,
        source: Optional[PublicationSource] = None,
        **kwargs
    ) -> Union[PublicationMetadata, str]:
        """
        Extract standardized metadata from a publication.
        
        Args:
            identifier: Publication identifier
            source: Specific source to use (default: auto-detect)
            **kwargs: Additional parameters
            
        Returns:
            PublicationMetadata or error string
        """
        try:
            # Determine provider
            if source:
                provider = self.registry.get_provider(source)
                if not provider:
                    return f"Provider {source.value} not available."
            else:
                provider = self._select_provider_for_identifier(identifier)
                if not provider:
                    return f"No suitable provider found for identifier: {identifier}"
            
            # Extract metadata
            metadata = provider.extract_publication_metadata(identifier, **kwargs)
            
            # Log the extraction
            self.data_manager.log_tool_usage(
                tool_name="extract_publication_metadata",
                parameters={
                    "identifier": identifier,
                    "provider": provider.source.value
                },
                description="Publication metadata extraction"
            )
            
            return metadata
            
        except Exception as e:
            logger.error(f"Metadata extraction error: {e}")
            return f"Metadata extraction error: {str(e)}"
    
    def search_datasets_directly(
        self,
        query: str,
        data_type: DatasetType,
        max_results: int = 5,
        filters: Optional[Dict[str, str]] = None
    ) -> str:
        """
        Search for datasets directly across omics databases.
        
        Args:
            query: Search query
            data_type: Type of omics data to search for
            max_results: Maximum results to return
            filters: Additional filters
            
        Returns:
            str: Formatted dataset search results
        """
        logger.info(f"Direct dataset search: {query[:50]}... for {data_type.value}")
        
        try:
            # Find providers that support this dataset type
            supporting_providers = self.registry.get_providers_for_dataset_type(data_type)
            
            if not supporting_providers:
                return f"No providers available for {data_type.value} dataset search."
            
            # Use the first supporting provider (could be enhanced to use multiple)
            provider = supporting_providers[0]
            
            # Route to appropriate provider method
            if provider.source == PublicationSource.GEO:
                # GEO provider uses search_publications method for direct dataset search
                results = provider.search_publications(
                    query=query,
                    max_results=max_results,
                    filters=filters
                )
            elif hasattr(provider, 'find_datasets_for_study'):
                # Legacy method from PubMedService - convert data_type
                results = provider.find_datasets_for_study(
                    query=query,
                    data_type=data_type,
                    filters=filters,
                    top_k=max_results
                )
            else:
                return f"Direct dataset search not supported by {provider.source.value} provider."
            
            # Log the search
            self.data_manager.log_tool_usage(
                tool_name="search_datasets_directly",
                parameters={
                    "query": query[:100],
                    "data_type": data_type.value,
                    "max_results": max_results,
                    "provider": provider.source.value
                },
                description="Direct dataset search"
            )
            
            return results
                
        except Exception as e:
            logger.error(f"Direct dataset search error: {e}")
            return f"Direct dataset search error: {str(e)}"
    
    def get_provider_capabilities(self) -> str:
        """
        Get information about available providers and their capabilities.
        
        Returns:
            str: Formatted provider capabilities report
        """
        if not self.registry.providers:
            return "No providers registered."
        
        report = "## Available Publication Providers\n\n"
        
        for source, provider in self.registry.providers.items():
            report += f"### {source.value.title()} Provider\n"
            
            # Basic info
            report += f"**Source**: {source.value}\n"
            report += f"**Supported Dataset Types**: {', '.join([dt.value for dt in provider.supported_dataset_types])}\n"
            
            # Features
            features = provider.get_supported_features()
            report += "**Features**:\n"
            for feature, supported in features.items():
                status = "✅" if supported else "❌"
                report += f"  - {feature.replace('_', ' ').title()}: {status}\n"
            
            report += "\n"
        
        # Default provider info
        if self.registry.default_provider:
            report += f"**Default Provider**: {self.registry.default_provider.value}\n\n"
        
        # Configuration info
        report += "### Service Configuration\n"
        report += f"- Multi-provider search: {'Enabled' if self.config.enable_multi_provider_search else 'Disabled'}\n"
        report += f"- Fallback enabled: {'Yes' if self.config.fallback_enabled else 'No'}\n"
        report += f"- Max results per provider: {self.config.max_results_per_provider}\n"
        
        return report
    
    # Helper methods
    
    def _search_across_providers(
        self,
        query: str,
        providers: List[BasePublicationProvider],
        max_results: int,
        filters: Optional[Dict[str, Any]],
        **kwargs
    ) -> str:
        """Search across multiple providers and aggregate results."""
        results_per_provider = max(1, max_results // len(providers))
        all_results = []
        
        for provider in providers:
            try:
                provider_results = provider.search_publications(
                    query=query,
                    max_results=results_per_provider,
                    filters=filters,
                    **kwargs
                )
                all_results.append(f"## Results from {provider.source.value.title()}\n{provider_results}")
            except Exception as e:
                logger.warning(f"Error searching {provider.source.value}: {e}")
                all_results.append(f"## {provider.source.value.title()} - Error\n{str(e)}")
        
        # Combine results
        combined = "\n\n---\n\n".join(all_results)
        header = f"# Multi-Provider Literature Search Results\n**Query**: {query}\n\n"
        
        return header + combined
    
    def _select_provider_for_identifier(self, identifier: str) -> Optional[BasePublicationProvider]:
        """Auto-select the best provider for a given identifier."""
        # Try each provider's validation method
        for provider in self.registry.providers.values():
            if provider.validate_identifier(identifier):
                return provider
        
        # Fallback to default provider
        return self.registry.get_default_provider()
    
    def _convert_to_legacy_data_type(self, data_type: DatasetType):
        """Convert new DatasetType to legacy format for backward compatibility."""
        # This is a helper for the legacy find_datasets_for_study method        
        mapping = {
            DatasetType.GEO: OmicsDataType.GEO,
            DatasetType.SRA: OmicsDataType.SRA,
            DatasetType.BIOPROJECT: OmicsDataType.BIOPROJECT,
            DatasetType.BIOSAMPLE: OmicsDataType.BIOSAMPLE,
            DatasetType.DBGAP: OmicsDataType.DBGAP
        }
        
        return mapping.get(data_type, OmicsDataType.GEO)
