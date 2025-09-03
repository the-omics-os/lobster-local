"""
Abstract base class for publication providers.

This module defines the interface that all publication providers must implement
to ensure consistent behavior across different publication sources.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from enum import Enum

from pydantic import BaseModel


class PublicationSource(Enum):
    """Enum for different publication sources."""
    PUBMED = "pubmed"
    BIORXIV = "biorxiv"
    MEDRXIV = "medrxiv"
    ARXIV = "arxiv"
    GEO = "geo"


class DatasetType(Enum):
    """Enum for different dataset types."""
    GEO = "geo"
    SRA = "sra"
    DBGAP = "dbgap"
    BIOPROJECT = "bioproject"
    BIOSAMPLE = "biosample"
    ARRAYEXPRESS = "arrayexpress"
    ENA = "ena"


class PublicationMetadata(BaseModel):
    """Standard publication metadata structure."""
    uid: str
    title: str
    journal: Optional[str] = None
    published: Optional[str] = None
    doi: Optional[str] = None
    pmid: Optional[str] = None
    abstract: Optional[str] = None
    authors: List[str] = []
    keywords: List[str] = []
    
    
class DatasetMetadata(BaseModel):
    """Standard dataset metadata structure."""
    accession: str
    title: str
    description: Optional[str] = None
    organism: Optional[str] = None
    platform: Optional[str] = None
    samples_count: Optional[int] = None
    date: Optional[str] = None
    data_type: Optional[DatasetType] = None
    source_url: Optional[str] = None


class BasePublicationProvider(ABC):
    """
    Abstract base class for publication providers.
    
    All publication providers must implement this interface to ensure
    consistent behavior and interoperability within the system.
    """
    
    @property
    @abstractmethod
    def source(self) -> PublicationSource:
        """Return the publication source this provider handles."""
        pass
    
    @property
    @abstractmethod
    def supported_dataset_types(self) -> List[DatasetType]:
        """Return list of dataset types this provider can discover."""
        pass
    
    @abstractmethod
    def search_publications(
        self,
        query: str,
        max_results: int = 5,
        filters: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> str:
        """
        Search for publications using the provider's specific implementation.
        
        Args:
            query: Search query string
            max_results: Maximum number of results to return
            filters: Optional filters to apply to the search
            **kwargs: Provider-specific additional parameters
            
        Returns:
            str: Formatted search results
        """
        pass
    
    @abstractmethod
    def find_datasets_from_publication(
        self, 
        identifier: str,
        dataset_types: Optional[List[DatasetType]] = None,
        **kwargs
    ) -> str:
        """
        Find datasets associated with a publication.
        
        Args:
            identifier: Publication identifier (DOI, PMID, etc.)
            dataset_types: Types of datasets to search for
            **kwargs: Provider-specific additional parameters
            
        Returns:
            str: Formatted list of discovered datasets
        """
        pass
    
    @abstractmethod
    def extract_publication_metadata(
        self, 
        identifier: str,
        **kwargs
    ) -> PublicationMetadata:
        """
        Extract standardized metadata from a publication.
        
        Args:
            identifier: Publication identifier (DOI, PMID, etc.)
            **kwargs: Provider-specific additional parameters
            
        Returns:
            PublicationMetadata: Standardized publication metadata
        """
        pass
    
    def validate_identifier(self, identifier: str) -> bool:
        """
        Validate that an identifier is supported by this provider.
        
        Args:
            identifier: Publication identifier to validate
            
        Returns:
            bool: True if identifier is valid for this provider
        """
        # Default implementation - can be overridden by providers
        return len(identifier.strip()) > 0
    
    def get_supported_features(self) -> Dict[str, bool]:
        """
        Return a dictionary of features supported by this provider.
        
        Returns:
            Dict[str, bool]: Feature support mapping
        """
        return {
            "literature_search": True,
            "dataset_discovery": len(self.supported_dataset_types) > 0,
            "metadata_extraction": True,
            "full_text_access": False,
            "advanced_filtering": False
        }
    
    def format_search_results(
        self, 
        results: List[PublicationMetadata], 
        query: str
    ) -> str:
        """
        Format search results in a standardized way.
        
        Args:
            results: List of publication metadata
            query: Original search query
            
        Returns:
            str: Formatted search results
        """
        if not results:
            return f"No publications found for query: {query}"
        
        formatted = f"## {self.source.value.title()} Search Results\n\n"
        formatted += f"**Query**: {query}\n"
        formatted += f"**Results**: {len(results)} publications found\n\n"
        
        for i, pub in enumerate(results, 1):
            formatted += f"### Result {i}/{len(results)}\n"
            formatted += f"**Title**: {pub.title}\n"
            
            if pub.pmid:
                formatted += f"**PMID**: [{pub.pmid}](https://pubmed.ncbi.nlm.nih.gov/{pub.pmid}/)\n"
            if pub.doi:
                formatted += f"**DOI**: {pub.doi}\n"
            if pub.journal:
                formatted += f"**Journal**: {pub.journal}\n"
            if pub.published:
                formatted += f"**Published**: {pub.published}\n"
            if pub.authors:
                formatted += f"**Authors**: {', '.join(pub.authors[:3])}{'...' if len(pub.authors) > 3 else ''}\n"
            
            if pub.abstract:
                abstract_preview = pub.abstract[:500]
                if len(pub.abstract) > 500:
                    abstract_preview += "..."
                formatted += f"**Abstract**: {abstract_preview}\n"
            
            formatted += "\n---\n\n"
        
        return formatted
    
    def format_dataset_results(
        self, 
        datasets: List[DatasetMetadata], 
        publication_id: str
    ) -> str:
        """
        Format dataset discovery results in a standardized way.
        
        Args:
            datasets: List of dataset metadata
            publication_id: Publication identifier used for search
            
        Returns:
            str: Formatted dataset results
        """
        if not datasets:
            return f"No datasets found for publication: {publication_id}"
        
        formatted = f"## Dataset Discovery Results\n\n"
        formatted += f"**Publication**: {publication_id}\n"
        formatted += f"**Datasets Found**: {len(datasets)}\n\n"
        
        # Group by dataset type
        by_type = {}
        for dataset in datasets:
            dtype = dataset.data_type or DatasetType.GEO
            if dtype not in by_type:
                by_type[dtype] = []
            by_type[dtype].append(dataset)
        
        for dtype, dataset_list in by_type.items():
            formatted += f"### {dtype.value.upper()} Datasets\n\n"
            
            for dataset in dataset_list:
                formatted += f"**Accession**: {dataset.accession}\n"
                formatted += f"**Title**: {dataset.title}\n"
                
                if dataset.description:
                    desc_preview = dataset.description[:200]
                    if len(dataset.description) > 200:
                        desc_preview += "..."
                    formatted += f"**Description**: {desc_preview}\n"
                
                if dataset.organism:
                    formatted += f"**Organism**: {dataset.organism}\n"
                if dataset.platform:
                    formatted += f"**Platform**: {dataset.platform}\n"
                if dataset.samples_count:
                    formatted += f"**Samples**: {dataset.samples_count}\n"
                if dataset.date:
                    formatted += f"**Date**: {dataset.date}\n"
                if dataset.source_url:
                    formatted += f"**URL**: [{dataset.accession}]({dataset.source_url})\n"
                
                formatted += "\n---\n\n"
        
        return formatted
