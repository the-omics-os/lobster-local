"""
Research Agent for literature discovery and dataset identification.

This agent specializes in searching scientific literature, discovering datasets,
and providing comprehensive research context using the modular publication service
architecture with DataManagerV2 integration.
"""

import re
from typing import List
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langchain_aws import ChatBedrockConverse

from datetime import date

from lobster.config.settings import get_settings
from lobster.core.data_manager_v2 import DataManagerV2
from lobster.tools.publication_service import PublicationService
from lobster.tools.providers.base_provider import DatasetType, PublicationSource
from lobster.utils.logger import get_logger

logger = get_logger(__name__)


def research_agent(
    data_manager: DataManagerV2, 
    callback_handler=None, 
    agent_name: str = "research_agent",
    handoff_tools: List = None
):
    """Create research agent using DataManagerV2 and modular publication service."""
    
    settings = get_settings()
    model_params = settings.get_agent_llm_params('research_agent')
    llm = ChatBedrockConverse(**model_params)
    
    if callback_handler and hasattr(llm, 'with_config'):
        llm = llm.with_config(callbacks=[callback_handler])
    
    # Initialize publication service
    publication_service = PublicationService(data_manager=data_manager)
    
    # Define tools
    @tool
    def search_literature(
        query: str,
        max_results: int = 5,
        sources: str = "pubmed",
        filters: str = None
    ) -> str:
        """
        Search for scientific literature across multiple sources.
        
        Args:
            query: Search query string
            max_results: Number of results to retrieve (default: 5, range: 1-20)
            sources: Publication sources to search (default: "pubmed", options: "pubmed,biorxiv,medrxiv")
            filters: Optional search filters as JSON string (e.g., '{"date_range": {"start": "2020", "end": "2024"}}')
        """
        try:
            # Parse sources
            source_list = []
            if sources:
                for source in sources.split(','):
                    source = source.strip().lower()
                    if source == 'pubmed':
                        source_list.append(PublicationSource.PUBMED)
                    elif source == 'biorxiv':
                        source_list.append(PublicationSource.BIORXIV)
                    elif source == 'medrxiv':
                        source_list.append(PublicationSource.MEDRXIV)
            
            # Parse filters if provided
            filter_dict = None
            if filters:
                import json
                try:
                    filter_dict = json.loads(filters)
                except json.JSONDecodeError:
                    logger.warning(f"Invalid filters JSON: {filters}")
            
            results = publication_service.search_literature(
                query=query,
                max_results=max_results,
                sources=source_list if source_list else None,
                filters=filter_dict
            )
            
            logger.info(f"Literature search completed for: {query[:50]}... (max_results={max_results})")
            return results
            
        except Exception as e:
            logger.error(f"Error searching literature: {e}")
            return f"Error searching literature: {str(e)}"

    @tool
    def find_datasets_from_publication(
        identifier: str,
        dataset_types: str = None,
        include_related: bool = True
    ) -> str:
        """
        Find datasets associated with a scientific publication.
        
        Args:
            identifier: Publication identifier (DOI or PMID)
            dataset_types: Types of datasets to search for, comma-separated (e.g., "geo,sra,arrayexpress")
            include_related: Whether to include related/linked datasets (default: True)
        """
        try:
            # Parse dataset types
            type_list = []
            if dataset_types:
                type_mapping = {
                    'geo': DatasetType.GEO,
                    'sra': DatasetType.SRA,
                    'arrayexpress': DatasetType.ARRAYEXPRESS,
                    'ena': DatasetType.ENA,
                    'bioproject': DatasetType.BIOPROJECT,
                    'biosample': DatasetType.BIOSAMPLE,
                    'dbgap': DatasetType.DBGAP
                }
                
                for dtype in dataset_types.split(','):
                    dtype = dtype.strip().lower()
                    if dtype in type_mapping:
                        type_list.append(type_mapping[dtype])
            
            results = publication_service.find_datasets_from_publication(
                identifier=identifier,
                dataset_types=type_list if type_list else None,
                include_related=include_related
            )
            
            logger.info(f"Dataset discovery completed for: {identifier}")
            return results
            
        except Exception as e:
            logger.error(f"Error finding datasets: {e}")
            return f"Error finding datasets from publication: {str(e)}"

    @tool
    def find_marker_genes(
        query: str,
        max_results: int = 5,
        filters: str = None
    ) -> str:
        """
        Find marker genes for cell types from literature.
        
        Args:
            query: Query with cell_type parameter (e.g., 'cell_type=T_cell disease=cancer')
            max_results: Number of results to retrieve (default: 5, range: 1-15)
            filters: Optional search filters as JSON string
        """
        try:
            # Parse parameters from query
            cell_type_match = re.search(r'cell[_\s]type[=\s]+([^,\s]+)', query)
            disease_match = re.search(r'disease[=\s]+([^,\s]+)', query)
            
            if not cell_type_match:
                return "Please specify cell_type parameter (e.g., 'cell_type=T_cell')"
            
            cell_type = cell_type_match.group(1).strip()
            disease = disease_match.group(1).strip() if disease_match else None
            
            # Build search query for marker genes
            search_query = f'"{cell_type}" marker genes'
            if disease:
                search_query += f' {disease}'
            
            # Parse filters if provided
            filter_dict = None
            if filters:
                import json
                try:
                    filter_dict = json.loads(filters)
                except json.JSONDecodeError:
                    logger.warning(f"Invalid filters JSON: {filters}")
            
            results = publication_service.search_literature(
                query=search_query,
                max_results=max_results,
                filters=filter_dict
            )
            
            # Add context header
            context_header = f"## Marker Gene Search Results for {cell_type}\n"
            if disease:
                context_header += f"**Disease context**: {disease}\n"
            context_header += f"**Search query**: {search_query}\n\n"
            
            logger.info(f"Marker gene search completed for {cell_type} (max_results={max_results})")
            return context_header + results
            
        except Exception as e:
            logger.error(f"Error finding marker genes: {e}")
            return f"Error finding marker genes: {str(e)}"

    @tool
    def discover_related_studies(
        identifier: str,
        research_topic: str = None,
        max_results: int = 5
    ) -> str:
        """
        Discover studies related to a given publication or research topic.
        
        Args:
            identifier: Publication identifier (DOI or PMID) to find related studies
            research_topic: Optional research topic to focus the search
            max_results: Number of results to retrieve (default: 5)
        """
        try:
            # First get metadata from the source publication
            metadata = publication_service.extract_publication_metadata(identifier)
            
            if isinstance(metadata, str):
                return f"Could not extract metadata for {identifier}: {metadata}"
            
            # Build search query based on metadata and research topic
            search_terms = []
            
            # Extract key terms from title
            if metadata.title:
                # Simple keyword extraction (could be enhanced with NLP)
                title_words = re.findall(r'\b[a-zA-Z]{4,}\b', metadata.title.lower())
                # Filter common words and take meaningful terms
                meaningful_terms = [w for w in title_words if w not in ['study', 'analysis', 'using', 'with', 'from', 'data']]
                search_terms.extend(meaningful_terms[:3])
            
            # Add research topic if provided
            if research_topic:
                search_terms.append(research_topic)
            
            # Build search query
            search_query = ' '.join(search_terms[:5])  # Limit to avoid too broad search
            
            if not search_query.strip():
                search_query = "related studies"
            
            results = publication_service.search_literature(
                query=search_query,
                max_results=max_results
            )
            
            # Add context header
            context_header = f"## Related Studies for {identifier}\n"
            context_header += f"**Source publication**: {metadata.title[:100]}...\n"
            context_header += f"**Search strategy**: {search_query}\n"
            if research_topic:
                context_header += f"**Research focus**: {research_topic}\n"
            context_header += "\n"
            
            logger.info(f"Related studies search completed for {identifier}")
            return context_header + results
            
        except Exception as e:
            logger.error(f"Error discovering related studies: {e}")
            return f"Error discovering related studies: {str(e)}"

    @tool
    def search_datasets_directly(
        query: str,
        data_type: str = "geo",
        max_results: int = 5,
        filters: str = None
    ) -> str:
        """
        Search for datasets directly across omics databases.
        
        Args:
            query: Search query for datasets
            data_type: Type of omics data (default: "geo", options: "geo,sra,bioproject,biosample,dbgap")
            max_results: Maximum results to return (default: 5)
            filters: Optional filters as JSON string (e.g., '{"organism": "human", "year": "2023"}')
        """
        try:
            # Map string to DatasetType
            type_mapping = {
                'geo': DatasetType.GEO,
                'sra': DatasetType.SRA,
                'bioproject': DatasetType.BIOPROJECT,
                'biosample': DatasetType.BIOSAMPLE,
                'dbgap': DatasetType.DBGAP,
                'arrayexpress': DatasetType.ARRAYEXPRESS,
                'ena': DatasetType.ENA
            }
            
            dataset_type = type_mapping.get(data_type.lower(), DatasetType.GEO)
            
            # Parse filters if provided
            filter_dict = None
            if filters:
                import json
                try:
                    filter_dict = json.loads(filters)
                except json.JSONDecodeError:
                    logger.warning(f"Invalid filters JSON: {filters}")
            
            results = publication_service.search_datasets_directly(
                query=query,
                data_type=dataset_type,
                max_results=max_results,
                filters=filter_dict
            )
            
            logger.info(f"Direct dataset search completed: {query[:50]}... ({data_type})")
            return results
            
        except Exception as e:
            logger.error(f"Error searching datasets directly: {e}")
            return f"Error searching datasets directly: {str(e)}"

    @tool
    def extract_publication_metadata(
        identifier: str,
        source: str = "auto"
    ) -> str:
        """
        Extract comprehensive metadata from a publication.
        
        Args:
            identifier: Publication identifier (DOI or PMID)
            source: Publication source (default: "auto", options: "auto,pubmed,biorxiv,medrxiv")
        """
        try:
            # Map source string to PublicationSource
            source_obj = None
            if source != "auto":
                source_mapping = {
                    'pubmed': PublicationSource.PUBMED,
                    'biorxiv': PublicationSource.BIORXIV,
                    'medrxiv': PublicationSource.MEDRXIV
                }
                source_obj = source_mapping.get(source.lower())
            
            metadata = publication_service.extract_publication_metadata(
                identifier=identifier,
                source=source_obj
            )
            
            if isinstance(metadata, str):
                return metadata  # Error message
            
            # Format metadata for display
            formatted = f"## Publication Metadata for {identifier}\n\n"
            formatted += f"**Title**: {metadata.title}\n"
            formatted += f"**UID**: {metadata.uid}\n"
            if metadata.journal:
                formatted += f"**Journal**: {metadata.journal}\n"
            if metadata.published:
                formatted += f"**Published**: {metadata.published}\n"
            if metadata.doi:
                formatted += f"**DOI**: {metadata.doi}\n"
            if metadata.pmid:
                formatted += f"**PMID**: {metadata.pmid}\n"
            if metadata.authors:
                formatted += f"**Authors**: {', '.join(metadata.authors[:5])}{'...' if len(metadata.authors) > 5 else ''}\n"
            if metadata.keywords:
                formatted += f"**Keywords**: {', '.join(metadata.keywords)}\n"
            
            if metadata.abstract:
                formatted += f"\n**Abstract**:\n{metadata.abstract[:1000]}{'...' if len(metadata.abstract) > 1000 else ''}\n"
            
            logger.info(f"Metadata extraction completed for: {identifier}")
            return formatted
            
        except Exception as e:
            logger.error(f"Error extracting metadata: {e}")
            return f"Error extracting publication metadata: {str(e)}"

    @tool
    def get_research_capabilities(self) -> str:
        """
        Get information about available research capabilities and providers.
        
        Returns:
            str: Formatted capabilities report
        """
        try:
            return publication_service.get_provider_capabilities()
        except Exception as e:
            logger.error(f"Error getting capabilities: {e}")
            return f"Error getting research capabilities: {str(e)}"

    base_tools = [
        search_literature,
        find_datasets_from_publication,
        find_marker_genes,
        discover_related_studies,
        search_datasets_directly,
        extract_publication_metadata,
        get_research_capabilities
    ]
    
    # Combine base tools with handoff tools if provided
    tools = base_tools + (handoff_tools or [])
    
    system_prompt = """
You are a research specialist focused on scientific literature discovery and dataset identification in bioinformatics and computational biology.

<Role>
Your primary expertise lies in comprehensive literature search, dataset discovery, and research context provision. You work closely with method experts who focus on parameter extraction and computational analysis.
</Role>

<Task>
Given a research inquiry, you will:
1. **Search scientific literature** across multiple sources (PubMed, bioRxiv, medRxiv)
2. **Discover and identify datasets** associated with publications or research topics
3. **Find biological markers** and gene signatures from literature
4. **Identify related studies** and research trends
5. **Extract publication metadata** for comprehensive research context
6. **Provide research landscape overview** for specific topics
7. **Bridge literature to datasets** for downstream analysis

Your role complements the method expert who handles computational parameter extraction and the data expert who manages dataset downloads and processing.
</Task>

<Available Research Tools>

### Literature Discovery
- `search_literature`: Multi-source literature search with advanced filtering
  * sources: "pubmed", "biorxiv", "medrxiv" (comma-separated)
  * filters: JSON string for date ranges, authors, journals, publication types
  * max_results: 1-20 (use 8-15 for comprehensive surveys, 3-5 for focused searches)

- `discover_related_studies`: Find studies related to a publication or topic
  * Automatically extracts key terms from source publications
  * Focuses on methodological or thematic relationships

- `extract_publication_metadata`: Comprehensive metadata extraction
  * Full bibliographic information, abstracts, author lists
  * Standardized format across different sources

### Dataset Discovery
- `find_datasets_from_publication`: Discover datasets from DOI/PMID
  * dataset_types: "geo,sra,arrayexpress,ena,bioproject,biosample,dbgap"
  * include_related: finds linked datasets through NCBI connections
  * Comprehensive dataset reports with download links

- `search_datasets_directly`: Direct omics database search
  * Search GEO, SRA, and other databases independently
  * Filters for organism, platform, date ranges
  * Useful when no specific publication is available

### Biological Discovery
- `find_marker_genes`: Literature-based marker gene identification
  * cell_type parameter required (e.g., "cell_type=T_cell")
  * Optional disease context
  * Cross-references multiple studies for consensus markers

### System Information
- `get_research_capabilities`: Available providers and features

<Research Strategies>

## Comprehensive Literature Review
```python
# 1. Start with broad search to understand landscape
search_literature(
    query="single-cell RNA-seq T cell exhaustion",
    max_results=10,
    sources="pubmed"
)

# 2. Find datasets for each relevant study
find_datasets_from_publication("10.1038/s41586-021-03659-0")

# 3. Discover related studies
discover_related_studies("10.1038/s41586-021-03659-0", "T cell dysfunction")
```

## Dataset-Focused Research
```python
# Search for datasets directly
search_datasets_directly(
    query="single-cell RNA-seq tumor microenvironment",
    data_type="geo",
    max_results=8,
    filters='{{"organism": "human", "year": "2023"}}'
)

# Find publications associated with interesting datasets  
discover_related_studies("GSE162498")
```

## Marker Gene Discovery
```python
# Find marker genes with disease context
find_marker_genes(
    query="cell_type=CD8_T_cell disease=cancer",
    max_results=6
)

# Cross-reference with recent literature
search_literature(
    query="CD8 T cell markers cancer immunotherapy",
    max_results=5,
    filters='{{"date_range": {{"start": "2022", "end": "2024"}}}}'
)
```

<Response Format>
Structure your research findings as:

## Literature Research Summary

### Key Publications Found
1. **[Title]** - PMID: [number] | DOI: [doi]
   - **Key finding**: Brief description
   - **Relevance**: Why important for the query
   - **Datasets**: Associated dataset accessions

### Associated Datasets
- **[Accession]** - [Description]
  - Platform: [technology]
  - Samples: [count]
  - Organism: [species]
  - **Download recommendation**: "Recommend data expert retrieve [accession] for [analysis purpose]"

### Biological Markers Identified
- **[Cell Type]**: Gene1, Gene2, Gene3 (Evidence: X studies)
- **Consensus markers**: Genes found across multiple studies
- **Context-specific**: Disease or condition-specific markers

### Research Trends & Gaps
- **Current methods**: What approaches are being used
- **Methodological consensus**: Areas of agreement in the field
- **Research gaps**: Understudied areas or missing datasets
- **Future directions**: Emerging approaches or technologies

### Recommendations for Method Expert
- **Parameter extraction needed**: Specific publications for computational methods
- **Method standardization**: Areas where parameter consensus is needed
- **Tool comparisons**: Studies comparing different software/approaches

### Next Steps
1. **For method expert**: "Extract computational parameters from PMID:[number]"
2. **For data expert**: "Download and prepare datasets: [accessions]"
3. **Additional literature**: "Search for [specific topic] if needed"

<Important Guidelines>
- **Use multiple sources**: Don't rely solely on PubMed for comprehensive research
- **Cross-reference findings**: Validate important findings across multiple studies
- **Consider recency**: Balance recent developments with foundational studies
- **Link literature to data**: Always identify associated datasets when available
- **Provide context**: Explain why specific studies or datasets are relevant
- **Note limitations**: Mention when literature is sparse or contradictory
- **Suggest follow-ups**: Recommend additional searches if gaps are identified

<Collaboration Notes>
- **Method Expert**: Will extract computational parameters from publications you identify
- **Data Expert**: Will download and process datasets you discover
- **Your Focus**: Literature discovery, dataset identification, research context

Today's date is {date}. Maximum iterations: 6
""".format(
    date=date.today()
)

    return create_react_agent(
        model=llm,
        tools=tools,
        prompt=system_prompt,
        name=agent_name
    )
