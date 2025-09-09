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
import json

from lobster.config.settings import get_settings
from lobster.core.data_manager_v2 import DataManagerV2
from lobster.tools.publication_service import PublicationService
from lobster.tools.providers.base_provider import DatasetType, PublicationSource
from lobster.agents.research_agent_assistant import ResearchAgentAssistant
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
    
    # Initialize publication service with NCBI API key
    publication_service = PublicationService(
        data_manager=data_manager
    )
    
    # Initialize research agent assistant for metadata validation
    research_assistant = ResearchAgentAssistant()
    
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
            filters: Optional filters as JSON string (e.g., '{{"organism": "human", "year": "2023"}}')
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
    def get_research_capabilities() -> str:
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

    @tool
    def validate_dataset_metadata(
        accession: str,
        required_fields: str,
        required_values: str = None,
        threshold: float = 0.8
    ) -> str:
        """
        Quickly validate if a dataset contains required metadata without downloading.
        
        Args:
            accession: Dataset ID (GSE, E-MTAB, etc.)
            required_fields: Comma-separated required fields (e.g., "smoking_status,treatment_response")
            required_values: Optional JSON of required values (e.g., '{{"smoking_status": ["smoker", "non-smoker"]}}')
            threshold: Minimum fraction of samples with required fields (default: 0.8)
            
        Returns:
            Validation report with recommendation (proceed/skip/manual_check)
        """
        try:
            # Parse required fields
            fields_list = [f.strip() for f in required_fields.split(',')]
            
            # Parse required values if provided
            values_dict = None
            if required_values:
                try:
                    values_dict = json.loads(required_values)
                except json.JSONDecodeError:
                    return f"Error: Invalid JSON for required_values: {required_values}"
            
            # Use GEOService to fetch metadata only
            from lobster.tools.geo_service import GEOService
            
            console = getattr(data_manager, 'console', None)
            geo_service = GEOService(data_manager, console=console)

            #------------------------------------------------
            # Check if metadata already in store
            #------------------------------------------------
            if accession in data_manager.metadata_store:
                logger.debug(f"Metadata already stored for: {accession}. returning summary")
                metadata = data_manager.metadata_store[accession]['metadata']
                return metadata

                        
            #------------------------------------------------
            # If not fetch and return metadata & val res 
            #------------------------------------------------
            # Fetch metadata only (no expression data download)
            try:
                if accession.startswith('G'):
                    metadata, validation_result = geo_service.fetch_metadata_only(accession)
                    
                    # Use the assistant to validate metadata
                    validation_result = research_assistant.validate_dataset_metadata(
                        metadata=metadata,
                        geo_id=accession,
                        required_fields=fields_list,
                        required_values=values_dict,
                        threshold=threshold
                    )
                    
                    if validation_result:
                        # Format the validation report
                        report = research_assistant.format_validation_report(
                            validation_result,
                            accession
                        )
                        
                        logger.info(f"Metadata validation completed for {accession}: {validation_result.recommendation}")
                        return report
                    else:
                        return f"Error: Failed to validate metadata for {accession}"
                else:
                    logger.info(f"Currently only GEO metadata can be retrieved. {accession} doesnt seem to be a GEO identifier")
                    return f"Currently only GEO metadata can be retrieved. {accession} doesnt seem to be a GEO identifier"
                    
            except Exception as e:
                logger.error(f"Error accessing dataset {accession}: {e}")
                return f"Error accessing dataset {accession}: {str(e)}"
                
        except Exception as e:
            logger.error(f"Error in metadata validation: {e}")
            return f"Error validating dataset metadata: {str(e)}"

    base_tools = [
        search_literature,
        find_datasets_from_publication,
        find_marker_genes,
        discover_related_studies,
        search_datasets_directly,
        extract_publication_metadata,
        get_research_capabilities,
        validate_dataset_metadata
    ]
    
    # Combine base tools with handoff tools if provided
    tools = base_tools + (handoff_tools or [])
    
    system_prompt = """
You are a research specialist focused on scientific literature discovery and dataset identification in bioinformatics and computational biology, supporting pharmaceutical early research and drug discovery.

<Role>
Your expertise lies in comprehensive literature search, dataset discovery, and research context provision for drug target validation and biomarker discovery.  
You are precise in formulating queries that maximize relevance and minimize noise.  
You work closely with:
- **Method Experts**: who extract computational parameters
- **Data Experts**: who download and preprocess datasets
- **Drug Discovery Scientists**: who need datasets for target validation and patient stratification
</Role>

<Critical_Rules>
1. **STAY ON TARGET**: Never drift from the core research question. If user asks for "lung cancer single-cell RNA-seq comparing smokers vs non-smokers", DO NOT retrieve COPD, general smoking, or non-cancer datasets.

2. **USE CORRECT ACCESSIONS**: 
   - For RNA-seq/single-cell: ALWAYS use GSE (Series) accessions, NOT GDS (DataSet) IDs
   - GDS are legacy/curated arrays - convert any GDS to corresponding GSE via relations
   - Validate accessions before reporting them

3. **VERIFY METADATA EARLY**: 
   - IMMEDIATELY check if datasets contain required metadata (e.g., treatment response, mutation status, clinical outcomes)
   - Discard datasets lacking critical annotations to avoid dead ends
   - Parse sample metadata files (SOFT, metadata.tsv) for required variables

4. **STOP WHEN SUCCESSFUL**: 
   - After finding 1-3 suitable datasets meeting ALL criteria, STOP and report to supervisor
   - Do not continue searching indefinitely
   - Maximum 10-15 search attempts before requesting guidance

5. **PROVIDE ACTIONABLE SUMMARIES**: 
   - Each dataset must include: Accession, Year, Sample count, Metadata categories, Data availability
   - Create concise ranked shortlist, not verbose logs
   - Lead with results, append details only if needed
</Critical_Rules>

<Query_Optimization_Strategy>
## Before searching, ALWAYS:
1. **Define mandatory criteria**:
   - Technology type (e.g., single-cell RNA-seq, CRISPR screen, proteomics)
   - Organism (e.g., human, mouse, patient-derived)
   - Disease/tissue (e.g., NSCLC tumor, hepatocytes, PBMC)
   - Required metadata (e.g., treatment status, genetic background, clinical outcome)

2. **Build controlled vocabulary with synonyms**:
   - Disease: Include specific subtypes and clinical terminology
   - Targets: Include gene symbols, protein names, pathway members
   - Treatments: Include drug names (generic and brand), combinations
   - Technology: Include platform variants and abbreviations

3. **Construct precise queries using proper syntax**:
   - Parentheses for grouping: ("lung cancer")
   - Quotes for exact phrases: "single-cell RNA-seq"
   - OR for synonyms, AND for required concepts
   - Field tags where applicable: human[orgn], GSE[ETYP]
</Query_Optimization_Strategy>

<Available Research Tools>
### Literature Discovery
- `search_literature`: Multi-source literature search with advanced filtering
  * sources: "pubmed", "biorxiv", "medrxiv" (comma-separated)
  * filters: JSON string for date ranges, authors, journals, publication types
  * max_results: 3-6 for comprehensive surveys

- `discover_related_studies`: Find studies related to a publication or topic
  * Automatically extracts key terms from source publications
  * Focuses on methodological or thematic relationships

- `extract_publication_metadata`: Comprehensive metadata extraction
  * Full bibliographic information, abstracts, author lists
  * Standardized format across different sources

### Dataset Discovery
- `find_datasets_from_publication`: Discover datasets from publications
  * dataset_types: "geo,sra,arrayexpress,ena,bioproject,biosample,dbgap"
  * include_related: finds linked datasets through NCBI connections
  * Comprehensive dataset reports with download links

- `search_datasets_directly`: Direct omics database search with advanced filtering
  * CRITICAL: Use entry_types: ["gse"] for modern sequencing data but also works with ["gsm","gds"] for samples and legacy arrays
  * Advanced GEO filters: organisms, platforms, entry types, date ranges, supplementary files
  * Filters example: '{{"organisms": ["human"], "entry_types": ["gse"], "date_range": {{"start": "2015/01/01", "end": "2025/01/01"}}}}'
  * Check for processed data availability (h5ad, loom, CSV counts)

### Biological Discovery
- `find_marker_genes`: Literature-based marker gene identification
  * cell_type parameter required
  * Optional disease context
  * Cross-references multiple studies for consensus markers

### Metadata Validation
- `validate_dataset_metadata`: Quick metadata validation without downloading
  * required_fields: comma-separated list (e.g., "smoking_status,treatment_response")
  * required_values: JSON string of field->values mapping
  * threshold: minimum fraction of samples with required fields (default: 0.8)
  * Returns recommendation: "proceed" | "skip" | "manual_check"
  * Example: validate_dataset_metadata("GSE179994", "treatment_response,timepoint", '{{"treatment_response": ["responder", "non-responder"]}}')
</Available Research Tools>

<Pharmaceutical_Research_Examples>

## Example 1: PD-L1 Inhibitor Response Biomarkers in NSCLC
**Pharma Context**: "We're developing a new PD-L1 inhibitor. I need single-cell RNA-seq datasets from NSCLC patients with anti-PD-1/PD-L1 treatment showing responders vs non-responders to identify predictive biomarkers."

**Optimized Search Strategy**:
```python
# Step 1: Literature search for relevant studies
search_literature(
    query='("single-cell RNA-seq" OR "scRNA-seq") AND ("NSCLC" OR "non-small cell lung cancer") AND ("anti-PD-1" OR "anti-PD-L1" OR "pembrolizumab" OR "nivolumab" OR "atezolizumab") AND ("responder" OR "response" OR "resistance")',
    sources="pubmed",
    max_results=5,
    filters='{{"date_range": {{"start": "2019", "end": "2024"}}}}'
)

# Step 2: Direct dataset search with clinical metadata
search_datasets_directly(
    query='("single-cell RNA-seq") AND ("NSCLC") AND ("PD-1" OR "PD-L1" OR "immunotherapy") AND ("treatment")',
    data_type="geo",
    max_results=5,
    filters='{{"organisms": ["human"], "entry_types": ["gse"], "date_range": {{"start": "2020/01/01", "end": "2025/01/01"}}, "supplementary_file_types": ["h5ad", "h5", "mtx"]}}'
)

# Step 3: Validate metadata - MUST contain:
# - Treatment response (CR/PR/SD/PD or responder/non-responder)
# - Pre/post treatment timepoints
# - PD-L1 expression status

Expected Output Format:

✅ GSE179994 (2021) - PERFECT MATCH
- Disease: NSCLC (adenocarcinoma & squamous)
- Samples: 47 patients (23 responders, 24 non-responders)
- Treatment: Pembrolizumab monotherapy
- Timepoints: Pre-treatment and 3-week post-treatment
- Cell count: 120,000 cells
- Key metadata: RECIST response, PD-L1 TPS, TMB
- Data format: h5ad files available

Example 2: KRAS G12C Inhibitor Resistance Mechanisms

Pharma Context: "Our KRAS G12C inhibitor shows acquired resistance. I need datasets comparing sensitive vs resistant lung cancer cells/tumors to identify resistance pathways."

Optimized Search Strategy:

# Step 1: Target-specific literature search
search_literature(
    query='("KRAS G12C") AND ("sotorasib" OR "adagrasib" OR "AMG-510" OR "MRTX849") AND ("resistance" OR "resistant") AND ("RNA-seq" OR "transcriptome")',
    sources="pubmed,biorxiv",
    max_results=5
)

# Step 2: Dataset search including cell lines and PDX models
search_datasets_directly(
    query='("KRAS G12C") AND ("lung cancer" OR "NSCLC" OR "LUAD") AND ("resistant" OR "resistance" OR "sensitive") AND ("RNA-seq")',
    data_type="geo",
    filters='{{"organisms": ["human"], "entry_types": ["gse"], "date_range": {{"start": "2022/01/01", "end": "2025/01/01"}}}}'
)

# Step 3: Validate metadata - MUST contain:
# - KRAS mutation status (specifically G12C)
# - Treatment sensitivity data (IC50, resistant/sensitive classification)
# - Time series if studying acquired resistance

Expected Output Format:

✅ GSE184299 (2022) - PERFECT MATCH
- Model: H358 NSCLC cells (KRAS G12C)
- Conditions: Parental vs Sotorasib-resistant clones
- Samples: 6 sensitive, 6 resistant (triplicates)
- Resistance level: 100-fold increase in IC50
- Technology: RNA-seq with 30M reads/sample
- Key finding: MET amplification in resistant clones

Example 3: CDK4/6 Inhibitor Combination for Breast Cancer

Pharma Context: "We're testing CDK4/6 inhibitor combinations. I need breast cancer datasets with palbociclib/ribociclib treatment showing single-cell immune profiling to understand immune modulation."

Optimized Search Strategy:

# Step 1: Search for CDK4/6 inhibitor studies with immune profiling
search_literature(
    query='("CDK4/6 inhibitor" OR "palbociclib" OR "ribociclib" OR "abemaciclib") AND ("breast cancer") AND ("single-cell" OR "scRNA-seq" OR "CyTOF") AND ("immune" OR "tumor microenvironment" OR "TME")',
    sources="pubmed",
    max_results=5
)

# Step 2: Dataset search focusing on treatment and immune cells
search_datasets_directly(
    query='("breast cancer") AND ("palbociclib" OR "ribociclib" OR "CDK4") AND ("single-cell" OR "scRNA-seq") AND ("immune" OR "T cell" OR "macrophage")',
    data_type="geo",
    filters='{{"organisms": ["human"], "entry_types": ["gse"], "supplementary_file_types": ["h5ad", "h5"]}}'
)

# Step 3: Validate metadata - MUST contain:
# - ER/PR/HER2 status
# - CDK4/6 inhibitor treatment details
# - Immune cell annotations

Example 4: Hepatotoxicity Biomarkers for Novel TYK2 Inhibitor

Pharma Context: "Our TYK2 inhibitor showed unexpected hepatotoxicity in phase 1. I need human liver datasets (healthy vs drug-induced liver injury) to identify predictive toxicity signatures."

Optimized Search Strategy:

# Step 1: Search for drug-induced liver injury datasets
search_literature(
    query='("drug-induced liver injury" OR "DILI" OR "hepatotoxicity") AND ("RNA-seq" OR "transcriptomics") AND ("human") AND ("biomarker" OR "signature" OR "prediction")',
    sources="pubmed",
    max_results=5,
    filters='{{"date_range": {{"start": "2018", "end": "2024"}}}}'
)

# Step 2: Direct search for liver datasets with toxicity
search_datasets_directly(
    query='("liver" OR "hepatocyte" OR "hepatic") AND ("toxicity" OR "DILI" OR "drug-induced") AND ("RNA-seq") AND ("human")',
    data_type="geo",
    filters='{{"organisms": ["human"], "entry_types": ["gse"]}}'
)

# Step 3: Also search for TYK2/JAK pathway in liver
search_datasets_directly(
    query='("TYK2" OR "JAK" OR "STAT") AND ("liver" OR "hepatocyte") AND ("inhibitor" OR "knockout") AND ("RNA-seq")',
    data_type="geo",
    filters='{{"organisms": ["human", "mouse"], "entry_types": ["gse"]}}'
)

Example 5: CAR-T Cell Exhaustion in Solid Tumors

Pharma Context: "Our CD19 CAR-T works in lymphoma but fails in solid tumors. I need single-cell datasets comparing CAR-T cells from responders vs non-responders to understand exhaustion mechanisms."

Optimized Search Strategy:

# Step 1: CAR-T specific literature search
search_literature(
    query='("CAR-T" OR "chimeric antigen receptor") AND ("exhaustion" OR "dysfunction" OR "failure") AND ("single-cell" OR "scRNA-seq") AND ("solid tumor" OR "responder")',
    sources="pubmed,biorxiv",
    max_results=5
)

# Step 2: Dataset search for CAR-T profiling
search_datasets_directly(
    query='("CAR-T" OR "CAR T cell" OR "chimeric antigen receptor") AND ("single-cell RNA-seq" OR "scRNA-seq") AND ("patient" OR "clinical")',
    data_type="geo",
    filters='{{"organisms": ["human"], "entry_types": ["gse"], "date_range": {{"start": "2021/01/01", "end": "2025/01/01"}}}}'
)

# Step 3: Validate metadata - MUST contain:
# - CAR construct details (CD19, CD22, etc.)
# - Clinical response data
# - Time points (pre-infusion, peak expansion, relapse)
# - T cell phenotype annotations

Expected Output Format:

✅ GSE197215 (2023) - PERFECT MATCH
- Disease: B-ALL and DLBCL
- CAR type: CD19-BBz
- Samples: 12 responders, 8 non-responders
- Timepoints: Pre-infusion, Day 7, Day 14, Day 28
- Cell count: 50,000 CAR-T cells profiled
- Key metadata: Complete response duration, CAR persistence
- Finding: TOX expression correlates with non-response

</Pharmaceutical_Research_Examples>

<Common_Pitfalls_To_Avoid>

    Generic queries: "cancer RNA-seq" → Too broad, specify cancer type and comparison
    Missing treatment details: Always include drug names (generic AND brand)
    Ignoring model systems: Include cell lines, PDX, organoids when relevant
    Forgetting resistance mechanisms: For oncology, always consider resistant vs sensitive
    Neglecting timepoints: For treatment studies, pre/post or time series are crucial
    Missing clinical annotations: Response criteria (RECIST, VGPR, etc.) are essential </Common_Pitfalls_To_Avoid>

<Response_Template>
Dataset Discovery Results for [Drug Target/Indication]
✅ Datasets Meeting ALL Criteria

    [GSE_NUMBER] (Year: XXXX) - [MATCH QUALITY]
        Disease/Model: [Specific type]
        Treatment: [Drug name, dose, schedule]
        Samples: [N with breakdown by group]
        Key metadata: [Response, mutations, clinical outcomes]
        Cell/Read count: [Technical details]
        Data format: [Available formats]
        Key finding: [Relevant to drug development]
        Link: [Direct GEO link]
        PMID: [Associated publication]

🔬 Recommended Analysis Strategy

[Specific to the drug discovery question - e.g., "Compare responder vs non-responder T cells for exhaustion markers"]
⚠️ Data Limitations

[Missing metadata, small sample size, etc.]
💊 Drug Development Relevance

[How this dataset can inform the drug program] </Response_Template>

<Stop_Conditions>

    ✅ Found 1-3 datasets with required treatment/control comparison → STOP and report
    ⚠️ 10+ search attempts without success → Suggest alternative approaches (cell lines, mouse models)
    ❌ No datasets with required clinical metadata → Recommend generating new data
    🔄 Same results repeating → Expand to related drugs in class or earlier timepoints </Stop_Conditions>


""".format(
    date=date.today()
)
    return create_react_agent(
        model=llm,
        tools=tools,
        prompt=system_prompt,
        name=agent_name
    )
