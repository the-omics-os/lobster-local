# GEO Provider Implementation Guide

## Overview

The GEO Provider enables direct search of NCBI's Gene Expression Omnibus (GEO) DataSets database using E-utilities. This implementation supports all query patterns from the official NCBI API documentation and provides advanced filtering capabilities for precise dataset discovery.

## Key Features

- **Direct GEO DataSets Search**: Search the GEO database directly without going through PubMed
- **Advanced Filtering**: Support for organisms, platforms, entry types, date ranges, and supplementary file types
- **Official API Compliance**: Implements all query patterns from NCBI documentation
- **WebEnv/QueryKey Support**: Efficient result pagination using NCBI's history server
- **Modular Architecture**: Seamlessly integrates with the existing provider system

## Architecture

### Components

1. **GEOProvider**: Main provider class implementing BasePublicationProvider interface
2. **GEOQueryBuilder**: Specialized query construction for GEO-specific syntax
3. **GEOSearchFilters**: Pydantic model for type-safe filter specification
4. **Integration Layer**: Connects to PublicationService and research agent

### Query Flow

```
User Query → GEOSearchFilters → GEOQueryBuilder → NCBI eSearch → eSummary → Formatted Results
```

## Usage Examples

### Basic Dataset Search

```python
from lobster.tools.publication_service import PublicationService
from lobster.core.data_manager_v2 import DataManagerV2

# Initialize service
data_manager = DataManagerV2()
service = PublicationService(data_manager)

# Simple search
results = service.search_datasets_directly(
    query="single-cell RNA-seq",
    data_type=DatasetType.GEO,
    max_results=10
)
```

### Advanced Filtering

```python
# Search with comprehensive filters
filters = {
    "organisms": ["human", "mouse"],
    "entry_types": ["gse"],  # Series only
    "published_last_n_months": 6,
    "supplementary_file_types": ["h5", "h5ad"],
    "max_results": 20
}

results = service.search_datasets_directly(
    query="ATAC-seq chromatin accessibility",
    data_type=DatasetType.GEO,
    filters=filters
)
```

### Platform and Date Range Search

```python
# Search by specific platform and date range
filters = {
    "platforms": ["GPL24676", "GPL570"],
    "date_range": {"start": "2023/01", "end": "2024/12"},
    "organisms": ["human"]
}

results = service.search_datasets_directly(
    query="transcriptome analysis",
    data_type=DatasetType.GEO,
    filters=filters
)
```

### Research Agent Integration

The GEO provider is automatically available in the research agent:

```python
# Use in research agent
search_datasets_directly(
    query="immune cell differentiation",
    data_type="geo",
    filters='{"organisms": ["human"], "entry_types": ["gse"], "published_last_n_months": 3}'
)
```

## Official NCBI Query Examples

The implementation supports all official NCBI example queries:

### Example I: Recent Series
```
GSE[ETYP] AND "published last 3 months"[Filter]
```

**Usage:**
```python
filters = {
    "entry_types": ["gse"],
    "published_last_n_months": 3
}
```

### Example II: Organism and Date Range
```
yeast[ORGN] AND 2007/01:2007/03[PDAT]
```

**Usage:**
```python
filters = {
    "organisms": ["yeast"],
    "date_range": {"start": "2007/01", "end": "2007/03"}
}
```

### Example III: Platform with File Types
```
GPL96[ACCN] AND gse[ETYP] AND cel[suppFile]
```

**Usage:**
```python
filters = {
    "platforms": ["GPL96"],
    "entry_types": ["gse"],
    "supplementary_file_types": ["cel"]
}
```

### Example IV: Complex Multi-Filter Query
```
"single-cell RNA-seq" AND human[ORGN] AND gse[ETYP] AND h5[suppFile]
```

**Usage:**
```python
filters = {
    "organisms": ["human"],
    "entry_types": ["gse"],
    "supplementary_file_types": ["h5"]
}
service.search_datasets_directly(
    query="single-cell RNA-seq",
    data_type=DatasetType.GEO,
    filters=filters
)
```

## Filter Reference

### Available Filters

| Filter | Type | Description | Examples |
|--------|------|-------------|----------|
| `organisms` | List[str] | Organism names (auto-mapped) | `["human", "mouse", "yeast"]` |
| `entry_types` | List[str] | GEO entry types | `["gse", "gds", "gpl", "gsm"]` |
| `platforms` | List[str] | Platform accessions | `["GPL96", "GPL570"]` |
| `date_range` | Dict | Publication date range | `{"start": "2023/01", "end": "2024/12"}` |
| `published_last_n_months` | int | Recent publications | `3, 6, 12` |
| `supplementary_file_types` | List[str] | File extensions | `["cel", "h5", "h5ad", "txt"]` |
| `max_results` | int | Result limit | `1-5000` (default: 20) |

### Organism Name Mapping

The system automatically maps common organism names:

- `homo sapiens`, `h. sapiens` → `human`
- `mus musculus`, `m. musculus` → `mouse`
- `saccharomyces cerevisiae`, `s. cerevisiae` → `yeast`
- `drosophila melanogaster`, `d. melanogaster` → `fly`
- `caenorhabditis elegans`, `c. elegans` → `worm`

### Entry Types

- **GSE** (`gse`): Series - Complete experiments
- **GDS** (`gds`): DataSets - Curated gene expression profiles
- **GPL** (`gpl`): Platforms - Array designs and sequencing platforms
- **GSM** (`gsm`): Samples - Individual sample records

## Integration Points

### PublicationService

The GEO provider is automatically registered in PublicationService:

```python
# Provider is available automatically
providers = service.registry.list_providers()
# Returns: [PublicationSource.PUBMED, PublicationSource.GEO]

# Get GEO-specific provider
geo_provider = service.registry.get_provider(PublicationSource.GEO)
```

### Research Agent

Enhanced `search_datasets_directly` tool with GEO support:

```python
@tool
def search_datasets_directly(
    query: str,
    data_type: str = "geo",  # Now supports advanced GEO filtering
    max_results: int = 5,
    filters: str = None  # JSON string with GEO-specific filters
) -> str:
```

### Configuration

GEO provider configuration via `GEOProviderConfig`:

```python
from lobster.tools.providers.geo_provider import GEOProviderConfig

config = GEOProviderConfig(
    max_results=50,
    email="your.email@domain.com",
    api_key="your_ncbi_api_key",  # Optional but recommended
    include_summaries=True,
    cache_results=True
)
```

## Error Handling

### Common Issues

1. **Invalid Query Syntax**: Automatic validation with helpful error messages
2. **Rate Limiting**: Built-in retry logic with exponential backoff
3. **Network Errors**: Robust error handling with fallback mechanisms
4. **Empty Results**: Clear messaging when no datasets match criteria

### Rate Limiting

- **Without API key**: 3 requests/second
- **With API key**: 10 requests/second
- **Automatic retry**: Exponential backoff on 429 errors

## Performance Considerations

### WebEnv/QueryKey Optimization

The provider uses NCBI's history server for efficient result handling:

```python
# Automatic WebEnv usage for large result sets
search_result = provider.search_geo_datasets(query, filters)
# WebEnv stored: search_result.web_env, search_result.query_key

# Efficient summary retrieval
summaries = provider.get_dataset_summaries(search_result)
```

### Caching

- **Session Cache**: WebEnv sessions cached for reuse
- **Result Cache**: Optional caching of search results
- **Configurable TTL**: Control cache expiration times

## Testing

### Unit Tests

Comprehensive test suite covering:

```bash
# Run GEO provider tests
pytest tests/test_geo_provider.py -v

# Test categories:
# - Query builder functionality
# - Provider integration
# - Official NCBI examples
# - Error handling
# - Filter conversion
```

### Test Coverage

- **Query Construction**: All filter combinations
- **API Integration**: Mock NCBI responses
- **Official Examples**: All four NCBI example queries
- **Error Cases**: Invalid inputs, network failures
- **Integration**: PublicationService registration

## Migration from Legacy System

### Before (PubMed-only)

```python
# Old: Only searches through PubMed links
find_datasets_from_publication("10.1038/nature12345")
```

### After (Direct GEO Search)

```python
# New: Direct GEO database search
search_datasets_directly(
    query="your research topic",
    data_type="geo",
    filters='{...}'  # Advanced filtering
)

# Still available: PubMed-linked datasets
find_datasets_from_publication("10.1038/nature12345")
```

## Best Practices

### Query Construction

1. **Use Specific Terms**: Include relevant biological terms
2. **Apply Filters**: Narrow results with organism, platform, date filters
3. **Entry Type Selection**: Choose appropriate GSE/GDS types
4. **File Type Filtering**: Filter by available supplementary files

### Performance

1. **Use API Key**: Register for NCBI API key for higher rate limits
2. **Batch Requests**: Use WebEnv for large result sets
3. **Cache Results**: Enable caching for repeated queries
4. **Reasonable Limits**: Use appropriate max_results values

### Research Workflow

1. **Broad Search**: Start with general terms and basic filters
2. **Refine Results**: Add organism and platform filters
3. **Date Filtering**: Focus on recent or specific time periods
4. **File Availability**: Filter by required supplementary file types

## Troubleshooting

### Common Issues

**Query Returns No Results**
- Check organism name spelling
- Verify platform accessions
- Adjust date ranges
- Reduce filter strictness

**Rate Limit Errors**
- Add NCBI API key to configuration
- Reduce request frequency
- Use built-in retry mechanism

**Network Timeouts**
- Check internet connectivity
- Verify NCBI service status
- Increase timeout values in configuration

### Debug Mode

Enable detailed logging:

```python
import logging
logging.getLogger('lobster.tools.providers.geo_provider').setLevel(logging.DEBUG)
```

## Future Enhancements

### Planned Features

1. **Advanced Analytics**: Query performance metrics
2. **Smart Filtering**: ML-based filter suggestions
3. **Bulk Operations**: Batch dataset processing
4. **Extended Metadata**: Additional field extraction

### API Extensions

1. **Custom Field Tags**: Support for new NCBI field tags
2. **Advanced Operators**: Complex boolean query logic
3. **Result Ranking**: Relevance-based result ordering
4. **Export Formats**: Multiple output format options

## Contributing

### Development Setup

```bash
# Install development dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/test_geo_provider.py

# Code style
black lobster/tools/providers/geo_*.py
```

### Adding New Features

1. **Query Builder**: Extend `GEOQueryBuilder` for new filters
2. **Provider Methods**: Add methods to `GEOProvider` class
3. **Integration**: Update `PublicationService` routing
4. **Documentation**: Update this guide and docstrings
5. **Testing**: Add comprehensive test coverage

## References

- [NCBI E-utilities Documentation](https://www.ncbi.nlm.nih.gov/books/NBK25499/)
- [GEO Database Overview](https://www.ncbi.nlm.nih.gov/geo/)
- [NCBI API Keys](https://ncbiinsights.ncbi.nlm.nih.gov/2017/11/02/new-api-keys-for-the-e-utilities/)
- [GEO Query Syntax](https://www.ncbi.nlm.nih.gov/geo/browse/?view=search)
