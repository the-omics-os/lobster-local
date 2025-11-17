# Archived Services - Phase 2 Migration

**Archive Date**: 2025-01-12
**Migration**: Phase 2 - Service Layer Consolidation
**New Service**: `ContentAccessService` (lobster/tools/content_access_service.py)

## Archived Services

### 1. `publication_service.py`
**Reason**: Replaced by ContentAccessService with capability-based routing
**Functionality**: Literature search, dataset discovery, metadata extraction
**Replacement**: Use `ContentAccessService` instead

### 2. `unified_content_service.py`
**Reason**: Merged into ContentAccessService as part of 3-tier cascade system
**Functionality**: Full-text extraction, PMC/PDF/webpage handling
**Replacement**: Use `ContentAccessService.get_full_content()` and related methods

## Migration Guide

### For Code Using PublicationService

**Before (Phase 1)**:
```python
from lobster.tools.publication_service import PublicationService

publication_service = PublicationService(data_manager=data_manager)
metadata = publication_service.extract_publication_metadata(identifier)
datasets = publication_service.find_datasets_from_publication(identifier)
results = publication_service.search_datasets_directly(query, data_type="geo")
capabilities = publication_service.get_provider_capabilities()
```

**After (Phase 2)**:
```python
from lobster.tools.content_access_service import ContentAccessService

content_service = ContentAccessService(data_manager=data_manager)
metadata = content_service.extract_metadata(identifier)
datasets = content_service.find_linked_datasets(identifier)
results = content_service.discover_datasets(query, dataset_type=DatasetType.GEO)
capabilities = content_service.query_capabilities()
```

### For Code Using UnifiedContentService

**Before (Phase 1)**:
```python
from lobster.tools.unified_content_service import UnifiedContentService

content_service = UnifiedContentService(
    cache_dir=Path(".lobster_workspace") / "literature_cache",
    data_manager=data_manager,
)
content = content_service.get_full_content(source, prefer_webpage=True)
methods = content_service.extract_methods_section(content)
```

**After (Phase 2)**:
```python
from lobster.tools.content_access_service import ContentAccessService

content_service = ContentAccessService(data_manager=data_manager)
content = content_service.get_full_content(source, prefer_webpage=True)
methods = content_service.extract_methods(content)
```

## Key Changes

### Method Name Changes

| Old Method | New Method | Parameter Changes |
|------------|------------|-------------------|
| `extract_publication_metadata()` | `extract_metadata()` | None |
| `find_datasets_from_publication()` | `find_linked_datasets()` | None |
| `search_datasets_directly()` | `discover_datasets()` | `data_type` → `dataset_type` |
| `get_provider_capabilities()` | `query_capabilities()` | None |
| `extract_methods_section()` | `extract_methods()` | Accepts content_result dict |

### Constructor Changes

- **PublicationService**: No constructor parameter changes
- **UnifiedContentService**: Removed `cache_dir` parameter (caching now handled by DataManagerV2)

### Architecture Improvements

1. **Capability-Based Routing**: Provider selection based on declared capabilities (ProviderCapability enum)
2. **Three-Tier Cascade**: PMC (Priority 10) → Webpage (Priority 50) → PDF (Priority 100)
3. **Session Caching**: Integrated with DataManagerV2 for workspace persistence
4. **W3C-PROV Tracking**: Full provenance logging for all operations
5. **10 Core Methods**: Complete API surface (discovery, metadata, content, system)

## Files Requiring Updates

### Primary Consumer (✅ Updated)
- `lobster/agents/research_agent.py` - Updated in Phase 2 completion

### Test Files (⏳ Pending Post-Commit)
- `tests/manual/test_unified_content_real.py`
- `tests/manual/test_fallback_chain_real.py`
- `tests/integration/test_pmc_integration.py`
- `tests/integration/test_publication_service_edge_cases.py`
- `tests/integration/test_geo_publication_edge_cases.py`
- `tests/integration/test_research_agent_resolution.py`
- `tests/integration/test_research_agent_pdf.py`
- `tests/unit/tools/test_unified_content_service.py`
- `tests/integration/test_real_pdf_extraction.py`
- `tests/unit/agents/test_research_agent.py`

### Documentation (⏳ Pending)
- `wiki/37-publication-intelligence-deep-dive.md`
- `wiki_update_report.md`

## Phase 2 Deliverables (Complete)

✅ **ContentAccessService** - 10 methods implemented
✅ **Capability Matrix** - All providers registered
✅ **Research Agent Migration** - Primary consumer updated
✅ **Archiving** - Legacy services preserved with git history
⏳ **Test Migration** - Deferred to post-commit

## Next Phase

**Phase 3**: Create `metadata_assistant` agent as specialist for discovery operations
- Follow `data_expert` LangGraph REACT agent pattern
- Extract 6 tools from research_agent (search_literature, discover_datasets, find_linked_datasets, extract_metadata, validate_metadata, query_capabilities)
- Register in agent_registry.py
- Integrate with supervisor

## References

- **Phase 2 Plan**: `kevin_notes/publisher/research_agent_refactor_phase_2.md`
- **Phase 3 Plan**: `kevin_notes/publisher/research_agent_refactor_phase_3.md`
- **Provider Matrix**: `kevin_notes/publisher/provider_capability_matrix.md`
- **ContentAccessService**: `lobster/tools/content_access_service.py`
