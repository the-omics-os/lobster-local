# Implementation Plan

## [Overview]
Split the current method_expert agent into two specialized agents: a research_agent focused on literature discovery and dataset identification, and a refined method_expert focused purely on computational parameter extraction and methodology analysis.

This refactoring addresses the current architectural issue where method_expert handles two distinct responsibilities: literature search/dataset discovery and computational method extraction. The new architecture will create cleaner separation of concerns, improve modularity, and prepare the system for future expansion with additional literature sources (bioRxiv, medRxiv) and advanced tools like knowledge graphs.

The implementation involves creating a new research_agent, refactoring the method_expert to focus on parameter extraction, developing a modular publication service architecture to replace the current pubmed_service, and updating the supervisor's delegation logic to coordinate between the two agents effectively.

## [Types]
New data structures and interfaces for the modular publication service and agent coordination.

**Publication Provider Interface:**
```python
class BasePublicationProvider(ABC):
    """Abstract base class for publication providers."""
    @abstractmethod
    def search_publications(self, query: str, **kwargs) -> str
    @abstractmethod
    def find_datasets_from_publication(self, identifier: str) -> str  
    @abstractmethod
    def extract_publication_metadata(self, identifier: str) -> str
```

**Provider Registry:**
```python
class ProviderRegistry:
    """Registry for managing publication providers."""
    providers: Dict[str, BasePublicationProvider]
    default_provider: str
```

**Agent State Enums:**
```python
class AgentType(Enum):
    RESEARCH = "research_agent"
    METHOD = "method_expert"
    DATA = "data_expert"
```

**Publication Source Types:**
```python
class PublicationSource(Enum):
    PUBMED = "pubmed"
    BIORXIV = "biorxiv" 
    MEDRXIV = "medrxiv"
    ARXIV = "arxiv"
```

## [Files]
File system changes including new files, modifications, and renames.

**New Files to Create:**
- `lobster/agents/research_agent.py` - New research agent for literature and dataset discovery
- `lobster/tools/publication_service.py` - Modular publication service replacing pubmed_service
- `lobster/tools/providers/base_provider.py` - Abstract base class for publication providers
- `lobster/tools/providers/pubmed_provider.py` - PubMed-specific implementation
- `lobster/tools/providers/biorxiv_provider.py` - bioRxiv stub implementation
- `lobster/tools/providers/medrxiv_provider.py` - medRxiv stub implementation
- `lobster/tools/providers/__init__.py` - Provider package initialization

**Files to Modify:**
- `lobster/agents/method_expert.py` - Remove literature search tools, focus on parameter extraction
- `lobster/agents/data_expert.py` - Remove GEO discovery functions, keep download/management
- `lobster/config/agent_registry.py` - Add research_agent configuration
- `lobster/agents/supervisor.py` - Update delegation logic for two-agent workflow
- `lobster/agents/graph.py` - Include research_agent in graph construction

**Files to Rename:**
- `lobster/tools/pubmed_service.py` → Archive for reference, replaced by new modular system

**Configuration Updates:**
- Agent registry entries for research_agent
- Supervisor prompt updates for new delegation patterns
- Import path updates where pubmed_service was used

## [Functions]
Function-level changes across the affected components.

**New Functions in research_agent.py:**
- `research_agent()` - Main agent factory function
- `search_literature()` - Multi-source literature search tool
- `find_datasets_from_publication()` - Dataset discovery from DOI/PMID
- `find_marker_genes()` - Biological marker gene search (moved from method_expert)
- `extract_publication_metadata()` - Publication metadata extraction
- `discover_related_studies()` - Find related publications

**Modified Functions in method_expert.py:**
- `method_expert()` - Remove literature search tools, keep parameter extraction tools
- Remove: `search_pubmed()`, `find_geo_from_doi()`, `find_marker_genes()`
- Keep: `find_method_parameters_from_doi()`, `find_protocol_information()`, `find_method_parameters_for_modality()`
- Add: `extract_computational_parameters()` - Enhanced parameter extraction
- Add: `analyze_methodology()` - Deep method analysis from pre-retrieved publications

**Modified Functions in data_expert.py:**
- Remove: `find_geo_from_doi()`, `find_geo_from_pmid()` - These move to research_agent
- Keep: All download and data management functions
- Enhance: Better integration with research_agent discovered datasets

**New Functions in publication_service.py:**
- `PublicationService.__init__()` - Service initialization with provider registry
- `search_across_sources()` - Multi-provider search orchestration
- `find_datasets()` - Unified dataset discovery
- `get_publication_metadata()` - Standardized metadata extraction
- `register_provider()` - Dynamic provider registration

**Provider Functions:**
- `PubMedProvider.search_publications()` - Enhanced PubMed search
- `PubMedProvider.find_datasets_from_publication()` - GEO/SRA discovery
- `BioRxivProvider.*` - Stub implementations for future development
- `MedRxivProvider.*` - Stub implementations for future development

## [Classes]
Class structure changes and new class definitions.

**New Classes:**

`BasePublicationProvider` (Abstract):
- Location: `lobster/tools/providers/base_provider.py`
- Purpose: Abstract interface for all publication providers
- Key Methods: `search_publications()`, `find_datasets_from_publication()`, `extract_publication_metadata()`
- Inheritance: ABC base class

`PubMedProvider`:
- Location: `lobster/tools/providers/pubmed_provider.py`  
- Purpose: PubMed-specific implementation with enhanced features
- Inherits: `BasePublicationProvider`
- Key Methods: All abstract methods + PubMed-specific utilities
- Integration: Uses existing PubMed functionality from current pubmed_service

`PublicationService`:
- Location: `lobster/tools/publication_service.py`
- Purpose: Main orchestrator for all publication operations
- Key Methods: `search_across_sources()`, `register_provider()`, `get_available_sources()`
- Composition: Contains provider registry and routing logic

`BioRxivProvider` & `MedRxivProvider`:
- Location: `lobster/tools/providers/biorxiv_provider.py` and `medrxiv_provider.py`
- Purpose: Stub implementations for future expansion
- Inherits: `BasePublicationProvider`
- Implementation: Minimal stubs that return "not implemented" messages

**Modified Classes:**

`AgentConfig` updates in `agent_registry.py`:
- Add research_agent configuration entry
- Update method_expert description to reflect new focus

**Removed Classes:**
- None (current PubMedService class will be refactored into PubMedProvider)

## [Dependencies]
Package and import dependency modifications.

**New Dependencies:**
- No new external packages required
- Enhanced internal module structure with providers package

**Import Changes:**
- `from lobster.tools.pubmed_service import PubMedService` → `from lobster.tools.publication_service import PublicationService`
- Add provider imports: `from lobster.tools.providers import PubMedProvider, BioRxivProvider, MedRxivProvider`
- Update agent imports to include research_agent

**Internal Dependencies:**
- research_agent depends on publication_service
- method_expert no longer depends on pubmed_service
- data_expert no longer imports publication-related functions
- supervisor imports both research_agent and updated method_expert

**Backward Compatibility:**
- Maintain compatibility by keeping old pubmed_service functions accessible through publication_service
- Gradual migration path for any external code using pubmed_service directly

## [Testing]
Comprehensive testing strategy for the refactored architecture.

**Test File Requirements:**
- `tests/test_research_agent.py` - Test new research agent functionality
- `tests/test_publication_service.py` - Test modular publication service
- `tests/test_providers/` - Directory for provider-specific tests
- `tests/test_method_expert_refactored.py` - Test updated method expert
- `tests/test_agent_coordination.py` - Test supervisor delegation between agents

**Integration Testing:**
- Test research_agent → method_expert workflow through supervisor
- Test publication service with different providers
- Test backward compatibility with existing workflows
- Validate dataset discovery and parameter extraction pipeline

**Test Data Requirements:**
- Sample DOIs and PMIDs for testing publication retrieval
- Mock responses for bioRxiv/medRxiv stub testing
- Sample publication content for parameter extraction testing

**Validation Strategies:**
- Unit tests for each provider implementation
- Integration tests for cross-agent workflows
- Performance tests to ensure no regression in literature search speed
- Functional tests for supervisor delegation logic

## [Implementation Order]
Logical sequence to minimize conflicts and ensure successful integration.

**Phase 1: Foundation (Steps 1-3)**
1. Create provider architecture and base classes
2. Implement PubMedProvider by refactoring existing PubMedService functionality
3. Create publication_service as orchestrator with provider registry

**Phase 2: Agent Refactoring (Steps 4-6)**
4. Create research_agent with literature search and dataset discovery tools
5. Refactor method_expert to remove literature functions and focus on parameter extraction
6. Update data_expert to remove publication-related functions

**Phase 3: Integration (Steps 7-9)**
7. Update agent_registry.py to include research_agent configuration
8. Update supervisor.py with new delegation logic for research/method agent coordination
9. Update graph.py to include research_agent in the multi-agent system

**Phase 4: Testing and Validation (Steps 10-12)**
10. Create comprehensive test suite for all components
11. Validate agent coordination workflows through supervisor
12. Test backward compatibility and migration path

**Phase 5: Documentation and Cleanup (Steps 13-15)**
13. Update all import statements and remove old pubmed_service references
14. Create migration guide and update documentation
15. Archive old pubmed_service.py and clean up unused code

**Dependencies Between Steps:**
- Steps 1-3 must complete before agent refactoring (Steps 4-6)
- Agent refactoring must complete before integration (Steps 7-9)
- All implementation phases must complete before testing (Steps 10-12)
- Testing must complete before final cleanup (Steps 13-15)

**Risk Mitigation:**
- Each phase includes validation checkpoints
- Maintain old code alongside new code until testing completes
- Implement feature flags for gradual rollout if needed
- Keep detailed rollback procedures for each phase
