# Agent Archive Notice

This directory contains deprecated agents and modules that have been archived during the research agent refactoring project.

## Archived Modules

### research_agent_assistant.py
**Archived:** 2025-01-12
**Reason:** Functionality split across new modular agents (Phase 3 refactoring)

**Original Functionality:**
1. **PDF Resolution & Access Discovery** (Primary Feature)
   - Automatic PMID/DOI → PDF URL resolution
   - Tiered waterfall strategy: PMC → bioRxiv/medRxiv → Publisher Direct → Suggestions
   - Batch publication resolution (up to 5 papers)
   - Access suggestion generation for paywalled papers
   - Resolution report formatting (single + batch)

2. **Metadata Validation** (Already Extracted in Phase 2)
   - Dataset metadata validation pre-download
   - Schema compliance checking
   - *Note:* This functionality was already extracted to `MetadataValidationService` in Phase 2

**Migration Plan:**

| Functionality | Migrated To | Target Phase | Status |
|---------------|-------------|--------------|--------|
| Sample ID mapping | `SampleMappingService` | Phase 3 | ✅ Complete |
| Metadata standardization | `MetadataStandardizationService` | Phase 3 | ✅ Complete |
| Dataset validation | `MetadataStandardizationService` | Phase 3 | ✅ Complete |
| Metadata reading | `MetadataStandardizationService` | Phase 3 | ✅ Complete |
| **PDF resolution** | `research_agent` (direct tools) | **Phase 4** | ⏳ Pending |
| **Batch PDF resolution** | `research_agent` (direct tools) | **Phase 4** | ⏳ Pending |
| **Access suggestions** | `research_agent` (direct tools) | **Phase 4** | ⏳ Pending |

**Phase 4 Integration Details:**

The PDF resolution features will be integrated directly into `research_agent` as tools:
- `resolve_paper_access()` - Replace resolve_publication_to_pdf()
- `get_full_content()` - Enhanced PDF extraction with auto-resolution
- Batch functionality will be handled by enhanced `extract_paper_methods()` tool
- Access suggestions will be integrated into content access service

**Why Archived:**
- **Phase 3 Goal:** Separate metadata operations into dedicated `metadata_assistant` agent with 4 specialized tools
- **Code Smell:** research_agent_assistant mixed PDF resolution (research domain) with metadata operations (data quality domain)
- **Architectural Improvement:** PDF resolution belongs in research_agent, metadata operations belong in metadata_assistant
- **Duplication:** Metadata validation was already extracted to MetadataValidationService in Phase 2

**Dependencies:**
- Uses `PublicationResolver` from `lobster.tools.providers.publication_resolver`
- LLM integration via `ChatBedrockConverse` for intelligent resolution decisions
- **Note:** PublicationResolver service will remain active and be used by research_agent in Phase 4

**Breaking Changes:**
- Direct imports of `ResearchAgentAssistant` will fail after archival
- Existing code should migrate to:
  - Metadata operations → Use `metadata_assistant` agent (Phase 3)
  - PDF resolution → Use `research_agent` tools (Phase 4)

**References:**
- Phase 3 Implementation: `kevin_notes/publisher/tmp_todo.md` (Tasks 1-7)
- Phase 4 Planning: `kevin_notes/publisher/tmp_todo.md` (Phase 4: research_agent Refactoring)
- New Metadata Services: `lobster/tools/sample_mapping_service.py`, `lobster/tools/metadata_standardization_service.py`
- New Metadata Agent: `lobster/agents/metadata_assistant/`

---

**For Developers:**
If you encounter a missing import for `research_agent_assistant`, check:
1. For metadata operations → Use `metadata_assistant` agent (registered in agent_registry)
2. For PDF resolution → Use `research_agent` tools (Phase 4 implementation pending)
3. If you need the archived code for reference → See `lobster/agents/archive/research_agent_assistant.py`
