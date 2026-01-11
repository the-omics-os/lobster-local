"""
Ontology models for disease/tissue/organism matching.

Migration-stable models for Phase 1 (keyword) -> Phase 2 (embeddings).
Part of the Strangler Fig migration pattern for ontology modernization.

Phase 1: JSON-backed keyword matching
Phase 2 (Q2 2026): ChromaDB embedding-based semantic search
"""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class DiseaseMatch(BaseModel):
    """
    Universal disease match result.

    Phase 1: confidence=1.0 (exact keyword match)
    Phase 2: confidence=0.0-1.0 (semantic similarity from embeddings)

    API contract remains stable across both phases.
    """

    disease_id: str = Field(
        description="Disease ID: 'crc' (Phase 1) or 'MONDO:0005575' (Phase 2)"
    )
    name: str = Field(description="Human-readable disease name")
    confidence: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Match confidence (1.0=exact keyword, <1.0=semantic)",
    )
    match_type: str = Field(
        default="exact_keyword",
        description="Method: 'exact_keyword' (Phase 1), 'semantic_embedding' (Phase 2)",
    )
    matched_term: str = Field(description="Which keyword/query triggered the match")
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Extensible: mondo_id, umls_cui, mesh_terms (Phase 2)",
    )


class DiseaseConcept(BaseModel):
    """
    Disease knowledge representation.

    Phase 1: Uses keywords only for matching
    Phase 2: Adds mondo_id, embeddings populated by ChromaDB

    Keywords are preserved in Phase 2 for hybrid matching (boost exact matches).
    """

    id: str = Field(description="Internal ID: 'crc', 'uc', 'cd', 'healthy'")
    name: str = Field(description="Display name: 'Colorectal Cancer'")
    keywords: List[str] = Field(
        description="Phase 1 matching keywords, Phase 2 boosting terms"
    )

    # Phase 2 fields (optional for backward compatibility)
    mondo_id: Optional[str] = Field(
        default=None, description="MONDO ontology ID (Phase 2)"
    )
    umls_cui: Optional[str] = Field(default=None, description="UMLS Concept ID")
    mesh_terms: List[str] = Field(
        default_factory=list, description="MeSH descriptor IDs"
    )


class DiseaseOntologyConfig(BaseModel):
    """
    Configuration schema for disease_ontology.json.

    Provides validation and documentation for the ontology config file.
    """

    version: str = Field(description="Config version (semver)")
    schema_version: str = Field(default="1.0", description="Schema format version")
    backend: str = Field(
        default="json", description="Backend type: 'json' (Phase 1) or 'embeddings' (Phase 2)"
    )
    description: str = Field(default="", description="Human-readable description")
    diseases: List[DiseaseConcept] = Field(
        description="List of disease concepts with keywords"
    )
