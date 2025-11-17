"""
Metabolomics schema definitions for metabolomics data.

This module defines the expected structure and metadata for metabolomics
datasets including LC-MS, GC-MS, and NMR platforms with appropriate validation
rules for sample-level and metabolite-level metadata standardization.
"""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator

from lobster.core.interfaces.validator import ValidationResult
from lobster.core.schemas.validation import FlexibleValidator

# =============================================================================
# ONTOLOGY FIELDS REMOVED - HANDLED BY EMBEDDING SERVICE
# =============================================================================
# The following fields have been removed from this schema and are now handled
# by the embedding-based ontology matching service:
#
# - organism      → NCBI Taxonomy ID (e.g., 9606 for Homo sapiens)
# - tissue        → UBERON term (e.g., UBERON:0000955 for brain)
#
# Users provide these as free-text strings during data upload.
# The metadata_assistant agent calls the embedding service to map
# strings to canonical ontology terms.
#
# Results are stored in adata.uns["ontology_mappings"], NOT in obs/var.
#
# See: docs/embedding-ontology-service.md
# Integration point: metadata_assistant.standardize_ontology_terms() tool
# =============================================================================


class MetabolomicsSchema:
    """
    Schema definitions for metabolomics data modalities.

    This class provides schema definitions for metabolomics data across
    LC-MS, GC-MS, and NMR platforms with appropriate metadata requirements
    and validation rules.
    """

    @staticmethod
    def get_metabolomics_schema() -> Dict[str, Any]:
        """
        Get schema for metabolomics data (LC-MS, GC-MS, NMR).

        Returns:
            Dict[str, Any]: Metabolomics schema definition
        """
        return {
            "modality": "metabolomics",
            "description": "Metabolomics data schema (LC-MS, GC-MS, NMR)",
            # obs: Observations (samples) metadata - DataFrame with samples as rows
            # Contains per-sample metadata including experimental conditions, technical metrics,
            # and analytical parameters specific to metabolomics experiments
            #
            # Example obs DataFrame:
            #            sample_id  subject_id timepoint condition sample_type   platform ionization_mode extraction_method  n_metabolites  total_intensity
            # Sample_1    Sample_1  Subject_001      Day0   Control       Serum      LC-MS        positive  Methanol-chloroform            234       1.2e8
            # Sample_2    Sample_2  Subject_001     Week4   Control       Serum      LC-MS        positive  Methanol-chloroform            245       1.3e8
            # Sample_3    Sample_3  Subject_002      Day0 Treatment       Serum      LC-MS        positive  Methanol-chloroform            228       1.1e8
            # Sample_4    Sample_4  Subject_002     Week4 Treatment       Serum      LC-MS        positive  Methanol-chloroform            251       1.4e8
            "obs": {
                "required": [],  # Columns that must be present - flexible for diverse experimental designs
                "optional": [  # Standard sample metadata for metabolomics experiments
                    # Core identifiers
                    "sample_id",  # Unique sample identifier
                    "subject_id",  # Subject/patient ID
                    "timepoint",  # Timepoint (Day0, Week4, etc.)
                    # Experimental design
                    "condition",  # Control, Treatment, Disease
                    "batch",  # Analytical batch
                    "replicate",  # Biological replicate
                    # Sample characteristics
                    # NOTE: organism and tissue fields removed - handled by embedding service
                    # See header comment for details on ontology integration
                    "sample_type",  # Serum, Urine, Tissue, Saliva
                    # Analytical metadata
                    "platform",  # LC-MS, GC-MS, NMR
                    "ionization_mode",  # positive, negative, both
                    "instrument_model",  # Orbitrap Fusion, QTOF, etc.
                    "acquisition_method",  # DDA, DIA, targeted
                    "polarity",  # Positive, Negative (alternative to ionization_mode)
                    # Sample preparation
                    "extraction_method",  # Methanol-chloroform, etc.
                    "derivatization",  # TMS, BSTFA, none
                    "internal_standards",  # 13C-labeled metabolites, etc.
                    # Quality metrics
                    "n_metabolites",  # Number of metabolites detected
                    "total_intensity",  # Total peak intensity
                    "median_intensity",  # Median intensity
                    "missing_values_pct",  # Percentage missing values
                    # Database accessions
                    "biosample_accession",  # SAMN identifier
                    "metabolights_sample_id",  # MetaboLights sample ID
                    "workbench_sample_id",  # Workbench sample ID
                ],
                "types": {  # Expected data types for validation and processing
                    "sample_id": "string",
                    "subject_id": "string",
                    "timepoint": "string",
                    "condition": "categorical",
                    "batch": "string",
                    "replicate": "string",
                    "sample_type": "categorical",
                    "platform": "categorical",
                    "ionization_mode": "categorical",
                    "instrument_model": "string",
                    "acquisition_method": "categorical",
                    "polarity": "categorical",
                    "extraction_method": "string",
                    "derivatization": "string",
                    "internal_standards": "string",
                    "n_metabolites": "numeric",
                    "total_intensity": "numeric",
                    "median_intensity": "numeric",
                    "missing_values_pct": "numeric",
                    "biosample_accession": "string",
                    "metabolights_sample_id": "string",
                    "workbench_sample_id": "string",
                },
            },
            # var: Variables (metabolites) metadata - DataFrame with metabolites as rows
            # Contains per-metabolite annotations including chemical identifiers, mass spec parameters,
            # and computational quality metrics
            #
            # Example var DataFrame:
            #                 metabolite_id metabolite_name          hmdb_id      chebi_id  pubchem_cid  kegg_id chemical_formula    mz  retention_time identification_level  is_identified  prevalence
            # MET_001              MET_001         Glucose      HMDB0000122  CHEBI:17234         5793  C00031          C6H12O6 180.06   5.2                    1           True         1.0
            # MET_002              MET_002        Lactate      HMDB0000190  CHEBI:422         612  C00186           C3H6O3  89.02   2.1                    1           True         0.95
            # MET_003              MET_003        Alanine      HMDB0000161  CHEBI:16449        5950  C00041           C3H7NO2  89.05   3.4                    1           True         0.88
            # MET_004              MET_004       Creatine      HMDB0000064  CHEBI:16919         586  C00300          C4H9N3O2 131.07   4.8                    2           True         0.75
            "var": {
                "required": [],  # Columns that must be present - flexible for different annotation levels
                "optional": [  # Standard metabolite metadata columns
                    # Primary identifiers
                    "metabolite_id",  # Internal ID (MET_001, etc.)
                    "metabolite_name",  # Common name
                    # Chemical identifiers
                    "hmdb_id",  # HMDB ID (HMDB0000001)
                    "chebi_id",  # ChEBI ID (CHEBI:15377)
                    "pubchem_cid",  # PubChem CID
                    "kegg_id",  # KEGG Compound ID (C00001)
                    "inchi",  # InChI string
                    "inchikey",  # InChIKey
                    "smiles",  # SMILES string
                    # Chemical properties
                    "chemical_formula",  # C6H12O6
                    "monoisotopic_mass",  # Exact mass (Da)
                    "molecular_weight",  # Average molecular weight
                    # MS-specific
                    "mz",  # m/z value observed
                    "retention_time",  # RT in minutes
                    "retention_index",  # RI (GC-MS)
                    "adduct",  # [M+H]+, [M-H]-, etc.
                    # Metabolite statistics
                    "n_samples",  # Number of samples with detection
                    "mean_intensity",  # Mean intensity across samples
                    "median_intensity",  # Median intensity
                    "prevalence",  # Proportion of samples detected
                    "cv",  # Coefficient of variation
                    "missing_values_pct",  # Missing value percentage
                    # Quality flags
                    "is_identified",  # MS2 confirmed (boolean)
                    "identification_level",  # MSI level 1-4
                    "is_internal_standard",  # Internal standard flag
                    "is_qc_compound",  # QC compound flag
                    # Biological annotations
                    "pathways",  # KEGG/Reactome pathways (list)
                    "class",  # Chemical class (lipid, amino acid, etc.)
                    "subclass",  # Chemical subclass
                ],
                "types": {  # Expected data types for validation and processing
                    "metabolite_id": "string",
                    "metabolite_name": "string",
                    "hmdb_id": "string",
                    "chebi_id": "string",
                    "pubchem_cid": "string",
                    "kegg_id": "string",
                    "inchi": "string",
                    "inchikey": "string",
                    "smiles": "string",
                    "chemical_formula": "string",
                    "monoisotopic_mass": "numeric",
                    "molecular_weight": "numeric",
                    "mz": "numeric",
                    "retention_time": "numeric",
                    "retention_index": "numeric",
                    "adduct": "string",
                    "n_samples": "numeric",
                    "mean_intensity": "numeric",
                    "median_intensity": "numeric",
                    "prevalence": "numeric",
                    "cv": "numeric",
                    "missing_values_pct": "numeric",
                    "is_identified": "boolean",
                    "identification_level": "string",
                    "is_internal_standard": "boolean",
                    "is_qc_compound": "boolean",
                    "pathways": "string",  # Can be comma-separated or list
                    "class": "categorical",
                    "subclass": "categorical",
                },
            },
            # layers: Alternative intensity matrices with same dimensions as X
            # Store different transformations/versions of the intensity data
            # Each layer is a 2D matrix: samples x metabolites, same shape as adata.X
            #
            # Example layers (4 samples x 4 metabolites):
            #
            # layers['raw_intensity'] (raw peak intensities):
            #          MET_001  MET_002  MET_003  MET_004
            # Sample_1   15000    28000        0      890
            # Sample_2   14200    31000     5200      920
            # Sample_3   16800    42000     2100      780
            # Sample_4   15600        0     8100      850
            #
            # layers['normalized'] (library-size normalized):
            #          MET_001  MET_002  MET_003  MET_004
            # Sample_1    1.34     2.50     0.00     0.08
            # Sample_2    1.19     2.60     0.44     0.08
            # Sample_3    1.40     3.50     0.18     0.07
            # Sample_4    1.30     0.00     0.68     0.07
            "layers": {
                "required": [],  # No layers are strictly required (main data stored in adata.X)
                "optional": [  # Common data transformations for metabolomics
                    "raw_intensity",  # Raw peak intensities
                    "normalized",  # Normalized intensities
                    "log2_intensity",  # Log2-transformed
                    "imputed",  # Imputed missing values
                    "batch_corrected",  # Batch-corrected
                    "pareto_scaled",  # Pareto scaling (common in metabolomics)
                    "auto_scaled",  # Auto-scaling (mean-centered + unit variance)
                ],
            },
            # obsm: Observations (samples) multidimensional annotations
            # Stores per-sample multidimensional data like embeddings or coordinates
            # Each entry is a 2D array: samples x dimensions
            #
            # Example obsm matrices:
            #
            # obsm['X_pca'] (PCA coordinates - 4 samples x 3 PCs):
            #          PC1    PC2   PC3
            # Sample_1 -8.5   3.2   1.1
            # Sample_2 -7.8   2.9   0.9
            # Sample_3  9.2  -3.5  -1.2
            # Sample_4  8.1  -3.1  -1.0
            #
            # obsm['X_plsda'] (PLS-DA coordinates - 4 samples x 2 components):
            #          PLSDA1  PLSDA2
            # Sample_1  -5.2     2.1
            # Sample_2  -4.9     1.8
            # Sample_3   5.4    -2.3
            # Sample_4   5.1    -2.0
            "obsm": {
                "required": [],  # No embeddings are required (generated during analysis)
                "optional": [  # Common dimensionality reduction results
                    "X_pca",  # PCA coordinates
                    "X_plsda",  # PLS-DA coordinates
                    "X_opls",  # OPLS coordinates
                    "X_umap",  # UMAP embedding
                    "X_tsne",  # t-SNE embedding
                ],
            },
            # uns: Unstructured annotations - global metadata and analysis parameters
            # Stores dataset-level information, analysis parameters, and complex results
            # Contains nested dictionaries, arrays, or objects that don't fit obs/var structure
            #
            # Example uns structure:
            # uns = {
            #     'metabolights_accession': 'MTBLS123',  # MetaboLights ID
            #     'workbench_accession': 'ST001234',      # Metabolomics Workbench ID
            #     'bioproject_accession': 'PRJNA123456',  # NCBI BioProject
            #     'investigation': {...},                 # ISA-Tab investigation
            #     'study': {...},                         # ISA-Tab study
            #     'assay': {...},                         # ISA-Tab assay
            #     'instrument_model': 'Orbitrap Fusion Lumos',
            #     'ionization_source': 'ESI',
            #     'peak_detection_method': 'XCMS',
            #     'differential_analysis': {...},         # Statistical results
            #     'pathway_analysis': {...}               # MSEA results
            # }
            "uns": {
                "required": [],  # No global metadata is strictly required
                "optional": [  # Common analysis metadata and computational results
                    # Study metadata
                    "title",  # Study title
                    "abstract",  # Study abstract
                    "publication_doi",  # DOI
                    # Database accessions
                    "metabolights_accession",  # MTBLS123
                    "workbench_accession",  # ST001234
                    "massive_accession",  # MSV000012345
                    "bioproject_accession",  # PRJNA123456
                    # ISA-Tab metadata (MetaboLights)
                    "investigation",  # Investigation metadata (dict)
                    "study",  # Study metadata (dict)
                    "assay",  # Assay metadata (dict)
                    # Instrument parameters
                    "instrument_model",  # Orbitrap Fusion Lumos
                    "ionization_source",  # ESI, APCI, EI
                    "mass_analyzer",  # Orbitrap, QTOF, QQQ
                    "resolution",  # Mass resolution (60000 FWHM)
                    "mass_range",  # m/z range [50, 1500]
                    "collision_energy",  # Collision energy (V or eV)
                    # Chromatography
                    "column_model",  # Column type
                    "column_length",  # Length (mm)
                    "column_diameter",  # Diameter (mm)
                    "particle_size",  # Particle size (μm)
                    "mobile_phase_a",  # Mobile phase A composition
                    "mobile_phase_b",  # Mobile phase B composition
                    "flow_rate",  # Flow rate (mL/min)
                    "gradient_program",  # Gradient profile (dict/list)
                    # Data processing
                    "peak_detection_method",  # XCMS, MZmine, etc.
                    "normalization_method",  # Total ion current, internal standard
                    "missing_value_method",  # Imputation method (KNN, minimum)
                    # Analysis results
                    "differential_analysis",  # Differential metabolite results
                    "pathway_analysis",  # MSEA/pathway enrichment
                    "statistical_tests",  # t-test, ANOVA, etc.
                    # Provenance
                    "provenance",  # W3C-PROV tracking
                    "processing_date",  # ISO 8601 timestamp
                    "lobster_version",  # Lobster version
                    # Ontology mappings (embedding service results)
                    "ontology_mappings",  # Organism/tissue ontology IDs
                ],
            },
        }

    @staticmethod
    def create_validator(
        strict: bool = False,
        ignore_warnings: Optional[List[str]] = None,
    ) -> FlexibleValidator:
        """
        Create a validator for metabolomics data.

        Args:
            strict: Whether to use strict validation
            ignore_warnings: List of warning types to ignore

        Returns:
            FlexibleValidator: Configured validator
        """
        schema = MetabolomicsSchema.get_metabolomics_schema()

        ignore_set = set(ignore_warnings) if ignore_warnings else set()

        # Add default ignored warnings for metabolomics
        ignore_set.update(
            [
                "Unexpected obs columns",
                "Unexpected var columns",
                "missing values",  # Common in metabolomics (30-70%)
                "Very sparse data",
            ]
        )

        validator = FlexibleValidator(
            schema=schema,
            name="MetabolomicsValidator",
            ignore_warnings=ignore_set,
        )

        # Add metabolomics-specific validation rules
        validator.add_custom_rule(
            "check_metabolite_identifiers", _validate_metabolite_ids
        )
        validator.add_custom_rule("check_intensity_data", _validate_intensity_data)
        validator.add_custom_rule("check_missing_values", _validate_missing_values)

        # Add cross-database accession validation
        validator.add_custom_rule(
            "check_cross_database_accessions",
            lambda adata: _validate_cross_database_accessions(
                adata, modality="metabolomics"
            ),
        )

        return validator

    @staticmethod
    def get_recommended_qc_thresholds() -> Dict[str, Any]:
        """
        Get recommended quality control thresholds for metabolomics.

        Returns:
            Dict[str, Any]: QC thresholds and recommendations
        """
        return {
            "min_metabolites_per_sample": 50,
            "max_missing_values_pct": 70.0,  # Metabolomics typically has 30-70% missing
            "min_samples_per_metabolite": 2,
            "min_intensity": 1000,  # Platform-dependent
            "max_cv_qc_samples": 30.0,  # CV in QC samples
            "min_identification_level": 4,  # MSI level (1-4, where 1 is best)
        }


def _validate_metabolite_ids(adata) -> "ValidationResult":
    """Validate metabolite identifier format and uniqueness."""
    from lobster.core.interfaces.validator import ValidationResult

    result = ValidationResult()

    # Check for metabolite IDs in var
    if "metabolite_id" in adata.var.columns:
        ids = adata.var["metabolite_id"]

        # Check for duplicates
        duplicates = ids.duplicated().sum()
        if duplicates > 0:
            result.add_warning(f"{duplicates} duplicate metabolite IDs found")

        # Check for missing IDs
        missing = ids.isna().sum()
        if missing > 0:
            result.add_warning(f"{missing} missing metabolite IDs")

    # Check for chemical identifiers
    chem_ids = ["hmdb_id", "chebi_id", "pubchem_cid", "kegg_id"]
    present_ids = [cid for cid in chem_ids if cid in adata.var.columns]

    if not present_ids:
        result.add_warning(
            "No chemical identifiers found (HMDB, ChEBI, PubChem, KEGG). "
            "Consider adding at least one standard identifier for metabolites."
        )

    return result


def _validate_intensity_data(adata) -> "ValidationResult":
    """Validate intensity data characteristics."""

    from lobster.core.interfaces.validator import ValidationResult

    result = ValidationResult()

    # Check for negative values
    if hasattr(adata.X, "min"):
        min_val = adata.X.min()
        if min_val < 0:
            result.add_warning(
                f"Negative values found in intensity matrix (min: {min_val}). "
                "This is unusual for metabolomics data."
            )

    # Check for zeros (common in metabolomics due to missing values)
    if hasattr(adata.X, "data"):  # Sparse matrix
        zero_pct = (adata.X.data == 0).sum() / adata.X.data.size * 100
    else:  # Dense matrix
        zero_pct = (adata.X == 0).sum() / adata.X.size * 100

    if zero_pct > 70:
        result.add_warning(
            f"{zero_pct:.1f}% of values are zero. High sparsity is common in "
            "metabolomics but may indicate data quality issues if >70%."
        )
    elif zero_pct > 30:
        result.add_info(
            f"{zero_pct:.1f}% of values are zero. This is typical for metabolomics data."
        )

    return result


def _validate_missing_values(adata) -> "ValidationResult":
    """Validate missing value patterns in metabolomics data."""
    import numpy as np

    from lobster.core.interfaces.validator import ValidationResult

    result = ValidationResult()

    # Calculate missing value percentage
    if hasattr(adata.X, "data"):  # Sparse matrix
        missing_pct = (adata.X.data == 0).sum() / adata.X.data.size * 100
    else:  # Dense matrix
        missing_pct = np.isnan(adata.X).sum() / adata.X.size * 100

    if missing_pct > 70:
        result.add_warning(
            f"{missing_pct:.1f}% missing values. Consider filtering metabolites "
            "with >50% missing or using appropriate imputation methods."
        )
    elif missing_pct > 30:
        result.add_info(
            f"{missing_pct:.1f}% missing values. This is within expected range "
            "for metabolomics data (30-70%)."
        )

    # Check for metabolites with high missing rates
    if hasattr(adata.X, "toarray"):
        X_dense = adata.X.toarray()
    else:
        X_dense = adata.X

    metabolite_missing = np.isnan(X_dense).sum(axis=0) / X_dense.shape[0] * 100
    high_missing = (metabolite_missing > 50).sum()

    if high_missing > 0:
        result.add_warning(
            f"{high_missing} metabolites have >50% missing values across samples. "
            "Consider filtering these features."
        )

    return result


def _validate_cross_database_accessions(
    adata, modality: str = "metabolomics"
) -> "ValidationResult":
    """
    Validate cross-database accession format and structure.

    Checks database accession fields in adata.uns against expected formats
    using the database_mappings registry.

    Args:
        adata: AnnData object to validate
        modality: Data modality (transcriptomics, proteomics, metabolomics, metagenomics)

    Returns:
        ValidationResult: Validation results with accession format errors/warnings
    """
    from lobster.core.interfaces.validator import ValidationResult
    from lobster.core.schemas.database_mappings import (
        get_accession_url,
        get_accessions_for_modality,
        validate_accession,
    )

    result = ValidationResult()

    # Get expected accessions for this modality
    expected_accessions = get_accessions_for_modality(modality)

    # Check each accession field in uns
    for field_name, accession_spec in expected_accessions.items():
        if field_name in adata.uns:
            value = adata.uns[field_name]

            # Skip empty/None values
            if value is None or (isinstance(value, str) and not value.strip()):
                continue

            # Validate accession format
            if not validate_accession(field_name, value):
                result.add_warning(
                    f"Invalid {accession_spec.database_name} accession format: '{value}' "
                    f"(expected pattern: {accession_spec.example})"
                )
            else:
                # Successful validation - add info with URL
                url = get_accession_url(field_name, value)
                if url:
                    result.add_info(
                        f"Valid {accession_spec.database_name} accession: {value} ({url})"
                    )

    return result


# =============================================================================
# Pydantic Metadata Schema for Sample-Level Metadata Standardization
# =============================================================================
# This schema is used by the metadata_assistant agent for cross-dataset
# metadata harmonization, standardization, and validation.
# Phase 3 addition for metabolomics metadata operations.
# =============================================================================


class MetabolomicsMetadataSchema(BaseModel):
    """
    Pydantic schema for metabolomics sample-level metadata standardization.

    This schema defines the expected structure for sample metadata across
    metabolomics experiments. It enforces controlled vocabularies and data
    types for consistent metadata representation across datasets.

    Used by metadata_assistant agent for:
    - Cross-dataset sample ID mapping
    - Metadata standardization and harmonization
    - Dataset completeness validation
    - Multi-omics integration preparation

    NOTE: organism and tissue fields have been removed. These are now handled
    by the embedding-based ontology matching service. See module header for details.

    Attributes:
        sample_id: Unique sample identifier (required)
        subject_id: Subject/patient identifier for biological replicates
        timepoint: Timepoint or developmental stage
        condition: Experimental condition (e.g., "Control", "Treatment")
        sample_type: Type of biological sample (e.g., "Serum", "Urine", "Tissue")
        platform: Analytical platform (e.g., "LC-MS", "GC-MS", "NMR")
        ionization_mode: Ionization mode for mass spec ("positive", "negative", "both")
        extraction_method: Metabolite extraction method
        derivatization: Derivatization method (if applicable)
        internal_standards: Internal standards used for normalization
        batch: Batch identifier for technical replicates
        additional_metadata: Flexible dict for custom fields
    """

    # Required fields
    sample_id: str = Field(..., description="Unique sample identifier", min_length=1)

    # Optional core fields
    subject_id: Optional[str] = Field(None, description="Subject/patient identifier")
    timepoint: Optional[str] = Field(
        None, description="Timepoint or developmental stage"
    )
    condition: str = Field(
        ..., description="Experimental condition (e.g., Control, Treatment)"
    )
    sample_type: Optional[str] = Field(
        None, description="Type of biological sample (Serum, Urine, Tissue, etc.)"
    )

    # NOTE: organism and tissue fields removed - handled by embedding service
    # See module header for details

    platform: str = Field(
        ..., description="Analytical platform (e.g., LC-MS, GC-MS, NMR)"
    )
    ionization_mode: str = Field(
        ..., description="Ionization mode (positive, negative, both)"
    )

    # Metabolomics-specific fields
    extraction_method: Optional[str] = Field(
        None, description="Metabolite extraction method"
    )
    derivatization: Optional[str] = Field(
        None, description="Derivatization method (if applicable)"
    )
    internal_standards: Optional[str] = Field(
        None, description="Internal standards used for normalization"
    )
    batch: Optional[str] = Field(None, description="Batch identifier")

    # Flexible additional metadata
    additional_metadata: Optional[Dict[str, Any]] = Field(
        default_factory=dict, description="Additional custom metadata fields"
    )

    class Config:
        """Pydantic configuration."""

        json_schema_extra = {
            "example": {
                "sample_id": "Sample_A_Rep1",
                "subject_id": "Subject_001",
                "timepoint": "Day0",
                "condition": "Control",
                "sample_type": "Serum",
                "platform": "LC-MS",
                "ionization_mode": "positive",
                "extraction_method": "Methanol-chloroform",
                "derivatization": "TMS",
                "internal_standards": "13C-labeled metabolites",
                "batch": "Batch1",
                "additional_metadata": {
                    "instrument": "Orbitrap Fusion",
                    "replicate": "Rep1",
                    "collection_time": "09:00",
                },
            }
        }

    @field_validator("platform")
    @classmethod
    def validate_platform(cls, v: str) -> str:
        """Validate platform is a known metabolomics platform."""
        allowed = {
            # Mass spectrometry
            "lc-ms",
            "lc_ms",
            "gc-ms",
            "gc_ms",
            "ce-ms",
            "ce_ms",
            "maldi-tof",
            "maldi_tof",
            "direct_infusion_ms",
            "dims",
            # NMR
            "nmr",
            "1h_nmr",
            "13c_nmr",
            # Other
            "ic-ms",
            "ic_ms",
        }
        v_lower = v.lower().replace("-", "_").replace(" ", "_")
        if v_lower not in allowed:
            # Allow unknown platforms
            return v
        # Normalize to uppercase with hyphen for consistency
        return v_lower.upper().replace("_", "-")

    @field_validator("ionization_mode")
    @classmethod
    def validate_ionization_mode(cls, v: str) -> str:
        """Validate ionization mode."""
        allowed = {"positive", "negative", "both", "pos", "neg", "dual"}
        v_lower = v.lower()
        if v_lower not in allowed:
            raise ValueError(f"ionization_mode must be one of {allowed}, got '{v}'")
        # Normalize to standard values
        if v_lower in {"pos"}:
            return "positive"
        if v_lower in {"neg"}:
            return "negative"
        if v_lower in {"dual"}:
            return "both"
        return v_lower

    @field_validator("condition")
    @classmethod
    def validate_condition(cls, v: str) -> str:
        """Ensure condition is not empty."""
        if not v or not v.strip():
            raise ValueError("condition cannot be empty")
        return v.strip()

    @field_validator("sample_id")
    @classmethod
    def validate_sample_id(cls, v: str) -> str:
        """Ensure sample_id is not empty and has no leading/trailing whitespace."""
        if not v or not v.strip():
            raise ValueError("sample_id cannot be empty")
        return v.strip()

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary representation.

        Returns:
            Dict[str, Any]: Dictionary with all fields including additional_metadata
        """
        base_dict = self.model_dump(exclude={"additional_metadata"}, exclude_none=True)
        if self.additional_metadata:
            base_dict.update(self.additional_metadata)
        return base_dict

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MetabolomicsMetadataSchema":
        """
        Create schema from dictionary, automatically handling unknown fields.

        Args:
            data: Dictionary with metadata fields

        Returns:
            MetabolomicsMetadataSchema: Validated schema instance
        """
        # Extract known fields
        known_fields = set(cls.model_fields.keys()) - {"additional_metadata"}
        schema_data = {k: v for k, v in data.items() if k in known_fields}

        # Put remaining fields in additional_metadata
        additional = {k: v for k, v in data.items() if k not in known_fields}
        if additional:
            schema_data["additional_metadata"] = additional

        return cls(**schema_data)
