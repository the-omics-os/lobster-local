# Proteomics Implementation Plan

## Overview
Complete the integration and enhancement of proteomics functionality in the Lobster AI system to ensure industry-standard capabilities and proper data management.

## Current Status
- ✅ Proteomics expert agent registered in agent registry
- ⚠️ Legacy code patterns need cleanup
- ❌ Missing stateless service layer
- ❌ Limited industry tool integration
- ⚠️ Basic functionality present but not optimized

## Implementation Phases

### Phase 1: Code Cleanup and Modernization
**Status: Partially Complete**
- ✅ Agent registered in registry
- ⏳ Remove legacy is_v2 compatibility checks from proteomics_expert.py
- ⏳ Update all tools to use DataManagerV2 directly

### Phase 2: Create Stateless Service Layer
**Status: Not Started**

#### ProteomicsPreprocessingService (New File: `lobster/tools/proteomics_preprocessing_service.py`)
- **Missing value imputation methods:**
  - KNN imputation
  - MinProb (minimum probability)
  - MNAR (Missing Not At Random) imputation
  - Mixed imputation strategies
- **Normalization methods:**
  - Median normalization
  - Quantile normalization
  - VSN (Variance Stabilizing Normalization)
  - Total sum normalization
- **Batch correction:**
  - ComBat for proteomics
  - Median centering
  - Reference-based correction

#### ProteomicsQualityService (New File: `lobster/tools/proteomics_quality_service.py`)
- **Quality metrics:**
  - Missing value patterns analysis
  - CV (Coefficient of Variation) assessment
  - Dynamic range evaluation
  - Sample correlation analysis
  - PCA outlier detection
- **Contaminant detection:**
  - Keratin contamination
  - Common contaminants database
  - Reverse hit identification
- **Technical replicate assessment:**
  - Reproducibility metrics
  - Technical variation quantification

#### ProteomicsAnalysisService (New File: `lobster/tools/proteomics_analysis_service.py`)
- **Statistical analysis:**
  - T-tests with missing value handling
  - ANOVA for multi-group comparisons
  - Mixed-effects models
  - Limma for proteomics
- **Pathway enrichment:**
  - GO enrichment analysis
  - KEGG pathway mapping
  - Reactome pathway analysis
  - String DB integration

#### ProteomicsDifferentialService (New File: `lobster/tools/proteomics_differential_service.py`)
- **Differential expression:**
  - MSstats integration
  - Perseus-like workflows
  - FDR control (Benjamini-Hochberg, q-value)
  - Volcano plot generation
- **Multiple testing correction:**
  - Peptide-level FDR
  - Protein-level FDR
  - Site-level FDR (for PTMs)

### Phase 3: Industry Standard Tool Integration
**Status: Not Started**

#### Format Support Enhancement
- **Add support for:**
  - MaxQuant output (proteinGroups.txt, peptides.txt)
  - MSFragger output
  - Spectronaut output
  - mzTab format
  - PRIDE XML format

#### Database Integration
- **UniProt integration:**
  - Protein annotation retrieval
  - Sequence information
  - PTM site mapping
- **PRIDE/ProteomeXchange:**
  - Dataset download capability
  - Metadata extraction
- **Peptide Atlas:**
  - Reference peptide information
  - Protein coverage data

#### Quantification Methods
- **Implement:**
  - iBAQ (Intensity-Based Absolute Quantification)
  - LFQ (Label-Free Quantification)
  - TMT/iTRAQ support
  - SILAC quantification
  - DIA quantification (OpenSWATH integration)

### Phase 4: Enhanced Data Management
**Status: Not Started**

#### Modality Naming Conventions
- Implement proteomics-specific naming:
  ```
  proteomics_ms_raw
  proteomics_ms_filtered
  proteomics_ms_normalized
  proteomics_ms_imputed
  proteomics_ms_differential
  ```

#### Replicate Handling
- Technical replicate averaging
- Biological replicate management
- Batch effect tracking
- Run order effects

#### Peptide-to-Protein Strategies
- Unique peptides only
- Razor peptide assignment
- Protein group handling
- Shared peptide distribution

### Phase 5: Validation and Quality Control
**Status: Not Started**

#### Comprehensive Validation Rules
- Minimum peptide requirements
- Protein FDR thresholds
- Quantification accuracy checks
- Missing value thresholds

#### Decoy Database Strategies
- Target-decoy approach
- Separate decoy search
- Entrapment databases

#### Match Between Runs
- RT alignment
- Mass accuracy checks
- Transfer quality metrics

## Files to Modify

### Primary Files
1. **`lobster/agents/proteomics_expert.py`**
   - Remove `is_v2` checks
   - Refactor tools to use services
   - Update to match transcriptomics pattern

### New Service Files to Create
1. `lobster/tools/proteomics_preprocessing_service.py`
2. `lobster/tools/proteomics_quality_service.py`
3. `lobster/tools/proteomics_analysis_service.py`
4. `lobster/tools/proteomics_differential_service.py`

### Supporting Files to Update
1. **`lobster/core/adapters/proteomics_adapter.py`**
   - Add MaxQuant format support
   - Enhance peptide mapping
   - Add PTM support

2. **`lobster/core/schemas/proteomics.py`**
   - Add PTM schema
   - Add spectral counting schema
   - Add DIA schema

## Testing Requirements

### Unit Tests
- Create `tests/unit/tools/test_proteomics_services.py`
- Test all normalization methods
- Test missing value imputation
- Test differential analysis

### Integration Tests
- End-to-end proteomics workflow
- MaxQuant output processing
- Database integration tests

## Success Criteria
1. All legacy code patterns removed
2. Complete service layer implementation
3. Support for major proteomics formats
4. Industry-standard analysis methods
5. Comprehensive validation and QC
6. Full test coverage

## Priority Order
1. **Critical**: Remove legacy code patterns
2. **Critical**: Implement core services
3. **High**: Add format support
4. **Medium**: Database integration
5. **Medium**: Advanced analysis features
6. **Low**: ML integration (excluded per requirements)
