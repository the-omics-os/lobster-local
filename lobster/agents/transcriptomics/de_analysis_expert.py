"""
Differential Expression Analysis Sub-Agent for specialized DE workflows.

This sub-agent handles all differential expression analysis tools including:
- Pseudobulk aggregation from single-cell data
- Formula-based experimental design
- pyDESeq2 differential expression analysis
- Iterative DE analysis with comparison
- Pathway enrichment analysis

CRITICAL SCIENTIFIC FIXES:
1. DESeq2 requires RAW INTEGER COUNTS from adata.raw.X (not normalized adata.X)
2. Minimum replicate threshold changed from 2 to 3 for stable variance estimation
3. Warning when any condition has fewer than 4 replicates (low statistical power)
"""

from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

from lobster.agents.transcriptomics.state import DEAnalysisExpertState
from lobster.config.llm_factory import create_llm
from lobster.config.settings import get_settings
from lobster.core import (
    AggregationError,
    DesignMatrixError,
    FormulaError,
    InsufficientCellsError,
    PseudobulkError,
)
from lobster.core.data_manager_v2 import DataManagerV2
from lobster.services.analysis.bulk_rnaseq_service import (
    BulkRNASeqError,
    BulkRNASeqService,
)
from lobster.services.analysis.differential_formula_service import (
    DifferentialFormulaService,
)
from lobster.services.analysis.pseudobulk_service import PseudobulkService
from lobster.utils.logger import get_logger

logger = get_logger(__name__)


class DEAnalysisError(Exception):
    """Base exception for differential expression analysis errors."""

    pass


class ModalityNotFoundError(DEAnalysisError):
    """Raised when requested modality doesn't exist."""

    pass


class InsufficientReplicatesError(DEAnalysisError):
    """Raised when there are insufficient replicates for stable variance estimation."""

    pass


def create_de_prompt() -> str:
    """Create the system prompt for the DE analysis expert agent."""
    return f"""
You are a specialized sub-agent for differential expression (DE) analysis in transcriptomics workflows.

<Role>
You handle all DE-related tasks for both single-cell (pseudobulk) and bulk RNA-seq data.
You are called by the parent transcriptomics_expert via delegation tools.
You report results back to the parent agent, not directly to users.
</Role>

<Critical Scientific Requirements>
**CRITICAL**: DESeq2/pyDESeq2 requires RAW INTEGER COUNTS, not normalized data.
- Always use adata.raw.X when extracting count matrices for DE analysis
- If adata.raw is not available, warn the user that results may be inaccurate
- Minimum 3 replicates per condition required for stable variance estimation
- Warn when any condition has fewer than 4 replicates (low statistical power)
</Critical Scientific Requirements>

<Available Tools>
## Pseudobulk Tools (Single-Cell to Bulk)
- `create_pseudobulk_matrix`: Aggregate single-cell data to pseudobulk
- `prepare_differential_expression_design`: Set up experimental design for DE

## DE Analysis Tools
- `run_pseudobulk_differential_expression`: Run pyDESeq2 on pseudobulk data
- `run_differential_expression_analysis`: Simple 2-group DE comparison
- `validate_experimental_design`: Validate design for statistical power

## Formula-Based DE Tools (Agent-Guided)
- `suggest_formula_for_design`: Analyze metadata and suggest formulas
- `construct_de_formula_interactive`: Build and validate formulas step-by-step
- `run_differential_expression_with_formula`: Execute formula-based DE

## Iteration & Comparison Tools
- `iterate_de_analysis`: Try different formulas/filters
- `compare_de_iterations`: Compare results between iterations

## Pathway Analysis
- `run_pathway_enrichment_analysis`: GO/KEGG pathway enrichment
</Available Tools>

<Workflow Guidelines>
1. Always validate experimental design before running DE analysis
2. Use adata.raw.X for count matrices (DESeq2 requirement)
3. Require minimum 3 replicates per condition
4. Warn when n < 4 per condition (low power)
5. Suggest appropriate formulas based on metadata structure
6. Support iterative analysis for formula refinement
</Workflow Guidelines>

Today's date: {date.today()}
""".strip()


def de_analysis_expert(
    data_manager: DataManagerV2,
    callback_handler=None,
    agent_name: str = "de_analysis_expert",
    delegation_tools: List = None,
    workspace_path: Optional[Path] = None,
):
    """
    Create differential expression analysis sub-agent.

    This agent handles all DE-related tasks for transcriptomics workflows.

    Args:
        data_manager: DataManagerV2 instance for modality management
        callback_handler: Optional LangChain callback handler
        agent_name: Name for the agent
        delegation_tools: List of delegation tools from parent agent

    Returns:
        LangGraph react agent configured for DE analysis
    """
    settings = get_settings()
    model_params = settings.get_agent_llm_params("de_analysis_expert")
    llm = create_llm("de_analysis_expert", model_params, workspace_path=workspace_path)

    # Normalize callbacks to a flat list (fix double-nesting bug)
    if callback_handler and hasattr(llm, "with_config"):
        callbacks = callback_handler if isinstance(callback_handler, list) else [callback_handler]
        llm = llm.with_config(callbacks=callbacks)

    # Initialize stateless services
    pseudobulk_service = PseudobulkService()
    bulk_rnaseq_service = BulkRNASeqService(data_manager=data_manager)
    formula_service = DifferentialFormulaService()

    analysis_results = {"summary": "", "details": {}}

    # -------------------------
    # HELPER FUNCTIONS
    # -------------------------
    def _extract_raw_counts(adata) -> Tuple[pd.DataFrame, bool]:
        """
        Extract raw counts from AnnData, preferring adata.raw.X.

        CRITICAL: DESeq2 requires raw integer counts, not normalized data.

        Args:
            adata: AnnData object

        Returns:
            Tuple of (count_matrix DataFrame, used_raw_flag)
        """
        used_raw = False

        # CRITICAL FIX: Use raw counts for DESeq2
        if adata.raw is not None:
            raw_data = adata.raw.X
            if hasattr(raw_data, "toarray"):
                raw_data = raw_data.toarray()
            count_matrix = pd.DataFrame(
                raw_data.T, index=adata.raw.var_names, columns=adata.obs_names
            )
            used_raw = True
            logger.info("Using adata.raw.X for count matrix (recommended for DESeq2)")
        else:
            # Fallback to adata.X with warning
            logger.warning(
                "No adata.raw found - using adata.X which may be normalized. "
                "DESeq2 requires raw counts for accurate results."
            )
            data = adata.X
            if hasattr(data, "toarray"):
                data = data.toarray()
            count_matrix = pd.DataFrame(
                data.T, index=adata.var_names, columns=adata.obs_names
            )

        return count_matrix, used_raw

    def _validate_replicate_counts(
        metadata: pd.DataFrame,
        groupby: str,
        min_replicates: int = 3,  # SCIENTIFIC FIX: Changed from 2 to 3
    ) -> Dict[str, Any]:
        """
        Validate replicate counts per condition.

        Args:
            metadata: Sample metadata DataFrame
            groupby: Column name for grouping
            min_replicates: Minimum required replicates (default: 3)

        Returns:
            Dict with validation results
        """
        group_counts = metadata[groupby].value_counts().to_dict()

        validation = {
            "valid": True,
            "group_counts": group_counts,
            "warnings": [],
            "errors": [],
        }

        for group, count in group_counts.items():
            if count < min_replicates:
                validation["valid"] = False
                validation["errors"].append(
                    f"Group '{group}' has only {count} replicates "
                    f"(minimum {min_replicates} required for stable variance estimation)"
                )
            # SCIENTIFIC FIX: Add warning when n < 4
            elif count < 4:
                validation["warnings"].append(
                    f"Group '{group}' has {count} replicates. "
                    f"Statistical power may be limited. 4+ replicates recommended."
                )

        return validation

    # -------------------------
    # PSEUDOBULK TOOLS
    # -------------------------
    @tool
    def create_pseudobulk_matrix(
        modality_name: str,
        sample_col: str,
        celltype_col: str,
        layer: str = None,
        min_cells: int = 10,
        aggregation_method: str = "sum",
        min_genes: int = 200,
        filter_zeros: bool = True,
        save_result: bool = True,
    ) -> str:
        """
        Aggregate single-cell data to pseudobulk matrix for differential expression analysis.

        IMPORTANT: This tool extracts RAW COUNTS from adata.raw.X for DESeq2 compatibility.

        Args:
            modality_name: Name of single-cell modality to aggregate
            sample_col: Column containing sample identifiers
            celltype_col: Column containing cell type identifiers
            layer: Layer to use for aggregation (default: None, uses raw counts)
            min_cells: Minimum cells per sample-celltype combination
            aggregation_method: Aggregation method ('sum' for DESeq2, 'mean', 'median')
            min_genes: Minimum genes detected per pseudobulk sample
            filter_zeros: Remove genes with all zeros after aggregation
            save_result: Whether to save the pseudobulk modality
        """
        try:
            # Validate modality exists
            if modality_name not in data_manager.list_modalities():
                raise ModalityNotFoundError(
                    f"Modality '{modality_name}' not found. "
                    f"Available: {data_manager.list_modalities()}"
                )

            # Get the single-cell modality
            adata = data_manager.get_modality(modality_name)
            logger.info(
                f"Creating pseudobulk matrix from '{modality_name}': "
                f"{adata.n_obs} cells x {adata.n_vars} genes"
            )

            # Validate required columns exist
            if sample_col not in adata.obs.columns:
                available_cols = list(adata.obs.columns)[:10]
                raise PseudobulkError(
                    f"Sample column '{sample_col}' not found. "
                    f"Available columns: {available_cols}..."
                )

            if celltype_col not in adata.obs.columns:
                available_cols = list(adata.obs.columns)[:10]
                raise PseudobulkError(
                    f"Cell type column '{celltype_col}' not found. "
                    f"Available columns: {available_cols}..."
                )

            # Use pseudobulk service (it uses raw counts internally)
            pseudobulk_adata = pseudobulk_service.aggregate_to_pseudobulk(
                adata=adata,
                sample_col=sample_col,
                celltype_col=celltype_col,
                layer=layer,
                min_cells=min_cells,
                aggregation_method=aggregation_method,
                min_genes=min_genes,
                filter_zeros=filter_zeros,
            )

            # Save as new modality
            pseudobulk_modality_name = f"{modality_name}_pseudobulk"
            data_manager.modalities[pseudobulk_modality_name] = pseudobulk_adata

            # Save to file if requested
            if save_result:
                save_path = f"{modality_name}_pseudobulk.h5ad"
                data_manager.save_modality(pseudobulk_modality_name, save_path)

            # Get aggregation statistics
            agg_stats = pseudobulk_adata.uns.get("aggregation_stats", {})

            # Log the operation
            data_manager.log_tool_usage(
                tool_name="create_pseudobulk_matrix",
                parameters={
                    "modality_name": modality_name,
                    "sample_col": sample_col,
                    "celltype_col": celltype_col,
                    "aggregation_method": aggregation_method,
                    "min_cells": min_cells,
                },
                description=f"Created pseudobulk matrix: {pseudobulk_adata.n_obs} samples x {pseudobulk_adata.n_vars} genes",
            )

            # Format response
            response = f"""Pseudobulk matrix created from single-cell data '{modality_name}'!

**Aggregation Results:**
- Original: {adata.n_obs:,} single cells -> {pseudobulk_adata.n_obs} pseudobulk samples
- Genes retained: {pseudobulk_adata.n_vars:,} / {adata.n_vars:,} ({pseudobulk_adata.n_vars/adata.n_vars*100:.1f}%)
- Aggregation method: {aggregation_method}
- Min cells threshold: {min_cells}

**Sample & Cell Type Distribution:**
- Unique samples: {agg_stats.get('n_samples', 'N/A')}
- Cell types: {agg_stats.get('n_cell_types', 'N/A')}
- Total cells aggregated: {agg_stats.get('total_cells_aggregated', 'N/A'):,}
- Mean cells per pseudobulk: {agg_stats.get('mean_cells_per_pseudobulk', 0):.1f}

**New modality created**: '{pseudobulk_modality_name}'"""

            if save_result:
                response += f"\n**Saved to**: {save_path}"

            response += "\n\nNext step: Use 'prepare_differential_expression_design' to set up statistical design for DE analysis."

            analysis_results["details"]["pseudobulk_aggregation"] = response
            return response

        except (
            PseudobulkError,
            AggregationError,
            InsufficientCellsError,
            ModalityNotFoundError,
        ) as e:
            logger.error(f"Error creating pseudobulk matrix: {e}")
            return f"Error creating pseudobulk matrix: {str(e)}"
        except Exception as e:
            logger.error(f"Unexpected error in pseudobulk creation: {e}")
            return f"Unexpected error: {str(e)}"

    @tool
    def prepare_differential_expression_design(
        modality_name: str,
        formula: str,
        contrast: List[str],
        reference_condition: str = None,
    ) -> str:
        """
        Prepare design matrix and validate experimental design for differential expression analysis.

        CRITICAL: Validates minimum replicate requirements (3+ per condition).

        Args:
            modality_name: Name of pseudobulk modality
            formula: R-style formula (e.g., "~condition + batch")
            contrast: Contrast specification [factor, level1, level2]
            reference_condition: Reference level for main condition
        """
        try:
            # Validate modality exists
            if modality_name not in data_manager.list_modalities():
                raise ModalityNotFoundError(
                    f"Modality '{modality_name}' not found. "
                    f"Available: {data_manager.list_modalities()}"
                )

            # Get the pseudobulk modality
            adata = data_manager.get_modality(modality_name)
            logger.info(
                f"Preparing DE design for '{modality_name}': "
                f"{adata.n_obs} samples x {adata.n_vars} genes"
            )

            # SCIENTIFIC FIX: Validate with min_replicates=3
            design_validation = bulk_rnaseq_service.validate_experimental_design(
                metadata=adata.obs, formula=formula, min_replicates=3
            )

            if not design_validation["valid"]:
                error_msg = "; ".join(design_validation["errors"])
                raise DesignMatrixError(f"Invalid experimental design: {error_msg}")

            # Additional replicate validation
            condition_col = contrast[0]
            replicate_validation = _validate_replicate_counts(
                adata.obs, condition_col, min_replicates=3
            )

            if not replicate_validation["valid"]:
                error_msg = "; ".join(replicate_validation["errors"])
                raise InsufficientReplicatesError(error_msg)

            # Create design matrix
            design_result = bulk_rnaseq_service.create_formula_design(
                metadata=adata.obs,
                condition_col=contrast[0],
                reference_condition=reference_condition,
            )

            # Store design information in modality
            adata.uns["formula_design"] = {
                "formula": formula,
                "contrast": contrast,
                "design_matrix_info": design_result,
                "validation_results": design_validation,
            }

            # Update modality with design info
            data_manager.modalities[modality_name] = adata

            # Log the operation
            data_manager.log_tool_usage(
                tool_name="prepare_differential_expression_design",
                parameters={
                    "modality_name": modality_name,
                    "formula": formula,
                    "contrast": contrast,
                },
                description=f"Prepared DE design for {adata.n_obs} pseudobulk samples",
            )

            # Format response with warnings
            response = f"""Differential expression design prepared for '{modality_name}'!

**Experimental Design:**
- Formula: {formula}
- Contrast: {contrast[1]} vs {contrast[2]} in {contrast[0]}
- Design matrix: {design_result['design_matrix'].shape[0]} samples x {design_result['design_matrix'].shape[1]} coefficients
- Matrix rank: {design_result['rank']} (full rank: {'Yes' if design_result['rank'] == design_result['n_coefficients'] else 'No'})

**Design Validation:**
- Valid: {'Yes' if design_validation['valid'] else 'No'}
- Warnings: {len(design_validation['warnings'])} ({', '.join(design_validation['warnings'][:2]) if design_validation['warnings'] else 'None'})

**Replicate Counts:**"""

            for group, count in replicate_validation["group_counts"].items():
                status = (
                    "OK"
                    if count >= 4
                    else "LOW POWER" if count >= 3 else "INSUFFICIENT"
                )
                response += f"\n- {group}: {count} replicates ({status})"

            if replicate_validation["warnings"]:
                response += "\n\n**Warnings:**"
                for warning in replicate_validation["warnings"]:
                    response += f"\n- {warning}"

            response += (
                "\n\n**Design information stored in**: adata.uns['formula_design']"
            )
            response += "\n\nNext step: Run 'run_pseudobulk_differential_expression' to perform pyDESeq2 analysis."

            analysis_results["details"]["de_design"] = response
            return response

        except (
            DesignMatrixError,
            FormulaError,
            ModalityNotFoundError,
            InsufficientReplicatesError,
        ) as e:
            logger.error(f"Error preparing DE design: {e}")
            return f"Error preparing differential expression design: {str(e)}"
        except Exception as e:
            logger.error(f"Unexpected error in DE design preparation: {e}")
            return f"Unexpected error: {str(e)}"

    @tool
    def run_pseudobulk_differential_expression(
        modality_name: str,
        alpha: float = 0.05,
        shrink_lfc: bool = True,
        n_cpus: int = 1,
        save_result: bool = True,
    ) -> str:
        """
        Run pyDESeq2 differential expression analysis on pseudobulk data.

        CRITICAL: Uses raw counts from adata.raw.X for accurate DESeq2 results.

        Args:
            modality_name: Name of pseudobulk modality with design prepared
            alpha: Significance threshold for multiple testing
            shrink_lfc: Whether to apply log fold change shrinkage
            n_cpus: Number of CPUs for parallel processing
            save_result: Whether to save results
        """
        try:
            # Validate modality exists
            if modality_name not in data_manager.list_modalities():
                raise ModalityNotFoundError(
                    f"Modality '{modality_name}' not found. "
                    f"Available: {data_manager.list_modalities()}"
                )

            # Get the pseudobulk modality
            adata = data_manager.get_modality(modality_name)
            logger.info(
                f"Running pyDESeq2 DE analysis on '{modality_name}': "
                f"{adata.n_obs} samples x {adata.n_vars} genes"
            )

            # Validate design exists
            if "formula_design" not in adata.uns:
                raise PseudobulkError(
                    "No design matrix prepared. Run 'prepare_differential_expression_design' first."
                )

            design_info = adata.uns["formula_design"]
            formula = design_info["formula"]
            contrast = design_info["contrast"]

            # CRITICAL: Extract raw counts for DESeq2
            count_matrix, used_raw = _extract_raw_counts(adata)

            raw_warning = ""
            if not used_raw:
                raw_warning = (
                    "\n\n**WARNING**: Using adata.X instead of adata.raw.X. "
                    "DESeq2 requires raw counts for accurate results. "
                    "If your data is normalized, results may be inaccurate."
                )

            # Run pyDESeq2 analysis using bulk RNA-seq service
            results_df, analysis_stats = (
                bulk_rnaseq_service.run_pydeseq2_from_pseudobulk(
                    pseudobulk_adata=adata,
                    formula=formula,
                    contrast=contrast,
                    alpha=alpha,
                    shrink_lfc=shrink_lfc,
                    n_cpus=n_cpus,
                )
            )

            # Store results in modality
            contrast_name = f"{contrast[0]}_{contrast[1]}_vs_{contrast[2]}"
            adata.uns[f"de_results_{contrast_name}"] = {
                "results_df": results_df,
                "analysis_stats": analysis_stats,
                "parameters": {
                    "alpha": alpha,
                    "shrink_lfc": shrink_lfc,
                    "formula": formula,
                    "contrast": contrast,
                    "used_raw_counts": used_raw,
                },
            }

            # Update modality with results
            data_manager.modalities[modality_name] = adata

            # Save results if requested
            if save_result:
                results_path = f"{modality_name}_de_results.csv"
                results_df.to_csv(results_path)

                save_path = f"{modality_name}_with_de_results.h5ad"
                data_manager.save_modality(modality_name, save_path)

            # Log the operation
            data_manager.log_tool_usage(
                tool_name="run_pseudobulk_differential_expression",
                parameters={
                    "modality_name": modality_name,
                    "formula": formula,
                    "contrast": contrast,
                    "alpha": alpha,
                    "shrink_lfc": shrink_lfc,
                },
                description=f"pyDESeq2 analysis: {analysis_stats['n_significant_genes']} significant genes found",
            )

            # Format response
            response = f"""pyDESeq2 Differential Expression Analysis Complete for '{modality_name}'!

**pyDESeq2 Analysis Results:**
- Contrast: {contrast[1]} vs {contrast[2]} in {contrast[0]}
- Genes tested: {analysis_stats['n_genes_tested']:,}
- Significant genes: {analysis_stats['n_significant_genes']:,} (alpha={alpha})
- Upregulated: {analysis_stats['n_upregulated']:,}
- Downregulated: {analysis_stats['n_downregulated']:,}

**Top Differentially Expressed Genes:**
**Upregulated ({contrast[1]} > {contrast[2]}):**
{chr(10).join([f"- {gene}" for gene in analysis_stats['top_upregulated'][:5]])}

**Downregulated ({contrast[1]} < {contrast[2]}):**
{chr(10).join([f"- {gene}" for gene in analysis_stats['top_downregulated'][:5]])}

**Analysis Parameters:**
- Formula: {formula}
- LFC shrinkage: {'Yes' if shrink_lfc else 'No'}
- Parallel CPUs: {n_cpus}
- Significance threshold: {alpha}
- Used raw counts: {'Yes' if used_raw else 'No (WARNING)'}

**Results stored in**: adata.uns['de_results_{contrast_name}']{raw_warning}"""

            if save_result:
                response += f"\n**Saved to**: {results_path} & {save_path}"

            response += "\n\nNext steps: Visualize results with volcano plots or run pathway enrichment analysis."

            analysis_results["details"]["differential_expression"] = response
            return response

        except (PseudobulkError, ModalityNotFoundError) as e:
            logger.error(f"Error in pseudobulk DE analysis: {e}")
            return f"Error in pseudobulk differential expression analysis: {str(e)}"
        except Exception as e:
            logger.error(f"Unexpected error in pseudobulk DE analysis: {e}")
            return f"Unexpected error: {str(e)}"

    @tool
    def validate_experimental_design(
        modality_name: str,
        formula: str,
    ) -> str:
        """
        Validate experimental design for statistical power and balance.

        CRITICAL: Requires minimum 3 replicates per condition (changed from 2).
        Warns when any condition has fewer than 4 replicates.

        Args:
            modality_name: Name of bulk RNA-seq or pseudobulk modality
            formula: R-style formula to validate
        """
        try:
            # Validate modality exists
            if modality_name not in data_manager.list_modalities():
                raise ModalityNotFoundError(
                    f"Modality '{modality_name}' not found. "
                    f"Available: {data_manager.list_modalities()}"
                )

            # Get modality
            adata = data_manager.get_modality(modality_name)
            metadata = adata.obs

            # SCIENTIFIC FIX: Validate design with min_replicates=3
            validation_result = formula_service.validate_experimental_design(
                metadata, formula, min_replicates=3
            )

            # Format response
            response = f"## Experimental Design Validation: `{formula}`\n\n"
            response += f"**Modality**: `{modality_name}`\n"
            response += f"**Samples**: {len(metadata)}\n\n"

            # Overall status
            if validation_result["valid"]:
                response += "**Overall Status**: PASSED - Design is valid for DESeq2 analysis\n\n"
            else:
                response += (
                    "**Overall Status**: FAILED - Design has issues (see below)\n\n"
                )

            # Design summary
            if validation_result.get("design_summary"):
                response += f"### Group Balance\n"
                for factor, counts in validation_result["design_summary"].items():
                    response += f"**{factor}**:\n"
                    for level, count in counts.items():
                        # SCIENTIFIC FIX: Indicate status based on replicate count
                        if count < 3:
                            status = "INSUFFICIENT"
                        elif count < 4:
                            status = "LOW POWER"
                        else:
                            status = "OK"
                        response += f"  - {level}: {count} samples ({status})\n"
                response += "\n"

            # Warnings
            if validation_result.get("warnings"):
                response += f"### Warnings\n"
                for warning in validation_result["warnings"]:
                    response += f"- {warning}\n"
                response += "\n"

            # Errors
            if validation_result.get("errors"):
                response += f"### Errors\n"
                for error in validation_result["errors"]:
                    response += f"- {error}\n"
                response += "\n"

            # Replicate requirements note
            response += "### Replicate Requirements\n"
            response += "- **Minimum required**: 3 replicates per condition (for stable variance estimation)\n"
            response += "- **Recommended**: 4+ replicates per condition (for adequate statistical power)\n"
            response += "- **Ideal**: 6+ replicates per condition (for publication-quality results)\n\n"

            if validation_result["valid"]:
                response += "**Conclusion**: Design is ready for pyDESeq2 analysis\n"
            else:
                response += (
                    "**Conclusion**: Please address issues before running analysis\n"
                )

            return response

        except ModalityNotFoundError as e:
            logger.error(f"Modality not found error: {e}")
            return f"Error: {str(e)}"
        except Exception as e:
            logger.error(f"Error validating experimental design: {e}")
            return f"Error: {str(e)}"

    @tool
    def suggest_formula_for_design(
        pseudobulk_modality: str,
        analysis_goal: Optional[str] = None,
        show_metadata_summary: bool = True,
    ) -> str:
        """
        Analyze metadata and suggest appropriate formulas for differential expression analysis.

        The agent examines the pseudobulk metadata structure and suggests 2-3 formula options
        based on available variables, explaining each in plain language with pros/cons.

        Args:
            pseudobulk_modality: Name of pseudobulk modality to analyze
            analysis_goal: Optional description of analysis goals
            show_metadata_summary: Whether to show detailed metadata summary
        """
        try:
            # Validate modality exists
            if pseudobulk_modality not in data_manager.list_modalities():
                return f"Modality '{pseudobulk_modality}' not found. Available: {data_manager.list_modalities()}"

            # Get the pseudobulk data
            adata = data_manager.get_modality(pseudobulk_modality)
            metadata = adata.obs
            n_samples = len(metadata)

            # Identify variable types and characteristics
            variable_analysis = {}
            for col in metadata.columns:
                if col.startswith("_") or col in ["n_cells", "total_counts"]:
                    continue

                series = metadata[col]
                if pd.api.types.is_numeric_dtype(series):
                    var_type = "continuous"
                    unique_vals = len(series.unique())
                    missing = series.isna().sum()
                else:
                    var_type = "categorical"
                    unique_vals = len(series.unique())
                    missing = series.isna().sum()

                variable_analysis[col] = {
                    "type": var_type,
                    "unique_values": unique_vals,
                    "missing_count": missing,
                    "sample_values": list(series.unique())[:5],
                }

            # Generate formula suggestions
            suggestions = []
            categorical_vars = [
                col
                for col, info in variable_analysis.items()
                if info["type"] == "categorical"
                and info["unique_values"] > 1
                and info["unique_values"] < n_samples / 2
            ]
            continuous_vars = [
                col
                for col, info in variable_analysis.items()
                if info["type"] == "continuous"
                and info["missing_count"] < n_samples / 2
            ]

            # Identify potential main condition and batch variables
            main_condition = None
            batch_vars = []

            for col in categorical_vars:
                unique_count = variable_analysis[col]["unique_values"]
                if unique_count == 2 and not main_condition:
                    main_condition = col
                elif col.lower() in ["batch", "sample", "donor", "patient", "subject"]:
                    batch_vars.append(col)
                elif unique_count > 2 and unique_count <= 6:
                    batch_vars.append(col)

            if main_condition:
                # Simple comparison
                suggestions.append(
                    {
                        "formula": f"~{main_condition}",
                        "complexity": "Simple",
                        "description": f"Compare {main_condition} groups directly",
                        "pros": [
                            "Maximum statistical power",
                            "Straightforward interpretation",
                            "Robust with small sample sizes",
                        ],
                        "cons": [
                            "Ignores potential confounders",
                            "May miss batch effects",
                        ],
                        "recommended_for": "Initial exploratory analysis or when confounders are minimal",
                        # SCIENTIFIC FIX: Changed from 6 to reflect min 3 per group
                        "min_samples": 6,  # 3 per group minimum
                    }
                )

                # Batch-corrected if batch variables available
                if batch_vars:
                    primary_batch = batch_vars[0]
                    suggestions.append(
                        {
                            "formula": f"~{main_condition} + {primary_batch}",
                            "complexity": "Batch-corrected",
                            "description": f"Compare {main_condition} while accounting for {primary_batch} effects",
                            "pros": [
                                "Controls for technical/batch variation",
                                "More reliable effect estimates",
                            ],
                            "cons": [
                                "Reduces degrees of freedom",
                                "Requires balanced design",
                            ],
                            "recommended_for": "Multi-batch experiments or when batch effects are suspected",
                            "min_samples": 8,
                        }
                    )

                # Full model with multiple covariates
                if len(batch_vars) > 1 or continuous_vars:
                    covariates = batch_vars[:2] + continuous_vars[:1]
                    formula_terms = [main_condition] + covariates
                    suggestions.append(
                        {
                            "formula": f"~{' + '.join(formula_terms)}",
                            "complexity": "Multi-factor",
                            "description": f"Comprehensive model accounting for {main_condition} and {len(covariates)} covariates",
                            "pros": [
                                "Controls for multiple confounders",
                                "Publication-ready analysis",
                                "Robust effect estimates",
                            ],
                            "cons": [
                                "Requires larger sample size",
                                "More complex interpretation",
                                "Risk of overfitting",
                            ],
                            "recommended_for": "Final analysis with adequate sample size and multiple known confounders",
                            "min_samples": max(
                                12, len(formula_terms) * 4
                            ),  # SCIENTIFIC FIX: Require more samples
                        }
                    )

            # Build response
            response = f"## Formula Design Analysis for '{pseudobulk_modality}'\n\n"

            if show_metadata_summary:
                response += "**Metadata Summary:**\n"
                response += f"- Samples: {n_samples}\n"
                response += f"- Variables analyzed: {len(variable_analysis)}\n"
                response += f"- Categorical variables: {len(categorical_vars)}\n"
                response += f"- Continuous variables: {len(continuous_vars)}\n\n"

                response += "**Key Variables:**\n"
                for col, info in list(variable_analysis.items())[:6]:
                    if col in categorical_vars + continuous_vars:
                        response += f"- **{col}**: {info['type']}, {info['unique_values']} levels"
                        if info["type"] == "categorical":
                            response += (
                                f" ({', '.join(map(str, info['sample_values']))})"
                            )
                        response += "\n"
                response += "\n"

            if analysis_goal:
                response += f"**Analysis Goal**: {analysis_goal}\n\n"

            if suggestions:
                response += "## Recommended Formula Options:\n\n"
                for i, suggestion in enumerate(suggestions, 1):
                    response += f"**{i}. {suggestion['complexity']} Model** *(recommended for {suggestion['recommended_for']})*\n"
                    response += f"   Formula: `{suggestion['formula']}`\n"
                    response += f"   Description: {suggestion['description']}\n"
                    response += f"   Pros: {', '.join(suggestion['pros'][:2])}\n"
                    response += f"   Cons: {', '.join(suggestion['cons'][:2])}\n"
                    response += (
                        f"   Min samples needed: {suggestion['min_samples']}\n\n"
                    )

                response += "**Recommendation**: Start with the simple model for exploration, then use the batch-corrected model if you see batch effects.\n\n"
                response += "**Next step**: Use `construct_de_formula_interactive` to build and validate your chosen formula."
            else:
                response += (
                    "**No suitable variables found for standard DE analysis.**\n"
                )
                response += "Please ensure your pseudobulk data has:\n"
                response += "- At least one categorical variable with 2+ levels (main condition)\n"
                response += "- Sufficient samples per group (minimum 3-4 replicates)\n"
                response += "- Proper metadata annotation\n\n"
                response += f"Available variables: {list(variable_analysis.keys())}"

            # Log the operation
            data_manager.log_tool_usage(
                tool_name="suggest_formula_for_design",
                parameters={
                    "pseudobulk_modality": pseudobulk_modality,
                    "analysis_goal": analysis_goal,
                    "n_suggestions": len(suggestions),
                },
                description=f"Generated {len(suggestions)} formula suggestions for {pseudobulk_modality}",
            )

            return response

        except Exception as e:
            logger.error(f"Error suggesting formulas: {e}")
            return f"Error analyzing design for formula suggestions: {str(e)}"

    @tool
    def construct_de_formula_interactive(
        pseudobulk_modality: str,
        main_variable: str,
        covariates: Optional[List[str]] = None,
        include_interactions: bool = False,
        validate_design: bool = True,
    ) -> str:
        """
        Build DE formula step-by-step with validation and preview.

        Constructs R-style formula, validates against metadata, shows design matrix preview,
        and provides warnings about potential statistical issues.

        Args:
            pseudobulk_modality: Name of pseudobulk modality
            main_variable: Primary variable of interest (main comparison)
            covariates: List of covariate variables to include
            include_interactions: Whether to include interaction terms
            validate_design: Whether to validate the experimental design
        """
        try:
            # Validate modality exists
            if pseudobulk_modality not in data_manager.list_modalities():
                return f"Modality '{pseudobulk_modality}' not found. Available: {data_manager.list_modalities()}"

            # Get the pseudobulk data
            adata = data_manager.get_modality(pseudobulk_modality)
            metadata = adata.obs

            # Validate main variable
            if main_variable not in metadata.columns:
                available_vars = [
                    col for col in metadata.columns if not col.startswith("_")
                ]
                return f"Main variable '{main_variable}' not found. Available: {available_vars}"

            # Build formula
            formula_terms = [main_variable]
            if covariates:
                missing_covariates = [
                    c for c in covariates if c not in metadata.columns
                ]
                if missing_covariates:
                    return f"Covariates not found: {missing_covariates}"
                formula_terms.extend(covariates)

            # Construct basic formula
            if include_interactions and covariates:
                interaction_term = f"{main_variable}*{covariates[0]}"
                formula = f"~{interaction_term}"
                if len(covariates) > 1:
                    formula += f" + {' + '.join(covariates[1:])}"
            else:
                formula = f"~{' + '.join(formula_terms)}"

            # Parse and validate formula
            try:
                formula_components = formula_service.parse_formula(formula, metadata)
                design_result = formula_service.construct_design_matrix(
                    formula_components, metadata
                )

                # Format response
                response = (
                    f"## Formula Construction Complete for '{pseudobulk_modality}'\n\n"
                )
                response += f"**Constructed Formula**: `{formula}`\n\n"

                response += "**Formula Components:**\n"
                response += f"- Main variable: {main_variable} ({formula_components['variable_info'][main_variable]['type']})\n"
                if covariates:
                    response += f"- Covariates: {', '.join(covariates)}\n"
                if include_interactions:
                    response += f"- Interactions: Yes (between {main_variable} and {covariates[0] if covariates else 'none'})\n"
                response += (
                    f"- Total terms: {len(formula_components['predictor_terms'])}\n\n"
                )

                # Design matrix preview
                response += "**Design Matrix Preview**:\n"
                response += f"- Dimensions: {design_result['design_matrix'].shape[0]} samples x {design_result['design_matrix'].shape[1]} coefficients\n"
                response += f"- Matrix rank: {design_result['rank']} (full rank: {'Yes' if design_result['rank'] == design_result['n_coefficients'] else 'WARNING'})\n"
                response += f"- Coefficient names: {', '.join(design_result['coefficient_names'][:5])}{'...' if len(design_result['coefficient_names']) > 5 else ''}\n\n"

                # Variable information
                response += "**Variable Details:**\n"
                for var, info in formula_components["variable_info"].items():
                    if info["type"] == "categorical":
                        response += f"- **{var}**: {info['n_levels']} levels, reference = '{info['reference_level']}'\n"
                    else:
                        response += f"- **{var}**: continuous variable\n"
                response += "\n"

                if validate_design:
                    # SCIENTIFIC FIX: Validate with min_replicates=3
                    validation = formula_service.validate_experimental_design(
                        metadata, formula, min_replicates=3
                    )

                    response += "**Design Validation**:\n"
                    response += (
                        f"- Valid design: {'Yes' if validation['valid'] else 'No'}\n"
                    )

                    if validation["warnings"]:
                        response += f"- Warnings ({len(validation['warnings'])}):\n"
                        for warning in validation["warnings"][:3]:
                            response += f"  - {warning}\n"

                    if validation["errors"]:
                        response += f"- Errors ({len(validation['errors'])}):\n"
                        for error in validation["errors"]:
                            response += f"  - {error}\n"

                    response += "\n**Sample Distribution:**\n"
                    for var, counts in validation.get("design_summary", {}).items():
                        response += f"- **{var}**: {dict(list(counts.items())[:4])}\n"

                response += "\n**Recommendations**:\n"
                if design_result["rank"] < design_result["n_coefficients"]:
                    response += "- WARNING: Design matrix is rank deficient - consider removing correlated variables\n"
                if validation.get("warnings"):
                    response += "- Review warnings above before proceeding\n"
                else:
                    response += "- Design looks good! Ready for differential expression analysis\n"

                response += "\n**Next step**: Use `run_differential_expression_with_formula` to execute the analysis."

                # Store formula in modality for later use
                adata.uns["constructed_formula"] = {
                    "formula": formula,
                    "main_variable": main_variable,
                    "covariates": covariates,
                    "include_interactions": include_interactions,
                    "formula_components": formula_components,
                    "design_result": design_result,
                    "validation": validation if validate_design else None,
                }
                data_manager.modalities[pseudobulk_modality] = adata

            except (FormulaError, DesignMatrixError) as e:
                response = "## Formula Construction Failed\n\n"
                response += f"**Formula**: `{formula}`\n"
                response += f"**Error**: {str(e)}\n\n"
                response += "**Suggestions**:\n"
                response += "- Check variable names are spelled correctly\n"
                response += "- Ensure variables have multiple levels (for categorical) or variation (for continuous)\n"
                response += "- Reduce model complexity if you have limited samples\n"
                response += f"- Available variables: {list(metadata.columns)[:10]}"
                return response

            # Log the operation
            data_manager.log_tool_usage(
                tool_name="construct_de_formula_interactive",
                parameters={
                    "pseudobulk_modality": pseudobulk_modality,
                    "formula": formula,
                    "main_variable": main_variable,
                    "covariates": covariates,
                    "include_interactions": include_interactions,
                },
                description=f"Constructed and validated formula: {formula}",
            )

            return response

        except Exception as e:
            logger.error(f"Error constructing formula: {e}")
            return f"Error in formula construction: {str(e)}"

    @tool
    def run_differential_expression_with_formula(
        pseudobulk_modality: str,
        formula: Optional[str] = None,
        contrast: Optional[List[str]] = None,
        reference_levels: Optional[dict] = None,
        alpha: float = 0.05,
        lfc_threshold: float = 0.0,
        save_results: bool = True,
    ) -> str:
        """
        Execute differential expression analysis with agent-guided formula.

        Uses pyDESeq2 for analysis with RAW COUNTS from adata.raw.X.

        Args:
            pseudobulk_modality: Name of pseudobulk modality
            formula: R-style formula (uses stored formula if None)
            contrast: Contrast specification [factor, level1, level2]
            reference_levels: Reference levels for categorical variables
            alpha: Significance threshold for adjusted p-values
            lfc_threshold: Log fold change threshold
            save_results: Whether to save results to files
        """
        try:
            # Validate modality exists
            if pseudobulk_modality not in data_manager.list_modalities():
                return f"Modality '{pseudobulk_modality}' not found. Available: {data_manager.list_modalities()}"

            # Get the pseudobulk data
            adata = data_manager.get_modality(pseudobulk_modality)

            # Use stored formula if none provided
            if formula is None:
                if "constructed_formula" in adata.uns:
                    formula = adata.uns["constructed_formula"]["formula"]
                    stored_info = adata.uns["constructed_formula"]
                    response_prefix = (
                        "Using stored formula from interactive construction:\n"
                    )
                else:
                    return "No formula provided and no stored formula found. Use `construct_de_formula_interactive` first or provide a formula."
            else:
                response_prefix = "Using provided formula:\n"
                stored_info = None

            # Auto-detect contrast if not provided
            if contrast is None and stored_info:
                main_var = stored_info["main_variable"]
                levels = list(adata.obs[main_var].unique())
                if len(levels) == 2:
                    contrast = [main_var, str(levels[1]), str(levels[0])]
                    response_prefix += (
                        f"Auto-detected contrast: {contrast[1]} vs {contrast[2]}\n"
                    )
                else:
                    return f"Multiple levels found for {main_var}: {levels}. Please specify contrast as [factor, level1, level2]."
            elif contrast is None:
                return "No contrast specified. Please provide contrast as [factor, level1, level2]."

            logger.info(
                f"Running DE analysis on '{pseudobulk_modality}' with formula: {formula}"
            )

            # SCIENTIFIC FIX: Validate design with min_replicates=3
            design_validation = bulk_rnaseq_service.validate_experimental_design(
                metadata=adata.obs, formula=formula, min_replicates=3
            )

            if not design_validation["valid"]:
                error_msgs = "; ".join(design_validation["errors"])
                return f"**Invalid experimental design**: {error_msgs}\n\nUse `construct_de_formula_interactive` to debug the design."

            # Create design matrix
            condition_col = contrast[0]
            reference_condition = (
                reference_levels.get(condition_col) if reference_levels else None
            )

            design_result = bulk_rnaseq_service.create_formula_design(
                metadata=adata.obs,
                condition_col=condition_col,
                reference_condition=reference_condition,
            )

            # Store design information
            adata.uns["de_formula_design"] = {
                "formula": formula,
                "contrast": contrast,
                "design_matrix_info": design_result,
                "validation_results": design_validation,
                "reference_levels": reference_levels,
            }

            # Run pyDESeq2 analysis
            results_df, analysis_stats = (
                bulk_rnaseq_service.run_pydeseq2_from_pseudobulk(
                    pseudobulk_adata=adata,
                    formula=formula,
                    contrast=contrast,
                    alpha=alpha,
                    shrink_lfc=True,
                    n_cpus=1,
                )
            )

            # Filter by LFC threshold if specified
            if lfc_threshold > 0:
                significant_mask = (results_df["padj"] < alpha) & (
                    abs(results_df["log2FoldChange"]) >= lfc_threshold
                )
                n_lfc_filtered = significant_mask.sum()
            else:
                n_lfc_filtered = analysis_stats["n_significant_genes"]

            # Store results in modality
            contrast_name = f"{contrast[0]}_{contrast[1]}_vs_{contrast[2]}"
            adata.uns[f"de_results_formula_{contrast_name}"] = {
                "results_df": results_df,
                "analysis_stats": analysis_stats,
                "parameters": {
                    "formula": formula,
                    "contrast": contrast,
                    "alpha": alpha,
                    "lfc_threshold": lfc_threshold,
                    "reference_levels": reference_levels,
                },
            }

            # Update modality
            data_manager.modalities[pseudobulk_modality] = adata

            # Save results if requested
            if save_results:
                results_path = f"{pseudobulk_modality}_formula_de_results.csv"
                results_df.to_csv(results_path)

                modality_path = f"{pseudobulk_modality}_with_formula_results.h5ad"
                data_manager.save_modality(pseudobulk_modality, modality_path)

            # Format response
            response = "## Differential Expression Analysis Complete\n\n"
            response += response_prefix
            response += f"**Formula**: `{formula}`\n"
            response += (
                f"**Contrast**: {contrast[1]} vs {contrast[2]} (in {contrast[0]})\n\n"
            )

            response += "**Results Summary**:\n"
            response += f"- Genes tested: {analysis_stats['n_genes_tested']:,}\n"
            response += f"- Significant genes (FDR < {alpha}): {analysis_stats['n_significant_genes']:,}\n"
            if lfc_threshold > 0:
                response += (
                    f"- Significant + |LFC| >= {lfc_threshold}: {n_lfc_filtered:,}\n"
                )
            response += f"- Upregulated ({contrast[1]} > {contrast[2]}): {analysis_stats['n_upregulated']:,}\n"
            response += f"- Downregulated ({contrast[1]} < {contrast[2]}): {analysis_stats['n_downregulated']:,}\n\n"

            response += "**Top Differentially Expressed Genes**:\n"
            response += "**Most Upregulated**:\n"
            for gene in analysis_stats["top_upregulated"][:5]:
                gene_data = results_df.loc[gene]
                response += f"- {gene}: LFC = {gene_data['log2FoldChange']:.2f}, FDR = {gene_data['padj']:.2e}\n"

            response += "\n**Most Downregulated**:\n"
            for gene in analysis_stats["top_downregulated"][:5]:
                gene_data = results_df.loc[gene]
                response += f"- {gene}: LFC = {gene_data['log2FoldChange']:.2f}, FDR = {gene_data['padj']:.2e}\n"

            response += f"\n**Results Storage**:\n"
            response += (
                f"- Stored in: adata.uns['de_results_formula_{contrast_name}']\n"
            )
            if save_results:
                response += f"- CSV file: {results_path}\n"
                response += f"- H5AD file: {modality_path}\n"

            response += "\n**Next steps**: Use `iterate_de_analysis` to try different formulas or `compare_de_iterations` to compare results."

            # Log the operation
            data_manager.log_tool_usage(
                tool_name="run_differential_expression_with_formula",
                parameters={
                    "pseudobulk_modality": pseudobulk_modality,
                    "formula": formula,
                    "contrast": contrast,
                    "alpha": alpha,
                    "lfc_threshold": lfc_threshold,
                },
                description=f"Formula-based DE analysis: {analysis_stats['n_significant_genes']} significant genes",
            )

            return response

        except Exception as e:
            logger.error(f"Error in formula-based DE analysis: {e}")
            return f"Error running differential expression with formula: {str(e)}"

    @tool
    def run_differential_expression_analysis(
        modality_name: str,
        groupby: str,
        group1: str,
        group2: str,
        method: str = "deseq2_like",
        min_expression_threshold: float = 1.0,
        min_fold_change: float = 1.5,
        min_pct_expressed: float = 0.1,
        max_out_pct_expressed: float = 0.5,
        save_result: bool = True,
    ) -> str:
        """
        Run differential expression analysis between two groups.

        CRITICAL: Uses raw counts from adata.raw.X for DESeq2 accuracy.
        Validates minimum replicate requirements (3+ per group).

        Args:
            modality_name: Name of the bulk RNA-seq or pseudobulk modality
            groupby: Column name for grouping (e.g., 'condition', 'treatment')
            group1: First group for comparison (e.g., 'control')
            group2: Second group for comparison (e.g., 'treatment')
            method: Analysis method ('deseq2_like', 'wilcoxon', 't_test')
            min_expression_threshold: Minimum expression threshold for gene filtering
            min_fold_change: Minimum fold-change threshold for biological significance (default: 1.5)
            min_pct_expressed: Minimum fraction expressing in group1 (default: 0.1)
            max_out_pct_expressed: Maximum fraction expressing in group2 (default: 0.5)
            save_result: Whether to save the results
        """
        try:
            # Validate modality exists
            if modality_name not in data_manager.list_modalities():
                raise ModalityNotFoundError(
                    f"Modality '{modality_name}' not found. "
                    f"Available: {data_manager.list_modalities()}"
                )

            # Get the modality
            adata = data_manager.get_modality(modality_name)
            logger.info(
                f"Running DE analysis on '{modality_name}': "
                f"{adata.n_obs} samples x {adata.n_vars} genes"
            )

            # Validate experimental design
            if groupby not in adata.obs.columns:
                available_columns = [
                    col
                    for col in adata.obs.columns
                    if col.lower() in ["condition", "treatment", "group", "batch"]
                ]
                return f"Grouping column '{groupby}' not found. Available experimental design columns: {available_columns}"

            # Check if groups exist
            available_groups = list(adata.obs[groupby].unique())
            if group1 not in available_groups:
                return f"Group '{group1}' not found in column '{groupby}'. Available groups: {available_groups}"
            if group2 not in available_groups:
                return f"Group '{group2}' not found in column '{groupby}'. Available groups: {available_groups}"

            # SCIENTIFIC FIX: Validate replicate counts with min_replicates=3
            replicate_validation = _validate_replicate_counts(
                adata.obs, groupby, min_replicates=3
            )

            if not replicate_validation["valid"]:
                error_msg = "; ".join(replicate_validation["errors"])
                return f"**Insufficient replicates**: {error_msg}\n\nMinimum 3 replicates per group required for stable variance estimation."

            # Get group counts for response
            group1_count = (adata.obs[groupby] == group1).sum()
            group2_count = (adata.obs[groupby] == group2).sum()

            # Use bulk service for differential expression
            adata_de, de_stats, ir = (
                bulk_rnaseq_service.run_differential_expression_analysis(
                    adata=adata,
                    groupby=groupby,
                    group1=group1,
                    group2=group2,
                    method=method,
                    min_expression_threshold=min_expression_threshold,
                    min_fold_change=min_fold_change,
                    min_pct_expressed=min_pct_expressed,
                    max_out_pct_expressed=max_out_pct_expressed,
                )
            )

            # Save as new modality
            de_modality_name = f"{modality_name}_de_{group1}_vs_{group2}"
            data_manager.modalities[de_modality_name] = adata_de

            # Save to file if requested
            if save_result:
                save_path = f"{modality_name}_de_{group1}_vs_{group2}.h5ad"
                data_manager.save_modality(de_modality_name, save_path)

            # Log the operation with IR for provenance tracking
            data_manager.log_tool_usage(
                tool_name="run_differential_expression_analysis",
                parameters={
                    "modality_name": modality_name,
                    "groupby": groupby,
                    "group1": group1,
                    "group2": group2,
                    "method": method,
                    "min_expression_threshold": min_expression_threshold,
                    "min_fold_change": min_fold_change,
                    "min_pct_expressed": min_pct_expressed,
                    "max_out_pct_expressed": max_out_pct_expressed,
                },
                description=f"DE analysis: {de_stats['n_significant_genes']} significant genes found",
                ir=ir,
            )

            # Format response
            response = f"## Differential Expression Analysis Complete for '{modality_name}'\n\n"
            response += f"**Analysis Results:**\n"
            response += f"- Comparison: {group1} ({group1_count} samples) vs {group2} ({group2_count} samples)\n"
            response += f"- Method: {de_stats['method']}\n"
            response += f"- Genes tested: {de_stats['n_genes_tested']:,}\n"
            response += f"- Significant genes (padj < 0.05): {de_stats['n_significant_genes']:,}\n\n"

            response += "**Differential Expression Summary:**\n"
            response += (
                f"- Upregulated in {group2}: {de_stats['n_upregulated']} genes\n"
            )
            response += (
                f"- Downregulated in {group2}: {de_stats['n_downregulated']} genes\n\n"
            )

            response += "**Top Upregulated Genes:**\n"
            for gene in de_stats["top_upregulated"][:5]:
                response += f"- {gene}\n"

            response += "\n**Top Downregulated Genes:**\n"
            for gene in de_stats["top_downregulated"][:5]:
                response += f"- {gene}\n"

            # Add replicate warnings if any
            if replicate_validation["warnings"]:
                response += "\n**Statistical Power Warnings:**\n"
                for warning in replicate_validation["warnings"]:
                    response += f"- {warning}\n"

            response += f"\n**New modality created**: '{de_modality_name}'"

            if save_result:
                response += f"\n**Saved to**: {save_path}"

            response += f"\n**Access detailed results**: adata.uns['{de_stats['de_results_key']}']\n"
            response += "\nUse the significant genes for pathway enrichment analysis or gene set analysis."

            analysis_results["details"]["differential_expression"] = response
            return response

        except (BulkRNASeqError, ModalityNotFoundError) as e:
            logger.error(f"Error in differential expression analysis: {e}")
            return f"Error running differential expression analysis: {str(e)}"
        except Exception as e:
            logger.error(f"Unexpected error in differential expression: {e}")
            return f"Unexpected error: {str(e)}"

    @tool
    def iterate_de_analysis(
        pseudobulk_modality: str,
        new_formula: Optional[str] = None,
        new_contrast: Optional[List[str]] = None,
        filter_criteria: Optional[dict] = None,
        compare_to_previous: bool = True,
        iteration_name: Optional[str] = None,
    ) -> str:
        """
        Support iterative analysis with formula/filter changes.

        Enables trying different formulas or filters, tracking iterations,
        and comparing results between analyses.

        Args:
            pseudobulk_modality: Name of pseudobulk modality
            new_formula: New formula to try (if None, modify existing)
            new_contrast: New contrast to test
            filter_criteria: Additional filtering criteria (e.g., {'min_lfc': 0.5})
            compare_to_previous: Whether to compare with previous iteration
            iteration_name: Custom name for this iteration
        """
        try:
            # Validate modality exists
            if pseudobulk_modality not in data_manager.list_modalities():
                return f"Modality '{pseudobulk_modality}' not found. Available: {data_manager.list_modalities()}"

            # Get the pseudobulk data
            adata = data_manager.get_modality(pseudobulk_modality)

            # Initialize iteration tracking if not exists
            if "de_iterations" not in adata.uns:
                adata.uns["de_iterations"] = {"iterations": [], "current_iteration": 0}

            iteration_tracker = adata.uns["de_iterations"]
            current_iter = iteration_tracker["current_iteration"] + 1

            # Determine iteration name
            if iteration_name is None:
                iteration_name = f"iteration_{current_iter}"

            # Get previous results for comparison
            previous_results = None
            previous_iteration = None
            if compare_to_previous and iteration_tracker["iterations"]:
                previous_iteration = iteration_tracker["iterations"][-1]
                prev_key = f"de_results_formula_{previous_iteration['contrast_name']}"
                if prev_key in adata.uns:
                    previous_results = adata.uns[prev_key]["results_df"]

            # Use existing formula/contrast if not provided
            if new_formula is None or new_contrast is None:
                if "de_formula_design" in adata.uns:
                    if new_formula is None:
                        new_formula = adata.uns["de_formula_design"]["formula"]
                    if new_contrast is None:
                        new_contrast = adata.uns["de_formula_design"]["contrast"]
                else:
                    return "No previous formula/contrast found and none provided. Run a DE analysis first."

            logger.info(
                f"Starting DE iteration '{iteration_name}' on '{pseudobulk_modality}'"
            )

            # Run the analysis with new parameters
            run_result = run_differential_expression_with_formula(
                pseudobulk_modality=pseudobulk_modality,
                formula=new_formula,
                contrast=new_contrast,
                alpha=0.05,
                lfc_threshold=(
                    filter_criteria.get("min_lfc", 0.0) if filter_criteria else 0.0
                ),
                save_results=False,
            )

            if "Error" in run_result:
                return f"Error in iteration '{iteration_name}': {run_result}"

            # Get current results
            contrast_name = f"{new_contrast[0]}_{new_contrast[1]}_vs_{new_contrast[2]}"
            current_key = f"de_results_formula_{contrast_name}"

            if current_key not in adata.uns:
                return "Results not found after analysis. Analysis may have failed."

            current_results = adata.uns[current_key]["results_df"]
            current_stats = adata.uns[current_key]["analysis_stats"]

            # Store iteration information
            iteration_info = {
                "name": iteration_name,
                "formula": new_formula,
                "contrast": new_contrast,
                "contrast_name": contrast_name,
                "n_significant": current_stats["n_significant_genes"],
                "timestamp": pd.Timestamp.now().isoformat(),
                "filter_criteria": filter_criteria or {},
            }

            # Compare with previous if requested
            comparison_results = None
            if compare_to_previous and previous_results is not None:
                current_sig = set(current_results[current_results["padj"] < 0.05].index)
                previous_sig = set(
                    previous_results[previous_results["padj"] < 0.05].index
                )

                overlap = len(current_sig & previous_sig)
                current_only = len(current_sig - previous_sig)
                previous_only = len(previous_sig - current_sig)

                common_genes = list(current_sig & previous_sig)
                if len(common_genes) > 3:
                    current_lfc = current_results.loc[common_genes, "log2FoldChange"]
                    previous_lfc = previous_results.loc[common_genes, "log2FoldChange"]
                    correlation = current_lfc.corr(previous_lfc)
                else:
                    correlation = None

                comparison_results = {
                    "overlap_genes": overlap,
                    "current_only": current_only,
                    "previous_only": previous_only,
                    "correlation": correlation,
                }

                iteration_info["comparison"] = comparison_results

            # Update iteration tracking
            iteration_tracker["iterations"].append(iteration_info)
            iteration_tracker["current_iteration"] = current_iter
            adata.uns["de_iterations"] = iteration_tracker

            # Update modality
            data_manager.modalities[pseudobulk_modality] = adata

            # Format response
            response = f"## DE Analysis Iteration '{iteration_name}' Complete\n\n"
            response += f"**Formula**: `{new_formula}`\n"
            response += f"**Contrast**: {new_contrast[1]} vs {new_contrast[2]} (in {new_contrast[0]})\n\n"

            response += "**Current Results**:\n"
            response += (
                f"- Significant genes: {current_stats['n_significant_genes']:,}\n"
            )
            response += f"- Upregulated: {current_stats['n_upregulated']:,}\n"
            response += f"- Downregulated: {current_stats['n_downregulated']:,}\n"

            if comparison_results:
                response += "\n**Comparison with Previous Iteration**:\n"
                response += f"- Overlapping significant genes: {comparison_results['overlap_genes']:,}\n"
                response += (
                    f"- New in current: {comparison_results['current_only']:,}\n"
                )
                response += (
                    f"- Lost from previous: {comparison_results['previous_only']:,}\n"
                )
                if comparison_results["correlation"] is not None:
                    response += f"- Fold change correlation: {comparison_results['correlation']:.3f}\n"

            response += f"\n**Iteration Summary**:\n"
            response += f"- Total iterations: {len(iteration_tracker['iterations'])}\n"
            response += f"- Current iteration: {current_iter}\n"

            response += f"\n**Results stored in**: adata.uns['de_results_formula_{contrast_name}']\n"
            response += "**Iteration tracking**: adata.uns['de_iterations']\n"

            response += "\n**Next steps**: Use `compare_de_iterations` to compare all iterations or continue iterating with different parameters."

            # Log the operation
            data_manager.log_tool_usage(
                tool_name="iterate_de_analysis",
                parameters={
                    "pseudobulk_modality": pseudobulk_modality,
                    "iteration_name": iteration_name,
                    "formula": new_formula,
                    "contrast": new_contrast,
                    "compare_to_previous": compare_to_previous,
                },
                description=f"DE iteration {current_iter}: {current_stats['n_significant_genes']} significant genes",
            )

            return response

        except Exception as e:
            logger.error(f"Error in DE iteration: {e}")
            return f"Error in DE analysis iteration: {str(e)}"

    @tool
    def compare_de_iterations(
        pseudobulk_modality: str,
        iteration_1: Optional[str] = None,
        iteration_2: Optional[str] = None,
        show_overlap: bool = True,
        show_unique: bool = True,
        save_comparison: bool = True,
    ) -> str:
        """
        Compare results between different DE analysis iterations.

        Shows overlapping and unique DEGs, correlation of fold changes, and helps
        users understand impact of formula changes.

        Args:
            pseudobulk_modality: Name of pseudobulk modality
            iteration_1: Name of first iteration (latest if None)
            iteration_2: Name of second iteration (second latest if None)
            show_overlap: Whether to show overlapping genes
            show_unique: Whether to show unique genes per iteration
            save_comparison: Whether to save comparison results
        """
        try:
            # Validate modality exists
            if pseudobulk_modality not in data_manager.list_modalities():
                return f"Modality '{pseudobulk_modality}' not found. Available: {data_manager.list_modalities()}"

            # Get the pseudobulk data
            adata = data_manager.get_modality(pseudobulk_modality)

            # Check iteration tracking exists
            if "de_iterations" not in adata.uns:
                return "No iteration tracking found. Run `iterate_de_analysis` first to create iterations."

            iteration_tracker = adata.uns["de_iterations"]
            iterations = iteration_tracker["iterations"]

            if len(iterations) < 2:
                return f"Only {len(iterations)} iteration(s) available. Need at least 2 for comparison."

            # Select iterations to compare
            if iteration_1 is None:
                iter1_info = iterations[-1]
            else:
                iter1_info = next(
                    (i for i in iterations if i["name"] == iteration_1), None
                )
                if not iter1_info:
                    available = [i["name"] for i in iterations]
                    return (
                        f"Iteration '{iteration_1}' not found. Available: {available}"
                    )

            if iteration_2 is None:
                iter2_info = iterations[-2] if len(iterations) >= 2 else iterations[0]
            else:
                iter2_info = next(
                    (i for i in iterations if i["name"] == iteration_2), None
                )
                if not iter2_info:
                    available = [i["name"] for i in iterations]
                    return (
                        f"Iteration '{iteration_2}' not found. Available: {available}"
                    )

            # Get results DataFrames
            iter1_key = f"de_results_formula_{iter1_info['contrast_name']}"
            iter2_key = f"de_results_formula_{iter2_info['contrast_name']}"

            if iter1_key not in adata.uns or iter2_key not in adata.uns:
                return f"Results not found for one or both iterations."

            results1 = adata.uns[iter1_key]["results_df"]
            results2 = adata.uns[iter2_key]["results_df"]

            # Get significant genes (FDR < 0.05)
            sig1 = set(results1[results1["padj"] < 0.05].index)
            sig2 = set(results2[results2["padj"] < 0.05].index)

            # Calculate overlaps
            overlap = sig1 & sig2
            unique1 = sig1 - sig2
            unique2 = sig2 - sig1

            # Calculate fold change correlation for overlapping genes
            if len(overlap) > 3:
                overlap_genes = list(overlap)
                lfc1 = results1.loc[overlap_genes, "log2FoldChange"]
                lfc2 = results2.loc[overlap_genes, "log2FoldChange"]
                correlation = lfc1.corr(lfc2)
            else:
                correlation = None

            # Format response
            response = "## DE Iteration Comparison\n\n"
            response += "**Comparing:**\n"
            response += (
                f"- Iteration 1: '{iter1_info['name']}' - {iter1_info['formula']}\n"
            )
            response += (
                f"- Iteration 2: '{iter2_info['name']}' - {iter2_info['formula']}\n\n"
            )

            response += "**Results Summary:**\n"
            response += f"- Iteration 1 significant genes: {len(sig1):,}\n"
            response += f"- Iteration 2 significant genes: {len(sig2):,}\n"
            response += f"- Overlapping genes: {len(overlap):,} ({len(overlap)/max(len(sig1), len(sig2))*100:.1f}%)\n"
            response += f"- Unique to iteration 1: {len(unique1):,}\n"
            response += f"- Unique to iteration 2: {len(unique2):,}\n"

            if correlation is not None:
                response += f"- Fold change correlation: {correlation:.3f}\n"

            if show_overlap and len(overlap) > 0:
                response += "\n**Top Overlapping Genes:**\n"
                overlap_df = results1.loc[list(overlap)]
                overlap_df = overlap_df.reindex(overlap_df["padj"].sort_values().index)

                for gene in list(overlap_df.index)[:10]:
                    lfc1 = results1.loc[gene, "log2FoldChange"]
                    lfc2 = results2.loc[gene, "log2FoldChange"]
                    response += f"- {gene}: LFC1={lfc1:.2f}, LFC2={lfc2:.2f}\n"

            if show_unique and (len(unique1) > 0 or len(unique2) > 0):
                response += "\n**Unique Significant Genes:**\n"

                if len(unique1) > 0:
                    response += (
                        f"**Only in '{iter1_info['name']}'** ({len(unique1)} genes):\n"
                    )
                    unique1_sorted = results1.loc[list(unique1)].sort_values("padj")
                    for gene in unique1_sorted.index[:8]:
                        lfc = results1.loc[gene, "log2FoldChange"]
                        fdr = results1.loc[gene, "padj"]
                        response += f"- {gene}: LFC={lfc:.2f}, FDR={fdr:.2e}\n"

                if len(unique2) > 0:
                    response += f"\n**Only in '{iter2_info['name']}'** ({len(unique2)} genes):\n"
                    unique2_sorted = results2.loc[list(unique2)].sort_values("padj")
                    for gene in unique2_sorted.index[:8]:
                        lfc = results2.loc[gene, "log2FoldChange"]
                        fdr = results2.loc[gene, "padj"]
                        response += f"- {gene}: LFC={lfc:.2f}, FDR={fdr:.2e}\n"

            # Analysis interpretation
            response += "\n**Interpretation:**\n"
            if correlation is not None:
                if correlation > 0.8:
                    response += f"- High correlation ({correlation:.3f}) suggests similar biological effects\n"
                elif correlation > 0.5:
                    response += f"- Moderate correlation ({correlation:.3f}) - some consistency but notable differences\n"
                else:
                    response += f"- Low correlation ({correlation:.3f}) - formulas capture different effects\n"

            overlap_percent = len(overlap) / max(len(sig1), len(sig2)) * 100
            if overlap_percent > 70:
                response += f"- High overlap ({overlap_percent:.1f}%) - formulas yield similar gene sets\n"
            elif overlap_percent > 40:
                response += f"- Moderate overlap ({overlap_percent:.1f}%) - some formula-specific effects\n"
            else:
                response += f"- Low overlap ({overlap_percent:.1f}%) - formulas capture different biology\n"

            # Save comparison if requested
            if save_comparison:
                comparison_data = {
                    "iteration_1": iter1_info,
                    "iteration_2": iter2_info,
                    "overlap_genes": list(overlap),
                    "unique_to_1": list(unique1),
                    "unique_to_2": list(unique2),
                    "correlation": correlation,
                    "summary_stats": {
                        "n_sig_1": len(sig1),
                        "n_sig_2": len(sig2),
                        "n_overlap": len(overlap),
                        "overlap_percent": overlap_percent,
                    },
                }

                comparison_key = (
                    f"iteration_comparison_{iter1_info['name']}_vs_{iter2_info['name']}"
                )
                if "iteration_comparisons" not in adata.uns:
                    adata.uns["iteration_comparisons"] = {}
                adata.uns["iteration_comparisons"][comparison_key] = comparison_data

                data_manager.modalities[pseudobulk_modality] = adata
                response += f"\n**Comparison saved**: adata.uns['iteration_comparisons']['{comparison_key}']\n"

            response += "\n**Next steps**: Choose the most appropriate formula based on biological interpretation and statistical robustness."

            # Log the operation
            data_manager.log_tool_usage(
                tool_name="compare_de_iterations",
                parameters={
                    "pseudobulk_modality": pseudobulk_modality,
                    "iteration_1": iter1_info["name"],
                    "iteration_2": iter2_info["name"],
                    "show_overlap": show_overlap,
                    "show_unique": show_unique,
                },
                description=f"Compared iterations: {len(overlap)} overlapping, {len(unique1)}+{len(unique2)} unique genes",
            )

            return response

        except Exception as e:
            logger.error(f"Error comparing DE iterations: {e}")
            return f"Error comparing DE iterations: {str(e)}"

    @tool
    def run_pathway_enrichment_analysis(
        gene_list: List[str],
        analysis_type: str = "GO",
        modality_name: str = None,
        save_result: bool = True,
    ) -> str:
        """
        Run pathway enrichment analysis on gene lists from differential expression results.

        Args:
            gene_list: List of genes for enrichment analysis
            analysis_type: Type of analysis ("GO" or "KEGG")
            modality_name: Optional modality name to extract genes from DE results
            save_result: Whether to save enrichment results
        """
        try:
            # If modality name provided, extract significant genes from it
            if modality_name and modality_name in data_manager.list_modalities():
                adata = data_manager.get_modality(modality_name)

                # Look for DE results in uns
                de_keys = [
                    key for key in adata.uns.keys() if key.startswith("de_results")
                ]
                if de_keys:
                    de_results = adata.uns[de_keys[0]]
                    if isinstance(de_results, dict) and "results_df" in de_results:
                        de_df = de_results["results_df"]
                        if "padj" in de_df.columns:
                            significant_genes = de_df[
                                de_df["padj"] < 0.05
                            ].index.tolist()
                            if significant_genes:
                                gene_list = significant_genes[:500]
                                logger.info(
                                    f"Extracted {len(gene_list)} significant genes from {modality_name}"
                                )

            if not gene_list or len(gene_list) == 0:
                return "No genes provided for enrichment analysis. Please provide a gene list or run differential expression analysis first."

            logger.info(f"Running pathway enrichment on {len(gene_list)} genes")

            # Use bulk service for pathway enrichment
            enrichment_df, enrichment_stats, ir = (
                bulk_rnaseq_service.run_pathway_enrichment(
                    gene_list=gene_list, analysis_type=analysis_type
                )
            )

            # Log the operation with IR for provenance tracking
            data_manager.log_tool_usage(
                tool_name="run_pathway_enrichment_analysis",
                parameters={
                    "gene_list_size": len(gene_list),
                    "analysis_type": analysis_type,
                    "modality_name": modality_name,
                },
                description=f"{analysis_type} enrichment: {enrichment_stats['n_significant_terms']} significant terms",
                ir=ir,
            )

            # Format response
            response = f"## {analysis_type} Pathway Enrichment Analysis Complete\n\n"
            response += "**Enrichment Results:**\n"
            response += f"- Genes analyzed: {enrichment_stats['n_genes_input']:,}\n"
            response += f"- Database: {enrichment_stats['enrichment_database']}\n"
            response += f"- Terms found: {enrichment_stats['n_terms_total']}\n"
            response += f"- Significant terms (p.adj < 0.05): {enrichment_stats['n_significant_terms']}\n\n"

            response += "**Top Enriched Pathways:**\n"
            for term in enrichment_stats["top_terms"][:8]:
                response += f"- {term}\n"

            if len(enrichment_stats["top_terms"]) > 8:
                remaining = len(enrichment_stats["top_terms"]) - 8
                response += f"... and {remaining} more pathways\n"

            response += "\nPathway enrichment reveals biological processes and pathways associated with differential expression."

            analysis_results["details"]["pathway_enrichment"] = response
            return response

        except BulkRNASeqError as e:
            logger.error(f"Error in pathway enrichment: {e}")
            return f"Error running pathway enrichment: {str(e)}"
        except Exception as e:
            logger.error(f"Unexpected error in pathway enrichment: {e}")
            return f"Unexpected error: {str(e)}"

    # -------------------------
    # TOOL REGISTRY
    # -------------------------
    base_tools = [
        # Pseudobulk tools
        create_pseudobulk_matrix,
        prepare_differential_expression_design,
        run_pseudobulk_differential_expression,
        # Design validation
        validate_experimental_design,
        # Formula-based DE tools
        suggest_formula_for_design,
        construct_de_formula_interactive,
        run_differential_expression_with_formula,
        # Simple 2-group DE
        run_differential_expression_analysis,
        # Iteration tools
        iterate_de_analysis,
        compare_de_iterations,
        # Pathway analysis
        run_pathway_enrichment_analysis,
    ]

    tools = base_tools + (delegation_tools or [])

    # -------------------------
    # CREATE AGENT
    # -------------------------
    system_prompt = create_de_prompt()

    return create_react_agent(
        model=llm,
        tools=tools,
        prompt=system_prompt,
        name=agent_name,
        state_schema=DEAnalysisExpertState,
    )
