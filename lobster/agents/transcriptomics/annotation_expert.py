"""
Annotation Expert Sub-Agent for single-cell RNA-seq cell type annotation.

This sub-agent handles all cell type annotation tools for single-cell data.
It is called by the parent transcriptomics_expert via delegation tools.

Tools included:
1. annotate_cell_types - Automated cell type annotation using marker databases
2. manually_annotate_clusters_interactive - Rich terminal interface for manual annotation
3. manually_annotate_clusters - Direct cluster-to-celltype assignment
4. collapse_clusters_to_celltype - Merge multiple clusters into a single cell type
5. mark_clusters_as_debris - Flag clusters as debris for QC
6. suggest_debris_clusters - Smart suggestions for potential debris clusters
7. review_annotation_assignments - Review current annotation coverage
8. apply_annotation_template - Apply tissue-specific annotation templates
9. export_annotation_mapping - Export annotations for reuse
10. import_annotation_mapping - Import and apply saved annotations
"""

import datetime
import json
from datetime import date
from typing import List, Optional

import pandas as pd
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

from lobster.agents.state import SingleCellExpertState
from lobster.config.llm_factory import create_llm
from lobster.config.settings import get_settings
from lobster.core.data_manager_v2 import DataManagerV2
from lobster.services.analysis.enhanced_singlecell_service import (
    EnhancedSingleCellService,
)
from lobster.services.metadata.manual_annotation_service import ManualAnnotationService
from lobster.services.templates.annotation_templates import (
    AnnotationTemplateService,
    TissueType,
)
from lobster.utils.logger import get_logger

logger = get_logger(__name__)


# Type alias for AnnotationExpertState - uses SingleCellExpertState for now
# as annotations share the same state structure
AnnotationExpertState = SingleCellExpertState


class AnnotationAgentError(Exception):
    """Base exception for annotation agent operations."""

    pass


class ModalityNotFoundError(AnnotationAgentError):
    """Raised when requested modality doesn't exist."""

    pass


def create_annotation_prompt() -> str:
    """Create the system prompt for the annotation expert sub-agent."""
    return f"""
You are an expert bioinformatician specializing in cell type annotation for single-cell RNA-seq data.

<Role>
You focus exclusively on cell type annotation tasks including:
- Automated annotation using marker gene databases
- Manual cluster annotation with rich terminal interfaces
- Debris cluster identification and removal
- Annotation quality assessment and validation
- Annotation import/export for reproducibility
- Tissue-specific annotation template application

**IMPORTANT**:
- You ONLY perform annotation tasks delegated by the transcriptomics_expert
- You report results back to the parent agent
- You validate annotation quality at each step
- You maintain annotation provenance for reproducibility
</Role>

<Available Annotation Tools>

## Automated Annotation:
- `annotate_cell_types`: Automated cell type annotation using marker gene expression patterns

## Manual Annotation:
- `manually_annotate_clusters_interactive`: Launch Rich terminal interface for manual annotation
- `manually_annotate_clusters`: Direct assignment of cell types to clusters
- `collapse_clusters_to_celltype`: Merge multiple clusters into a single cell type
- `mark_clusters_as_debris`: Flag clusters as debris for quality control
- `suggest_debris_clusters`: Get smart suggestions for potential debris clusters

## Annotation Management:
- `review_annotation_assignments`: Review current annotation coverage and quality
- `apply_annotation_template`: Apply predefined tissue-specific annotation templates
- `export_annotation_mapping`: Export annotation mapping for reuse
- `import_annotation_mapping`: Import and apply saved annotation mappings

<Annotation Best Practices>

**Cell Type Annotation Protocol**

IMPORTANT: Built-in marker gene lists are PRELIMINARY and NOT scientifically validated.
They lack evidence scoring (AUC, logFC, specificity), reference atlas validation,
and tissue/context-specific optimization.

**MANDATORY STEPS before annotation:**

1. ALWAYS verify clustering quality before annotation
2. Check for marker gene data availability
3. Consider tissue context when selecting annotation approach
4. Validate annotations against known markers
5. Review cells with low confidence for manual curation
6. Document annotation decisions for reproducibility

**Confidence Scoring:**
When reference_markers are provided, annotation generates per-cell metrics:
- cell_type_confidence: Pearson correlation score (0-1)
- cell_type_top3: Top 3 cell type predictions
- annotation_entropy: Shannon entropy (lower = more confident)
- annotation_quality: Categorical flag (high/medium/low)

Quality thresholds:
- HIGH: confidence > 0.5 AND entropy < 0.8
- MEDIUM: confidence > 0.3 AND entropy < 1.0
- LOW: All other cases

**Debris Cluster Identification:**
Common debris indicators:
- Low gene counts (< 200 genes/cell)
- High mitochondrial percentage (> 50%)
- Low UMI counts (< 500 UMI/cell)
- Unusual expression profiles

<Important Guidelines>
1. **Validate modality existence** before any annotation operation
2. **Use descriptive modality names** for traceability
3. **Save intermediate results** for reproducibility
4. **Monitor annotation quality** at each step
5. **Document annotation decisions** in provenance logs
6. **Consider tissue context** when suggesting cell types
7. **Always provide confidence metrics** when available

Today's date: {date.today()}
""".strip()


def annotation_expert(
    data_manager: DataManagerV2,
    callback_handler=None,
    agent_name: str = "annotation_expert",
    delegation_tools: list = None,
):
    """
    Factory function for annotation expert sub-agent.

    This sub-agent handles all cell type annotation tools for single-cell data.
    It is delegated to by the transcriptomics_expert for annotation tasks.

    Args:
        data_manager: DataManagerV2 instance for modality management
        callback_handler: Optional callback handler for LLM responses
        agent_name: Name of the agent for routing
        delegation_tools: Optional list of delegation tools from parent agent

    Returns:
        A LangGraph ReAct agent configured for cell type annotation
    """

    settings = get_settings()
    model_params = settings.get_agent_llm_params("annotation_expert")
    llm = create_llm("annotation_expert", model_params)

    # Normalize callbacks to a flat list (fix double-nesting bug)
    if callback_handler and hasattr(llm, "with_config"):
        callbacks = callback_handler if isinstance(callback_handler, list) else [callback_handler]
        llm = llm.with_config(callbacks=callbacks)

    # Initialize services for annotation
    manual_annotation_service = ManualAnnotationService()
    template_service = AnnotationTemplateService()
    singlecell_service = EnhancedSingleCellService()

    # Track analysis results
    analysis_results = {"summary": "", "details": {}}

    # -------------------------
    # ANNOTATION TOOLS
    # -------------------------

    @tool
    def annotate_cell_types(
        modality_name: str, reference_markers: dict = None, save_result: bool = True
    ) -> str:
        """
        Annotate single-cell clusters with cell types based on marker gene expression patterns.

        Args:
            modality_name: Name of the single-cell modality with clustering results
            reference_markers: Custom marker genes dict (None to use defaults)
            save_result: Whether to save annotated modality
        """
        try:
            # Validate modality exists
            if modality_name not in data_manager.list_modalities():
                raise ModalityNotFoundError(
                    f"Modality '{modality_name}' not found. Available: {data_manager.list_modalities()}"
                )

            # Get the modality
            adata = data_manager.get_modality(modality_name)
            logger.info(
                f"Annotating cell types in single-cell modality '{modality_name}': {adata.shape[0]} cells x {adata.shape[1]} genes"
            )

            # Use singlecell service for cell type annotation
            adata_annotated, annotation_stats, ir = (
                singlecell_service.annotate_cell_types(
                    adata=adata, reference_markers=reference_markers
                )
            )

            # Save as new modality
            annotated_modality_name = f"{modality_name}_annotated"
            data_manager.modalities[annotated_modality_name] = adata_annotated

            # Save to file if requested
            if save_result:
                save_path = f"{modality_name}_annotated.h5ad"
                data_manager.save_modality(annotated_modality_name, save_path)

            # Log the operation
            data_manager.log_tool_usage(
                tool_name="annotate_cell_types",
                parameters={
                    "modality_name": modality_name,
                    "reference_markers": "custom" if reference_markers else "default",
                },
                description=f"Annotated {annotation_stats['n_cell_types_identified']} cell types in single-cell data {modality_name}",
                ir=ir,
            )

            # Format professional response
            response = f"""Successfully annotated cell types in single-cell modality '{modality_name}'!

**Single-cell Annotation Results:**
- Cell types identified: {annotation_stats['n_cell_types_identified']}
- Clusters annotated: {annotation_stats['n_clusters']}
- Marker sets used: {annotation_stats['n_marker_sets']}

**Single-cell Type Distribution:**"""

            for cell_type, count in list(annotation_stats["cell_type_counts"].items())[
                :8
            ]:
                response += f"\n- {cell_type}: {count} cells"

            if len(annotation_stats["cell_type_counts"]) > 8:
                remaining = len(annotation_stats["cell_type_counts"]) - 8
                response += f"\n... and {remaining} more types"

            # Add confidence distribution if available
            if "confidence_mean" in annotation_stats:
                response += f"\n\n**Confidence Scoring:**"
                response += (
                    f"\n- Mean confidence: {annotation_stats['confidence_mean']:.3f}"
                )
                response += f"\n- Median confidence: {annotation_stats['confidence_median']:.3f}"
                response += (
                    f"\n- Std deviation: {annotation_stats['confidence_std']:.3f}"
                )

                response += f"\n\n**Annotation Quality Distribution:**"
                quality_dist = annotation_stats["quality_distribution"]
                response += f"\n- High confidence: {quality_dist['high']} cells"
                response += f"\n- Medium confidence: {quality_dist['medium']} cells"
                response += f"\n- Low confidence: {quality_dist['low']} cells"

                response += "\n\n**Note**: Per-cell confidence scores available in:"
                response += (
                    "\n  - adata.obs['cell_type_confidence']: Correlation score (0-1)"
                )
                response += "\n  - adata.obs['cell_type_top3']: Top 3 predictions"
                response += "\n  - adata.obs['annotation_entropy']: Shannon entropy"
                response += "\n  - adata.obs['annotation_quality']: Quality flag (high/medium/low)"

            response += f"\n\n**New modality created**: '{annotated_modality_name}'"

            if save_result:
                response += f"\n**Saved to**: {save_path}"

            response += "\n**Cell type annotations added to**: adata.obs['cell_type']"
            response += "\n\nProceed with cell type-specific downstream analysis or comparative studies."

            analysis_results["details"]["cell_type_annotation"] = response
            return response

        except (ModalityNotFoundError,) as e:
            logger.error(f"Error in single-cell cell type annotation: {e}")
            return f"Error annotating cell types in single-cell data: {str(e)}"
        except Exception as e:
            logger.error(f"Unexpected error in single-cell cell type annotation: {e}")
            return f"Unexpected error: {str(e)}"

    @tool
    def manually_annotate_clusters_interactive(
        modality_name: str, cluster_col: str = "leiden", save_result: bool = True
    ) -> str:
        """
        Launch Rich terminal interface for manual cluster annotation with color synchronization.

        Args:
            modality_name: Name of clustered single-cell modality
            cluster_col: Column containing cluster assignments
            save_result: Whether to save annotated modality
        """
        try:
            # Validate modality exists
            if modality_name not in data_manager.list_modalities():
                raise ModalityNotFoundError(
                    f"Modality '{modality_name}' not found. Available: {data_manager.list_modalities()}"
                )

            # Get the modality
            adata = data_manager.get_modality(modality_name)
            logger.info(
                f"Launching interactive annotation for '{modality_name}': {adata.shape[0]} cells x {adata.shape[1]} genes"
            )

            # Validate cluster column exists
            if cluster_col not in adata.obs.columns:
                available_cols = [
                    col
                    for col in adata.obs.columns
                    if col in ["leiden", "cell_type", "louvain"]
                ]
                return f"Cluster column '{cluster_col}' not found. Available: {available_cols}"

            # Initialize annotation session
            annotation_state = manual_annotation_service.initialize_annotation_session(
                adata=adata, cluster_key=cluster_col
            )

            # Launch Rich terminal interface
            cell_type_mapping = manual_annotation_service.rich_annotation_interface()

            # Apply annotations to data
            adata_annotated = manual_annotation_service.apply_annotations_to_adata(
                adata=adata,
                cluster_key=cluster_col,
                cell_type_column="cell_type_manual",
            )

            # Save as new modality
            annotated_modality_name = f"{modality_name}_manually_annotated"
            data_manager.modalities[annotated_modality_name] = adata_annotated

            # Save to file if requested
            if save_result:
                save_path = f"{modality_name}_manually_annotated.h5ad"
                data_manager.save_modality(annotated_modality_name, save_path)

            # Log the operation
            data_manager.log_tool_usage(
                tool_name="manually_annotate_clusters_interactive",
                parameters={
                    "modality_name": modality_name,
                    "cluster_col": cluster_col,
                    "n_annotations": len(cell_type_mapping),
                },
                description=f"Manual annotation completed for {len(cell_type_mapping)} clusters",
            )

            # Validate results
            validation = manual_annotation_service.validate_annotation_coverage(
                adata_annotated, "cell_type_manual"
            )

            # Format response
            response = f"""Manual cluster annotation completed for '{modality_name}'!

**Interactive Annotation Results:**
- Total clusters: {len(annotation_state.clusters)}
- Manually annotated: {len(cell_type_mapping)}
- Marked as debris: {len(annotation_state.debris_clusters)}
- Coverage: {validation['coverage_percentage']:.1f}%

**Color-Synchronized Interface:**
- Rich terminal colors matched UMAP plot colors
- Visual cluster identification completed
- Expert-guided annotation workflow

**Cell Type Distribution:**"""

            for cell_type, count in list(validation["cell_type_counts"].items())[:8]:
                response += f"\n- {cell_type}: {count} cells"

            response += f"\n\n**New modality created**: '{annotated_modality_name}'"
            response += "\n**Manual annotations in**: adata.obs['cell_type_manual']"

            if save_result:
                response += f"\n**Saved to**: {save_path}"

            response += "\n\nManual annotation complete! Use for downstream analysis or pseudobulk aggregation."

            analysis_results["details"]["manual_annotation"] = response
            return response

        except Exception as e:
            logger.error(f"Error in interactive manual annotation: {e}")
            return f"Error in manual cluster annotation: {str(e)}"

    @tool
    def manually_annotate_clusters(
        modality_name: str,
        annotations: dict,
        cluster_col: str = "leiden",
        save_result: bool = True,
    ) -> str:
        """
        Directly assign cell types to clusters without interactive interface.

        Args:
            modality_name: Name of clustered single-cell modality
            annotations: Dictionary mapping cluster IDs to cell type names
            cluster_col: Column containing cluster assignments
            save_result: Whether to save annotated modality
        """
        try:
            # Validate modality exists
            if modality_name not in data_manager.list_modalities():
                raise ModalityNotFoundError(
                    f"Modality '{modality_name}' not found. Available: {data_manager.list_modalities()}"
                )

            # Get the modality
            adata = data_manager.get_modality(modality_name)

            # Validate cluster column exists
            if cluster_col not in adata.obs.columns:
                return f"Cluster column '{cluster_col}' not found."

            # Apply annotations directly
            adata_copy = adata.copy()
            cell_type_mapping = {}

            for cluster_id, cell_type in annotations.items():
                cell_type_mapping[str(cluster_id)] = cell_type

            # Create cell type column
            adata_copy.obs["cell_type_manual"] = (
                adata_copy.obs[cluster_col]
                .astype(str)
                .map(cell_type_mapping)
                .fillna("Unassigned")
            )

            # Save as new modality
            annotated_modality_name = f"{modality_name}_manually_annotated"
            data_manager.modalities[annotated_modality_name] = adata_copy

            # Save to file if requested
            if save_result:
                save_path = f"{modality_name}_manually_annotated.h5ad"
                data_manager.save_modality(annotated_modality_name, save_path)

            # Log the operation
            data_manager.log_tool_usage(
                tool_name="manually_annotate_clusters",
                parameters={
                    "modality_name": modality_name,
                    "cluster_col": cluster_col,
                    "annotations": annotations,
                },
                description=f"Direct manual annotation of {len(annotations)} clusters",
            )

            response = f"""Manual cluster annotation applied to '{modality_name}'!

**Annotation Results:**
- Clusters annotated: {len(annotations)}
- Cell types assigned: {len(set(annotations.values()))}

**Annotations Applied:**"""

            for cluster_id, cell_type in list(annotations.items())[:10]:
                response += f"\n- Cluster {cluster_id}: {cell_type}"

            if len(annotations) > 10:
                response += f"\n... and {len(annotations) - 10} more clusters"

            response += f"\n\n**New modality created**: '{annotated_modality_name}'"

            if save_result:
                response += f"\n**Saved to**: {save_path}"

            return response

        except Exception as e:
            logger.error(f"Error in manual annotation: {e}")
            return f"Error applying manual annotations: {str(e)}"

    @tool
    def collapse_clusters_to_celltype(
        modality_name: str,
        cluster_list: List[str],
        cell_type_name: str,
        cluster_col: str = "leiden",
        save_result: bool = True,
    ) -> str:
        """
        Collapse multiple clusters into a single cell type.

        Args:
            modality_name: Name of single-cell modality
            cluster_list: List of cluster IDs to collapse
            cell_type_name: New cell type name for collapsed clusters
            cluster_col: Column containing cluster assignments
            save_result: Whether to save result
        """
        try:
            # Validate modality exists
            if modality_name not in data_manager.list_modalities():
                raise ModalityNotFoundError(
                    f"Modality '{modality_name}' not found. Available: {data_manager.list_modalities()}"
                )

            # Get the modality
            adata = data_manager.get_modality(modality_name)

            # Validate clusters exist
            unique_clusters = set(adata.obs[cluster_col].astype(str).unique())
            invalid_clusters = [
                c for c in cluster_list if str(c) not in unique_clusters
            ]
            if invalid_clusters:
                return f"Invalid cluster IDs: {invalid_clusters}. Available: {sorted(unique_clusters)}"

            # Create collapsed annotation
            adata_copy = adata.copy()

            # Create or update manual cell type column
            if "cell_type_manual" not in adata_copy.obs:
                adata_copy.obs["cell_type_manual"] = "Unassigned"

            # Apply collapse
            for cluster_id in cluster_list:
                mask = adata_copy.obs[cluster_col].astype(str) == str(cluster_id)
                adata_copy.obs.loc[mask, "cell_type_manual"] = cell_type_name

            # Calculate statistics
            total_cells_collapsed = sum(
                (adata_copy.obs[cluster_col].astype(str) == str(c)).sum()
                for c in cluster_list
            )

            # Save as new modality
            collapsed_modality_name = f"{modality_name}_collapsed"
            data_manager.modalities[collapsed_modality_name] = adata_copy

            # Save to file if requested
            if save_result:
                save_path = f"{modality_name}_collapsed.h5ad"
                data_manager.save_modality(collapsed_modality_name, save_path)

            # Log the operation
            data_manager.log_tool_usage(
                tool_name="collapse_clusters_to_celltype",
                parameters={
                    "modality_name": modality_name,
                    "cluster_list": cluster_list,
                    "cell_type_name": cell_type_name,
                    "cluster_col": cluster_col,
                },
                description=f"Collapsed {len(cluster_list)} clusters into '{cell_type_name}'",
            )

            response = f"""Successfully collapsed clusters in '{modality_name}'!

**Collapse Results:**
- Clusters collapsed: {', '.join(cluster_list)}
- New cell type: {cell_type_name}
- Total cells affected: {total_cells_collapsed:,}

**New modality created**: '{collapsed_modality_name}'"""

            if save_result:
                response += f"\n**Saved to**: {save_path}"

            response += f"\n\nClusters {', '.join(cluster_list)} are now annotated as '{cell_type_name}'."

            return response

        except Exception as e:
            logger.error(f"Error collapsing clusters: {e}")
            return f"Error collapsing clusters: {str(e)}"

    @tool
    def mark_clusters_as_debris(
        modality_name: str,
        debris_clusters: List[str],
        remove_debris: bool = False,
        cluster_col: str = "leiden",
        save_result: bool = True,
    ) -> str:
        """
        Mark specified clusters as debris for quality control.

        Args:
            modality_name: Name of single-cell modality
            debris_clusters: List of cluster IDs to mark as debris
            remove_debris: Whether to remove debris clusters from data
            cluster_col: Column containing cluster assignments
            save_result: Whether to save result
        """
        try:
            # Validate modality exists
            if modality_name not in data_manager.list_modalities():
                raise ModalityNotFoundError(
                    f"Modality '{modality_name}' not found. Available: {data_manager.list_modalities()}"
                )

            # Get the modality
            adata = data_manager.get_modality(modality_name)

            # Validate clusters exist
            unique_clusters = set(adata.obs[cluster_col].astype(str).unique())
            invalid_clusters = [
                c for c in debris_clusters if str(c) not in unique_clusters
            ]
            if invalid_clusters:
                return f"Invalid cluster IDs: {invalid_clusters}. Available: {sorted(unique_clusters)}"

            adata_copy = adata.copy()

            # Mark debris clusters
            if "cell_type_manual" not in adata_copy.obs:
                adata_copy.obs["cell_type_manual"] = "Unassigned"

            debris_mask = (
                adata_copy.obs[cluster_col]
                .astype(str)
                .isin([str(c) for c in debris_clusters])
            )
            adata_copy.obs.loc[debris_mask, "cell_type_manual"] = "Debris"

            # Add debris flag
            adata_copy.obs["is_debris"] = False
            adata_copy.obs.loc[debris_mask, "is_debris"] = True

            # Optionally remove debris
            if remove_debris:
                adata_copy = adata_copy[~debris_mask].copy()

            total_debris_cells = debris_mask.sum()

            # Save as new modality
            debris_modality_name = f"{modality_name}_debris_marked"
            if remove_debris:
                debris_modality_name = f"{modality_name}_debris_removed"

            data_manager.modalities[debris_modality_name] = adata_copy

            # Save to file if requested
            if save_result:
                save_path = f"{debris_modality_name}.h5ad"
                data_manager.save_modality(debris_modality_name, save_path)

            # Log the operation
            data_manager.log_tool_usage(
                tool_name="mark_clusters_as_debris",
                parameters={
                    "modality_name": modality_name,
                    "debris_clusters": debris_clusters,
                    "remove_debris": remove_debris,
                    "cluster_col": cluster_col,
                },
                description=f"Marked {len(debris_clusters)} clusters as debris ({total_debris_cells} cells)",
            )

            response = f"""Successfully marked debris clusters in '{modality_name}'!

**Debris Marking Results:**
- Clusters marked: {', '.join(debris_clusters)}
- Total debris cells: {total_debris_cells:,}
- Action: {'Removed' if remove_debris else 'Marked only'}

**New modality created**: '{debris_modality_name}'"""

            if save_result:
                response += f"\n**Saved to**: {save_path}"

            if remove_debris:
                remaining_cells = adata_copy.n_obs
                response += f"\n**Remaining cells**: {remaining_cells:,} ({remaining_cells/adata.n_obs*100:.1f}%)"
            else:
                response += "\n**Debris flag added**: adata.obs['is_debris']"

            return response

        except Exception as e:
            logger.error(f"Error marking debris clusters: {e}")
            return f"Error marking clusters as debris: {str(e)}"

    @tool
    def suggest_debris_clusters(
        modality_name: str,
        min_genes: int = 200,
        max_mt_percent: float = 50,
        min_umi: int = 500,
        cluster_col: str = "leiden",
    ) -> str:
        """
        Get smart suggestions for potential debris clusters based on QC metrics.

        Args:
            modality_name: Name of single-cell modality
            min_genes: Minimum genes per cell threshold
            max_mt_percent: Maximum mitochondrial percentage
            min_umi: Minimum UMI count threshold
            cluster_col: Column containing cluster assignments
        """
        try:
            # Validate modality exists
            if modality_name not in data_manager.list_modalities():
                raise ModalityNotFoundError(
                    f"Modality '{modality_name}' not found. Available: {data_manager.list_modalities()}"
                )

            # Get the modality
            adata = data_manager.get_modality(modality_name)

            # Get suggestions using manual annotation service
            suggested_debris = manual_annotation_service.suggest_debris_clusters(
                adata=adata,
                min_genes=min_genes,
                max_mt_percent=max_mt_percent,
                min_umi=min_umi,
            )

            if not suggested_debris:
                return f"No debris clusters suggested based on QC thresholds (min_genes={min_genes}, max_mt%={max_mt_percent}, min_umi={min_umi})"

            # Get cluster statistics for suggestions
            response = f"""Smart debris cluster suggestions for '{modality_name}':

**QC-Based Suggestions:**
- Clusters flagged: {len(suggested_debris)}
- Thresholds used: min_genes={min_genes}, max_mt%={max_mt_percent}, min_umi={min_umi}

**Suggested Debris Clusters:**"""

            for cluster_id in suggested_debris[:10]:
                cluster_mask = adata.obs[cluster_col].astype(str) == cluster_id
                n_cells = cluster_mask.sum()

                # Get QC stats for cluster
                if cluster_mask.sum() > 0:
                    mean_genes = (
                        adata.obs.loc[cluster_mask, "n_genes"].mean()
                        if "n_genes" in adata.obs
                        else 0
                    )
                    mean_mt = (
                        adata.obs.loc[cluster_mask, "percent_mito"].mean()
                        if "percent_mito" in adata.obs
                        else 0
                    )
                    mean_umi = (
                        adata.obs.loc[cluster_mask, "n_counts"].mean()
                        if "n_counts" in adata.obs
                        else 0
                    )

                    response += f"\n- Cluster {cluster_id}: {n_cells} cells (genes: {mean_genes:.0f}, MT: {mean_mt:.1f}%, UMI: {mean_umi:.0f})"

            if len(suggested_debris) > 10:
                response += f"\n... and {len(suggested_debris) - 10} more clusters"

            response += "\n\n**Recommendation:**"
            response += "\nUse 'mark_clusters_as_debris' to apply these suggestions."
            response += f"\nExample: mark_clusters_as_debris('{modality_name}', {suggested_debris[:5]})"

            return response

        except Exception as e:
            logger.error(f"Error suggesting debris clusters: {e}")
            return f"Error suggesting debris clusters: {str(e)}"

    @tool
    def review_annotation_assignments(
        modality_name: str,
        annotation_col: str = "cell_type_manual",
        show_unassigned: bool = True,
        show_debris: bool = True,
    ) -> str:
        """
        Review current manual annotation assignments.

        Args:
            modality_name: Name of modality with annotations
            annotation_col: Column containing cell type annotations
            show_unassigned: Whether to show unassigned clusters
            show_debris: Whether to show debris clusters
        """
        try:
            # Validate modality exists
            if modality_name not in data_manager.list_modalities():
                raise ModalityNotFoundError(
                    f"Modality '{modality_name}' not found. Available: {data_manager.list_modalities()}"
                )

            # Get the modality
            adata = data_manager.get_modality(modality_name)

            if annotation_col not in adata.obs.columns:
                return f"Annotation column '{annotation_col}' not found. Available columns: {list(adata.obs.columns)[:10]}"

            # Validate annotation coverage
            validation = manual_annotation_service.validate_annotation_coverage(
                adata, annotation_col
            )

            response = f"""Annotation review for '{modality_name}':

**Coverage Summary:**
- Total cells: {validation['total_cells']:,}
- Annotated cells: {validation['annotated_cells']:,} ({validation['coverage_percentage']:.1f}%)
- Unassigned cells: {validation['unassigned_cells']:,}
- Debris cells: {validation['debris_cells']:,}
- Unique cell types: {validation['unique_cell_types']}

**Cell Type Distribution:**"""

            # Show all cell types
            for cell_type, count in validation["cell_type_counts"].items():
                if cell_type == "Unassigned" and not show_unassigned:
                    continue
                if cell_type == "Debris" and not show_debris:
                    continue

                percentage = (count / validation["total_cells"]) * 100
                response += f"\n- {cell_type}: {count:,} cells ({percentage:.1f}%)"

            # Add quality assessment
            if validation["coverage_percentage"] >= 90:
                response += "\n\n**Quality**: Excellent annotation coverage"
            elif validation["coverage_percentage"] >= 70:
                response += "\n\n**Quality**: Good annotation coverage, consider annotating remaining clusters"
            else:
                response += (
                    "\n\n**Quality**: Low annotation coverage, more annotation needed"
                )

            return response

        except Exception as e:
            logger.error(f"Error reviewing annotations: {e}")
            return f"Error reviewing annotation assignments: {str(e)}"

    @tool
    def apply_annotation_template(
        modality_name: str,
        tissue_type: str,
        cluster_col: str = "leiden",
        expression_threshold: float = 0.5,
        save_result: bool = True,
    ) -> str:
        """
        Apply predefined tissue-specific annotation template to suggest cell types.

        Args:
            modality_name: Name of single-cell modality
            tissue_type: Type of tissue (pbmc, brain, lung, heart, kidney, liver, intestine, skin, tumor)
            cluster_col: Column containing cluster assignments
            expression_threshold: Minimum expression for marker detection
            save_result: Whether to save annotated modality
        """
        try:
            # Validate modality exists
            if modality_name not in data_manager.list_modalities():
                raise ModalityNotFoundError(
                    f"Modality '{modality_name}' not found. Available: {data_manager.list_modalities()}"
                )

            # Get the modality
            adata = data_manager.get_modality(modality_name)

            # Validate tissue type
            try:
                tissue_enum = TissueType(tissue_type.lower())
            except ValueError:
                available_tissues = [t.value for t in TissueType]
                return f"Invalid tissue type '{tissue_type}'. Available: {available_tissues}"

            # Apply template
            cluster_suggestions = template_service.apply_template_to_clusters(
                adata=adata,
                tissue_type=tissue_enum,
                cluster_col=cluster_col,
                expression_threshold=expression_threshold,
            )

            if not cluster_suggestions:
                return (
                    f"No template suggestions generated for tissue type '{tissue_type}'"
                )

            # Apply suggestions to data
            adata_copy = adata.copy()
            adata_copy.obs["cell_type_template"] = (
                adata_copy.obs[cluster_col]
                .astype(str)
                .map(cluster_suggestions)
                .fillna("Unknown")
            )

            # Save as new modality
            template_modality_name = f"{modality_name}_template_{tissue_type}"
            data_manager.modalities[template_modality_name] = adata_copy

            # Save to file if requested
            if save_result:
                save_path = f"{modality_name}_template_{tissue_type}.h5ad"
                data_manager.save_modality(template_modality_name, save_path)

            # Log the operation
            data_manager.log_tool_usage(
                tool_name="apply_annotation_template",
                parameters={
                    "modality_name": modality_name,
                    "tissue_type": tissue_type,
                    "cluster_col": cluster_col,
                    "expression_threshold": expression_threshold,
                },
                description=f"Applied {tissue_type} template: {len(cluster_suggestions)} clusters annotated",
            )

            # Get template cell types
            template = template_service.get_template(tissue_enum)
            available_types = list(template.keys()) if template else []

            response = f"""Applied {tissue_type.upper()} annotation template to '{modality_name}'!

**Template Application Results:**
- Tissue type: {tissue_type.upper()}
- Clusters analyzed: {len(cluster_suggestions)}
- Expression threshold: {expression_threshold}

**Suggested Annotations:**"""

            # Show suggestions
            suggestion_counts = {}
            for cluster_id, cell_type in cluster_suggestions.items():
                if cell_type not in suggestion_counts:
                    suggestion_counts[cell_type] = []
                suggestion_counts[cell_type].append(cluster_id)

            for cell_type, clusters in suggestion_counts.items():
                response += f"\n- {cell_type}: clusters {', '.join(sorted(clusters))}"

            response += f"\n\n**Available cell types in {tissue_type} template:**"
            response += f"\n{', '.join(available_types[:8])}"
            if len(available_types) > 8:
                response += f"... and {len(available_types) - 8} more"

            response += f"\n\n**New modality created**: '{template_modality_name}'"
            response += "\n**Template suggestions in**: adata.obs['cell_type_template']"

            if save_result:
                response += f"\n**Saved to**: {save_path}"

            response += "\n\n**Next steps:** Review suggestions and refine with manual annotation if needed."

            return response

        except Exception as e:
            logger.error(f"Error applying annotation template: {e}")
            return f"Error applying annotation template: {str(e)}"

    @tool
    def export_annotation_mapping(
        modality_name: str,
        annotation_col: str = "cell_type_manual",
        output_filename: str = "annotation_mapping.json",
        format: str = "json",
    ) -> str:
        """
        Export annotation mapping for reuse in other analyses.

        Args:
            modality_name: Name of annotated modality
            annotation_col: Column containing cell type annotations
            output_filename: Output filename
            format: Export format ('json' or 'csv')
        """
        try:
            # Validate modality exists
            if modality_name not in data_manager.list_modalities():
                raise ModalityNotFoundError(
                    f"Modality '{modality_name}' not found. Available: {data_manager.list_modalities()}"
                )

            # Get the modality
            adata = data_manager.get_modality(modality_name)

            if annotation_col not in adata.obs.columns:
                return f"Annotation column '{annotation_col}' not found."

            # Create export data
            export_data = {
                "modality_name": modality_name,
                "annotation_column": annotation_col,
                "export_timestamp": datetime.datetime.now().isoformat(),
                "total_cells": adata.n_obs,
                "cell_type_mapping": {},
                "cell_type_counts": adata.obs[annotation_col].value_counts().to_dict(),
            }

            # Create cluster-to-celltype mapping if cluster info available
            cluster_cols = [
                col for col in adata.obs.columns if col in ["leiden", "louvain"]
            ]
            if cluster_cols:
                cluster_col = cluster_cols[0]
                cluster_mapping = {}
                for cluster_id in adata.obs[cluster_col].unique():
                    cluster_mask = adata.obs[cluster_col] == cluster_id
                    most_common_type = (
                        adata.obs.loc[cluster_mask, annotation_col].mode().iloc[0]
                    )
                    cluster_mapping[str(cluster_id)] = most_common_type

                export_data["cluster_to_celltype"] = cluster_mapping
                export_data["cluster_column"] = cluster_col

            # Export based on format
            if format.lower() == "json":
                with open(output_filename, "w") as f:
                    json.dump(export_data, f, indent=2)
            elif format.lower() == "csv":
                # Export as CSV
                df_data = []
                for cell_type, count in export_data["cell_type_counts"].items():
                    df_data.append(
                        {
                            "cell_type": cell_type,
                            "cell_count": count,
                            "percentage": (count / export_data["total_cells"]) * 100,
                        }
                    )

                df = pd.DataFrame(df_data)
                df.to_csv(output_filename, index=False)
            else:
                return f"Unsupported export format: {format}. Use 'json' or 'csv'."

            # Log the operation
            data_manager.log_tool_usage(
                tool_name="export_annotation_mapping",
                parameters={
                    "modality_name": modality_name,
                    "annotation_col": annotation_col,
                    "output_filename": output_filename,
                    "format": format,
                },
                description=f"Exported annotation mapping with {len(export_data['cell_type_counts'])} cell types",
            )

            response = f"""Successfully exported annotation mapping for '{modality_name}'!

**Export Details:**
- Annotation column: {annotation_col}
- Output file: {output_filename}
- Format: {format.upper()}
- Cell types: {len(export_data['cell_type_counts'])}

**Exported Data:**
- Total cells: {export_data['total_cells']:,}
- Cell type counts included
- Cluster mapping included (if available)
- Export timestamp: {export_data['export_timestamp']}

**File created**: {output_filename}

Use this mapping to apply consistent annotations to similar datasets."""

            return response

        except Exception as e:
            logger.error(f"Error exporting annotation mapping: {e}")
            return f"Error exporting annotation mapping: {str(e)}"

    @tool
    def import_annotation_mapping(
        modality_name: str,
        mapping_file: str,
        preview_only: bool = False,
        save_result: bool = True,
    ) -> str:
        """
        Import and apply annotation mapping from previous analysis.

        Args:
            modality_name: Name of modality to annotate
            mapping_file: Path to mapping file (JSON format)
            preview_only: If True, only show what would be applied
            save_result: Whether to save annotated modality
        """
        try:
            # Validate modality exists
            if modality_name not in data_manager.list_modalities():
                raise ModalityNotFoundError(
                    f"Modality '{modality_name}' not found. Available: {data_manager.list_modalities()}"
                )

            # Get the modality
            adata = data_manager.get_modality(modality_name)

            # Load mapping file
            with open(mapping_file, "r") as f:
                mapping_data = json.load(f)

            if preview_only:
                response = f"""Preview of annotation mapping from '{mapping_file}':

**Mapping File Details:**
- Source modality: {mapping_data.get('modality_name', 'N/A')}
- Annotation column: {mapping_data.get('annotation_column', 'N/A')}
- Export timestamp: {mapping_data.get('export_timestamp', 'N/A')}

**Cell Types in Mapping:**"""

                for cell_type, count in mapping_data.get(
                    "cell_type_counts", {}
                ).items():
                    response += f"\n- {cell_type}: {count} cells"

                if "cluster_to_celltype" in mapping_data:
                    response += "\n\n**Cluster Mappings:**"
                    cluster_mapping = mapping_data["cluster_to_celltype"]
                    for cluster_id, cell_type in list(cluster_mapping.items())[:10]:
                        response += f"\n- Cluster {cluster_id}: {cell_type}"

                    if len(cluster_mapping) > 10:
                        response += (
                            f"\n... and {len(cluster_mapping) - 10} more clusters"
                        )

                response += f"\n\nUse preview_only=False to apply this mapping to '{modality_name}'."
                return response

            # Apply mapping
            adata_copy = adata.copy()

            if (
                "cluster_to_celltype" in mapping_data
                and "cluster_column" in mapping_data
            ):
                cluster_col = mapping_data["cluster_column"]
                cluster_mapping = mapping_data["cluster_to_celltype"]

                if cluster_col in adata_copy.obs.columns:
                    adata_copy.obs["cell_type_imported"] = (
                        adata_copy.obs[cluster_col]
                        .astype(str)
                        .map(cluster_mapping)
                        .fillna("Unassigned")
                    )
                else:
                    return f"Cluster column '{cluster_col}' from mapping not found in modality."
            else:
                return "Mapping file does not contain cluster-to-celltype information."

            # Save as new modality
            imported_modality_name = f"{modality_name}_imported_annotations"
            data_manager.modalities[imported_modality_name] = adata_copy

            # Save to file if requested
            if save_result:
                save_path = f"{modality_name}_imported_annotations.h5ad"
                data_manager.save_modality(imported_modality_name, save_path)

            # Log the operation
            data_manager.log_tool_usage(
                tool_name="import_annotation_mapping",
                parameters={
                    "modality_name": modality_name,
                    "mapping_file": mapping_file,
                    "preview_only": preview_only,
                },
                description=f"Imported annotation mapping from {mapping_file}",
            )

            # Validate imported annotations
            validation = manual_annotation_service.validate_annotation_coverage(
                adata_copy, "cell_type_imported"
            )

            response = f"""Successfully imported annotation mapping to '{modality_name}'!

**Import Results:**
- Mapping file: {mapping_file}
- Clusters mapped: {len(cluster_mapping)}
- Coverage: {validation['coverage_percentage']:.1f}%

**Imported Cell Types:**"""

            for cell_type, count in list(validation["cell_type_counts"].items())[:8]:
                response += f"\n- {cell_type}: {count:,} cells"

            response += f"\n\n**New modality created**: '{imported_modality_name}'"
            response += "\n**Imported annotations in**: adata.obs['cell_type_imported']"

            if save_result:
                response += f"\n**Saved to**: {save_path}"

            return response

        except FileNotFoundError:
            return f"Mapping file not found: {mapping_file}"
        except Exception as e:
            logger.error(f"Error importing annotation mapping: {e}")
            return f"Error importing annotation mapping: {str(e)}"

    # -------------------------
    # TOOL REGISTRY
    # -------------------------
    base_tools = [
        # Automated annotation
        annotate_cell_types,
        # Manual annotation
        manually_annotate_clusters_interactive,
        manually_annotate_clusters,
        collapse_clusters_to_celltype,
        mark_clusters_as_debris,
        suggest_debris_clusters,
        # Annotation management
        review_annotation_assignments,
        apply_annotation_template,
        export_annotation_mapping,
        import_annotation_mapping,
    ]

    tools = base_tools + (delegation_tools or [])

    # -------------------------
    # SYSTEM PROMPT
    # -------------------------
    system_prompt = create_annotation_prompt()

    return create_react_agent(
        model=llm,
        tools=tools,
        prompt=system_prompt,
        name=agent_name,
        state_schema=AnnotationExpertState,
    )
