"""
Modality Management Service for centralized modality CRUD operations.

This service provides a unified interface for managing modalities in DataManagerV2,
including listing, inspection, removal, compatibility validation, and loading.
All operations include W3C-PROV compliant provenance tracking.
"""

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

from lobster.core.analysis_ir import AnalysisStep
from lobster.core.data_manager_v2 import DataManagerV2
from lobster.utils.logger import get_logger

logger = get_logger(__name__)


class ModalityManagementService:
    """Service for centralized modality CRUD operations with provenance tracking."""

    def __init__(self, data_manager: DataManagerV2):
        """
        Initialize the ModalityManagementService.

        Args:
            data_manager: DataManagerV2 instance for modality operations
        """
        self.data_manager = data_manager

    def list_modalities(
        self, filter_pattern: Optional[str] = None
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any], AnalysisStep]:
        """
        List all available modalities with optional filtering.

        Args:
            filter_pattern: Optional glob-style pattern to filter modality names
                          (e.g., "geo_gse*", "*clustered", "bulk_*")

        Returns:
            Tuple containing:
            - List of modality info dicts (name, n_obs, n_vars, obs_cols, var_cols)
            - Statistics dict (total_count, matched_count, filter_pattern)
            - AnalysisStep for provenance tracking
        """
        try:
            all_modalities = self.data_manager.list_modalities()

            # Apply filter if provided
            if filter_pattern:
                import fnmatch

                filtered = [
                    name
                    for name in all_modalities
                    if fnmatch.fnmatch(name, filter_pattern)
                ]
            else:
                filtered = all_modalities

            # Gather info for each modality
            modality_info = []
            for name in filtered:
                try:
                    adata = self.data_manager.get_modality(name)
                    info = {
                        "name": name,
                        "n_obs": adata.n_obs,
                        "n_vars": adata.n_vars,
                        "obs_columns": list(adata.obs.columns),
                        "var_columns": list(adata.var.columns),
                    }
                    modality_info.append(info)
                except Exception as e:
                    logger.warning(
                        f"Could not retrieve info for modality '{name}': {e}"
                    )
                    modality_info.append({"name": name, "error": str(e)})

            stats = {
                "total_modalities": len(all_modalities),
                "matched_modalities": len(filtered),
                "filter_pattern": filter_pattern,
                "timestamp": datetime.now().isoformat(),
            }

            ir = self._create_list_ir(filter_pattern=filter_pattern)

            logger.info(
                f"Listed {len(filtered)} modalities"
                + (f" (filtered by '{filter_pattern}')" if filter_pattern else "")
            )

            return modality_info, stats, ir

        except Exception as e:
            logger.error(f"Error listing modalities: {e}")
            raise

    def get_modality_info(
        self, modality_name: str
    ) -> Tuple[Dict[str, Any], Dict[str, Any], AnalysisStep]:
        """
        Get detailed information about a specific modality.

        Args:
            modality_name: Name of the modality to inspect

        Returns:
            Tuple containing:
            - Detailed info dict (shape, layers, obsm keys, varm keys, uns keys, etc.)
            - Statistics dict (summary metrics)
            - AnalysisStep for provenance tracking
        """
        try:
            if modality_name not in self.data_manager.list_modalities():
                raise ValueError(
                    f"Modality '{modality_name}' not found. "
                    f"Available: {', '.join(self.data_manager.list_modalities())}"
                )

            adata = self.data_manager.get_modality(modality_name)

            # Gather comprehensive info
            info = {
                "name": modality_name,
                "shape": {"n_obs": adata.n_obs, "n_vars": adata.n_vars},
                "obs_columns": list(adata.obs.columns),
                "var_columns": list(adata.var.columns),
                "layers": list(adata.layers.keys()) if hasattr(adata, "layers") else [],
                "obsm_keys": list(adata.obsm.keys()) if hasattr(adata, "obsm") else [],
                "varm_keys": list(adata.varm.keys()) if hasattr(adata, "varm") else [],
                "uns_keys": list(adata.uns.keys()) if hasattr(adata, "uns") else [],
                "is_sparse": sparse.issparse(adata.X),
            }

            # Get quality metrics if available
            try:
                quality_metrics = self.data_manager.get_quality_metrics(modality_name)
                info["quality_metrics"] = quality_metrics
            except Exception:
                info["quality_metrics"] = {}

            stats = {
                "modality_name": modality_name,
                "total_cells": adata.n_obs,
                "total_features": adata.n_vars,
                "layer_count": len(info["layers"]),
                "obsm_count": len(info["obsm_keys"]),
                "timestamp": datetime.now().isoformat(),
            }

            ir = self._create_get_info_ir(modality_name=modality_name)

            logger.info(
                f"Retrieved info for modality '{modality_name}': "
                f"{adata.n_obs} obs × {adata.n_vars} vars"
            )

            return info, stats, ir

        except Exception as e:
            logger.error(f"Error getting modality info for '{modality_name}': {e}")
            raise

    def remove_modality(
        self, modality_name: str
    ) -> Tuple[bool, Dict[str, Any], AnalysisStep]:
        """
        Remove a modality from DataManagerV2.

        Args:
            modality_name: Name of the modality to remove

        Returns:
            Tuple containing:
            - Success boolean
            - Statistics dict (removed_modality, remaining_count)
            - AnalysisStep for provenance tracking
        """
        try:
            if modality_name not in self.data_manager.list_modalities():
                raise ValueError(
                    f"Modality '{modality_name}' not found. "
                    f"Available: {', '.join(self.data_manager.list_modalities())}"
                )

            # Get info before removal for stats
            adata = self.data_manager.get_modality(modality_name)
            n_obs = adata.n_obs
            n_vars = adata.n_vars

            # Remove modality
            self.data_manager.remove_modality(modality_name)

            remaining = self.data_manager.list_modalities()

            stats = {
                "removed_modality": modality_name,
                "shape": {"n_obs": n_obs, "n_vars": n_vars},
                "remaining_modalities": len(remaining),
                "timestamp": datetime.now().isoformat(),
            }

            ir = self._create_remove_ir(modality_name=modality_name)

            logger.info(
                f"Removed modality '{modality_name}' ({n_obs} × {n_vars}). "
                f"{len(remaining)} modalities remaining."
            )

            return True, stats, ir

        except Exception as e:
            logger.error(f"Error removing modality '{modality_name}': {e}")
            raise

    def validate_compatibility(
        self, modality_names: List[str]
    ) -> Tuple[Dict[str, Any], Dict[str, Any], AnalysisStep]:
        """
        Validate compatibility between multiple modalities for integration.

        Checks:
        - Observation (sample) alignment
        - Variable (feature) overlap
        - Batch effects (via obs columns)
        - Schema compatibility
        - Recommends integration strategy

        Args:
            modality_names: List of modality names to validate for compatibility

        Returns:
            Tuple containing:
            - Validation result dict (compatible, issues, recommendations)
            - Statistics dict (overlap_rate, shared_obs, shared_vars)
            - AnalysisStep for provenance tracking
        """
        try:
            if len(modality_names) < 2:
                raise ValueError(
                    "Need at least 2 modalities to validate compatibility. "
                    f"Provided: {len(modality_names)}"
                )

            # Validate all modalities exist
            available = self.data_manager.list_modalities()
            missing = [name for name in modality_names if name not in available]
            if missing:
                raise ValueError(
                    f"Modalities not found: {missing}. Available: {', '.join(available)}"
                )

            # Load all modalities
            modalities = {
                name: self.data_manager.get_modality(name) for name in modality_names
            }

            # Check observation overlap
            obs_indices = [set(adata.obs_names) for adata in modalities.values()]
            shared_obs = set.intersection(*obs_indices)
            obs_overlap_rate = len(shared_obs) / max(len(idx) for idx in obs_indices)

            # Check variable overlap
            var_indices = [set(adata.var_names) for adata in modalities.values()]
            shared_vars = set.intersection(*var_indices)
            var_overlap_rate = len(shared_vars) / max(len(idx) for idx in var_indices)

            # Detect batch effects (check for common batch columns)
            batch_columns = []
            first_obs_cols = set(modalities[modality_names[0]].obs.columns)
            common_obs_cols = first_obs_cols.intersection(
                *[set(adata.obs.columns) for adata in list(modalities.values())[1:]]
            )
            potential_batch_cols = [
                col
                for col in common_obs_cols
                if "batch" in col.lower() or "sample" in col.lower()
            ]
            batch_columns = potential_batch_cols

            # Determine compatibility
            issues = []
            compatible = True

            if obs_overlap_rate < 0.5:
                issues.append(
                    f"Low observation overlap ({obs_overlap_rate:.1%}). "
                    "Consider cohort-level integration."
                )
                compatible = False

            if var_overlap_rate < 0.3 and len(shared_vars) == 0:
                issues.append(
                    "No shared features. Cross-modal integration required "
                    "(e.g., pathway-level or correlation-based)."
                )
                compatible = False

            if not batch_columns:
                issues.append(
                    "No common batch columns detected. "
                    "May need manual batch annotation before integration."
                )

            # Recommendations
            recommendations = []
            if obs_overlap_rate >= 0.9:
                recommendations.append(
                    "High observation overlap - sample-level integration recommended"
                )
            elif obs_overlap_rate >= 0.5:
                recommendations.append(
                    "Medium observation overlap - consider metadata matching or cohort-level"
                )
            else:
                recommendations.append(
                    "Low observation overlap - cohort-level integration only"
                )

            if var_overlap_rate > 0:
                recommendations.append(
                    f"Feature overlap detected ({len(shared_vars)} shared) - "
                    "direct integration possible"
                )
            else:
                recommendations.append(
                    "No feature overlap - use pathway-level or correlation-based integration"
                )

            validation = {
                "compatible": compatible,
                "modalities": modality_names,
                "shared_observations": len(shared_obs),
                "shared_variables": len(shared_vars),
                "observation_overlap_rate": obs_overlap_rate,
                "variable_overlap_rate": var_overlap_rate,
                "batch_columns": batch_columns,
                "issues": issues,
                "recommendations": recommendations,
            }

            stats = {
                "modality_count": len(modality_names),
                "observation_overlap_rate": obs_overlap_rate,
                "variable_overlap_rate": var_overlap_rate,
                "compatibility_status": (
                    "compatible" if compatible else "issues_detected"
                ),
                "timestamp": datetime.now().isoformat(),
            }

            ir = self._create_validate_ir(modality_names=modality_names)

            logger.info(
                f"Validated compatibility for {len(modality_names)} modalities: "
                f"{'Compatible' if compatible else 'Issues detected'}"
            )

            return validation, stats, ir

        except Exception as e:
            logger.error(f"Error validating modality compatibility: {e}")
            raise

    def load_modality(
        self,
        modality_name: str,
        file_path: str,
        adapter: str,
        dataset_id: Optional[str] = None,
        dataset_type: str = "custom",
        validate: bool = True,
        **kwargs: Any,
    ) -> Tuple[ad.AnnData, Dict[str, Any], AnalysisStep]:
        """
        Load a data file as a modality using the modular adapter system.

        Args:
            modality_name: Name for the new modality
            file_path: Path to the data file
            adapter: Adapter to use (e.g., 'transcriptomics_single_cell', 'proteomics_ms')
            dataset_id: Optional dataset identifier for metadata
            dataset_type: Source type (e.g., 'custom', 'geo', 'local')
            validate: Whether to validate against schema (default: True)
            **kwargs: Additional adapter-specific parameters

        Returns:
            Tuple containing:
            - Loaded AnnData object
            - Statistics dict (shape, adapter, file_path, validation_status)
            - AnalysisStep for provenance tracking
        """
        try:
            # Validate file exists
            file_path_obj = Path(file_path)
            if not file_path_obj.exists():
                raise FileNotFoundError(f"File not found: {file_path}")

            # Check if modality already exists
            if modality_name in self.data_manager.list_modalities():
                raise ValueError(
                    f"Modality '{modality_name}' already exists. "
                    "Use remove_modality() first or choose a different name."
                )

            # Check if adapter is available
            available_adapters = list(self.data_manager.adapters.keys())
            if adapter not in available_adapters:
                raise ValueError(
                    f"Adapter '{adapter}' not available. "
                    f"Available adapters: {', '.join(available_adapters)}"
                )

            # Load using DataManagerV2
            adata = self.data_manager.load_modality(
                name=modality_name,
                source=file_path_obj,
                adapter=adapter,
                validate=validate,
                dataset_id=dataset_id,
                dataset_type=dataset_type,
                **kwargs,
            )

            # Get quality metrics
            quality_metrics = self.data_manager.get_quality_metrics(modality_name)

            stats = {
                "modality_name": modality_name,
                "shape": {"n_obs": adata.n_obs, "n_vars": adata.n_vars},
                "adapter": adapter,
                "file_path": str(file_path_obj),
                "dataset_type": dataset_type,
                "validation_status": "passed" if validate else "skipped",
                "quality_metrics_count": len(
                    [
                        k
                        for k, v in quality_metrics.items()
                        if isinstance(v, (int, float))
                    ]
                ),
                "timestamp": datetime.now().isoformat(),
            }

            ir = self._create_load_ir(
                modality_name=modality_name,
                file_path=str(file_path_obj),
                adapter=adapter,
                dataset_type=dataset_type,
            )

            logger.info(
                f"Loaded modality '{modality_name}' from {file_path_obj.name}: "
                f"{adata.n_obs} obs × {adata.n_vars} vars (adapter: {adapter})"
            )

            return adata, stats, ir

        except Exception as e:
            logger.error(f"Error loading modality '{modality_name}': {e}")
            raise

    # ============================================================
    # Provenance IR Creation Helpers
    # ============================================================

    def _create_list_ir(self, filter_pattern: Optional[str]) -> AnalysisStep:
        """Create provenance IR for list_modalities operation."""
        return AnalysisStep(
            operation="modality_management.list_modalities",
            tool_name="ModalityManagementService.list_modalities",
            description=(
                f"List all available modalities"
                + (f" matching pattern '{filter_pattern}'" if filter_pattern else "")
            ),
            library="lobster",
            imports=[
                "from lobster.tools.modality_management_service import ModalityManagementService"
            ],
            code_template="""# List modalities
service = ModalityManagementService(data_manager)
modality_info, stats, ir = service.list_modalities(
    filter_pattern={{ filter_pattern if filter_pattern else 'None' }}
)
print(f"Found {stats['matched_modalities']} modalities")
for info in modality_info:
    print(f"  - {info['name']}: {info['n_obs']} × {info['n_vars']}")
""",
            parameters={"filter_pattern": filter_pattern},
            parameter_schema={
                "filter_pattern": {
                    "type": "string",
                    "optional": True,
                    "description": "Glob-style pattern to filter modality names",
                }
            },
            input_entities=[],
            output_entities=["modality_list"],
        )

    def _create_get_info_ir(self, modality_name: str) -> AnalysisStep:
        """Create provenance IR for get_modality_info operation."""
        return AnalysisStep(
            operation="modality_management.get_modality_info",
            tool_name="ModalityManagementService.get_modality_info",
            description=f"Retrieve detailed information for modality '{modality_name}'",
            library="lobster",
            imports=[
                "from lobster.tools.modality_management_service import ModalityManagementService"
            ],
            code_template="""# Get modality info
service = ModalityManagementService(data_manager)
info, stats, ir = service.get_modality_info(modality_name="{{ modality_name }}")
print(f"Modality: {info['name']}")
print(f"Shape: {info['shape']['n_obs']} × {info['shape']['n_vars']}")
print(f"Layers: {', '.join(info['layers'])}")
""",
            parameters={"modality_name": modality_name},
            parameter_schema={
                "modality_name": {
                    "type": "string",
                    "required": True,
                    "description": "Name of the modality to inspect",
                }
            },
            input_entities=[modality_name],
            output_entities=["modality_info"],
        )

    def _create_remove_ir(self, modality_name: str) -> AnalysisStep:
        """Create provenance IR for remove_modality operation."""
        return AnalysisStep(
            operation="modality_management.remove_modality",
            tool_name="ModalityManagementService.remove_modality",
            description=f"Remove modality '{modality_name}' from workspace",
            library="lobster",
            imports=[
                "from lobster.tools.modality_management_service import ModalityManagementService"
            ],
            code_template="""# Remove modality
service = ModalityManagementService(data_manager)
success, stats, ir = service.remove_modality(modality_name="{{ modality_name }}")
if success:
    print(f"Removed modality: {stats['removed_modality']}")
    print(f"Remaining modalities: {stats['remaining_modalities']}")
""",
            parameters={"modality_name": modality_name},
            parameter_schema={
                "modality_name": {
                    "type": "string",
                    "required": True,
                    "description": "Name of the modality to remove",
                }
            },
            input_entities=[modality_name],
            output_entities=[],
        )

    def _create_validate_ir(self, modality_names: List[str]) -> AnalysisStep:
        """Create provenance IR for validate_compatibility operation."""
        return AnalysisStep(
            operation="modality_management.validate_compatibility",
            tool_name="ModalityManagementService.validate_compatibility",
            description=f"Validate compatibility between {len(modality_names)} modalities",
            library="lobster",
            imports=[
                "from lobster.tools.modality_management_service import ModalityManagementService"
            ],
            code_template="""# Validate modality compatibility
service = ModalityManagementService(data_manager)
validation, stats, ir = service.validate_compatibility(
    modality_names={{ modality_names }}
)
print(f"Compatibility: {'Compatible' if validation['compatible'] else 'Issues detected'}")
print(f"Observation overlap: {validation['observation_overlap_rate']:.1%}")
print(f"Recommendations: {', '.join(validation['recommendations'])}")
""",
            parameters={"modality_names": modality_names},
            parameter_schema={
                "modality_names": {
                    "type": "list",
                    "required": True,
                    "description": "List of modality names to validate for compatibility",
                }
            },
            input_entities=modality_names,
            output_entities=["compatibility_report"],
        )

    def _create_load_ir(
        self, modality_name: str, file_path: str, adapter: str, dataset_type: str
    ) -> AnalysisStep:
        """Create provenance IR for load_modality operation."""
        return AnalysisStep(
            operation="modality_management.load_modality",
            tool_name="ModalityManagementService.load_modality",
            description=f"Load data file as modality '{modality_name}' using {adapter} adapter",
            library="lobster",
            imports=[
                "from lobster.tools.modality_management_service import ModalityManagementService"
            ],
            code_template="""# Load modality from file
service = ModalityManagementService(data_manager)
adata, stats, ir = service.load_modality(
    modality_name="{{ modality_name }}",
    file_path="{{ file_path }}",
    adapter="{{ adapter }}",
    dataset_type="{{ dataset_type }}",
    validate=True
)
print(f"Loaded modality: {stats['modality_name']}")
print(f"Shape: {stats['shape']['n_obs']} × {stats['shape']['n_vars']}")
print(f"Adapter: {stats['adapter']}")
""",
            parameters={
                "modality_name": modality_name,
                "file_path": file_path,
                "adapter": adapter,
                "dataset_type": dataset_type,
            },
            parameter_schema={
                "modality_name": {
                    "type": "string",
                    "required": True,
                    "description": "Name for the new modality",
                },
                "file_path": {
                    "type": "string",
                    "required": True,
                    "description": "Path to the data file",
                },
                "adapter": {
                    "type": "string",
                    "required": True,
                    "description": "Adapter to use for loading",
                },
                "dataset_type": {
                    "type": "string",
                    "optional": True,
                    "default": "custom",
                    "description": "Source type (custom, geo, local)",
                },
            },
            input_entities=[file_path],
            output_entities=[modality_name],
        )
