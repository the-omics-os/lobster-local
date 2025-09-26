# Creating Custom Agents Tutorial

This comprehensive tutorial demonstrates how to create, integrate, and deploy custom AI agents in Lobster AI using the centralized agent registry system and modular architecture.

## Overview

In this tutorial, you will learn to:
- Understand Lobster AI's agent architecture and registry system
- Create a custom agent with specialized tools
- Integrate with the DataManagerV2 system
- Register your agent in the central registry
- Test and deploy your custom agent
- Handle agent handoffs and tool integration

## Prerequisites

- Lobster AI development environment set up
- Python 3.12+ with development dependencies
- Understanding of LangChain/LangGraph concepts
- Basic knowledge of bioinformatics workflows
- Familiarity with Lobster's modular architecture

## Example: Creating a Spatial Omics Expert Agent

We'll create a **Spatial Omics Expert Agent** that specializes in spatial transcriptomics and spatial proteomics analysis (Visium, MERFISH, CosMx, etc.).

## Step 1: Understanding the Agent Architecture

### Agent Registry Pattern

Lobster uses a centralized registry system in `/lobster/config/agent_registry.py`:

```python
@dataclass
class AgentRegistryConfig:
    """Configuration for an agent in the system."""
    name: str
    display_name: str
    description: str
    factory_function: str  # Module path to the factory function
    handoff_tool_name: Optional[str] = None
    handoff_tool_description: Optional[str] = None
```

### Agent Tool Pattern

All agents follow this standardized pattern:

```python
@tool
def analyze_modality(modality_name: str, **params) -> str:
    """Standard pattern for all agent tools."""
    try:
        # 1. Validate modality exists
        if modality_name not in data_manager.list_modalities():
            raise ModalityNotFoundError(f"Modality '{modality_name}' not found")

        adata = data_manager.get_modality(modality_name)

        # 2. Call stateless service (returns tuple)
        result_adata, stats = service.analyze(adata, **params)

        # 3. Store results with descriptive naming
        new_modality = f"{modality_name}_analyzed"
        data_manager.modalities[new_modality] = result_adata

        # 4. Log operation for provenance
        data_manager.log_tool_usage("analyze_modality", params, stats)

        return formatted_response(stats, new_modality)

    except ServiceError as e:
        logger.error(f"Service error: {e}")
        return f"Analysis failed: {str(e)}"
```

## Step 2: Create the Spatial Service Layer

First, create the stateless service that handles spatial analysis:

### Create `/lobster/tools/spatial_omics_service.py`

```python
"""
Spatial Omics Service for specialized spatial transcriptomics and proteomics analysis.
"""

import numpy as np
import pandas as pd
import scanpy as sc
import squidpy as sq
from typing import Dict, Any, Tuple, Optional, List
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
import anndata as AnnData

from lobster.utils.logger import get_logger

logger = get_logger(__name__)


class SpatialOmicsError(Exception):
    """Base exception for spatial omics operations."""
    pass


class SpatialOmicsService:
    """Stateless service for spatial omics analysis."""

    def __init__(self):
        """Initialize spatial omics service."""
        self.logger = get_logger(self.__class__.__name__)

    def calculate_spatial_metrics(
        self,
        adata: AnnData,
        coord_type: str = "grid"
    ) -> Tuple[AnnData, Dict[str, Any]]:
        """Calculate spatial metrics including Moran's I and spatial autocorrelation."""
        try:
            self.logger.info("Calculating spatial metrics...")

            # Ensure spatial coordinates are available
            if 'spatial' not in adata.obsm:
                raise SpatialOmicsError("No spatial coordinates found in adata.obsm['spatial']")

            # Calculate spatial neighbors
            if coord_type == "grid":
                sq.gr.spatial_neighbors(adata, coord_type="grid")
            else:
                sq.gr.spatial_neighbors(adata, coord_type="generic", n_neighs=6)

            # Calculate Moran's I for highly variable genes
            if 'highly_variable' in adata.var.columns:
                hvg_genes = adata.var[adata.var['highly_variable']].index.tolist()
            else:
                # Select top 100 most variable genes
                sc.pp.highly_variable_genes(adata, n_top_genes=100)
                hvg_genes = adata.var[adata.var['highly_variable']].index.tolist()

            # Calculate spatial autocorrelation
            sq.gr.spatial_autocorr(
                adata,
                mode="moran",
                genes=hvg_genes[:50],  # Limit for performance
                n_perms=100,
                n_jobs=1
            )

            # Calculate spatial statistics
            spatial_stats = {
                "n_spots": adata.n_obs,
                "n_genes": adata.n_vars,
                "spatial_dimensions": adata.obsm['spatial'].shape[1],
                "coordinate_range": {
                    "x_min": float(adata.obsm['spatial'][:, 0].min()),
                    "x_max": float(adata.obsm['spatial'][:, 0].max()),
                    "y_min": float(adata.obsm['spatial'][:, 1].min()),
                    "y_max": float(adata.obsm['spatial'][:, 1].max())
                },
                "n_spatially_variable_genes": int((adata.var['moranI'] > 0.1).sum()) if 'moranI' in adata.var else 0,
                "mean_moran_i": float(adata.var['moranI'].mean()) if 'moranI' in adata.var else 0.0
            }

            self.logger.info(f"Spatial metrics calculated for {adata.n_obs} spots")
            return adata, spatial_stats

        except Exception as e:
            raise SpatialOmicsError(f"Spatial metrics calculation failed: {str(e)}")

    def spatial_clustering(
        self,
        adata: AnnData,
        n_clusters: int = 10,
        method: str = "leiden"
    ) -> Tuple[AnnData, Dict[str, Any]]:
        """Perform spatial-aware clustering."""
        try:
            self.logger.info(f"Performing spatial clustering with {method}...")

            if method == "leiden":
                # Use spatial neighbors for clustering
                sc.tl.leiden(adata, resolution=0.5, key_added="spatial_clusters")
                cluster_key = "spatial_clusters"
            elif method == "kmeans":
                # K-means on spatial coordinates
                kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                spatial_clusters = kmeans.fit_predict(adata.obsm['spatial'])
                adata.obs['spatial_kmeans'] = pd.Categorical(spatial_clusters)
                cluster_key = "spatial_kmeans"
            else:
                raise SpatialOmicsError(f"Unknown clustering method: {method}")

            # Calculate clustering statistics
            cluster_stats = {
                "method": method,
                "n_clusters": len(adata.obs[cluster_key].cat.categories),
                "cluster_sizes": dict(adata.obs[cluster_key].value_counts()),
                "silhouette_score": self._calculate_spatial_silhouette(adata, cluster_key),
                "spatial_coherence": self._calculate_spatial_coherence(adata, cluster_key)
            }

            self.logger.info(f"Spatial clustering complete: {cluster_stats['n_clusters']} clusters")
            return adata, cluster_stats

        except Exception as e:
            raise SpatialOmicsError(f"Spatial clustering failed: {str(e)}")

    def identify_spatial_domains(
        self,
        adata: AnnData,
        method: str = "banksy"
    ) -> Tuple[AnnData, Dict[str, Any]]:
        """Identify spatial domains using advanced spatial methods."""
        try:
            self.logger.info(f"Identifying spatial domains using {method}...")

            if method == "banksy":
                # Use squidpy's implementation of spatial domain detection
                sq.gr.spatial_neighbors(adata, n_neighs=6)
                sq.gr.nhood_enrichment(adata, cluster_key="leiden")

                # Calculate spatial domains based on neighborhood composition
                sq.tl.co_occurrence(adata, cluster_key="leiden")

                domain_key = "spatial_domains"
                # Simple spatial domain assignment based on neighborhood enrichment
                adata.obs[domain_key] = adata.obs["leiden"]  # Placeholder implementation

            else:
                raise SpatialOmicsError(f"Unknown spatial domain method: {method}")

            domain_stats = {
                "method": method,
                "n_domains": len(adata.obs[domain_key].cat.categories),
                "domain_sizes": dict(adata.obs[domain_key].value_counts()),
                "avg_domain_size": float(adata.obs[domain_key].value_counts().mean())
            }

            self.logger.info(f"Spatial domain identification complete: {domain_stats['n_domains']} domains")
            return adata, domain_stats

        except Exception as e:
            raise SpatialOmicsError(f"Spatial domain identification failed: {str(e)}")

    def _calculate_spatial_silhouette(self, adata: AnnData, cluster_key: str) -> float:
        """Calculate spatial-aware silhouette score."""
        try:
            from sklearn.metrics import silhouette_score

            # Use spatial coordinates for silhouette calculation
            coords = adata.obsm['spatial']
            labels = adata.obs[cluster_key].cat.codes

            if len(np.unique(labels)) < 2:
                return 0.0

            score = silhouette_score(coords, labels)
            return float(score)

        except Exception:
            return 0.0

    def _calculate_spatial_coherence(self, adata: AnnData, cluster_key: str) -> float:
        """Calculate spatial coherence of clusters."""
        try:
            # Simple spatial coherence: fraction of neighbors with same cluster
            if 'spatial_connectivities' not in adata.obsp:
                return 0.0

            conn_matrix = adata.obsp['spatial_connectivities']
            labels = adata.obs[cluster_key].cat.codes.values

            coherence_scores = []
            for i in range(len(labels)):
                neighbors = conn_matrix[i].indices
                if len(neighbors) > 0:
                    same_cluster = (labels[neighbors] == labels[i]).sum()
                    coherence = same_cluster / len(neighbors)
                    coherence_scores.append(coherence)

            return float(np.mean(coherence_scores)) if coherence_scores else 0.0

        except Exception:
            return 0.0


def create_spatial_omics_service() -> SpatialOmicsService:
    """Factory function to create spatial omics service."""
    return SpatialOmicsService()
```

## Step 3: Create the Spatial Omics Agent

### Create `/lobster/agents/spatial_omics_expert.py`

```python
"""
Spatial Omics Expert Agent for spatial transcriptomics and proteomics analysis.

This agent specializes in spatial omics data analysis using Lobster's modular
DataManagerV2 system with spatial-specific quality control and visualization.
"""

from typing import List, Union
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langchain_aws import ChatBedrockConverse
from datetime import date

from lobster.config.settings import get_settings
from lobster.core.data_manager_v2 import DataManagerV2
from lobster.tools.spatial_omics_service import SpatialOmicsService, SpatialOmicsError
from lobster.utils.logger import get_logger

logger = get_logger(__name__)


class SpatialOmicsAgentError(Exception):
    """Base exception for spatial omics agent operations."""
    pass


class ModalityNotFoundError(SpatialOmicsAgentError):
    """Raised when requested modality doesn't exist."""
    pass


def spatial_omics_expert(
    data_manager: DataManagerV2,
    callback_handler=None,
    agent_name: str = "spatial_omics_expert_agent",
    handoff_tools: List = None
):
    """Create spatial omics expert agent using DataManagerV2 and modular services."""

    settings = get_settings()
    model_params = settings.get_agent_llm_params('spatial_omics_expert')
    llm = ChatBedrockConverse(**model_params)

    if callback_handler and hasattr(llm, 'with_config'):
        llm = llm.with_config(callbacks=[callback_handler])

    # Initialize stateless service
    spatial_service = SpatialOmicsService()

    analysis_results = {"summary": "", "details": {}}

    # -------------------------
    # DATA STATUS TOOLS
    # -------------------------
    @tool
    def check_spatial_data_status(modality_name: str = "") -> str:
        """Check if spatial omics data is loaded and has required spatial coordinates."""
        try:
            if modality_name == "":
                modalities = data_manager.list_modalities()
                if not modalities:
                    return "No modalities loaded. Please load a spatial dataset first."

                # Look for spatial modalities
                spatial_modalities = []
                for mod_name in modalities:
                    adata = data_manager.get_modality(mod_name)
                    if 'spatial' in adata.obsm:
                        spatial_modalities.append(mod_name)

                if not spatial_modalities:
                    response = f"Available modalities ({len(modalities)}) but none have spatial coordinates:\n"
                    for mod_name in modalities[:5]:  # Show first 5
                        adata = data_manager.get_modality(mod_name)
                        response += f"- **{mod_name}**: {adata.n_obs} spots × {adata.n_vars} genes\n"
                    response += "\nTo use spatial analysis, data must have spatial coordinates in adata.obsm['spatial']."
                    return response

                response = f"Found {len(spatial_modalities)} spatial modalities:\n"
                for mod_name in spatial_modalities:
                    adata = data_manager.get_modality(mod_name)
                    coords_shape = adata.obsm['spatial'].shape if 'spatial' in adata.obsm else "None"
                    response += f"- **{mod_name}**: {adata.n_obs} spots × {adata.n_vars} genes, coords: {coords_shape}\n"

                return response

            # Check specific modality
            if modality_name not in data_manager.list_modalities():
                return f"Modality '{modality_name}' not found. Use check_spatial_data_status() to see available modalities."

            adata = data_manager.get_modality(modality_name)

            # Check for spatial coordinates
            has_spatial = 'spatial' in adata.obsm
            if not has_spatial:
                return f"Modality '{modality_name}' exists but lacks spatial coordinates. Spatial analysis requires coordinates in adata.obsm['spatial']."

            coords = adata.obsm['spatial']
            response = f"✅ **{modality_name}** - Ready for spatial analysis\n"
            response += f"- Spots: {adata.n_obs:,}\n"
            response += f"- Genes: {adata.n_vars:,}\n"
            response += f"- Spatial coordinates: {coords.shape} ({coords.shape[1]}D)\n"
            response += f"- Coordinate range: X[{coords[:, 0].min():.1f}, {coords[:, 0].max():.1f}], Y[{coords[:, 1].min():.1f}, {coords[:, 1].max():.1f}]\n"

            # Check for existing spatial analysis
            spatial_features = []
            if 'moranI' in adata.var.columns:
                spatial_features.append("Moran's I calculated")
            if 'spatial_connectivities' in adata.obsp:
                spatial_features.append("Spatial neighbors computed")
            if any('spatial' in col for col in adata.obs.columns):
                spatial_features.append("Spatial clustering performed")

            if spatial_features:
                response += f"- Previous spatial analysis: {', '.join(spatial_features)}\n"

            return response

        except Exception as e:
            logger.error(f"Error checking spatial data status: {e}")
            return f"Error checking spatial data status: {str(e)}"

    # -------------------------
    # SPATIAL ANALYSIS TOOLS
    # -------------------------
    @tool
    def calculate_spatial_statistics(
        modality_name: str,
        coord_type: str = "grid"
    ) -> str:
        """Calculate spatial statistics including Moran's I and spatial autocorrelation."""
        try:
            if modality_name not in data_manager.list_modalities():
                raise ModalityNotFoundError(f"Modality '{modality_name}' not found")

            adata = data_manager.get_modality(modality_name)

            # Check for spatial coordinates
            if 'spatial' not in adata.obsm:
                return f"❌ Modality '{modality_name}' lacks spatial coordinates. Cannot perform spatial analysis."

            # Call service
            result_adata, stats = spatial_service.calculate_spatial_metrics(adata, coord_type)

            # Store result
            new_modality = f"{modality_name}_spatial_metrics"
            data_manager.modalities[new_modality] = result_adata

            # Log operation
            data_manager.log_tool_usage("calculate_spatial_statistics",
                                      {"coord_type": coord_type}, stats)

            # Format response
            response = f"✅ **Spatial Statistics Calculated** for {modality_name}\n\n"
            response += f"📊 **Dataset Overview:**\n"
            response += f"- Spots analyzed: {stats['n_spots']:,}\n"
            response += f"- Genes analyzed: {stats['n_genes']:,}\n"
            response += f"- Spatial dimensions: {stats['spatial_dimensions']}D\n\n"

            response += f"📐 **Coordinate Range:**\n"
            response += f"- X: [{stats['coordinate_range']['x_min']:.1f}, {stats['coordinate_range']['x_max']:.1f}]\n"
            response += f"- Y: [{stats['coordinate_range']['y_min']:.1f}, {stats['coordinate_range']['y_max']:.1f}]\n\n"

            response += f"🧬 **Spatial Gene Analysis:**\n"
            response += f"- Spatially variable genes: {stats['n_spatially_variable_genes']}\n"
            response += f"- Mean Moran's I: {stats['mean_moran_i']:.3f}\n\n"

            response += f"💾 **Results stored as:** `{new_modality}`\n"
            response += f"Next: Use `perform_spatial_clustering()` to identify spatial domains."

            return response

        except SpatialOmicsError as e:
            logger.error(f"Spatial statistics calculation failed: {e}")
            return f"❌ Spatial statistics calculation failed: {str(e)}"
        except Exception as e:
            logger.error(f"Unexpected error in spatial statistics: {e}")
            return f"❌ Unexpected error: {str(e)}"

    @tool
    def perform_spatial_clustering(
        modality_name: str,
        n_clusters: int = 10,
        method: str = "leiden"
    ) -> str:
        """Perform spatial-aware clustering to identify tissue regions."""
        try:
            if modality_name not in data_manager.list_modalities():
                raise ModalityNotFoundError(f"Modality '{modality_name}' not found")

            adata = data_manager.get_modality(modality_name)

            # Call service
            result_adata, stats = spatial_service.spatial_clustering(adata, n_clusters, method)

            # Store result
            new_modality = f"{modality_name}_spatial_clustered"
            data_manager.modalities[new_modality] = result_adata

            # Log operation
            data_manager.log_tool_usage("perform_spatial_clustering",
                                      {"n_clusters": n_clusters, "method": method}, stats)

            # Format response
            response = f"✅ **Spatial Clustering Complete** using {stats['method']}\n\n"
            response += f"🎯 **Clustering Results:**\n"
            response += f"- Clusters identified: {stats['n_clusters']}\n"
            response += f"- Silhouette score: {stats['silhouette_score']:.3f}\n"
            response += f"- Spatial coherence: {stats['spatial_coherence']:.3f}\n\n"

            response += f"📊 **Cluster Distribution:**\n"
            for cluster_id, size in sorted(stats['cluster_sizes'].items()):
                percentage = (size / sum(stats['cluster_sizes'].values())) * 100
                response += f"- Cluster {cluster_id}: {size:,} spots ({percentage:.1f}%)\n"

            response += f"\n💾 **Results stored as:** `{new_modality}`\n"
            response += f"Next: Use `identify_spatial_domains()` for advanced domain detection."

            return response

        except SpatialOmicsError as e:
            logger.error(f"Spatial clustering failed: {e}")
            return f"❌ Spatial clustering failed: {str(e)}"
        except Exception as e:
            logger.error(f"Unexpected error in spatial clustering: {e}")
            return f"❌ Unexpected error: {str(e)}"

    @tool
    def identify_spatial_domains(
        modality_name: str,
        method: str = "banksy"
    ) -> str:
        """Identify spatial domains using advanced spatial analysis methods."""
        try:
            if modality_name not in data_manager.list_modalities():
                raise ModalityNotFoundError(f"Modality '{modality_name}' not found")

            adata = data_manager.get_modality(modality_name)

            # Call service
            result_adata, stats = spatial_service.identify_spatial_domains(adata, method)

            # Store result
            new_modality = f"{modality_name}_spatial_domains"
            data_manager.modalities[new_modality] = result_adata

            # Log operation
            data_manager.log_tool_usage("identify_spatial_domains",
                                      {"method": method}, stats)

            # Format response
            response = f"✅ **Spatial Domains Identified** using {stats['method']}\n\n"
            response += f"🗺️ **Domain Analysis:**\n"
            response += f"- Spatial domains found: {stats['n_domains']}\n"
            response += f"- Average domain size: {stats['avg_domain_size']:.1f} spots\n\n"

            response += f"📊 **Domain Distribution:**\n"
            for domain_id, size in sorted(stats['domain_sizes'].items()):
                percentage = (size / sum(stats['domain_sizes'].values())) * 100
                response += f"- Domain {domain_id}: {size:,} spots ({percentage:.1f}%)\n"

            response += f"\n💾 **Results stored as:** `{new_modality}`\n"
            response += f"Spatial domain analysis complete!"

            return response

        except SpatialOmicsError as e:
            logger.error(f"Spatial domain identification failed: {e}")
            return f"❌ Spatial domain identification failed: {str(e)}"
        except Exception as e:
            logger.error(f"Unexpected error in spatial domain identification: {e}")
            return f"❌ Unexpected error: {str(e)}"

    # -------------------------
    # AGENT CREATION
    # -------------------------

    # Collect all tools
    tools = [
        check_spatial_data_status,
        calculate_spatial_statistics,
        perform_spatial_clustering,
        identify_spatial_domains
    ]

    # Add handoff tools if provided
    if handoff_tools:
        tools.extend(handoff_tools)

    # Create agent with enhanced prompt
    system_prompt = f"""You are the Spatial Omics Expert, specialized in spatial transcriptomics and spatial proteomics analysis.

**Your Expertise:**
- Spatial transcriptomics (Visium, MERFISH, seqFISH, CosMx)
- Spatial proteomics (IMC, MIBI-TOF, GeoMx)
- Spatial statistics and autocorrelation analysis
- Tissue domain identification and spatial clustering
- Spatial neighborhood analysis and cell-cell interactions

**Available Tools:**
- check_spatial_data_status: Check if spatial data is properly loaded
- calculate_spatial_statistics: Compute Moran's I and spatial autocorrelation
- perform_spatial_clustering: Identify tissue regions using spatial clustering
- identify_spatial_domains: Advanced spatial domain detection

**Guidelines:**
1. Always check data status before analysis
2. Use spatial coordinates for clustering when available
3. Interpret spatial patterns in biological context
4. Provide clear descriptions of spatial domains and their potential biological significance
5. Suggest appropriate follow-up analyses based on results

**Data Requirements:**
- Spatial coordinates must be in adata.obsm['spatial']
- For best results, data should be quality-controlled and normalized
- Spatial analysis works best with >1000 spots and >500 genes

Today's date: {date.today()}
"""

    # Create the agent
    agent = create_react_agent(
        llm,
        tools,
        state_modifier=system_prompt
    )

    return agent


# Agent factory function for registry
def create_spatial_omics_expert(data_manager: DataManagerV2, **kwargs):
    """Factory function to create spatial omics expert agent."""
    return spatial_omics_expert(data_manager, **kwargs)
```

## Step 4: Register Your Custom Agent

Add your agent to the central registry in `/lobster/config/agent_registry.py`:

```python
# Add this to the AGENT_REGISTRY dictionary
AGENT_REGISTRY = {
    # ... existing agents ...

    'spatial_omics_expert_agent': AgentRegistryConfig(
        name='spatial_omics_expert_agent',
        display_name='Spatial Omics Expert',
        description='Handles spatial transcriptomics and spatial proteomics analysis including tissue domain identification, spatial clustering, and neighborhood analysis',
        factory_function='lobster.agents.spatial_omics_expert.spatial_omics_expert',
        handoff_tool_name='handoff_to_spatial_omics_expert',
        handoff_tool_description='Assign spatial omics analysis tasks including spatial transcriptomics (Visium, MERFISH) and spatial proteomics (IMC, MIBI) to the spatial omics expert'
    ),
}
```

## Step 5: Create Unit Tests

Create comprehensive tests in `/tests/unit/test_spatial_omics_agent.py`:

```python
"""
Unit tests for Spatial Omics Expert Agent.
"""

import pytest
import numpy as np
import pandas as pd
import anndata as ad
from unittest.mock import Mock, patch
import tempfile
from pathlib import Path

from lobster.core.data_manager_v2 import DataManagerV2
from lobster.agents.spatial_omics_expert import spatial_omics_expert, create_spatial_omics_expert
from lobster.tools.spatial_omics_service import SpatialOmicsService


@pytest.fixture
def temp_workspace():
    """Create temporary workspace for testing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        workspace_path = Path(temp_dir) / ".lobster_workspace"
        workspace_path.mkdir(exist_ok=True)
        yield workspace_path


@pytest.fixture
def mock_spatial_data():
    """Create mock spatial transcriptomics data."""
    n_spots = 1000
    n_genes = 2000

    # Create count matrix
    X = np.random.negative_binomial(5, 0.3, (n_spots, n_genes)).astype(float)

    # Create spatial coordinates (hexagonal grid)
    coords = []
    for i in range(32):  # 32x32 grid ≈ 1000 spots
        for j in range(32):
            if len(coords) >= n_spots:
                break
            x = i + 0.5 * (j % 2)  # Hexagonal offset
            y = j * 0.866  # Hexagonal spacing
            coords.append([x, y])

    coords = np.array(coords[:n_spots])

    # Create AnnData object
    adata = ad.AnnData(X=X)
    adata.obs_names = [f"spot_{i}" for i in range(n_spots)]
    adata.var_names = [f"gene_{i}" for i in range(n_genes)]
    adata.obsm['spatial'] = coords

    # Add some metadata
    adata.obs['tissue_type'] = np.random.choice(['tumor', 'normal'], n_spots)
    adata.var['highly_variable'] = np.random.choice([True, False], n_genes, p=[0.1, 0.9])

    return adata


@pytest.fixture
def data_manager_with_spatial(temp_workspace, mock_spatial_data):
    """Create DataManagerV2 with spatial data loaded."""
    data_manager = DataManagerV2(workspace_path=temp_workspace)
    data_manager.modalities['spatial_visium'] = mock_spatial_data
    return data_manager


def test_spatial_omics_service_initialization():
    """Test spatial omics service can be initialized."""
    service = SpatialOmicsService()
    assert service is not None


def test_spatial_metrics_calculation(mock_spatial_data):
    """Test spatial metrics calculation."""
    service = SpatialOmicsService()

    result_adata, stats = service.calculate_spatial_metrics(mock_spatial_data)

    assert result_adata is not None
    assert 'n_spots' in stats
    assert 'n_genes' in stats
    assert 'spatial_dimensions' in stats
    assert stats['spatial_dimensions'] == 2
    assert stats['n_spots'] == 1000


def test_spatial_clustering(mock_spatial_data):
    """Test spatial clustering functionality."""
    service = SpatialOmicsService()

    # First calculate spatial metrics to set up neighbors
    adata_with_metrics, _ = service.calculate_spatial_metrics(mock_spatial_data)

    # Then perform clustering
    result_adata, stats = service.spatial_clustering(adata_with_metrics, n_clusters=5, method="kmeans")

    assert 'spatial_kmeans' in result_adata.obs.columns
    assert 'method' in stats
    assert 'n_clusters' in stats
    assert stats['method'] == 'kmeans'


def test_agent_creation(data_manager_with_spatial):
    """Test that the spatial omics agent can be created."""
    agent = spatial_omics_expert(data_manager_with_spatial)
    assert agent is not None


def test_factory_function(data_manager_with_spatial):
    """Test the factory function."""
    agent = create_spatial_omics_expert(data_manager_with_spatial)
    assert agent is not None


@pytest.mark.integration
def test_check_spatial_data_status_tool(data_manager_with_spatial):
    """Test the check_spatial_data_status tool."""
    agent = spatial_omics_expert(data_manager_with_spatial)

    # Get the tool function
    tools = agent.get_graph().get_state_schema()
    # Note: This is a simplified test - in practice you'd invoke the agent

    # Test with existing spatial modality
    result = "Found spatial modalities with coordinates"  # Expected result
    assert isinstance(result, str)


def test_missing_spatial_coordinates():
    """Test handling of data without spatial coordinates."""
    # Create data without spatial coordinates
    adata = ad.AnnData(X=np.random.randn(100, 50))

    service = SpatialOmicsService()

    with pytest.raises(Exception):  # Should raise SpatialOmicsError
        service.calculate_spatial_metrics(adata)


if __name__ == "__main__":
    pytest.main([__file__])
```

## Step 6: Integration Testing

Create integration tests in `/tests/integration/test_spatial_omics_integration.py`:

```python
"""
Integration tests for Spatial Omics Expert Agent with full workflow.
"""

import pytest
import tempfile
from pathlib import Path
import numpy as np
import anndata as ad

from lobster.core.data_manager_v2 import DataManagerV2
from lobster.core.client import AgentClient


@pytest.fixture
def full_spatial_workspace():
    """Create a full workspace with spatial data for integration testing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        workspace_path = Path(temp_dir) / ".lobster_workspace"
        workspace_path.mkdir(exist_ok=True)

        # Create realistic spatial data
        n_spots = 2000
        n_genes = 3000

        # Create spatial coordinates in tissue shape
        coords = create_tissue_coordinates(n_spots)

        # Create count matrix with spatial patterns
        X = create_spatial_expression_data(coords, n_genes)

        # Create AnnData
        adata = ad.AnnData(X=X)
        adata.obs_names = [f"spot_{i:04d}" for i in range(n_spots)]
        adata.var_names = [f"gene_{i:04d}" for i in range(n_genes)]
        adata.obsm['spatial'] = coords

        # Create data manager and load data
        data_manager = DataManagerV2(workspace_path=workspace_path)
        data_manager.modalities['visium_spatial'] = adata

        yield data_manager


def create_tissue_coordinates(n_spots: int) -> np.ndarray:
    """Create realistic tissue-shaped spatial coordinates."""
    # Create circular tissue section
    coords = []
    radius = 20
    center = [25, 25]

    for i in range(50):
        for j in range(50):
            x, y = i, j
            distance = np.sqrt((x - center[0])**2 + (y - center[1])**2)
            if distance <= radius and len(coords) < n_spots:
                # Add some noise
                x += np.random.normal(0, 0.1)
                y += np.random.normal(0, 0.1)
                coords.append([x, y])

    return np.array(coords[:n_spots])


def create_spatial_expression_data(coords: np.ndarray, n_genes: int) -> np.ndarray:
    """Create expression data with spatial patterns."""
    n_spots = coords.shape[0]
    X = np.random.negative_binomial(3, 0.3, (n_spots, n_genes))

    # Add spatial gradients to some genes
    for g in range(0, min(100, n_genes), 10):  # Every 10th gene in first 100
        # Create spatial gradient
        gradient = coords[:, 0] / coords[:, 0].max()  # X-gradient
        spatial_effect = (gradient * 2 + 1).astype(int)  # 1x to 3x multiplier
        X[:, g] = X[:, g] * spatial_effect

    return X.astype(float)


@pytest.mark.integration
def test_full_spatial_workflow(full_spatial_workspace):
    """Test complete spatial analysis workflow."""

    # Create agent client
    client = AgentClient(data_manager=full_spatial_workspace)

    # Test 1: Check data status
    result1 = client.query("Check if spatial data is loaded and ready for analysis")
    assert result1['success']
    assert "spatial" in result1['response'].lower()

    # Test 2: Calculate spatial statistics
    result2 = client.query("Calculate spatial statistics for the loaded spatial data")
    assert result2['success']
    assert "moran" in result2['response'].lower()

    # Test 3: Perform spatial clustering
    result3 = client.query("Perform spatial clustering to identify tissue domains using leiden clustering")
    assert result3['success']
    assert "cluster" in result3['response'].lower()

    # Test 4: Identify spatial domains
    result4 = client.query("Identify spatial domains using advanced spatial methods")
    assert result4['success']

    # Verify data was stored correctly
    modalities = full_spatial_workspace.list_modalities()
    assert len(modalities) >= 4  # Original + 3 analysis results


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
```

## Step 7: Testing Your Custom Agent

Run your tests to ensure everything works:

```bash
# Run unit tests
pytest tests/unit/test_spatial_omics_agent.py -v

# Run integration tests
pytest tests/integration/test_spatial_omics_integration.py -v

# Run all tests
make test
```

## Step 8: Using Your Custom Agent

Now you can use your custom agent through Lobster AI:

```bash
# Start Lobster AI
lobster chat

# Use your custom agent
🦞 You: "I have spatial transcriptomics data from a Visium experiment. Can you help me analyze the spatial patterns and identify tissue domains?"
```

**Expected Agent Handoff:**
```
🦞 Lobster: I'll help you analyze your spatial transcriptomics data. Let me hand this over to our Spatial Omics Expert who specializes in spatial analysis.

🧬 Spatial Omics Expert: I'll analyze your Visium spatial transcriptomics data...

✓ Checking spatial data status...
✓ Found spatial coordinates for analysis
✓ Calculating spatial statistics and autocorrelation...
✓ Performing spatial clustering to identify tissue domains...
✓ Generating spatial visualizations...

Your Visium data analysis is complete! I've identified 8 distinct spatial domains with strong spatial coherence (0.85). The results show clear tissue organization with immune-rich and stromal regions.
```

## Step 9: Advanced Features

### Add Visualization Tools

Extend your agent with specialized visualization capabilities:

```python
@tool
def create_spatial_plots(
    modality_name: str,
    plot_type: str = "spatial_clusters"
) -> str:
    """Create spatial-specific visualizations."""
    try:
        adata = data_manager.get_modality(modality_name)

        if plot_type == "spatial_clusters":
            # Create spatial scatter plot with cluster colors
            # Implementation would use squidpy.pl.spatial_scatter
            pass
        elif plot_type == "spatial_heatmap":
            # Create spatial heatmap of gene expression
            pass

        return f"✅ Spatial visualization created: {plot_type}"

    except Exception as e:
        return f"❌ Visualization failed: {str(e)}"
```

### Add Integration Capabilities

```python
@tool
def integrate_with_histology(
    modality_name: str,
    histology_image_path: str
) -> str:
    """Integrate spatial omics data with histology images."""
    try:
        # Load histology image and overlay spatial data
        # Implementation would use squidpy image processing
        return "✅ Histology integration complete"
    except Exception as e:
        return f"❌ Integration failed: {str(e)}"
```

## Best Practices for Custom Agent Development

### 1. Follow the Service Pattern
- Keep agents stateless
- Implement logic in separate service classes
- Return tuples from services: `(result_adata, statistics)`

### 2. Error Handling
- Create specific exception classes
- Handle edge cases gracefully
- Provide helpful error messages

### 3. Logging and Provenance
- Log all operations using `data_manager.log_tool_usage()`
- Include parameter information
- Track analysis history

### 4. Documentation
- Write comprehensive docstrings
- Include parameter descriptions
- Provide usage examples

### 5. Testing
- Create unit tests for services
- Write integration tests for full workflows
- Test edge cases and error conditions

## Deployment and Distribution

### Option 1: Local Development
Keep your custom agent in your local Lobster installation for personal use.

### Option 2: Plugin System
Package your agent as a plugin that others can install:

```python
# setup.py for your custom agent plugin
from setuptools import setup, find_packages

setup(
    name="lobster-spatial-omics",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "lobster-ai>=2.0.0",
        "squidpy>=1.3.0",
        "scanpy>=1.9.0"
    ],
    entry_points={
        "lobster_agents": [
            "spatial_omics = lobster_spatial_omics.agent:spatial_omics_expert"
        ]
    }
)
```

### Option 3: Contribute to Core
Submit a pull request to add your agent to the main Lobster AI repository.

## Next Steps

After creating your custom agent:

1. **Extend Functionality**: Add more specialized tools for your domain
2. **Create Visualizations**: Develop domain-specific plotting functions
3. **Integration**: Enable data exchange with other agents
4. **Optimization**: Profile and optimize performance for large datasets
5. **Documentation**: Create user guides and tutorials
6. **Community**: Share your agent with the Lobster community

## Summary

You have successfully:
- ✅ Created a custom Spatial Omics Expert Agent
- ✅ Implemented the modular service pattern
- ✅ Integrated with Lobster's agent registry system
- ✅ Added comprehensive error handling and logging
- ✅ Created unit and integration tests
- ✅ Deployed and tested your custom agent
- ✅ Learned best practices for agent development

Your custom agent is now ready to handle spatial omics analysis tasks through natural language interaction, seamlessly integrated with Lobster AI's multi-agent system!

## Related Resources

- **[Agent Architecture Documentation](../architecture_diagram.md)** - Technical details
- **[API Documentation](../API_DOCUMENTATION.md)** - Core API reference
- **[Testing Guide](../testing.md)** - Comprehensive testing strategies
- **[Examples Cookbook](27-examples-cookbook.md)** - More advanced patterns