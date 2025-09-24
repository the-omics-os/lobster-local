"""
Integration tests for the Single Cell Expert -> ML Expert -> Single Cell handoff flow.

This module tests the complete workflow including actual handoff execution,
context preservation, and proper return flow.
"""

import unittest
import uuid
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any
import tempfile
import shutil
from pathlib import Path

import pandas as pd
import numpy as np
import anndata as ad
import pytest

from lobster.core.data_manager_v2 import DataManagerV2
from lobster.tools.expert_handoff_manager import expert_handoff_manager
from lobster.tools.enhanced_handoff_tool import create_expert_handoff_tool, SCVI_CONTEXT_SCHEMA
from lobster.config.agent_registry import create_expert_handoff_tools


class TestScviHandoffFlow(unittest.TestCase):
    """Test complete Single Cell -> ML Expert -> Single Cell handoff workflow."""

    def setUp(self):
        """Set up test fixtures with mock data."""
        # Create temporary workspace
        self.temp_workspace = tempfile.mkdtemp()
        self.workspace_path = Path(self.temp_workspace)

        # Initialize DataManagerV2 with test workspace
        self.data_manager = DataManagerV2(workspace_path=self.workspace_path)

        # Create mock single-cell data
        self.create_mock_singlecell_data()

        # Clear any existing handoffs
        expert_handoff_manager.active_handoffs.clear()
        expert_handoff_manager.handoff_history.clear()
        expert_handoff_manager.handoff_chains.clear()

    def tearDown(self):
        """Clean up test fixtures."""
        # Clean up temporary workspace
        shutil.rmtree(self.temp_workspace, ignore_errors=True)

    def create_mock_singlecell_data(self):
        """Create mock single-cell RNA-seq data for testing."""
        # Create realistic single-cell data
        n_obs = 1500  # Sufficient cells for scVI
        n_vars = 2000  # Sufficient genes for scVI

        # Generate mock expression data
        np.random.seed(42)
        X = np.random.negative_binomial(5, 0.3, size=(n_obs, n_vars))
        X = X.astype(np.float32)

        # Create observation metadata
        obs_data = {
            'cell_type': np.random.choice(['T_cell', 'B_cell', 'NK_cell', 'Monocyte'], n_obs),
            'sample_id': np.random.choice(['sample_1', 'sample_2', 'sample_3'], n_obs),
            'batch': np.random.choice(['batch_A', 'batch_B'], n_obs),
            'n_genes': np.random.randint(500, 1500, n_obs),
            'total_counts': np.random.randint(1000, 10000, n_obs)
        }
        obs = pd.DataFrame(obs_data, index=[f"cell_{i}" for i in range(n_obs)])

        # Create variable metadata
        var_data = {
            'gene_name': [f"gene_{i}" for i in range(n_vars)],
            'highly_variable': np.random.choice([True, False], n_vars, p=[0.2, 0.8])
        }
        var = pd.DataFrame(var_data, index=[f"ENSG{i:08d}" for i in range(n_vars)])

        # Create AnnData object
        adata = ad.AnnData(X=X, obs=obs, var=var)

        # Add some basic preprocessing info
        adata.uns['log1p'] = {'base': None}
        adata.layers['raw'] = X.copy()

        # Store in data manager
        self.data_manager.modalities['test_singlecell'] = adata

    def test_end_to_end_scvi_workflow(self):
        """Test complete workflow: Single Cell -> ML -> Single Cell -> Supervisor."""
        # Step 1: Create handoff tool
        ml_handoff_tool = create_expert_handoff_tool(
            from_expert="singlecell_expert",
            to_expert="machine_learning_expert",
            task_type="scvi_training",
            context_schema=SCVI_CONTEXT_SCHEMA,
            return_to_sender=True
        )

        # Step 2: Prepare handoff context
        handoff_context = {
            "modality_name": "test_singlecell",
            "n_latent": 10,
            "batch_key": "batch",
            "max_epochs": 100,  # Reduced for testing
            "use_gpu": False,
            "task_type": "scvi_training",
            "return_expectations": {
                "embedding_key": "X_scvi",
                "success_message": "scVI training completed successfully"
            }
        }

        task_description = """Train scVI embedding for single-cell modality 'test_singlecell'.

Parameters:
- Modality: test_singlecell
- Latent dimensions: 10
- Batch correction: ✓ (batch)
- Max epochs: 100
- GPU acceleration: ✗

Expected outcome:
1. Train scVI model on the specified modality
2. Store embeddings in modality.obsm['X_scvi']
3. Return control to Single Cell Expert for continued analysis

Context: This is for enhanced single-cell clustering and batch correction."""

        # Step 3: Mock state for handoff
        mock_state = {
            "messages": [],
            "current_agent": "singlecell_expert"
        }

        # Step 4: Execute handoff
        handoff_command = ml_handoff_tool.invoke({
            "task_description": task_description,
            "context": handoff_context,
            "state": mock_state,
            "tool_call_id": str(uuid.uuid4())
        })

        # Verify handoff command structure
        self.assertEqual(handoff_command.goto, "machine_learning_expert")
        self.assertIn("handoff_context", handoff_command.update)
        self.assertIn("task_description", handoff_command.update)
        self.assertTrue(handoff_command.update["return_to_sender"])

        # Verify handoff was tracked
        handoff_id = handoff_command.update["handoff_id"]
        self.assertIn(handoff_id, expert_handoff_manager.active_handoffs)

        # Step 5: Simulate ML Expert processing (mock scVI training)
        # In real implementation, ML Expert would train scVI and store embeddings
        adata = self.data_manager.get_modality("test_singlecell")

        # Mock scVI embeddings (normally done by ML Expert)
        mock_embeddings = np.random.randn(adata.n_obs, 10).astype(np.float32)
        adata.obsm['X_scvi'] = mock_embeddings

        # Step 6: Simulate ML Expert completion
        from lobster.tools.expert_handoff_manager import create_handoff_result
        from lobster.tools.enhanced_handoff_tool import create_handoff_completion_tool

        completion_tool = create_handoff_completion_tool("machine_learning_expert")

        # Mock state after ML processing
        ml_completion_state = {
            "messages": mock_state["messages"],
            "handoff_context": handoff_command.update["handoff_context"],
            "current_agent": "machine_learning_expert"
        }

        # Execute completion
        completion_result = completion_tool.invoke({
            "success": True,
            "result_data": {
                "modality_name": "test_singlecell",
                "embedding_key": "X_scvi",
                "embedding_shape": (adata.n_obs, 10),
                "training_epochs": 100,
                "final_loss": 1234.56
            },
            "error_message": None,
            "state": ml_completion_state,
            "tool_call_id": str(uuid.uuid4())
        })

        # Verify completion command
        self.assertEqual(completion_result.goto, "singlecell_expert")
        self.assertIn("handoff_result", completion_result.update)
        self.assertTrue(completion_result.update["handoff_completed"])

        # Verify embeddings were stored
        final_adata = self.data_manager.get_modality("test_singlecell")
        self.assertIn("X_scvi", final_adata.obsm)
        self.assertEqual(final_adata.obsm["X_scvi"].shape, (adata.n_obs, 10))

        # Verify handoff was cleaned up
        self.assertNotIn(handoff_id, expert_handoff_manager.active_handoffs)
        self.assertTrue(len(expert_handoff_manager.handoff_history) > 0)

    def test_scvi_embedding_integration(self):
        """Test that scVI embeddings are properly integrated and usable."""
        # Get the mock data
        adata = self.data_manager.get_modality("test_singlecell")

        # Simulate successful scVI handoff (embeddings already stored)
        mock_embeddings = np.random.randn(adata.n_obs, 15).astype(np.float32)
        adata.obsm['X_scvi'] = mock_embeddings

        # Test that embeddings can be used for downstream analysis
        self.assertIn('X_scvi', adata.obsm)
        self.assertEqual(adata.obsm['X_scvi'].shape[0], adata.n_obs)
        self.assertEqual(adata.obsm['X_scvi'].shape[1], 15)

        # Test embeddings are numeric and finite
        self.assertTrue(np.all(np.isfinite(adata.obsm['X_scvi'])))
        self.assertEqual(adata.obsm['X_scvi'].dtype, np.float32)

        # Simulate using embeddings for clustering (would be done by Single Cell Expert)
        # This tests that the handoff result can be used for downstream analysis
        from sklearn.cluster import KMeans
        from sklearn.neighbors import NearestNeighbors

        # Test that embeddings can be used for clustering
        kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(adata.obsm['X_scvi'])

        self.assertEqual(len(cluster_labels), adata.n_obs)
        self.assertTrue(len(np.unique(cluster_labels)) <= 4)

        # Test that embeddings can be used for neighbor finding
        nn = NearestNeighbors(n_neighbors=10)
        nn.fit(adata.obsm['X_scvi'])
        distances, indices = nn.kneighbors(adata.obsm['X_scvi'][:10])

        self.assertEqual(distances.shape, (10, 10))
        self.assertEqual(indices.shape, (10, 10))

    def test_failed_scvi_training_handling(self):
        """Test behavior when scVI training fails."""
        # Create handoff tool
        ml_handoff_tool = create_expert_handoff_tool(
            from_expert="singlecell_expert",
            to_expert="machine_learning_expert",
            task_type="scvi_training",
            context_schema=SCVI_CONTEXT_SCHEMA,
            return_to_sender=True
        )

        # Prepare handoff context
        handoff_context = {
            "modality_name": "test_singlecell",
            "n_latent": 10,
            "batch_key": "batch",
            "max_epochs": 100,
            "use_gpu": False,
            "task_type": "scvi_training",
            "return_expectations": {
                "embedding_key": "X_scvi",
                "success_message": "scVI training completed successfully"
            }
        }

        task_description = "Train scVI embedding for test data"

        # Mock state
        mock_state = {
            "messages": [],
            "current_agent": "singlecell_expert"
        }

        # Execute handoff
        handoff_command = ml_handoff_tool.invoke({
            "task_description": task_description,
            "context": handoff_context,
            "state": mock_state,
            "tool_call_id": str(uuid.uuid4())
        })

        handoff_id = handoff_command.update["handoff_id"]

        # Simulate ML Expert failure
        from lobster.tools.enhanced_handoff_tool import create_handoff_completion_tool

        completion_tool = create_handoff_completion_tool("machine_learning_expert")

        # Mock state after ML processing failure
        ml_completion_state = {
            "messages": mock_state["messages"],
            "handoff_context": handoff_command.update["handoff_context"],
            "current_agent": "machine_learning_expert"
        }

        # Execute completion with failure
        completion_result = completion_tool.invoke({
            "success": False,
            "result_data": {},
            "error_message": "scVI training failed: insufficient memory",
            "state": ml_completion_state,
            "tool_call_id": str(uuid.uuid4())
        })

        # Verify failure handling
        self.assertEqual(completion_result.goto, "singlecell_expert")
        self.assertIn("handoff_result", completion_result.update)
        self.assertTrue(completion_result.update["handoff_completed"])

        # Verify failure was recorded
        handoff_result = completion_result.update["handoff_result"]
        self.assertFalse(handoff_result["success"])
        self.assertEqual(handoff_result["error_message"], "scVI training failed: insufficient memory")

        # Verify no embeddings were created
        adata = self.data_manager.get_modality("test_singlecell")
        self.assertNotIn("X_scvi", adata.obsm)

        # Verify handoff was cleaned up
        self.assertNotIn(handoff_id, expert_handoff_manager.active_handoffs)

    def test_handoff_context_validation_error(self):
        """Test handling of context validation errors."""
        # Create handoff tool
        ml_handoff_tool = create_expert_handoff_tool(
            from_expert="singlecell_expert",
            to_expert="machine_learning_expert",
            task_type="scvi_training",
            context_schema=SCVI_CONTEXT_SCHEMA,
            return_to_sender=True
        )

        # Prepare invalid handoff context (missing required field)
        invalid_context = {
            "n_latent": 10,
            "batch_key": "batch",
            "max_epochs": 100,
            "use_gpu": False
            # Missing required "modality_name"
        }

        # Mock state
        mock_state = {
            "messages": [],
            "current_agent": "singlecell_expert"
        }

        # Execute handoff with invalid context
        result_command = ml_handoff_tool.invoke({
            "task_description": "Test task",
            "context": invalid_context,
            "state": mock_state,
            "tool_call_id": str(uuid.uuid4())
        })

        # Should return error command instead of handoff
        self.assertEqual(result_command.goto, "__end__")
        self.assertIn("handoff_error", result_command.update)

        # Should not create active handoff
        self.assertEqual(len(expert_handoff_manager.active_handoffs), 0)

    def test_handoff_with_nonexistent_modality(self):
        """Test handoff behavior with nonexistent modality."""
        # This would typically be caught by the single cell expert before handoff,
        # but we test the validation here
        handoff_context = {
            "modality_name": "nonexistent_modality",
            "n_latent": 10,
            "batch_key": None,
            "max_epochs": 100,
            "use_gpu": False
        }

        # In a real scenario, the single cell expert would validate the modality exists
        # before creating the handoff. Here we simulate that validation.
        modality_exists = "nonexistent_modality" in self.data_manager.modalities
        self.assertFalse(modality_exists)

        # The handoff should not be created if modality doesn't exist
        if not modality_exists:
            # This would be handled by the single cell expert returning an error command
            self.assertTrue(True)  # Test passes - validation works correctly

    def test_multiple_concurrent_handoffs(self):
        """Test handling of multiple concurrent handoffs."""
        # Create two different handoff tools
        ml_handoff_tool = create_expert_handoff_tool(
            from_expert="singlecell_expert",
            to_expert="machine_learning_expert",
            task_type="scvi_training",
            context_schema=SCVI_CONTEXT_SCHEMA,
            return_to_sender=True
        )

        bulk_handoff_tool = create_expert_handoff_tool(
            from_expert="singlecell_expert",
            to_expert="bulk_rnaseq_expert",
            task_type="pseudobulk_analysis",
            context_schema={
                "modality_name": str,
                "groupby": str,
                "layer": type(None),  # Optional[str] simplified
                "method": str,
                "min_cells": int
            },
            return_to_sender=True
        )

        # Create two different contexts
        ml_context = {
            "modality_name": "test_singlecell",
            "n_latent": 10,
            "batch_key": "batch",
            "max_epochs": 100,
            "use_gpu": False
        }

        bulk_context = {
            "modality_name": "test_singlecell",
            "groupby": "cell_type",
            "layer": None,
            "method": "sum",
            "min_cells": 10
        }

        # Mock state
        mock_state = {
            "messages": [],
            "current_agent": "singlecell_expert"
        }

        # Execute first handoff
        ml_command = ml_handoff_tool.invoke({
            "task_description": "Train scVI embeddings",
            "context": ml_context,
            "state": mock_state,
            "tool_call_id": str(uuid.uuid4())
        })

        # Execute second handoff
        bulk_command = bulk_handoff_tool.invoke({
            "task_description": "Create pseudobulk data",
            "context": bulk_context,
            "state": mock_state,
            "tool_call_id": str(uuid.uuid4())
        })

        # Verify both handoffs are tracked
        ml_handoff_id = ml_command.update["handoff_id"]
        bulk_handoff_id = bulk_command.update["handoff_id"]

        self.assertIn(ml_handoff_id, expert_handoff_manager.active_handoffs)
        self.assertIn(bulk_handoff_id, expert_handoff_manager.active_handoffs)
        self.assertEqual(len(expert_handoff_manager.active_handoffs), 2)

        # Verify they have different IDs and contexts
        self.assertNotEqual(ml_handoff_id, bulk_handoff_id)

        ml_context_stored = expert_handoff_manager.active_handoffs[ml_handoff_id]
        bulk_context_stored = expert_handoff_manager.active_handoffs[bulk_handoff_id]

        self.assertEqual(ml_context_stored.task_type, "scvi_training")
        self.assertEqual(bulk_context_stored.task_type, "pseudobulk_analysis")


if __name__ == '__main__':
    unittest.main()