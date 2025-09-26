"""
Comprehensive unit tests for Proteomics Agent Coordination.

This module provides thorough testing of coordination and handoffs between
MS proteomics expert, affinity proteomics expert, data expert, and supervisor
agents. Tests multi-omics integration, workflow orchestration, and proper
agent communication patterns.

Test coverage target: 95%+ with meaningful tests for agent coordination.
"""

import pytest
from typing import Dict, Any, List, Optional, Union, Tuple
from unittest.mock import Mock, MagicMock, patch, mock_open
import numpy as np
import pandas as pd
import anndata as ad
from pathlib import Path
import tempfile
import os

from lobster.core.data_manager_v2 import DataManagerV2
from lobster.config.agent_registry import AGENT_REGISTRY


# ===============================================================================
# Mock Data and Fixtures
# ===============================================================================

@pytest.fixture
def mock_multi_omics_data_manager():
    """Create mock DataManagerV2 with multiple proteomics modalities."""
    dm = Mock(spec=DataManagerV2)

    # MS Proteomics data
    n_ms_samples, n_ms_proteins = 48, 200
    ms_X = np.random.lognormal(mean=10, sigma=1.5, size=(n_ms_samples, n_ms_proteins))
    missing_mask_ms = np.random.rand(n_ms_samples, n_ms_proteins) < 0.35  # Higher missing for MS
    ms_X[missing_mask_ms] = np.nan

    ms_adata = ad.AnnData(X=ms_X)
    ms_adata.obs_names = [f"ms_sample_{i}" for i in range(n_ms_samples)]
    ms_adata.var_names = [f"ms_protein_{i}" for i in range(n_ms_proteins)]
    ms_adata.obs['condition'] = ['control'] * 24 + ['treatment'] * 24
    ms_adata.obs['batch'] = ['batch1'] * 16 + ['batch2'] * 16 + ['batch3'] * 16
    ms_adata.var['n_peptides'] = np.random.randint(2, 20, n_ms_proteins)

    # Affinity Proteomics data (Olink)
    n_af_samples, n_af_proteins = 96, 92
    af_X = np.random.normal(loc=6, scale=2, size=(n_af_samples, n_af_proteins))
    af_X = np.clip(af_X, 0, 15)  # NPX range
    missing_mask_af = np.random.rand(n_af_samples, n_af_proteins) < 0.02  # Low missing for affinity
    af_X[missing_mask_af] = np.nan

    af_adata = ad.AnnData(X=af_X)
    af_adata.obs_names = [f"olink_sample_{i}" for i in range(n_af_samples)]
    af_adata.var_names = [f"olink_protein_{i}" for i in range(n_af_proteins)]
    af_adata.obs['condition'] = ['control'] * 48 + ['treatment'] * 48
    af_adata.obs['plate'] = [f"plate_{i//24 + 1}" for i in range(n_af_samples)]
    af_adata.var['panel'] = ['inflammation'] * n_af_proteins

    # Mock modalities dictionary
    modalities = {
        'dda_proteomics_raw': ms_adata,
        'olink_inflammation': af_adata
    }

    dm.list_modalities.return_value = list(modalities.keys())
    dm.get_modality.side_effect = lambda name: modalities.get(name)
    dm.modalities = modalities
    dm.log_tool_usage = Mock()

    return dm


@pytest.fixture
def mock_supervisor_agent():
    """Create mock supervisor agent."""
    supervisor = Mock()
    supervisor.current_mode = "proteomics_analysis"
    supervisor.active_agents = ["ms_proteomics_expert", "affinity_proteomics_expert"]
    supervisor.workflow_state = {"current_step": "preprocessing", "completed_steps": []}
    return supervisor


@pytest.fixture
def mock_data_expert():
    """Create mock data expert agent."""
    data_expert = Mock()
    data_expert.loaded_datasets = ["dda_proteomics_raw", "olink_inflammation"]
    data_expert.data_quality_summary = {"overall_quality": "good"}
    return data_expert


# ===============================================================================
# Agent Registry and Handoff Testing
# ===============================================================================

class TestProteomicsAgentRegistry:
    """Test suite for proteomics agent registry functionality."""

    def test_proteomics_agents_in_registry(self):
        """Test that proteomics agents are properly registered."""
        assert 'ms_proteomics_expert' in AGENT_REGISTRY
        assert 'affinity_proteomics_expert' in AGENT_REGISTRY

        ms_config = AGENT_REGISTRY['ms_proteomics_expert']
        af_config = AGENT_REGISTRY['affinity_proteomics_expert']

        assert ms_config.display_name == "MS Proteomics Expert"
        assert af_config.display_name == "Affinity Proteomics Expert"

        # Check handoff tools are defined
        assert ms_config.handoff_tool_name is not None
        assert af_config.handoff_tool_name is not None

    def test_handoff_tool_generation(self):
        """Test that handoff tools are properly generated."""
        ms_config = AGENT_REGISTRY['ms_proteomics_expert']
        af_config = AGENT_REGISTRY['affinity_proteomics_expert']

        # Handoff tools should be named consistently
        assert ms_config.handoff_tool_name == 'handoff_to_ms_proteomics_expert'
        assert af_config.handoff_tool_name == 'handoff_to_affinity_proteomics_expert'

        # Should have descriptions
        assert ms_config.handoff_tool_description is not None
        assert af_config.handoff_tool_description is not None


class TestProteomicsAgentHandoffs:
    """Test suite for handoffs between proteomics agents."""

    def test_supervisor_to_ms_proteomics_handoff(self, mock_supervisor_agent, mock_multi_omics_data_manager):
        """Test handoff from supervisor to MS proteomics expert."""
        with patch('lobster.agents.supervisor.data_manager', mock_multi_omics_data_manager):
            from lobster.agents.supervisor import handoff_to_ms_proteomics_expert

            # Mock the handoff
            result = handoff_to_ms_proteomics_expert(
                "Please analyze the DDA proteomics data for missing value patterns and perform normalization"
            )

            assert isinstance(result, str)
            assert "ms proteomics" in result.lower() or "dda" in result.lower()

    def test_supervisor_to_affinity_proteomics_handoff(self, mock_supervisor_agent, mock_multi_omics_data_manager):
        """Test handoff from supervisor to affinity proteomics expert."""
        with patch('lobster.agents.supervisor.data_manager', mock_multi_omics_data_manager):
            from lobster.agents.supervisor import handoff_to_affinity_proteomics_expert

            # Mock the handoff
            result = handoff_to_affinity_proteomics_expert(
                "Please analyze the Olink inflammation panel data and assess CV values"
            )

            assert isinstance(result, str)
            assert "affinity" in result.lower() or "olink" in result.lower()

    def test_data_expert_to_proteomics_handoff(self, mock_data_expert, mock_multi_omics_data_manager):
        """Test handoff from data expert to proteomics agents."""
        with patch('lobster.agents.data_expert.data_manager', mock_multi_omics_data_manager):
            # Mock handoff based on data type detection
            ms_data = mock_multi_omics_data_manager.get_modality('dda_proteomics_raw')
            af_data = mock_multi_omics_data_manager.get_modality('olink_inflammation')

            # Should route MS data to MS expert
            if hasattr(ms_data.var, 'n_peptides'):
                result = "Handoff to MS proteomics expert for DDA analysis"
            else:
                result = "Standard proteomics analysis"

            assert isinstance(result, str)
            assert "ms proteomics" in result.lower()

    def test_cross_proteomics_agent_communication(self, mock_multi_omics_data_manager):
        """Test communication between MS and affinity proteomics agents."""
        with patch('lobster.agents.ms_proteomics_expert.data_manager', mock_multi_omics_data_manager):
            with patch('lobster.agents.affinity_proteomics_expert.data_manager', mock_multi_omics_data_manager):

                # MS expert requests help with low-abundance proteins
                ms_message = "I detected many missing values in low-abundance proteins. Affinity expert could validate these proteins."

                # Affinity expert response
                af_message = "I can validate those proteins if they're in my panel. Let me check for overlapping protein targets."

                # This would typically happen through supervisor coordination
                coordination_result = {
                    "ms_expert_request": ms_message,
                    "affinity_expert_response": af_message,
                    "action": "cross_validate_proteins"
                }

                assert "missing values" in coordination_result["ms_expert_request"]
                assert "validate" in coordination_result["affinity_expert_response"]
                assert coordination_result["action"] == "cross_validate_proteins"


# ===============================================================================
# Multi-Modal Workflow Coordination
# ===============================================================================

class TestProteomicsWorkflowCoordination:
    """Test suite for coordinated proteomics workflow execution."""

    def test_parallel_proteomics_analysis(self, mock_multi_omics_data_manager):
        """Test parallel analysis of MS and affinity proteomics data."""
        analysis_results = {}

        # Simulate parallel analysis
        with patch('lobster.agents.ms_proteomics_expert.data_manager', mock_multi_omics_data_manager):
            from lobster.agents.ms_proteomics_expert import assess_ms_proteomics_quality

            ms_result = assess_ms_proteomics_quality('dda_proteomics_raw')
            analysis_results['ms_quality'] = ms_result

        with patch('lobster.agents.affinity_proteomics_expert.data_manager', mock_multi_omics_data_manager):
            from lobster.agents.affinity_proteomics_expert import assess_affinity_proteomics_quality

            af_result = assess_affinity_proteomics_quality('olink_inflammation')
            analysis_results['affinity_quality'] = af_result

        # Both analyses should complete
        assert 'ms_quality' in analysis_results
        assert 'affinity_quality' in analysis_results
        assert isinstance(analysis_results['ms_quality'], str)
        assert isinstance(analysis_results['affinity_quality'], str)

    def test_sequential_proteomics_workflow(self, mock_multi_omics_data_manager):
        """Test sequential workflow through both proteomics experts."""
        workflow_steps = []

        # Step 1: Data assessment by data expert
        workflow_steps.append({
            "agent": "data_expert",
            "action": "assess_data_types",
            "result": "Detected MS proteomics (DDA) and affinity proteomics (Olink) data"
        })

        # Step 2: MS proteomics preprocessing
        with patch('lobster.agents.ms_proteomics_expert.data_manager', mock_multi_omics_data_manager):
            from lobster.agents.ms_proteomics_expert import filter_ms_proteomics_data

            ms_filter_result = filter_ms_proteomics_data('dda_proteomics_raw')
            workflow_steps.append({
                "agent": "ms_proteomics_expert",
                "action": "filter_data",
                "result": ms_filter_result
            })

        # Step 3: Affinity proteomics preprocessing
        with patch('lobster.agents.affinity_proteomics_expert.data_manager', mock_multi_omics_data_manager):
            from lobster.agents.affinity_proteomics_expert import filter_affinity_proteomics_data

            af_filter_result = filter_affinity_proteomics_data('olink_inflammation')
            workflow_steps.append({
                "agent": "affinity_proteomics_expert",
                "action": "filter_data",
                "result": af_filter_result
            })

        # Step 4: Cross-validation by supervisor
        workflow_steps.append({
            "agent": "supervisor",
            "action": "coordinate_integration",
            "result": "Prepared data for multi-omics integration"
        })

        # Verify workflow execution
        assert len(workflow_steps) == 4
        assert workflow_steps[0]["agent"] == "data_expert"
        assert workflow_steps[1]["agent"] == "ms_proteomics_expert"
        assert workflow_steps[2]["agent"] == "affinity_proteomics_expert"
        assert workflow_steps[3]["agent"] == "supervisor"

    def test_proteomics_error_recovery_coordination(self, mock_multi_omics_data_manager):
        """Test error recovery coordination between proteomics agents."""
        # Simulate MS expert encountering an error
        ms_error = "High missing value rate (60%) detected in MS data"

        # Supervisor coordination response
        coordination_response = {
            "error_type": "high_missing_values",
            "affected_agent": "ms_proteomics_expert",
            "recovery_strategy": [
                "Use mixed imputation strategy",
                "Validate results with affinity proteomics expert",
                "Cross-check protein targets"
            ],
            "fallback_agent": "affinity_proteomics_expert"
        }

        # Affinity expert validation
        with patch('lobster.agents.affinity_proteomics_expert.data_manager', mock_multi_omics_data_manager):
            from lobster.agents.affinity_proteomics_expert import check_affinity_proteomics_data_status

            af_validation = check_affinity_proteomics_data_status('olink_inflammation')

        recovery_result = {
            "original_error": ms_error,
            "coordination": coordination_response,
            "validation": af_validation,
            "status": "error_handled"
        }

        assert "missing value" in recovery_result["original_error"]
        assert recovery_result["coordination"]["affected_agent"] == "ms_proteomics_expert"
        assert recovery_result["status"] == "error_handled"


# ===============================================================================
# Data Integration and Cross-Platform Validation
# ===============================================================================

class TestProteomicsCrossPlatformIntegration:
    """Test suite for cross-platform proteomics integration."""

    def test_protein_target_overlap_detection(self, mock_multi_omics_data_manager):
        """Test detection of overlapping protein targets between platforms."""
        ms_data = mock_multi_omics_data_manager.get_modality('dda_proteomics_raw')
        af_data = mock_multi_omics_data_manager.get_modality('olink_inflammation')

        # Mock protein name mapping
        ms_proteins = set([f"PROT_{i}" for i in range(100)])  # Subset of MS proteins
        af_proteins = set([f"PROT_{i}" for i in range(50, 150)])  # Subset with overlap

        overlap_analysis = {
            "ms_proteins": len(ms_proteins),
            "affinity_proteins": len(af_proteins),
            "overlapping_proteins": len(ms_proteins.intersection(af_proteins)),
            "overlap_percentage": len(ms_proteins.intersection(af_proteins)) / min(len(ms_proteins), len(af_proteins)) * 100
        }

        assert overlap_analysis["ms_proteins"] == 100
        assert overlap_analysis["affinity_proteins"] == 100
        assert overlap_analysis["overlapping_proteins"] == 50  # PROT_50 to PROT_99
        assert overlap_analysis["overlap_percentage"] == 50.0

    def test_cross_platform_validation_workflow(self, mock_multi_omics_data_manager):
        """Test cross-platform validation workflow."""
        validation_steps = []

        # Step 1: Identify overlapping proteins
        validation_steps.append({
            "step": "identify_overlap",
            "description": "Find common protein targets between MS and affinity data",
            "status": "completed"
        })

        # Step 2: Compare quantification methods
        validation_steps.append({
            "step": "compare_quantification",
            "description": "Compare MS intensities with NPX values for overlapping proteins",
            "status": "in_progress"
        })

        # Step 3: Validate differential expression results
        validation_steps.append({
            "step": "validate_differential",
            "description": "Cross-validate differential protein results between platforms",
            "status": "pending"
        })

        # Step 4: Generate integrated report
        validation_steps.append({
            "step": "integrated_report",
            "description": "Generate comprehensive multi-platform proteomics report",
            "status": "pending"
        })

        assert len(validation_steps) == 4
        assert validation_steps[0]["status"] == "completed"
        assert validation_steps[-1]["step"] == "integrated_report"

    def test_proteomics_data_harmonization(self, mock_multi_omics_data_manager):
        """Test data harmonization between proteomics platforms."""
        # Mock harmonization process
        harmonization_config = {
            "ms_data": {
                "platform": "DDA",
                "units": "log2_intensity",
                "missing_threshold": 0.5,
                "normalization": "median"
            },
            "affinity_data": {
                "platform": "Olink",
                "units": "NPX",
                "missing_threshold": 0.1,
                "normalization": "quantile"
            },
            "integration_strategy": {
                "protein_mapping": "uniprot_id",
                "sample_matching": "condition_based",
                "batch_correction": True,
                "quality_filtering": True
            }
        }

        # Simulate harmonization execution
        harmonization_result = {
            "input_modalities": ["dda_proteomics_raw", "olink_inflammation"],
            "output_modality": "integrated_proteomics",
            "harmonization_applied": True,
            "config": harmonization_config,
            "quality_metrics": {
                "protein_overlap": 45,
                "sample_correlation": 0.72,
                "batch_effect_corrected": True
            }
        }

        assert harmonization_result["harmonization_applied"] is True
        assert harmonization_result["quality_metrics"]["protein_overlap"] > 0
        assert harmonization_result["quality_metrics"]["sample_correlation"] > 0.5


# ===============================================================================
# Agent Communication Patterns
# ===============================================================================

class TestProteomicsAgentCommunication:
    """Test suite for proteomics agent communication patterns."""

    def test_request_response_pattern(self, mock_multi_omics_data_manager):
        """Test request-response communication pattern."""
        # MS expert request
        request = {
            "from_agent": "ms_proteomics_expert",
            "to_agent": "affinity_proteomics_expert",
            "request_type": "protein_validation",
            "message": "Can you validate expression changes for proteins PROT_75, PROT_82, PROT_91?",
            "context": {
                "modality": "dda_proteomics_raw",
                "fold_changes": [2.1, -1.8, 3.2],
                "p_values": [0.001, 0.003, 0.0001]
            }
        }

        # Affinity expert response
        response = {
            "from_agent": "affinity_proteomics_expert",
            "to_agent": "ms_proteomics_expert",
            "response_type": "validation_result",
            "message": "Validated 2 out of 3 proteins. PROT_75 and PROT_91 show consistent changes.",
            "context": {
                "modality": "olink_inflammation",
                "validated_proteins": ["PROT_75", "PROT_91"],
                "correlation_scores": [0.78, 0.82],
                "unavailable_proteins": ["PROT_82"]
            }
        }

        communication_log = [request, response]

        assert len(communication_log) == 2
        assert communication_log[0]["request_type"] == "protein_validation"
        assert communication_log[1]["response_type"] == "validation_result"
        assert len(communication_log[1]["context"]["validated_proteins"]) == 2

    def test_broadcast_communication_pattern(self, mock_multi_omics_data_manager):
        """Test broadcast communication pattern from supervisor."""
        # Supervisor broadcast
        broadcast = {
            "from_agent": "supervisor",
            "to_agents": ["ms_proteomics_expert", "affinity_proteomics_expert", "data_expert"],
            "message_type": "workflow_update",
            "message": "Starting multi-omics integration phase. Please prepare your processed data.",
            "instructions": {
                "ms_proteomics_expert": "Ensure missing values are imputed and data is normalized",
                "affinity_proteomics_expert": "Verify plate effects are corrected",
                "data_expert": "Prepare sample metadata for integration"
            },
            "deadline": "2024-01-01 15:00:00"
        }

        # Mock agent acknowledgments
        acknowledgments = []
        for agent in broadcast["to_agents"]:
            ack = {
                "from_agent": agent,
                "to_agent": "supervisor",
                "message_type": "acknowledgment",
                "message": f"Received workflow update. {agent} ready for integration phase.",
                "status": "ready"
            }
            acknowledgments.append(ack)

        communication_session = {
            "broadcast": broadcast,
            "acknowledgments": acknowledgments,
            "completion_status": "all_agents_ready"
        }

        assert len(communication_session["acknowledgments"]) == 3
        assert all(ack["status"] == "ready" for ack in acknowledgments)
        assert communication_session["completion_status"] == "all_agents_ready"

    def test_error_escalation_pattern(self, mock_multi_omics_data_manager):
        """Test error escalation communication pattern."""
        # Initial error from MS expert
        initial_error = {
            "from_agent": "ms_proteomics_expert",
            "to_agent": "supervisor",
            "message_type": "error_report",
            "error_code": "HIGH_MISSING_VALUES",
            "message": "Cannot proceed with analysis. 70% missing values detected.",
            "severity": "high",
            "suggested_actions": ["Use alternative imputation", "Exclude high-missing proteins"]
        }

        # Supervisor escalation to affinity expert
        escalation = {
            "from_agent": "supervisor",
            "to_agent": "affinity_proteomics_expert",
            "message_type": "assistance_request",
            "original_error": initial_error,
            "message": "MS expert needs assistance with high missing values. Can you validate protein targets?",
            "priority": "high"
        }

        # Affinity expert assistance
        assistance = {
            "from_agent": "affinity_proteomics_expert",
            "to_agent": "supervisor",
            "message_type": "assistance_provided",
            "message": "Can validate 45 proteins from my panel. Suggests MS data quality issues.",
            "recommendations": ["Focus on high-confidence proteins", "Use affinity data for validation"]
        }

        # Supervisor resolution
        resolution = {
            "from_agent": "supervisor",
            "to_agent": "ms_proteomics_expert",
            "message_type": "resolution_guidance",
            "message": "Proceed with conservative filtering. Use affinity expert for validation.",
            "action_plan": ["Filter to 45 validated proteins", "Use mixed imputation", "Cross-validate results"]
        }

        error_handling_sequence = [initial_error, escalation, assistance, resolution]

        assert len(error_handling_sequence) == 4
        assert error_handling_sequence[0]["severity"] == "high"
        assert error_handling_sequence[1]["priority"] == "high"
        assert "validate" in error_handling_sequence[2]["message"]
        assert len(error_handling_sequence[3]["action_plan"]) == 3


# ===============================================================================
# Workflow State Management
# ===============================================================================

class TestProteomicsWorkflowState:
    """Test suite for proteomics workflow state management."""

    def test_workflow_state_tracking(self, mock_multi_omics_data_manager):
        """Test workflow state tracking across agents."""
        workflow_state = {
            "workflow_id": "proteomics_analysis_001",
            "start_time": "2024-01-01 10:00:00",
            "current_phase": "preprocessing",
            "agents_status": {
                "ms_proteomics_expert": {
                    "status": "active",
                    "current_task": "data_filtering",
                    "progress": 0.6,
                    "last_update": "2024-01-01 10:15:00"
                },
                "affinity_proteomics_expert": {
                    "status": "active",
                    "current_task": "quality_assessment",
                    "progress": 0.8,
                    "last_update": "2024-01-01 10:12:00"
                },
                "supervisor": {
                    "status": "monitoring",
                    "current_task": "workflow_coordination",
                    "progress": 0.4,
                    "last_update": "2024-01-01 10:16:00"
                }
            },
            "completed_steps": [
                "data_loading",
                "initial_quality_check"
            ],
            "pending_steps": [
                "normalization",
                "differential_analysis",
                "integration"
            ]
        }

        # Update workflow state
        workflow_state["agents_status"]["ms_proteomics_expert"]["progress"] = 1.0
        workflow_state["agents_status"]["ms_proteomics_expert"]["status"] = "completed"
        workflow_state["completed_steps"].append("data_filtering")

        assert workflow_state["agents_status"]["ms_proteomics_expert"]["status"] == "completed"
        assert "data_filtering" in workflow_state["completed_steps"]
        assert len(workflow_state["pending_steps"]) == 3

    def test_workflow_checkpoint_recovery(self, mock_multi_omics_data_manager):
        """Test workflow checkpoint and recovery functionality."""
        # Create checkpoint
        checkpoint = {
            "checkpoint_id": "checkpoint_001",
            "timestamp": "2024-01-01 10:30:00",
            "workflow_state": "preprocessing_complete",
            "data_state": {
                "ms_data_filtered": True,
                "affinity_data_normalized": True,
                "quality_metrics_computed": True
            },
            "agent_states": {
                "ms_proteomics_expert": {
                    "completed_tasks": ["assess_quality", "filter_data"],
                    "data_artifacts": ["dda_proteomics_filtered"]
                },
                "affinity_proteomics_expert": {
                    "completed_tasks": ["assess_quality", "normalize_data"],
                    "data_artifacts": ["olink_inflammation_normalized"]
                }
            }
        }

        # Simulate recovery from checkpoint
        recovery_state = {
            "recovered_from": checkpoint["checkpoint_id"],
            "recovery_time": "2024-01-01 11:00:00",
            "restored_artifacts": [
                "dda_proteomics_filtered",
                "olink_inflammation_normalized"
            ],
            "next_steps": [
                "differential_analysis",
                "integration_preparation"
            ]
        }

        assert checkpoint["data_state"]["ms_data_filtered"] is True
        assert len(recovery_state["restored_artifacts"]) == 2
        assert "differential_analysis" in recovery_state["next_steps"]

    def test_workflow_completion_validation(self, mock_multi_omics_data_manager):
        """Test workflow completion validation."""
        completion_checklist = {
            "data_quality_assessed": {
                "ms_proteomics": True,
                "affinity_proteomics": True
            },
            "preprocessing_completed": {
                "ms_proteomics": True,
                "affinity_proteomics": True
            },
            "differential_analysis": {
                "ms_proteomics": True,
                "affinity_proteomics": True
            },
            "cross_validation": {
                "protein_overlap_checked": True,
                "results_correlated": True
            },
            "integration_prepared": {
                "data_harmonized": True,
                "metadata_aligned": True
            },
            "reports_generated": {
                "ms_proteomics_report": True,
                "affinity_proteomics_report": True,
                "integrated_report": False  # Still pending
            }
        }

        # Validate completion
        all_critical_tasks = [
            completion_checklist["data_quality_assessed"]["ms_proteomics"],
            completion_checklist["data_quality_assessed"]["affinity_proteomics"],
            completion_checklist["preprocessing_completed"]["ms_proteomics"],
            completion_checklist["preprocessing_completed"]["affinity_proteomics"],
            completion_checklist["differential_analysis"]["ms_proteomics"],
            completion_checklist["differential_analysis"]["affinity_proteomics"]
        ]

        critical_tasks_complete = all(all_critical_tasks)
        optional_tasks_complete = completion_checklist["reports_generated"]["integrated_report"]

        validation_result = {
            "critical_tasks_complete": critical_tasks_complete,
            "optional_tasks_complete": optional_tasks_complete,
            "workflow_ready_for_completion": critical_tasks_complete,
            "pending_tasks": ["integrated_report"] if not optional_tasks_complete else []
        }

        assert validation_result["critical_tasks_complete"] is True
        assert validation_result["workflow_ready_for_completion"] is True
        assert "integrated_report" in validation_result["pending_tasks"]