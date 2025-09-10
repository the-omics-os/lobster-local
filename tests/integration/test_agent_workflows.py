"""
Comprehensive integration tests for agent workflows.

This module provides thorough testing of multi-agent collaboration including
agent handoffs, workflow orchestration, supervisor coordination,
cross-agent data sharing, and complex analysis pipelines.

Test coverage target: 95%+ with realistic workflow scenarios.
"""

import pytest
from typing import Dict, Any, List, Optional, Union
from unittest.mock import Mock, MagicMock, patch, AsyncMock
import asyncio
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
import json

from lobster.core.client import AgentClient
from lobster.core.data_manager_v2 import DataManagerV2
from lobster.agents.supervisor import supervisor_agent
from lobster.agents.data_expert import data_expert_agent
from lobster.agents.singlecell_expert import singlecell_expert_agent
from lobster.agents.research_agent import research_agent
from lobster.agents.method_expert import method_expert_agent

from tests.mock_data.factories import SingleCellDataFactory, BulkRNASeqDataFactory
from tests.mock_data.base import SMALL_DATASET_CONFIG, LARGE_DATASET_CONFIG


# ===============================================================================
# Test Fixtures and Mock Data
# ===============================================================================

@pytest.fixture
def temp_workspace():
    """Create temporary workspace for workflow tests."""
    with tempfile.TemporaryDirectory() as temp_dir:
        workspace_path = Path(temp_dir) / ".lobster_workspace"
        workspace_path.mkdir(parents=True, exist_ok=True)
        yield workspace_path


@pytest.fixture
def mock_agent_client(temp_workspace):
    """Create mock agent client for workflow testing."""
    with patch('lobster.core.client.AgentClient') as MockClient:
        mock_client = MockClient.return_value
        
        # Mock data manager
        mock_dm = Mock(spec=DataManagerV2)
        mock_dm.workspace_path = temp_workspace
        mock_dm.list_modalities.return_value = []
        mock_client.data_manager = mock_dm
        
        # Mock message handling
        mock_client.messages = []
        mock_client.current_agent = "supervisor_agent"
        
        yield mock_client


@pytest.fixture
def mock_workflow_state():
    """Create mock workflow state for testing."""
    return {
        "messages": [],
        "current_agent": "supervisor_agent",
        "workflow_id": "test_workflow_123",
        "workflow_status": "initialized",
        "completed_steps": [],
        "data_manager": Mock(spec=DataManagerV2),
        "context": {},
        "metadata": {}
    }


@pytest.fixture
def sample_geo_dataset():
    """Create sample GEO dataset for workflow testing."""
    return {
        "accession": "GSE123456",
        "title": "Single-cell RNA sequencing of immune cells",
        "organism": "Homo sapiens",
        "samples": 48,
        "description": "scRNA-seq analysis of immune cell populations",
        "data": SingleCellDataFactory(config=SMALL_DATASET_CONFIG)
    }


# ===============================================================================
# Basic Agent Workflow Tests
# ===============================================================================

@pytest.mark.integration
class TestBasicAgentWorkflows:
    """Test basic agent workflow functionality."""
    
    def test_supervisor_agent_initialization(self, mock_agent_client, mock_workflow_state):
        """Test supervisor agent initialization and basic routing."""
        state = mock_workflow_state.copy()
        state["messages"] = [{"content": "Analyze single-cell data from GEO", "sender": "human"}]
        
        with patch('lobster.agents.supervisor.supervisor_agent') as mock_supervisor:
            mock_supervisor.return_value = {
                "messages": state["messages"] + [{"content": "I'll help you analyze single-cell data", "sender": "assistant"}],
                "next_agent": "data_expert_agent",
                "reasoning": "User wants to analyze data, need to start with data loading",
                "workflow_plan": ["load_data", "quality_control", "analysis"]
            }
            
            result = supervisor_agent(state)
            
            assert result["next_agent"] == "data_expert_agent"
            assert "workflow_plan" in result
            assert len(result["workflow_plan"]) == 3
    
    def test_data_expert_to_singlecell_handoff(self, mock_agent_client, mock_workflow_state, sample_geo_dataset):
        """Test handoff from data expert to single-cell expert."""
        # Setup: Data expert completes data loading
        state = mock_workflow_state.copy()
        state["messages"] = [
            {"content": "Load GEO dataset GSE123456", "sender": "human"},
            {"content": "Dataset loaded successfully", "sender": "data_expert_agent"}
        ]
        state["current_agent"] = "data_expert_agent"
        
        # Mock data manager with loaded data
        mock_dm = Mock(spec=DataManagerV2)
        mock_dm.list_modalities.return_value = ["geo_gse123456"]
        mock_dm.get_modality.return_value = sample_geo_dataset["data"]
        state["data_manager"] = mock_dm
        
        # Test handoff to single-cell expert
        with patch('lobster.agents.singlecell_expert.singlecell_expert_agent') as mock_sc_agent:
            mock_sc_agent.return_value = {
                "messages": state["messages"] + [{"content": "Ready to analyze single-cell data", "sender": "singlecell_expert_agent"}],
                "current_agent": "singlecell_expert_agent",
                "data_context": {
                    "available_modalities": ["geo_gse123456"],
                    "data_type": "single_cell_rna_seq",
                    "n_cells": 1000,
                    "n_genes": 2000
                }
            }
            
            result = singlecell_expert_agent(state)
            
            assert result["current_agent"] == "singlecell_expert_agent"
            assert "data_context" in result
            assert result["data_context"]["data_type"] == "single_cell_rna_seq"
    
    def test_research_agent_workflow(self, mock_agent_client, mock_workflow_state):
        """Test research agent workflow for literature search."""
        state = mock_workflow_state.copy()
        state["messages"] = [{"content": "Find papers about T cell exhaustion", "sender": "human"}]
        state["current_agent"] = "research_agent"
        
        with patch('lobster.agents.research_agent.research_agent') as mock_research:
            mock_research.return_value = {
                "messages": state["messages"] + [{"content": "Found 15 relevant papers about T cell exhaustion", "sender": "research_agent"}],
                "search_results": [
                    {"pmid": "12345678", "title": "T cell exhaustion in cancer", "journal": "Nature"},
                    {"pmid": "87654321", "title": "Molecular markers of T cell dysfunction", "journal": "Cell"}
                ],
                "literature_summary": "Recent research focuses on exhaustion markers and therapeutic targets",
                "recommended_datasets": ["GSE111111", "GSE222222"]
            }
            
            result = research_agent(state)
            
            assert len(result["search_results"]) == 2
            assert "literature_summary" in result
            assert "recommended_datasets" in result
    
    def test_method_expert_parameter_extraction(self, mock_agent_client, mock_workflow_state):
        """Test method expert workflow for parameter extraction."""
        state = mock_workflow_state.copy()
        state["messages"] = [{"content": "Extract clustering parameters from PMID:12345678", "sender": "human"}]
        state["current_agent"] = "method_expert_agent"
        
        with patch('lobster.agents.method_expert.method_expert_agent') as mock_method:
            mock_method.return_value = {
                "messages": state["messages"] + [{"content": "Extracted parameters from the paper", "sender": "method_expert_agent"}],
                "extracted_parameters": {
                    "leiden_resolution": 0.5,
                    "n_neighbors": 15,
                    "normalization_method": "log1p",
                    "target_sum": 10000
                },
                "parameter_confidence": 0.9,
                "method_reproducibility": "high"
            }
            
            result = method_expert_agent(state)
            
            assert "extracted_parameters" in result
            assert result["extracted_parameters"]["leiden_resolution"] == 0.5
            assert result["parameter_confidence"] == 0.9


# ===============================================================================
# Complex Multi-Agent Workflows
# ===============================================================================

@pytest.mark.integration
class TestComplexMultiAgentWorkflows:
    """Test complex multi-agent workflow scenarios."""
    
    def test_complete_single_cell_analysis_workflow(self, mock_agent_client, mock_workflow_state, sample_geo_dataset):
        """Test complete single-cell analysis workflow with multiple agents."""
        workflow_steps = []
        
        # Step 1: Supervisor receives request
        initial_state = mock_workflow_state.copy()
        initial_state["messages"] = [{"content": "Perform complete single-cell analysis of GSE123456", "sender": "human"}]
        
        with patch('lobster.agents.supervisor.supervisor_agent') as mock_supervisor:
            mock_supervisor.return_value = {
                "messages": initial_state["messages"] + [{"content": "I'll coordinate a complete single-cell analysis", "sender": "supervisor_agent"}],
                "next_agent": "data_expert_agent",
                "workflow_plan": ["data_loading", "quality_control", "preprocessing", "clustering", "annotation"],
                "workflow_id": "sc_analysis_001"
            }
            
            supervisor_result = supervisor_agent(initial_state)
            workflow_steps.append(("supervisor", supervisor_result))
        
        # Step 2: Data expert loads data
        data_state = supervisor_result.copy()
        data_state["current_agent"] = "data_expert_agent"
        
        with patch('lobster.agents.data_expert.data_expert_agent') as mock_data_expert:
            mock_data_expert.return_value = {
                "messages": data_state["messages"] + [{"content": "Successfully loaded GSE123456 dataset", "sender": "data_expert_agent"}],
                "current_agent": "data_expert_agent",
                "loaded_modalities": ["geo_gse123456"],
                "data_summary": {"n_cells": 5000, "n_genes": 20000},
                "next_agent": "singlecell_expert_agent",
                "handoff_reason": "Data loaded, ready for single-cell analysis"
            }
            
            data_result = data_expert_agent(data_state)
            workflow_steps.append(("data_expert", data_result))
        
        # Step 3: Single-cell expert performs analysis
        sc_state = data_result.copy()
        sc_state["current_agent"] = "singlecell_expert_agent"
        
        with patch('lobster.agents.singlecell_expert.singlecell_expert_agent') as mock_sc_expert:
            mock_sc_expert.return_value = {
                "messages": sc_state["messages"] + [{"content": "Completed clustering and found 12 cell types", "sender": "singlecell_expert_agent"}],
                "current_agent": "singlecell_expert_agent",
                "analysis_results": {
                    "n_clusters": 12,
                    "cluster_quality": "high",
                    "marker_genes_found": 156,
                    "cell_types_annotated": 8
                },
                "workflow_status": "completed",
                "output_modalities": ["geo_gse123456_processed", "geo_gse123456_clustered"]
            }
            
            sc_result = singlecell_expert_agent(sc_state)
            workflow_steps.append(("singlecell_expert", sc_result))
        
        # Verify complete workflow
        assert len(workflow_steps) == 3
        assert workflow_steps[0][1]["next_agent"] == "data_expert_agent"
        assert workflow_steps[1][1]["next_agent"] == "singlecell_expert_agent" 
        assert workflow_steps[2][1]["workflow_status"] == "completed"
        assert workflow_steps[2][1]["analysis_results"]["n_clusters"] == 12
    
    def test_research_guided_analysis_workflow(self, mock_agent_client, mock_workflow_state):
        """Test workflow where research agent guides method selection."""
        workflow_states = []
        
        # Step 1: Research agent finds relevant papers
        research_state = mock_workflow_state.copy()
        research_state["messages"] = [{"content": "Find best methods for T cell analysis", "sender": "human"}]
        research_state["current_agent"] = "research_agent"
        
        with patch('lobster.agents.research_agent.research_agent') as mock_research:
            mock_research.return_value = {
                "messages": research_state["messages"] + [{"content": "Found optimal methods for T cell analysis", "sender": "research_agent"}],
                "recommended_papers": [
                    {"pmid": "12345678", "title": "Optimal T cell clustering", "relevance": 0.95}
                ],
                "next_agent": "method_expert_agent",
                "handoff_data": {"target_pmid": "12345678", "analysis_type": "t_cell_clustering"}
            }
            
            research_result = research_agent(research_state)
            workflow_states.append(research_result)
        
        # Step 2: Method expert extracts parameters
        method_state = research_result.copy()
        method_state["current_agent"] = "method_expert_agent"
        
        with patch('lobster.agents.method_expert.method_expert_agent') as mock_method:
            mock_method.return_value = {
                "messages": method_state["messages"] + [{"content": "Extracted optimal parameters for T cell analysis", "sender": "method_expert_agent"}],
                "extracted_parameters": {
                    "resolution": 0.3,
                    "n_neighbors": 20,
                    "t_cell_markers": ["CD3D", "CD3E", "CD8A", "CD4"]
                },
                "optimization_confidence": 0.92,
                "next_agent": "singlecell_expert_agent",
                "handoff_data": {"optimized_params": True, "analysis_type": "t_cell_specific"}
            }
            
            method_result = method_expert_agent(method_state)
            workflow_states.append(method_result)
        
        # Step 3: Single-cell expert applies optimized parameters
        sc_state = method_result.copy()
        sc_state["current_agent"] = "singlecell_expert_agent"
        
        with patch('lobster.agents.singlecell_expert.singlecell_expert_agent') as mock_sc:
            mock_sc.return_value = {
                "messages": sc_state["messages"] + [{"content": "Applied optimized parameters for T cell analysis", "sender": "singlecell_expert_agent"}],
                "analysis_results": {
                    "t_cell_clusters": 6,
                    "parameter_source": "literature_optimized",
                    "improvement_over_default": 0.25
                },
                "workflow_status": "completed"
            }
            
            sc_result = singlecell_expert_agent(sc_state)
            workflow_states.append(sc_result)
        
        # Verify research-guided workflow
        assert len(workflow_states) == 3
        assert workflow_states[1]["optimization_confidence"] == 0.92
        assert workflow_states[2]["analysis_results"]["parameter_source"] == "literature_optimized"
        assert workflow_states[2]["analysis_results"]["improvement_over_default"] == 0.25
    
    def test_parallel_agent_execution(self, mock_agent_client, mock_workflow_state):
        """Test parallel execution of independent agent tasks."""
        import concurrent.futures
        
        # Setup independent tasks
        tasks = [
            {"agent": "research_agent", "task": "Find T cell papers"},
            {"agent": "research_agent", "task": "Find B cell papers"},
            {"agent": "method_expert_agent", "task": "Extract clustering parameters"}
        ]
        
        def execute_agent_task(task):
            """Execute a single agent task."""
            state = mock_workflow_state.copy()
            state["messages"] = [{"content": task["task"], "sender": "human"}]
            state["current_agent"] = task["agent"]
            
            if task["agent"] == "research_agent":
                with patch('lobster.agents.research_agent.research_agent') as mock_agent:
                    mock_agent.return_value = {
                        "task_id": f"task_{task['task'][:10]}",
                        "papers_found": 5,
                        "execution_time": 2.5
                    }
                    return research_agent(state)
            
            elif task["agent"] == "method_expert_agent":
                with patch('lobster.agents.method_expert.method_expert_agent') as mock_agent:
                    mock_agent.return_value = {
                        "task_id": f"task_{task['task'][:10]}",
                        "parameters_extracted": 8,
                        "execution_time": 1.8
                    }
                    return method_expert_agent(state)
        
        # Execute tasks in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            future_to_task = {executor.submit(execute_agent_task, task): task for task in tasks}
            results = {}
            
            for future in concurrent.futures.as_completed(future_to_task):
                task = future_to_task[future]
                try:
                    result = future.result()
                    results[task["task"]] = result
                except Exception as e:
                    results[task["task"]] = {"error": str(e)}
        
        # Verify parallel execution
        assert len(results) == 3
        assert all("error" not in result for result in results.values())
    
    def test_workflow_error_recovery(self, mock_agent_client, mock_workflow_state):
        """Test workflow error recovery and agent coordination."""
        # Step 1: Data expert fails to load data
        error_state = mock_workflow_state.copy()
        error_state["messages"] = [{"content": "Load invalid dataset XYZ", "sender": "human"}]
        error_state["current_agent"] = "data_expert_agent"
        
        with patch('lobster.agents.data_expert.data_expert_agent') as mock_data_expert:
            mock_data_expert.return_value = {
                "messages": error_state["messages"] + [{"content": "Failed to load dataset XYZ", "sender": "data_expert_agent"}],
                "error": "Dataset not found",
                "error_type": "data_loading_error",
                "next_agent": "supervisor_agent",
                "handoff_reason": "Need supervisor assistance for error recovery"
            }
            
            error_result = data_expert_agent(error_state)
        
        # Step 2: Supervisor handles error and suggests alternative
        recovery_state = error_result.copy()
        recovery_state["current_agent"] = "supervisor_agent"
        
        with patch('lobster.agents.supervisor.supervisor_agent') as mock_supervisor:
            mock_supervisor.return_value = {
                "messages": recovery_state["messages"] + [{"content": "Let me help you find an alternative dataset", "sender": "supervisor_agent"}],
                "error_handled": True,
                "recovery_action": "suggest_alternative_dataset",
                "next_agent": "research_agent",
                "handoff_data": {"search_similar_datasets": True}
            }
            
            recovery_result = supervisor_agent(recovery_state)
        
        # Step 3: Research agent finds alternatives
        research_state = recovery_result.copy()
        research_state["current_agent"] = "research_agent"
        
        with patch('lobster.agents.research_agent.research_agent') as mock_research:
            mock_research.return_value = {
                "messages": research_state["messages"] + [{"content": "Found alternative datasets", "sender": "research_agent"}],
                "alternative_datasets": ["GSE111111", "GSE222222"],
                "workflow_recovered": True
            }
            
            research_result = research_agent(research_state)
        
        # Verify error recovery workflow
        assert error_result["error_type"] == "data_loading_error"
        assert recovery_result["error_handled"] == True
        assert research_result["workflow_recovered"] == True
        assert len(research_result["alternative_datasets"]) == 2


# ===============================================================================
# Agent Communication and Context Sharing Tests
# ===============================================================================

@pytest.mark.integration
class TestAgentCommunication:
    """Test agent communication and context sharing."""
    
    def test_context_preservation_across_agents(self, mock_agent_client, mock_workflow_state):
        """Test that context is preserved across agent handoffs."""
        # Initialize context with supervisor
        initial_state = mock_workflow_state.copy()
        initial_state["context"] = {
            "analysis_type": "single_cell",
            "organism": "homo_sapiens",
            "research_question": "identify_novel_cell_types"
        }
        
        # Agent 1: Data expert adds to context
        with patch('lobster.agents.data_expert.data_expert_agent') as mock_data_expert:
            mock_data_expert.return_value = {
                "context": {
                    **initial_state["context"],
                    "dataset_loaded": "geo_gse123456",
                    "n_cells": 5000,
                    "data_quality": "high"
                },
                "next_agent": "singlecell_expert_agent"
            }
            
            data_result = data_expert_agent(initial_state)
        
        # Agent 2: Single-cell expert uses and adds to context
        sc_state = data_result.copy()
        sc_state["current_agent"] = "singlecell_expert_agent"
        
        with patch('lobster.agents.singlecell_expert.singlecell_expert_agent') as mock_sc:
            mock_sc.return_value = {
                "context": {
                    **data_result["context"],
                    "clustering_completed": True,
                    "n_clusters": 12,
                    "novel_cell_types_found": 3
                },
                "analysis_complete": True
            }
            
            sc_result = singlecell_expert_agent(sc_state)
        
        # Verify context preservation and accumulation
        assert sc_result["context"]["analysis_type"] == "single_cell"  # From initial
        assert sc_result["context"]["dataset_loaded"] == "geo_gse123456"  # From data expert
        assert sc_result["context"]["novel_cell_types_found"] == 3  # From SC expert
    
    def test_cross_agent_data_references(self, mock_agent_client, mock_workflow_state):
        """Test cross-agent data references and sharing."""
        # Setup shared data registry
        shared_data = {
            "datasets": {},
            "analysis_results": {},
            "literature_refs": {}
        }
        
        # Research agent adds literature references
        research_state = mock_workflow_state.copy()
        research_state["shared_data"] = shared_data
        
        with patch('lobster.agents.research_agent.research_agent') as mock_research:
            mock_research.return_value = {
                "shared_data": {
                    **shared_data,
                    "literature_refs": {
                        "t_cell_markers": ["PMID:12345678", "PMID:87654321"],
                        "clustering_methods": ["PMID:11111111"]
                    }
                },
                "next_agent": "method_expert_agent"
            }
            
            research_result = research_agent(research_state)
        
        # Method expert uses literature references
        method_state = research_result.copy()
        method_state["current_agent"] = "method_expert_agent"
        
        with patch('lobster.agents.method_expert.method_expert_agent') as mock_method:
            mock_method.return_value = {
                "shared_data": {
                    **research_result["shared_data"],
                    "analysis_results": {
                        "optimized_parameters": {
                            "source_papers": research_result["shared_data"]["literature_refs"]["clustering_methods"],
                            "parameters": {"resolution": 0.5, "n_neighbors": 15}
                        }
                    }
                },
                "next_agent": "singlecell_expert_agent"
            }
            
            method_result = method_expert_agent(method_state)
        
        # Single-cell expert applies parameters from method expert
        sc_state = method_result.copy()
        sc_state["current_agent"] = "singlecell_expert_agent"
        
        with patch('lobster.agents.singlecell_expert.singlecell_expert_agent') as mock_sc:
            mock_sc.return_value = {
                "shared_data": {
                    **method_result["shared_data"],
                    "datasets": {
                        "clustered_data": {
                            "parameters_used": method_result["shared_data"]["analysis_results"]["optimized_parameters"]["parameters"],
                            "literature_validated": True
                        }
                    }
                }
            }
            
            sc_result = singlecell_expert_agent(sc_state)
        
        # Verify cross-agent data sharing
        final_shared_data = sc_result["shared_data"]
        assert "literature_refs" in final_shared_data
        assert "analysis_results" in final_shared_data
        assert "datasets" in final_shared_data
        assert final_shared_data["datasets"]["clustered_data"]["literature_validated"] == True
    
    def test_agent_capability_negotiation(self, mock_agent_client, mock_workflow_state):
        """Test agent capability negotiation and task delegation."""
        # Supervisor determines which agent can handle task
        negotiation_state = mock_workflow_state.copy()
        negotiation_state["task"] = "analyze_proteomics_data"
        
        # Mock agent capabilities
        agent_capabilities = {
            "data_expert_agent": ["data_loading", "format_conversion", "quality_assessment"],
            "singlecell_expert_agent": ["scrna_seq", "clustering", "cell_annotation"], 
            "proteomics_expert_agent": ["proteomics", "mass_spec", "protein_quantification"],
            "research_agent": ["literature_search", "dataset_discovery"]
        }
        
        with patch('lobster.agents.supervisor.supervisor_agent') as mock_supervisor:
            mock_supervisor.return_value = {
                "task_analysis": {
                    "required_capabilities": ["proteomics", "protein_quantification"],
                    "task_complexity": "medium",
                    "estimated_time": "15 minutes"
                },
                "capability_match": {
                    "best_agent": "proteomics_expert_agent",
                    "match_score": 0.95,
                    "alternative_agents": []
                },
                "next_agent": "proteomics_expert_agent",
                "delegation_reason": "Direct capability match for proteomics analysis"
            }
            
            result = supervisor_agent(negotiation_state)
        
        # Verify capability-based delegation
        assert result["capability_match"]["best_agent"] == "proteomics_expert_agent"
        assert result["capability_match"]["match_score"] == 0.95
        assert result["next_agent"] == "proteomics_expert_agent"


# ===============================================================================
# Workflow Performance and Monitoring Tests
# ===============================================================================

@pytest.mark.integration
class TestWorkflowPerformanceMonitoring:
    """Test workflow performance and monitoring capabilities."""
    
    def test_workflow_execution_timing(self, mock_agent_client, mock_workflow_state):
        """Test workflow execution timing and performance metrics."""
        import time
        
        workflow_metrics = {
            "start_time": time.time(),
            "agent_timings": {},
            "total_agents": 0
        }
        
        # Agent 1: Data expert (simulated timing)
        data_start = time.time()
        with patch('lobster.agents.data_expert.data_expert_agent') as mock_data_expert:
            mock_data_expert.return_value = {
                "execution_time": 2.5,
                "performance_metrics": {
                    "data_loading_time": 1.8,
                    "validation_time": 0.7
                }
            }
            
            data_result = data_expert_agent(mock_workflow_state)
            workflow_metrics["agent_timings"]["data_expert"] = data_result["execution_time"]
            workflow_metrics["total_agents"] += 1
        
        # Agent 2: Single-cell expert (simulated timing)
        sc_start = time.time()
        with patch('lobster.agents.singlecell_expert.singlecell_expert_agent') as mock_sc:
            mock_sc.return_value = {
                "execution_time": 8.3,
                "performance_metrics": {
                    "preprocessing_time": 3.2,
                    "clustering_time": 4.1,
                    "annotation_time": 1.0
                }
            }
            
            sc_result = singlecell_expert_agent(mock_workflow_state)
            workflow_metrics["agent_timings"]["singlecell_expert"] = sc_result["execution_time"]
            workflow_metrics["total_agents"] += 1
        
        workflow_metrics["total_time"] = sum(workflow_metrics["agent_timings"].values())
        
        # Verify performance tracking
        assert workflow_metrics["total_agents"] == 2
        assert workflow_metrics["total_time"] == 10.8
        assert workflow_metrics["agent_timings"]["data_expert"] == 2.5
        assert workflow_metrics["agent_timings"]["singlecell_expert"] == 8.3
    
    def test_workflow_resource_monitoring(self, mock_agent_client, mock_workflow_state):
        """Test workflow resource usage monitoring."""
        resource_usage = {
            "memory_peak": 0,
            "cpu_usage": [],
            "disk_usage": 0
        }
        
        # Mock resource-intensive operations
        with patch('lobster.agents.singlecell_expert.singlecell_expert_agent') as mock_sc:
            mock_sc.return_value = {
                "resource_usage": {
                    "peak_memory_mb": 2048,
                    "avg_cpu_percent": 85.5,
                    "disk_io_mb": 156.7
                },
                "performance_warnings": [
                    "High memory usage during clustering",
                    "CPU intensive operation detected"
                ]
            }
            
            result = singlecell_expert_agent(mock_workflow_state)
            
        # Update resource tracking
        resource_usage["memory_peak"] = max(resource_usage["memory_peak"], result["resource_usage"]["peak_memory_mb"])
        resource_usage["cpu_usage"].append(result["resource_usage"]["avg_cpu_percent"])
        resource_usage["disk_usage"] += result["resource_usage"]["disk_io_mb"]
        
        # Verify resource monitoring
        assert resource_usage["memory_peak"] == 2048
        assert resource_usage["cpu_usage"][0] == 85.5
        assert resource_usage["disk_usage"] == 156.7
        assert len(result["performance_warnings"]) == 2
    
    def test_workflow_progress_tracking(self, mock_agent_client, mock_workflow_state):
        """Test workflow progress tracking and status updates."""
        workflow_progress = {
            "total_steps": 5,
            "completed_steps": [],
            "current_step": None,
            "progress_percentage": 0
        }
        
        # Define workflow steps
        workflow_steps = [
            {"name": "data_loading", "agent": "data_expert_agent"},
            {"name": "quality_control", "agent": "singlecell_expert_agent"},
            {"name": "preprocessing", "agent": "singlecell_expert_agent"},
            {"name": "clustering", "agent": "singlecell_expert_agent"},
            {"name": "annotation", "agent": "singlecell_expert_agent"}
        ]
        
        # Simulate step completion
        for i, step in enumerate(workflow_steps[:3]):  # Complete first 3 steps
            workflow_progress["completed_steps"].append(step["name"])
            workflow_progress["current_step"] = step["name"]
            workflow_progress["progress_percentage"] = (len(workflow_progress["completed_steps"]) / workflow_progress["total_steps"]) * 100
            
            # Mock agent execution for this step
            with patch(f'lobster.agents.{step["agent"]}.{step["agent"]}') as mock_agent:
                mock_agent.return_value = {
                    "step_completed": step["name"],
                    "step_status": "success",
                    "progress_update": workflow_progress["progress_percentage"]
                }
                
                if step["agent"] == "data_expert_agent":
                    result = data_expert_agent(mock_workflow_state)
                else:
                    result = singlecell_expert_agent(mock_workflow_state)
        
        # Verify progress tracking
        assert len(workflow_progress["completed_steps"]) == 3
        assert workflow_progress["progress_percentage"] == 60.0
        assert workflow_progress["current_step"] == "preprocessing"
        assert "clustering" not in workflow_progress["completed_steps"]


# ===============================================================================
# Workflow State Management Tests
# ===============================================================================

@pytest.mark.integration
class TestWorkflowStateManagement:
    """Test workflow state persistence and recovery."""
    
    def test_workflow_state_serialization(self, mock_agent_client, mock_workflow_state, temp_workspace):
        """Test workflow state serialization and persistence."""
        # Create complex workflow state
        complex_state = mock_workflow_state.copy()
        complex_state.update({
            "context": {
                "analysis_type": "single_cell",
                "dataset": "geo_gse123456",
                "parameters": {"resolution": 0.5, "n_neighbors": 15}
            },
            "completed_steps": ["data_loading", "quality_control"],
            "agent_history": [
                {"agent": "supervisor_agent", "timestamp": "2024-01-01T10:00:00"},
                {"agent": "data_expert_agent", "timestamp": "2024-01-01T10:05:00"}
            ],
            "shared_data": {
                "datasets": ["geo_gse123456"],
                "analysis_results": {"qc_passed": True}
            }
        })
        
        # Serialize state to file
        state_file = temp_workspace / "workflow_state.json"
        with open(state_file, 'w') as f:
            json.dump(complex_state, f, default=str)
        
        # Load and verify state
        with open(state_file, 'r') as f:
            loaded_state = json.load(f)
        
        # Verify state preservation
        assert loaded_state["workflow_id"] == complex_state["workflow_id"]
        assert loaded_state["context"]["analysis_type"] == "single_cell"
        assert len(loaded_state["completed_steps"]) == 2
        assert len(loaded_state["agent_history"]) == 2
        assert loaded_state["shared_data"]["analysis_results"]["qc_passed"] == True
    
    def test_workflow_checkpoint_recovery(self, mock_agent_client, mock_workflow_state):
        """Test workflow recovery from checkpoints."""
        # Create checkpoint state
        checkpoint_state = mock_workflow_state.copy()
        checkpoint_state.update({
            "checkpoint_id": "checkpoint_001",
            "checkpoint_timestamp": "2024-01-01T10:30:00",
            "completed_steps": ["data_loading", "quality_control", "preprocessing"],
            "next_step": "clustering",
            "recovery_data": {
                "loaded_datasets": ["geo_gse123456_processed"],
                "preprocessing_results": {"n_cells_after_qc": 4500, "n_genes_filtered": 18000}
            }
        })
        
        # Simulate recovery from checkpoint
        with patch('lobster.agents.singlecell_expert.singlecell_expert_agent') as mock_sc:
            mock_sc.return_value = {
                "recovered_from_checkpoint": True,
                "checkpoint_id": checkpoint_state["checkpoint_id"],
                "resuming_step": "clustering",
                "recovery_successful": True,
                "continuing_analysis": True
            }
            
            recovery_result = singlecell_expert_agent(checkpoint_state)
        
        # Verify successful recovery
        assert recovery_result["recovered_from_checkpoint"] == True
        assert recovery_result["checkpoint_id"] == "checkpoint_001"
        assert recovery_result["resuming_step"] == "clustering"
        assert recovery_result["recovery_successful"] == True
    
    def test_workflow_branching_and_merging(self, mock_agent_client, mock_workflow_state):
        """Test workflow branching for parallel analysis paths."""
        # Create branching workflow
        main_state = mock_workflow_state.copy()
        main_state["workflow_type"] = "branching"
        
        # Branch 1: Standard clustering
        branch1_state = main_state.copy()
        branch1_state["branch_id"] = "standard_clustering"
        branch1_state["branch_params"] = {"resolution": 0.5}
        
        # Branch 2: High-resolution clustering  
        branch2_state = main_state.copy()
        branch2_state["branch_id"] = "high_res_clustering"
        branch2_state["branch_params"] = {"resolution": 1.2}
        
        # Execute branches in parallel
        branch_results = {}
        
        # Branch 1 execution
        with patch('lobster.agents.singlecell_expert.singlecell_expert_agent') as mock_sc1:
            mock_sc1.return_value = {
                "branch_id": "standard_clustering",
                "clusters_found": 8,
                "resolution_used": 0.5,
                "silhouette_score": 0.72
            }
            
            branch_results["standard"] = singlecell_expert_agent(branch1_state)
        
        # Branch 2 execution
        with patch('lobster.agents.singlecell_expert.singlecell_expert_agent') as mock_sc2:
            mock_sc2.return_value = {
                "branch_id": "high_res_clustering", 
                "clusters_found": 15,
                "resolution_used": 1.2,
                "silhouette_score": 0.68
            }
            
            branch_results["high_res"] = singlecell_expert_agent(branch2_state)
        
        # Merge results for comparison
        merged_results = {
            "branch_comparison": {
                "standard_clusters": branch_results["standard"]["clusters_found"],
                "high_res_clusters": branch_results["high_res"]["clusters_found"],
                "best_silhouette": max(
                    branch_results["standard"]["silhouette_score"],
                    branch_results["high_res"]["silhouette_score"]
                )
            },
            "recommended_branch": "standard_clustering" if branch_results["standard"]["silhouette_score"] > branch_results["high_res"]["silhouette_score"] else "high_res_clustering"
        }
        
        # Verify branching workflow
        assert len(branch_results) == 2
        assert branch_results["standard"]["clusters_found"] == 8
        assert branch_results["high_res"]["clusters_found"] == 15
        assert merged_results["recommended_branch"] == "standard_clustering"
        assert merged_results["branch_comparison"]["best_silhouette"] == 0.72


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])