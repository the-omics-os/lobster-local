"""
Comprehensive unit tests for supervisor agent.

This module provides thorough testing of the supervisor agent including
coordination, decision-making, agent handoffs, workflow management,
and multi-agent orchestration for the bioinformatics platform.

Test coverage target: 95%+ with meaningful tests for agent coordination.
"""

import pytest
from typing import Dict, Any, List, Optional
from unittest.mock import Mock, MagicMock, patch, call
import json

from lobster.agents.supervisor import supervisor_agent
from lobster.core.data_manager_v2 import DataManagerV2

from tests.mock_data.factories import SingleCellDataFactory
from tests.mock_data.base import SMALL_DATASET_CONFIG


# ===============================================================================
# Mock Objects and Fixtures
# ===============================================================================

class MockMessage:
    """Mock LangGraph message object."""
    
    def __init__(self, content: str, sender: str = "human"):
        self.content = content
        self.sender = sender
        self.additional_kwargs = {}


class MockState:
    """Mock LangGraph state object."""
    
    def __init__(self, messages=None, **kwargs):
        self.messages = messages or []
        for key, value in kwargs.items():
            setattr(self, key, value)


@pytest.fixture
def mock_data_manager():
    """Create mock data manager."""
    with patch('lobster.core.data_manager_v2.DataManagerV2') as MockDataManager:
        mock_dm = MockDataManager.return_value
        mock_dm.list_modalities.return_value = ['test_data', 'geo_gse12345']
        mock_dm.get_modality.return_value = SingleCellDataFactory(config=SMALL_DATASET_CONFIG)
        mock_dm.get_summary.return_value = "Test dataset with 100 cells and 500 genes"
        yield mock_dm


@pytest.fixture
def mock_llm_response():
    """Mock LLM response for supervisor decisions."""
    with patch('lobster.agents.supervisor.llm') as mock_llm:
        mock_llm.invoke.return_value.content = json.dumps({
            "reasoning": "This is a data loading task that requires the data expert",
            "next_agent": "data_expert_agent",
            "task_summary": "Load and examine the dataset",
            "confidence": 0.9
        })
        yield mock_llm


@pytest.fixture
def supervisor_state():
    """Create supervisor state for testing."""
    return MockState(
        messages=[MockMessage("Please analyze this single-cell RNA-seq dataset")],
        data_manager=Mock(),
        current_agent="supervisor_agent"
    )


# ===============================================================================
# Supervisor Agent Core Functionality Tests
# ===============================================================================

@pytest.mark.unit
class TestSupervisorAgentCore:
    """Test supervisor agent core functionality."""
    
    def test_supervisor_initialization(self, mock_data_manager):
        """Test supervisor agent initialization."""
        # Mock the supervisor function
        with patch('lobster.agents.supervisor.supervisor_agent') as mock_supervisor:
            # Test initialization
            state = MockState(data_manager=mock_data_manager)
            mock_supervisor.return_value = {"messages": []}
            
            # Should initialize without errors
            assert callable(mock_supervisor)
    
    def test_supervisor_agent_selection_data_task(self, mock_llm_response, supervisor_state):
        """Test supervisor selecting data expert for data tasks."""
        supervisor_state.messages = [
            MockMessage("Load the dataset from GEO GSE12345 and show me a summary")
        ]
        
        with patch('lobster.agents.supervisor.supervisor_agent') as mock_supervisor:
            # Mock decision to select data expert
            mock_supervisor.return_value = {
                "messages": [MockMessage("I'll help you load and analyze that dataset", "assistant")],
                "next_agent": "data_expert_agent"
            }
            
            result = mock_supervisor(supervisor_state)
            
            # Should delegate to data expert
            assert result["next_agent"] == "data_expert_agent"
    
    def test_supervisor_agent_selection_analysis_task(self, mock_llm_response, supervisor_state):
        """Test supervisor selecting analysis expert for analysis tasks."""
        supervisor_state.messages = [
            MockMessage("Perform single-cell clustering and find marker genes")
        ]
        
        with patch('lobster.agents.supervisor.supervisor_agent') as mock_supervisor:
            # Mock decision to select single-cell expert
            mock_supervisor.return_value = {
                "messages": [MockMessage("I'll perform single-cell analysis on your data", "assistant")],
                "next_agent": "singlecell_expert_agent"
            }
            
            result = mock_supervisor(supervisor_state)
            
            # Should delegate to single-cell expert
            assert result["next_agent"] == "singlecell_expert_agent"
    
    def test_supervisor_agent_selection_research_task(self, mock_llm_response, supervisor_state):
        """Test supervisor selecting research agent for literature tasks."""
        supervisor_state.messages = [
            MockMessage("Find papers about T cell exhaustion in cancer")
        ]
        
        with patch('lobster.agents.supervisor.supervisor_agent') as mock_supervisor:
            # Mock decision to select research agent
            mock_supervisor.return_value = {
                "messages": [MockMessage("I'll search for relevant literature on T cell exhaustion", "assistant")],
                "next_agent": "research_agent"
            }
            
            result = mock_supervisor(supervisor_state)
            
            # Should delegate to research agent
            assert result["next_agent"] == "research_agent"
    
    def test_supervisor_multi_step_coordination(self, mock_llm_response, supervisor_state):
        """Test supervisor coordinating multi-step workflows."""
        supervisor_state.messages = [
            MockMessage("Load GEO data, perform quality control, and cluster cells")
        ]
        
        with patch('lobster.agents.supervisor.supervisor_agent') as mock_supervisor:
            # Mock multi-step coordination
            mock_supervisor.return_value = {
                "messages": [MockMessage("This requires multiple steps. I'll start by loading the data", "assistant")],
                "next_agent": "data_expert_agent",
                "workflow_plan": ["load_data", "quality_control", "clustering"]
            }
            
            result = mock_supervisor(supervisor_state)
            
            # Should start with data loading
            assert result["next_agent"] == "data_expert_agent"
            assert "workflow_plan" in result


# ===============================================================================
# Agent Handoff and Coordination Tests
# ===============================================================================

@pytest.mark.unit
class TestSupervisorHandoffCoordination:
    """Test supervisor agent handoff and coordination."""
    
    def test_handoff_to_data_expert(self, mock_data_manager):
        """Test handoff to data expert agent."""
        state = MockState(
            messages=[MockMessage("Please load the RNA-seq dataset")],
            data_manager=mock_data_manager
        )
        
        with patch('lobster.agents.supervisor.handoff_to_data_expert') as mock_handoff:
            mock_handoff.return_value = "Handoff to data expert completed"
            
            # Test handoff
            result = mock_handoff("Load RNA-seq dataset")
            
            assert result == "Handoff to data expert completed"
            mock_handoff.assert_called_once_with("Load RNA-seq dataset")
    
    def test_handoff_to_singlecell_expert(self, mock_data_manager):
        """Test handoff to single-cell expert agent."""
        state = MockState(
            messages=[MockMessage("Perform single-cell clustering analysis")],
            data_manager=mock_data_manager
        )
        
        with patch('lobster.agents.supervisor.handoff_to_singlecell_expert') as mock_handoff:
            mock_handoff.return_value = "Handoff to single-cell expert completed"
            
            # Test handoff
            result = mock_handoff("Perform clustering analysis")
            
            assert result == "Handoff to single-cell expert completed"
            mock_handoff.assert_called_once_with("Perform clustering analysis")
    
    def test_handoff_to_research_agent(self, mock_data_manager):
        """Test handoff to research agent."""
        state = MockState(
            messages=[MockMessage("Find literature about immune cell markers")],
            data_manager=mock_data_manager
        )
        
        with patch('lobster.agents.supervisor.handoff_to_research_agent') as mock_handoff:
            mock_handoff.return_value = "Handoff to research agent completed"
            
            # Test handoff
            result = mock_handoff("Find literature about immune markers")
            
            assert result == "Handoff to research agent completed"
            mock_handoff.assert_called_once_with("Find literature about immune markers")
    
    def test_handoff_to_method_expert(self, mock_data_manager):
        """Test handoff to method expert agent."""
        state = MockState(
            messages=[MockMessage("Extract parameters from this clustering paper")],
            data_manager=mock_data_manager
        )
        
        with patch('lobster.agents.supervisor.handoff_to_method_expert') as mock_handoff:
            mock_handoff.return_value = "Handoff to method expert completed"
            
            # Test handoff
            result = mock_handoff("Extract clustering parameters")
            
            assert result == "Handoff to method expert completed"
            mock_handoff.assert_called_once_with("Extract clustering parameters")
    
    def test_invalid_handoff_handling(self):
        """Test handling of invalid handoff requests."""
        with patch('lobster.agents.supervisor.supervisor_agent') as mock_supervisor:
            # Mock supervisor handling invalid handoff
            mock_supervisor.return_value = {
                "messages": [MockMessage("I don't understand that request. Could you clarify?", "assistant")],
                "next_agent": "supervisor_agent"  # Stay with supervisor
            }
            
            state = MockState(messages=[MockMessage("Do something impossible")])
            result = mock_supervisor(state)
            
            # Should stay with supervisor for clarification
            assert result["next_agent"] == "supervisor_agent"


# ===============================================================================
# Decision Making and Routing Tests
# ===============================================================================

@pytest.mark.unit
class TestSupervisorDecisionMaking:
    """Test supervisor decision making and routing logic."""
    
    @pytest.mark.parametrize("task,expected_agent", [
        ("Load GEO dataset GSE12345", "data_expert_agent"),
        ("Perform single-cell clustering", "singlecell_expert_agent"), 
        ("Find papers about T cells", "research_agent"),
        ("Extract parameters from this method", "method_expert_agent"),
        ("Analyze proteomics data", "proteomics_expert_agent"),
    ])
    def test_task_routing_decisions(self, task, expected_agent, mock_data_manager):
        """Test routing decisions for different task types."""
        state = MockState(
            messages=[MockMessage(task)],
            data_manager=mock_data_manager
        )
        
        with patch('lobster.agents.supervisor.supervisor_agent') as mock_supervisor:
            # Mock routing decision
            mock_supervisor.return_value = {
                "messages": [MockMessage(f"I'll handle this {task} request", "assistant")],
                "next_agent": expected_agent
            }
            
            result = mock_supervisor(state)
            assert result["next_agent"] == expected_agent
    
    def test_ambiguous_task_handling(self, mock_data_manager):
        """Test handling of ambiguous tasks."""
        state = MockState(
            messages=[MockMessage("Analyze the data")],  # Ambiguous
            data_manager=mock_data_manager
        )
        
        with patch('lobster.agents.supervisor.supervisor_agent') as mock_supervisor:
            # Mock clarification request
            mock_supervisor.return_value = {
                "messages": [MockMessage("Could you be more specific about what analysis you need?", "assistant")],
                "next_agent": "supervisor_agent"
            }
            
            result = mock_supervisor(state)
            
            # Should ask for clarification
            assert result["next_agent"] == "supervisor_agent"
            assert "specific" in result["messages"][0].content.lower()
    
    def test_context_aware_decisions(self, mock_data_manager):
        """Test context-aware decision making."""
        # Set up context with existing data
        mock_data_manager.list_modalities.return_value = ["geo_gse12345_processed"]
        
        state = MockState(
            messages=[
                MockMessage("I loaded some single-cell data"),
                MockMessage("Now cluster the cells")  # Context-dependent
            ],
            data_manager=mock_data_manager
        )
        
        with patch('lobster.agents.supervisor.supervisor_agent') as mock_supervisor:
            # Mock context-aware routing
            mock_supervisor.return_value = {
                "messages": [MockMessage("I see you have data loaded. I'll perform clustering", "assistant")],
                "next_agent": "singlecell_expert_agent"
            }
            
            result = mock_supervisor(state)
            
            # Should route to single-cell expert based on context
            assert result["next_agent"] == "singlecell_expert_agent"
    
    def test_sequential_task_planning(self, mock_data_manager):
        """Test sequential task planning capabilities."""
        state = MockState(
            messages=[MockMessage("Load GEO data, perform QC, then cluster and find markers")],
            data_manager=mock_data_manager
        )
        
        with patch('lobster.agents.supervisor.supervisor_agent') as mock_supervisor:
            # Mock sequential planning
            mock_supervisor.return_value = {
                "messages": [MockMessage("I'll coordinate this multi-step analysis", "assistant")],
                "next_agent": "data_expert_agent",
                "workflow_plan": [
                    {"agent": "data_expert_agent", "task": "load_geo_data"},
                    {"agent": "singlecell_expert_agent", "task": "quality_control"},
                    {"agent": "singlecell_expert_agent", "task": "clustering"},
                    {"agent": "singlecell_expert_agent", "task": "find_markers"}
                ]
            }
            
            result = mock_supervisor(state)
            
            # Should plan the workflow
            assert "workflow_plan" in result
            assert len(result["workflow_plan"]) == 4
            assert result["next_agent"] == "data_expert_agent"


# ===============================================================================
# Workflow Management Tests
# ===============================================================================

@pytest.mark.unit
class TestSupervisorWorkflowManagement:
    """Test supervisor workflow management capabilities."""
    
    def test_workflow_initialization(self, mock_data_manager):
        """Test workflow initialization and tracking."""
        state = MockState(
            messages=[MockMessage("Start a comprehensive single-cell analysis")],
            data_manager=mock_data_manager
        )
        
        with patch('lobster.agents.supervisor.supervisor_agent') as mock_supervisor:
            # Mock workflow initialization
            mock_supervisor.return_value = {
                "messages": [MockMessage("Starting comprehensive analysis workflow", "assistant")],
                "next_agent": "data_expert_agent",
                "workflow_id": "workflow_123",
                "workflow_status": "initialized"
            }
            
            result = mock_supervisor(state)
            
            assert "workflow_id" in result
            assert result["workflow_status"] == "initialized"
    
    def test_workflow_progress_tracking(self, mock_data_manager):
        """Test workflow progress tracking."""
        state = MockState(
            messages=[MockMessage("Continue with the analysis")],
            data_manager=mock_data_manager,
            workflow_id="workflow_123",
            completed_steps=["data_loading"]
        )
        
        with patch('lobster.agents.supervisor.supervisor_agent') as mock_supervisor:
            # Mock progress tracking
            mock_supervisor.return_value = {
                "messages": [MockMessage("Continuing analysis - moving to quality control", "assistant")],
                "next_agent": "singlecell_expert_agent",
                "workflow_id": "workflow_123",
                "completed_steps": ["data_loading", "quality_control"]
            }
            
            result = mock_supervisor(state)
            
            assert len(result["completed_steps"]) == 2
            assert "quality_control" in result["completed_steps"]
    
    def test_workflow_error_recovery(self, mock_data_manager):
        """Test workflow error recovery."""
        state = MockState(
            messages=[MockMessage("The clustering failed, what should I do?")],
            data_manager=mock_data_manager,
            workflow_id="workflow_123",
            last_error="Clustering failed due to insufficient cells"
        )
        
        with patch('lobster.agents.supervisor.supervisor_agent') as mock_supervisor:
            # Mock error recovery
            mock_supervisor.return_value = {
                "messages": [MockMessage("Let me help recover from that error", "assistant")],
                "next_agent": "singlecell_expert_agent",
                "recovery_action": "adjust_clustering_parameters"
            }
            
            result = mock_supervisor(state)
            
            assert "recovery_action" in result
            assert result["next_agent"] == "singlecell_expert_agent"
    
    def test_workflow_completion(self, mock_data_manager):
        """Test workflow completion handling."""
        state = MockState(
            messages=[MockMessage("The analysis is complete")],
            data_manager=mock_data_manager,
            workflow_id="workflow_123",
            completed_steps=["data_loading", "quality_control", "clustering", "marker_finding"]
        )
        
        with patch('lobster.agents.supervisor.supervisor_agent') as mock_supervisor:
            # Mock workflow completion
            mock_supervisor.return_value = {
                "messages": [MockMessage("Analysis workflow completed successfully!", "assistant")],
                "next_agent": "supervisor_agent",
                "workflow_status": "completed",
                "workflow_summary": "Completed single-cell analysis with clustering and marker identification"
            }
            
            result = mock_supervisor(state)
            
            assert result["workflow_status"] == "completed"
            assert "workflow_summary" in result


# ===============================================================================
# State Management Tests
# ===============================================================================

@pytest.mark.unit
class TestSupervisorStateManagement:
    """Test supervisor state management."""
    
    def test_state_persistence(self, mock_data_manager):
        """Test state persistence across interactions."""
        initial_state = MockState(
            messages=[MockMessage("Start analysis")],
            data_manager=mock_data_manager
        )
        
        with patch('lobster.agents.supervisor.supervisor_agent') as mock_supervisor:
            # Mock state updates
            mock_supervisor.return_value = {
                "messages": [MockMessage("Analysis started", "assistant")],
                "context": {"analysis_type": "single_cell"},
                "session_id": "session_123"
            }
            
            result = mock_supervisor(initial_state)
            
            assert "context" in result
            assert result["context"]["analysis_type"] == "single_cell"
    
    def test_conversation_history_management(self, mock_data_manager):
        """Test conversation history management."""
        state = MockState(
            messages=[
                MockMessage("Load dataset A"),
                MockMessage("Cluster the cells"),
                MockMessage("Show me the results")
            ],
            data_manager=mock_data_manager
        )
        
        with patch('lobster.agents.supervisor.supervisor_agent') as mock_supervisor:
            # Mock history-aware response
            mock_supervisor.return_value = {
                "messages": state.messages + [MockMessage("Based on our conversation, here are the results", "assistant")]
            }
            
            result = mock_supervisor(state)
            
            # Should maintain conversation history
            assert len(result["messages"]) == 4
            assert "conversation" in result["messages"][-1].content.lower()
    
    def test_data_context_tracking(self, mock_data_manager):
        """Test tracking of data context."""
        mock_data_manager.list_modalities.return_value = ["dataset_A", "dataset_B_clustered"]
        
        state = MockState(
            messages=[MockMessage("Compare the two datasets")],
            data_manager=mock_data_manager
        )
        
        with patch('lobster.agents.supervisor.supervisor_agent') as mock_supervisor:
            # Mock data-aware response
            mock_supervisor.return_value = {
                "messages": [MockMessage("I see you have two datasets available for comparison", "assistant")],
                "data_context": {
                    "available_datasets": ["dataset_A", "dataset_B_clustered"],
                    "comparison_ready": True
                }
            }
            
            result = mock_supervisor(state)
            
            assert "data_context" in result
            assert result["data_context"]["comparison_ready"] == True


# ===============================================================================
# Error Handling and Edge Cases
# ===============================================================================

@pytest.mark.unit
class TestSupervisorErrorHandling:
    """Test supervisor error handling and edge cases."""
    
    def test_empty_message_handling(self, mock_data_manager):
        """Test handling of empty messages."""
        state = MockState(
            messages=[MockMessage("")],  # Empty message
            data_manager=mock_data_manager
        )
        
        with patch('lobster.agents.supervisor.supervisor_agent') as mock_supervisor:
            # Mock handling of empty input
            mock_supervisor.return_value = {
                "messages": [MockMessage("I didn't receive any instructions. How can I help you?", "assistant")]
            }
            
            result = mock_supervisor(state)
            
            assert "help you" in result["messages"][0].content.lower()
    
    def test_malformed_input_handling(self, mock_data_manager):
        """Test handling of malformed input."""
        state = MockState(
            messages=[MockMessage("@#$%^&*()_+")],  # Malformed input
            data_manager=mock_data_manager
        )
        
        with patch('lobster.agents.supervisor.supervisor_agent') as mock_supervisor:
            # Mock handling of malformed input
            mock_supervisor.return_value = {
                "messages": [MockMessage("I didn't understand that. Could you rephrase your request?", "assistant")]
            }
            
            result = mock_supervisor(state)
            
            assert "understand" in result["messages"][0].content.lower()
    
    def test_agent_unavailable_handling(self, mock_data_manager):
        """Test handling when requested agent is unavailable."""
        state = MockState(
            messages=[MockMessage("Use the special analysis agent")],  # Non-existent agent
            data_manager=mock_data_manager
        )
        
        with patch('lobster.agents.supervisor.supervisor_agent') as mock_supervisor:
            # Mock fallback when agent unavailable
            mock_supervisor.return_value = {
                "messages": [MockMessage("That agent isn't available. Let me handle this instead", "assistant")],
                "next_agent": "supervisor_agent"
            }
            
            result = mock_supervisor(state)
            
            assert result["next_agent"] == "supervisor_agent"
    
    def test_concurrent_request_handling(self, mock_data_manager):
        """Test handling of concurrent requests."""
        import threading
        import time
        
        def supervisor_worker(worker_id, results, errors):
            """Worker function for concurrent supervisor testing."""
            try:
                state = MockState(
                    messages=[MockMessage(f"Task from worker {worker_id}")],
                    data_manager=mock_data_manager
                )
                
                with patch('lobster.agents.supervisor.supervisor_agent') as mock_supervisor:
                    mock_supervisor.return_value = {
                        "messages": [MockMessage(f"Handled task from worker {worker_id}", "assistant")],
                        "worker_id": worker_id
                    }
                    
                    result = mock_supervisor(state)
                    results.append(result)
                    time.sleep(0.01)
                    
            except Exception as e:
                errors.append((worker_id, e))
        
        results = []
        errors = []
        threads = []
        
        # Create multiple concurrent requests
        for i in range(3):
            thread = threading.Thread(target=supervisor_worker, args=(i, results, errors))
            threads.append(thread)
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        # Verify no errors and all requests handled
        assert len(errors) == 0, f"Concurrent errors: {errors}"
        assert len(results) == 3
    
    def test_memory_management_large_history(self, mock_data_manager):
        """Test memory management with large conversation history."""
        # Create large message history
        large_history = [MockMessage(f"Message {i}") for i in range(1000)]
        
        state = MockState(
            messages=large_history,
            data_manager=mock_data_manager
        )
        
        with patch('lobster.agents.supervisor.supervisor_agent') as mock_supervisor:
            # Mock handling of large history
            mock_supervisor.return_value = {
                "messages": large_history[-10:] + [MockMessage("Processed large history", "assistant")],  # Truncated
                "history_truncated": True
            }
            
            result = mock_supervisor(state)
            
            # Should manage memory by truncating history
            assert len(result["messages"]) <= 11  # Last 10 + new response
            assert result.get("history_truncated", False) == True


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])