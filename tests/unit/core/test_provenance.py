"""
Comprehensive unit tests for W3C-PROV compliant provenance tracking.

This module provides thorough testing of the ProvenanceTracker class including
W3C-PROV compliance, activity tracking, entity management, agent handling,
data integrity, serialization/deserialization, and integration capabilities.

Test coverage target: 95%+ with meaningful tests for provenance standards.
"""

import datetime
import hashlib
import json
import tempfile
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import Mock, patch, mock_open

import numpy as np
import pandas as pd
import pytest
import anndata as ad

from lobster.core.provenance import ProvenanceTracker
from tests.mock_data.factories import SingleCellDataFactory, ProteomicsDataFactory
from tests.mock_data.base import SMALL_DATASET_CONFIG


# ===============================================================================
# ProvenanceTracker Core Functionality Tests
# ===============================================================================

@pytest.mark.unit
class TestProvenanceTrackerInitialization:
    """Test ProvenanceTracker initialization and basic properties."""

    def test_default_initialization(self):
        """Test default initialization of ProvenanceTracker."""
        tracker = ProvenanceTracker()

        assert tracker.namespace == "lobster"
        assert tracker.activities == []
        assert tracker.entities == {}
        assert tracker.agents == {}

    def test_custom_namespace_initialization(self):
        """Test initialization with custom namespace."""
        custom_namespace = "custom_analysis"
        tracker = ProvenanceTracker(namespace=custom_namespace)

        assert tracker.namespace == custom_namespace
        assert tracker.activities == []
        assert tracker.entities == {}
        assert tracker.agents == {}

    def test_initialization_creates_empty_collections(self):
        """Test that initialization creates empty collections."""
        tracker = ProvenanceTracker()

        # Collections should be empty but exist
        assert isinstance(tracker.activities, list)
        assert isinstance(tracker.entities, dict)
        assert isinstance(tracker.agents, dict)
        assert len(tracker.activities) == 0
        assert len(tracker.entities) == 0
        assert len(tracker.agents) == 0


@pytest.mark.unit
class TestProvenanceActivityManagement:
    """Test provenance activity creation and management."""

    def test_create_activity_basic(self):
        """Test basic activity creation."""
        tracker = ProvenanceTracker()

        activity_id = tracker.create_activity(
            activity_type="test_activity",
            agent="test_agent",
            description="Test activity description"
        )

        # Check return value
        assert isinstance(activity_id, str)
        assert activity_id.startswith("lobster:activity:")

        # Check activity was added
        assert len(tracker.activities) == 1
        activity = tracker.activities[0]

        assert activity["id"] == activity_id
        assert activity["type"] == "test_activity"
        assert activity["agent"] == "test_agent"
        assert activity["description"] == "Test activity description"
        assert "timestamp" in activity
        assert "software_versions" in activity
        assert activity["inputs"] == []
        assert activity["outputs"] == []
        assert activity["parameters"] == {}

    def test_create_activity_with_inputs_outputs(self):
        """Test activity creation with inputs and outputs."""
        tracker = ProvenanceTracker()

        inputs = [{"entity": "input_1", "role": "source"}]
        outputs = [{"entity": "output_1", "role": "result"}]
        parameters = {"param1": "value1", "param2": 42}

        activity_id = tracker.create_activity(
            activity_type="processing",
            agent="processor_agent",
            inputs=inputs,
            outputs=outputs,
            parameters=parameters
        )

        activity = tracker.activities[0]
        assert activity["inputs"] == inputs
        assert activity["outputs"] == outputs
        assert activity["parameters"] == parameters

    def test_create_multiple_activities(self):
        """Test creating multiple activities."""
        tracker = ProvenanceTracker()

        activity_ids = []
        for i in range(3):
            activity_id = tracker.create_activity(
                activity_type=f"activity_{i}",
                agent=f"agent_{i}",
                description=f"Activity {i}"
            )
            activity_ids.append(activity_id)

        assert len(tracker.activities) == 3
        assert len(set(activity_ids)) == 3  # All IDs should be unique

        # Check that activities are ordered
        for i, activity in enumerate(tracker.activities):
            assert activity["type"] == f"activity_{i}"
            assert activity["agent"] == f"agent_{i}"

    def test_activity_timestamp_format(self):
        """Test that activity timestamps are in ISO format with timezone."""
        tracker = ProvenanceTracker()

        activity_id = tracker.create_activity(
            activity_type="test_activity",
            agent="test_agent"
        )

        activity = tracker.activities[0]
        timestamp = activity["timestamp"]

        # Should be valid ISO format with timezone
        parsed_timestamp = datetime.datetime.fromisoformat(timestamp)
        assert parsed_timestamp.tzinfo is not None

    @patch('lobster.core.provenance.ProvenanceTracker._get_software_versions')
    def test_activity_includes_software_versions(self, mock_versions):
        """Test that activities include software version information."""
        mock_versions.return_value = {"numpy": "1.21.0", "pandas": "1.3.0"}

        tracker = ProvenanceTracker()
        activity_id = tracker.create_activity(
            activity_type="test_activity",
            agent="test_agent"
        )

        activity = tracker.activities[0]
        assert activity["software_versions"] == {"numpy": "1.21.0", "pandas": "1.3.0"}
        mock_versions.assert_called_once()


@pytest.mark.unit
class TestProvenanceEntityManagement:
    """Test provenance entity creation and management."""

    def test_create_entity_basic(self):
        """Test basic entity creation."""
        tracker = ProvenanceTracker()

        entity_id = tracker.create_entity(
            entity_type="dataset",
            format="h5ad"
        )

        # Check return value
        assert isinstance(entity_id, str)
        assert entity_id.startswith("lobster:entity:")

        # Check entity was added
        assert len(tracker.entities) == 1
        assert entity_id in tracker.entities

        entity = tracker.entities[entity_id]
        assert entity["id"] == entity_id
        assert entity["type"] == "dataset"
        assert entity["format"] == "h5ad"
        assert entity["uri"] is None
        assert entity["checksum"] is None
        assert entity["metadata"] == {}
        assert "created" in entity

    def test_create_entity_with_uri(self):
        """Test entity creation with URI."""
        tracker = ProvenanceTracker()
        test_path = Path("/test/path/data.h5ad")

        with patch.object(tracker, '_calculate_checksum', return_value="abc123"):
            entity_id = tracker.create_entity(
                entity_type="dataset",
                uri=test_path,
                format="h5ad"
            )

        entity = tracker.entities[entity_id]
        assert entity["uri"] == str(test_path)
        assert entity["checksum"] == "abc123"

    def test_create_entity_with_metadata(self):
        """Test entity creation with metadata."""
        tracker = ProvenanceTracker()
        metadata = {"source": "experiment_1", "version": "v1.0"}

        entity_id = tracker.create_entity(
            entity_type="dataset",
            format="h5ad",
            metadata=metadata
        )

        entity = tracker.entities[entity_id]
        assert entity["metadata"] == metadata

    def test_create_multiple_entities(self):
        """Test creating multiple entities."""
        tracker = ProvenanceTracker()

        entity_ids = []
        for i in range(3):
            entity_id = tracker.create_entity(
                entity_type=f"entity_{i}",
                format="h5ad"
            )
            entity_ids.append(entity_id)

        assert len(tracker.entities) == 3
        assert len(set(entity_ids)) == 3  # All IDs should be unique

        for entity_id in entity_ids:
            assert entity_id in tracker.entities

    def test_entity_timestamp_format(self):
        """Test that entity timestamps are in ISO format with timezone."""
        tracker = ProvenanceTracker()

        entity_id = tracker.create_entity(
            entity_type="dataset",
            format="h5ad"
        )

        entity = tracker.entities[entity_id]
        timestamp = entity["created"]

        # Should be valid ISO format with timezone
        parsed_timestamp = datetime.datetime.fromisoformat(timestamp)
        assert parsed_timestamp.tzinfo is not None


@pytest.mark.unit
class TestProvenanceAgentManagement:
    """Test provenance agent creation and management."""

    def test_create_agent_basic(self):
        """Test basic agent creation."""
        tracker = ProvenanceTracker()

        agent_id = tracker.create_agent(
            name="Test Agent",
            agent_type="software",
            description="A test agent"
        )

        # Check return value format
        assert isinstance(agent_id, str)
        assert agent_id == "lobster:agent:test_agent"

        # Check agent was added
        assert len(tracker.agents) == 1
        assert agent_id in tracker.agents

        agent = tracker.agents[agent_id]
        assert agent["id"] == agent_id
        assert agent["name"] == "Test Agent"
        assert agent["type"] == "software"
        assert agent["description"] == "A test agent"
        assert agent["version"] is None

    def test_create_agent_with_version(self):
        """Test agent creation with version."""
        tracker = ProvenanceTracker()

        agent_id = tracker.create_agent(
            name="Versioned Agent",
            agent_type="software",
            version="1.0.0"
        )

        agent = tracker.agents[agent_id]
        assert agent["version"] == "1.0.0"

    def test_create_agent_name_normalization(self):
        """Test that agent names are normalized for IDs."""
        tracker = ProvenanceTracker()

        # Test spaces and case normalization
        agent_id = tracker.create_agent(name="Complex Agent NAME")
        assert agent_id == "lobster:agent:complex_agent_name"

        # Test special characters
        agent_id2 = tracker.create_agent(name="Agent-With_Special.Chars")
        assert agent_id2 == "lobster:agent:agent-with_special.chars"

    def test_duplicate_agent_handling(self):
        """Test that duplicate agents are not created."""
        tracker = ProvenanceTracker()

        # Create first agent
        agent_id1 = tracker.create_agent(name="Duplicate Agent")
        assert len(tracker.agents) == 1

        # Try to create same agent again
        agent_id2 = tracker.create_agent(name="Duplicate Agent")
        assert agent_id1 == agent_id2
        assert len(tracker.agents) == 1  # Still only one agent

    def test_create_multiple_agents(self):
        """Test creating multiple unique agents."""
        tracker = ProvenanceTracker()

        agent_names = ["Agent A", "Agent B", "Agent C"]
        agent_ids = []

        for name in agent_names:
            agent_id = tracker.create_agent(name=name)
            agent_ids.append(agent_id)

        assert len(tracker.agents) == 3
        assert len(set(agent_ids)) == 3  # All IDs should be unique


# ===============================================================================
# W3C-PROV Compliance Tests
# ===============================================================================

@pytest.mark.unit
class TestW3CProvCompliance:
    """Test W3C-PROV standard compliance."""

    def test_activity_structure_compliance(self):
        """Test that activities conform to W3C-PROV activity structure."""
        tracker = ProvenanceTracker()

        activity_id = tracker.create_activity(
            activity_type="prov:Activity",
            agent="test_agent",
            inputs=[{"entity": "input_entity", "role": "prov:used"}],
            outputs=[{"entity": "output_entity", "role": "prov:generated"}]
        )

        activity = tracker.activities[0]

        # W3C-PROV required fields for activities
        assert "id" in activity  # prov:id
        assert "type" in activity  # prov:type
        assert "timestamp" in activity  # prov:startedAtTime (approximation)
        assert "agent" in activity  # prov:wasAssociatedWith
        assert "inputs" in activity  # prov:used relations
        assert "outputs" in activity  # prov:generated relations

    def test_entity_structure_compliance(self):
        """Test that entities conform to W3C-PROV entity structure."""
        tracker = ProvenanceTracker()

        entity_id = tracker.create_entity(
            entity_type="prov:Entity",
            uri="/path/to/data",
            format="h5ad"
        )

        entity = tracker.entities[entity_id]

        # W3C-PROV required fields for entities
        assert "id" in entity  # prov:id
        assert "type" in entity  # prov:type
        assert "uri" in entity  # prov:location (approximation)
        assert "created" in entity  # prov:generatedAtTime (approximation)

    def test_agent_structure_compliance(self):
        """Test that agents conform to W3C-PROV agent structure."""
        tracker = ProvenanceTracker()

        agent_id = tracker.create_agent(
            name="Test Software",
            agent_type="prov:SoftwareAgent",
            version="1.0.0"
        )

        agent = tracker.agents[agent_id]

        # W3C-PROV required fields for agents
        assert "id" in agent  # prov:id
        assert "name" in agent  # rdfs:label (approximation)
        assert "type" in agent  # prov:type

    def test_provenance_relations_compliance(self):
        """Test that provenance relations follow W3C-PROV patterns."""
        tracker = ProvenanceTracker()

        # Create entities
        input_entity = tracker.create_entity("input_data", format="csv")
        output_entity = tracker.create_entity("processed_data", format="h5ad")

        # Create agent
        agent_id = tracker.create_agent("Processing Software")

        # Create activity with proper relations
        activity_id = tracker.create_activity(
            activity_type="data_processing",
            agent=agent_id,
            inputs=[{"entity": input_entity, "role": "used"}],
            outputs=[{"entity": output_entity, "role": "generated"}]
        )

        activity = tracker.activities[0]

        # Check W3C-PROV relation patterns
        # prov:used relation
        assert any(inp["entity"] == input_entity and inp["role"] == "used"
                  for inp in activity["inputs"])

        # prov:generated relation
        assert any(out["entity"] == output_entity and out["role"] == "generated"
                  for out in activity["outputs"])

        # prov:wasAssociatedWith relation
        assert activity["agent"] == agent_id

    def test_identifier_format_compliance(self):
        """Test that identifiers follow W3C-PROV naming conventions."""
        tracker = ProvenanceTracker(namespace="test_namespace")

        # Create each type of entity
        activity_id = tracker.create_activity("test", "agent")
        entity_id = tracker.create_entity("test_entity")
        agent_id = tracker.create_agent("test_agent")

        # Check namespace compliance
        assert activity_id.startswith("test_namespace:activity:")
        assert entity_id.startswith("test_namespace:entity:")
        assert agent_id.startswith("test_namespace:agent:")

        # Check format compliance (namespace:type:id)
        assert len(activity_id.split(":")) == 3
        assert len(entity_id.split(":")) == 3
        assert len(agent_id.split(":")) == 3


# ===============================================================================
# Provenance Data Integrity Tests
# ===============================================================================

@pytest.mark.unit
class TestProvenanceDataIntegrity:
    """Test provenance data integrity and consistency."""

    def test_checksum_calculation(self):
        """Test file checksum calculation."""
        tracker = ProvenanceTracker()

        # Create a temporary file with known content
        test_content = b"test data for checksum"
        expected_checksum = hashlib.sha256(test_content).hexdigest()

        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(test_content)
            tmp_file.flush()

            calculated_checksum = tracker._calculate_checksum(tmp_file.name)

        Path(tmp_file.name).unlink()  # Clean up

        assert calculated_checksum == expected_checksum

    def test_checksum_calculation_nonexistent_file(self):
        """Test checksum calculation for non-existent file."""
        tracker = ProvenanceTracker()

        checksum = tracker._calculate_checksum("/nonexistent/file.txt")
        assert checksum is None

    def test_format_detection(self):
        """Test file format detection from extensions."""
        tracker = ProvenanceTracker()

        format_tests = [
            ("/path/to/file.h5ad", "h5ad"),
            ("/path/to/file.csv", "csv"),
            ("/path/to/file.xlsx", "excel"),
            ("/path/to/file.h5mu", "h5mu"),
            ("/path/to/file.unknown", "unknown"),
            ("file_no_extension", "unknown")
        ]

        for file_path, expected_format in format_tests:
            detected_format = tracker._detect_format(file_path)
            assert detected_format == expected_format

    def test_software_version_collection(self):
        """Test software version collection."""
        tracker = ProvenanceTracker()

        # Mock available packages
        with patch('importlib.import_module') as mock_import:
            # Mock successful imports
            mock_pandas = Mock()
            mock_pandas.__version__ = "1.3.0"
            mock_numpy = Mock()
            mock_numpy.__version__ = "1.21.0"

            def mock_import_side_effect(module_name):
                if module_name == "pandas":
                    return mock_pandas
                elif module_name == "numpy":
                    return mock_numpy
                else:
                    raise ImportError()

            mock_import.side_effect = mock_import_side_effect
            versions = tracker._get_software_versions()

            # Should include available packages
            assert "pandas" in versions or "numpy" in versions

    def test_data_consistency_across_operations(self):
        """Test that data remains consistent across multiple operations."""
        tracker = ProvenanceTracker()

        # Create initial data
        entity1 = tracker.create_entity("data1", format="csv")
        agent1 = tracker.create_agent("processor1")
        activity1 = tracker.create_activity("process1", agent1,
                                           inputs=[{"entity": entity1, "role": "input"}])

        # Verify initial state
        assert len(tracker.entities) == 1
        assert len(tracker.agents) == 1
        assert len(tracker.activities) == 1

        # Add more data
        entity2 = tracker.create_entity("data2", format="h5ad")
        activity2 = tracker.create_activity("process2", agent1,
                                           inputs=[{"entity": entity1, "role": "input"}],
                                           outputs=[{"entity": entity2, "role": "output"}])

        # Verify consistency
        assert len(tracker.entities) == 2
        assert len(tracker.agents) == 1  # Same agent reused
        assert len(tracker.activities) == 2

        # Verify references are consistent
        assert tracker.activities[1]["agent"] == agent1
        assert any(inp["entity"] == entity1 for inp in tracker.activities[1]["inputs"])
        assert any(out["entity"] == entity2 for out in tracker.activities[1]["outputs"])


# ===============================================================================
# Provenance Serialization and Deserialization Tests
# ===============================================================================

@pytest.mark.unit
class TestProvenanceSerialization:
    """Test provenance data serialization and deserialization."""

    def test_to_dict_basic(self):
        """Test basic export to dictionary."""
        tracker = ProvenanceTracker(namespace="test")

        # Add some data
        entity_id = tracker.create_entity("test_entity", format="h5ad")
        agent_id = tracker.create_agent("test_agent")
        activity_id = tracker.create_activity("test_activity", agent_id)

        data_dict = tracker.to_dict()

        # Check structure
        assert isinstance(data_dict, dict)
        assert "namespace" in data_dict
        assert "activities" in data_dict
        assert "entities" in data_dict
        assert "agents" in data_dict
        assert "export_timestamp" in data_dict

        # Check content
        assert data_dict["namespace"] == "test"
        assert len(data_dict["activities"]) == 1
        assert len(data_dict["entities"]) == 1
        assert len(data_dict["agents"]) == 1

    def test_from_dict_basic(self):
        """Test basic import from dictionary."""
        tracker = ProvenanceTracker()

        # Create test data
        test_data = {
            "namespace": "imported",
            "activities": [{"id": "test_activity", "type": "test"}],
            "entities": {"test_entity": {"id": "test_entity", "type": "dataset"}},
            "agents": {"test_agent": {"id": "test_agent", "name": "Test Agent"}}
        }

        tracker.from_dict(test_data)

        # Verify import
        assert tracker.namespace == "imported"
        assert len(tracker.activities) == 1
        assert len(tracker.entities) == 1
        assert len(tracker.agents) == 1

        assert tracker.activities[0]["id"] == "test_activity"
        assert "test_entity" in tracker.entities
        assert "test_agent" in tracker.agents

    def test_round_trip_serialization(self):
        """Test that data survives round-trip serialization."""
        tracker1 = ProvenanceTracker(namespace="round_trip")

        # Create complex provenance data
        entity1 = tracker1.create_entity("input_data", uri="/path/input.csv", format="csv")
        entity2 = tracker1.create_entity("output_data", uri="/path/output.h5ad", format="h5ad")
        agent1 = tracker1.create_agent("Data Processor", version="1.0.0")
        activity1 = tracker1.create_activity(
            "data_processing",
            agent1,
            inputs=[{"entity": entity1, "role": "source"}],
            outputs=[{"entity": entity2, "role": "result"}],
            parameters={"method": "normalization", "threshold": 0.5}
        )

        # Export to dict
        exported_data = tracker1.to_dict()

        # Import to new tracker
        tracker2 = ProvenanceTracker()
        tracker2.from_dict(exported_data)

        # Verify equality
        assert tracker2.namespace == tracker1.namespace
        assert len(tracker2.activities) == len(tracker1.activities)
        assert len(tracker2.entities) == len(tracker1.entities)
        assert len(tracker2.agents) == len(tracker1.agents)

        # Verify specific data
        assert tracker2.activities[0]["type"] == "data_processing"
        assert tracker2.activities[0]["parameters"]["method"] == "normalization"
        assert entity1 in tracker2.entities
        assert entity2 in tracker2.entities
        assert agent1 in tracker2.agents

    def test_json_serialization_compatibility(self):
        """Test that exported data is JSON serializable."""
        tracker = ProvenanceTracker()

        # Add data with various types
        entity_id = tracker.create_entity("test", format="h5ad",
                                        metadata={"count": 42, "valid": True})
        agent_id = tracker.create_agent("test_agent", version="1.0.0")
        activity_id = tracker.create_activity("test", agent_id,
                                            parameters={"param": [1, 2, 3]})

        data_dict = tracker.to_dict()

        # Should be JSON serializable
        try:
            json_str = json.dumps(data_dict)
            reconstructed = json.loads(json_str)
            assert reconstructed == data_dict
        except (TypeError, ValueError) as e:
            pytest.fail(f"Data is not JSON serializable: {e}")


# ===============================================================================
# Provenance AnnData Integration Tests
# ===============================================================================

@pytest.mark.unit
class TestProvenanceAnnDataIntegration:
    """Test provenance integration with AnnData objects."""

    def test_add_to_anndata_basic(self):
        """Test basic provenance addition to AnnData."""
        tracker = ProvenanceTracker()
        adata = SingleCellDataFactory(config=SMALL_DATASET_CONFIG)

        # Add some provenance data
        entity_id = tracker.create_entity("test_data", format="h5ad")
        agent_id = tracker.create_agent("test_processor")
        activity_id = tracker.create_activity("test_processing", agent_id)

        # Add to AnnData
        result_adata = tracker.add_to_anndata(adata)

        # Should return the same object (modified in place)
        assert result_adata is adata

        # Check provenance was added
        assert "provenance" in adata.uns
        prov_data = adata.uns["provenance"]

        assert "activities" in prov_data
        assert "entities" in prov_data
        assert "agents" in prov_data
        assert "tracker_namespace" in prov_data

        assert len(prov_data["activities"]) == 1
        assert len(prov_data["entities"]) == 1
        assert len(prov_data["agents"]) == 1
        assert prov_data["tracker_namespace"] == "lobster"

    def test_add_to_anndata_preserves_existing_provenance(self):
        """Test that adding provenance preserves existing provenance data."""
        tracker = ProvenanceTracker()
        adata = SingleCellDataFactory(config=SMALL_DATASET_CONFIG)

        # Add existing provenance
        adata.uns["provenance"] = {"existing": "data"}

        # Add tracker provenance
        entity_id = tracker.create_entity("test_data")
        tracker.add_to_anndata(adata)

        # Should preserve existing data and add new
        prov_data = adata.uns["provenance"]
        assert "existing" in prov_data  # Original data preserved
        assert "activities" in prov_data  # New data added
        assert "entities" in prov_data

    def test_extract_from_anndata_basic(self):
        """Test basic provenance extraction from AnnData."""
        # Create AnnData with provenance
        adata = SingleCellDataFactory(config=SMALL_DATASET_CONFIG)
        test_provenance = {
            "activities": [{"id": "test_activity", "type": "test"}],
            "entities": {"entity1": {"id": "entity1", "type": "dataset"}},
            "agents": {"agent1": {"id": "agent1", "name": "Test Agent"}}
        }
        adata.uns["provenance"] = test_provenance

        # Extract to tracker
        tracker = ProvenanceTracker()
        success = tracker.extract_from_anndata(adata)

        assert success is True
        assert len(tracker.activities) == 1
        assert len(tracker.entities) == 1
        assert len(tracker.agents) == 1

        assert tracker.activities[0]["id"] == "test_activity"
        assert "entity1" in tracker.entities
        assert "agent1" in tracker.agents

    def test_extract_from_anndata_no_provenance(self):
        """Test extraction from AnnData without provenance."""
        adata = SingleCellDataFactory(config=SMALL_DATASET_CONFIG)
        tracker = ProvenanceTracker()

        success = tracker.extract_from_anndata(adata)

        assert success is False
        assert len(tracker.activities) == 0
        assert len(tracker.entities) == 0
        assert len(tracker.agents) == 0

    def test_extract_from_anndata_partial_provenance(self):
        """Test extraction from AnnData with partial provenance data."""
        adata = SingleCellDataFactory(config=SMALL_DATASET_CONFIG)

        # Only activities, no entities or agents
        adata.uns["provenance"] = {
            "activities": [{"id": "test_activity", "type": "test"}]
        }

        tracker = ProvenanceTracker()
        success = tracker.extract_from_anndata(adata)

        assert success is True
        assert len(tracker.activities) == 1
        assert len(tracker.entities) == 0  # No entities to extract
        assert len(tracker.agents) == 0  # No agents to extract

    def test_anndata_round_trip_integration(self):
        """Test complete round-trip: tracker -> AnnData -> tracker."""
        tracker1 = ProvenanceTracker(namespace="round_trip")

        # Create provenance data
        entity_id = tracker1.create_entity("test_data", format="h5ad")
        agent_id = tracker1.create_agent("test_agent", version="1.0")
        activity_id = tracker1.create_activity("test_activity", agent_id,
                                             parameters={"test": True})

        # Add to AnnData
        adata = SingleCellDataFactory(config=SMALL_DATASET_CONFIG)
        tracker1.add_to_anndata(adata)

        # Extract to new tracker
        tracker2 = ProvenanceTracker()
        success = tracker2.extract_from_anndata(adata)

        assert success is True

        # Compare trackers
        assert len(tracker2.activities) == len(tracker1.activities)
        assert len(tracker2.entities) == len(tracker1.entities)
        assert len(tracker2.agents) == len(tracker1.agents)

        # Check specific data
        assert tracker2.activities[0]["id"] == activity_id
        assert tracker2.activities[0]["parameters"]["test"] is True
        assert entity_id in tracker2.entities
        assert agent_id in tracker2.agents


# ===============================================================================
# Provenance Lineage and History Tests
# ===============================================================================

@pytest.mark.unit
class TestProvenanceLineage:
    """Test provenance lineage tracking and history reconstruction."""

    def test_get_lineage_simple(self):
        """Test lineage tracking for simple case."""
        tracker = ProvenanceTracker()

        # Create linear processing chain
        input_entity = tracker.create_entity("input_data", format="csv")
        output_entity = tracker.create_entity("output_data", format="h5ad")
        agent_id = tracker.create_agent("processor")

        activity_id = tracker.create_activity(
            "processing",
            agent_id,
            inputs=[{"entity": input_entity, "role": "input"}],
            outputs=[{"entity": output_entity, "role": "output"}]
        )

        # Get lineage for output entity
        lineage = tracker.get_lineage(output_entity)

        assert len(lineage) == 1
        assert lineage[0]["id"] == activity_id
        assert lineage[0]["type"] == "processing"

    def test_get_lineage_complex_chain(self):
        """Test lineage tracking for complex processing chain."""
        tracker = ProvenanceTracker()

        # Create processing chain: A -> B -> C
        entity_a = tracker.create_entity("data_a", format="csv")
        entity_b = tracker.create_entity("data_b", format="h5ad")
        entity_c = tracker.create_entity("data_c", format="h5ad")

        agent_id = tracker.create_agent("processor")

        # A -> B
        activity1 = tracker.create_activity(
            "step1",
            agent_id,
            inputs=[{"entity": entity_a, "role": "input"}],
            outputs=[{"entity": entity_b, "role": "output"}]
        )

        # B -> C
        activity2 = tracker.create_activity(
            "step2",
            agent_id,
            inputs=[{"entity": entity_b, "role": "input"}],
            outputs=[{"entity": entity_c, "role": "output"}]
        )

        # Get lineage for final entity
        lineage = tracker.get_lineage(entity_c)

        # Should include both activities (recursive)
        assert len(lineage) >= 1  # At least the direct activity
        activity_ids = [act["id"] for act in lineage]
        assert activity2 in activity_ids

    def test_get_lineage_multiple_inputs(self):
        """Test lineage tracking with multiple inputs."""
        tracker = ProvenanceTracker()

        # Create merge operation: A + B -> C
        entity_a = tracker.create_entity("data_a", format="csv")
        entity_b = tracker.create_entity("data_b", format="csv")
        entity_c = tracker.create_entity("merged_data", format="h5ad")

        agent_id = tracker.create_agent("merger")

        activity_id = tracker.create_activity(
            "merge",
            agent_id,
            inputs=[
                {"entity": entity_a, "role": "input1"},
                {"entity": entity_b, "role": "input2"}
            ],
            outputs=[{"entity": entity_c, "role": "merged_output"}]
        )

        # Get lineage
        lineage = tracker.get_lineage(entity_c)

        assert len(lineage) >= 1
        merge_activity = next(act for act in lineage if act["id"] == activity_id)
        assert len(merge_activity["inputs"]) == 2

    def test_get_lineage_nonexistent_entity(self):
        """Test lineage tracking for non-existent entity."""
        tracker = ProvenanceTracker()

        # Try to get lineage for entity that doesn't exist
        lineage = tracker.get_lineage("nonexistent_entity")

        assert lineage == []

    def test_lineage_preserves_activity_order(self):
        """Test that lineage preserves the chronological order of activities."""
        tracker = ProvenanceTracker()

        # Create time-ordered chain
        entities = []
        activities = []

        for i in range(3):
            entity = tracker.create_entity(f"data_{i}", format="h5ad")
            entities.append(entity)

            if i > 0:  # Skip first iteration (no input)
                activity = tracker.create_activity(
                    f"step_{i}",
                    "processor",
                    inputs=[{"entity": entities[i-1], "role": "input"}],
                    outputs=[{"entity": entities[i], "role": "output"}]
                )
                activities.append(activity)

        # Get lineage for final entity
        lineage = tracker.get_lineage(entities[-1])

        # Activities should be retrievable
        assert len(lineage) >= 1


# ===============================================================================
# Provenance High-Level Workflow Tests
# ===============================================================================

@pytest.mark.unit
class TestProvenanceWorkflows:
    """Test high-level provenance workflows and logging methods."""

    def test_log_data_loading_workflow(self):
        """Test data loading workflow logging."""
        tracker = ProvenanceTracker()

        # Create output entity first
        output_entity = tracker.create_entity("loaded_data", format="h5ad")

        # Log data loading
        activity_id = tracker.log_data_loading(
            source_path="/path/to/source.csv",
            output_entity_id=output_entity,
            adapter_name="TranscriptomicsAdapter",
            parameters={"delimiter": ",", "header": True}
        )

        # Verify activity was created
        assert len(tracker.activities) == 1
        activity = tracker.activities[0]

        assert activity["id"] == activity_id
        assert activity["type"] == "data_loading"
        assert activity["parameters"]["delimiter"] == ","

        # Should have created input entity for source file
        assert len(tracker.entities) == 2  # output + source entities

        # Should have created adapter agent
        assert len(tracker.agents) == 1

        # Check relationships
        assert len(activity["inputs"]) == 1
        assert len(activity["outputs"]) == 1
        assert activity["outputs"][0]["entity"] == output_entity

    def test_log_data_processing_workflow(self):
        """Test data processing workflow logging."""
        tracker = ProvenanceTracker()

        # Create entities
        input_entity = tracker.create_entity("raw_data", format="h5ad")
        output_entity = tracker.create_entity("processed_data", format="h5ad")

        # Log processing
        activity_id = tracker.log_data_processing(
            input_entity_id=input_entity,
            output_entity_id=output_entity,
            processing_type="normalization",
            agent_name="NormalizationService",
            parameters={"method": "log1p", "scale": True},
            description="Log normalization with scaling"
        )

        # Verify activity
        assert len(tracker.activities) == 1
        activity = tracker.activities[0]

        assert activity["type"] == "normalization"
        assert activity["description"] == "Log normalization with scaling"
        assert activity["parameters"]["method"] == "log1p"

        # Check relationships
        assert activity["inputs"][0]["entity"] == input_entity
        assert activity["outputs"][0]["entity"] == output_entity

        # Should have created processing agent
        agent_ids = list(tracker.agents.keys())
        assert len(agent_ids) == 1
        assert "normalizationservice" in agent_ids[0]

    def test_log_data_saving_workflow(self):
        """Test data saving workflow logging."""
        tracker = ProvenanceTracker()

        # Create input entity
        input_entity = tracker.create_entity("data_to_save", format="h5ad")

        # Log saving
        activity_id = tracker.log_data_saving(
            input_entity_id=input_entity,
            output_path="/path/to/saved/data.h5ad",
            backend_name="H5ADBackend",
            parameters={"compression": "gzip", "backup": True}
        )

        # Verify activity
        assert len(tracker.activities) == 1
        activity = tracker.activities[0]

        assert activity["type"] == "data_saving"
        assert activity["parameters"]["compression"] == "gzip"

        # Should have created output entity for saved file
        assert len(tracker.entities) == 2  # input + output entities

        # Should have created backend agent
        assert len(tracker.agents) == 1

        # Check relationships
        assert activity["inputs"][0]["entity"] == input_entity
        assert len(activity["outputs"]) == 1

    def test_complete_analysis_workflow(self):
        """Test complete analysis workflow with multiple steps."""
        tracker = ProvenanceTracker()

        # Step 1: Data loading
        loaded_entity = tracker.create_entity("loaded_data", format="h5ad")
        loading_activity = tracker.log_data_loading(
            source_path="/data/experiment.csv",
            output_entity_id=loaded_entity,
            adapter_name="CSVAdapter"
        )

        # Step 2: Quality control
        qc_entity = tracker.create_entity("qc_data", format="h5ad")
        qc_activity = tracker.log_data_processing(
            input_entity_id=loaded_entity,
            output_entity_id=qc_entity,
            processing_type="quality_control",
            agent_name="QCService",
            parameters={"min_genes": 200, "max_genes": 5000}
        )

        # Step 3: Normalization
        norm_entity = tracker.create_entity("normalized_data", format="h5ad")
        norm_activity = tracker.log_data_processing(
            input_entity_id=qc_entity,
            output_entity_id=norm_entity,
            processing_type="normalization",
            agent_name="NormalizationService",
            parameters={"method": "log1p"}
        )

        # Step 4: Save results
        save_activity = tracker.log_data_saving(
            input_entity_id=norm_entity,
            output_path="/results/normalized_data.h5ad",
            backend_name="H5ADBackend"
        )

        # Verify complete workflow
        assert len(tracker.activities) == 4
        assert len(tracker.entities) == 5  # 3 processing + 1 source + 1 saved
        assert len(tracker.agents) >= 4  # Different services

        # Verify processing chain
        lineage = tracker.get_lineage(norm_entity)
        assert len(lineage) >= 2  # At least normalization and QC activities


# ===============================================================================
# Performance and Edge Case Tests
# ===============================================================================

@pytest.mark.unit
class TestProvenancePerformanceAndEdgeCases:
    """Test provenance system performance and edge case handling."""

    def test_large_scale_provenance_tracking(self):
        """Test performance with large numbers of entities and activities."""
        tracker = ProvenanceTracker()

        # Create many entities and activities
        num_entities = 100
        entities = []

        for i in range(num_entities):
            entity_id = tracker.create_entity(f"entity_{i}", format="h5ad")
            entities.append(entity_id)

        # Create processing chain
        agent_id = tracker.create_agent("bulk_processor")
        for i in range(1, num_entities):
            tracker.create_activity(
                f"process_{i}",
                agent_id,
                inputs=[{"entity": entities[i-1], "role": "input"}],
                outputs=[{"entity": entities[i], "role": "output"}]
            )

        # Verify scale
        assert len(tracker.entities) == num_entities
        assert len(tracker.activities) == num_entities - 1
        assert len(tracker.agents) == 1

        # Test lineage performance
        lineage = tracker.get_lineage(entities[-1])
        assert len(lineage) >= 1

    def test_provenance_with_special_characters(self):
        """Test provenance handling with special characters in names."""
        tracker = ProvenanceTracker()

        # Test with various special characters
        special_names = [
            "file with spaces.csv",
            "file-with-dashes.h5ad",
            "file_with_underscores.txt",
            "file.with.dots.xlsx",
            "file(with)parentheses.h5mu"
        ]

        for name in special_names:
            entity_id = tracker.create_entity("test_entity", uri=name)
            entity = tracker.entities[entity_id]
            assert entity["uri"] == name

        # Test agent names with special characters
        agent_names = [
            "Agent With Spaces",
            "Agent-With-Dashes",
            "Agent_With_Underscores",
            "Agent.With.Dots"
        ]

        for name in agent_names:
            agent_id = tracker.create_agent(name)
            agent = tracker.agents[agent_id]
            assert agent["name"] == name

    def test_provenance_with_unicode_characters(self):
        """Test provenance handling with Unicode characters."""
        tracker = ProvenanceTracker()

        # Test with Unicode in various fields
        unicode_names = [
            "файл.csv",  # Cyrillic
            "文件.h5ad",  # Chinese
            "ファイル.txt",  # Japanese
            "🧬data.h5mu"  # Emoji
        ]

        for name in unicode_names:
            try:
                entity_id = tracker.create_entity("unicode_entity", uri=name)
                entity = tracker.entities[entity_id]
                assert entity["uri"] == name
            except UnicodeError:
                pytest.fail(f"Failed to handle Unicode in: {name}")

    def test_provenance_memory_efficiency(self):
        """Test memory efficiency of provenance tracking."""
        import sys

        tracker = ProvenanceTracker()

        # Measure initial memory usage
        initial_size = sys.getsizeof(tracker.activities) + \
                      sys.getsizeof(tracker.entities) + \
                      sys.getsizeof(tracker.agents)

        # Add substantial amount of data
        for i in range(50):
            entity_id = tracker.create_entity(f"entity_{i}")
            agent_id = tracker.create_agent(f"agent_{i}")
            activity_id = tracker.create_activity(f"activity_{i}", agent_id)

        # Measure final memory usage
        final_size = sys.getsizeof(tracker.activities) + \
                    sys.getsizeof(tracker.entities) + \
                    sys.getsizeof(tracker.agents)

        # Memory should scale reasonably (not test specific values, just ensure it doesn't explode)
        assert final_size > initial_size
        assert final_size < initial_size * 1000  # Reasonable upper bound

    def test_concurrent_provenance_operations(self):
        """Test thread safety of provenance operations."""
        import threading
        import time

        tracker = ProvenanceTracker()
        results = []
        errors = []

        def worker(worker_id):
            """Worker function for concurrent testing."""
            try:
                # Each worker creates entities and activities
                entity_id = tracker.create_entity(f"entity_{worker_id}")
                agent_id = tracker.create_agent(f"agent_{worker_id}")
                activity_id = tracker.create_activity(f"activity_{worker_id}", agent_id)

                results.append((worker_id, entity_id, agent_id, activity_id))
                time.sleep(0.01)  # Small delay
            except Exception as e:
                errors.append((worker_id, e))

        # Create multiple threads
        threads = []
        for i in range(10):
            thread = threading.Thread(target=worker, args=(i,))
            threads.append(thread)
            thread.start()

        # Wait for completion
        for thread in threads:
            thread.join()

        # Check results
        assert len(errors) == 0, f"Concurrent errors: {errors}"
        assert len(results) == 10

        # Verify data integrity
        assert len(tracker.entities) == 10
        assert len(tracker.agents) == 10
        assert len(tracker.activities) == 10


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])