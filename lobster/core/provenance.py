"""
Provenance tracking utilities for the modular DataManager architecture.

This module provides W3C-PROV-like provenance tracking for complete
reproducibility and audit trail of data processing operations.
"""

import datetime
import hashlib
import logging
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import anndata

logger = logging.getLogger(__name__)


class ProvenanceTracker:
    """
    W3C-PROV-like provenance tracking system.
    
    This class tracks data processing activities, entities, and agents
    to provide a complete audit trail and enable reproducibility.
    """

    def __init__(self, namespace: str = "lobster"):
        """
        Initialize the provenance tracker.

        Args:
            namespace: Namespace for provenance identifiers
        """
        self.namespace = namespace
        self.activities: List[Dict[str, Any]] = []
        self.entities: Dict[str, Dict[str, Any]] = {}
        self.agents: Dict[str, Dict[str, Any]] = {}
        self.logger = logger

    def create_activity(
        self,
        activity_type: str,
        agent: str,
        inputs: Optional[List[Dict[str, Any]]] = None,
        outputs: Optional[List[Dict[str, Any]]] = None,
        parameters: Optional[Dict[str, Any]] = None,
        description: Optional[str] = None
    ) -> str:
        """
        Create a new provenance activity record.

        Args:
            activity_type: Type of activity (e.g., 'data_loading', 'normalization')
            agent: Agent performing the activity (e.g., 'TranscriptomicsAdapter')
            inputs: List of input entities
            outputs: List of output entities
            parameters: Parameters used in the activity
            description: Human-readable description

        Returns:
            str: Unique activity ID
        """
        activity_id = f"{self.namespace}:activity:{uuid.uuid4()}"
        timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()

        activity = {
            "id": activity_id,
            "type": activity_type,
            "agent": agent,
            "timestamp": timestamp,
            "inputs": inputs or [],
            "outputs": outputs or [],
            "parameters": parameters or {},
            "description": description,
            "software_versions": self._get_software_versions()
        }

        self.activities.append(activity)
        self.logger.debug(f"Created activity: {activity_id} ({activity_type})")
        
        return activity_id

    def create_entity(
        self,
        entity_type: str,
        uri: Union[str, Path] = None,
        checksum: Optional[str] = None,
        format: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Create a new provenance entity record.

        Args:
            entity_type: Type of entity (e.g., 'dataset', 'plot', 'result')
            uri: URI or path to the entity
            checksum: Optional checksum for integrity verification
            format: File format or data type
            metadata: Additional metadata

        Returns:
            str: Unique entity ID
        """
        entity_id = f"{self.namespace}:entity:{uuid.uuid4()}"
        
        # Calculate checksum if not provided and entity is a file
        if checksum is None and isinstance(uri, (str, Path)):
            checksum = self._calculate_checksum(uri)

        entity = {
            "id": entity_id,
            "type": entity_type,
            "uri": str(uri) if uri else None,
            "checksum": checksum,
            "format": format,
            "metadata": metadata or {},
            "created": datetime.datetime.now(datetime.timezone.utc).isoformat()
        }

        self.entities[entity_id] = entity
        self.logger.debug(f"Created entity: {entity_id} ({entity_type})")
        
        return entity_id

    def create_agent(
        self,
        name: str,
        agent_type: str = "software",
        version: Optional[str] = None,
        description: Optional[str] = None
    ) -> str:
        """
        Create a new provenance agent record.

        Args:
            name: Name of the agent
            agent_type: Type of agent ('software', 'person', 'organization')
            version: Version of the agent
            description: Description of the agent

        Returns:
            str: Unique agent ID
        """
        agent_id = f"{self.namespace}:agent:{name.replace(' ', '_').lower()}"

        if agent_id not in self.agents:
            agent = {
                "id": agent_id,
                "name": name,
                "type": agent_type,
                "version": version,
                "description": description
            }
            self.agents[agent_id] = agent
            self.logger.debug(f"Created agent: {agent_id}")

        return agent_id

    def log_data_loading(
        self,
        source_path: Union[str, Path],
        output_entity_id: str,
        adapter_name: str,
        parameters: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Log a data loading activity.

        Args:
            source_path: Path to source data file
            output_entity_id: ID of the loaded data entity
            adapter_name: Name of the adapter used
            parameters: Loading parameters

        Returns:
            str: Activity ID
        """
        # Create input entity for source file
        input_entity_id = self.create_entity(
            entity_type="source_file",
            uri=source_path,
            format=self._detect_format(source_path)
        )

        # Create agent for adapter
        agent_id = self.create_agent(
            name=adapter_name,
            agent_type="software",
            description="Data adapter for loading biological data"
        )

        # Create loading activity
        activity_id = self.create_activity(
            activity_type="data_loading",
            agent=agent_id,
            inputs=[{"entity": input_entity_id, "role": "source"}],
            outputs=[{"entity": output_entity_id, "role": "loaded_data"}],
            parameters=parameters,
            description=f"Loaded data from {source_path} using {adapter_name}"
        )

        return activity_id

    def log_data_processing(
        self,
        input_entity_id: str,
        output_entity_id: str,
        processing_type: str,
        agent_name: str,
        parameters: Optional[Dict[str, Any]] = None,
        description: Optional[str] = None
    ) -> str:
        """
        Log a data processing activity.

        Args:
            input_entity_id: ID of input data entity
            output_entity_id: ID of output data entity
            processing_type: Type of processing (e.g., 'normalization', 'filtering')
            agent_name: Name of the processing agent
            parameters: Processing parameters
            description: Description of the processing step

        Returns:
            str: Activity ID
        """
        # Create agent
        agent_id = self.create_agent(
            name=agent_name,
            agent_type="software",
            description="Data processing agent"
        )

        # Create processing activity
        activity_id = self.create_activity(
            activity_type=processing_type,
            agent=agent_id,
            inputs=[{"entity": input_entity_id, "role": "input_data"}],
            outputs=[{"entity": output_entity_id, "role": "processed_data"}],
            parameters=parameters,
            description=description or f"Applied {processing_type} to data"
        )

        return activity_id

    def log_data_saving(
        self,
        input_entity_id: str,
        output_path: Union[str, Path],
        backend_name: str,
        parameters: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Log a data saving activity.

        Args:
            input_entity_id: ID of input data entity
            output_path: Path where data was saved
            backend_name: Name of the storage backend
            parameters: Saving parameters

        Returns:
            str: Activity ID
        """
        # Create output entity for saved file
        output_entity_id = self.create_entity(
            entity_type="saved_file",
            uri=output_path,
            format=self._detect_format(output_path)
        )

        # Create agent for backend
        agent_id = self.create_agent(
            name=backend_name,
            agent_type="software",
            description="Data storage backend"
        )

        # Create saving activity
        activity_id = self.create_activity(
            activity_type="data_saving",
            agent=agent_id,
            inputs=[{"entity": input_entity_id, "role": "data_to_save"}],
            outputs=[{"entity": output_entity_id, "role": "saved_file"}],
            parameters=parameters,
            description=f"Saved data to {output_path} using {backend_name}"
        )

        return activity_id

    def add_to_anndata(self, adata: anndata.AnnData) -> anndata.AnnData:
        """
        Add provenance information to AnnData object.

        Args:
            adata: AnnData object to annotate

        Returns:
            anndata.AnnData: AnnData with provenance information
        """
        if "provenance" not in adata.uns:
            adata.uns["provenance"] = {}

        adata.uns["provenance"]["activities"] = self.activities.copy()
        adata.uns["provenance"]["entities"] = self.entities.copy()
        adata.uns["provenance"]["agents"] = self.agents.copy()
        adata.uns["provenance"]["tracker_namespace"] = self.namespace

        return adata

    def extract_from_anndata(self, adata: anndata.AnnData) -> bool:
        """
        Extract provenance information from AnnData object.

        Args:
            adata: AnnData object containing provenance

        Returns:
            bool: True if provenance was found and extracted
        """
        if "provenance" not in adata.uns:
            return False

        prov_data = adata.uns["provenance"]
        
        if "activities" in prov_data:
            self.activities.extend(prov_data["activities"])
        
        if "entities" in prov_data:
            self.entities.update(prov_data["entities"])
        
        if "agents" in prov_data:
            self.agents.update(prov_data["agents"])

        return True

    def get_lineage(self, entity_id: str) -> List[Dict[str, Any]]:
        """
        Get the complete lineage of an entity.

        Args:
            entity_id: ID of the entity to trace

        Returns:
            List[Dict[str, Any]]: List of activities in the lineage
        """
        lineage = []
        
        # Find activities that produced this entity
        for activity in self.activities:
            for output in activity.get("outputs", []):
                if output.get("entity") == entity_id:
                    lineage.append(activity)
                    # Recursively find activities that produced the inputs
                    for input_ref in activity.get("inputs", []):
                        input_entity_id = input_ref.get("entity")
                        if input_entity_id:
                            parent_lineage = self.get_lineage(input_entity_id)
                            lineage.extend(parent_lineage)
                    break
        
        return lineage

    def to_dict(self) -> Dict[str, Any]:
        """
        Export provenance data as dictionary.

        Returns:
            Dict[str, Any]: Complete provenance data
        """
        return {
            "namespace": self.namespace,
            "activities": self.activities,
            "entities": self.entities,
            "agents": self.agents,
            "export_timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat()
        }

    def from_dict(self, data: Dict[str, Any]) -> None:
        """
        Import provenance data from dictionary.

        Args:
            data: Provenance data dictionary
        """
        self.namespace = data.get("namespace", self.namespace)
        self.activities = data.get("activities", [])
        self.entities = data.get("entities", {})
        self.agents = data.get("agents", {})

    def _calculate_checksum(self, path: Union[str, Path]) -> Optional[str]:
        """Calculate SHA256 checksum of a file."""
        try:
            path = Path(path)
            if not path.exists():
                return None
            
            sha256_hash = hashlib.sha256()
            with open(path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    sha256_hash.update(chunk)
            
            return sha256_hash.hexdigest()
        except Exception as e:
            self.logger.warning(f"Failed to calculate checksum for {path}: {e}")
            return None

    def _detect_format(self, path: Union[str, Path]) -> str:
        """Detect file format from extension."""
        path = Path(path)
        extension = path.suffix.lower()
        
        format_mapping = {
            '.h5ad': 'h5ad',
            '.h5': 'h5',
            '.csv': 'csv',
            '.tsv': 'tsv',
            '.txt': 'txt',
            '.xlsx': 'excel',
            '.xls': 'excel',
            '.h5mu': 'h5mu',
            '.png': 'png',
            '.pdf': 'pdf',
            '.svg': 'svg'
        }
        
        return format_mapping.get(extension, 'unknown')

    def _get_software_versions(self) -> Dict[str, str]:
        """Get versions of key software packages."""
        versions = {}
        
        try:
            import scanpy
            versions["scanpy"] = scanpy.__version__
        except ImportError:
            pass
        
        try:
            import anndata
            versions["anndata"] = anndata.__version__
        except ImportError:
            pass
        
        try:
            import pandas
            versions["pandas"] = pandas.__version__
        except ImportError:
            pass
        
        try:
            import numpy
            versions["numpy"] = numpy.__version__
        except ImportError:
            pass
        
        try:
            # Try to get lobster version
            from lobster.version import __version__
            versions["lobster"] = __version__
        except ImportError:
            versions["lobster"] = "unknown"
        
        return versions
