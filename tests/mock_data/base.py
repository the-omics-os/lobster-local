"""
Base classes and configuration for mock data generation.

This module provides the foundational classes and configuration
for generating synthetic biological datasets for testing.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
import json


@dataclass
class MockDataConfig:
    """Configuration for synthetic biological data generation."""
    
    # Random seed for reproducibility
    seed: int = 42
    
    # Default dataset sizes
    default_cell_count: int = 1000
    default_gene_count: int = 2000
    default_sample_count: int = 24
    default_protein_count: int = 500
    
    # Data characteristics
    sparsity_level: float = 0.7  # Fraction of zeros in single-cell data
    noise_level: float = 0.1     # Amount of random noise to add
    
    # Cell type simulation
    n_cell_types: int = 5
    cell_type_proportions: Optional[List[float]] = None
    
    # Gene expression parameters
    mean_expression: float = 2.0
    expression_variance: float = 1.5
    
    # Quality control parameters
    mt_gene_fraction_range: Tuple[float, float] = (0.05, 0.25)
    ribo_gene_fraction_range: Tuple[float, float] = (0.10, 0.40)
    
    # Batch effect simulation  
    batch_effects: bool = True
    n_batches: int = 3
    batch_effect_strength: float = 0.3
    
    # Missing data simulation (for proteomics)
    missing_data_rate: float = 0.2
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            field.name: getattr(self, field.name) 
            for field in self.__dataclass_fields__.values()
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "MockDataConfig":
        """Create configuration from dictionary."""
        return cls(**config_dict)
    
    def save(self, path: Path) -> None:
        """Save configuration to JSON file."""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, path: Path) -> "MockDataConfig":
        """Load configuration from JSON file."""
        with open(path, 'r') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)


class TestDataRegistry:
    """Registry for managing test datasets and their metadata."""
    
    def __init__(self):
        self._datasets: Dict[str, Dict[str, Any]] = {}
        self._metadata: Dict[str, MockDataConfig] = {}
    
    def register_dataset(self, name: str, dataset: Any, config: MockDataConfig) -> None:
        """Register a test dataset with its configuration."""
        self._datasets[name] = dataset
        self._metadata[name] = config
    
    def get_dataset(self, name: str) -> Any:
        """Retrieve a registered dataset."""
        if name not in self._datasets:
            raise KeyError(f"Dataset '{name}' not found in registry")
        return self._datasets[name]
    
    def get_metadata(self, name: str) -> MockDataConfig:
        """Retrieve metadata for a registered dataset.""" 
        if name not in self._metadata:
            raise KeyError(f"Metadata for dataset '{name}' not found")
        return self._metadata[name]
    
    def list_datasets(self) -> List[str]:
        """List all registered datasets."""
        return list(self._datasets.keys())
    
    def remove_dataset(self, name: str) -> None:
        """Remove a dataset from the registry."""
        self._datasets.pop(name, None)
        self._metadata.pop(name, None)
    
    def clear(self) -> None:
        """Clear all datasets from the registry."""
        self._datasets.clear()
        self._metadata.clear()
    
    def to_json(self) -> str:
        """Export registry metadata to JSON."""
        export_data = {
            name: config.to_dict() 
            for name, config in self._metadata.items()
        }
        return json.dumps(export_data, indent=2)
    
    def save_registry(self, path: Path) -> None:
        """Save registry metadata to file."""
        with open(path, 'w') as f:
            f.write(self.to_json())


# Global test data registry instance
test_data_registry = TestDataRegistry()


# Predefined configurations for common test scenarios
SMALL_DATASET_CONFIG = MockDataConfig(
    seed=42,
    default_cell_count=100,
    default_gene_count=500,
    default_sample_count=6,
    default_protein_count=100,
    n_cell_types=3,
    n_batches=2
)

MEDIUM_DATASET_CONFIG = MockDataConfig(
    seed=42,
    default_cell_count=1000,
    default_gene_count=2000,
    default_sample_count=24,
    default_protein_count=500,
    n_cell_types=5,
    n_batches=3
)

LARGE_DATASET_CONFIG = MockDataConfig(
    seed=42,
    default_cell_count=10000,
    default_gene_count=5000,
    default_sample_count=48,
    default_protein_count=1000,
    n_cell_types=10,
    n_batches=4
)

# Configuration for datasets with specific characteristics
HIGH_NOISE_CONFIG = MockDataConfig(
    seed=42,
    noise_level=0.5,
    sparsity_level=0.8,
    batch_effect_strength=0.5
)

LOW_QUALITY_CONFIG = MockDataConfig(
    seed=42,
    mt_gene_fraction_range=(0.20, 0.50),  # High mitochondrial content
    missing_data_rate=0.5,  # High missing data rate
    noise_level=0.3
)

BATCH_EFFECT_CONFIG = MockDataConfig(
    seed=42,
    batch_effects=True,
    n_batches=5,
    batch_effect_strength=0.8
)