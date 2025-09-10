"""
Mock data generation utilities for Lobster AI testing framework.

This module provides comprehensive synthetic biological data generation
that maintains statistical properties of real datasets while being 
completely reproducible for testing purposes.
"""

from .factories import (
    SingleCellDataFactory,
    BulkRNASeqDataFactory,
    ProteomicsDataFactory,
    MultiModalDataFactory
)

from .generators import (
    generate_synthetic_single_cell,
    generate_synthetic_bulk_rnaseq,
    generate_synthetic_proteomics,
    generate_mock_geo_response,
    generate_test_workspace_state
)

from .base import MockDataConfig, TestDataRegistry

__all__ = [
    # Factories
    "SingleCellDataFactory",
    "BulkRNASeqDataFactory", 
    "ProteomicsDataFactory",
    "MultiModalDataFactory",
    
    # Generators
    "generate_synthetic_single_cell",
    "generate_synthetic_bulk_rnaseq", 
    "generate_synthetic_proteomics",
    "generate_mock_geo_response",
    "generate_test_workspace_state",
    
    # Configuration
    "MockDataConfig",
    "TestDataRegistry"
]