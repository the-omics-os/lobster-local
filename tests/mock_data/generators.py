"""
High-level generator functions for creating synthetic biological datasets.

This module provides convenient functions for generating test data
without needing to work with factories directly.
"""

import numpy as np
import pandas as pd
import anndata as ad
from typing import Optional, Dict, Any, List, Union
from pathlib import Path
import json
from datetime import datetime, timedelta
from faker import Faker

from .base import MockDataConfig, MEDIUM_DATASET_CONFIG
from .factories import (
    SingleCellDataFactory,
    BulkRNASeqDataFactory, 
    ProteomicsDataFactory,
    MultiModalDataFactory
)

fake = Faker()
Faker.seed(42)


def generate_synthetic_single_cell(
    n_cells: Optional[int] = None,
    n_genes: Optional[int] = None,
    n_cell_types: Optional[int] = None,
    config: Optional[MockDataConfig] = None,
    **kwargs
) -> ad.AnnData:
    """
    Generate synthetic single-cell RNA-seq data.
    
    Args:
        n_cells: Number of cells to generate
        n_genes: Number of genes to generate
        n_cell_types: Number of cell types to simulate
        config: Configuration object for data generation
        **kwargs: Additional parameters passed to factory
    
    Returns:
        AnnData object with synthetic single-cell data
    """
    if config is None:
        config = MEDIUM_DATASET_CONFIG
    
    # Override config values if provided
    if n_cells is not None:
        config.default_cell_count = n_cells
    if n_genes is not None:
        config.default_gene_count = n_genes
    if n_cell_types is not None:
        config.n_cell_types = n_cell_types
    
    return SingleCellDataFactory._create(ad.AnnData, config=config, **kwargs)


def generate_synthetic_bulk_rnaseq(
    n_samples: Optional[int] = None,
    n_genes: Optional[int] = None,
    config: Optional[MockDataConfig] = None,
    **kwargs
) -> ad.AnnData:
    """
    Generate synthetic bulk RNA-seq data.
    
    Args:
        n_samples: Number of samples to generate
        n_genes: Number of genes to generate
        config: Configuration object for data generation
        **kwargs: Additional parameters passed to factory
    
    Returns:
        AnnData object with synthetic bulk RNA-seq data
    """
    if config is None:
        config = MEDIUM_DATASET_CONFIG
    
    if n_samples is not None:
        config.default_sample_count = n_samples
    if n_genes is not None:
        config.default_gene_count = n_genes
    
    return BulkRNASeqDataFactory._create(ad.AnnData, config=config, **kwargs)


def generate_synthetic_proteomics(
    n_samples: Optional[int] = None,
    n_proteins: Optional[int] = None,
    missing_rate: Optional[float] = None,
    config: Optional[MockDataConfig] = None,
    **kwargs
) -> ad.AnnData:
    """
    Generate synthetic proteomics data.
    
    Args:
        n_samples: Number of samples to generate
        n_proteins: Number of proteins to generate
        missing_rate: Fraction of missing values
        config: Configuration object for data generation
        **kwargs: Additional parameters passed to factory
    
    Returns:
        AnnData object with synthetic proteomics data
    """
    if config is None:
        config = MEDIUM_DATASET_CONFIG
    
    if n_samples is not None:
        config.default_sample_count = n_samples
    if n_proteins is not None:
        config.default_protein_count = n_proteins
    if missing_rate is not None:
        config.missing_data_rate = missing_rate
    
    return ProteomicsDataFactory._create(ad.AnnData, config=config, **kwargs)


def generate_multimodal_data(
    n_cells: Optional[int] = None,
    n_genes: Optional[int] = None,
    n_proteins: Optional[int] = None,
    config: Optional[MockDataConfig] = None,
    **kwargs
) -> Dict[str, ad.AnnData]:
    """
    Generate synthetic multi-modal data (RNA + Protein).
    
    Args:
        n_cells: Number of cells to generate
        n_genes: Number of genes to generate
        n_proteins: Number of proteins to generate
        config: Configuration object for data generation
        **kwargs: Additional parameters passed to factory
    
    Returns:
        Dictionary with 'rna' and 'protein' AnnData objects
    """
    if config is None:
        config = MEDIUM_DATASET_CONFIG
    
    if n_cells is not None:
        config.default_cell_count = n_cells
    if n_genes is not None:
        config.default_gene_count = n_genes
    if n_proteins is not None:
        config.default_protein_count = n_proteins
    
    return MultiModalDataFactory._create(dict, config=config, **kwargs)


def generate_mock_geo_response(
    gse_id: str = "GSE123456",
    organism: str = "Homo sapiens",
    n_samples: int = 12,
    data_type: str = "single_cell",
    include_files: bool = True
) -> Dict[str, Any]:
    """
    Generate mock GEO dataset response for testing API interactions.
    
    Args:
        gse_id: GEO Series ID
        organism: Species name
        n_samples: Number of samples in the dataset
        data_type: Type of data (single_cell, bulk_rnaseq, proteomics)
        include_files: Whether to include file listings
    
    Returns:
        Dictionary mimicking GEO API response
    """
    # Generate realistic titles and descriptions
    titles = {
        "single_cell": "Single-cell RNA sequencing of {} tissue",
        "bulk_rnaseq": "Bulk RNA-seq analysis of {} under {} treatment", 
        "proteomics": "Quantitative proteomics profiling of {} samples"
    }
    
    tissues = ["brain", "liver", "kidney", "heart", "lung", "muscle", "skin"]
    treatments = ["drug treatment", "knockout", "overexpression", "stress conditions"]
    
    tissue = fake.random_element(tissues)
    treatment = fake.random_element(treatments)
    
    if data_type == "single_cell":
        title = titles[data_type].format(tissue)
    elif data_type == "bulk_rnaseq":
        title = titles[data_type].format(tissue, treatment)
    else:
        title = titles[data_type].format(tissue)
    
    # Generate samples
    samples = []
    for i in range(n_samples):
        sample = {
            "gsm_id": f"GSM{123456 + i}",
            "title": f"Sample {i+1}",
            "characteristics": {
                "tissue": tissue,
                "cell_type": fake.random_element(["T_cell", "B_cell", "Monocyte", "NK_cell"]),
                "treatment": fake.random_element(["Control", "Treatment"]),
                "time_point": f"{fake.random_int(0, 72)}h",
                "batch": f"Batch_{fake.random_int(1, 3)}"
            },
            "supplementary_files": []
        }
        
        if include_files:
            if data_type == "single_cell":
                sample["supplementary_files"] = [
                    f"{sample['gsm_id']}_matrix.mtx.gz",
                    f"{sample['gsm_id']}_features.tsv.gz",
                    f"{sample['gsm_id']}_barcodes.tsv.gz"
                ]
            elif data_type == "bulk_rnaseq":
                sample["supplementary_files"] = [
                    f"{sample['gsm_id']}_counts.txt.gz"
                ]
            else:  # proteomics
                sample["supplementary_files"] = [
                    f"{sample['gsm_id']}_intensities.xlsx"
                ]
        
        samples.append(sample)
    
    # Generate supplementary files at series level
    series_files = []
    if include_files:
        if data_type == "single_cell":
            series_files = [
                f"{gse_id}_matrix.mtx.gz",
                f"{gse_id}_features.tsv.gz",
                f"{gse_id}_barcodes.tsv.gz",
                f"{gse_id}_metadata.txt"
            ]
        elif data_type == "bulk_rnaseq":
            series_files = [
                f"{gse_id}_counts_matrix.txt.gz",
                f"{gse_id}_sample_metadata.txt",
                f"{gse_id}_gene_annotations.txt"
            ]
        else:  # proteomics
            series_files = [
                f"{gse_id}_protein_intensities.xlsx",
                f"{gse_id}_sample_info.txt",
                f"{gse_id}_protein_annotations.txt"
            ]
    
    return {
        "gse_id": gse_id,
        "title": title,
        "summary": f"Comprehensive {data_type.replace('_', '-')} analysis of {organism} {tissue} tissue. "
                  f"This study includes {n_samples} samples processed using state-of-the-art methods.",
        "organism": organism,
        "platform": _get_platform_for_data_type(data_type),
        "submission_date": (datetime.now() - timedelta(days=fake.random_int(30, 365))).strftime("%Y-%m-%d"),
        "last_update": (datetime.now() - timedelta(days=fake.random_int(1, 30))).strftime("%Y-%m-%d"),
        "sample_count": n_samples,
        "data_type": data_type,
        "samples": samples,
        "supplementary_files": series_files,
        "contact": {
            "name": fake.name(),
            "email": fake.email(),
            "institution": fake.company()
        },
        "pmid": str(fake.random_int(10000000, 99999999)),
        "keywords": _get_keywords_for_data_type(data_type, tissue, treatment)
    }


def generate_test_workspace_state(
    modalities: Optional[List[str]] = None,
    plots: Optional[List[Dict]] = None,
    metadata: Optional[Dict] = None,
    workspace_path: Optional[Path] = None
) -> Dict[str, Any]:
    """
    Generate a controlled workspace state for testing.
    
    Args:
        modalities: List of modality names to include
        plots: List of plot metadata dictionaries
        metadata: Additional workspace metadata
        workspace_path: Path to workspace directory
    
    Returns:
        Dictionary representing workspace state
    """
    if modalities is None:
        modalities = ["test_single_cell", "test_bulk", "test_proteomics"]
    
    if plots is None:
        plots = [
            {
                "name": "qc_metrics",
                "type": "violin",
                "description": "Quality control metrics",
                "file_path": "plots/qc_metrics.html",
                "created": datetime.now().isoformat()
            },
            {
                "name": "umap_clusters", 
                "type": "scatter",
                "description": "UMAP visualization with clusters",
                "file_path": "plots/umap_clusters.html",
                "created": datetime.now().isoformat()
            }
        ]
    
    if metadata is None:
        metadata = {
            "created": datetime.now().isoformat(),
            "last_modified": datetime.now().isoformat(),
            "total_analyses": len(modalities),
            "workspace_version": "2.0.0"
        }
    
    return {
        "workspace_path": str(workspace_path) if workspace_path else "/tmp/test_workspace",
        "modalities": {
            name: {
                "type": _infer_modality_type(name),
                "file_path": f"data/{name}.h5ad",
                "created": datetime.now().isoformat(),
                "size_mb": fake.random_int(1, 100)
            }
            for name in modalities
        },
        "plots": plots,
        "metadata": metadata,
        "tool_usage_history": [
            {
                "tool": "download_geo_dataset",
                "parameters": {"gse_id": "GSE123456"},
                "timestamp": datetime.now().isoformat(),
                "success": True
            },
            {
                "tool": "perform_quality_control", 
                "parameters": {"modality": "test_single_cell"},
                "timestamp": datetime.now().isoformat(),
                "success": True
            }
        ]
    }


def generate_batch_datasets(
    n_datasets: int = 5,
    data_types: Optional[List[str]] = None,
    config: Optional[MockDataConfig] = None
) -> Dict[str, ad.AnnData]:
    """
    Generate multiple datasets for batch testing.
    
    Args:
        n_datasets: Number of datasets to generate
        data_types: Types of datasets to generate
        config: Configuration for data generation
    
    Returns:
        Dictionary mapping dataset names to AnnData objects
    """
    if data_types is None:
        data_types = ["single_cell", "bulk_rnaseq", "proteomics"]
    
    if config is None:
        config = MEDIUM_DATASET_CONFIG
    
    datasets = {}
    
    for i in range(n_datasets):
        data_type = fake.random_element(data_types)
        dataset_name = f"test_{data_type}_{i:02d}"
        
        if data_type == "single_cell":
            datasets[dataset_name] = generate_synthetic_single_cell(config=config)
        elif data_type == "bulk_rnaseq":
            datasets[dataset_name] = generate_synthetic_bulk_rnaseq(config=config)
        elif data_type == "proteomics":
            datasets[dataset_name] = generate_synthetic_proteomics(config=config)
    
    return datasets


# Helper functions
def _get_platform_for_data_type(data_type: str) -> str:
    """Get appropriate GEO platform for data type."""
    platforms = {
        "single_cell": "GPL24676",  # Illumina NovaSeq 6000
        "bulk_rnaseq": "GPL20301",  # Illumina HiSeq 4000
        "proteomics": "GPL28304"    # Orbitrap Fusion Lumos
    }
    return platforms.get(data_type, "GPL24676")


def _get_keywords_for_data_type(data_type: str, tissue: str, treatment: str) -> List[str]:
    """Generate appropriate keywords for data type."""
    base_keywords = {
        "single_cell": ["single-cell", "scRNA-seq", "transcriptomics"],
        "bulk_rnaseq": ["bulk RNA-seq", "gene expression", "transcriptomics"],
        "proteomics": ["proteomics", "mass spectrometry", "protein expression"]
    }
    
    keywords = base_keywords.get(data_type, ["genomics"])
    keywords.extend([tissue, treatment, "biomarker discovery"])
    
    return keywords


def _infer_modality_type(name: str) -> str:
    """Infer modality type from name."""
    name_lower = name.lower()
    if "single" in name_lower or "sc" in name_lower:
        return "single_cell"
    elif "bulk" in name_lower:
        return "bulk_rnaseq"
    elif "prot" in name_lower:
        return "proteomics"
    else:
        return "unknown"