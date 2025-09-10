"""
Factory classes for generating synthetic biological datasets.

This module uses the factory pattern to create consistent, 
reproducible test datasets for various biological data types.
"""

import numpy as np
import pandas as pd
import anndata as ad
from typing import Optional, List, Dict, Any, Tuple
from pathlib import Path
import factory
from factory import Factory, Sequence, LazyAttribute
from faker import Faker

from .base import MockDataConfig, MEDIUM_DATASET_CONFIG


fake = Faker()
Faker.seed(42)


class BaseDataFactory(Factory):
    """Base factory for biological datasets."""
    
    class Meta:
        abstract = True
    
    config = factory.SubFactory(lambda: MEDIUM_DATASET_CONFIG)
    
    @classmethod
    def _setup_numpy_seed(cls, config: MockDataConfig) -> None:
        """Set up numpy random seed from config."""
        np.random.seed(config.seed)


class SingleCellDataFactory(BaseDataFactory):
    """Factory for generating synthetic single-cell RNA-seq data."""
    
    n_cells = LazyAttribute(lambda obj: obj.config.default_cell_count)
    n_genes = LazyAttribute(lambda obj: obj.config.default_gene_count)
    n_cell_types = LazyAttribute(lambda obj: obj.config.n_cell_types)
    sparsity = LazyAttribute(lambda obj: obj.config.sparsity_level)
    
    class Meta:
        model = ad.AnnData
    
    @classmethod
    def _create(cls, model_class, **kwargs):
        """Create a synthetic single-cell AnnData object."""
        config = kwargs.get('config', MEDIUM_DATASET_CONFIG)
        n_cells = kwargs.get('n_cells', config.default_cell_count)
        n_genes = kwargs.get('n_genes', config.default_gene_count)
        n_cell_types = kwargs.get('n_cell_types', config.n_cell_types)
        sparsity = kwargs.get('sparsity', config.sparsity_level)
        
        cls._setup_numpy_seed(config)
        
        # Generate count matrix with negative binomial distribution
        # This simulates realistic single-cell RNA-seq count distributions
        X = np.random.negative_binomial(
            n=5, p=0.3, size=(n_cells, n_genes)
        ).astype(np.float32)
        
        # Add sparsity (zeros) to make it realistic
        zero_mask = np.random.random((n_cells, n_genes)) < sparsity
        X[zero_mask] = 0
        
        # Add some highly expressed genes (housekeeping genes)
        n_housekeeping = min(50, n_genes // 20)
        housekeeping_idx = np.random.choice(n_genes, n_housekeeping, replace=False)
        X[:, housekeeping_idx] *= np.random.uniform(2, 5, size=(n_cells, n_housekeeping))
        
        # Create cell and gene names
        cell_names = [f"Cell_{fake.uuid4()[:8]}" for _ in range(n_cells)]
        gene_names = [f"Gene_{i:04d}" for i in range(n_genes)]
        
        # Create AnnData object
        adata = ad.AnnData(
            X=X,
            obs=pd.DataFrame(index=cell_names),
            var=pd.DataFrame(index=gene_names)
        )
        
        # Add realistic gene metadata
        adata.var["gene_ids"] = [f"ENSG{i:011d}" for i in range(n_genes)]
        adata.var["feature_types"] = ["Gene Expression"] * n_genes
        adata.var["chromosome"] = np.random.choice(
            [f"chr{i}" for i in range(1, 23)] + ["chrX", "chrY"],
            size=n_genes
        )
        
        # Mark some genes as mitochondrial and ribosomal
        mt_genes = np.random.choice(n_genes, size=int(0.02 * n_genes), replace=False)
        ribo_genes = np.random.choice(n_genes, size=int(0.05 * n_genes), replace=False)
        
        adata.var["is_mt"] = False
        adata.var["is_ribo"] = False
        adata.var.iloc[mt_genes, adata.var.columns.get_loc("is_mt")] = True
        adata.var.iloc[ribo_genes, adata.var.columns.get_loc("is_ribo")] = True
        
        # Add cell metadata
        adata.obs["total_counts"] = np.array(X.sum(axis=1))
        adata.obs["n_genes_by_counts"] = np.array((X > 0).sum(axis=1))
        
        # Calculate mitochondrial and ribosomal gene percentages
        mt_counts = X[:, adata.var["is_mt"]].sum(axis=1)
        ribo_counts = X[:, adata.var["is_ribo"]].sum(axis=1)
        total_counts = adata.obs["total_counts"]
        
        adata.obs["pct_counts_mt"] = (mt_counts / total_counts * 100).fillna(0)
        adata.obs["pct_counts_ribo"] = (ribo_counts / total_counts * 100).fillna(0)
        
        # Add cell type assignments
        cell_types = [f"CellType_{i}" for i in range(n_cell_types)]
        
        # Create cell type proportions if not provided
        if config.cell_type_proportions is None:
            proportions = np.random.dirichlet(np.ones(n_cell_types))
        else:
            proportions = np.array(config.cell_type_proportions)
        
        # Assign cell types based on proportions
        cell_type_assignments = np.random.choice(
            cell_types, size=n_cells, p=proportions
        )
        adata.obs["cell_type"] = cell_type_assignments
        
        # Add batch information if batch effects are enabled
        if config.batch_effects:
            batch_names = [f"Batch_{i+1}" for i in range(config.n_batches)]
            adata.obs["batch"] = np.random.choice(batch_names, size=n_cells)
            
            # Add batch effects to gene expression
            if config.batch_effect_strength > 0:
                for i, batch in enumerate(batch_names):
                    batch_mask = adata.obs["batch"] == batch
                    batch_effect = np.random.normal(
                        1.0, config.batch_effect_strength, size=n_genes
                    )
                    adata.X[batch_mask, :] *= batch_effect
        
        # Add technical metadata
        adata.obs["doublet_score"] = np.random.beta(2, 8, size=n_cells)  # Most cells low doublet score
        adata.obs["is_doublet"] = adata.obs["doublet_score"] > 0.5
        
        # Add sample metadata
        adata.obs["sample_id"] = np.random.choice(
            [f"Sample_{i:02d}" for i in range(4)], size=n_cells
        )
        
        return adata


class BulkRNASeqDataFactory(BaseDataFactory):
    """Factory for generating synthetic bulk RNA-seq data."""
    
    n_samples = LazyAttribute(lambda obj: obj.config.default_sample_count)
    n_genes = LazyAttribute(lambda obj: obj.config.default_gene_count)
    
    class Meta:
        model = ad.AnnData
    
    @classmethod
    def _create(cls, model_class, **kwargs):
        """Create a synthetic bulk RNA-seq AnnData object."""
        config = kwargs.get('config', MEDIUM_DATASET_CONFIG)
        n_samples = kwargs.get('n_samples', config.default_sample_count)
        n_genes = kwargs.get('n_genes', config.default_gene_count)
        
        cls._setup_numpy_seed(config)
        
        # Generate count matrix with higher counts than single-cell
        # Use negative binomial with higher n parameter for bulk data
        X = np.random.negative_binomial(
            n=20, p=0.1, size=(n_samples, n_genes)
        ).astype(np.float32)
        
        # Add some noise
        if config.noise_level > 0:
            noise = np.random.normal(0, config.noise_level * X.mean(), X.shape)
            X = np.maximum(0, X + noise)
        
        # Create sample and gene names
        sample_names = [f"Sample_{i:02d}" for i in range(n_samples)]
        gene_names = [f"Gene_{i:04d}" for i in range(n_genes)]
        
        adata = ad.AnnData(
            X=X,
            obs=pd.DataFrame(index=sample_names),
            var=pd.DataFrame(index=gene_names)
        )
        
        # Add gene metadata
        adata.var["gene_ids"] = [f"ENSG{i:011d}" for i in range(n_genes)]
        adata.var["gene_name"] = [f"GENE{i}" for i in range(n_genes)]
        adata.var["biotype"] = np.random.choice(
            ["protein_coding", "lncRNA", "miRNA", "pseudogene"],
            size=n_genes,
            p=[0.7, 0.15, 0.05, 0.1]
        )
        adata.var["chromosome"] = np.random.choice(
            [f"chr{i}" for i in range(1, 23)] + ["chrX", "chrY"],
            size=n_genes
        )
        
        # Add sample metadata
        n_conditions = 2
        samples_per_condition = n_samples // n_conditions
        conditions = (["Treatment"] * samples_per_condition + 
                     ["Control"] * (n_samples - samples_per_condition))
        adata.obs["condition"] = conditions
        
        # Add batch information
        if config.batch_effects:
            n_batches = min(config.n_batches, n_samples // 2)
            batch_names = [f"Batch_{i+1}" for i in range(n_batches)]
            adata.obs["batch"] = np.random.choice(batch_names, size=n_samples)
        
        # Add demographic information
        adata.obs["sex"] = np.random.choice(["M", "F"], size=n_samples)
        adata.obs["age"] = np.random.randint(20, 80, size=n_samples)
        adata.obs["tissue"] = np.random.choice(
            ["Brain", "Liver", "Kidney", "Heart"], size=n_samples
        )
        
        # Add technical metadata
        adata.obs["library_size"] = np.array(X.sum(axis=1))
        adata.obs["n_detected_genes"] = np.array((X > 0).sum(axis=1))
        adata.obs["rna_integrity_number"] = np.random.uniform(6.0, 10.0, size=n_samples)
        
        return adata


class ProteomicsDataFactory(BaseDataFactory):
    """Factory for generating synthetic proteomics data."""
    
    n_samples = LazyAttribute(lambda obj: obj.config.default_sample_count * 2)  # More samples for proteomics
    n_proteins = LazyAttribute(lambda obj: obj.config.default_protein_count)
    missing_rate = LazyAttribute(lambda obj: obj.config.missing_data_rate)
    
    class Meta:
        model = ad.AnnData
    
    @classmethod
    def _create(cls, model_class, **kwargs):
        """Create a synthetic proteomics AnnData object."""
        config = kwargs.get('config', MEDIUM_DATASET_CONFIG)
        n_samples = kwargs.get('n_samples', config.default_sample_count * 2)
        n_proteins = kwargs.get('n_proteins', config.default_protein_count)
        missing_rate = kwargs.get('missing_rate', config.missing_data_rate)
        
        cls._setup_numpy_seed(config)
        
        # Generate intensity matrix with log-normal distribution
        # This is typical for proteomics data
        X = np.random.lognormal(
            mean=10, sigma=2, size=(n_samples, n_proteins)
        ).astype(np.float32)
        
        # Add missing values (common in proteomics)
        missing_mask = np.random.random((n_samples, n_proteins)) < missing_rate
        X[missing_mask] = np.nan
        
        # Create sample and protein names
        sample_names = [f"Sample_{i:03d}" for i in range(n_samples)]
        protein_names = [f"Protein_{i:03d}" for i in range(n_proteins)]
        
        adata = ad.AnnData(
            X=X,
            obs=pd.DataFrame(index=sample_names),
            var=pd.DataFrame(index=protein_names)
        )
        
        # Add protein metadata
        adata.var["protein_ids"] = [f"P{i:05d}" for i in range(n_proteins)]
        adata.var["protein_names"] = [f"PROT{i}" for i in range(n_proteins)]
        adata.var["molecular_weight"] = np.random.uniform(10, 200, n_proteins)
        adata.var["isoelectric_point"] = np.random.uniform(3, 12, n_proteins)
        adata.var["subcellular_location"] = np.random.choice(
            ["Cytoplasm", "Nucleus", "Membrane", "Mitochondria", "ER"],
            size=n_proteins
        )
        
        # Add sample metadata
        n_conditions = 3
        samples_per_condition = n_samples // n_conditions
        conditions = (["Disease"] * samples_per_condition +
                     ["Healthy"] * samples_per_condition +
                     ["Control"] * (n_samples - 2 * samples_per_condition))
        adata.obs["condition"] = conditions
        
        # Add additional metadata
        adata.obs["tissue"] = np.random.choice(
            ["Brain", "Liver", "Kidney", "Heart", "Lung"], size=n_samples
        )
        adata.obs["age"] = np.random.randint(25, 85, size=n_samples)
        adata.obs["sex"] = np.random.choice(["M", "F"], size=n_samples)
        
        # Add batch information
        if config.batch_effects:
            batch_names = [f"Batch_{i+1}" for i in range(config.n_batches)]
            adata.obs["batch"] = np.random.choice(batch_names, size=n_samples)
        
        # Add technical metadata
        adata.obs["total_protein_intensity"] = np.nansum(X, axis=1)
        adata.obs["n_detected_proteins"] = np.sum(~np.isnan(X), axis=1)
        adata.obs["missing_value_pct"] = np.sum(np.isnan(X), axis=1) / n_proteins * 100
        
        return adata


class MultiModalDataFactory(BaseDataFactory):
    """Factory for generating synthetic multi-modal datasets."""
    
    class Meta:
        model = dict  # Returns a dictionary of AnnData objects
    
    @classmethod
    def _create(cls, model_class, **kwargs):
        """Create synthetic multi-modal data (RNA + Protein)."""
        config = kwargs.get('config', MEDIUM_DATASET_CONFIG)
        
        # Create single-cell RNA-seq data
        rna_data = SingleCellDataFactory._create(ad.AnnData, config=config)
        
        # Create corresponding protein data with same cells
        n_cells = rna_data.n_obs
        n_proteins = config.default_protein_count
        
        cls._setup_numpy_seed(config)
        
        # Generate protein expression data
        protein_X = np.random.lognormal(
            mean=5, sigma=1.5, size=(n_cells, n_proteins)
        ).astype(np.float32)
        
        # Add missing values
        missing_mask = np.random.random((n_cells, n_proteins)) < 0.3
        protein_X[missing_mask] = np.nan
        
        # Create protein AnnData with same cell names
        protein_names = [f"Protein_{i:03d}" for i in range(n_proteins)]
        protein_data = ad.AnnData(
            X=protein_X,
            obs=rna_data.obs.copy(),  # Same cell metadata
            var=pd.DataFrame(index=protein_names)
        )
        
        # Add protein-specific metadata
        protein_data.var["protein_ids"] = [f"P{i:05d}" for i in range(n_proteins)]
        protein_data.var["protein_names"] = [f"PROT{i}" for i in range(n_proteins)]
        protein_data.var["molecular_weight"] = np.random.uniform(10, 200, n_proteins)
        
        return {
            "rna": rna_data,
            "protein": protein_data
        }


# Factory aliases for convenience
ScRNASeqFactory = SingleCellDataFactory
BulkFactory = BulkRNASeqDataFactory
ProtFactory = ProteomicsDataFactory
MultiModalFactory = MultiModalDataFactory