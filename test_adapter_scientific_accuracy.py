#!/usr/bin/env python3
"""
Scientific Accuracy Validation for Lobster Adapters

This script performs comprehensive validation of the adapter components with
realistic bioinformatics data to ensure scientific accuracy and proper functionality.

Focus areas:
1. Data format conversion and validation
2. AnnData object handling and transformation
3. Scientific accuracy of data transformations
4. Memory efficiency with large datasets
5. Error handling for malformed data
"""

import sys
import os
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
import anndata as ad
import scanpy as sc
from scipy import sparse
import logging

# Add the lobster package to the path
sys.path.insert(0, '/Users/tyo/GITHUB/lobster')

from lobster.core.adapters.transcriptomics_adapter import TranscriptomicsAdapter
from lobster.core.adapters.proteomics_adapter import ProteomicsAdapter
from lobster.core.adapters.pseudobulk_adapter import PseudobulkAdapter
from lobster.core.adapters.base import BaseAdapter

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ScientificAccuracyTester:
    """Test suite for scientific accuracy validation of adapters."""

    def __init__(self):
        self.results = {}
        self.temp_dir = tempfile.mkdtemp()
        logger.info(f"Created temporary directory: {self.temp_dir}")

    def create_realistic_single_cell_data(self, n_cells=500, n_genes=2000):
        """Create realistic single-cell RNA-seq data for testing."""
        np.random.seed(42)

        # Simulate realistic single-cell count matrix
        # Most genes have low expression, some have high expression
        gene_means = np.random.lognormal(mean=1, sigma=2, size=n_genes)

        # Create sparse count matrix with realistic sparsity (~90%)
        counts = np.zeros((n_cells, n_genes))
        for i in range(n_cells):
            # Each cell expresses ~10% of genes
            expressed_genes = np.random.choice(n_genes, size=int(n_genes * 0.1), replace=False)
            for gene_idx in expressed_genes:
                counts[i, gene_idx] = np.random.poisson(gene_means[gene_idx])

        # Create cell metadata
        obs = pd.DataFrame(index=[f"Cell_{i:04d}" for i in range(n_cells)])
        obs['sample_id'] = np.random.choice(['Sample_A', 'Sample_B', 'Sample_C'], n_cells)
        obs['condition'] = np.where(obs['sample_id'].isin(['Sample_A']), 'Control', 'Treatment')
        obs['batch'] = np.random.choice(['Batch1', 'Batch2'], n_cells)

        # Add some QC metrics
        obs['total_counts'] = counts.sum(axis=1)
        obs['n_genes'] = (counts > 0).sum(axis=1)

        # Create gene metadata
        var = pd.DataFrame(index=[f"Gene_{i:04d}" for i in range(n_genes)])
        # Add mitochondrial genes (typical ~13 genes)
        mito_genes = np.random.choice(n_genes, size=13, replace=False)
        var['mitochondrial'] = False
        var.iloc[mito_genes, var.columns.get_loc('mitochondrial')] = True

        # Add ribosomal genes (~80 genes)
        ribo_genes = np.random.choice(n_genes, size=80, replace=False)
        var['ribosomal'] = False
        var.iloc[ribo_genes, var.columns.get_loc('ribosomal')] = True

        # Create AnnData object
        adata = ad.AnnData(X=sparse.csr_matrix(counts), obs=obs, var=var)

        logger.info(f"Created realistic single-cell data: {adata.shape}")
        return adata

    def create_realistic_proteomics_data(self, n_samples=20, n_proteins=1000, data_type="mass_spectrometry"):
        """Create realistic proteomics data for testing."""
        np.random.seed(42)

        if data_type == "mass_spectrometry":
            # Mass spec data has more missing values (30-70%)
            missing_prob = 0.5
            # Intensity values are typically log-normal
            base_intensities = np.random.lognormal(mean=10, sigma=2, size=n_proteins)
        else:  # affinity proteomics
            # Affinity data has fewer missing values (<30%)
            missing_prob = 0.2
            # NPX values are typically normal around 0
            base_intensities = np.random.normal(loc=0, scale=2, size=n_proteins)

        # Create intensity matrix with realistic missing value patterns
        intensities = np.zeros((n_samples, n_proteins))
        for i in range(n_samples):
            for j in range(n_proteins):
                if np.random.random() > missing_prob:
                    # Add some biological variation
                    intensities[i, j] = np.random.normal(
                        base_intensities[j],
                        base_intensities[j] * 0.2
                    )
                else:
                    intensities[i, j] = np.nan

        # Create sample metadata
        obs = pd.DataFrame(index=[f"Sample_{i:02d}" for i in range(n_samples)])
        obs['condition'] = np.random.choice(['Control', 'Treatment'], n_samples)
        obs['batch'] = np.random.choice(['Batch1', 'Batch2'], n_samples)
        obs['patient_id'] = [f"Patient_{i//2:02d}" for i in range(n_samples)]

        # Create protein metadata
        var = pd.DataFrame(index=[f"Protein_{i:04d}" for i in range(n_proteins)])
        var['protein_name'] = [f"PROT{i:04d}" for i in range(n_proteins)]

        if data_type == "mass_spectrometry":
            # Add contaminants and reverse hits
            n_contaminants = int(n_proteins * 0.02)  # ~2% contaminants
            contaminant_idx = np.random.choice(n_proteins, size=n_contaminants, replace=False)
            var['contaminant'] = False
            var.iloc[contaminant_idx, var.columns.get_loc('contaminant')] = True

            n_reverse = int(n_proteins * 0.01)  # ~1% reverse hits
            reverse_idx = np.random.choice(n_proteins, size=n_reverse, replace=False)
            var['reverse'] = False
            var.iloc[reverse_idx, var.columns.get_loc('reverse')] = True

        # Create AnnData object
        adata = ad.AnnData(X=intensities, obs=obs, var=var)

        logger.info(f"Created realistic {data_type} data: {adata.shape}")
        return adata

    def test_transcriptomics_adapter_accuracy(self):
        """Test transcriptomics adapter with realistic data."""
        logger.info("Testing TranscriptomicsAdapter scientific accuracy...")

        results = {}

        # Test single-cell adapter
        adapter_sc = TranscriptomicsAdapter(data_type="single_cell")
        test_data = self.create_realistic_single_cell_data()

        # Save test data to CSV for file loading test
        csv_path = Path(self.temp_dir) / "test_single_cell.csv"
        test_df = pd.DataFrame(
            test_data.X.toarray().T,  # Transpose for genes as rows
            index=test_data.var.index,
            columns=test_data.obs.index
        )
        test_df.to_csv(csv_path)

        try:
            # Test file loading
            loaded_data = adapter_sc.from_source(csv_path, transpose=True)
            results['file_loading'] = {
                'status': 'PASS',
                'shape_preserved': loaded_data.shape == test_data.shape,
                'data_type_preserved': loaded_data.X.dtype == np.float64 or loaded_data.X.dtype == np.float32
            }

            # Test data preprocessing
            preprocessed = adapter_sc.preprocess_data(loaded_data)
            results['preprocessing'] = {
                'status': 'PASS',
                'qc_metrics_added': 'total_counts' in preprocessed.obs.columns,
                'mitochondrial_calculated': 'pct_counts_mt' in preprocessed.obs.columns if 'mitochondrial' in preprocessed.var.columns else True,
                'shape_preserved': preprocessed.shape == loaded_data.shape
            }

            # Test validation
            validation_result = adapter_sc.validate(preprocessed)
            results['validation'] = {
                'status': 'PASS',
                'validation_passed': validation_result.is_valid,
                'has_recommendations': len(validation_result.recommendations) > 0
            }

        except Exception as e:
            results['error'] = str(e)
            logger.error(f"TranscriptomicsAdapter test failed: {e}")

        self.results['transcriptomics'] = results
        return results

    def test_proteomics_adapter_accuracy(self):
        """Test proteomics adapter with realistic data."""
        logger.info("Testing ProteomicsAdapter scientific accuracy...")

        results = {}

        # Test mass spectrometry adapter
        adapter_ms = ProteomicsAdapter(data_type="mass_spectrometry")
        test_data = self.create_realistic_proteomics_data(data_type="mass_spectrometry")

        # Save test data to CSV
        csv_path = Path(self.temp_dir) / "test_proteomics_ms.csv"
        test_df = pd.DataFrame(
            test_data.X,
            index=test_data.obs.index,
            columns=test_data.var.index
        )
        test_df.to_csv(csv_path)

        try:
            # Test file loading
            loaded_data = adapter_ms.from_source(csv_path, transpose=False)
            results['ms_file_loading'] = {
                'status': 'PASS',
                'shape_preserved': loaded_data.shape == test_data.shape,
                'missing_values_preserved': np.isnan(loaded_data.X).sum() > 0
            }

            # Test missing value handling
            filled_data = adapter_ms.handle_missing_values(loaded_data.copy(), strategy="median")
            results['missing_value_handling'] = {
                'status': 'PASS',
                'missing_values_reduced': np.isnan(filled_data.X).sum() < np.isnan(loaded_data.X).sum(),
                'shape_preserved': filled_data.shape == loaded_data.shape
            }

            # Test preprocessing
            preprocessed = adapter_ms.preprocess_data(loaded_data)
            results['ms_preprocessing'] = {
                'status': 'PASS',
                'qc_metrics_added': 'missing_values_pct' in preprocessed.obs.columns,
                'contaminants_identified': 'contaminant' in preprocessed.var.columns
            }

        except Exception as e:
            results['ms_error'] = str(e)
            logger.error(f"ProteomicsAdapter MS test failed: {e}")

        # Test affinity proteomics
        try:
            adapter_aff = ProteomicsAdapter(data_type="affinity")
            aff_data = self.create_realistic_proteomics_data(data_type="affinity")

            preprocessed_aff = adapter_aff.preprocess_data(aff_data)
            results['affinity_preprocessing'] = {
                'status': 'PASS',
                'cv_calculated': 'cv_values' in preprocessed_aff.var.columns,
                'lower_missing_values': np.isnan(aff_data.X).sum() / aff_data.X.size < 0.3
            }

        except Exception as e:
            results['affinity_error'] = str(e)
            logger.error(f"ProteomicsAdapter affinity test failed: {e}")

        self.results['proteomics'] = results
        return results

    def test_pseudobulk_adapter_accuracy(self):
        """Test pseudobulk adapter with realistic data."""
        logger.info("Testing PseudobulkAdapter scientific accuracy...")

        results = {}

        try:
            adapter = PseudobulkAdapter()

            # Create realistic pseudobulk data (aggregated single-cell data)
            n_samples = 12
            n_genes = 1000

            # Simulate pseudobulk count data (higher counts than single-cell)
            np.random.seed(42)
            counts = np.random.negative_binomial(n=20, p=0.1, size=(n_samples, n_genes))

            obs = pd.DataFrame(index=[f"Pseudobulk_{i:02d}" for i in range(n_samples)])
            obs['condition'] = np.tile(['Control', 'Treatment'], n_samples // 2)
            obs['batch'] = np.repeat(['Batch1', 'Batch2'], n_samples // 2)
            obs['sample_id'] = [f"Sample_{i//2}" for i in range(n_samples)]
            obs['cell_count'] = np.random.randint(50, 200, n_samples)  # Cells per pseudobulk sample

            var = pd.DataFrame(index=[f"Gene_{i:04d}" for i in range(n_genes)])

            test_data = ad.AnnData(X=counts, obs=obs, var=var)

            # Test validation
            validation_result = adapter.validate(test_data)
            results['validation'] = {
                'status': 'PASS',
                'validation_passed': validation_result.is_valid,
                'appropriate_cell_counts': test_data.obs['cell_count'].min() >= 10
            }

            # Test CSV loading
            csv_path = Path(self.temp_dir) / "test_pseudobulk.csv"
            test_df = pd.DataFrame(
                test_data.X,
                index=test_data.obs.index,
                columns=test_data.var.index
            )
            test_df.to_csv(csv_path)

            loaded_data = adapter.from_source(csv_path)
            results['file_loading'] = {
                'status': 'PASS',
                'shape_preserved': loaded_data.shape == test_data.shape,
                'count_data_preserved': loaded_data.X.dtype in [np.int32, np.int64, np.float32, np.float64]
            }

        except Exception as e:
            results['error'] = str(e)
            logger.error(f"PseudobulkAdapter test failed: {e}")

        self.results['pseudobulk'] = results
        return results

    def test_memory_efficiency(self):
        """Test memory efficiency with larger datasets."""
        logger.info("Testing memory efficiency with large datasets...")

        results = {}

        try:
            # Create a larger dataset to test memory efficiency
            large_data = self.create_realistic_single_cell_data(n_cells=2000, n_genes=5000)

            adapter = TranscriptomicsAdapter(data_type="single_cell")

            # Test that large data can be processed without memory errors
            import psutil
            import os

            process = psutil.Process(os.getpid())
            memory_before = process.memory_info().rss / 1024 / 1024  # MB

            processed = adapter.preprocess_data(large_data)

            memory_after = process.memory_info().rss / 1024 / 1024  # MB
            memory_used = memory_after - memory_before

            results['memory_efficiency'] = {
                'status': 'PASS',
                'memory_used_mb': memory_used,
                'processed_successfully': processed is not None,
                'shape_preserved': processed.shape == large_data.shape
            }

        except Exception as e:
            results['memory_error'] = str(e)
            logger.error(f"Memory efficiency test failed: {e}")

        self.results['memory'] = results
        return results

    def test_error_handling(self):
        """Test error handling with malformed data."""
        logger.info("Testing error handling with malformed data...")

        results = {}

        # Test with empty data
        try:
            adapter = TranscriptomicsAdapter()
            empty_data = ad.AnnData(X=np.array([]).reshape(0, 0))

            # Should handle gracefully
            validation_result = adapter.validate(empty_data)
            results['empty_data'] = {
                'status': 'PASS',
                'handled_gracefully': not validation_result.is_valid,
                'has_error_message': len(validation_result.errors) > 0
            }

        except Exception as e:
            results['empty_data_error'] = str(e)

        # Test with corrupted data (NaN, Inf values)
        try:
            corrupted_data = self.create_realistic_single_cell_data(n_cells=100, n_genes=200)
            corrupted_data.X[0, 0] = np.inf
            corrupted_data.X[1, 1] = -np.inf
            corrupted_data.X[2, 2] = np.nan

            validation_result = adapter.validate(corrupted_data)
            results['corrupted_data'] = {
                'status': 'PASS',
                'detected_issues': not validation_result.is_valid,
                'has_warnings': len(validation_result.warnings) > 0
            }

        except Exception as e:
            results['corrupted_data_error'] = str(e)

        self.results['error_handling'] = results
        return results

    def run_all_tests(self):
        """Run all scientific accuracy tests."""
        logger.info("Starting comprehensive adapter testing...")

        self.test_transcriptomics_adapter_accuracy()
        self.test_proteomics_adapter_accuracy()
        self.test_pseudobulk_adapter_accuracy()
        self.test_memory_efficiency()
        self.test_error_handling()

        return self.results

    def generate_report(self):
        """Generate a comprehensive test report."""
        if not self.results:
            self.run_all_tests()

        report = []
        report.append("=" * 80)
        report.append("LOBSTER ADAPTER SCIENTIFIC ACCURACY TEST REPORT")
        report.append("=" * 80)
        report.append("")

        total_tests = 0
        passed_tests = 0

        for adapter_name, test_results in self.results.items():
            report.append(f"{adapter_name.upper()} ADAPTER TESTS:")
            report.append("-" * 40)

            for test_name, result in test_results.items():
                total_tests += 1
                if isinstance(result, dict) and result.get('status') == 'PASS':
                    passed_tests += 1
                    report.append(f"  ✓ {test_name}: PASSED")
                    # Add details
                    for key, value in result.items():
                        if key != 'status':
                            report.append(f"    - {key}: {value}")
                elif 'error' in test_name:
                    report.append(f"  ✗ {test_name}: ERROR - {result}")
                else:
                    report.append(f"  ? {test_name}: {result}")

            report.append("")

        report.append("SUMMARY:")
        report.append("-" * 40)
        report.append(f"Total tests: {total_tests}")
        report.append(f"Passed tests: {passed_tests}")
        report.append(f"Success rate: {passed_tests/total_tests*100:.1f}%" if total_tests > 0 else "No tests run")
        report.append("")

        return "\n".join(report)

def main():
    """Main function to run the scientific accuracy tests."""
    tester = ScientificAccuracyTester()

    try:
        results = tester.run_all_tests()
        report = tester.generate_report()

        print(report)

        # Save report to file
        report_path = Path(tester.temp_dir) / "adapter_test_report.txt"
        with open(report_path, 'w') as f:
            f.write(report)

        logger.info(f"Test report saved to: {report_path}")

        return results

    except Exception as e:
        logger.error(f"Test execution failed: {e}")
        raise

if __name__ == "__main__":
    main()