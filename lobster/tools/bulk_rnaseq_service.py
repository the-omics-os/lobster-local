"""
Bulk RNA-seq analysis service.

This service provides methods for analyzing bulk RNA-seq data including
quality control, quantification, and differential expression analysis.
"""

import os
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import anndata
import numpy as np
import pandas as pd

from lobster.core import FormulaError
from lobster.tools.differential_formula_service import DifferentialFormulaService
from lobster.utils.logger import get_logger

logger = get_logger(__name__)


class BulkRNASeqError(Exception):
    """Base exception for bulk RNA-seq operations."""

    pass


class PyDESeq2Error(BulkRNASeqError):
    """Exception for pyDESeq2-related operations."""

    pass


class BulkRNASeqService:
    """
    Stateless service for bulk RNA-seq analysis workflows.

    This class provides methods for quality control, quantification,
    and differential expression analysis of bulk RNA-seq data.
    """

    def __init__(self, results_dir: Optional[Path] = None):
        """
        Initialize the bulk RNA-seq service.

        Args:
            results_dir: Optional directory for storing analysis results
        """
        logger.debug("Initializing stateless BulkRNASeqService")

        # Set up results directory
        if results_dir is None:
            data_dir = Path("data")
            data_dir.mkdir(exist_ok=True)
            self.results_dir = data_dir / "bulk_results"
        else:
            self.results_dir = Path(results_dir)

        self.results_dir.mkdir(exist_ok=True)

        # Initialize formula service
        self.formula_service = DifferentialFormulaService()

        logger.debug(
            f"BulkRNASeqService initialized with results_dir: {self.results_dir}"
        )

    def run_fastqc(self, fastq_files: List[str]) -> str:
        """
        Run FastQC quality control on FASTQ files.

        Args:
            fastq_files: List of FASTQ file paths

        Returns:
            str: Analysis results summary
        """
        logger.info(f"Starting FastQC analysis on {len(fastq_files)} FASTQ files")
        logger.debug(f"Input FASTQ files: {fastq_files}")

        try:
            # Validate input files
            valid_files = []
            for file_path in fastq_files:
                if os.path.exists(file_path):
                    file_size = os.path.getsize(file_path)
                    logger.debug(f"File {file_path}: exists, size={file_size} bytes")
                    valid_files.append(file_path)
                else:
                    logger.warning(f"FASTQ file not found: {file_path}")

            if not valid_files:
                logger.error("No valid FASTQ files found for FastQC analysis")
                return "No valid FASTQ files found for analysis"

            logger.info(f"Processing {len(valid_files)} valid FASTQ files")

            # Create output directory
            qc_dir = self.results_dir / "fastqc"
            logger.debug(f"Creating FastQC output directory: {qc_dir}")
            qc_dir.mkdir(exist_ok=True)

            # Build FastQC command
            cmd = ["fastqc", "-o", str(qc_dir)] + valid_files
            logger.info(
                f"Running FastQC command: {' '.join(cmd[:3])} ... ({len(valid_files)} files)"
            )

            # Run FastQC
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=300
            )  # 5 minute timeout

            if result.returncode != 0:
                logger.error(f"FastQC failed with return code {result.returncode}")
                logger.error(f"FastQC stderr: {result.stderr}")
                logger.debug(f"FastQC stdout: {result.stdout}")
                raise RuntimeError(f"FastQC failed: {result.stderr}")

            logger.info("FastQC completed successfully")
            logger.debug(f"FastQC stdout: {result.stdout}")

            # Parse results and create summary
            summary = self._parse_fastqc_results(qc_dir)
            logger.info(f"Generated FastQC summary: {summary}")

            return f"""FastQC Analysis Complete!

**Files Analyzed:** {len(valid_files)}
**Output Directory:** {qc_dir}

**Quality Summary:**
{summary}

Next suggested step: Run MultiQC to aggregate results or proceed with quantification using Salmon/Kallisto."""

        except subprocess.TimeoutExpired:
            logger.error("FastQC analysis timed out after 5 minutes")
            return "FastQC analysis timed out. Large files may require more time."
        except Exception as e:
            logger.exception(f"Error in FastQC analysis: {e}")
            return f"Error running FastQC: {str(e)}"

    def run_multiqc(self, input_dir: Optional[str] = None) -> str:
        """
        Run MultiQC to aggregate quality control results.

        Args:
            input_dir: Directory containing QC results

        Returns:
            str: MultiQC results summary
        """
        logger.info("Starting MultiQC analysis to aggregate QC results")

        try:
            if input_dir is None:
                input_dir = str(self.results_dir)
                logger.debug(f"Using default input directory: {input_dir}")
            else:
                logger.debug(f"Using provided input directory: {input_dir}")

            # Validate input directory
            if not os.path.exists(input_dir):
                logger.error(f"Input directory does not exist: {input_dir}")
                return f"Input directory not found: {input_dir}"

            # Check for QC files
            qc_files = []
            for ext in ["*.html", "*.txt", "*.log", "*.json"]:
                qc_files.extend(Path(input_dir).rglob(ext))
            logger.info(f"Found {len(qc_files)} potential QC files in {input_dir}")

            # Create output directory
            multiqc_dir = self.results_dir / "multiqc"
            logger.debug(f"Creating MultiQC output directory: {multiqc_dir}")
            multiqc_dir.mkdir(exist_ok=True)

            # Build MultiQC command
            cmd = ["multiqc", input_dir, "-o", str(multiqc_dir), "--force"]
            logger.info(f"Running MultiQC command: {' '.join(cmd)}")

            # Run MultiQC
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=180
            )  # 3 minute timeout

            if result.returncode != 0:
                logger.error(f"MultiQC failed with return code {result.returncode}")
                logger.error(f"MultiQC stderr: {result.stderr}")
                logger.debug(f"MultiQC stdout: {result.stdout}")
                raise RuntimeError(f"MultiQC failed: {result.stderr}")

            logger.info("MultiQC completed successfully")
            logger.debug(f"MultiQC stdout: {result.stdout}")

            # Check if report was generated
            report_path = multiqc_dir / "multiqc_report.html"
            if report_path.exists():
                report_size = os.path.getsize(report_path)
                logger.info(
                    f"MultiQC report generated: {report_path} (size: {report_size} bytes)"
                )
            else:
                logger.warning("MultiQC report file not found after execution")

            return f"""MultiQC Analysis Complete!

**Report Generated:** {multiqc_dir}/multiqc_report.html
**Data Directory:** {multiqc_dir}/multiqc_data/

The MultiQC report aggregates all quality control metrics in an interactive HTML report.

Next suggested step: Proceed with quantification if quality looks good, or investigate problematic samples."""

        except subprocess.TimeoutExpired:
            logger.error("MultiQC analysis timed out after 3 minutes")
            return "MultiQC analysis timed out. Try with a smaller input directory."
        except Exception as e:
            logger.exception(f"Error in MultiQC analysis: {e}")
            return f"Error running MultiQC: {str(e)}"

    def run_salmon_quantification(
        self,
        fastq_files: List[str],
        index_path: Optional[str] = None,
        sample_names: Optional[List[str]] = None,
        transcriptome_index: Optional[str] = None,
    ) -> str:
        """
        Run Salmon quantification on FASTQ files.

        Args:
            fastq_files: List of FASTQ file paths
            index_path: Path to Salmon index (deprecated, use transcriptome_index)
            sample_names: Optional list of sample names
            transcriptome_index: Path to Salmon transcriptome index

        Returns:
            str: Quantification results summary
        """
        logger.info(f"Starting Salmon quantification on {len(fastq_files)} FASTQ files")

        # Handle both parameter names for backward compatibility
        if transcriptome_index is not None:
            actual_index_path = transcriptome_index
        elif index_path is not None:
            actual_index_path = index_path
        else:
            raise BulkRNASeqError(
                "Either index_path or transcriptome_index must be provided"
            )

        logger.debug(f"Index path: {actual_index_path}")
        logger.debug(f"Input files: {fastq_files}")

        try:
            # Validate index path
            if not os.path.exists(actual_index_path):
                logger.error(f"Salmon index not found: {actual_index_path}")
                return f"Salmon index not found: {actual_index_path}"

            # Validate FASTQ files
            valid_files = []
            for file_path in fastq_files:
                if os.path.exists(file_path):
                    file_size = os.path.getsize(file_path)
                    logger.debug(
                        f"FASTQ file {file_path}: exists, size={file_size} bytes"
                    )
                    valid_files.append(file_path)
                else:
                    logger.warning(f"FASTQ file not found: {file_path}")

            if not valid_files:
                logger.error("No valid FASTQ files found for Salmon quantification")
                return "No valid FASTQ files found for quantification"

            # Generate sample names if not provided
            if sample_names is None:
                sample_names = [f"sample_{i+1}" for i in range(len(valid_files))]
                logger.debug(f"Generated sample names: {sample_names}")

            logger.info(f"Processing {len(valid_files)} samples with Salmon")

            # Create output directory
            salmon_dir = self.results_dir / "salmon"
            logger.debug(f"Creating Salmon output directory: {salmon_dir}")
            salmon_dir.mkdir(exist_ok=True)

            results = []
            failed_samples = []

            for i, (fastq_file, sample_name) in enumerate(
                zip(valid_files, sample_names)
            ):
                logger.info(
                    f"Processing sample {i+1}/{len(valid_files)}: {sample_name}"
                )
                sample_dir = salmon_dir / sample_name

                # Build Salmon command
                cmd = [
                    "salmon",
                    "quant",
                    "-i",
                    actual_index_path,
                    "-l",
                    "A",  # Auto-detect library type
                    "-r",
                    fastq_file,
                    "-o",
                    str(sample_dir),
                    "--validateMappings",
                    "--threads",
                    "1",  # Use single thread for better logging
                ]

                logger.debug(
                    f"Running Salmon command: {' '.join(cmd[:6])} ... (full command logged)"
                )

                # Run Salmon with timeout
                result = subprocess.run(
                    cmd, capture_output=True, text=True, timeout=600
                )  # 10 minute timeout

                if result.returncode != 0:
                    logger.error(
                        f"Salmon failed for {sample_name} with return code {result.returncode}"
                    )
                    logger.error(f"Salmon stderr: {result.stderr}")
                    logger.debug(f"Salmon stdout: {result.stdout}")
                    failed_samples.append(sample_name)
                    continue

                logger.info(f"Salmon completed successfully for {sample_name}")
                logger.debug(f"Salmon stdout: {result.stdout}")

                # Verify output files
                quant_file = sample_dir / "quant.sf"
                if quant_file.exists():
                    quant_size = os.path.getsize(quant_file)
                    logger.debug(
                        f"Quantification file created: {quant_file} (size: {quant_size} bytes)"
                    )
                    results.append(sample_name)
                else:
                    logger.error(
                        f"Quantification file not found for {sample_name}: {quant_file}"
                    )
                    failed_samples.append(sample_name)

            # Load and combine quantification results
            if results:
                logger.info(
                    f"Combining quantification results from {len(results)} successful samples"
                )
                combined_data = self._combine_salmon_results(salmon_dir, results)
                logger.info(f"Combined expression matrix shape: {combined_data.shape}")

            if failed_samples:
                logger.warning(
                    f"Failed to process {len(failed_samples)} samples: {failed_samples}"
                )

            return f"""Salmon Quantification Complete!

**Samples Processed:** {len(results)}/{len(valid_files)}
**Failed Samples:** {len(failed_samples)} ({failed_samples if failed_samples else 'None'})
**Output Directory:** {salmon_dir}
**Combined Expression Matrix:** {combined_data.shape if results else 'N/A'}

Next suggested step: Import quantification data with tximport for differential expression analysis."""

        except subprocess.TimeoutExpired:
            logger.error("Salmon quantification timed out after 10 minutes")
            return "Salmon quantification timed out. Consider using smaller files or increasing timeout."
        except Exception as e:
            logger.exception(f"Error in Salmon quantification: {e}")
            return f"Error running Salmon: {str(e)}"

    def run_differential_expression_analysis(
        self,
        adata: anndata.AnnData,
        groupby: str,
        group1: str,
        group2: str,
        method: str = "deseq2_like",
        min_expression_threshold: float = 1.0,
    ) -> Tuple[anndata.AnnData, Dict[str, Any]]:
        """
        Run differential expression analysis on bulk RNA-seq data.

        Args:
            adata: AnnData object with bulk RNA-seq data
            groupby: Column name for grouping (e.g., 'condition', 'treatment')
            group1: First group for comparison (e.g., 'control')
            group2: Second group for comparison (e.g., 'treatment')
            method: Analysis method ('deseq2_like', 'wilcoxon', 't_test')
            min_expression_threshold: Minimum expression threshold for filtering

        Returns:
            Tuple[anndata.AnnData, Dict[str, Any]]: AnnData with results and DE stats

        Raises:
            BulkRNASeqError: If differential expression analysis fails
        """
        try:
            # Validate required parameters
            if adata is None:
                raise BulkRNASeqError("AnnData object is required")
            if groupby is None:
                raise BulkRNASeqError("groupby parameter is required")
            if group1 is None or group2 is None:
                raise BulkRNASeqError("group1 and group2 parameters are required")

            logger.info(
                f"Running differential expression analysis: {group1} vs {group2}"
            )

            # Create working copy
            adata_de = adata.copy()

            # Validate groupby column and groups exist
            if groupby not in adata_de.obs.columns:
                raise BulkRNASeqError(
                    f"Group column '{groupby}' not found in observations"
                )

            available_groups = adata_de.obs[groupby].unique()
            if group1 not in available_groups:
                raise BulkRNASeqError(
                    f"Group '{group1}' not found in {groupby}. Available: {available_groups}"
                )
            if group2 not in available_groups:
                raise BulkRNASeqError(
                    f"Group '{group2}' not found in {groupby}. Available: {available_groups}"
                )

            # Filter genes by expression threshold
            if min_expression_threshold > 0:
                gene_filter = (adata_de.X > min_expression_threshold).sum(axis=0) >= 2
                if hasattr(gene_filter, "A1"):
                    gene_filter = gene_filter.A1
                adata_de = adata_de[:, gene_filter].copy()
                logger.info(
                    f"Filtered to {adata_de.n_vars} genes above expression threshold"
                )

            # Create comparison matrix
            group1_mask = adata_de.obs[groupby] == group1
            group2_mask = adata_de.obs[groupby] == group2

            group1_data = adata_de[group1_mask]
            group2_data = adata_de[group2_mask]

            if group1_data.n_obs == 0 or group2_data.n_obs == 0:
                raise BulkRNASeqError(
                    f"One or both groups have no samples: {group1}={group1_data.n_obs}, {group2}={group2_data.n_obs}"
                )

            # Run differential expression based on method
            if method == "deseq2_like":
                results_df = self._run_deseq2_like_analysis(
                    group1_data, group2_data, group1, group2
                )
            elif method == "wilcoxon":
                results_df = self._run_wilcoxon_test(
                    group1_data, group2_data, group1, group2
                )
            elif method == "t_test":
                results_df = self._run_ttest_analysis(
                    group1_data, group2_data, group1, group2
                )
            else:
                raise BulkRNASeqError(
                    f"Unknown differential expression method: {method}"
                )

            # Add results to AnnData
            adata_de.uns[f"de_results_{group1}_vs_{group2}"] = results_df.to_dict()

            # Count significant genes
            significant_genes = results_df[results_df["padj"] < 0.05]
            upregulated = significant_genes[significant_genes["log2FoldChange"] > 0]
            downregulated = significant_genes[significant_genes["log2FoldChange"] < 0]

            # Compile analysis statistics
            de_stats = {
                "analysis_type": "differential_expression",
                "method": method,
                "groupby": groupby,
                "group1": group1,
                "group2": group2,
                "n_samples_group1": group1_data.n_obs,
                "n_samples_group2": group2_data.n_obs,
                "n_genes_tested": len(results_df),
                "n_significant_genes": len(significant_genes),
                "n_upregulated": len(upregulated),
                "n_downregulated": len(downregulated),
                "significant_genes_list": significant_genes.index.tolist()[
                    :100
                ],  # Top 100
                "top_upregulated": upregulated.nlargest(
                    10, "log2FoldChange"
                ).index.tolist(),
                "top_downregulated": downregulated.nsmallest(
                    10, "log2FoldChange"
                ).index.tolist(),
                "de_results_key": f"de_results_{group1}_vs_{group2}",
            }

            logger.info(
                f"Differential expression completed: {len(significant_genes)} significant genes found"
            )

            return adata_de, de_stats

        except Exception as e:
            logger.exception(f"Error in differential expression analysis: {e}")
            raise BulkRNASeqError(f"Differential expression analysis failed: {str(e)}")

    def run_pathway_enrichment(
        self,
        gene_list: List[str],
        analysis_type: str = "GO",
        background_genes: Optional[List[str]] = None,
        organism: Optional[str] = None,
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Run pathway enrichment analysis on a gene list using GSEApy.

        Supports multiple enrichment databases:
        - GO: Gene Ontology (Biological Process, Molecular Function, Cellular Component)
        - KEGG: KEGG pathway database
        - Reactome: Reactome pathway database
        - WikiPathways: WikiPathways database

        Args:
            gene_list: List of genes for enrichment analysis (gene symbols)
            analysis_type: Type of analysis ("GO", "KEGG", "Reactome", "WikiPathways")
            background_genes: Background gene set (None for default - all detected genes)
            organism: Organism name (default: "human", supports "mouse", "rat", etc.)

        Returns:
            Tuple[pd.DataFrame, Dict[str, Any]]: Enrichment results and stats
                - DataFrame contains: Term, Overlap, P-value, Adjusted P-value, Genes
                - Dict contains summary statistics

        Raises:
            BulkRNASeqError: If enrichment analysis fails
        """
        try:
            # Validate inputs
            if not gene_list or len(gene_list) == 0:
                raise BulkRNASeqError(
                    "Empty gene list provided for enrichment analysis"
                )

            # Set default organism
            if organism is None:
                organism = "human"

            logger.info(
                f"Running {analysis_type} pathway enrichment on {len(gene_list)} genes for {organism}"
            )

            # Try to import GSEApy
            try:
                import gseapy as gp
            except ImportError:
                logger.error("GSEApy not installed")
                raise BulkRNASeqError(
                    "GSEApy not installed. Install with: pip install gseapy"
                )

            # Map analysis type to Enrichr gene set libraries
            library_map = {
                "GO": [
                    "GO_Biological_Process_2023",
                    "GO_Molecular_Function_2023",
                    "GO_Cellular_Component_2023",
                ],
                "KEGG": ["KEGG_2021_Human"],
                "Reactome": ["Reactome_2022"],
                "WikiPathways": ["WikiPathway_2023_Human"],
            }

            # Get organism-specific libraries
            if organism.lower() == "mouse":
                library_map["KEGG"] = ["KEGG_2021_Mouse"]
            elif organism.lower() == "rat":
                library_map["KEGG"] = ["KEGG_2021_Rat"]

            if analysis_type not in library_map:
                raise BulkRNASeqError(
                    f"Unknown analysis type '{analysis_type}'. "
                    f"Supported types: {list(library_map.keys())}"
                )

            gene_sets = library_map[analysis_type]
            logger.info(f"Using gene set libraries: {gene_sets}")

            # Run enrichment analysis
            all_results = []
            for gene_set in gene_sets:
                try:
                    logger.info(f"Running enrichment for {gene_set}")
                    enr_result = gp.enrichr(
                        gene_list=gene_list,
                        gene_sets=[gene_set],
                        organism=organism.capitalize(),
                        background=background_genes,
                        cutoff=0.05,  # Adjusted p-value cutoff
                    )

                    if enr_result.results is not None and not enr_result.results.empty:
                        results_df = enr_result.results.copy()
                        results_df["gene_set_library"] = gene_set
                        all_results.append(results_df)
                        logger.info(
                            f"Found {len(results_df)} enriched terms in {gene_set}"
                        )
                except Exception as e:
                    logger.warning(f"Failed to run enrichment for {gene_set}: {e}")
                    continue

            if not all_results:
                logger.warning(
                    f"No significant enrichment found for {analysis_type} analysis"
                )
                # Return empty results
                enrichment_df = pd.DataFrame(
                    columns=[
                        "Term",
                        "Overlap",
                        "P-value",
                        "Adjusted P-value",
                        "Genes",
                        "gene_set_library",
                    ]
                )
            else:
                # Combine results from all gene sets
                enrichment_df = pd.concat(all_results, ignore_index=True)

                # Sort by adjusted p-value
                enrichment_df = enrichment_df.sort_values("Adjusted P-value")

            # Generate analysis statistics
            significant_terms = enrichment_df[enrichment_df["Adjusted P-value"] < 0.05]

            enrichment_stats = {
                "analysis_type": "pathway_enrichment",
                "enrichment_database": analysis_type,
                "n_genes_input": len(gene_list),
                "n_terms_total": len(enrichment_df),
                "n_significant_terms": len(significant_terms),
                "organism": organism,
                "gene_set_libraries": gene_sets,
                "top_terms": (
                    significant_terms["Term"].head(10).tolist()
                    if len(significant_terms) > 0
                    else []
                ),
            }

            logger.info(
                f"Pathway enrichment completed: {len(significant_terms)} significant terms found"
            )

            return enrichment_df, enrichment_stats

        except Exception as e:
            if isinstance(e, BulkRNASeqError):
                raise
            else:
                logger.exception(f"Error in pathway enrichment analysis: {e}")
                raise BulkRNASeqError(f"Pathway enrichment analysis failed: {e}")

    def _run_deseq2_like_analysis(
        self,
        group1_data: anndata.AnnData,
        group2_data: anndata.AnnData,
        group1_name: str,
        group2_name: str,
    ) -> pd.DataFrame:
        """Run DESeq2-like differential expression analysis."""

        if group1_data is None or group2_data is None:
            raise BulkRNASeqError("group1_data and group2_data are required")

        logger.info(f"Running DESeq2-like analysis: {group1_name} vs {group2_name}")

        # Extract expression matrices
        if hasattr(group1_data.X, "toarray"):
            group1_expr = group1_data.X.toarray()
            group2_expr = group2_data.X.toarray()
        else:
            group1_expr = group1_data.X
            group2_expr = group2_data.X

        n_genes = group1_data.n_vars
        gene_names = group1_data.var_names

        # Calculate basic statistics
        group1_mean = np.mean(group1_expr, axis=0)
        group2_mean = np.mean(group2_expr, axis=0)

        # Calculate fold changes (add pseudocount to avoid log(0))
        log2_fold_change = np.log2((group2_mean + 1) / (group1_mean + 1))

        # Simple statistical test (t-test) for p-values
        from scipy import stats

        p_values = []
        for i in range(n_genes):
            if group1_expr.shape[0] > 1 and group2_expr.shape[0] > 1:
                _, p_val = stats.ttest_ind(group1_expr[:, i], group2_expr[:, i])
                p_values.append(p_val)
            else:
                p_values.append(1.0)  # No test possible with single sample

        p_values = np.array(p_values)

        # Multiple testing correction (Benjamini-Hochberg)
        from statsmodels.stats.multitest import multipletests

        _, p_adjusted, _, _ = multipletests(p_values, method="fdr_bh")

        # Create results DataFrame
        results_df = pd.DataFrame(
            {
                "baseMean": (group1_mean + group2_mean) / 2,
                "log2FoldChange": log2_fold_change,
                "lfcSE": np.ones(n_genes),  # Simplified
                "stat": log2_fold_change / np.ones(n_genes),  # Simplified
                "pvalue": p_values,
                "padj": p_adjusted,
            },
            index=gene_names,
        )

        return results_df.dropna()

    def _run_wilcoxon_test(
        self,
        group1_data: anndata.AnnData,
        group2_data: anndata.AnnData,
        group1_name: str,
        group2_name: str,
    ) -> pd.DataFrame:
        """Run Wilcoxon rank-sum test for differential expression."""

        if group1_data is None or group2_data is None:
            raise BulkRNASeqError("group1_data and group2_data are required")

        logger.info(f"Running Wilcoxon test: {group1_name} vs {group2_name}")

        from scipy import stats

        # Extract expression matrices
        if hasattr(group1_data.X, "toarray"):
            group1_expr = group1_data.X.toarray()
            group2_expr = group2_data.X.toarray()
        else:
            group1_expr = group1_data.X
            group2_expr = group2_data.X

        n_genes = group1_data.n_vars
        gene_names = group1_data.var_names

        # Calculate statistics
        group1_mean = np.mean(group1_expr, axis=0)
        group2_mean = np.mean(group2_expr, axis=0)
        log2_fold_change = np.log2((group2_mean + 1) / (group1_mean + 1))

        # Wilcoxon test for each gene
        p_values = []
        for i in range(n_genes):
            if group1_expr.shape[0] > 1 and group2_expr.shape[0] > 1:
                _, p_val = stats.ranksums(group1_expr[:, i], group2_expr[:, i])
                p_values.append(p_val)
            else:
                p_values.append(1.0)

        p_values = np.array(p_values)

        # Multiple testing correction
        from statsmodels.stats.multitest import multipletests

        _, p_adjusted, _, _ = multipletests(p_values, method="fdr_bh")

        results_df = pd.DataFrame(
            {
                "baseMean": (group1_mean + group2_mean) / 2,
                "log2FoldChange": log2_fold_change,
                "lfcSE": np.ones(n_genes),
                "stat": log2_fold_change / np.ones(n_genes),
                "pvalue": p_values,
                "padj": p_adjusted,
            },
            index=gene_names,
        )

        return results_df.dropna()

    def _run_ttest_analysis(
        self,
        group1_data: anndata.AnnData,
        group2_data: anndata.AnnData,
        group1_name: str,
        group2_name: str,
    ) -> pd.DataFrame:
        """Run t-test for differential expression."""

        if group1_data is None or group2_data is None:
            raise BulkRNASeqError("group1_data and group2_data are required")

        logger.info(f"Running t-test: {group1_name} vs {group2_name}")

        from scipy import stats

        # Extract expression matrices
        if hasattr(group1_data.X, "toarray"):
            group1_expr = group1_data.X.toarray()
            group2_expr = group2_data.X.toarray()
        else:
            group1_expr = group1_data.X
            group2_expr = group2_data.X

        n_genes = group1_data.n_vars
        gene_names = group1_data.var_names

        # Calculate statistics
        group1_mean = np.mean(group1_expr, axis=0)
        group2_mean = np.mean(group2_expr, axis=0)
        log2_fold_change = np.log2((group2_mean + 1) / (group1_mean + 1))

        # T-test for each gene
        p_values = []
        t_stats = []
        for i in range(n_genes):
            if group1_expr.shape[0] > 1 and group2_expr.shape[0] > 1:
                t_stat, p_val = stats.ttest_ind(group1_expr[:, i], group2_expr[:, i])
                t_stats.append(t_stat)
                p_values.append(p_val)
            else:
                t_stats.append(0.0)
                p_values.append(1.0)

        p_values = np.array(p_values)
        t_stats = np.array(t_stats)

        # Multiple testing correction
        from statsmodels.stats.multitest import multipletests

        _, p_adjusted, _, _ = multipletests(p_values, method="fdr_bh")

        results_df = pd.DataFrame(
            {
                "baseMean": (group1_mean + group2_mean) / 2,
                "log2FoldChange": log2_fold_change,
                "lfcSE": np.ones(n_genes),
                "stat": t_stats,
                "pvalue": p_values,
                "padj": p_adjusted,
            },
            index=gene_names,
        )

        return results_df.dropna()

    def _parse_fastqc_results(self, qc_dir: Path) -> str:
        """Parse FastQC results and create summary."""
        html_files = list(qc_dir.glob("*.html"))
        return f"Generated {len(html_files)} FastQC reports. Check individual HTML files for detailed quality metrics."

    def _combine_salmon_results(
        self, salmon_dir: Path, sample_names: List[str]
    ) -> pd.DataFrame:
        """
        Combine Salmon quantification results into a single matrix.

        DEPRECATED: Use merge_salmon_results() for enhanced functionality.
        This method is kept for backward compatibility.
        """
        dfs = []
        for sample in sample_names:
            quant_file = salmon_dir / sample / "quant.sf"
            if quant_file.exists():
                df = pd.read_csv(quant_file, sep="\t")
                df = df.set_index("Name")["TPM"]
                df.name = sample
                dfs.append(df)

        if dfs:
            combined = pd.concat(dfs, axis=1)
            return combined.fillna(0)
        else:
            raise ValueError("No valid Salmon results found")

    def merge_salmon_results(
        self,
        salmon_dir: Path,
        sample_names: Optional[List[str]] = None,
        value_column: str = "TPM",
        gene_id_column: str = "Name",
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Merge per-sample Salmon quantification files into unified expression matrix.

        Args:
            salmon_dir: Directory containing per-sample subdirectories with quant.sf files
            sample_names: Optional list of sample names; auto-detected if None
            value_column: Column to extract (TPM, NumReads, etc.)
            gene_id_column: Column containing gene/transcript IDs

        Returns:
            Tuple of (merged_dataframe, metadata_dict)
            - merged_dataframe: Samples as columns, genes as rows
            - metadata_dict: Detection statistics and file info

        Raises:
            FileNotFoundError: If no valid quant.sf files found
            ValueError: If data format is invalid
        """
        # Auto-detect sample names if not provided
        if sample_names is None:
            sample_names = [
                d.name
                for d in salmon_dir.iterdir()
                if d.is_dir() and (d / "quant.sf").exists()
            ]
            logger.info(f"Auto-detected {len(sample_names)} Salmon samples")

        if not sample_names:
            raise FileNotFoundError(
                f"No Salmon quantification directories found in {salmon_dir}"
            )

        # Merge per-sample files
        dfs = []
        successful_samples = []
        failed_samples = []

        for sample in sample_names:
            quant_file = salmon_dir / sample / "quant.sf"

            if not quant_file.exists():
                logger.warning(f"Missing quant.sf for sample {sample}")
                failed_samples.append(sample)
                continue

            try:
                df = pd.read_csv(quant_file, sep="\t")

                # Validate required columns
                if gene_id_column not in df.columns:
                    raise ValueError(f"Missing gene ID column '{gene_id_column}'")
                if value_column not in df.columns:
                    raise ValueError(f"Missing value column '{value_column}'")

                # Extract values
                df = df.set_index(gene_id_column)[value_column]
                df.name = sample
                dfs.append(df)
                successful_samples.append(sample)

            except Exception as e:
                logger.error(f"Failed to read {sample}: {e}")
                failed_samples.append(sample)

        if not dfs:
            raise ValueError("No valid Salmon quantification files could be loaded")

        # Combine into matrix
        combined = pd.concat(dfs, axis=1)
        combined = combined.fillna(0)  # Missing genes set to 0

        # Generate metadata
        metadata = {
            "quantification_tool": "Salmon",
            "n_samples": len(successful_samples),
            "n_genes": len(combined),
            "value_type": value_column,
            "successful_samples": successful_samples,
            "failed_samples": failed_samples,
            "source_directory": str(salmon_dir),
        }

        logger.info(
            f"Merged Salmon results: {len(successful_samples)} samples × {len(combined)} genes"
        )

        return combined, metadata

    def merge_kallisto_results(
        self,
        kallisto_dir: Path,
        sample_names: Optional[List[str]] = None,
        value_column: str = "tpm",
        gene_id_column: str = "target_id",
        use_h5: bool = False,
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Merge per-sample Kallisto quantification files into unified expression matrix.

        Kallisto produces three output formats:
        - abundance.tsv: Text format (default)
        - abundance.h5: HDF5 format (faster, requires h5py)
        - abundance.txt: Alternative text format

        Args:
            kallisto_dir: Directory containing per-sample subdirectories
            sample_names: Optional list of sample names; auto-detected if None
            value_column: Column to extract (tpm, est_counts, etc.)
            gene_id_column: Column containing gene/transcript IDs
            use_h5: If True, prefer HDF5 format over TSV

        Returns:
            Tuple of (merged_dataframe, metadata_dict)

        Raises:
            FileNotFoundError: If no valid abundance files found
            ImportError: If use_h5=True but h5py not available
        """
        # Check H5 support
        if use_h5:
            try:
                import h5py  # noqa: F401
            except ImportError:
                logger.warning("h5py not available, falling back to TSV format")
                use_h5 = False

        # Auto-detect sample names
        if sample_names is None:
            sample_names = []
            for d in kallisto_dir.iterdir():
                if not d.is_dir():
                    continue

                # Check for any Kallisto output format
                if (
                    (d / "abundance.tsv").exists()
                    or (d / "abundance.h5").exists()
                    or (d / "abundance.txt").exists()
                ):
                    sample_names.append(d.name)

            logger.info(f"Auto-detected {len(sample_names)} Kallisto samples")

        if not sample_names:
            raise FileNotFoundError(
                f"No Kallisto quantification directories found in {kallisto_dir}"
            )

        # Merge per-sample files
        dfs = []
        successful_samples = []
        failed_samples = []
        formats_used = {"h5": 0, "tsv": 0, "txt": 0}

        for sample in sample_names:
            sample_dir = kallisto_dir / sample

            # Try formats in priority order
            df = None
            format_used = None

            if use_h5 and (sample_dir / "abundance.h5").exists():
                try:
                    df = self._read_kallisto_h5(
                        sample_dir / "abundance.h5", value_column, gene_id_column
                    )
                    format_used = "h5"
                except Exception as e:
                    logger.warning(f"Failed to read H5 for {sample}: {e}, trying TSV")

            if df is None and (sample_dir / "abundance.tsv").exists():
                try:
                    df = self._read_kallisto_tsv(
                        sample_dir / "abundance.tsv", value_column, gene_id_column
                    )
                    format_used = "tsv"
                except Exception as e:
                    logger.warning(f"Failed to read TSV for {sample}: {e}")

            if df is None and (sample_dir / "abundance.txt").exists():
                try:
                    df = self._read_kallisto_tsv(
                        sample_dir / "abundance.txt", value_column, gene_id_column
                    )
                    format_used = "txt"
                except Exception as e:
                    logger.error(f"Failed to read TXT for {sample}: {e}")

            if df is not None:
                df.name = sample
                dfs.append(df)
                successful_samples.append(sample)
                formats_used[format_used] += 1
            else:
                failed_samples.append(sample)

        if not dfs:
            raise ValueError("No valid Kallisto quantification files could be loaded")

        # Combine into matrix
        combined = pd.concat(dfs, axis=1)
        combined = combined.fillna(0)

        # Generate metadata
        metadata = {
            "quantification_tool": "Kallisto",
            "n_samples": len(successful_samples),
            "n_genes": len(combined),
            "value_type": value_column,
            "successful_samples": successful_samples,
            "failed_samples": failed_samples,
            "formats_used": formats_used,
            "source_directory": str(kallisto_dir),
        }

        logger.info(
            f"Merged Kallisto results: {len(successful_samples)} samples × {len(combined)} genes"
        )

        return combined, metadata

    def _read_kallisto_h5(
        self,
        h5_file: Path,
        value_column: str,
        gene_id_column: str,
    ) -> pd.Series:
        """Read Kallisto HDF5 abundance file."""
        import h5py

        with h5py.File(h5_file, "r") as f:
            # Kallisto H5 structure: /aux/ids, /est_counts, /tpm, etc.
            gene_ids = f["aux"]["ids"][:]
            if isinstance(gene_ids[0], bytes):
                gene_ids = [gid.decode("utf-8") for gid in gene_ids]

            # Map column name to H5 dataset
            h5_column_map = {
                "tpm": "tpm",
                "est_counts": "est_counts",
                "eff_length": "eff_length",
            }

            if value_column not in h5_column_map:
                raise ValueError(f"Invalid value column '{value_column}' for H5 format")

            values = f[h5_column_map[value_column]][:]

        return pd.Series(values, index=gene_ids, name=h5_file.parent.name)

    def _read_kallisto_tsv(
        self,
        tsv_file: Path,
        value_column: str,
        gene_id_column: str,
    ) -> pd.Series:
        """Read Kallisto TSV abundance file."""
        df = pd.read_csv(tsv_file, sep="\t")

        # Validate columns
        if gene_id_column not in df.columns:
            raise ValueError(f"Missing gene ID column '{gene_id_column}'")
        if value_column not in df.columns:
            raise ValueError(f"Missing value column '{value_column}'")

        return df.set_index(gene_id_column)[value_column]

    def load_from_quantification_files(
        self,
        quantification_dir: Path,
        tool: str = "auto",
        sample_names: Optional[List[str]] = None,
        **kwargs,
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Load bulk RNA-seq quantification files (Kallisto or Salmon) into expression matrix.

        Args:
            quantification_dir: Directory containing per-sample quantification files
            tool: Quantification tool ("kallisto", "salmon", or "auto" for detection)
            sample_names: Optional list of sample names
            **kwargs: Additional arguments passed to tool-specific loaders

        Returns:
            Tuple of (expression_dataframe, metadata_dict)

        Example:
            >>> df, metadata = service.load_from_quantification_files(
            ...     Path("GSE130036_RAW/"),
            ...     tool="auto"
            ... )
            >>> print(df.shape)  # (60000 genes, 4 samples)
        """
        # Auto-detect tool if needed
        if tool == "auto":
            tool = self._detect_quantification_tool(quantification_dir)
            logger.info(f"Auto-detected quantification tool: {tool}")

        # Route to appropriate loader
        if tool == "kallisto":
            return self.merge_kallisto_results(
                quantification_dir, sample_names, **kwargs
            )
        elif tool == "salmon":
            return self.merge_salmon_results(quantification_dir, sample_names, **kwargs)
        else:
            raise ValueError(f"Unsupported quantification tool: {tool}")

    def _detect_quantification_tool(self, directory: Path) -> str:
        """Auto-detect whether directory contains Kallisto or Salmon files."""
        kallisto_count = 0
        salmon_count = 0

        for subdir in directory.iterdir():
            if not subdir.is_dir():
                continue

            # Check for Kallisto signatures
            if (
                (subdir / "abundance.tsv").exists()
                or (subdir / "abundance.h5").exists()
                or (subdir / "abundance.txt").exists()
            ):
                kallisto_count += 1

            # Check for Salmon signatures
            if (subdir / "quant.sf").exists() or (subdir / "quant.genes.sf").exists():
                salmon_count += 1

        if kallisto_count > salmon_count:
            return "kallisto"
        elif salmon_count > 0:
            return "salmon"
        else:
            raise ValueError(
                f"No Kallisto or Salmon quantification files detected in {directory}"
            )

    def run_pydeseq2_analysis(
        self,
        count_matrix: Optional[pd.DataFrame] = None,
        metadata: Optional[pd.DataFrame] = None,
        formula: Optional[str] = None,
        contrast: Optional[List[str]] = None,
        alpha: float = 0.05,
        shrink_lfc: bool = True,
        n_cpus: int = 1,
    ) -> pd.DataFrame:
        """
        Run pyDESeq2 differential expression analysis.

        Args:
            count_matrix: Count matrix (genes x samples)
            metadata: Sample metadata DataFrame
            formula: R-style formula string (e.g., "~condition + batch")
            contrast: Contrast specification [factor, level1, level2]
            alpha: Significance threshold for multiple testing
            shrink_lfc: Whether to apply log fold change shrinkage
            n_cpus: Number of CPUs for parallel processing

        Returns:
            pd.DataFrame: Differential expression results

        Raises:
            PyDESeq2Error: If pyDESeq2 analysis fails
        """
        try:
            logger.info(f"Running pyDESeq2 analysis with formula: {formula}")

            # Validate pyDESeq2 installation
            installation_status = self.validate_pydeseq2_setup()
            if not all(installation_status.values()):
                missing_deps = [k for k, v in installation_status.items() if not v]
                raise PyDESeq2Error(f"Missing pyDESeq2 dependencies: {missing_deps}")

            # Import pyDESeq2 components
            from pydeseq2.dds import DeseqDataSet
            from pydeseq2.default_inference import DefaultInference
            from pydeseq2.ds import DeseqStats

            # Validate inputs
            self._validate_deseq2_inputs(count_matrix, metadata, formula, contrast)

            # Parse formula and validate design
            design_info = self.formula_service.parse_formula(formula, metadata)
            self.formula_service.construct_design_matrix(
                design_info, metadata, contrast
            )

            # Ensure count matrix is integer and properly oriented (samples x genes for pyDESeq2)
            count_matrix_int = count_matrix.T.astype(
                int
            )  # Transpose to samples x genes

            # Align metadata with count matrix
            aligned_metadata = metadata.loc[count_matrix_int.index].copy()

            # Create inference object with parallel processing
            inference = DefaultInference(n_cpus=n_cpus)

            # Initialize DESeq2 dataset
            logger.info("Initializing DESeq2 dataset...")
            dds = DeseqDataSet(
                counts=count_matrix_int,
                metadata=aligned_metadata,
                design=formula,
                inference=inference,
            )

            # Fit dispersion and log fold changes
            logger.info("Fitting DESeq2 model...")
            dds.deseq2()

            # Perform statistical testing
            logger.info(f"Running statistical tests for contrast: {contrast}")
            ds = DeseqStats(dds, contrast=contrast, alpha=alpha, inference=inference)
            ds.summary()

            # Optional LFC shrinkage for better estimates
            if shrink_lfc:
                logger.info("Applying log fold change shrinkage...")
                try:
                    # Construct coefficient name for shrinkage
                    factor, level1, level2 = contrast
                    coeff_name = f"{factor}_{level1}_vs_{level2}"
                    ds.lfc_shrink(coeff=coeff_name)
                except Exception as e:
                    logger.warning(f"LFC shrinkage failed, continuing without: {e}")

            # Extract results
            results_df = ds.results_df.copy()

            # Add additional statistics
            results_df = self._enhance_deseq2_results(results_df, dds, contrast)

            logger.info(f"pyDESeq2 analysis completed: {len(results_df)} genes tested")

            return results_df

        except Exception as e:
            if isinstance(e, PyDESeq2Error):
                raise
            else:
                logger.exception(f"Error in pyDESeq2 analysis: {e}")
                raise PyDESeq2Error(f"pyDESeq2 analysis failed: {e}")

    def run_pydeseq2_from_pseudobulk(
        self,
        pseudobulk_adata: anndata.AnnData,
        formula: str,
        contrast: List[str],
        count_layer: str = None,
        **kwargs,
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Run pyDESeq2 analysis on pseudobulk data.

        Args:
            pseudobulk_adata: Pseudobulk AnnData object
            formula: R-style formula string
            contrast: Contrast specification [factor, level1, level2]
            count_layer: Layer containing count data (default: X)
            **kwargs: Additional arguments for run_pydeseq2_analysis

        Returns:
            Tuple[pd.DataFrame, Dict[str, Any]]: Results DataFrame and analysis stats

        Raises:
            PyDESeq2Error: If analysis fails
        """
        try:
            logger.info("Running pyDESeq2 analysis on pseudobulk data")

            # Extract count matrix
            if count_layer and count_layer in pseudobulk_adata.layers:
                count_matrix = pd.DataFrame(
                    pseudobulk_adata.layers[count_layer].T,
                    index=pseudobulk_adata.var_names,
                    columns=pseudobulk_adata.obs_names,
                )
            else:
                count_matrix = pd.DataFrame(
                    pseudobulk_adata.X.T,
                    index=pseudobulk_adata.var_names,
                    columns=pseudobulk_adata.obs_names,
                )

            # Extract metadata
            metadata = pseudobulk_adata.obs.copy()

            # Run pyDESeq2 analysis
            results_df = self.run_pydeseq2_analysis(
                count_matrix, metadata, formula, contrast, **kwargs
            )

            # Create analysis statistics
            significant_genes = results_df[
                results_df["padj"] < kwargs.get("alpha", 0.05)
            ]
            upregulated = significant_genes[significant_genes["log2FoldChange"] > 0]
            downregulated = significant_genes[significant_genes["log2FoldChange"] < 0]

            analysis_stats = {
                "analysis_type": "pydeseq2_pseudobulk",
                "formula": formula,
                "contrast": contrast,
                "n_pseudobulk_samples": pseudobulk_adata.n_obs,
                "n_genes_tested": len(results_df),
                "n_significant_genes": len(significant_genes),
                "n_upregulated": len(upregulated),
                "n_downregulated": len(downregulated),
                "significance_threshold": kwargs.get("alpha", 0.05),
                "top_upregulated": upregulated.nlargest(
                    10, "log2FoldChange"
                ).index.tolist(),
                "top_downregulated": downregulated.nsmallest(
                    10, "log2FoldChange"
                ).index.tolist(),
            }

            logger.info(
                f"pyDESeq2 pseudobulk analysis completed: {len(significant_genes)} significant genes"
            )

            return results_df, analysis_stats

        except Exception as e:
            logger.exception(f"Error in pyDESeq2 pseudobulk analysis: {e}")
            raise PyDESeq2Error(f"pyDESeq2 pseudobulk analysis failed: {e}")

    def validate_pydeseq2_setup(self) -> Dict[str, bool]:
        """
        Validate pyDESeq2 installation and dependencies.

        Returns:
            Dict[str, bool]: Installation status for each component
        """
        status = {}

        try:
            from pydeseq2.dds import DeseqDataSet  # noqa: F401
            from pydeseq2.ds import DeseqStats  # noqa: F401

            status["pydeseq2"] = True
            status["pydeseq2_available"] = True  # Add expected key for tests
            logger.debug("pyDESeq2 core components available")
        except ImportError as e:
            logger.warning(f"pyDESeq2 not available: {e}")
            status["pydeseq2"] = False
            status["pydeseq2_available"] = False

        try:
            from pydeseq2.default_inference import DefaultInference  # noqa: F401

            status["pydeseq2_inference"] = True
            logger.debug("pyDESeq2 inference components available")
        except ImportError as e:
            logger.warning(f"pyDESeq2 inference not available: {e}")
            status["pydeseq2_inference"] = False

        try:
            import numba

            status["numba"] = True
            logger.debug(f"numba version {numba.__version__} available")
        except ImportError:
            logger.warning("numba not available - pyDESeq2 performance may be reduced")
            status["numba"] = False

        try:
            import statsmodels

            status["statsmodels"] = True
            logger.debug(f"statsmodels version {statsmodels.__version__} available")
        except ImportError:
            logger.warning("statsmodels not available")
            status["statsmodels"] = False

        return status

    def create_formula_design(
        self,
        metadata: pd.DataFrame,
        condition_col: Optional[str] = None,
        batch_col: Optional[str] = None,
        reference_condition: Optional[str] = None,
        condition_column: Optional[str] = None,
        batch_column: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Create design matrix for common experimental designs.

        Args:
            metadata: Sample metadata DataFrame
            condition_col: Main condition column name
            batch_col: Optional batch effect column name
            reference_condition: Reference level for condition
            condition_column: Condition column (alternative name)
            batch_column: Batch column (alternative name)

        Returns:
            Dict[str, Any]: Design matrix information

        Raises:
            FormulaError: If design construction fails
        """
        try:
            if metadata is None:
                raise FormulaError("metadata is required")

            # Handle parameter name variations
            actual_condition_col = condition_column or condition_col
            actual_batch_col = batch_column or batch_col

            if actual_condition_col is None:
                raise FormulaError("condition column must be specified")

            return self.formula_service.create_simple_design(
                metadata, actual_condition_col, actual_batch_col, reference_condition
            )
        except Exception as e:
            raise FormulaError(f"Failed to create design matrix: {e}")

    def validate_experimental_design(
        self, metadata: pd.DataFrame, formula: str, min_replicates: int = 2
    ) -> Dict[str, Any]:
        """
        Validate experimental design for statistical analysis.

        Args:
            metadata: Sample metadata DataFrame
            formula: R-style formula string
            min_replicates: Minimum replicates per group

        Returns:
            Dict[str, Any]: Validation results
        """
        try:
            if metadata is None or formula is None:
                return {
                    "valid": False,
                    "errors": ["metadata and formula are required"],
                    "warnings": [],
                    "design_summary": {},
                }

            return self.formula_service.validate_experimental_design(
                metadata, formula, min_replicates
            )
        except Exception as e:
            return {
                "valid": False,
                "errors": [str(e)],
                "warnings": [],
                "design_summary": {},
            }

    def _validate_deseq2_inputs(
        self,
        count_matrix: pd.DataFrame,
        metadata: pd.DataFrame,
        formula: str,
        contrast: List[str],
    ) -> None:
        """Validate inputs for pyDESeq2 analysis."""

        if count_matrix is None:
            raise PyDESeq2Error("Count matrix is required")

        # Check count matrix
        if count_matrix.empty:
            raise PyDESeq2Error("Count matrix is empty")

        if not count_matrix.dtypes.apply(lambda x: np.issubdtype(x, np.number)).all():
            raise PyDESeq2Error("Count matrix contains non-numeric data")

        if (count_matrix < 0).any().any():
            raise PyDESeq2Error("Count matrix contains negative values")

        # Check metadata
        if metadata.empty:
            raise PyDESeq2Error("Metadata is empty")

        # Check sample alignment
        count_samples = set(count_matrix.columns)
        metadata_samples = set(metadata.index)

        if not count_samples.issubset(metadata_samples):
            missing = count_samples - metadata_samples
            raise PyDESeq2Error(
                f"Samples in count matrix missing from metadata: {missing}"
            )

        # Validate contrast
        if len(contrast) != 3:
            raise PyDESeq2Error("Contrast must be [factor, level1, level2]")

        factor, level1, level2 = contrast

        if factor not in metadata.columns:
            raise PyDESeq2Error(f"Contrast factor '{factor}' not found in metadata")

        factor_levels = metadata[factor].unique()
        if level1 not in factor_levels:
            raise PyDESeq2Error(
                f"Contrast level '{level1}' not found in factor '{factor}'"
            )

        if level2 not in factor_levels:
            raise PyDESeq2Error(
                f"Contrast level '{level2}' not found in factor '{factor}'"
            )

    def _enhance_deseq2_results(
        self, results_df: pd.DataFrame, dds, contrast: List[str]
    ) -> pd.DataFrame:
        """Enhance pyDESeq2 results with additional statistics."""

        if results_df is None or contrast is None:
            raise PyDESeq2Error("results_df and contrast are required")

        # Add contrast information
        results_df["contrast"] = f"{contrast[0]}_{contrast[1]}_vs_{contrast[2]}"

        # Add significance categories
        alpha = 0.05  # Default significance threshold
        results_df["significant"] = (results_df["padj"] < alpha) & (
            ~results_df["padj"].isna()
        )

        # Add regulation direction
        results_df["regulation"] = "unchanged"
        results_df.loc[
            (results_df["significant"]) & (results_df["log2FoldChange"] > 0),
            "regulation",
        ] = "upregulated"
        results_df.loc[
            (results_df["significant"]) & (results_df["log2FoldChange"] < 0),
            "regulation",
        ] = "downregulated"

        # Add rank based on adjusted p-value
        results_df["rank"] = results_df["padj"].rank(method="min", na_option="bottom")

        return results_df
