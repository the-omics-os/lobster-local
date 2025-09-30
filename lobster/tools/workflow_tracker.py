"""
Workflow tracker for differential expression analysis iterations.

This module provides lightweight tracking of DE analysis iterations, enabling
users to compare results between different formulas, parameters, and filtering
criteria through conversational agent interaction.
"""

import json
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union
import pandas as pd
import numpy as np
from pathlib import Path

from lobster.utils.logger import get_logger

logger = get_logger(__name__)


class WorkflowTracker:
    """
    Lightweight tracker for DE analysis iterations.
    
    This class manages iteration tracking for differential expression analyses,
    providing comparison capabilities and workflow state management without
    complex UI components.
    """

    def __init__(self, data_manager):
        """
        Initialize workflow tracker.
        
        Args:
            data_manager: DataManagerV2 instance for accessing modalities
        """
        self.data_manager = data_manager
        self.logger = logger
        
        # Iteration storage
        self.iterations = {}  # {modality_name: {iteration_id: iteration_data}}
        self.current_iteration_ids = {}  # {modality_name: current_iteration_counter}
        
        # Comparison cache
        self.comparison_cache = {}

    def track_iteration(
        self,
        modality_name: str,
        formula: str,
        contrast: List[str],
        results_df: pd.DataFrame,
        analysis_stats: Dict[str, Any],
        parameters: Dict[str, Any],
        iteration_name: Optional[str] = None
    ) -> str:
        """
        Track a new analysis iteration.
        
        Args:
            modality_name: Name of the modality being analyzed
            formula: Formula used for analysis
            contrast: Contrast specification [factor, level1, level2]
            results_df: Results DataFrame from DE analysis
            analysis_stats: Analysis statistics dictionary
            parameters: Analysis parameters used
            iteration_name: Custom iteration name
            
        Returns:
            str: Iteration ID for reference
        """
        try:
            # Initialize tracking for this modality if needed
            if modality_name not in self.iterations:
                self.iterations[modality_name] = {}
                self.current_iteration_ids[modality_name] = 0
            
            # Generate iteration ID
            self.current_iteration_ids[modality_name] += 1
            iteration_id = self.current_iteration_ids[modality_name]
            
            if iteration_name is None:
                iteration_name = f"iteration_{iteration_id}"
            
            # Extract key results
            significant_genes = set(results_df[results_df['padj'] < parameters.get('alpha', 0.05)].index)
            
            # Store iteration data
            iteration_data = {
                'id': iteration_id,
                'name': iteration_name,
                'timestamp': datetime.now().isoformat(),
                'formula': formula,
                'contrast': contrast,
                'contrast_name': f"{contrast[0]}_{contrast[1]}_vs_{contrast[2]}",
                'parameters': parameters,
                'n_significant_genes': len(significant_genes),
                'n_upregulated': analysis_stats.get('n_upregulated', 0),
                'n_downregulated': analysis_stats.get('n_downregulated', 0),
                'top_genes': {
                    'upregulated': analysis_stats.get('top_upregulated', [])[:10],
                    'downregulated': analysis_stats.get('top_downregulated', [])[:10]
                },
                'significant_genes': list(significant_genes),
                'results_summary': {
                    'n_genes_tested': analysis_stats.get('n_genes_tested', 0),
                    'alpha': parameters.get('alpha', 0.05),
                    'lfc_threshold': parameters.get('lfc_threshold', 0.0)
                }
            }
            
            # Store in tracker
            self.iterations[modality_name][iteration_id] = iteration_data
            
            self.logger.info(
                f"Tracked iteration '{iteration_name}' for modality '{modality_name}': "
                f"{len(significant_genes)} significant genes"
            )
            
            return f"{modality_name}_{iteration_id}"
            
        except Exception as e:
            self.logger.error(f"Error tracking iteration: {e}")
            raise

    def compare_iterations(
        self,
        modality_name: str,
        iteration_1: Union[int, str],
        iteration_2: Union[int, str]
    ) -> Dict[str, Any]:
        """
        Compare results between two iterations.
        
        Args:
            modality_name: Name of the modality
            iteration_1: First iteration ID or name
            iteration_2: Second iteration ID or name
            
        Returns:
            Dict[str, Any]: Comparison results
        """
        try:
            if modality_name not in self.iterations:
                raise ValueError(f"No iterations tracked for modality '{modality_name}'")
            
            modality_iterations = self.iterations[modality_name]
            
            # Find iterations
            iter1_data = self._find_iteration(modality_iterations, iteration_1)
            iter2_data = self._find_iteration(modality_iterations, iteration_2)
            
            if not iter1_data or not iter2_data:
                available = list(modality_iterations.keys())
                raise ValueError(f"Iteration not found. Available: {available}")
            
            # Get significant gene sets
            sig1 = set(iter1_data['significant_genes'])
            sig2 = set(iter2_data['significant_genes'])
            
            # Calculate overlaps
            overlap = sig1 & sig2
            unique_to_1 = sig1 - sig2
            unique_to_2 = sig2 - sig1
            
            # Calculate overlap statistics
            total_unique = len(sig1 | sig2)
            overlap_percent = len(overlap) / max(len(sig1), len(sig2)) * 100 if max(len(sig1), len(sig2)) > 0 else 0
            
            # Compare top genes
            top1_up = set(iter1_data['top_genes']['upregulated'][:5])
            top2_up = set(iter2_data['top_genes']['upregulated'][:5])
            top_overlap = len(top1_up & top2_up)
            
            # Prepare comparison result
            comparison = {
                'modality_name': modality_name,
                'iteration_1': {
                    'id': iter1_data['id'],
                    'name': iter1_data['name'],
                    'formula': iter1_data['formula'],
                    'n_significant': len(sig1)
                },
                'iteration_2': {
                    'id': iter2_data['id'],
                    'name': iter2_data['name'], 
                    'formula': iter2_data['formula'],
                    'n_significant': len(sig2)
                },
                'overlap_stats': {
                    'overlapping_genes': len(overlap),
                    'unique_to_1': len(unique_to_1),
                    'unique_to_2': len(unique_to_2),
                    'total_unique_genes': total_unique,
                    'overlap_percentage': overlap_percent
                },
                'gene_lists': {
                    'overlap': list(overlap),
                    'unique_to_1': list(unique_to_1)[:20],  # Limit for response size
                    'unique_to_2': list(unique_to_2)[:20]
                },
                'top_gene_overlap': top_overlap,
                'comparison_timestamp': datetime.now().isoformat()
            }
            
            # Cache comparison
            cache_key = f"{modality_name}_{iter1_data['id']}_vs_{iter2_data['id']}"
            self.comparison_cache[cache_key] = comparison
            
            self.logger.info(
                f"Compared iterations {iter1_data['name']} vs {iter2_data['name']}: "
                f"{len(overlap)} overlap, {len(unique_to_1)}+{len(unique_to_2)} unique"
            )
            
            return comparison
            
        except Exception as e:
            self.logger.error(f"Error comparing iterations: {e}")
            raise

    def get_iteration_summary(self, modality_name: str) -> str:
        """
        Generate summary of all iterations for a modality.
        
        Args:
            modality_name: Name of the modality
            
        Returns:
            str: Formatted summary of iterations
        """
        try:
            if modality_name not in self.iterations:
                return f"No iterations tracked for modality '{modality_name}'"
            
            modality_iterations = self.iterations[modality_name]
            
            if not modality_iterations:
                return f"No iterations found for modality '{modality_name}'"
            
            summary = f"Iteration Summary for '{modality_name}':\n\n"
            summary += f"Total iterations: {len(modality_iterations)}\n\n"
            
            # Sort by iteration ID
            sorted_iterations = sorted(modality_iterations.items(), key=lambda x: x[1]['id'])
            
            for iter_id, iter_data in sorted_iterations:
                summary += f"**{iter_data['name']}** (ID: {iter_id})\n"
                summary += f"• Formula: {iter_data['formula']}\n"
                summary += f"• Contrast: {iter_data['contrast'][1]} vs {iter_data['contrast'][2]}\n"
                summary += f"• Significant genes: {iter_data['n_significant_genes']:,}\n"
                summary += f"• Timestamp: {iter_data['timestamp']}\n\n"
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Error generating iteration summary: {e}")
            return f"Error generating summary: {str(e)}"

    def suggest_next_iteration(
        self,
        modality_name: str,
        current_results: Optional[pd.DataFrame] = None
    ) -> List[str]:
        """
        Suggest potential next analyses based on current results.
        
        Args:
            modality_name: Name of the modality
            current_results: Current DE results DataFrame
            
        Returns:
            List[str]: List of suggestions for next iterations
        """
        try:
            suggestions = []
            
            if modality_name not in self.iterations:
                suggestions.append("Run initial DE analysis to establish baseline")
                return suggestions
            
            modality_iterations = self.iterations[modality_name]
            
            if not modality_iterations:
                suggestions.append("No previous iterations - run initial analysis")
                return suggestions
            
            # Get latest iteration
            latest_iter = max(modality_iterations.values(), key=lambda x: x['id'])
            
            # Analyze latest results
            n_significant = latest_iter['n_significant_genes']
            formula = latest_iter['formula']
            
            # Suggest based on current state
            if n_significant == 0:
                suggestions.extend([
                    "No significant genes found - try simpler model or check data quality",
                    "Consider lowering significance threshold or LFC threshold",
                    "Verify experimental design has sufficient power"
                ])
            elif n_significant < 10:
                suggestions.extend([
                    "Few significant genes - consider simpler model for more power",
                    "Try removing least important covariates",
                    "Check if batch effects are over-corrected"
                ])
            elif n_significant > 5000:
                suggestions.extend([
                    "Many significant genes - consider more stringent thresholds",
                    "Add covariates to control for confounders",
                    "Apply stricter log fold change threshold"
                ])
            else:
                suggestions.extend([
                    "Good number of significant genes - try alternative models for validation",
                    "Consider interaction terms if biologically justified",
                    "Test robustness with different filtering criteria"
                ])
            
            # Formula-specific suggestions
            if '+' not in formula:
                suggestions.append("Try adding batch correction or covariates")
            elif len(formula.split('+')) >= 4:
                suggestions.append("Consider simplifying model - may be overfitted")
            else:
                suggestions.append("Try interaction terms if condition effects vary by batch")
            
            return suggestions[:4]  # Limit to top 4 suggestions
            
        except Exception as e:
            self.logger.error(f"Error suggesting next iteration: {e}")
            return [f"Error generating suggestions: {str(e)}"]

    def export_iteration_history(
        self,
        modality_name: str,
        output_path: Optional[str] = None
    ) -> str:
        """
        Export iteration history for a modality.
        
        Args:
            modality_name: Name of the modality
            output_path: Optional output file path
            
        Returns:
            str: Path to exported file
        """
        try:
            if modality_name not in self.iterations:
                raise ValueError(f"No iterations tracked for modality '{modality_name}'")
            
            # Prepare export data
            export_data = {
                'modality_name': modality_name,
                'export_timestamp': datetime.now().isoformat(),
                'total_iterations': len(self.iterations[modality_name]),
                'iterations': self.iterations[modality_name],
                'comparisons': {k: v for k, v in self.comparison_cache.items() 
                               if k.startswith(modality_name)}
            }
            
            # Set output path
            if output_path is None:
                output_path = f"{modality_name}_iteration_history.json"
            
            # Write to file
            with open(output_path, 'w') as f:
                json.dump(export_data, f, indent=2, default=str)
            
            self.logger.info(f"Exported iteration history for '{modality_name}' to {output_path}")
            return output_path
            
        except Exception as e:
            self.logger.error(f"Error exporting iteration history: {e}")
            raise

    def import_iteration_history(
        self,
        import_path: str
    ) -> str:
        """
        Import iteration history from file.
        
        Args:
            import_path: Path to iteration history file
            
        Returns:
            str: Imported modality name
        """
        try:
            with open(import_path, 'r') as f:
                import_data = json.load(f)
            
            modality_name = import_data['modality_name']
            
            # Import iterations
            self.iterations[modality_name] = import_data['iterations']
            
            # Import comparisons
            for comp_key, comp_data in import_data.get('comparisons', {}).items():
                self.comparison_cache[comp_key] = comp_data
            
            # Update iteration counter
            if import_data['iterations']:
                max_id = max(int(iter_data['id']) for iter_data in import_data['iterations'].values())
                self.current_iteration_ids[modality_name] = max_id
            else:
                self.current_iteration_ids[modality_name] = 0
            
            self.logger.info(
                f"Imported {import_data['total_iterations']} iterations for '{modality_name}'"
            )
            
            return modality_name
            
        except Exception as e:
            self.logger.error(f"Error importing iteration history: {e}")
            raise

    def get_best_iteration(
        self,
        modality_name: str,
        metric: str = 'n_significant_genes'
    ) -> Optional[Dict[str, Any]]:
        """
        Get the best iteration based on specified metric.
        
        Args:
            modality_name: Name of the modality
            metric: Metric to optimize ('n_significant_genes', 'balance', etc.)
            
        Returns:
            Optional[Dict[str, Any]]: Best iteration data or None
        """
        try:
            if modality_name not in self.iterations or not self.iterations[modality_name]:
                return None
            
            iterations = list(self.iterations[modality_name].values())
            
            if metric == 'n_significant_genes':
                # Find iteration with optimal number of significant genes (not too few, not too many)
                scores = []
                for iter_data in iterations:
                    n_sig = iter_data['n_significant_genes']
                    # Score based on reasonable range (50-2000 genes)
                    if 50 <= n_sig <= 2000:
                        score = 1.0  # Optimal range
                    elif n_sig < 50:
                        score = n_sig / 50.0  # Penalize too few
                    else:
                        score = 2000.0 / n_sig  # Penalize too many
                    scores.append(score)
                
                best_idx = np.argmax(scores)
                return iterations[best_idx]
            
            elif metric == 'latest':
                return max(iterations, key=lambda x: x['id'])
            
            elif metric == 'simplest':
                # Find iteration with simplest formula (fewest terms)
                return min(iterations, key=lambda x: len(x['formula'].split('+')))
            
            else:
                # Default to most significant genes
                return max(iterations, key=lambda x: x['n_significant_genes'])
                
        except Exception as e:
            self.logger.error(f"Error finding best iteration: {e}")
            return None

    def cleanup_iterations(
        self,
        modality_name: str,
        keep_last_n: int = 5
    ) -> int:
        """
        Clean up old iterations to save memory.
        
        Args:
            modality_name: Name of the modality
            keep_last_n: Number of recent iterations to keep
            
        Returns:
            int: Number of iterations removed
        """
        try:
            if modality_name not in self.iterations:
                return 0
            
            modality_iterations = self.iterations[modality_name]
            
            if len(modality_iterations) <= keep_last_n:
                return 0
            
            # Sort by iteration ID and keep only the last N
            sorted_iterations = sorted(modality_iterations.items(), key=lambda x: x[1]['id'])
            to_remove = sorted_iterations[:-keep_last_n]
            
            # Remove old iterations
            removed_count = 0
            for iter_id, _ in to_remove:
                del modality_iterations[iter_id]
                removed_count += 1
            
            # Clean up related comparisons
            cache_keys_to_remove = []
            for cache_key in self.comparison_cache:
                if any(str(removed_id) in cache_key for removed_id, _ in to_remove):
                    cache_keys_to_remove.append(cache_key)
            
            for cache_key in cache_keys_to_remove:
                del self.comparison_cache[cache_key]
            
            self.logger.info(f"Cleaned up {removed_count} old iterations for '{modality_name}'")
            return removed_count
            
        except Exception as e:
            self.logger.error(f"Error cleaning up iterations: {e}")
            return 0

    def _find_iteration(
        self,
        modality_iterations: Dict[int, Dict[str, Any]],
        identifier: Union[int, str]
    ) -> Optional[Dict[str, Any]]:
        """Find iteration by ID or name."""
        
        # Try by ID first
        if isinstance(identifier, int) and identifier in modality_iterations:
            return modality_iterations[identifier]
        
        # Try by name
        for iter_data in modality_iterations.values():
            if iter_data['name'] == identifier:
                return iter_data
        
        return None

    def generate_comparison_report(
        self,
        modality_name: str,
        iteration_1: Union[int, str],
        iteration_2: Union[int, str],
        include_gene_lists: bool = True,
        max_genes_per_list: int = 20
    ) -> str:
        """
        Generate detailed comparison report between iterations.
        
        Args:
            modality_name: Name of the modality
            iteration_1: First iteration ID or name
            iteration_2: Second iteration ID or name
            include_gene_lists: Whether to include gene lists
            max_genes_per_list: Maximum genes per list in report
            
        Returns:
            str: Formatted comparison report
        """
        try:
            comparison = self.compare_iterations(modality_name, iteration_1, iteration_2)
            
            report = f"# DE Iteration Comparison Report\n\n"
            report += f"**Modality**: {modality_name}\n"
            report += f"**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
            
            # Iteration details
            report += f"## Iterations Compared\n\n"
            report += f"**Iteration 1**: {comparison['iteration_1']['name']}\n"
            report += f"• Formula: `{comparison['iteration_1']['formula']}`\n"
            report += f"• Significant genes: {comparison['iteration_1']['n_significant']:,}\n\n"
            
            report += f"**Iteration 2**: {comparison['iteration_2']['name']}\n"
            report += f"• Formula: `{comparison['iteration_2']['formula']}`\n"
            report += f"• Significant genes: {comparison['iteration_2']['n_significant']:,}\n\n"
            
            # Overlap statistics
            overlap_stats = comparison['overlap_stats']
            report += f"## Overlap Analysis\n\n"
            report += f"• **Overlapping genes**: {overlap_stats['overlapping_genes']:,} ({overlap_stats['overlap_percentage']:.1f}%)\n"
            report += f"• **Unique to iteration 1**: {overlap_stats['unique_to_1']:,}\n"
            report += f"• **Unique to iteration 2**: {overlap_stats['unique_to_2']:,}\n"
            report += f"• **Total unique genes**: {overlap_stats['total_unique_genes']:,}\n\n"
            
            # Interpretation
            report += f"## Interpretation\n\n"
            if overlap_stats['overlap_percentage'] > 70:
                report += f"**High overlap** ({overlap_stats['overlap_percentage']:.1f}%) indicates formulas capture similar biology.\n"
            elif overlap_stats['overlap_percentage'] > 40:
                report += f"**Moderate overlap** ({overlap_stats['overlap_percentage']:.1f}%) suggests some formula-specific effects.\n"
            else:
                report += f"**Low overlap** ({overlap_stats['overlap_percentage']:.1f}%) indicates formulas capture different biological processes.\n"
            
            # Gene lists
            if include_gene_lists:
                gene_lists = comparison['gene_lists']
                
                if gene_lists['overlap']:
                    report += f"\n### Overlapping Genes\n"
                    overlap_genes = gene_lists['overlap'][:max_genes_per_list]
                    report += f"{', '.join(overlap_genes)}\n"
                    
                    if len(gene_lists['overlap']) > max_genes_per_list:
                        report += f"... and {len(gene_lists['overlap']) - max_genes_per_list} more\n"
                
                if gene_lists['unique_to_1']:
                    report += f"\n### Unique to {comparison['iteration_1']['name']}\n"
                    unique1_genes = gene_lists['unique_to_1'][:max_genes_per_list]
                    report += f"{', '.join(unique1_genes)}\n"
                    
                    if len(gene_lists['unique_to_1']) > max_genes_per_list:
                        report += f"... and more\n"
                
                if gene_lists['unique_to_2']:
                    report += f"\n### Unique to {comparison['iteration_2']['name']}\n"
                    unique2_genes = gene_lists['unique_to_2'][:max_genes_per_list]
                    report += f"{', '.join(unique2_genes)}\n"
                    
                    if len(gene_lists['unique_to_2']) > max_genes_per_list:
                        report += f"... and more\n"
            
            return report
            
        except Exception as e:
            self.logger.error(f"Error generating comparison report: {e}")
            return f"Error generating report: {str(e)}"

    def get_iteration_by_name(
        self,
        modality_name: str,
        iteration_name: str
    ) -> Optional[Dict[str, Any]]:
        """
        Get iteration data by name.
        
        Args:
            modality_name: Name of the modality
            iteration_name: Name of the iteration
            
        Returns:
            Optional[Dict[str, Any]]: Iteration data or None if not found
        """
        if modality_name not in self.iterations:
            return None
        
        return self._find_iteration(self.iterations[modality_name], iteration_name)

    def list_iterations(self, modality_name: str) -> List[str]:
        """
        List all iteration names for a modality.
        
        Args:
            modality_name: Name of the modality
            
        Returns:
            List[str]: List of iteration names
        """
        if modality_name not in self.iterations:
            return []
        
        return [iter_data['name'] for iter_data in self.iterations[modality_name].values()]

    def clear_iterations(self, modality_name: str) -> int:
        """
        Clear all iterations for a modality.
        
        Args:
            modality_name: Name of the modality
            
        Returns:
            int: Number of iterations cleared
        """
        if modality_name not in self.iterations:
            return 0
        
        count = len(self.iterations[modality_name])
        self.iterations[modality_name] = {}
        self.current_iteration_ids[modality_name] = 0
        
        # Clear related comparisons
        cache_keys_to_remove = [k for k in self.comparison_cache.keys() if k.startswith(modality_name)]
        for cache_key in cache_keys_to_remove:
            del self.comparison_cache[cache_key]
        
        self.logger.info(f"Cleared {count} iterations for '{modality_name}'")
        return count


class IterationComparisonResult:
    """
    Structured result for iteration comparisons.
    
    Provides formatted access to comparison results with helper methods
    for agent response generation.
    """
    
    def __init__(self, comparison_data: Dict[str, Any]):
        """Initialize from comparison data dictionary."""
        self.data = comparison_data
        
    @property
    def overlap_genes(self) -> List[str]:
        """Get list of overlapping genes."""
        return self.data.get('gene_lists', {}).get('overlap', [])
    
    @property
    def unique_to_first(self) -> List[str]:
        """Get genes unique to first iteration."""
        return self.data.get('gene_lists', {}).get('unique_to_1', [])
    
    @property
    def unique_to_second(self) -> List[str]:
        """Get genes unique to second iteration."""
        return self.data.get('gene_lists', {}).get('unique_to_2', [])
    
    @property
    def overlap_percentage(self) -> float:
        """Get overlap percentage."""
        return self.data.get('overlap_stats', {}).get('overlap_percentage', 0.0)
    
    def format_summary(self) -> str:
        """Format a concise summary for agent responses."""
        try:
            iter1 = self.data['iteration_1']
            iter2 = self.data['iteration_2']
            stats = self.data['overlap_stats']
            
            summary = f"**Comparison: {iter1['name']} vs {iter2['name']}**\n\n"
            summary += f"• Iteration 1: {iter1['n_significant']:,} significant genes\n"
            summary += f"• Iteration 2: {iter2['n_significant']:,} significant genes\n"
            summary += f"• Overlap: {stats['overlapping_genes']:,} genes ({stats['overlap_percentage']:.1f}%)\n"
            summary += f"• Unique genes: {stats['unique_to_1']:,} + {stats['unique_to_2']:,}\n"
            
            return summary
            
        except Exception as e:
            return f"Error formatting summary: {str(e)}"
    
    def get_interpretation(self) -> str:
        """Get biological interpretation of the comparison."""
        overlap_pct = self.overlap_percentage
        
        if overlap_pct > 80:
            return "High similarity - formulas capture similar biological effects"
        elif overlap_pct > 50:
            return "Moderate similarity - some consistent effects with formula-specific differences"
        elif overlap_pct > 20:
            return "Low similarity - formulas capture substantially different biology"
        else:
            return "Very low similarity - formulas may be detecting different biological processes"
