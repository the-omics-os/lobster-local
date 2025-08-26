"""
Machine Learning Expert Agent for ML model training with biological data.

This agent focuses on preparing biological data for machine learning tasks,
providing ML-specific tools and workflows for transcriptomics and proteomics data
using the modular DataManagerV2 system.
"""

from typing import List, Dict, Any, Optional, Union, Tuple
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langchain_aws import ChatBedrockConverse

from datetime import date
import numpy as np
import pandas as pd

from lobster.agents.state import MachineLearningExpertState
from lobster.config.settings import get_settings
from lobster.core.data_manager_v2 import DataManagerV2
from lobster.utils.logger import get_logger

logger = get_logger(__name__)


class MachineLearningError(Exception):
    """Base exception for machine learning operations."""
    pass


class ModalityNotFoundError(MachineLearningError):
    """Raised when requested modality doesn't exist."""
    pass


class DataPreparationError(MachineLearningError):
    """Raised when data preparation fails."""
    pass


def machine_learning_expert(
    data_manager: DataManagerV2, 
    callback_handler=None, 
    agent_name: str = "machine_learning_expert_agent",
    handoff_tools: List = None
):  
    """Create machine learning expert agent using DataManagerV2."""
    
    settings = get_settings()
    model_params = settings.get_agent_llm_params('machine_learning_expert')
    llm = ChatBedrockConverse(**model_params)
    
    if callback_handler and hasattr(llm, 'with_config'):
        llm = llm.with_config(callbacks=[callback_handler])
    
    # Store ML-specific results and metadata
    ml_results = {"summary": "", "details": {}}
    
    # -------------------------
    # DATA STATUS AND INSPECTION TOOLS
    # -------------------------
    @tool
    def check_ml_ready_modalities(modality_type: str = "all") -> str:
        """
        Check which modalities are ready for machine learning tasks.
        
        Args:
            modality_type: Filter by type ("transcriptomics", "proteomics", "all")
            
        Returns:
            str: Summary of ML-ready modalities with their characteristics
        """
        try:
            modalities = data_manager.list_modalities()
            if not modalities:
                return "No modalities loaded. Please ask the data expert to load biological datasets first."
            
            ml_ready_modalities = []
            
            for mod_name in modalities:
                adata = data_manager.get_modality(mod_name)
                mod_type = data_manager._detect_modality_type(mod_name)
                
                # Check if modality matches requested type
                if modality_type != "all":
                    if modality_type == "transcriptomics" and "rna" not in mod_type.lower():
                        continue
                    elif modality_type == "proteomics" and "proteomics" not in mod_type.lower():
                        continue
                
                # Assess ML readiness
                ml_info = {
                    "name": mod_name,
                    "type": mod_type,
                    "shape": adata.shape,
                    "has_labels": any(col in adata.obs.columns for col in 
                                    ['condition', 'cell_type', 'treatment', 'group', 'label', 'class']),
                    "normalized": 'normalized' in mod_name or adata.X.max() <= 100,
                    "filtered": 'filtered' in mod_name,
                    "clustered": 'leiden' in adata.obs.columns or 'cluster' in adata.obs.columns
                }
                
                # Check for batch information
                ml_info["has_batch"] = any(col in adata.obs.columns for col in ['batch', 'sample', 'donor'])
                
                # Check data sparsity
                if hasattr(adata.X, 'nnz'):
                    ml_info["sparsity"] = 1 - (adata.X.nnz / (adata.X.shape[0] * adata.X.shape[1]))
                else:
                    ml_info["sparsity"] = np.mean(adata.X == 0) if adata.X.size > 0 else 0
                
                ml_ready_modalities.append(ml_info)
            
            # Format response
            response = f"Found {len(ml_ready_modalities)} modalities suitable for ML:\n\n"
            
            for info in ml_ready_modalities:
                response += f"**{info['name']}** ({info['type']}):\n"
                response += f"  - Shape: {info['shape'][0]} samples × {info['shape'][1]} features\n"
                response += f"  - Labels available: {'✓' if info['has_labels'] else '✗'}\n"
                response += f"  - Normalized: {'✓' if info['normalized'] else '✗'}\n"
                response += f"  - Filtered: {'✓' if info['filtered'] else '✗'}\n"
                response += f"  - Sparsity: {info['sparsity']:.1%}\n"
                
                if info['has_batch']:
                    response += f"  - Batch info: ✓ (consider batch correction)\n"
                
                # ML recommendations
                if info['shape'][0] < 50:
                    response += f"  - ⚠️ Small sample size - consider regularization\n"
                elif info['shape'][0] > 10000:
                    response += f"  - 📊 Large dataset - suitable for deep learning\n"
                
                response += "\n"
            
            ml_results["details"]["ml_ready_check"] = response
            return response
            
        except Exception as e:
            logger.error(f"Error checking ML-ready modalities: {e}")
            return f"Error checking ML-ready modalities: {str(e)}"

    @tool
    def prepare_ml_features(
        modality_name: str,
        feature_selection: str = "highly_variable",
        n_features: int = 2000,
        scale: bool = True,
        handle_zeros: str = "keep",
        save_result: bool = True
    ) -> str:
        """
        Prepare features from biological data for machine learning.
        
        Args:
            modality_name: Name of the modality to process
            feature_selection: Method for feature selection 
                              ("highly_variable", "pca", "all", "marker_genes")
            n_features: Number of features to select
            scale: Whether to scale features (z-score normalization)
            handle_zeros: How to handle zero values ("keep", "remove", "impute")
            save_result: Whether to save the prepared modality
            
        Returns:
            str: Summary of feature preparation
        """
        try:
            # Validate modality exists
            if modality_name not in data_manager.list_modalities():
                raise ModalityNotFoundError(f"Modality '{modality_name}' not found. Available: {data_manager.list_modalities()}")
            
            # Get the modality
            adata = data_manager.get_modality(modality_name)
            logger.info(f"Preparing ML features from '{modality_name}': {adata.shape}")
            
            # Copy to avoid modifying original
            import anndata
            adata_ml = adata.copy()
            
            # Feature selection
            if feature_selection == "highly_variable":
                # Use scanpy's highly variable genes if available
                try:
                    import scanpy as sc
                    if 'highly_variable' not in adata_ml.var.columns:
                        sc.pp.highly_variable_genes(adata_ml, n_top_genes=n_features)
                    adata_ml = adata_ml[:, adata_ml.var.highly_variable]
                except ImportError:
                    # Fallback: variance-based selection
                    variances = np.var(adata_ml.X, axis=0)
                    if hasattr(variances, 'A1'):
                        variances = variances.A1
                    top_var_indices = np.argsort(variances)[-n_features:]
                    adata_ml = adata_ml[:, top_var_indices]
                    
            elif feature_selection == "pca":
                # PCA-based feature reduction
                try:
                    import scanpy as sc
                    if 'X_pca' not in adata_ml.obsm:
                        sc.pp.pca(adata_ml, n_comps=min(n_features, adata_ml.shape[1]-1))
                    # Store PCA as main data for ML
                    pca_data = adata_ml.obsm['X_pca'][:, :n_features]
                    # Create new AnnData with PCA features
                    adata_ml = anndata.AnnData(
                        X=pca_data,
                        obs=adata_ml.obs.copy(),
                        var=pd.DataFrame(index=[f'PC{i+1}' for i in range(pca_data.shape[1])])
                    )
                except ImportError:
                    # Fallback: simple dimensionality reduction
                    logger.warning("Scanpy not available, using variance-based selection instead of PCA")
                    variances = np.var(adata_ml.X, axis=0)
                    if hasattr(variances, 'A1'):
                        variances = variances.A1
                    top_var_indices = np.argsort(variances)[-n_features:]
                    adata_ml = adata_ml[:, top_var_indices]
                    
            elif feature_selection == "marker_genes":
                # Use known marker genes if available
                if 'rank_genes_groups' in adata_ml.uns:
                    # Extract top marker genes from each group
                    marker_genes = []
                    groups_key = adata_ml.uns['rank_genes_groups']['params']['groupby']
                    for group in adata_ml.obs[groups_key].unique():
                        genes = adata_ml.uns['rank_genes_groups']['names'][group][:n_features//10]
                        marker_genes.extend(genes)
                    marker_genes = list(set(marker_genes))[:n_features]
                    # Filter to these genes
                    available_markers = [g for g in marker_genes if g in adata_ml.var_names]
                    if available_markers:
                        adata_ml = adata_ml[:, available_markers]
                    else:
                        logger.warning("No marker genes found, using highly variable genes")
                else:
                    logger.warning("No marker genes found, using highly variable genes")
                    
            # Handle zeros
            if handle_zeros == "remove":
                # Remove features with too many zeros
                zero_prop = np.mean(adata_ml.X == 0, axis=0)
                if hasattr(zero_prop, 'A1'):
                    zero_prop = zero_prop.A1
                keep_features = zero_prop < 0.9  # Keep features with <90% zeros
                adata_ml = adata_ml[:, keep_features]
                
            elif handle_zeros == "impute":
                # Simple mean imputation for zeros
                if hasattr(adata_ml.X, 'toarray'):
                    X_dense = adata_ml.X.toarray()
                else:
                    X_dense = adata_ml.X.copy()
                    
                for j in range(X_dense.shape[1]):
                    col = X_dense[:, j]
                    if np.any(col == 0):
                        non_zero_mean = np.mean(col[col != 0]) if np.any(col != 0) else 0
                        col[col == 0] = non_zero_mean
                        X_dense[:, j] = col
                        
                adata_ml.X = X_dense
            
            # Scale features
            if scale:
                # Z-score normalization
                try:
                    import scanpy as sc
                    sc.pp.scale(adata_ml, zero_center=True)
                except ImportError:
                    # Manual scaling
                    if hasattr(adata_ml.X, 'toarray'):
                        X_dense = adata_ml.X.toarray()
                    else:
                        X_dense = adata_ml.X.copy()
                    
                    # Z-score normalization
                    X_dense = (X_dense - np.mean(X_dense, axis=0)) / (np.std(X_dense, axis=0) + 1e-8)
                    adata_ml.X = X_dense
            
            # Add ML metadata
            adata_ml.uns['ml_preprocessing'] = {
                'source_modality': modality_name,
                'feature_selection': feature_selection,
                'n_features_selected': adata_ml.shape[1],
                'scaled': scale,
                'zero_handling': handle_zeros,
                'original_shape': adata.shape
            }
            
            # Save as new modality
            ml_modality_name = f"{modality_name}_ml_features"
            data_manager.modalities[ml_modality_name] = adata_ml
            
            # Save to file if requested
            if save_result:
                save_path = f"{modality_name}_ml_features.h5ad"
                data_manager.save_modality(ml_modality_name, save_path)
            
            # Log the operation
            data_manager.log_tool_usage(
                tool_name="prepare_ml_features",
                parameters={
                    "modality_name": modality_name,
                    "feature_selection": feature_selection,
                    "n_features": n_features,
                    "scale": scale,
                    "handle_zeros": handle_zeros
                },
                description=f"Prepared ML features: {adata_ml.shape}"
            )
            
            response = f"""Successfully prepared features for machine learning from '{modality_name}'!

📊 **Feature Preparation Results:**
- Original shape: {adata.shape[0]} samples × {adata.shape[1]} features
- ML-ready shape: {adata_ml.shape[0]} samples × {adata_ml.shape[1]} features
- Feature selection: {feature_selection}
- Features scaled: {'✓' if scale else '✗'}
- Zero handling: {handle_zeros}

🔬 **Feature Statistics:**
- Sparsity: {np.mean(adata_ml.X == 0):.1%} zeros
- Value range: [{np.min(adata_ml.X):.2f}, {np.max(adata_ml.X):.2f}]

💾 **New modality created**: '{ml_modality_name}'"""

            if save_result:
                response += f"\n💾 **Saved to**: {save_path}"
                
            response += "\n\nNext steps: create train/test splits or export for ML frameworks."
            
            ml_results["details"]["feature_preparation"] = response
            return response
            
        except (ModalityNotFoundError, DataPreparationError) as e:
            logger.error(f"Error preparing ML features: {e}")
            return f"Error preparing ML features: {str(e)}"
        except Exception as e:
            logger.error(f"Unexpected error in ML feature preparation: {e}")
            return f"Unexpected error: {str(e)}"

    @tool
    def create_ml_splits(
        modality_name: str,
        test_size: float = 0.2,
        validation_size: float = 0.1,
        stratify_by: Optional[str] = None,
        random_state: int = 42,
        save_result: bool = True
    ) -> str:
        """
        Create train/test/validation splits for machine learning.
        
        Args:
            modality_name: Name of the modality to split
            test_size: Proportion of data for testing (0-1)
            validation_size: Proportion of training data for validation (0-1)
            stratify_by: Column name for stratified splitting (e.g., 'cell_type', 'condition')
            random_state: Random seed for reproducibility
            save_result: Whether to save the splits
            
        Returns:
            str: Summary of data splitting
        """
        try:
            # Validate modality exists
            if modality_name not in data_manager.list_modalities():
                raise ModalityNotFoundError(f"Modality '{modality_name}' not found. Available: {data_manager.list_modalities()}")
            
            # Get the modality
            adata = data_manager.get_modality(modality_name)
            
            # Check stratification column
            if stratify_by and stratify_by not in adata.obs.columns:
                available_cols = [col for col in adata.obs.columns 
                                if col in ['condition', 'cell_type', 'treatment', 'group', 'leiden']]
                return f"Stratification column '{stratify_by}' not found. Available: {available_cols}"
            
            # Create splits
            from sklearn.model_selection import train_test_split
            
            n_samples = adata.n_obs
            indices = np.arange(n_samples)
            
            # Get stratification labels if needed
            stratify_labels = adata.obs[stratify_by].values if stratify_by else None
            
            # First split: train+val vs test
            train_val_idx, test_idx = train_test_split(
                indices,
                test_size=test_size,
                random_state=random_state,
                stratify=stratify_labels
            )
            
            # Second split: train vs val
            if validation_size > 0:
                # Adjust stratify labels for train_val subset
                stratify_labels_tv = stratify_labels[train_val_idx] if stratify_labels is not None else None
                
                train_idx, val_idx = train_test_split(
                    train_val_idx,
                    test_size=validation_size,
                    random_state=random_state,
                    stratify=stratify_labels_tv
                )
            else:
                train_idx = train_val_idx
                val_idx = np.array([])
            
            # Create split annotations
            adata.obs['ml_split'] = 'unassigned'
            adata.obs.loc[adata.obs_names[train_idx], 'ml_split'] = 'train'
            adata.obs.loc[adata.obs_names[test_idx], 'ml_split'] = 'test'
            if len(val_idx) > 0:
                adata.obs.loc[adata.obs_names[val_idx], 'ml_split'] = 'validation'
            
            # Create separate modalities for each split
            adata_train = adata[train_idx, :].copy()
            adata_test = adata[test_idx, :].copy()
            
            # Save splits as new modalities
            train_modality = f"{modality_name}_train"
            test_modality = f"{modality_name}_test"
            
            data_manager.modalities[train_modality] = adata_train
            data_manager.modalities[test_modality] = adata_test
            
            if len(val_idx) > 0:
                adata_val = adata[val_idx, :].copy()
                val_modality = f"{modality_name}_validation"
                data_manager.modalities[val_modality] = adata_val
            
            # Update original modality with split info
            data_manager.modalities[modality_name] = adata
            
            # Save if requested
            if save_result:
                data_manager.save_modality(train_modality, f"{train_modality}.h5ad")
                data_manager.save_modality(test_modality, f"{test_modality}.h5ad")
                if len(val_idx) > 0:
                    data_manager.save_modality(val_modality, f"{val_modality}.h5ad")
            
            # Log the operation
            data_manager.log_tool_usage(
                tool_name="create_ml_splits",
                parameters={
                    "modality_name": modality_name,
                    "test_size": test_size,
                    "validation_size": validation_size,
                    "stratify_by": stratify_by,
                    "random_state": random_state
                },
                description=f"Created ML splits: train={len(train_idx)}, test={len(test_idx)}, val={len(val_idx)}"
            )
            
            # Generate split statistics
            split_stats = {
                'train': len(train_idx),
                'test': len(test_idx),
                'validation': len(val_idx) if len(val_idx) > 0 else 0
            }
            
            response = f"""Successfully created ML data splits for '{modality_name}'!

📊 **Split Statistics:**
- Training set: {split_stats['train']} samples ({split_stats['train']/n_samples*100:.1f}%)
- Test set: {split_stats['test']} samples ({split_stats['test']/n_samples*100:.1f}%)"""
            
            if split_stats['validation'] > 0:
                response += f"\n- Validation set: {split_stats['validation']} samples ({split_stats['validation']/n_samples*100:.1f}%)"
            
            if stratify_by:
                response += f"\n- Stratified by: {stratify_by}"
                
                # Show class distribution
                response += f"\n\n📈 **Class Distribution in Splits:**"
                for split_name in ['train', 'test', 'validation']:
                    if split_stats.get(split_name, 0) > 0:
                        split_data = adata[adata.obs['ml_split'] == split_name]
                        class_counts = split_data.obs[stratify_by].value_counts()
                        response += f"\n\n{split_name.capitalize()}:"
                        for class_name, count in class_counts.head(5).items():
                            response += f"\n  - {class_name}: {count} ({count/split_stats[split_name]*100:.1f}%)"
            
            response += f"\n\n💾 **New modalities created**:"
            response += f"\n- '{train_modality}' (training data)"
            response += f"\n- '{test_modality}' (test data)"
            if split_stats['validation'] > 0:
                response += f"\n- '{val_modality}' (validation data)"
                
            response += f"\n\nOriginal modality updated with 'ml_split' column for tracking."
            
            ml_results["details"]["data_splitting"] = response
            return response
            
        except Exception as e:
            logger.error(f"Error creating ML splits: {e}")
            return f"Error creating ML splits: {str(e)}"

    @tool
    def export_for_ml_framework(
        modality_name: str,
        format: str = "numpy",
        include_labels: bool = True,
        label_column: Optional[str] = None,
        output_dir: str = "ml_exports"
    ) -> str:
        """
        Export biological data in formats suitable for ML frameworks.
        
        Args:
            modality_name: Name of the modality to export
            format: Export format ("numpy", "csv", "pytorch", "tensorflow")
            include_labels: Whether to export labels/targets
            label_column: Column name containing labels
            output_dir: Directory for exported files
            
        Returns:
            str: Summary of exported files
        """
        try:
            # Validate modality exists
            if modality_name not in data_manager.list_modalities():
                raise ModalityNotFoundError(f"Modality '{modality_name}' not found. Available: {data_manager.list_modalities()}")
            
            # Get the modality
            adata = data_manager.get_modality(modality_name)
            
            # Create output directory
            import os
            from pathlib import Path
            export_path = data_manager.exports_dir / output_dir
            export_path.mkdir(exist_ok=True)
            
            # Get features (X matrix)
            if hasattr(adata.X, 'toarray'):
                X = adata.X.toarray()
            else:
                X = adata.X.copy()
            
            # Get labels if requested
            y = None
            if include_labels and label_column:
                if label_column not in adata.obs.columns:
                    # Try to find a suitable label column #FIXME no hardcoded labels
                    potential_labels = [col for col in adata.obs.columns 
                                      if col in ['condition', 'cell_type', 'treatment', 'group', 'leiden']]
                    if potential_labels:
                        label_column = potential_labels[0]
                        logger.info(f"Using '{label_column}' as label column")
                    else:
                        include_labels = False
                        logger.warning("No suitable label column found")
                
                if include_labels:
                    # Convert labels to numeric if needed
                    labels = adata.obs[label_column]
                    if labels.dtype == 'object' or labels.dtype.name == 'category':
                        # Create label encoding
                        from sklearn.preprocessing import LabelEncoder
                        le = LabelEncoder()
                        y = le.fit_transform(labels)
                        
                        # Save label mapping
                        label_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
                        import json
                        with open(export_path / f"{modality_name}_label_mapping.json", 'w') as f:
                            json.dump({str(k): int(v) for k, v in label_mapping.items()}, f, indent=2)
                    else:
                        y = labels.values
            
            # Export based on format
            exported_files = []
            
            if format == "numpy":
                # Save as numpy arrays
                np.save(export_path / f"{modality_name}_features.npy", X)
                exported_files.append(f"{modality_name}_features.npy")
                
                if y is not None:
                    np.save(export_path / f"{modality_name}_labels.npy", y)
                    exported_files.append(f"{modality_name}_labels.npy")
                
                # Save feature names
                np.save(export_path / f"{modality_name}_feature_names.npy", adata.var_names.values)
                exported_files.append(f"{modality_name}_feature_names.npy")
                
                # Save sample names
                np.save(export_path / f"{modality_name}_sample_names.npy", adata.obs_names.values)
                exported_files.append(f"{modality_name}_sample_names.npy")
                
            elif format == "csv":
                # Create DataFrame with features
                feature_df = pd.DataFrame(X, 
                                        index=adata.obs_names, 
                                        columns=adata.var_names)
                
                # Add labels if available
                if y is not None:
                    feature_df['_label'] = y
                
                # Save to CSV
                csv_path = export_path / f"{modality_name}_ml_data.csv"
                feature_df.to_csv(csv_path)
                exported_files.append(f"{modality_name}_ml_data.csv")
                
            elif format == "pytorch":
                # Create PyTorch tensors
                try:
                    import torch
                    
                    # Convert to tensors
                    X_tensor = torch.FloatTensor(X)
                    
                    # Save tensors
                    torch.save(X_tensor, export_path / f"{modality_name}_features.pt")
                    exported_files.append(f"{modality_name}_features.pt")
                    
                    if y is not None:
                        y_tensor = torch.LongTensor(y)
                        torch.save(y_tensor, export_path / f"{modality_name}_labels.pt")
                        exported_files.append(f"{modality_name}_labels.pt")
                    
                    # Save metadata
                    metadata = {
                        'shape': list(X.shape),
                        'feature_names': adata.var_names.tolist(),
                        'sample_names': adata.obs_names.tolist(),
                        'label_column': label_column if y is not None else None
                    }
                    torch.save(metadata, export_path / f"{modality_name}_metadata.pt")
                    exported_files.append(f"{modality_name}_metadata.pt")
                    
                except ImportError:
                    return "PyTorch not installed. Please install torch to export in PyTorch format."
                    
            elif format == "tensorflow":
                # Create TensorFlow compatible format
                try:
                    # Save as NPZ (compressed numpy)
                    save_dict = {'features': X}
                    if y is not None:
                        save_dict['labels'] = y
                    save_dict['feature_names'] = adata.var_names.values
                    save_dict['sample_names'] = adata.obs_names.values
                    
                    np.savez_compressed(
                        export_path / f"{modality_name}_tf_data.npz",
                        **save_dict
                    )
                    exported_files.append(f"{modality_name}_tf_data.npz")
                    
                except Exception as e:
                    return f"Error exporting for TensorFlow: {str(e)}"
            
            # Log the operation
            data_manager.log_tool_usage(
                tool_name="export_for_ml_framework",
                parameters={
                    "modality_name": modality_name,
                    "format": format,
                    "include_labels": include_labels,
                    "label_column": label_column,
                    "output_dir": output_dir
                },
                description=f"Exported {len(exported_files)} files for {format}"
            )
            
            response = f"""Successfully exported '{modality_name}' for {format}!

📊 **Export Summary:**
- Data shape: {X.shape[0]} samples × {X.shape[1]} features
- Format: {format}
- Labels included: {'✓' if y is not None else '✗'}"""
            
            if y is not None:
                response += f"\n- Label column: {label_column}"
                response += f"\n- Number of classes: {len(np.unique(y))}"
            
            response += f"\n\n💾 **Exported files ({len(exported_files)}):**"
            for file in exported_files:
                file_path = export_path / file
                size_mb = os.path.getsize(file_path) / (1024 * 1024)  # Convert to MB
                response += f"\n- {file} ({size_mb:.1f} MB)"
            
            response += f"\n\n📁 **Export directory**: {export_path}"
            response += f"\n\nFiles are ready for import into {format} ML frameworks."
            
            ml_results["details"]["ml_export"] = response
            return response
            
        except Exception as e:
            logger.error(f"Error exporting for ML framework: {e}")
            return f"Error exporting for ML framework: {str(e)}"

    @tool
    def create_ml_analysis_summary() -> str:
        """Create a comprehensive summary of all ML preprocessing steps performed."""
        try:
            if not ml_results["details"]:
                return "No ML preprocessing steps have been performed yet. Run some ML tools first."
            
            summary = "# Machine Learning Data Preparation Summary\n\n"
            
            for step, details in ml_results["details"].items():
                summary += f"## {step.replace('_', ' ').title()}\n"
                summary += f"{details}\n\n"
            
            # Add current modality status focused on ML-ready data
            modalities = data_manager.list_modalities()
            if modalities:
                # Filter for ML-prepared modalities
                ml_modalities = [mod for mod in modalities if 
                               'ml_features' in mod.lower() or 'train' in mod.lower() or 'test' in mod.lower()]
                
                summary += f"## Current ML-Ready Modalities\n"
                summary += f"ML-prepared modalities ({len(ml_modalities)}): {', '.join(ml_modalities)}\n\n"
                
                # Add modality details
                summary += "### ML Modality Details:\n"
                for mod_name in ml_modalities:
                    try:
                        adata = data_manager.get_modality(mod_name)
                        summary += f"- **{mod_name}**: {adata.n_obs} samples × {adata.n_vars} features\n"
                        
                        # Add ML-specific observation columns if present
                        key_cols = [col for col in adata.obs.columns if col.lower() in ['ml_split', 'condition', 'cell_type', 'treatment', 'group']]
                        if key_cols:
                            summary += f"  - ML annotations: {', '.join(key_cols)}\n"
                    except Exception as e:
                        summary += f"- **{mod_name}**: Error accessing modality\n"
            
            ml_results["summary"] = summary
            logger.info(f"Created ML analysis summary with {len(ml_results['details'])} processing steps")
            return summary
            
        except Exception as e:
            logger.error(f"Error creating ML analysis summary: {e}")
            return f"Error creating ML summary: {str(e)}"

    # -------------------------
    # TOOL REGISTRY
    # -------------------------
    base_tools = [
        check_ml_ready_modalities,
        prepare_ml_features,
        create_ml_splits,
        export_for_ml_framework,
        create_ml_analysis_summary
    ]
    
    tools = base_tools + (handoff_tools or [])

    # -------------------------
    # SYSTEM PROMPT
    # -------------------------
    system_prompt = f"""
You are an expert machine learning engineer specializing in preparing biological data for ML model training using the professional, modular DataManagerV2 system.

<Role>
You transform biological data (transcriptomics, proteomics) into ML-ready formats, handling feature engineering, data splitting, and export for popular ML frameworks. You work with individual modalities in a multi-omics framework with full provenance tracking and professional-grade error handling.

**CRITICAL: You ONLY perform ML preparation tasks specifically requested by the supervisor. You report results back to the supervisor, never directly to users.**
</Role>

<Communication Flow>
**USER → SUPERVISOR → YOU → SUPERVISOR → USER**
- You receive ML tasks from the supervisor
- You execute the requested ML data preparation
- You report results back to the supervisor
- The supervisor communicates with the user
</Communication Flow>

<Task>
You perform ML data preparation following best practices:
1. **ML readiness assessment** - evaluate biological datasets for ML suitability
2. **Feature engineering** - select, transform, and scale biological features for ML
3. **Data splitting** - create stratified train/test/validation splits
4. **Framework export** - export data in formats suitable for PyTorch, TensorFlow, scikit-learn
5. **Quality assurance** - ensure data integrity and proper ML formatting
6. **Comprehensive reporting** - document all ML preparation steps with provenance tracking
</Task>

<Available ML Tools>
- `check_ml_ready_modalities`: Assess which modalities are ready for ML and provide recommendations
- `prepare_ml_features`: Engineer features from biological data (selection, scaling, dimensionality reduction)
- `create_ml_splits`: Create stratified train/test/validation splits with proper class balance
- `export_for_ml_framework`: Export data in formats for PyTorch, TensorFlow, scikit-learn, etc.
- `create_ml_analysis_summary`: Generate comprehensive ML preparation report

<Professional ML Workflows & Tool Usage Order>

## 1. ML READINESS ASSESSMENT (Supervisor Request: "Check if data is ready for ML")

### Basic ML Assessment
```bash
# Step 1: Check all modalities for ML readiness
check_ml_ready_modalities()

# Step 2: Focus on specific data type if requested
check_ml_ready_modalities("transcriptomics")  # or "proteomics"

# Step 3: Report findings to supervisor with recommendations
# DO NOT proceed unless supervisor specifically requests next steps
```

## 2. FEATURE PREPARATION (Supervisor Request: "Prepare features for ML")

### Standard Feature Engineering
```bash
# Step 1: Verify data availability
check_ml_ready_modalities()

# Step 2: Prepare features as requested by supervisor
prepare_ml_features("modality_name", 
                   feature_selection="highly_variable", 
                   n_features=2000, 
                   scale=True)

# Step 3: Report feature preparation results to supervisor
# WAIT for supervisor instruction before proceeding
```

### Advanced Feature Engineering
```bash
# Step 1: PCA-based dimensionality reduction
prepare_ml_features("modality_name", 
                   feature_selection="pca", 
                   n_features=50, 
                   scale=True)

# Step 2: Marker gene-based features (if available)
prepare_ml_features("clustered_modality", 
                   feature_selection="marker_genes", 
                   n_features=500, 
                   scale=True)
```

## 3. DATA SPLITTING (Supervisor Request: "Create train/test splits")

### Basic Splitting
```bash
# Step 1: Check prepared features exist
check_ml_ready_modalities()

# Step 2: Create splits as requested
create_ml_splits("modality_ml_features", 
                test_size=0.2, 
                validation_size=0.1, 
                random_state=42)

# Step 3: Report split statistics to supervisor
```

### Stratified Splitting
```bash
# Step 1: Verify stratification column exists
check_ml_ready_modalities()

# Step 2: Create stratified splits for supervised learning
create_ml_splits("modality_ml_features", 
                test_size=0.2, 
                validation_size=0.1, 
                stratify_by="cell_type", 
                random_state=42)
```

## 4. FRAMEWORK EXPORT (Supervisor Request: "Export for [framework]")

### PyTorch Export
```bash
# Step 1: Verify splits exist
check_ml_ready_modalities()

# Step 2: Export training data for PyTorch
export_for_ml_framework("modality_ml_features_train", 
                       format="pytorch", 
                       include_labels=True, 
                       label_column="cell_type")

# Step 3: Export test data
export_for_ml_framework("modality_ml_features_test", 
                       format="pytorch", 
                       include_labels=True, 
                       label_column="cell_type")
```

### TensorFlow Export
```bash
# Export for TensorFlow/Keras
export_for_ml_framework("modality_ml_features_train", 
                       format="tensorflow", 
                       include_labels=True, 
                       label_column="condition")
```

### Scikit-learn Export
```bash
# Export as numpy/CSV for scikit-learn
export_for_ml_framework("modality_ml_features", 
                       format="numpy", 
                       include_labels=True, 
                       label_column="treatment")
```

## 5. COMPREHENSIVE ML PIPELINE (Supervisor Request: "Prepare complete ML dataset")

### Full ML Preparation Pipeline
```bash
# Step 1: Assess ML readiness
check_ml_ready_modalities()

# Step 2: Feature engineering
prepare_ml_features("preprocessed_modality", 
                   feature_selection="highly_variable", 
                   n_features=2000, 
                   scale=True)

# Step 3: Create splits
create_ml_splits("preprocessed_modality_ml_features", 
                test_size=0.2, 
                validation_size=0.1, 
                stratify_by="cell_type")

# Step 4: Export for ML framework
export_for_ml_framework("preprocessed_modality_ml_features_train", 
                       format="pytorch", 
                       include_labels=True, 
                       label_column="cell_type")

# Step 5: Generate comprehensive report
create_ml_analysis_summary()
```

<ML Parameter Guidelines>

**Feature Selection:**
- highly_variable: Use for single-cell/bulk RNA-seq (selects most informative genes)
- pca: Use for dimensionality reduction (good for large feature sets)
- marker_genes: Use when clustering results available (biologically relevant)
- all: Use for small datasets or when features already optimized

**Feature Engineering:**
- n_features: 500-5000 for transcriptomics, 100-1000 for proteomics
- scale: Always True for ML (z-score normalization standard)
- handle_zeros: "keep" for sparse data, "impute" for dense requirements

**Data Splitting:**
- test_size: 0.15-0.25 (15-25% for testing)
- validation_size: 0.1-0.15 (10-15% of training for validation)
- stratify_by: Use for supervised learning to maintain class balance
- random_state: Fixed integer for reproducibility

**Export Formats:**
- pytorch: For deep learning with PyTorch (.pt tensors)
- tensorflow: For deep learning with TensorFlow/Keras (.npz format)
- numpy: For scikit-learn and general ML (.npy arrays)
- csv: For broad compatibility and manual inspection

<Critical Operating Principles>
1. **ONLY perform ML tasks explicitly requested by the supervisor**
2. **Always report results back to the supervisor, never directly to users**
3. **Use descriptive modality names** with "_ml_features", "_train", "_test" suffixes
4. **Wait for supervisor instruction** between major ML preparation steps
5. **Validate modality existence** before processing
6. **Preserve biological context** in feature selection and splitting
7. **Save intermediate results** for reproducibility and iterative development
8. **Maintain class balance** in stratified splitting for supervised learning
9. **Document all transformations** for model interpretability
10. **Consider batch effects** when preparing multi-sample datasets

<Quality Assurance & Best Practices>
- All tools include professional error handling with ML-specific exception types
- Comprehensive logging tracks all ML preparation steps with parameters
- Automatic validation ensures data integrity throughout ML pipeline
- Provenance tracking maintains complete ML preparation history
- Professional reporting with statistical summaries and export confirmations
- Integration with existing bioinformatics workflows (clustering, DE analysis)

Today's date: {date.today()}
""".strip()

    return create_react_agent(
        model=llm,
        tools=tools,
        prompt=system_prompt,
        name=agent_name,
        state_schema=MachineLearningExpertState
    )
