"""
scVI Embedding Service

Deep learning-based dimensionality reduction and batch correction using scVI
(single-cell Variational Inference). This service provides state-of-the-art
embeddings for single-cell RNA-seq data.
"""

from typing import Optional, Dict, Any, Tuple, List
import logging
from pathlib import Path

# Import GPU detector without ML dependencies
from .gpu_detector import GPUDetector, format_installation_message

logger = logging.getLogger(__name__)


class ScviEmbeddingService:
    """
    Service for training scVI embeddings on single-cell data.
    
    This service handles:
    - Dependency checking and installation guidance
    - scVI model training with optimal parameters
    - Embedding generation and storage
    - GPU/CPU device management
    - Batch correction capabilities
    """
    
    def __init__(self):
        self.availability_info = GPUDetector.check_scvi_availability()
        self.device = None
        self.model = None
        
    def check_availability(self) -> Dict[str, Any]:
        """
        Check if scVI dependencies are available.
        
        Returns:
            Dictionary with availability status and installation guidance
        """
        return self.availability_info.copy()
    
    def get_installation_message(self) -> str:
        """Get user-friendly installation guidance message."""
        return format_installation_message(self.availability_info)
    
    def _import_scvi_dependencies(self):
        """
        Import scVI dependencies with helpful error messages.
        
        Raises:
            ImportError: If dependencies are not available with installation guidance
        """
        if not self.availability_info["ready_for_scvi"]:
            message = self.get_installation_message()
            raise ImportError(f"scVI dependencies not available.\n{message}")
        
        try:
            import torch
            import scvi
            return torch, scvi
        except ImportError as e:
            hardware_rec = self.availability_info["hardware_recommendation"]
            raise ImportError(
                f"Failed to import scVI dependencies: {e}\n"
                f"Try installing with: {hardware_rec['command']}"
            )
    
    def setup_device(self, force_cpu: bool = False) -> str:
        """
        Set up the optimal compute device for training.
        
        Args:
            force_cpu: Force CPU usage even if GPU is available
            
        Returns:
            Device string used ("cuda", "mps", or "cpu")
        """
        torch, _ = self._import_scvi_dependencies()
        
        if force_cpu:
            device = "cpu"
        else:
            hardware_rec = GPUDetector.get_hardware_recommendation()
            recommended_device = hardware_rec["device"]
            
            # Verify the device is actually available
            if recommended_device == "cuda" and torch.cuda.is_available():
                device = "cuda"
            elif recommended_device == "mps" and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        
        self.device = device
        logger.info(f"Using device: {device}")
        return device
    
    def train_scvi_embedding(
        self,
        adata,
        batch_key: Optional[str] = None,
        categorical_covariate_keys: Optional[List[str]] = None,
        continuous_covariate_keys: Optional[List[str]] = None,
        layer: Optional[str] = None,
        n_latent: int = 10,
        max_epochs: Optional[int] = None,
        early_stopping: bool = True,
        save_path: Optional[str] = None,
        force_cpu: bool = False,
        **model_kwargs
    ) -> Tuple[Any, Dict[str, Any]]:
        """
        Train scVI model and generate embeddings.
        
        Args:
            adata: AnnData object with single-cell data
            batch_key: Key in adata.obs for batch information
            categorical_covariate_keys: List of categorical covariate keys
            continuous_covariate_keys: List of continuous covariate keys
            layer: Layer to use for training (None for X)
            n_latent: Number of latent dimensions (embedding size)
            max_epochs: Maximum training epochs (auto-calculated if None)
            early_stopping: Whether to use early stopping
            save_path: Path to save trained model
            force_cpu: Force CPU usage
            **model_kwargs: Additional arguments for scVI model
            
        Returns:
            Tuple of (trained_model, training_info_dict)
        """
        torch, scvi = self._import_scvi_dependencies()
        
        # Setup device
        device = self.setup_device(force_cpu=force_cpu)
        
        # Prepare data for scVI
        logger.info("Setting up scVI data...")
        
        # Setup AnnData for scVI
        scvi.model.SCVI.setup_anndata(
            adata,
            layer=layer,
            batch_key=batch_key,
            categorical_covariate_keys=categorical_covariate_keys,
            continuous_covariate_keys=continuous_covariate_keys
        )
        
        # Create model
        logger.info(f"Creating scVI model with {n_latent} latent dimensions...")
        model = scvi.model.SCVI(
            adata,
            n_latent=n_latent,
            **model_kwargs
        )
        
        # Calculate epochs if not specified
        if max_epochs is None:
            n_cells = adata.n_obs
            if n_cells < 5000:
                max_epochs = 400
            elif n_cells < 20000:
                max_epochs = 200
            else:
                max_epochs = 100
        
        # Train model
        logger.info(f"Training scVI model for up to {max_epochs} epochs on {device}...")
        
        train_kwargs = {
            "max_epochs": max_epochs,
            "use_gpu": device != "cpu",
            "early_stopping": early_stopping,
            "early_stopping_patience": 10
        }
        
        model.train(**train_kwargs)
        
        # Generate embeddings
        logger.info("Generating embeddings...")
        latent_representation = model.get_latent_representation()
        
        # Store embeddings in AnnData object
        adata.obsm["X_scvi"] = latent_representation
        
        # Save model if requested
        if save_path:
            model.save(save_path, overwrite=True)
            logger.info(f"Model saved to {save_path}")
        
        # Collect training information
        training_info = {
            "n_latent": n_latent,
            "max_epochs": max_epochs,
            "device": device,
            "n_cells": adata.n_obs,
            "n_genes": adata.n_vars,
            "batch_key": batch_key,
            "embedding_shape": latent_representation.shape,
            "model_saved": save_path is not None
        }
        
        self.model = model
        logger.info("scVI training completed successfully!")
        
        return model, training_info
    
    def get_embeddings(self, adata, model=None) -> Optional[any]:
        """
        Extract embeddings from trained model or AnnData object.
        
        Args:
            adata: AnnData object
            model: Trained scVI model (uses self.model if None)
            
        Returns:
            Embedding array or None if not available
        """
        # Try to get from AnnData first
        if "X_scvi" in adata.obsm:
            return adata.obsm["X_scvi"]
        
        # Try to generate from model
        if model is not None:
            return model.get_latent_representation()
        elif self.model is not None:
            return self.model.get_latent_representation()
        
        return None
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the current model.
        
        Returns:
            Dictionary with model information
        """
        if self.model is None:
            return {"model_available": False}
        
        try:
            # Get basic model info
            info = {
                "model_available": True,
                "model_type": type(self.model).__name__,
                "device": self.device,
                "n_latent": getattr(self.model.module, 'n_latent', None),
                "is_trained": hasattr(self.model, 'is_trained_') and self.model.is_trained_
            }
            
            # Add training history if available
            if hasattr(self.model, 'history') and self.model.history is not None:
                history = self.model.history
                info["training_history"] = {
                    "final_elbo": history['elbo_train'].iloc[-1] if 'elbo_train' in history else None,
                    "epochs_trained": len(history) if len(history) > 0 else 0
                }
            
            return info
            
        except Exception as e:
            return {
                "model_available": True,
                "error": str(e)
            }
