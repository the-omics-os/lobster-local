"""
Unit tests for scVI embedding service.

Tests scVI functionality with conditional imports - works with and without scVI installed.
"""

import pytest
from unittest.mock import patch, MagicMock, Mock
import numpy as np
import pandas as pd

from lobster.tools.scvi_embedding_service import ScviEmbeddingService

# Check if scVI is available for conditional testing
try:
    import scvi
    import torch
    SCVI_AVAILABLE = True
except ImportError:
    SCVI_AVAILABLE = False


class TestScviEmbeddingServiceBasics:
    """Test basic functionality that doesn't require scVI."""
    
    def test_service_initialization(self):
        """Test service initialization."""
        service = ScviEmbeddingService()
        
        assert service.availability_info is not None
        assert service.device is None
        assert service.model is None
    
    def test_check_availability(self):
        """Test availability checking."""
        service = ScviEmbeddingService()
        availability = service.check_availability()
        
        assert "torch_available" in availability
        assert "scvi_available" in availability
        assert "ready_for_scvi" in availability
        assert "hardware_recommendation" in availability
    
    def test_get_installation_message(self):
        """Test installation message generation."""
        service = ScviEmbeddingService()
        message = service.get_installation_message()
        
        assert isinstance(message, str)
        assert len(message) > 0
        
        # Should contain either ready message or installation instructions
        assert ("✅ scVI is ready!" in message) or ("❌ Missing dependencies" in message)


class TestScviEmbeddingServiceWithoutDependencies:
    """Test behavior when scVI dependencies are not available."""
    
    def test_import_dependencies_fails_gracefully(self):
        """Test graceful handling when dependencies are missing."""
        with patch.object(ScviEmbeddingService, 'check_availability') as mock_check:
            mock_check.return_value = {"ready_for_scvi": False}
            
            service = ScviEmbeddingService()
            
            with pytest.raises(ImportError) as exc_info:
                service._import_scvi_dependencies()
            
            assert "scVI dependencies not available" in str(exc_info.value)
    
    def test_train_scvi_embedding_without_dependencies(self):
        """Test training fails gracefully without dependencies."""
        with patch.object(ScviEmbeddingService, 'check_availability') as mock_check:
            mock_check.return_value = {"ready_for_scvi": False}
            
            service = ScviEmbeddingService()
            
            # Create mock AnnData
            mock_adata = MagicMock()
            mock_adata.n_obs = 1000
            mock_adata.n_vars = 2000
            
            with pytest.raises(ImportError):
                service.train_scvi_embedding(mock_adata)


@pytest.mark.skipif(not SCVI_AVAILABLE, reason="scVI dependencies not installed")
class TestScviEmbeddingServiceWithDependencies:
    """Test functionality when scVI dependencies are available."""
    
    def test_setup_device_cpu(self):
        """Test device setup for CPU."""
        service = ScviEmbeddingService()
        
        with patch.object(service, '_import_scvi_dependencies') as mock_import:
            mock_torch = MagicMock()
            mock_torch.cuda.is_available.return_value = False
            mock_torch.backends.mps.is_available.return_value = False
            mock_import.return_value = (mock_torch, MagicMock())
            
            device = service.setup_device(force_cpu=True)
            
            assert device == "cpu"
            assert service.device == "cpu"
    
    def test_setup_device_cuda_available(self):
        """Test device setup when CUDA is available."""
        service = ScviEmbeddingService()
        
        with patch.object(service, '_import_scvi_dependencies') as mock_import, \
             patch.object(service, 'check_availability') as mock_check:
            
            mock_torch = MagicMock()
            mock_torch.cuda.is_available.return_value = True
            mock_torch.backends.mps.is_available.return_value = False
            mock_import.return_value = (mock_torch, MagicMock())
            
            # Mock hardware recommendation for CUDA
            service.availability_info = {
                "ready_for_scvi": True,
                "hardware_recommendation": {"device": "cuda"}
            }
            
            device = service.setup_device()
            
            assert device == "cuda"
    
    def test_get_embeddings_from_adata(self):
        """Test extracting embeddings from AnnData object."""
        service = ScviEmbeddingService()
        
        # Create mock AnnData with embeddings
        mock_adata = MagicMock()
        mock_embeddings = np.random.randn(100, 10)
        mock_adata.obsm = {"X_scvi": mock_embeddings}
        
        embeddings = service.get_embeddings(mock_adata)
        
        assert np.array_equal(embeddings, mock_embeddings)
    
    def test_get_embeddings_from_model(self):
        """Test extracting embeddings from trained model."""
        service = ScviEmbeddingService()
        
        # Create mock model and AnnData
        mock_model = MagicMock()
        mock_embeddings = np.random.randn(100, 10)
        mock_model.get_latent_representation.return_value = mock_embeddings
        
        mock_adata = MagicMock()
        mock_adata.obsm = {}  # No pre-existing embeddings
        
        embeddings = service.get_embeddings(mock_adata, model=mock_model)
        
        assert np.array_equal(embeddings, mock_embeddings)
        mock_model.get_latent_representation.assert_called_once()
    
    def test_get_model_info_no_model(self):
        """Test model info when no model is available."""
        service = ScviEmbeddingService()
        
        info = service.get_model_info()
        
        assert info["model_available"] is False
    
    def test_get_model_info_with_model(self):
        """Test model info when model is available."""
        service = ScviEmbeddingService()
        
        # Create mock model
        mock_model = MagicMock()
        mock_model.module.n_latent = 10
        mock_model.is_trained_ = True
        service.model = mock_model
        service.device = "cpu"
        
        info = service.get_model_info()
        
        assert info["model_available"] is True
        assert info["device"] == "cpu"
        assert info["n_latent"] == 10
        assert info["is_trained"] is True


@pytest.mark.skipif(not SCVI_AVAILABLE, reason="scVI dependencies not installed")  
class TestScviTrainingIntegration:
    """Integration tests for scVI training workflow."""
    
    @patch('scvi.model.SCVI')
    def test_train_scvi_embedding_basic(self, mock_scvi_class):
        """Test basic scVI training workflow."""
        service = ScviEmbeddingService()
        
        # Mock the trained model
        mock_model = MagicMock()
        mock_embeddings = np.random.randn(100, 10)
        mock_model.get_latent_representation.return_value = mock_embeddings
        mock_model.history = pd.DataFrame({'elbo_train': [-1000, -950, -900]})
        mock_scvi_class.return_value = mock_model
        
        # Create mock AnnData
        mock_adata = MagicMock()
        mock_adata.n_obs = 100
        mock_adata.n_vars = 2000
        mock_adata.obsm = {}
        
        # Mock the service setup
        with patch.object(service, '_import_scvi_dependencies') as mock_import, \
             patch.object(service, 'setup_device') as mock_setup_device:
            
            mock_torch = MagicMock()
            mock_scvi = MagicMock()
            mock_scvi.model.SCVI = mock_scvi_class
            mock_import.return_value = (mock_torch, mock_scvi)
            mock_setup_device.return_value = "cpu"
            
            # Train the model
            model, training_info = service.train_scvi_embedding(
                adata=mock_adata,
                n_latent=10,
                max_epochs=100
            )
            
            # Verify results
            assert model == mock_model
            assert training_info["n_latent"] == 10
            assert training_info["device"] == "cpu"
            assert training_info["n_cells"] == 100
            assert training_info["n_genes"] == 2000
            assert training_info["embedding_shape"] == mock_embeddings.shape
            
            # Verify scVI setup was called
            mock_scvi.model.SCVI.setup_anndata.assert_called_once()
            mock_scvi_class.assert_called_once()
            mock_model.train.assert_called_once()
            mock_model.get_latent_representation.assert_called_once()
    
    def test_train_scvi_embedding_with_batch_correction(self):
        """Test scVI training with batch correction."""
        service = ScviEmbeddingService()
        
        # Create mock AnnData with batch information
        mock_adata = MagicMock()
        mock_adata.n_obs = 100
        mock_adata.n_vars = 2000
        mock_adata.obsm = {}
        
        with patch.object(service, '_import_scvi_dependencies') as mock_import, \
             patch.object(service, 'setup_device') as mock_setup_device:
            
            mock_torch = MagicMock()
            mock_scvi = MagicMock()
            mock_model = MagicMock()
            mock_embeddings = np.random.randn(100, 10)
            mock_model.get_latent_representation.return_value = mock_embeddings
            mock_scvi.model.SCVI.return_value = mock_model
            mock_import.return_value = (mock_torch, mock_scvi)
            mock_setup_device.return_value = "cpu"
            
            model, training_info = service.train_scvi_embedding(
                adata=mock_adata,
                batch_key="sample",
                n_latent=15
            )
            
            # Verify batch key was used
            assert training_info["batch_key"] == "sample"
            mock_scvi.model.SCVI.setup_anndata.assert_called_once()
            call_kwargs = mock_scvi.model.SCVI.setup_anndata.call_args[1]
            assert call_kwargs["batch_key"] == "sample"


class TestErrorHandling:
    """Test error handling scenarios."""
    
    def test_service_with_import_error(self):
        """Test service behavior when imports fail."""
        with patch('lobster.tools.scvi_embedding_service.GPUDetector') as mock_detector:
            mock_detector.check_scvi_availability.side_effect = ImportError("Test import error")
            
            # Service should still initialize but availability check should handle error
            service = ScviEmbeddingService()
            availability = service.check_availability()
            
            # Should return a dict with error information
            assert isinstance(availability, dict)
    
    def test_invalid_parameters(self):
        """Test handling of invalid parameters."""
        service = ScviEmbeddingService()
        
        # Test with None AnnData
        with pytest.raises((TypeError, AttributeError)):
            service.train_scvi_embedding(None)


class TestConditionalSkip:
    """Test that provides useful information when scVI is not available."""
    
    @pytest.mark.skipif(SCVI_AVAILABLE, reason="scVI is installed - testing unavailable scenario")
    def test_scvi_not_available_message(self):
        """Test that we get helpful messages when scVI is not available."""
        # This test only runs when scVI is NOT installed
        service = ScviEmbeddingService()
        
        # The service should detect scVI is not available
        availability = service.check_availability()
        assert availability["ready_for_scvi"] is False
        
        message = service.get_installation_message()
        assert "pip install" in message.lower() or "missing" in message.lower()
    
    @pytest.mark.skipif(not SCVI_AVAILABLE, reason="scVI dependencies not installed")
    def test_scvi_available_message(self):
        """Test that we get ready messages when scVI is available."""
        # This test only runs when scVI IS installed
        service = ScviEmbeddingService()
        
        # The service should detect scVI is available
        availability = service.check_availability()
        assert availability["ready_for_scvi"] is True
        
        message = service.get_installation_message()
        assert "✅ scVI is ready!" in message


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "-s"])
