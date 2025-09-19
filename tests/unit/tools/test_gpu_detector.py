"""
Unit tests for GPU detection utility.

Tests GPU detection functionality without requiring actual GPU hardware.
"""

import pytest
from unittest.mock import patch, MagicMock
import subprocess

from lobster.tools.gpu_detector import (
    GPUDetector, 
    get_scvi_device_recommendation, 
    format_installation_message
)


class TestGPUDetector:
    """Test GPU detection functionality."""
    
    def test_check_nvidia_gpu_available(self):
        """Test NVIDIA GPU detection when nvidia-smi succeeds."""
        with patch('subprocess.run') as mock_run:
            mock_result = MagicMock()
            mock_result.returncode = 0
            mock_result.stdout = "NVIDIA GeForce RTX 3080, 10240 MiB\n"
            mock_run.return_value = mock_result
            
            available, info = GPUDetector.check_nvidia_gpu()
            
            assert available is True
            assert "RTX 3080" in info
            
    def test_check_nvidia_gpu_not_available(self):
        """Test NVIDIA GPU detection when nvidia-smi fails."""
        with patch('subprocess.run') as mock_run:
            mock_run.side_effect = FileNotFoundError()
            
            available, info = GPUDetector.check_nvidia_gpu()
            
            assert available is False
            assert info is None
    
    def test_check_apple_silicon_mac(self):
        """Test Apple Silicon detection on Mac."""
        with patch('platform.system') as mock_system, \
             patch('subprocess.run') as mock_run:
            
            mock_system.return_value = "Darwin"
            mock_result = MagicMock()
            mock_result.returncode = 0
            mock_result.stdout = "Apple M1"
            mock_run.return_value = mock_result
            
            is_apple_silicon = GPUDetector.check_apple_silicon()
            
            assert is_apple_silicon is True
    
    def test_check_apple_silicon_non_mac(self):
        """Test Apple Silicon detection on non-Mac systems."""
        with patch('platform.system') as mock_system:
            mock_system.return_value = "Linux"
            
            is_apple_silicon = GPUDetector.check_apple_silicon()
            
            assert is_apple_silicon is False
    
    def test_get_hardware_recommendation_nvidia(self):
        """Test hardware recommendation for NVIDIA GPU systems."""
        with patch.object(GPUDetector, 'check_nvidia_gpu') as mock_nvidia, \
             patch.object(GPUDetector, 'check_apple_silicon') as mock_apple:
            
            mock_nvidia.return_value = (True, "RTX 3080, 10240 MiB")
            mock_apple.return_value = False
            
            recommendation = GPUDetector.get_hardware_recommendation()
            
            assert recommendation["profile"] == "scvi-gpu"
            assert recommendation["device"] == "cuda"
            assert "RTX 3080" in recommendation["info"]
            
    def test_get_hardware_recommendation_apple_silicon(self):
        """Test hardware recommendation for Apple Silicon."""
        with patch.object(GPUDetector, 'check_nvidia_gpu') as mock_nvidia, \
             patch.object(GPUDetector, 'check_apple_silicon') as mock_apple:
            
            mock_nvidia.return_value = (False, None)
            mock_apple.return_value = True
            
            recommendation = GPUDetector.get_hardware_recommendation()
            
            assert recommendation["profile"] == "scvi-mac"
            assert recommendation["device"] == "mps"
            assert "Apple Silicon" in recommendation["info"]
    
    def test_get_hardware_recommendation_cpu_only(self):
        """Test hardware recommendation for CPU-only systems."""
        with patch.object(GPUDetector, 'check_nvidia_gpu') as mock_nvidia, \
             patch.object(GPUDetector, 'check_apple_silicon') as mock_apple:
            
            mock_nvidia.return_value = (False, None)
            mock_apple.return_value = False
            
            recommendation = GPUDetector.get_hardware_recommendation()
            
            assert recommendation["profile"] == "scvi-cpu"
            assert recommendation["device"] == "cpu"
            assert "No GPU detected" in recommendation["info"]
    
    def test_check_scvi_availability_ready(self):
        """Test scVI availability check when dependencies are installed."""
        with patch('importlib.util.find_spec') as mock_spec, \
             patch.object(GPUDetector, 'get_hardware_recommendation') as mock_hardware:
            
            # Mock torch and scvi as available
            mock_spec.return_value = MagicMock()
            mock_hardware.return_value = {
                "profile": "scvi-cpu",
                "device": "cpu",
                "info": "No GPU detected",
                "command": "pip install torch scvi-tools"
            }
            
            availability = GPUDetector.check_scvi_availability()
            
            assert availability["torch_available"] is True
            assert availability["scvi_available"] is True
            assert availability["ready_for_scvi"] is True
            assert availability["installation_needed"] is False
    
    def test_check_scvi_availability_not_ready(self):
        """Test scVI availability check when dependencies are missing."""
        with patch('importlib.util.find_spec') as mock_spec, \
             patch.object(GPUDetector, 'get_hardware_recommendation') as mock_hardware:
            
            # Mock torch and scvi as not available
            mock_spec.return_value = None
            mock_hardware.return_value = {
                "profile": "scvi-cpu",
                "device": "cpu",
                "info": "No GPU detected",
                "command": "pip install torch scvi-tools"
            }
            
            availability = GPUDetector.check_scvi_availability()
            
            assert availability["torch_available"] is False
            assert availability["scvi_available"] is False
            assert availability["ready_for_scvi"] is False
            assert availability["installation_needed"] is True


class TestUtilityFunctions:
    """Test utility functions."""
    
    def test_get_scvi_device_recommendation(self):
        """Test device recommendation utility function."""
        with patch.object(GPUDetector, 'get_hardware_recommendation') as mock_hardware:
            mock_hardware.return_value = {"device": "cuda"}
            
            device = get_scvi_device_recommendation()
            
            assert device == "cuda"
    
    def test_format_installation_message_ready(self):
        """Test formatting installation message when ready."""
        availability_info = {
            "ready_for_scvi": True,
            "hardware_recommendation": {
                "device": "cuda",
                "info": "NVIDIA GPU detected"
            }
        }
        
        message = format_installation_message(availability_info)
        
        assert "✅ scVI is ready!" in message
        assert "cuda" in message.lower()
        
    def test_format_installation_message_not_ready(self):
        """Test formatting installation message when not ready."""
        availability_info = {
            "ready_for_scvi": False,
            "torch_available": False,
            "scvi_available": False,
            "hardware_recommendation": {
                "device": "cpu",
                "info": "No GPU detected",
                "command": "pip install torch scvi-tools"
            }
        }
        
        message = format_installation_message(availability_info)
        
        assert "❌ Missing dependencies" in message
        assert "PyTorch" in message
        assert "scVI" in message
        assert "pip install" in message


if __name__ == "__main__":
    pytest.main([__file__])
