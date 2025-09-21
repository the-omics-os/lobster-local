"""
GPU Detection Utility

Provides GPU availability detection without requiring PyTorch installation.
Used by scVI integration to determine hardware capabilities.
"""

import os
import subprocess
from typing import Dict, Optional, Tuple
from lobster.utils import IS_MACOS


class GPUDetector:
    """Detects GPU availability and capabilities without importing ML libraries."""
    
    @staticmethod
    def check_nvidia_gpu() -> Tuple[bool, Optional[str]]:
        """
        Check for NVIDIA GPU using nvidia-smi command.
        
        Returns:
            Tuple of (gpu_available: bool, gpu_info: str or None)
        """
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader'],
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode == 0 and result.stdout.strip():
                gpu_info = result.stdout.strip().split('\n')[0]  # First GPU
                return True, gpu_info
            return False, None
        except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.SubprocessError):
            return False, None
    
    @staticmethod
    def check_apple_silicon() -> bool:
        """Check if running on Apple Silicon (M1/M2/M3) Mac."""
        if not IS_MACOS:
            return False
        
        try:
            result = subprocess.run(
                ['sysctl', '-n', 'machdep.cpu.brand_string'],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                cpu_info = result.stdout.strip().lower()
                return 'apple' in cpu_info or 'm1' in cpu_info or 'm2' in cpu_info or 'm3' in cpu_info
            return False
        except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.SubprocessError):
            return False
    
    @staticmethod
    def get_hardware_recommendation() -> Dict[str, str]:
        """
        Get hardware-specific recommendations for scVI installation.
        
        Returns:
            Dictionary with installation profile and device recommendation
        """
        nvidia_available, gpu_info = GPUDetector.check_nvidia_gpu()
        apple_silicon = GPUDetector.check_apple_silicon()
        
        if nvidia_available:
            return {
                "profile": "scvi-gpu",
                "device": "cuda",
                "info": f"NVIDIA GPU detected: {gpu_info}",
                "command": "pip install 'lobster[scvi-gpu]'"
            }
        elif apple_silicon:
            return {
                "profile": "scvi-mac",
                "device": "mps",
                "info": "Apple Silicon Mac detected - MPS acceleration available",
                "command": "pip install 'lobster[scvi-mac]'"
            }
        else:
            return {
                "profile": "scvi-cpu",
                "device": "cpu",
                "info": "No GPU detected - CPU-only mode",
                "command": "pip install 'lobster[scvi-cpu]'"
            }
    
    @staticmethod
    def check_scvi_availability() -> Dict[str, any]:
        """
        Check if scVI and PyTorch are available without importing them.
        
        Returns:
            Dictionary with availability status and recommendations
        """
        try:
            import importlib.util
            
            # Check PyTorch
            torch_spec = importlib.util.find_spec("torch")
            torch_available = torch_spec is not None
            
            # Check scVI
            scvi_spec = importlib.util.find_spec("scvi")
            scvi_available = scvi_spec is not None
            
            hardware_rec = GPUDetector.get_hardware_recommendation()
            
            return {
                "torch_available": torch_available,
                "scvi_available": scvi_available,
                "ready_for_scvi": torch_available and scvi_available,
                "hardware_recommendation": hardware_rec,
                "installation_needed": not (torch_available and scvi_available)
            }
            
        except Exception as e:
            return {
                "torch_available": False,
                "scvi_available": False,
                "ready_for_scvi": False,
                "hardware_recommendation": GPUDetector.get_hardware_recommendation(),
                "installation_needed": True,
                "error": str(e)
            }


def get_scvi_device_recommendation() -> str:
    """
    Get the recommended device string for scVI based on hardware.
    
    Returns:
        Device string: "cuda", "mps", or "cpu"
    """
    recommendation = GPUDetector.get_hardware_recommendation()
    return recommendation["device"]


def format_installation_message(availability_info: Dict[str, any]) -> str:
    """
    Format a user-friendly message about scVI installation status.
    
    Args:
        availability_info: Result from check_scvi_availability()
        
    Returns:
        Formatted message string
    """
    if availability_info["ready_for_scvi"]:
        device = availability_info["hardware_recommendation"]["device"]
        info = availability_info["hardware_recommendation"]["info"]
        return f"✅ scVI is ready! Device: {device} ({info})"
    
    hardware_rec = availability_info["hardware_recommendation"]
    missing = []
    if not availability_info["torch_available"]:
        missing.append("PyTorch")
    if not availability_info["scvi_available"]:
        missing.append("scVI")
    
    message = f"❌ Missing dependencies: {', '.join(missing)}\n"
    message += f"💡 Recommended installation: {hardware_rec['command']}\n"
    message += f"🔧 Hardware detected: {hardware_rec['info']}"
    
    return message


if __name__ == "__main__":
    # Quick test when run directly
    availability = GPUDetector.check_scvi_availability()
    print(format_installation_message(availability))
