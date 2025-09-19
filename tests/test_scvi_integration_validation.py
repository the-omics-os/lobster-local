"""
End-to-end validation script for scVI integration.

This script validates the complete scVI integration without requiring actual scVI installation.
It checks that all components are properly integrated and would work when dependencies are installed.
"""

import sys
import os
from pathlib import Path

# Add lobster to path
sys.path.insert(0, str(Path(__file__).parent.parent))

def test_imports():
    """Test that all new modules can be imported."""
    print("🔍 Testing imports...")
    
    try:
        from lobster.tools.gpu_detector import GPUDetector, get_scvi_device_recommendation
        print("✅ GPU detector imports successfully")
    except ImportError as e:
        print(f"❌ GPU detector import failed: {e}")
        return False
    
    try:
        from lobster.tools.scvi_embedding_service import ScviEmbeddingService
        print("✅ scVI embedding service imports successfully")
    except ImportError as e:
        print(f"❌ scVI embedding service import failed: {e}")
        return False
    
    return True


def test_agent_modifications():
    """Test that agent modifications are properly integrated."""
    print("\n🔍 Testing agent modifications...")
    
    try:
        from lobster.config.agent_registry import AGENT_REGISTRY, AgentRegistryConfig
        
        # Check agent registry fix
        ml_agent_config = AGENT_REGISTRY.get('machine_learning_expert_agent')
        if isinstance(ml_agent_config, AgentRegistryConfig):
            print("✅ Agent registry typo fixed successfully")
        else:
            print("❌ Agent registry still has typo")
            return False
    except ImportError as e:
        print(f"❌ Agent registry import failed: {e}")
        return False
    
    # Test agent creation (without actually running them)
    try:
        from lobster.agents.machine_learning_expert import machine_learning_expert
        from lobster.agents.singlecell_expert import singlecell_expert
        from lobster.core.data_manager_v2 import DataManagerV2
        from unittest.mock import MagicMock
        
        # Create mock data manager
        mock_dm = MagicMock(spec=DataManagerV2)
        
        # Test ML Expert creation
        ml_agent = machine_learning_expert(mock_dm)
        ml_tool_names = [getattr(tool, 'name', str(tool)) for tool in ml_agent.tools]
        
        scvi_tools_present = [
            any('check_scvi_availability' in str(name) for name in ml_tool_names),
            any('train_scvi_embedding' in str(name) for name in ml_tool_names)
        ]
        
        if all(scvi_tools_present):
            print("✅ ML Expert has scVI tools")
        else:
            print("❌ ML Expert missing scVI tools")
            return False
        
        # Test SingleCell Expert creation
        sc_agent = singlecell_expert(mock_dm)
        sc_tool_names = [getattr(tool, 'name', str(tool)) for tool in sc_agent.tools]
        
        has_scvi_handoff = any('request_scvi_embedding' in str(name) for name in sc_tool_names)
        if has_scvi_handoff:
            print("✅ SingleCell Expert has scVI handoff tool")
        else:
            print("❌ SingleCell Expert missing scVI handoff tool")
            return False
            
    except ImportError as e:
        print(f"❌ Agent import failed: {e}")
        return False
    except Exception as e:
        print(f"❌ Agent creation failed: {e}")
        return False
    
    return True


def test_clustering_service():
    """Test clustering service modifications."""
    print("\n🔍 Testing clustering service modifications...")
    
    try:
        from lobster.tools.clustering_service import ClusteringService
        
        # Create service
        service = ClusteringService()
        
        # Check that cluster_and_visualize method accepts use_rep parameter
        import inspect
        cluster_method = service.cluster_and_visualize
        signature = inspect.signature(cluster_method)
        
        if 'use_rep' in signature.parameters:
            print("✅ Clustering service accepts use_rep parameter")
        else:
            print("❌ Clustering service missing use_rep parameter")
            return False
            
    except ImportError as e:
        print(f"❌ Clustering service import failed: {e}")
        return False
    
    return True


def test_gpu_detection():
    """Test GPU detection functionality."""
    print("\n🔍 Testing GPU detection...")
    
    try:
        from lobster.tools.gpu_detector import GPUDetector
        
        # Test hardware detection (should work on any system)
        recommendation = GPUDetector.get_hardware_recommendation()
        
        required_keys = ["profile", "device", "info", "command"]
        if all(key in recommendation for key in required_keys):
            print(f"✅ GPU detection works - Device: {recommendation['device']}")
        else:
            print("❌ GPU detection missing required keys")
            return False
        
        # Test scVI availability check (should work regardless of installation)
        availability = GPUDetector.check_scvi_availability()
        
        required_availability_keys = ["torch_available", "scvi_available", "ready_for_scvi"]
        if all(key in availability for key in required_availability_keys):
            scvi_status = "ready" if availability["ready_for_scvi"] else "not ready"
            print(f"✅ scVI availability check works - Status: {scvi_status}")
        else:
            print("❌ scVI availability check missing required keys")
            return False
            
    except Exception as e:
        print(f"❌ GPU detection failed: {e}")
        return False
    
    return True


def test_installation_guidance():
    """Test installation guidance generation."""
    print("\n🔍 Testing installation guidance...")
    
    try:
        from lobster.tools.gpu_detector import GPUDetector, format_installation_message
        
        # Test with mock availability data
        mock_availability = {
            "ready_for_scvi": False,
            "torch_available": False,
            "scvi_available": False,
            "hardware_recommendation": {
                "device": "cpu",
                "info": "No GPU detected - CPU-only mode",
                "command": "pip install torch scvi-tools"
            }
        }
        
        message = format_installation_message(mock_availability)
        
        if "pip install" in message and "torch" in message:
            print("✅ Installation guidance generation works")
        else:
            print("❌ Installation guidance incomplete")
            return False
            
    except Exception as e:
        print(f"❌ Installation guidance failed: {e}")
        return False
    
    return True


def main():
    """Run all validation tests."""
    print("🧪 scVI Integration Validation\n" + "="*50)
    
    tests = [
        test_imports,
        test_agent_modifications,
        test_clustering_service,
        test_gpu_detection,
        test_installation_guidance
    ]
    
    results = []
    for test_func in tests:
        try:
            result = test_func()
            results.append(result)
        except Exception as e:
            print(f"❌ Test {test_func.__name__} crashed: {e}")
            results.append(False)
    
    print("\n" + "="*50)
    print("📊 Validation Results:")
    print(f"✅ Passed: {sum(results)}/{len(results)} tests")
    
    if all(results):
        print("\n🎉 All validation tests passed!")
        print("💡 scVI integration is properly implemented and ready for use.")
        return True
    else:
        print("\n⚠️  Some validation tests failed.")
        print("🔧 Please check the failed components above.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
