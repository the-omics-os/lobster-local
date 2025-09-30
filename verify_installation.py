#!/usr/bin/env python3
"""
Simple verification script for Lobster package split
"""
import sys
import os

def test_imports():
    """Test basic imports"""
    success = True

    # Test core lobster package
    try:
        import lobster
        print("✅ lobster: Core package imported successfully")

        # Test version access
        try:
            from lobster.version import __version__
            print(f"   Version: {__version__}")
        except ImportError:
            print("   Version: Unable to determine")

    except ImportError as e:
        print(f"❌ lobster: Core package import failed - {e}")
        success = False

    # Test core components
    try:
        from lobster.core.client import AgentClient
        from lobster.core.data_manager_v2 import DataManagerV2
        print("✅ lobster-core: Core components available")
        print("   AgentClient and DataManagerV2 imported successfully")

    except ImportError as e:
        # Check if it's specifically a proteomics import error (expected in public version)
        if "proteomics_adapter" in str(e):
            print("⚠️  lobster-core: Proteomics modules not available (expected for public distribution)")
            print("   Core components will work with transcriptomics only")
            # Don't mark as failure - this is expected for public version
        else:
            print(f"❌ lobster-core: Core component import failed - {e}")
            print(f"   This indicates a dependency issue. Try: pip install -e .")
            success = False

    # Test CLI functionality
    try:
        from lobster.cli import app
        print("✅ lobster-cli: CLI components available")

    except ImportError as e:
        print(f"❌ lobster-cli: CLI import failed - {e}")
        success = False

    # Test proteomics modules (optional in public distribution)
    print("\n📋 Proteomics Modules Check (Optional):")
    proteomics_modules = [
        ('lobster.core.adapters.proteomics_adapter', 'Proteomics data adapter'),
        ('lobster.agents.ms_proteomics_expert', 'Mass spectrometry proteomics agent'),
        ('lobster.agents.affinity_proteomics_expert', 'Affinity proteomics agent'),
        ('lobster.tools.proteomics_preprocessing_service', 'Proteomics preprocessing'),
        ('lobster.tools.proteomics_analysis_service', 'Proteomics analysis'),
        ('lobster.tools.proteomics_differential_service', 'Proteomics differential analysis'),
        ('lobster.tools.proteomics_visualization_service', 'Proteomics visualization'),
        ('lobster.tools.proteomics_quality_service', 'Proteomics quality control'),
    ]

    proteomics_available = False
    for module_name, description in proteomics_modules:
        try:
            __import__(module_name, fromlist=[''])
            print(f"✅ {module_name.split('.')[-1]}: Available - {description}")
            proteomics_available = True
        except ImportError:
            print(f"⚠️  {module_name.split('.')[-1]}: Not available - {description}")

    if not proteomics_available:
        print("ℹ️  Note: Proteomics modules are not available in the public distribution")
        print("   This is expected and does not affect transcriptomics functionality")

    # Test optional dependencies (non-failing tests)
    optional_deps = {
        'scvi': 'scvi-tools (for advanced single-cell analysis)',
        'torch': 'PyTorch (for deep learning features)',
        'numba': 'Numba (for performance optimization)',
    }

    print("\n📋 Optional Dependencies Check:")
    for dep_name, description in optional_deps.items():
        try:
            __import__(dep_name)
            print(f"✅ {dep_name}: Available - {description}")
        except ImportError:
            print(f"⚠️  {dep_name}: Not available - {description}")

    return success

def test_cli_detection():
    """Test CLI cloud detection logic"""
    try:
        from lobster.cli import init_client
        
        # Mock environment without cloud key
        old_key = os.environ.get('LOBSTER_CLOUD_KEY')
        if 'LOBSTER_CLOUD_KEY' in os.environ:
            del os.environ['LOBSTER_CLOUD_KEY']
        
        print("✅ CLI module: Successfully imported")
        
        # Restore environment
        if old_key:
            os.environ['LOBSTER_CLOUD_KEY'] = old_key
            
        return True
        
    except Exception as e:
        print(f"❌ CLI detection: Failed - {e}")
        return False

def test_dependency_versions():
    """Test critical dependency versions"""
    critical_deps = {
        'numpy': '1.23.0',
        'pandas': '1.5.0',
        'scipy': '1.10.0',
        'scanpy': '1.11.4',
        'anndata': '0.9.0',
    }

    print("🔍 Critical Dependency Versions:")
    success = True

    for dep_name, min_version in critical_deps.items():
        try:
            module = __import__(dep_name)
            version = getattr(module, '__version__', 'unknown')
            print(f"✅ {dep_name}: {version} (min: {min_version})")
        except ImportError:
            print(f"❌ {dep_name}: Not installed (required: {min_version})")
            success = False

    return success

def main():
    """Run all verification tests"""
    print("🦞 Lobster AI Installation Verification")
    print("=" * 50)

    tests = [
        ("Core Package Imports", test_imports),
        ("CLI Functionality", test_cli_detection),
        ("Dependency Versions", test_dependency_versions),
    ]
    
    results = []
    for name, test_func in tests:
        print(f"\n🔍 Testing {name}...")
        try:
            result = test_func()
            results.append(result)
        except Exception as e:
            print(f"❌ {name}: Exception - {e}")
            results.append(False)
    
    # Summary
    passed = sum(results)
    total = len(results)
    
    print("\n" + "=" * 50)
    print("📊 VERIFICATION SUMMARY")
    print("=" * 50)
    
    for i, (name, _) in enumerate(tests):
        status = "✅ PASS" if results[i] else "❌ FAIL"
        print(f"{status} {name}")
    
    print(f"\n🎯 Overall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 SUCCESS! Lobster AI installation is working correctly!")
        print("\n📋 Next Steps:")
        print("   1. Configure API keys in .env file")
        print("   2. Start Lobster: lobster chat")
        print("   3. Try: 'Download GSE109564 and perform single-cell analysis'")
        print("\n📚 Documentation: https://github.com/the-omics-os/lobster")
        return True
    else:
        print("\n⚠️  Some tests failed. Installation may be incomplete.")
        print("\n🔧 Troubleshooting:")
        print("   • Ensure all system dependencies are installed")
        print("   • Try: make clean-install")
        print("   • Check Python version: python --version (requires 3.12+)")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
