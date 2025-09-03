#!/usr/bin/env python3
"""
Lobster AI Open Source Installation Test
Verifies that the open source components work correctly
"""

import sys
import os
import subprocess
from pathlib import Path

def test_imports():
    """Test that core components can be imported."""
    print("🔍 Testing package imports...")
    
    # Test core lobster import
    try:
        import lobster
        print("   ✅ Main lobster package imported")
    except ImportError as e:
        print(f"   ❌ Main package import failed: {e}")
        return False
    
    # Test core components
    try:
        from lobster.core.client import AgentClient
        from lobster.core.data_manager_v2 import DataManagerV2
        print("   ✅ Core components imported")
    except ImportError as e:
        print(f"   ❌ Core components failed: {e}")
        return False
    
    # Test CLI
    try:
        from lobster.cli import app
        print("   ✅ CLI module imported")
    except ImportError as e:
        print(f"   ❌ CLI import failed: {e}")
        return False
    
    # Test that cloud components are NOT available (as expected)
    try:
        import lobster_cloud
        print("   ⚠️  WARNING: lobster_cloud still available (should be private)")
        return False
    except ImportError:
        print("   ✅ Cloud components properly excluded")
    
    return True

def test_cli_functionality():
    """Test basic CLI functionality."""
    print("\n🖥️  Testing CLI functionality...")
    
    # Test CLI help command
    try:
        result = subprocess.run([
            sys.executable, "-m", "lobster.cli", "--help"
        ], capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            print("   ✅ CLI help command works")
            if "Multi-Agent Bioinformatics" in result.stdout:
                print("   ✅ CLI description correct")
            else:
                print("   ⚠️  CLI description might need updating")
        else:
            print(f"   ❌ CLI help failed: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print("   ❌ CLI help command timed out")
        return False
    except Exception as e:
        print(f"   ❌ CLI test error: {e}")
        return False
    
    return True

def test_cloud_fallback():
    """Test that cloud key detection works gracefully."""
    print("\n☁️  Testing cloud key detection...")
    
    # Set a test cloud key to see if fallback works
    env = os.environ.copy()
    env['LOBSTER_CLOUD_KEY'] = 'test-key-open-source'
    
    try:
        # This should detect the key but fall back gracefully
        result = subprocess.run([
            sys.executable, "-c", 
            """
import os
os.environ['LOBSTER_CLOUD_KEY'] = 'test-key'
from lobster.cli import init_client
try:
    client = init_client()
    print("SUCCESS: Fallback to local mode works")
except Exception as e:
    print(f"ERROR: {e}")
            """
        ], capture_output=True, text=True, timeout=30, env=env)
        
        if "SUCCESS" in result.stdout:
            print("   ✅ Cloud key fallback works properly")
        elif "Lobster Cloud Not Available" in result.stderr:
            print("   ✅ Proper cloud unavailable message shown")
        else:
            print(f"   ❌ Unexpected behavior: {result.stdout}{result.stderr}")
            return False
            
    except Exception as e:
        print(f"   ❌ Cloud fallback test error: {e}")
        return False
    
    return True

def test_basic_functionality():
    """Test basic bioinformatics functionality."""
    print("\n🧬 Testing basic functionality...")
    
    try:
        from lobster.core.client import AgentClient
        from lobster.core.data_manager_v2 import DataManagerV2
        
        # Initialize with test workspace
        workspace = Path("./test_workspace")
        data_manager = DataManagerV2(workspace_path=workspace)
        
        # Test that we can create a client
        client = AgentClient(data_manager=data_manager)
        
        # Test basic status
        status = client.get_status()
        if status and "session_id" in status:
            print("   ✅ Basic client functionality works")
        else:
            print("   ❌ Client status check failed")
            return False
        
        # Clean up test workspace
        import shutil
        if workspace.exists():
            shutil.rmtree(workspace)
        
        print("   ✅ Basic bioinformatics setup works")
        
    except Exception as e:
        print(f"   ❌ Basic functionality test failed: {e}")
        return False
    
    return True

def main():
    """Run all tests."""
    print("🦞 Lobster AI Open Source Verification")
    print("=====================================")
    
    tests = [
        ("Package Imports", test_imports),
        ("CLI Functionality", test_cli_functionality), 
        ("Cloud Fallback", test_cloud_fallback),
        ("Basic Functionality", test_basic_functionality)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"   💥 {test_name} crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "="*50)
    print("📊 TEST SUMMARY")
    print("="*50)
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
        if result:
            passed += 1
    
    print("-" * 50)
    print(f"🎯 Overall: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("\n🎉 All tests passed! Lobster AI Open Source is ready!")
        print("👥 Ready for community use and contributions")
        return 0
    else:
        print("\n⚠️  Some tests failed. Review issues before public release.")
        return 1

if __name__ == "__main__":
    exit(main())
