#!/usr/bin/env python3
"""
Test script to validate both local and cloud versions of Lobster
"""

import os
import sys
import subprocess
from pathlib import Path

def test_imports():
    """Test that all packages can be imported correctly"""
    print("🔍 Testing package imports...")
    
    try:
        # Test core package
        from lobster_core.interfaces.base_client import BaseLobsterClient, BaseDataManager
        print("   ✓ lobster-core imports successful")
        
        # Test cloud package
        from lobster_cloud.client import CloudLobsterClient
        print("   ✓ lobster-cloud imports successful")
        
        # Test local package (should be available as 'lobster')
        from lobster.core.client import AgentClient
        from lobster.core.data_manager_v2 import DataManagerV2
        print("   ✓ lobster-local imports successful")
        
        return True
        
    except ImportError as e:
        print(f"   ✗ Import failed: {e}")
        return False

def test_cli_local():
    """Test CLI in local mode"""
    print("\n🖥️  Testing Local CLI Mode...")
    
    # Make sure LOBSTER_CLOUD_KEY is not set
    env = os.environ.copy()
    env.pop('LOBSTER_CLOUD_KEY', None)
    
    try:
        # Test simple query command
        result = subprocess.run([
            sys.executable, '-m', 'lobster.cli', 'query', 
            'What is the difference between DNA and RNA?'
        ], 
        env=env, 
        capture_output=True, 
        text=True, 
        timeout=60
        )
        
        if result.returncode == 0:
            print("   ✓ Local CLI query successful")
            print(f"   📄 Sample output: {result.stdout[:200]}...")
            return True
        else:
            print(f"   ✗ Local CLI query failed: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print("   ✗ Local CLI query timed out")
        return False
    except Exception as e:
        print(f"   ✗ Local CLI test error: {e}")
        return False

def test_cli_cloud_mock():
    """Test CLI in cloud mode (with mock API key)"""
    print("\n☁️  Testing Cloud CLI Mode (Mock)...")
    
    # Set mock API key
    env = os.environ.copy()
    env['LOBSTER_CLOUD_KEY'] = 'test-mock-key-12345'
    env['LOBSTER_ENDPOINT'] = 'https://mock-endpoint.example.com'
    
    try:
        # Test that it tries to use cloud mode (will fail due to mock endpoint)
        result = subprocess.run([
            sys.executable, '-m', 'lobster.cli', 'query', 
            'Test cloud mode'
        ], 
        env=env, 
        capture_output=True, 
        text=True, 
        timeout=30
        )
        
        # Should fail due to connection error but show cloud mode was attempted
        output = result.stdout + result.stderr
        if "Using Lobster Cloud" in output or "Could not connect to Lobster Cloud" in output:
            print("   ✓ Cloud mode detection successful")
            print("   📄 Expected connection error to mock endpoint")
            return True
        else:
            print(f"   ✗ Cloud mode not detected: {output[:200]}...")
            return False
            
    except subprocess.TimeoutExpired:
        print("   ✓ Cloud mode attempted (timeout expected with mock endpoint)")
        return True
    except Exception as e:
        print(f"   ✗ Cloud CLI test error: {e}")
        return False

def test_direct_import():
    """Test direct use of the cloud client"""
    print("\n🔌 Testing Direct Cloud Client Usage...")
    
    try:
        from lobster_cloud.client import CloudLobsterClient
        
        # Create client with mock credentials
        client = CloudLobsterClient(
            api_key="test-key", 
            endpoint="https://mock.example.com"
        )
        
        # Test get_status (will fail but should handle gracefully)
        status = client.get_status()
        
        if isinstance(status, dict) and "error" in status:
            print("   ✓ Cloud client handles connection errors gracefully")
            print(f"   📄 Error message: {status.get('error', 'Unknown')[:100]}...")
            return True
        else:
            print(f"   ✗ Unexpected status response: {status}")
            return False
            
    except Exception as e:
        print(f"   ✗ Direct cloud client test error: {e}")
        return False

def test_package_structure():
    """Test that the package structure is correct"""
    print("\n📦 Testing Package Structure...")
    
    checks = []
    
    # Check that core package has proper structure
    try:
        import lobster_core
        checks.append(("lobster_core version", hasattr(lobster_core, '__version__')))
        
        from lobster_core.interfaces import BaseLobsterClient
        checks.append(("BaseLobsterClient available", True))
        
    except Exception as e:
        checks.append(("lobster_core structure", False))
    
    # Check that cloud package has proper structure  
    try:
        import lobster_cloud
        checks.append(("lobster_cloud version", hasattr(lobster_cloud, '__version__')))
        
        from lobster_cloud import CloudLobsterClient
        checks.append(("CloudLobsterClient available", True))
        
    except Exception as e:
        checks.append(("lobster_cloud structure", False))
    
    # Check that the CLI is accessible
    try:
        import lobster.cli
        checks.append(("CLI module accessible", True))
    except Exception as e:
        checks.append(("CLI module accessible", False))
    
    # Print results
    all_passed = True
    for check_name, passed in checks:
        status = "✓" if passed else "✗"
        print(f"   {status} {check_name}")
        if not passed:
            all_passed = False
    
    return all_passed

def main():
    """Run all tests"""
    print("🦞 Lobster Cloud/Local Split Test Suite")
    print("=" * 50)
    
    tests = [
        ("Package Imports", test_imports),
        ("Package Structure", test_package_structure),
        ("Local CLI Mode", test_cli_local),
        ("Cloud CLI Mode", test_cli_cloud_mock),
        ("Direct Cloud Client", test_direct_import),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"\n❌ {test_name} failed with exception: {e}")
            results[test_name] = False
    
    # Print summary
    print("\n" + "=" * 50)
    print("📊 Test Results Summary")
    print("=" * 50)
    
    passed = sum(1 for result in results.values() if result)
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status:8} {test_name}")
    
    print(f"\n🎯 Overall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The package split is working correctly.")
        print("\n📋 Quick usage guide:")
        print("   • Local mode: lobster query 'Your question'")
        print("   • Cloud mode: LOBSTER_CLOUD_KEY=your-key lobster query 'Your question'")
        print("   • Interactive: lobster chat")
        return True
    else:
        print("⚠️  Some tests failed. Please check the package installation.")
        print("\n🔧 Try running: ./dev_install.sh")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
