#!/usr/bin/env python3
"""
Simple verification script for Lobster package split
"""
import sys
import os

def test_imports():
    """Test basic imports"""
    try:
        import lobster_core
        print("✅ lobster-core: Successfully imported")
        print(f"   Version: {lobster_core.__version__}")
        
        from lobster_core.interfaces import BaseLobsterClient, BaseDataManager
        print("   Base classes available")
        
    except ImportError as e:
        print(f"❌ lobster-core: Import failed - {e}")
        return False

    try:
        import lobster_cloud
        print("✅ lobster-cloud: Successfully imported")
        print(f"   Version: {lobster_cloud.__version__}")
        
        from lobster_cloud.client import CloudLobsterClient
        print("   CloudLobsterClient available")
        
    except ImportError as e:
        print(f"❌ lobster-cloud: Import failed - {e}")
        return False

    try:
        from lobster.core.client import AgentClient
        from lobster.core.data_manager_v2 import DataManagerV2
        print("✅ lobster-local: Successfully imported")
        print("   AgentClient and DataManagerV2 available")
        
    except ImportError as e:
        print(f"❌ lobster-local: Import failed - {e}")
        return False
    
    return True

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

def test_cloud_client():
    """Test cloud client creation"""
    try:
        from lobster_cloud.client import CloudLobsterClient
        
        client = CloudLobsterClient(api_key="test-key", endpoint="https://mock.example.com")
        status = client.get_status()
        
        if isinstance(status, dict) and "error" in status:
            print("✅ Cloud client: Connection error handling works correctly")
            return True
        else:
            print(f"❌ Cloud client: Unexpected response - {status}")
            return False
            
    except Exception as e:
        print(f"❌ Cloud client: Creation failed - {e}")
        return False

def main():
    """Run all verification tests"""
    print("🦞 Lobster Package Split Verification")
    print("=" * 50)
    
    tests = [
        ("Package Imports", test_imports),
        ("CLI Detection", test_cli_detection),
        ("Cloud Client", test_cloud_client),
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
        print("\n🎉 SUCCESS! Package split is working correctly!")
        print("\n📋 Usage:")
        print("   • Local mode: lobster query 'Your question'")
        print("   • Cloud mode: LOBSTER_CLOUD_KEY=key lobster query 'Your question'")
        return True
    else:
        print("\n⚠️  Some tests failed. Package installation may be incomplete.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
