#!/usr/bin/env python3
"""
Simple test script to verify GDS support in the Lobster platform.
Tests the ability to fetch metadata from GDS identifiers.
"""

import sys
import urllib.request
import urllib.parse
import json

def test_gds_api_call():
    """Test direct NCBI E-utilities call for GDS5826"""
    print("Testing direct NCBI E-utilities API call for GDS5826...")
    
    # Build the API URL
    base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
    params = {
        'db': 'gds',
        'id': '5826',
        'retmode': 'json'
    }
    
    url = f"{base_url}?{urllib.parse.urlencode(params)}"
    print(f"API URL: {url}")
    
    try:
        with urllib.request.urlopen(url) as response:
            data = json.loads(response.read().decode())
            
        print("\n✅ API call successful!")
        print(f"Response keys: {list(data.keys())}")
        
        if 'result' in data and '5826' in data['result']:
            gds_info = data['result']['5826']
            print(f"Title: {gds_info.get('title', 'N/A')}")
            print(f"Summary: {gds_info.get('summary', 'N/A')[:100]}...")
            print(f"GSE: {gds_info.get('gse', 'N/A')}")
            
            if 'gse' in gds_info:
                print(f"\n✅ Found GSE mapping: GDS5826 → {gds_info['gse']}")
                return True
            else:
                print("\n❌ No GSE mapping found")
                return False
        else:
            print("\n❌ Unexpected response structure")
            return False
            
    except Exception as e:
        print(f"\n❌ API call failed: {e}")
        return False

def test_geo_service():
    """Test GEOService with GDS identifier"""
    print("\n" + "="*50)
    print("Testing GEOService with GDS identifier...")
    
    try:
        # Import required modules
        from lobster.tools.geo_service import GEOService
        from lobster.core.data_manager_v2 import DataManagerV2
        
        # Create minimal DataManagerV2 instance
        data_manager = DataManagerV2()
        geo_service = GEOService(data_manager)
        
        # Test fetch_metadata_only with GDS identifier
        print("Calling geo_service.fetch_metadata_only('GDS5826')...")
        
        # This should use our new GDS handling logic
        result = geo_service.fetch_metadata_only('GDS5826')
        
        print(f"\n✅ GEOService call successful!")
        print(f"Result type: {type(result)}")
        print(f"Result preview: {str(result)[:200]}...")
        
        return True
        
    except Exception as e:
        print(f"\n❌ GEOService test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_ncbi_query_builder():
    """Test NCBIQueryBuilder enum for GDS support"""
    print("\n" + "="*50)
    print("Testing NCBIQueryBuilder enum for GDS support...")
    
    try:
        from lobster.tools.providers.ncbi_query_builder import NCBIDatabase
        
        # Check if GDS enum exists
        gds_db = NCBIDatabase.GDS
        print(f"✅ GDS database enum found: {gds_db.value}")
        
        # Check all available databases
        all_dbs = [db.value for db in NCBIDatabase]
        print(f"Available databases: {all_dbs}")
        
        if 'gds' in all_dbs:
            print("✅ GDS database is properly registered")
            return True
        else:
            print("❌ GDS database not found in enum")
            return False
            
    except Exception as e:
        print(f"❌ NCBIQueryBuilder test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 Testing GDS Support Implementation")
    print("="*50)
    
    tests = [
        ("NCBI API Call", test_gds_api_call),
        ("NCBI Query Builder", test_ncbi_query_builder),
        ("GEO Service", test_geo_service),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results[test_name] = False
    
    # Summary
    print("\n" + "="*50)
    print("🏁 TEST SUMMARY")
    print("="*50)
    
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {test_name}")
    
    overall_success = all(results.values())
    if overall_success:
        print("\n🎉 All tests passed! GDS support is working correctly.")
    else:
        print(f"\n⚠️  {sum(results.values())}/{len(results)} tests passed. Some issues detected.")
    
    return overall_success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
