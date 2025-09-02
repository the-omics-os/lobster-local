#!/usr/bin/env python3
"""
Test script to verify NCBI query builder fixes.

This script tests that the problematic "published last X months"[Filter] syntax
is no longer generated and that proper date range filters are used instead.
"""

import sys
import os

# Add the lobster directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '.'))

from lobster.tools.providers.ncbi_query_builder import GEOQueryBuilder, NCBIDatabase
from datetime import datetime, timedelta


def test_date_range_formatting():
    """Test that date ranges are properly formatted."""
    print("🧪 Testing date range formatting...")
    
    builder = GEOQueryBuilder()
    
    # Test 1: Date range with start and end
    filters = {
        'date_range': {
            'start': '2020/01/01',
            'end': '2024/12/31'
        }
    }
    
    query = builder.build_query('lung cancer', filters)
    print(f"✅ Date range query: {query}")
    
    # Should contain proper PDAT syntax, not Filter syntax
    assert 'Filter' not in query, f"Query should not contain [Filter]: {query}"
    assert 'PDAT' in query, f"Query should contain [PDAT]: {query}"
    assert '"2020/01/01"[PDAT]' in query, f"Query should contain proper date format: {query}"
    
    return query


def test_no_published_last_months():
    """Test that published_last_n_months filter is no longer supported."""
    print("🧪 Testing removal of published_last_n_months...")
    
    builder = GEOQueryBuilder()
    
    # This should no longer generate any filter since published_last_n_months is removed
    filters = {
        'organism': 'human',
        'published_last_n_months': 120  # This should be ignored
    }
    
    query = builder.build_query('single cell', filters)
    print(f"✅ Query without published_last_n_months: {query}")
    
    # Should NOT contain the problematic Filter syntax
    assert '"published last 120 months"[Filter]' not in query, f"Query should not contain problematic filter: {query}"
    assert 'Filter' not in query, f"Query should not contain any [Filter]: {query}"
    assert 'human[ORGN]' in query, f"Query should contain organism filter: {query}"
    
    return query


def test_combined_filters():
    """Test combination of various filters."""
    print("🧪 Testing combined filters...")
    
    builder = GEOQueryBuilder()
    
    filters = {
        'organism': 'human',
        'entry_type': 'gse',
        'date_range': {
            'start': '2023/01/01',
            'end': '2024/01/01'
        }
    }
    
    query = builder.build_query('RNA-seq', filters)
    print(f"✅ Combined filters query: {query}")
    
    # Verify all expected components
    assert 'RNA-seq' in query, f"Query should contain search term: {query}"
    assert 'human[ORGN]' in query, f"Query should contain organism: {query}"
    assert 'gse[ETYP]' in query, f"Query should contain entry type: {query}"
    assert '"2023/01/01"[PDAT]' in query, f"Query should contain date range: {query}"
    assert 'Filter' not in query, f"Query should not contain [Filter]: {query}"
    
    return query


def test_original_problematic_scenario():
    """Test the original scenario that caused the bug."""
    print("🧪 Testing original problematic scenario...")
    
    builder = GEOQueryBuilder()
    
    # Simulate the original query that caused the issue
    search_terms = 'GSE[ACCN] lung adenocarcinoma "single cell" "smoking status"'
    filters = {
        'organism': 'human',
        'gse': True,  # Entry type filter
        'date_range': {
            'start': '2015/01/01',  # Equivalent to "last 120 months" from 2025
            'end': '2025/01/01'
        }
    }
    
    query = builder.build_query(search_terms, filters)
    print(f"✅ Original scenario (fixed): {query}")
    
    # Verify the problematic syntax is gone
    assert '"published last' not in query.lower(), f"Query should not contain 'published last': {query}"
    assert '[Filter]' not in query, f"Query should not contain [Filter]: {query}"
    
    # Verify proper components are present
    assert 'human[ORGN]' in query, f"Query should contain organism filter: {query}"
    assert 'gse[ETYP]' in query, f"Query should contain entry type: {query}"
    assert '[PDAT]' in query, f"Query should contain date filter: {query}"
    
    return query


def test_query_validation():
    """Test query validation works properly."""
    print("🧪 Testing query validation...")
    
    builder = GEOQueryBuilder()
    
    # Test valid queries
    valid_queries = [
        'cancer AND human[ORGN]',
        '"single cell" AND gse[ETYP]',
        'RNA-seq AND ("2020/01/01"[PDAT] : "2024/01/01"[PDAT])'
    ]
    
    for query in valid_queries:
        is_valid = builder.validate_query(query)
        print(f"✅ Valid query: {query} -> {is_valid}")
        assert is_valid, f"Query should be valid: {query}"
    
    # Test invalid queries
    invalid_queries = [
        'cancer AND human[ORGN',  # Missing closing bracket
        '"single cell AND gse[ETYP]',  # Unmatched quote
        'RNA-seq AND (human[ORGN] AND gse[ETYP]'  # Unmatched parenthesis
    ]
    
    for query in invalid_queries:
        is_valid = builder.validate_query(query)
        print(f"❌ Invalid query: {query} -> {is_valid}")
        assert not is_valid, f"Query should be invalid: {query}"


def create_corrected_url_example():
    """Create an example of the corrected URL."""
    print("🧪 Creating corrected URL example...")
    
    builder = GEOQueryBuilder()
    
    # Original problematic search
    search_terms = 'GSE[ACCN] lung adenocarcinoma "single cell" "smoking status"'
    filters = {
        'organism': 'human',
        'gse': True,
        'date_range': {
            'start': '2015/01/01',
            'end': '2025/01/01'
        }
    }
    
    query = builder.build_query(search_terms, filters)
    encoded_query = builder.url_encode(query)
    
    # Simulate the URL construction
    base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    url_params = {
        'db': 'gds',
        'term': encoded_query,
        'retmode': 'json',
        'retmax': '8',
        'tool': 'lobster',
        'email': 'kevin.yar@homara.ai',
        'usehistory': 'y'
    }
    
    # Create URL
    from urllib.parse import urlencode
    full_url = f"{base_url}?{urlencode(url_params)}"
    
    print(f"✅ Corrected URL: {full_url}")
    print(f"✅ Query part: {query}")
    
    return full_url, query


def main():
    """Run all tests."""
    print("🔧 Testing NCBI Query Builder Fixes")
    print("=" * 50)
    
    try:
        # Run all tests
        test_date_range_formatting()
        print()
        
        test_no_published_last_months()
        print()
        
        test_combined_filters()
        print()
        
        test_original_problematic_scenario()
        print()
        
        test_query_validation()
        print()
        
        create_corrected_url_example()
        print()
        
        print("🎉 All tests passed! NCBI query builder fixes are working correctly.")
        print("\n📋 Summary of fixes:")
        print("• ❌ Removed problematic 'published last N months'[Filter] syntax")
        print("• ✅ Added proper date range filtering using [PDAT] field")
        print("• ✅ Improved date formatting for NCBI E-utilities")
        print("• ✅ Enhanced query validation")
        print("• ✅ All URLs will now be valid for NCBI API calls")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
