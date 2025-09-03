#!/usr/bin/env python3
"""
Test script to verify robust null handling in DataExpertAssistant.

This script tests various null value scenarios to ensure the code handles them properly.
"""

import sys
import os

# Add the lobster directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'lobster'))

from lobster.agents.data_expert_assistant import DataExpertAssistant

def test_null_sanitization():
    """Test the null value sanitization function."""
    print("Testing null value sanitization...")
    
    assistant = DataExpertAssistant()
    
    # Test cases with various null representations
    test_cases = [
        # Case 1: Actual Python None values
        {
            "input": {
                "summary_file_name": None,
                "summary_file_type": None,
                "raw_data_available": None
            },
            "expected": {
                "summary_file_name": "NA",
                "summary_file_type": "NA", 
                "raw_data_available": False
            }
        },
        
        # Case 2: String null representations
        {
            "input": {
                "summary_file_name": "null",
                "summary_file_type": "None",
                "processed_matrix_name": "NULL",
                "cell_annotation_name": "n/a",
                "raw_data_available": "false"
            },
            "expected": {
                "summary_file_name": "NA",
                "summary_file_type": "NA",
                "processed_matrix_name": "NA",
                "cell_annotation_name": "NA",
                "raw_data_available": "false"  # String "false" should remain as string
            }
        },
        
        # Case 3: Empty strings and whitespace
        {
            "input": {
                "summary_file_name": "",
                "summary_file_type": "   ",
                "processed_matrix_name": " NA ",
                "raw_data_available": True
            },
            "expected": {
                "summary_file_name": "NA",
                "summary_file_type": "NA",
                "processed_matrix_name": "NA",
                "raw_data_available": True
            }
        },
        
        # Case 4: Valid values should remain unchanged
        {
            "input": {
                "summary_file_name": "GSE123456_summary",
                "summary_file_type": "xlsx",
                "processed_matrix_name": "GSE123456_matrix",
                "raw_data_available": True
            },
            "expected": {
                "summary_file_name": "GSE123456_summary",
                "summary_file_type": "xlsx",
                "processed_matrix_name": "GSE123456_matrix", 
                "raw_data_available": True
            }
        }
    ]
    
    all_passed = True
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\nTest Case {i}:")
        print(f"Input: {test_case['input']}")
        
        result = assistant._sanitize_null_values(test_case["input"])
        print(f"Result: {result}")
        print(f"Expected: {test_case['expected']}")
        
        # Check if result matches expected
        passed = True
        for key, expected_val in test_case["expected"].items():
            if key not in result:
                print(f"❌ Missing key: {key}")
                passed = False
            elif result[key] != expected_val:
                print(f"❌ Mismatch for {key}: got {result[key]}, expected {expected_val}")
                passed = False
        
        if passed:
            print("✅ PASSED")
        else:
            print("❌ FAILED")
            all_passed = False
    
    return all_passed

def test_file_validation():
    """Test the _has_valid_file function."""
    print("\n" + "="*50)
    print("Testing file validation...")
    
    assistant = DataExpertAssistant()
    
    test_cases = [
        ("valid_filename", True),
        ("GSE123456_matrix", True), 
        ("", False),
        ("   ", False),
        ("NA", False),
        ("N/A", False),
        ("null", False),
        ("None", False),
        ("NULL", False),
        (None, False),  # This should be handled gracefully
    ]
    
    all_passed = True
    
    for filename, expected in test_cases:
        try:
            result = assistant._has_valid_file(filename)
            if result == expected:
                print(f"✅ '{filename}' -> {result} (expected {expected})")
            else:
                print(f"❌ '{filename}' -> {result} (expected {expected})")
                all_passed = False
        except Exception as e:
            print(f"❌ '{filename}' -> Error: {e}")
            all_passed = False
    
    return all_passed

def test_format_display():
    """Test the _format_file_display function."""
    print("\n" + "="*50)
    print("Testing file display formatting...")
    
    assistant = DataExpertAssistant()
    
    test_cases = [
        ("valid_file", "txt", "valid_file.txt"),
        ("GSE123456_matrix", "csv", "GSE123456_matrix.csv"),
        ("", "txt", "Not found"),
        ("NA", "txt", "Not found"),
        ("valid_file", "", "valid_file"),
        ("valid_file", "NA", "valid_file"),
        ("", "", "Not found"),
        ("NA", "NA", "Not found"),
    ]
    
    all_passed = True
    
    for filename, filetype, expected in test_cases:
        try:
            result = assistant._format_file_display(filename, filetype)
            if result == expected:
                print(f"✅ ('{filename}', '{filetype}') -> '{result}'")
            else:
                print(f"❌ ('{filename}', '{filetype}') -> '{result}' (expected '{expected}')")
                all_passed = False
        except Exception as e:
            print(f"❌ ('{filename}', '{filetype}') -> Error: {e}")
            all_passed = False
    
    return all_passed

def main():
    """Run all tests."""
    print("🧪 Testing Robust Null Handling in DataExpertAssistant")
    print("=" * 60)
    
    test_results = []
    
    # Run tests
    test_results.append(test_null_sanitization())
    test_results.append(test_file_validation())
    test_results.append(test_format_display())
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY:")
    print(f"Null Sanitization: {'✅ PASSED' if test_results[0] else '❌ FAILED'}")
    print(f"File Validation: {'✅ PASSED' if test_results[1] else '❌ FAILED'}")  
    print(f"Display Formatting: {'✅ PASSED' if test_results[2] else '❌ FAILED'}")
    
    if all(test_results):
        print("\n🎉 All tests PASSED! The null handling implementation is robust.")
        return 0
    else:
        print("\n💥 Some tests FAILED. Please check the implementation.")
        return 1

if __name__ == "__main__":
    exit(main())
