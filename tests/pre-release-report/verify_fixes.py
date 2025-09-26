#!/usr/bin/env python3
"""
Service Fix Verification Script

This script helps verify that critical fixes have been implemented correctly
by running targeted tests for the most problematic services.

Usage:
    python verify_fixes.py                    # Run all verifications
    python verify_fixes.py --critical         # Only critical services
    python verify_fixes.py --service clustering # Test specific service
"""

import argparse
import importlib
import json
import sys
from pathlib import Path
from typing import Dict, List, Any
import traceback

# Add parent directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

def test_service_import(service_name: str) -> Dict[str, Any]:
    """Test if a service can be imported successfully."""
    try:
        module_path = f"lobster.tools.{service_name.replace('.py', '')}"
        module = importlib.import_module(module_path)
        return {
            "status": "PASS",
            "message": f"Successfully imported {service_name}",
            "module": module
        }
    except Exception as e:
        return {
            "status": "FAIL",
            "message": f"Import failed: {str(e)}",
            "error": traceback.format_exc()
        }

def test_service_instantiation(service_name: str, module) -> Dict[str, Any]:
    """Test if a service can be instantiated."""
    try:
        # Get the service class name (convert filename to class name)
        class_name = ''.join([word.capitalize() for word in service_name.replace('.py', '').split('_')])
        service_class = getattr(module, class_name)
        service_instance = service_class()
        return {
            "status": "PASS",
            "message": f"Successfully instantiated {class_name}",
            "instance": service_instance
        }
    except Exception as e:
        return {
            "status": "FAIL",
            "message": f"Instantiation failed: {str(e)}",
            "error": traceback.format_exc()
        }

def run_basic_functionality_test(service_name: str, instance) -> Dict[str, Any]:
    """Run basic functionality tests for a service."""
    try:
        # Test that the service has expected methods
        expected_methods = ['__init__']

        # Service-specific method checks
        if 'clustering' in service_name:
            expected_methods.extend(['perform_clustering', 'compute_umap'])
        elif 'proteomics_visualization' in service_name:
            expected_methods.extend(['create_heatmap', 'create_volcano_plot'])
        elif 'proteomics_quality' in service_name:
            expected_methods.extend(['assess_missing_values', 'calculate_cv'])
        elif 'pseudobulk' in service_name:
            expected_methods.extend(['aggregate_to_pseudobulk'])

        missing_methods = []
        for method in expected_methods:
            if not hasattr(instance, method):
                missing_methods.append(method)

        if missing_methods:
            return {
                "status": "PARTIAL",
                "message": f"Missing expected methods: {missing_methods}"
            }

        return {
            "status": "PASS",
            "message": "All expected methods present"
        }

    except Exception as e:
        return {
            "status": "FAIL",
            "message": f"Basic functionality test failed: {str(e)}",
            "error": traceback.format_exc()
        }

def run_service_verification(service_name: str) -> Dict[str, Any]:
    """Run complete verification for a single service."""
    print(f"\n🧪 Testing {service_name}...")

    results = {
        "service": service_name,
        "tests": {},
        "overall_status": "UNKNOWN"
    }

    # Test 1: Import
    import_result = test_service_import(service_name)
    results["tests"]["import"] = import_result
    print(f"  Import: {import_result['status']} - {import_result['message']}")

    if import_result["status"] != "PASS":
        results["overall_status"] = "FAIL"
        return results

    # Test 2: Instantiation
    instantiation_result = test_service_instantiation(service_name, import_result["module"])
    results["tests"]["instantiation"] = instantiation_result
    print(f"  Instantiation: {instantiation_result['status']} - {instantiation_result['message']}")

    if instantiation_result["status"] != "PASS":
        results["overall_status"] = "FAIL"
        return results

    # Test 3: Basic functionality
    functionality_result = run_basic_functionality_test(service_name, instantiation_result["instance"])
    results["tests"]["functionality"] = functionality_result
    print(f"  Functionality: {functionality_result['status']} - {functionality_result['message']}")

    # Determine overall status
    statuses = [test["status"] for test in results["tests"].values()]
    if all(status == "PASS" for status in statuses):
        results["overall_status"] = "PASS"
    elif any(status == "FAIL" for status in statuses):
        results["overall_status"] = "FAIL"
    else:
        results["overall_status"] = "PARTIAL"

    return results

def main():
    parser = argparse.ArgumentParser(description="Verify service fixes")
    parser.add_argument("--critical", action="store_true",
                       help="Only test critical priority services")
    parser.add_argument("--service", type=str,
                       help="Test specific service")
    args = parser.parse_args()

    # Define service categories
    critical_services = [
        "proteomics_visualization_service.py",
        "proteomics_quality_service.py"
    ]

    high_priority_services = [
        "clustering_service.py",
        "pseudobulk_service.py",
        "proteomics_differential_service.py"
    ]

    all_services = [
        # Transcriptomics
        "preprocessing_service.py",
        "quality_service.py",
        "clustering_service.py",
        "enhanced_singlecell_service.py",
        "bulk_rnaseq_service.py",
        "pseudobulk_service.py",
        "differential_formula_service.py",
        "concatenation_service.py",
        # Proteomics
        "proteomics_preprocessing_service.py",
        "proteomics_quality_service.py",
        "proteomics_analysis_service.py",
        "proteomics_differential_service.py",
        "proteomics_visualization_service.py",
        # Data/Publication
        "geo_service.py",
        "publication_service.py",
        "visualization_service.py"
    ]

    # Determine which services to test
    if args.service:
        if not args.service.endswith('.py'):
            args.service += '.py'
        services_to_test = [args.service]
    elif args.critical:
        services_to_test = critical_services
    else:
        services_to_test = critical_services + high_priority_services

    print(f"🦞 Lobster AI Services Fix Verification")
    print(f"Testing {len(services_to_test)} services...")

    all_results = []

    for service in services_to_test:
        result = run_service_verification(service)
        all_results.append(result)

    # Summary
    print(f"\n📊 VERIFICATION SUMMARY")
    print(f"{'='*50}")

    pass_count = sum(1 for r in all_results if r["overall_status"] == "PASS")
    partial_count = sum(1 for r in all_results if r["overall_status"] == "PARTIAL")
    fail_count = sum(1 for r in all_results if r["overall_status"] == "FAIL")

    print(f"✅ PASS: {pass_count}")
    print(f"⚠️  PARTIAL: {partial_count}")
    print(f"❌ FAIL: {fail_count}")
    print(f"📈 Success Rate: {pass_count/len(all_results)*100:.1f}%")

    # Save detailed results
    results_file = Path(__file__).parent / "verification_results.json"
    with open(results_file, 'w') as f:
        json.dump({
            "timestamp": "2025-09-25",
            "services_tested": len(all_results),
            "summary": {
                "pass": pass_count,
                "partial": partial_count,
                "fail": fail_count,
                "success_rate": f"{pass_count/len(all_results)*100:.1f}%"
            },
            "detailed_results": all_results
        }, f, indent=2)

    print(f"\n💾 Detailed results saved to: {results_file}")

    # Exit with appropriate code
    sys.exit(0 if fail_count == 0 else 1)

if __name__ == "__main__":
    main()