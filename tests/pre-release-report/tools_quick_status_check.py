#!/usr/bin/env python3
"""
Quick Status Check for Lobster AI Tools Testing
Provides rapid assessment of critical testing gaps and failures.

Usage:
    python tests/pre-release-report/tools_quick_status_check.py
"""

import subprocess
import sys
from pathlib import Path

def run_command(cmd, capture_output=True):
    """Run a command and return result."""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=capture_output, text=True)
        return result.returncode == 0, result.stdout, result.stderr
    except Exception as e:
        return False, "", str(e)

def check_critical_services_import():
    """Check if critical services can be imported."""
    critical_services = [
        "lobster.tools.preprocessing_service.PreprocessingService",
        "lobster.tools.quality_service.QualityService",
        "lobster.tools.clustering_service.ClusteringService",
        "lobster.tools.enhanced_singlecell_service.EnhancedSingleCellService",
        "lobster.tools.geo_service.GEOService",
        "lobster.tools.visualization_service.SingleCellVisualizationService"
    ]

    print("🔍 CHECKING CRITICAL SERVICE IMPORTS...")
    import_results = {}

    for service in critical_services:
        module_path, class_name = service.rsplit('.', 1)
        try:
            exec(f"from {module_path} import {class_name}")
            import_results[service] = "✅ OK"
        except Exception as e:
            import_results[service] = f"❌ FAILED: {str(e)[:50]}..."

    return import_results

def run_working_tests():
    """Run tests that are known to work."""
    print("\n🧪 RUNNING WORKING TESTS...")

    working_tests = [
        "tests/unit/tools/test_concatenation_service.py",
        "tests/unit/tools/test_manual_annotation_service.py"
    ]

    for test_file in working_tests:
        success, stdout, stderr = run_command(f".venv/bin/pytest {test_file} -v --tb=no")
        if success:
            print(f"✅ {test_file}: PASSED")
        else:
            print(f"❌ {test_file}: FAILED")

    return True

def check_critical_failures():
    """Check for critical test failures."""
    print("\n🔥 CHECKING CRITICAL SERVICE FAILURES...")

    critical_test_files = [
        "tests/unit/tools/test_pseudobulk_service.py",
        "tests/unit/tools/test_differential_formula_service.py",
        "tests/unit/tools/test_bulk_rnaseq_service.py"
    ]

    failure_summary = {}

    for test_file in critical_test_files:
        if Path(test_file).exists():
            success, stdout, stderr = run_command(f".venv/bin/pytest {test_file} --tb=no -q")
            if not success:
                # Count failures
                failure_count = stderr.count('FAILED') + stdout.count('FAILED')
                failure_summary[test_file] = f"❌ {failure_count} failures"
            else:
                failure_summary[test_file] = "✅ All passed"
        else:
            failure_summary[test_file] = "⚠️ File not found"

    return failure_summary

def check_test_coverage():
    """Get basic coverage information."""
    print("\n📊 CHECKING TEST COVERAGE...")

    success, stdout, stderr = run_command(
        ".venv/bin/pytest tests/unit/tools/test_concatenation_service.py tests/unit/tools/test_manual_annotation_service.py --cov=lobster.tools --cov-report=term --tb=no -q"
    )

    if success and "TOTAL" in stdout:
        lines = stdout.split('\n')
        for line in lines:
            if "TOTAL" in line:
                parts = line.split()
                if len(parts) >= 4:
                    coverage_pct = parts[-1]
                    return f"Current coverage: {coverage_pct}"

    return "Coverage check failed"

def count_services_without_tests():
    """Count services that have no unit tests."""
    tools_dir = Path("lobster/tools")
    tests_dir = Path("tests/unit/tools")

    if not tools_dir.exists() or not tests_dir.exists():
        return "❌ Directories not found"

    # Get all service files
    service_files = [f for f in tools_dir.glob("*.py") if f.name != "__init__.py"]

    # Get existing test files
    test_files = set()
    for test_file in tests_dir.glob("test_*.py"):
        # Extract service name from test file
        service_name = test_file.name.replace("test_", "").replace(".py", "") + ".py"
        test_files.add(service_name)

    services_without_tests = []
    for service_file in service_files:
        if service_file.name not in test_files:
            services_without_tests.append(service_file.name)

    return len(services_without_tests), len(service_files), services_without_tests

def generate_status_report():
    """Generate comprehensive status report."""
    print("=" * 60)
    print("🧪 LOBSTER AI TOOLS - QUICK STATUS CHECK")
    print("=" * 60)

    # Import checks
    import_results = check_critical_services_import()
    for service, status in import_results.items():
        print(f"  {service.split('.')[-1]}: {status}")

    # Working tests
    run_working_tests()

    # Critical failures
    failure_summary = check_critical_failures()
    for test_file, status in failure_summary.items():
        test_name = Path(test_file).name
        print(f"  {test_name}: {status}")

    # Coverage
    coverage_info = check_test_coverage()
    print(f"\n📊 {coverage_info}")

    # Services without tests
    untested_count, total_count, untested_services = count_services_without_tests()
    print(f"\n📋 SERVICES WITHOUT TESTS: {untested_count}/{total_count}")

    if untested_count > 0:
        print("\n🚨 CRITICAL SERVICES MISSING TESTS:")
        critical_services = [
            "preprocessing_service.py",
            "quality_service.py",
            "clustering_service.py",
            "enhanced_singlecell_service.py",
            "geo_service.py",
            "visualization_service.py"
        ]

        for service in critical_services:
            if service in untested_services:
                print(f"  ❌ {service}")
            else:
                print(f"  ✅ {service}")

    # Overall status
    print("\n" + "=" * 60)
    if untested_count < 5 and len([r for r in import_results.values() if "FAILED" in r]) == 0:
        print("🟡 STATUS: IMPROVING - Some critical issues remain")
    elif untested_count > 15:
        print("🔴 STATUS: RELEASE BLOCKED - Major testing gaps")
    else:
        print("🟠 STATUS: NEEDS WORK - Significant issues to address")
    print("=" * 60)

if __name__ == "__main__":
    generate_status_report()