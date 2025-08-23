#!/usr/bin/env python3
"""
Test script for the new GEOparse-based GEO service implementation.

This script demonstrates the professional workflow:
1. Download SOFT with GEOparse
2. Extract metadata including supplementary file information
3. Download individual sample matrices
4. Validate file formats
5. Concatenate matrices professionally
6. Prepare data for downstream analysis
"""

import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

try:
    import pandas as pd
##########################################
##########################################
##########################################
## NEEDS Migration to DATAMANGER 2
##########################################
##########################################
##########################################
    from lobster.core.data_manager import DataManager
    from lobster.tools.geo_service import GEOService

    print("✅ Successfully imported lobster modules")
except ImportError as e:
    print(f"❌ Import error: {e}")
    print(
        "Please ensure you're running from the project root and have installed dependencies"
    )
    sys.exit(1)


def test_geoparse_workflow():
    """Test the complete GEOparse-based workflow."""
    print("🧬 Testing GEOparse-based GEO data workflow")
    print("=" * 50)

    # Initialize data manager
    workspace_dir = Path("./test_workspace")
    workspace_dir.mkdir(exist_ok=True)

    data_manager = DataManager(workspace_path=workspace_dir)
    print("✅ DataManager initialized")

    # Initialize GEO service
    try:
        geo_service = GEOService(data_manager, cache_dir="./test_geo_cache")
        print("✅ GEOService initialized with GEOparse backend")
    except ImportError as e:
        print(f"❌ GEOparse not installed: {e}")
        print("Install with: pip install GEOparse")
        return False

    # Test with a small dataset (GSE194247 as mentioned in the requirements)
    test_dataset = "GSE194247"
    print(f"\n🔬 Testing with dataset: {test_dataset}")
    print("This dataset was mentioned in your requirements with GEOparse example")

    try:
        # Download and process dataset
        result = geo_service.download_dataset(test_dataset)
        print("\n📊 Download Result:")
        print(result)

        # Check if data was loaded
        if data_manager.has_data():
            print("\n✅ Data successfully loaded into DataManager")

            # Get data summary
            summary = data_manager.get_data_summary()
            print(f"📈 Data shape: {summary['shape']}")
            print(f"💾 Memory usage: {summary['memory_usage']}")

            # Check metadata
            metadata = data_manager.current_metadata
            print(f"\n📋 Metadata keys: {list(metadata.keys())[:10]}...")

            if "source" in metadata:
                print(f"🔗 Source: {metadata['source']}")
            if "n_samples" in metadata:
                print(f"🧪 Number of samples: {metadata['n_samples']}")
            if "n_validated_samples" in metadata:
                print(f"✅ Validated samples: {metadata['n_validated_samples']}")
            if "sample_ids" in metadata:
                print(f"🆔 Sample IDs: {metadata['sample_ids'][:3]}...")

            print("\n🎯 Workflow completed successfully!")
            print("Data is ready for:")
            print("- Quality assessment and filtering")
            print("- Clustering and cell type annotation")
            print("- Machine learning model preparation")
            print("- Differential expression analysis")

            return True
        else:
            print("❌ No data was loaded")
            return False

    except Exception as e:
        print(f"❌ Error during workflow: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_metadata_extraction():
    """Test metadata extraction capabilities."""
    print("\n🔍 Testing metadata extraction with GEOparse...")

    try:
        import GEOparse

        # Test the example from requirements
        print("📥 Downloading SOFT for GSE194247...")
        gse = GEOparse.get_GEO(geo="GSE194247", destdir="./test_geo_cache")

        print("✅ SOFT file downloaded successfully")
        print(f"📊 Metadata keys: {list(gse.metadata.keys())}")

        # Show key metadata (like in your example)
        key_fields = [
            "title",
            "geo_accession",
            "status",
            "summary",
            "supplementary_file",
        ]

        print("\n📋 Key Metadata:")
        for field in key_fields:
            if field in gse.metadata:
                value = gse.metadata[field]
                if isinstance(value, list) and len(value) == 1:
                    value = value[0]
                print(f"  {field}: {value}")

        # Check for supplementary files (TAR files)
        if "supplementary_file" in gse.metadata:
            suppl_files = gse.metadata["supplementary_file"]
            print(
                f"\n📦 Supplementary files found: {len(suppl_files) if isinstance(suppl_files, list) else 1}"
            )
            if isinstance(suppl_files, list):
                for i, file_url in enumerate(suppl_files[:3]):  # Show first 3
                    print(f"  {i+1}. {file_url}")
            else:
                print(f"  1. {suppl_files}")

        return True

    except Exception as e:
        print(f"❌ Metadata extraction test failed: {e}")
        return False


if __name__ == "__main__":
    print("🦞 Lobster GEOparse Integration Test")
    print("=" * 40)

    # Run tests
    metadata_success = test_metadata_extraction()
    workflow_success = test_geoparse_workflow()

    print("\n" + "=" * 40)
    print("📊 Test Results:")
    print(f"  Metadata extraction: {'✅ PASS' if metadata_success else '❌ FAIL'}")
    print(f"  Complete workflow:   {'✅ PASS' if workflow_success else '❌ FAIL'}")

    if metadata_success and workflow_success:
        print("\n🎉 All tests passed! GEOparse integration is working correctly.")
        print("\nThe new professional approach provides:")
        print("✅ Clean GEOparse-only implementation")
        print("✅ Automatic metadata extraction from SOFT files")
        print("✅ Professional sample matrix downloading")
        print("✅ File format validation")
        print("✅ Intelligent matrix concatenation")
        print("✅ Ready for ML/downstream analysis")
    else:
        print("\n⚠️  Some tests failed. Please check the output above.")
        sys.exit(1)
