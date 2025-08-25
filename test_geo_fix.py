#!/usr/bin/env python3
"""Test script to verify the GEO dataset ID bug fix."""

import pandas as pd
from lobster.core.data_manager_v2 import DataManagerV2

def test_geo_fix():
    """Test that dataset_id and other metadata kwargs are handled correctly."""
    print("Testing GEO dataset ID bug fix...")
    
    # Initialize DataManagerV2
    data_manager = DataManagerV2()
    
    # Create a simple test DataFrame
    test_data = pd.DataFrame({
        'Gene1': [1.0, 2.0, 3.0],
        'Gene2': [4.0, 5.0, 6.0],
        'Gene3': [7.0, 8.0, 9.0]
    }, index=['Cell1', 'Cell2', 'Cell3'])
    
    # Test loading with metadata fields that were causing the error
    try:
        adata = data_manager.load_modality(
            name="test_geo_modality",
            source=test_data,
            adapter="transcriptomics_single_cell",
            validate=True,
            dataset_id="GSE123456",  # This was causing the error
            dataset_type="GEO",
            source_metadata={"test": "metadata"},
            processing_date="2025-01-24",
            download_source="test",
            processing_method="test_method"
        )
        
        print("✓ Successfully loaded modality with metadata fields")
        print(f"  Shape: {adata.shape}")
        print(f"  Metadata stored in uns: {list(adata.uns.keys())}")
        
        # Verify metadata was stored correctly
        assert "dataset_id" in adata.uns
        assert adata.uns["dataset_id"] == "GSE123456"
        print("✓ Metadata correctly stored in uns")
        
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_geo_fix()
    if success:
        print("\n✅ All tests passed! The bug has been fixed.")
    else:
        print("\n❌ Tests failed! The bug is not fixed.")
