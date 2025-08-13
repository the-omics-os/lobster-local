import pandas as pd
import numpy as np
from pathlib import Path

def create_test_files():
    """Create test data files for file upload testing."""
    test_data_dir = Path("tests/data")
    test_data_dir.mkdir(exist_ok=True)
    
    # Single-cell expression matrix
    np.random.seed(42)
    sc_data = pd.DataFrame(
        np.random.negative_binomial(5, 0.3, (1000, 500)),
        index=[f"Cell_{i:04d}" for i in range(1000)],
        columns=[f"Gene_{i:04d}" for i in range(500)]
    )
    sc_data.to_csv(test_data_dir / "single_cell_matrix.csv")
    sc_data.to_excel(test_data_dir / "single_cell_matrix.xlsx")
    
    # Bulk RNA-seq expression matrix
    bulk_data = pd.DataFrame(
        np.random.negative_binomial(20, 0.1, (12, 2000)),
        index=[f"Sample_{i:02d}" for i in range(12)],
        columns=[f"Gene_{i:05d}" for i in range(2000)]
    )
    bulk_data.to_csv(test_data_dir / "bulk_rnaseq_matrix.csv")
    
    # Sample metadata
    metadata = pd.DataFrame({
        'condition': ['control'] * 6 + ['treatment'] * 6,
        'batch': [1, 1, 1, 2, 2, 2] * 2,
        'age': np.random.randint(20, 80, 12)
    }, index=bulk_data.index)
    metadata.to_csv(test_data_dir / "sample_metadata.csv")
    
    # Small FASTQ files for testing
    fastq_content = """@read1
ATCGATCGATCGATCGATCGATCGATCGATCG
+
IIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII
@read2
TCGATCGATCGATCGATCGATCGATCGATCGA
+
IIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII
"""
    
    for i in range(3):
        with open(test_data_dir / f"sample_{i}.fastq", "w") as f:
            f.write(fastq_content * 100)  # Make it bigger
    
    print(f"Test data files created in {test_data_dir}")

if __name__ == "__main__":
    create_test_files()