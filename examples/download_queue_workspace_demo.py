"""
Demo script showing supervisor reading download_queue via get_content_from_workspace.

This script demonstrates the complete workflow of:
1. Research agent adding entries to download queue (simulated)
2. Supervisor reading queue entries via workspace tool
3. Filtering by status and retrieving details

Usage:
    python examples/download_queue_workspace_demo.py
"""

from pathlib import Path
from tempfile import TemporaryDirectory

from lobster.core.data_manager_v2 import DataManagerV2
from lobster.core.schemas.download_queue import (
    DownloadQueueEntry,
    DownloadStatus,
    StrategyConfig,
)
from lobster.tools.workspace_tool import create_get_content_from_workspace_tool


def main():
    """Demonstrate download_queue workspace functionality."""
    print("=" * 80)
    print("Download Queue Workspace Demo")
    print("=" * 80)
    print()

    # Create temporary workspace
    with TemporaryDirectory() as tmpdir:
        workspace_path = Path(tmpdir) / "workspace"
        workspace_path.mkdir()

        # Initialize DataManagerV2
        data_manager = DataManagerV2(workspace_path=workspace_path)

        # Create workspace tool
        get_content = create_get_content_from_workspace_tool(data_manager)

        print("Step 1: Adding entries to download queue...")
        print("-" * 80)

        # Add entry 1 (PENDING with strategy)
        strategy1 = StrategyConfig(
            strategy_name="MATRIX_FIRST",
            concatenation_strategy="auto",
            confidence=0.95,
            rationale="Processed matrix available with complete metadata",
        )

        entry1 = DownloadQueueEntry(
            entry_id="demo_entry_001",
            dataset_id="GSE180759",
            database="geo",
            priority=7,
            status=DownloadStatus.PENDING,
            metadata={"n_samples": 48, "platform": "GPL96"},
            recommended_strategy=strategy1,
            validation_result={"is_valid": True, "warnings": []},
        )
        data_manager.download_queue.add_entry(entry1)
        print(f"✓ Added entry: {entry1.entry_id} ({entry1.dataset_id})")

        # Add entry 2 (COMPLETED)
        entry2 = DownloadQueueEntry(
            entry_id="demo_entry_002",
            dataset_id="GSE123456",
            database="geo",
            priority=5,
            status=DownloadStatus.COMPLETED,
            modality_name="geo_gse123456",
            metadata={"n_samples": 100, "organism": "Homo sapiens"},
        )
        data_manager.download_queue.add_entry(entry2)
        print(f"✓ Added entry: {entry2.entry_id} ({entry2.dataset_id})")

        # Add entry 3 (PENDING)
        entry3 = DownloadQueueEntry(
            entry_id="demo_entry_003",
            dataset_id="GSE999999",
            database="geo",
            priority=3,
            status=DownloadStatus.PENDING,
        )
        data_manager.download_queue.add_entry(entry3)
        print(f"✓ Added entry: {entry3.entry_id} ({entry3.dataset_id})")

        print()
        print("Step 2: Supervisor reads all download queue entries...")
        print("-" * 80)
        result = get_content.invoke({"workspace": "download_queue"})
        print(result)

        print()
        print("Step 3: Supervisor filters by PENDING status...")
        print("-" * 80)
        result = get_content.invoke(
            {"workspace": "download_queue", "status_filter": "PENDING"}
        )
        print(result)

        print()
        print("Step 4: Supervisor reads specific entry details...")
        print("-" * 80)
        result = get_content.invoke(
            {
                "identifier": "demo_entry_001",
                "workspace": "download_queue",
                "level": "summary",
            }
        )
        print(result)

        print()
        print("Step 5: Supervisor retrieves validation result...")
        print("-" * 80)
        result = get_content.invoke(
            {
                "identifier": "demo_entry_001",
                "workspace": "download_queue",
                "level": "validation",
            }
        )
        print(result)

        print()
        print("Step 6: Supervisor retrieves download strategy...")
        print("-" * 80)
        result = get_content.invoke(
            {
                "identifier": "demo_entry_001",
                "workspace": "download_queue",
                "level": "strategy",
            }
        )
        print(result)

        print()
        print("=" * 80)
        print("Demo completed successfully!")
        print("=" * 80)


if __name__ == "__main__":
    main()
