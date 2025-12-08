#!/usr/bin/env python3
"""
Download OGB datasets locally for transfer to cluster.
This avoids repeated downloads on the cluster and potential network issues.
"""

import os
from pathlib import Path
from typing import Union

from ogb.graphproppred import PygGraphPropPredDataset


def download_ogb_datasets() -> None:
    """Download OGB datasets to local directory."""

    # Create local data directory
    local_data_dir = "./graph_datasets_local"
    os.makedirs(local_data_dir, exist_ok=True)

    print("🚀 Starting OGB dataset downloads...")
    print(f"📁 Download directory: {os.path.abspath(local_data_dir)}")

    # Download ogbg-molhiv
    print("\n📊 Downloading ogbg-molhiv...")
    try:
        molhiv = PygGraphPropPredDataset(name="ogbg-molhiv", root=local_data_dir)
        print(f"✅ ogbg-molhiv downloaded: {len(molhiv)} graphs")
        print(
            f"   Size on disk: ~{get_dir_size(os.path.join(local_data_dir, 'ogbg_molhiv')):.1f} MB"
        )
    except Exception as e:
        print(f"❌ Failed to download ogbg-molhiv: {e}")

    # Download ogbg-molpcba
    print("\n📊 Downloading ogbg-molpcba...")
    try:
        molpcba = PygGraphPropPredDataset(name="ogbg-molpcba", root=local_data_dir)
        print(f"✅ ogbg-molpcba downloaded: {len(molpcba)} graphs")
        print(
            f"   Size on disk: ~{get_dir_size(os.path.join(local_data_dir, 'ogbg_molpcba')):.1f} MB"
        )
    except Exception as e:
        print(f"❌ Failed to download ogbg-molpcba: {e}")

    total_size = get_dir_size(local_data_dir)
    print("\n🎉 Downloads complete!")
    print(f"📁 Total size: ~{total_size:.1f} MB")
    print(f"📍 Location: {os.path.abspath(local_data_dir)}")

    print("\n📤 To transfer to cluster, run:")
    print(f"cd {local_data_dir}")
    print(
        "scp -r * rpellegrinext@login.rc.fas.harvard.edu:/n/holylabs/LABS/mweber_lab/Everyone/rpellegrin/graph_moes/graph_datasets/"
    )


def get_dir_size(path: Union[str, Path]) -> float:
    """Get directory size in MB."""
    if not os.path.exists(path):
        return 0

    total_size = 0
    for dirpath, dirnames, filenames in os.walk(path):
        for filename in filenames:
            filepath = os.path.join(dirpath, filename)
            if os.path.exists(filepath):
                total_size += os.path.getsize(filepath)
    return total_size / (1024 * 1024)  # Convert to MB


if __name__ == "__main__":
    download_ogb_datasets()
