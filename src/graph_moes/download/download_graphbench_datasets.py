#!/usr/bin/env python3
"""
Download GraphBench datasets locally for transfer to cluster.
This avoids repeated downloads on the cluster and potential network issues.

GraphBench is a comprehensive benchmarking suite that spans diverse domains and tasks,
including node-level, edge-level, graph-level, and generative settings.
See: https://github.com/graphbench/package and https://arxiv.org/pdf/2512.04475
"""

import os
from pathlib import Path
from typing import List, Optional, Union

try:
    from graphbench.loader import Loader  # type: ignore
except ImportError as exc:
    raise ImportError(
        "graphbench-lib is not installed. Install it with: pip install graphbench-lib"
    ) from exc


def get_dir_size(path: Union[str, Path]) -> float:
    """Get directory size in MB."""
    if not os.path.exists(path):
        return 0.0

    total_size = 0.0
    for dirpath, _dirnames, filenames in os.walk(path):
        for filename in filenames:
            filepath = os.path.join(dirpath, filename)
            if os.path.exists(filepath):
                total_size += os.path.getsize(filepath)
    return total_size / (1024 * 1024)  # Convert to MB


def download_graphbench_dataset(
    dataset_name: str, root_dir: Union[str, Path], verbose: bool = True
) -> Optional[Loader]:
    """
    Download a single GraphBench dataset.

    Args:
        dataset_name: Name of the GraphBench dataset to download
        root_dir: Root directory where datasets will be stored
        verbose: Whether to print progress messages

    Returns:
        Loader object if successful, None otherwise
    """
    if verbose:
        print(f"\n📊 Downloading GraphBench dataset: {dataset_name}...")

    try:
        loader = Loader(
            dataset_names=dataset_name,
            root=root_dir,
            pre_filter=None,
            pre_transform=None,
            transform=None,
        )
        dataset = loader.load()

        if verbose:
            if hasattr(dataset, "__len__"):
                print(f"✅ {dataset_name} downloaded: {len(dataset)} samples")
            else:
                print(f"✅ {dataset_name} downloaded successfully")

            # Check dataset directory size
            dataset_path = os.path.join(root_dir, dataset_name)
            if os.path.exists(dataset_path):
                size = get_dir_size(dataset_path)
                print(f"   Size on disk: ~{size:.1f} MB")

        return loader

    except (ValueError, RuntimeError, OSError, ImportError) as e:
        if verbose:
            print(f"❌ Failed to download {dataset_name}: {e}")
        return None


def download_graphbench_datasets(
    root_dir: Optional[Union[str, Path]] = None,
    dataset_names: Optional[List[str]] = None,
) -> None:
    """
    Download GraphBench datasets to local directory.

    Args:
        root_dir: Root directory for dataset downloads. Defaults to ./graph_datasets_graphbench
        dataset_names: List of specific datasets to download. If None, downloads a default set
    """
    if root_dir is None:
        # Use local data directory for downloads
        root_dir = "./graph_datasets"

    root_dir = Path(root_dir)
    os.makedirs(root_dir, exist_ok=True)

    print("🚀 Starting GraphBench dataset downloads...")
    print(f"📁 Download directory: {os.path.abspath(root_dir)}")

    # Default datasets to download based on GraphBench domains
    # These are the domain-level dataset names that load all datasets in that domain
    if dataset_names is None:
        dataset_names = [
            # Graph classification tasks
            "socialnetwork",  # Social media datasets
            "co",  # Combinatorial optimization
            "sat",  # SAT solving
            # Algorithmic reasoning (these are separate difficulty levels)
            "algorithmic_reasoning_easy",
            "algorithmic_reasoning_medium",
            "algorithmic_reasoning_hard",
            # Electronic circuits
            "electronic_circuits",
            # Chip design
            "chipdesign",
            # Weather forecasting
            "weather",
        ]

    print(f"\n📋 Will download {len(dataset_names)} dataset(s):")
    for name in dataset_names:
        print(f"   - {name}")

    successful_downloads = []
    failed_downloads = []

    for dataset_name in dataset_names:
        loader = download_graphbench_dataset(dataset_name, root_dir, verbose=True)
        if loader is not None:
            successful_downloads.append(dataset_name)
        else:
            failed_downloads.append(dataset_name)

    # Summary
    total_size = get_dir_size(root_dir)
    print("\n🎉 Downloads complete!")
    print(f"📁 Total size: ~{total_size:.1f} MB")
    print(f"📍 Location: {os.path.abspath(root_dir)}")

    if successful_downloads:
        print(f"\n✅ Successfully downloaded {len(successful_downloads)} dataset(s):")
        for name in successful_downloads:
            print(f"   - {name}")

    if failed_downloads:
        print(f"\n❌ Failed to download {len(failed_downloads)} dataset(s):")
        for name in failed_downloads:
            print(f"   - {name}")

    print("\n📤 To transfer to cluster, run:")
    print(f"cd {root_dir}")
    print(
        "scp -r * rpellegrinext@login.rc.fas.harvard.edu:/n/holylabs/LABS/mweber_lab/Everyone/rpellegrin/graph_moes/graph_datasets/"
    )


if __name__ == "__main__":
    download_graphbench_datasets()
