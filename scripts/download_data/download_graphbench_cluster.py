#!/usr/bin/env python3
"""
Download GraphBench datasets on the cluster.
This script downloads datasets once so experiments don't need to download them repeatedly.

It checks if datasets already exist and handles corrupted downloads by deleting and retrying.
"""

import gc
import shutil
import sys
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

try:
    from graphbench.loader import Loader  # type: ignore
except ImportError:
    print(
        "❌ graphbench-lib is not installed. Install it with: pip install graphbench-lib"
    )
    sys.exit(1)


def check_dataset_exists(dataset_name: str, root_dir: Path) -> bool:
    """Check if a dataset directory exists and appears complete."""
    dataset_path = root_dir / dataset_name
    if not dataset_path.exists():
        return False

    # Check for common GraphBench dataset subdirectories
    # Different datasets have different structures
    if (dataset_path / "processed").exists() or (dataset_path / "raw").exists():
        return True

    # For algoreas (combinatorial optimization), check for subdirectories
    algoreas_path = root_dir / "algoreas"
    if dataset_name == "co" and algoreas_path.exists():
        # Check if algoreas has subdirectories
        subdirs = [d for d in algoreas_path.iterdir() if d.is_dir()]
        if len(subdirs) > 0:
            return True

    # If directory exists but structure is unclear, assume it exists
    # (will fail during load if corrupted)
    return True


def download_graphbench_dataset(
    dataset_name: str, root_dir: Path, retry: bool = True
) -> bool:
    """
    Download a single GraphBench dataset with error handling.

    Args:
        dataset_name: Name of the GraphBench dataset to download
        root_dir: Root directory where datasets will be stored
        retry: Whether to retry if download fails (deletes corrupted files first)

    Returns:
        True if successful, False otherwise
    """
    print(f"\n📊 Processing GraphBench dataset: {dataset_name}...")

    # Check if dataset already exists
    if check_dataset_exists(dataset_name, root_dir):
        print(f"   ✅ {dataset_name} already exists, skipping download")
        # Try to verify it's not corrupted by attempting to load
        try:
            loader = Loader(
                dataset_names=dataset_name,
                root=str(root_dir),
                pre_filter=None,
                pre_transform=None,
                transform=None,
            )
            # Just check if we can create the loader, don't actually load
            print(f"   ✅ {dataset_name} appears to be valid")
            # Free memory
            del loader
            gc.collect()
            return True
        except Exception as e:
            error_msg = str(e)
            if any(
                keyword in error_msg.lower()
                for keyword in ["zlib", "decompressing", "eof", "corrupted", "invalid"]
            ):
                print(f"   ⚠️  {dataset_name} appears corrupted, will re-download...")
                # Delete corrupted dataset
                dataset_path = root_dir / dataset_name
                algoreas_path = root_dir / "algoreas"
                if dataset_path.exists():
                    shutil.rmtree(dataset_path)
                    print(f"   🗑️  Deleted corrupted {dataset_name} directory")
                if dataset_name == "co" and algoreas_path.exists():
                    shutil.rmtree(algoreas_path)
                    print("   🗑️  Deleted corrupted algoreas directory")
            else:
                print(f"   ⚠️  {dataset_name} exists but may have issues: {e}")
                if not retry:
                    return False

    # Download the dataset with retry logic for rate limits
    max_retries = 3
    retry_delay = 60  # Wait 60 seconds between retries for rate limits

    for attempt in range(max_retries):
        try:
            if attempt > 0:
                wait_time = retry_delay * attempt  # Exponential backoff
                print(
                    f"   ⏳ Retrying download (attempt {attempt + 1}/{max_retries}) after {wait_time}s..."
                )
                time.sleep(wait_time)
            else:
                print(f"   ⏳ Downloading {dataset_name}...")

            loader = Loader(
                dataset_names=dataset_name,
                root=str(root_dir),
                pre_filter=None,
                pre_transform=None,
                transform=None,
            )
            dataset = loader.load()

            if hasattr(dataset, "__len__"):
                print(
                    f"   ✅ {dataset_name} downloaded successfully: {len(dataset)} samples"
                )
            else:
                print(f"   ✅ {dataset_name} downloaded successfully")

            # Explicitly free memory after loading
            del dataset
            del loader
            gc.collect()

            return True

        except Exception as e:
            error_msg = str(e)
            error_type = type(e).__name__

            # Check if it's a rate limit error (HTTP 429)
            if (
                "429" in error_msg
                or "Too Many Requests" in error_msg
                or "rate limit" in error_msg.lower()
            ):
                if attempt < max_retries - 1:
                    print(
                        f"   ⚠️  Rate limited (HTTP 429), will retry in {retry_delay * (attempt + 1)}s..."
                    )
                    continue  # Retry with backoff
                else:
                    print(
                        f"   ❌ Rate limited after {max_retries} attempts. Please wait and try again later."
                    )
                    return False

            # Check if it's a download/extraction error
            if any(
                keyword in error_msg.lower()
                for keyword in [
                    "zlib",
                    "decompressing",
                    "eof",
                    "corrupted",
                    "invalid",
                    "end-of-stream",
                    "tarfile",
                ]
            ) or error_type in ["EOFError", "zlib.error"]:
                print(f"   ❌ Download failed due to corrupted file: {error_type}: {e}")
                if retry:
                    print("   🔄 Will retry after cleaning up...")
                    # Clean up any partial downloads
                    dataset_path = root_dir / dataset_name
                    algoreas_path = root_dir / "algoreas"
                    if dataset_path.exists():
                        shutil.rmtree(dataset_path)
                    if dataset_name == "co" and algoreas_path.exists():
                        shutil.rmtree(algoreas_path)
                    # Retry once
                    return download_graphbench_dataset(
                        dataset_name, root_dir, retry=False
                    )
                return False
            else:
                # For other errors, only retry if it's not the last attempt
                if attempt < max_retries - 1:
                    print(f"   ⚠️  Error: {error_type}: {e}, will retry...")
                    continue
                else:
                    print(f"   ❌ Failed to download {dataset_name}: {error_type}: {e}")
                    return False

    return False


def main() -> None:
    """Main function to download GraphBench datasets."""
    # Use cluster data directory
    data_directory = Path(
        "/n/holylabs/LABS/mweber_lab/Everyone/rpellegrin/graph_moes/graph_datasets"
    )
    data_directory.mkdir(parents=True, exist_ok=True)

    print("🚀 Starting GraphBench dataset downloads on cluster...")
    print(f"📁 Download directory: {data_directory}")

    # Datasets needed for graph classification experiments
    dataset_names = [
        "socialnetwork",  # Social media datasets
        "co",  # Combinatorial optimization (may be problematic)
        "sat",  # SAT solving
        "algorithmic_reasoning_easy",  # Algorithmic reasoning (easy)
        "algorithmic_reasoning_medium",  # Algorithmic reasoning (medium)
        "algorithmic_reasoning_hard",  # Algorithmic reasoning (hard)
        "electronic_circuits",  # Electronic circuits
        "chipdesign",  # Chip design
        # Note: weather is for regression, not included here
    ]

    print(f"\n📋 Will process {len(dataset_names)} dataset(s):")
    for name in dataset_names:
        print(f"   - {name}")

    successful_downloads = []
    failed_downloads = []

    for dataset_name in dataset_names:
        if download_graphbench_dataset(dataset_name, data_directory):
            successful_downloads.append(dataset_name)
        else:
            failed_downloads.append(dataset_name)

        # Force garbage collection between datasets to free memory
        gc.collect()

    # Summary
    print("\n" + "=" * 60)
    print("🎉 Download process complete!")
    print("=" * 60)

    if successful_downloads:
        print(f"\n✅ Successfully processed {len(successful_downloads)} dataset(s):")
        for name in successful_downloads:
            print(f"   - {name}")

    if failed_downloads:
        print(f"\n❌ Failed to download {len(failed_downloads)} dataset(s):")
        for name in failed_downloads:
            print(f"   - {name}")
        print("\n💡 Tip: You can re-run this script to retry failed downloads.")
        print("   Corrupted datasets will be automatically deleted and re-downloaded.")

    print(f"\n📁 Datasets location: {data_directory}")
    print("\n✅ Experiments can now run without downloading datasets!")


if __name__ == "__main__":
    main()
