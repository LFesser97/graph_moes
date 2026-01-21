#!/usr/bin/env python3
"""
Download a single GraphBench dataset.

This script downloads one GraphBench dataset at a time, designed to be called
from bash scripts for sequential downloads with better control and logging.
"""

import gc
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

try:
    from graphbench.loader import Loader  # type: ignore
except ImportError:
    print(
        "❌ graphbench-lib is not installed. Install it with: pip install graphbench-lib"
    )
    sys.exit(1)


def download_single_dataset(
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
    dataset_path = root_dir / dataset_name
    algoreas_path = root_dir / "algoreas"

    if dataset_path.exists() or (dataset_name == "co" and algoreas_path.exists()):
        print(f"   ✅ {dataset_name} already exists, verifying...")
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
                if dataset_path.exists():
                    import shutil

                    shutil.rmtree(dataset_path)
                    print(f"   🗑️  Deleted corrupted {dataset_name} directory")
                if dataset_name == "co" and algoreas_path.exists():
                    import shutil

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
                import time

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
                    import shutil

                    if dataset_path.exists():
                        shutil.rmtree(dataset_path)
                    if dataset_name == "co" and algoreas_path.exists():
                        shutil.rmtree(algoreas_path)
                    # Retry once
                    return download_single_dataset(dataset_name, root_dir, retry=False)
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
    """Main function to download a single GraphBench dataset."""
    if len(sys.argv) < 3:
        print(
            "Usage: python download_graphbench_single.py <dataset_name> <root_directory>"
        )
        sys.exit(1)

    dataset_name = sys.argv[1]
    root_directory = Path(sys.argv[2])

    root_directory.mkdir(parents=True, exist_ok=True)

    success = download_single_dataset(dataset_name, root_directory)

    if success:
        print(f"\n✅ Successfully downloaded {dataset_name}")
        sys.exit(0)
    else:
        print(f"\n❌ Failed to download {dataset_name}")
        sys.exit(1)


if __name__ == "__main__":
    main()
