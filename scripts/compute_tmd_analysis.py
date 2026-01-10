"""Script to compute TMD and class-distance ratios for datasets."""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from torch_geometric.datasets import GNNBenchmarkDataset, TUDataset

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from graph_moes.tmd import (
    compute_class_distance_ratios,
    compute_tmd_matrix,
    extract_labels,
    save_tmd_results,
)


def load_dataset(
    dataset_name: str, data_directory: str
) -> tuple[List, np.ndarray, int, dict]:
    """Load a dataset and extract labels.

    Supports multiple dataset types:
    - TU datasets: MUTAG, ENZYMES, PROTEINS, IMDB-BINARY, COLLAB, REDDIT-BINARY, etc.
    - GNN Benchmark datasets: MNIST, CIFAR10, PATTERN

    Args:
        dataset_name: Name of the dataset (e.g., 'MUTAG', 'ENZYMES', 'MNIST', 'PATTERN')
        data_directory: Root directory for datasets

    Returns:
        Tuple of (dataset_list, labels, num_classes, label_counts)

    Raises:
        ValueError: If dataset cannot be loaded or is not supported
    """
    print(f"\n📊 Loading {dataset_name} dataset...")

    # Try TU datasets first
    tu_datasets = [
        "MUTAG",
        "ENZYMES",
        "PROTEINS",
        "IMDB-BINARY",
        "COLLAB",
        "REDDIT-BINARY",
        "MalNetTiny",
    ]

    # Try GNN Benchmark datasets
    gnn_benchmark_datasets = ["MNIST", "CIFAR10", "PATTERN"]

    dataset = None
    if dataset_name.upper() in tu_datasets:
        try:
            dataset = list(TUDataset(root=data_directory, name=dataset_name))
        except Exception as e:
            print(f"❌ Failed to load {dataset_name} as TU dataset: {e}")
            raise ValueError(f"Could not load {dataset_name} as TU dataset: {e}")
    elif dataset_name.upper() in gnn_benchmark_datasets:
        try:
            dataset = list(
                GNNBenchmarkDataset(root=data_directory, name=dataset_name.upper())
            )
        except Exception as e:
            print(f"❌ Failed to load {dataset_name} as GNN Benchmark dataset: {e}")
            raise ValueError(
                f"Could not load {dataset_name} as GNN Benchmark dataset: {e}"
            )
    else:
        # Try as TU dataset first, then GNN Benchmark
        try:
            dataset = list(TUDataset(root=data_directory, name=dataset_name))
        except Exception:
            try:
                dataset = list(
                    GNNBenchmarkDataset(root=data_directory, name=dataset_name.upper())
                )
            except Exception as e:
                print(
                    f"❌ Failed to load {dataset_name} as either TU or GNN Benchmark dataset: {e}"
                )
                raise ValueError(
                    f"Dataset {dataset_name} not recognized. Supported types: "
                    f"TU datasets ({', '.join(tu_datasets)}) or "
                    f"GNN Benchmark datasets ({', '.join(gnn_benchmark_datasets)})"
                ) from e

    # Extract labels
    try:
        labels, num_classes, label_counts = extract_labels(dataset, dataset_name)
        print(f"✅ {dataset_name} loaded: {len(dataset)} graphs, {num_classes} classes")
        return dataset, labels, num_classes, label_counts
    except Exception as e:
        print(f"❌ Failed to extract labels from {dataset_name}: {e}")
        raise ValueError(f"Could not extract labels from {dataset_name}: {e}") from e


def save_results(
    dataset_name: str,
    tmd_matrix: np.ndarray,
    ratios: np.ndarray,
    labels: np.ndarray,
    stats_dict: dict,
    output_dir: str,
) -> None:
    """Save TMD computation results to files.

    Args:
        dataset_name: Name of the dataset
        tmd_matrix: TMD distance matrix
        ratios: Class-distance ratios
        labels: Graph labels
        stats_dict: Statistics dictionary
        output_dir: Output directory for results
    """
    file_paths = save_tmd_results(
        dataset_name, tmd_matrix, ratios, labels, stats_dict, output_dir
    )

    print(f"💾 Saved TMD matrix to {file_paths['tmd_matrix']}")
    print(f"💾 Saved class-distance ratios to {file_paths['class_ratios']}")
    print(f"💾 Saved statistics to {file_paths['stats']}")


def main() -> None:
    """Main function to compute TMD and class-distance ratios."""
    parser = argparse.ArgumentParser(
        description="Compute TMD and class-distance ratios for datasets"
    )
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="+",
        default=["MUTAG", "ENZYMES"],
        help="Dataset names to process (default: MUTAG ENZYMES)",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Data directory (default: tries local then cluster path)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="tmd_results",
        help="Output directory for results (default: tmd_results)",
    )
    parser.add_argument(
        "--w",
        type=float,
        default=1.0,
        help="TMD weighting constant (default: 1.0)",
    )
    parser.add_argument(
        "--L",
        type=int,
        default=4,
        help="TMD computation tree depth (default: 4)",
    )
    parser.add_argument(
        "--cache",
        action="store_true",
        help="Use cached TMD matrices if available",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Force recomputation even if cache exists",
    )

    args = parser.parse_args()

    # Determine data directory
    if args.data_dir:
        data_directory = args.data_dir
    else:
        # Try local first, then cluster
        local_dir = os.path.join(project_root, "graph_datasets")
        cluster_dir = (
            "/n/holylabs/LABS/mweber_lab/Everyone/rpellegrin/graph_moes/graph_datasets"
        )

        if os.path.exists(local_dir):
            data_directory = local_dir
            print(f"📁 Using local data directory: {data_directory}")
        elif os.path.exists(cluster_dir):
            data_directory = cluster_dir
            print(f"📁 Using cluster data directory: {data_directory}")
        else:
            print(f"❌ Data directory not found at {local_dir} or {cluster_dir}")
            print("   Please specify --data-dir")
            sys.exit(1)

    # Process each dataset
    for dataset_name in args.datasets:
        print(f"\n{'='*60}")
        print(f"Processing {dataset_name}")
        print(f"{'='*60}")

        try:
            # Load dataset
            dataset, labels, num_classes, label_counts = load_dataset(
                dataset_name, data_directory
            )

            # Determine cache path
            cache_path = None
            if args.cache and not args.no_cache:
                cache_path = os.path.join(
                    args.output_dir, f"{dataset_name.lower()}_tmd_matrix.npy"
                )

            # Compute TMD matrix
            tmd_matrix = compute_tmd_matrix(
                dataset,
                w=args.w,
                L=args.L,
                verbose=True,
                cache_path=cache_path if not args.no_cache else None,
            )

            # Compute class-distance ratios
            ratios, stats_dict = compute_class_distance_ratios(
                tmd_matrix, labels, verbose=True
            )

            # Save results
            save_results(
                dataset_name,
                tmd_matrix,
                ratios,
                labels,
                stats_dict,
                args.output_dir,
            )

            print(f"\n✅ {dataset_name} analysis complete!")

        except Exception as e:
            print(f"\n❌ Error processing {dataset_name}: {e}")
            import traceback

            traceback.print_exc()
            continue

    print(f"\n🎉 All datasets processed!")


if __name__ == "__main__":
    main()
