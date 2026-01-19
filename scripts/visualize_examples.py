"""Script to visualize example graphs from each dataset.

This script loads all available datasets and creates visualizations
of example graphs from each dataset for inspection and analysis.
"""

import argparse
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch_geometric.data import Data
from torch_geometric.datasets import ZINC, GNNBenchmarkDataset, TUDataset

from graph_moes.plotting.visualize_graphs import (
    sample_graphs,
    visualize_graph_grid,
    visualize_single_graph,
)

# Add src directory to path to import graph_moes modules
project_root = Path(__file__).parent.parent
src_dir = project_root / "src"
sys.path.insert(0, str(src_dir))


def _convert_lrgb(dataset_tuple: torch.Tensor) -> Data:
    """Convert LRGB dataset tuple format to PyTorch Geometric Data object.

    Args:
        dataset_tuple: Tuple containing (x, edge_attr, edge_index, y) tensors

    Returns:
        PyTorch Geometric Data object with node features, edges, and labels
    """
    x = dataset_tuple[0]
    edge_attr = dataset_tuple[1]
    edge_index = dataset_tuple[2]
    y = dataset_tuple[3]

    return Data(x=x, edge_index=edge_index, y=y, edge_attr=edge_attr)


def load_all_datasets(data_directory: str) -> dict:
    """
    Load all available datasets.

    Args:
        data_directory: Root directory for datasets

    Returns:
        Dictionary mapping dataset names to lists of graphs
    """
    datasets = {}
    os.makedirs(data_directory, exist_ok=True)

    print("📊 Loading datasets for visualization...")

    # TU datasets
    tu_datasets = [
        "MUTAG",
        "ENZYMES",
        "PROTEINS",
        "IMDB-BINARY",
        "COLLAB",
        "REDDIT-BINARY",
    ]
    for name in tu_datasets:
        try:
            print(f"  ⏳ Loading {name}...")
            dataset = list(TUDataset(root=data_directory, name=name))
            datasets[name.lower().replace("-", "_")] = dataset
            print(f"  ✅ {name} loaded: {len(dataset)} graphs")
        except Exception as e:
            print(f"  ⚠️  Failed to load {name}: {e}")

    # GNN Benchmark datasets
    benchmark_datasets = ["MNIST", "CIFAR10", "PATTERN"]
    for name in benchmark_datasets:
        try:
            print(f"  ⏳ Loading {name}...")
            dataset = list(GNNBenchmarkDataset(root=data_directory, name=name))
            datasets[name.lower()] = dataset
            print(f"  ✅ {name} loaded: {len(dataset)} graphs")
        except Exception as e:
            print(f"  ⚠️  Failed to load {name}: {e}")

    # ZINC dataset (regression)
    try:
        print("  ⏳ Loading ZINC...")
        train_dataset = ZINC(root=data_directory, subset=True, split="train")
        zinc = [train_dataset[i] for i in range(min(100, len(train_dataset)))]
        datasets["zinc"] = zinc
        print(f"  ✅ ZINC loaded: {len(zinc)} graphs (sampled)")
    except Exception as e:
        print(f"  ⚠️  Failed to load ZINC: {e}")

    # Peptides-func dataset (LRGB)
    try:
        print("  ⏳ Loading Peptides-func...")
        peptides_func_path = os.path.join(data_directory, "peptidesfunc")
        if os.path.exists(peptides_func_path):
            peptides_train = torch.load(
                os.path.join(peptides_func_path, "train.pt"), weights_only=False
            )
            peptides_val = torch.load(
                os.path.join(peptides_func_path, "val.pt"), weights_only=False
            )
            peptides_test = torch.load(
                os.path.join(peptides_func_path, "test.pt"), weights_only=False
            )
            peptides_func = (
                [_convert_lrgb(peptides_train[i]) for i in range(len(peptides_train))]
                + [_convert_lrgb(peptides_val[i]) for i in range(len(peptides_val))]
                + [_convert_lrgb(peptides_test[i]) for i in range(len(peptides_test))]
            )
            datasets["peptides_func"] = peptides_func
            print(f"  ✅ Peptides-func loaded: {len(peptides_func)} graphs")
        else:
            print(f"  ⚠️  Peptides-func directory not found at {peptides_func_path}")
    except Exception as e:
        print(f"  ⚠️  Failed to load Peptides-func: {e}")

    # GraphBench datasets
    try:
        from graph_moes.download.load_graphbench import load_graphbench_dataset

        # All GraphBench datasets that are available/downloaded
        graphbench_datasets = [
            "socialnetwork",  # Social media datasets
            "co",  # Combinatorial optimization
            "sat",  # SAT solving
            "algorithmic_reasoning_easy",  # Algorithmic reasoning (easy)
            "algorithmic_reasoning_medium",  # Algorithmic reasoning (medium)
            "algorithmic_reasoning_hard",  # Algorithmic reasoning (hard)
            "electronic_circuits",  # Electronic circuits
            "chipdesign",  # Chip design
            "weather",  # Weather forecasting
        ]
        for dataset_name in graphbench_datasets:
            try:
                print(f"  ⏳ Loading GraphBench: {dataset_name}...")
                graphbench_data = load_graphbench_dataset(
                    dataset_name=dataset_name, root=data_directory
                )
                datasets[f"graphbench_{dataset_name}"] = graphbench_data
                print(
                    f"  ✅ GraphBench {dataset_name} loaded: {len(graphbench_data)} graphs"
                )
            except Exception as e:
                print(f"  ⚠️  Failed to load GraphBench {dataset_name}: {e}")
    except ImportError:
        print("  ⚠️  GraphBench not available (graphbench-lib not installed)")

    return datasets


def visualize_dataset_examples(
    dataset_name: str,
    graphs: list,
    n_examples: int = 8,
    output_dir: str = "visualizations",
    figsize: tuple = (16, 16),
) -> None:
    """
    Visualize example graphs from a dataset.

    Args:
        dataset_name: Name of the dataset
        graphs: List of graphs from the dataset
        n_examples: Number of example graphs to visualize
        output_dir: Directory to save visualizations
        figsize: Figure size for the grid
    """
    if not graphs:
        print(f"  ⚠️  Skipping {dataset_name}: No graphs available")
        return

    os.makedirs(output_dir, exist_ok=True)

    # First, create a single example graph visualization
    print(f"  📊 Creating single example visualization for {dataset_name}...")
    example_graph = graphs[0]  # Use first graph as example
    single_fig = visualize_single_graph(
        example_graph,
        figsize=(10, 10),
        title=f"Example Graph from {dataset_name.upper().replace('_', '-')} Dataset",
    )
    single_output_path = os.path.join(output_dir, f"{dataset_name}_example_single.png")
    single_fig.savefig(single_output_path, dpi=150, bbox_inches="tight")
    print(f"  💾 Saved single example: {single_output_path}")
    plt.close(single_fig)

    # Then create grid visualization with multiple examples
    sampled = sample_graphs(
        graphs, n_samples=min(n_examples, len(graphs)), random_seed=42
    )

    print(f"  📊 Creating grid visualization for {dataset_name}...")
    fig = visualize_graph_grid(
        sampled,
        n_cols=4,
        figsize=figsize,
        dataset_name=dataset_name.upper().replace("_", "-"),
    )

    # Save grid figure
    output_path = os.path.join(output_dir, f"{dataset_name}_examples.png")
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"  💾 Saved grid: {output_path}")

    plt.close(fig)


def main() -> None:
    """Main function to visualize examples from all datasets."""
    parser = argparse.ArgumentParser(
        description="Visualize example graphs from each dataset"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="./graph_datasets",
        help="Directory containing datasets",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="visualizations",
        help="Directory to save visualizations",
    )
    parser.add_argument(
        "--n_examples",
        type=int,
        default=8,
        help="Number of example graphs to visualize per dataset",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Specific dataset to visualize (if None, visualizes all)",
    )

    args = parser.parse_args()

    # Load datasets
    datasets = load_all_datasets(args.data_dir)

    if not datasets:
        print("❌ No datasets loaded. Exiting.")
        return

    print(f"\n📊 Found {len(datasets)} datasets")
    print("🎨 Creating visualizations...\n")

    # Visualize each dataset
    for dataset_name, graphs in datasets.items():
        if args.dataset and dataset_name != args.dataset:
            continue

        try:
            visualize_dataset_examples(
                dataset_name=dataset_name,
                graphs=graphs,
                n_examples=args.n_examples,
                output_dir=args.output_dir,
            )
        except Exception as e:
            print(f"  ❌ Error visualizing {dataset_name}: {e}")

    print(f"\n✅ Visualization complete! Check {args.output_dir} for results.")


if __name__ == "__main__":
    main()
