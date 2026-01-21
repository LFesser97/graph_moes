"""Plot label distributions by graph index for classification datasets.

This script investigates label ordering in datasets by plotting each graph's
label as a function of its index in the dataset. This helps identify if there
are patterns in label ordering that might explain accuracy patterns in results.

For example, if COLLAB has all graphs with label 0 in indices 0-2499 and all
graphs with label 1 in indices 2500-4999, this would explain why accuracy plots
show a clear divide by index.
"""

import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch_geometric.data import Data

# Add src to path for imports
project_root = Path(__file__).parent.parent.parent
src_path = project_root / "src"
if src_path.exists() and str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

try:
    from ogb.graphproppred import PygGraphPropPredDataset
    from torch_geometric.datasets import GNNBenchmarkDataset, TUDataset
except ImportError as e:
    print(f"❌ Failed to import required modules: {e}")
    sys.exit(1)


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


def extract_labels(dataset, dataset_name: str):
    """Extract labels from a dataset.

    Args:
        dataset: List of PyTorch Geometric Data objects or dataset object
        dataset_name: Name of the dataset (for handling special cases)

    Returns:
        Tuple of (labels_array, num_classes, label_counts)
        - labels_array: numpy array of labels for each graph
        - num_classes: number of unique classes
        - label_counts: dictionary mapping label to count
    """
    labels = []

    # Handle different dataset formats
    if hasattr(dataset, "__getitem__"):
        # List or dataset object
        for i in range(len(dataset)):
            graph = dataset[i]
            # Extract label
            if hasattr(graph, "y"):
                y = graph.y
                # Handle different label formats
                if isinstance(y, torch.Tensor):
                    if y.dim() == 0:  # Scalar
                        label = y.item()
                    elif y.dim() == 1 and len(y) == 1:  # 1D tensor with 1 element
                        label = y[0].item()
                    elif y.dim() == 1:  # 1D tensor (multi-label or single)
                        if len(y) == 1:
                            # Single label
                            label = y[0].item()
                        else:
                            # Multi-label: use first active label or convert to hash
                            # Find first non-zero label
                            non_zero = torch.nonzero(y, as_tuple=False)
                            if len(non_zero) > 0:
                                label = y[non_zero[0]].item()
                            else:
                                # All zeros, use 0
                                label = 0
                    else:
                        # Multi-dimensional, use first element
                        label = y.flatten()[0].item()
                else:
                    label = int(y)
                labels.append(label)
            else:
                labels.append(0)  # Default if no label

    labels_array = np.array(labels)

    # Handle NaN values (filter them out or replace with a default)
    nan_mask = np.isnan(labels_array)
    if np.any(nan_mask):
        print(f"     ⚠️  Warning: Found {np.sum(nan_mask)} NaN labels, replacing with 0")
        labels_array[nan_mask] = 0

    unique_labels = np.unique(labels_array)
    num_classes = len(unique_labels)

    # Count labels (handle NaN if still present)
    label_counts = {}
    for label in unique_labels:
        if not np.isnan(label):
            label_counts[int(label)] = int(np.sum(labels_array == label))

    return labels_array, num_classes, label_counts


def plot_label_distribution(
    dataset_name: str,
    labels: np.ndarray,
    num_classes: int,
    label_counts: dict,
    output_dir: str = "visualizations/labels",
) -> str:
    """Plot label distribution by graph index.

    Args:
        dataset_name: Name of the dataset
        labels: Array of labels for each graph
        num_classes: Number of unique classes
        label_counts: Dictionary mapping label to count
        output_dir: Directory to save the plot

    Returns:
        Path to saved plot file
    """
    fig, ax = plt.subplots(figsize=(14, 6))

    # Create indices
    indices = np.arange(len(labels))

    # Use a colormap with distinct colors for each class
    if num_classes <= 10:
        cmap = plt.cm.tab10
    elif num_classes <= 20:
        cmap = plt.cm.tab20
    else:
        cmap = plt.cm.viridis

    # Plot each class with different color
    for label in np.unique(labels):
        mask = labels == label
        ax.scatter(
            indices[mask],
            labels[mask],
            label=f"Class {int(label)} (n={label_counts[int(label)]})",
            alpha=0.6,
            s=10,
            c=[cmap(label / max(num_classes - 1, 1))],
        )

    # Set labels and title
    ax.set_xlabel("Graph Index", fontsize=12)
    ax.set_ylabel("Label (Class)", fontsize=12)
    ax.set_title(
        f"Label Distribution by Index: {dataset_name.upper()}",
        fontsize=14,
        fontweight="bold",
    )

    # Add grid
    ax.grid(True, alpha=0.3, linestyle="--", axis="y")

    # Set y-axis to show all classes
    ax.set_ylim(-0.5, num_classes - 0.5)
    ax.set_yticks(range(num_classes))

    # Add legend
    ax.legend(loc="upper right", fontsize=9, ncol=2 if num_classes > 5 else 1)

    # Add statistics text
    sorted_labels = sorted(label_counts.items())
    label_distribution = ", ".join(
        [f"Class {label}: {count}" for label, count in sorted_labels]
    )

    # Calculate label ordering statistics
    # Check if labels are ordered (all same label in contiguous blocks)
    transitions = 0
    for i in range(1, len(labels)):
        if labels[i] != labels[i - 1]:
            transitions += 1
            # Check if this transition is expected (only allowed once per class change)
            if i > 1 and labels[i] == labels[i - 2]:
                break

    stats_text = (
        f"Total graphs: {len(labels)} | Classes: {num_classes} | "
        # f"Label transitions: {transitions} | Ordering: {ordering_status}\n"
        f"Distribution: {label_distribution}"
    )

    ax.text(
        0.5,
        -0.15,
        stats_text,
        transform=ax.transAxes,
        ha="center",
        fontsize=9,
        style="italic",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.3),
    )

    plt.tight_layout()

    # Save the plot
    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, f"{dataset_name}_label_distribution.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return plot_path


def main():
    """Main function to plot label distributions for all classification datasets."""

    # Use local data directory (adjust path if needed)
    data_directory = os.path.join(project_root, "data")

    # Alternative: use graph_datasets directory or cluster path if data doesn't exist
    graph_datasets_directory = os.path.join(project_root, "graph_datasets")
    cluster_data_directory = (
        "/n/holylabs/LABS/mweber_lab/Everyone/rpellegrin/graph_moes/graph_datasets"
    )

    if os.path.exists(data_directory):
        print(f"📁 Using local data directory: {data_directory}")
        use_directory = data_directory
    elif os.path.exists(graph_datasets_directory):
        print(f"📁 Using graph_datasets directory: {graph_datasets_directory}")
        use_directory = graph_datasets_directory
    elif os.path.exists(cluster_data_directory):
        print(f"📁 Using cluster data directory: {cluster_data_directory}")
        use_directory = cluster_data_directory
    else:
        print(
            f"❌ Data directory not found at {data_directory}, {graph_datasets_directory}, or {cluster_data_directory}"
        )
        print("   Please update the path in the script.")
        sys.exit(1)

    output_dir = os.path.join(project_root, "visualizations", "labels")
    os.makedirs(output_dir, exist_ok=True)
    print(f"💾 Output directory: {output_dir}")

    # Dictionary to store datasets and their names
    datasets_to_plot = {}

    print("\n📊 Loading classification datasets...")

    # TU Datasets (classification)
    tu_datasets = [
        "MUTAG",
        "ENZYMES",
        "PROTEINS",
        "IMDB-BINARY",
        "COLLAB",
        "REDDIT-BINARY",
    ]
    for ds_name in tu_datasets:
        try:
            print(f"  ⏳ Loading {ds_name}...")
            dataset = list(TUDataset(root=use_directory, name=ds_name))
            datasets_to_plot[ds_name.lower().replace("-", "_")] = dataset
            print(f"  ✅ {ds_name} loaded: {len(dataset)} graphs")
        except Exception as e:
            print(f"  ⚠️  Failed to load {ds_name}: {e}")

    # GNN Benchmark datasets
    gnn_benchmark = ["MNIST", "CIFAR10", "PATTERN"]
    for ds_name in gnn_benchmark:
        try:
            print(f"  ⏳ Loading {ds_name}...")
            dataset = list(GNNBenchmarkDataset(root=use_directory, name=ds_name))
            datasets_to_plot[ds_name.lower()] = dataset
            print(f"  ✅ {ds_name} loaded: {len(dataset)} graphs")
        except Exception as e:
            print(f"  ⚠️  Failed to load {ds_name}: {e}")

    # OGB datasets (graph classification)
    # Note: Skip ogbg-ppa if it prompts for update (EOFError)
    ogb_datasets = ["ogbg-molhiv", "ogbg-molpcba"]
    for ds_name in ogb_datasets:
        try:
            print(f"  ⏳ Loading {ds_name}...")
            dataset = PygGraphPropPredDataset(name=ds_name, root=use_directory)
            # Convert to list (sample first 10000 for large datasets to speed up)
            if len(dataset) > 10000:
                print(
                    f"     Large dataset ({len(dataset)} graphs), sampling first 10000 for visualization..."
                )
                dataset_list = [dataset[i] for i in range(10000)]
            else:
                dataset_list = [dataset[i] for i in range(len(dataset))]
            short_name = ds_name.replace("ogbg-", "").replace("-", "_")
            datasets_to_plot[short_name] = dataset_list
            print(f"  ✅ {ds_name} loaded: {len(dataset_list)} graphs")
        except (EOFError, KeyboardInterrupt):
            print(f"  ⚠️  Skipped {ds_name}: interactive prompt encountered")
        except Exception as e:
            print(f"  ⚠️  Failed to load {ds_name}: {e}")

    # Peptides-func dataset (LRGB)
    try:
        print("  ⏳ Loading Peptides-func...")
        peptides_func_path = os.path.join(use_directory, "peptidesfunc")
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
            datasets_to_plot["peptides_func"] = peptides_func
            print(f"  ✅ Peptides-func loaded: {len(peptides_func)} graphs")
        else:
            print(f"  ⚠️  Peptides-func directory not found at {peptides_func_path}")
    except Exception as e:
        print(f"  ⚠️  Failed to load Peptides-func: {e}")

    print(
        f"\n📈 Generating label distribution plots for {len(datasets_to_plot)} datasets..."
    )

    # Plot each dataset
    for dataset_name, dataset in datasets_to_plot.items():
        try:
            print(f"\n  📊 Processing {dataset_name.upper()}...")

            # Extract labels
            labels, num_classes, label_counts = extract_labels(dataset, dataset_name)

            print(f"     Graphs: {len(labels)}")
            print(f"     Classes: {num_classes}")
            print(f"     Label distribution: {label_counts}")

            # Check for ordered labels
            if len(labels) > 0:
                # Count transitions between labels
                transitions = sum(
                    1 for i in range(1, len(labels)) if labels[i] != labels[i - 1]
                )
                print(f"     Label transitions: {transitions}")

                # Check if labels are mostly ordered (few transitions relative to dataset size)
                if transitions < len(labels) * 0.1:  # Less than 10% transitions
                    print(
                        f"     ⚠️  WARNING: Labels appear to be heavily ordered (only {transitions} transitions)"
                    )
                elif transitions > len(labels) * 0.9:  # More than 90% transitions
                    print("     ✅ Labels appear well-mixed (many transitions)")

            # Generate plot
            plot_path = plot_label_distribution(
                dataset_name, labels, num_classes, label_counts, output_dir
            )
            print(f"     ✅ Plot saved: {plot_path}")

        except Exception as e:
            print(f"     ❌ Error processing {dataset_name}: {e}")
            import traceback

            traceback.print_exc()

    print(f"\n🎉 Done! All plots saved to: {output_dir}")


if __name__ == "__main__":
    main()
