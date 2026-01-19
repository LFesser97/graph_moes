#!/usr/bin/env python3
"""
Generate sorted average accuracy plots for all existing results.

This script processes all pickle files in the results directory and generates
plots ordered by average accuracy (instead of by graph index).
"""

import os
import pickle
import sys
from pathlib import Path

# Add src to path for imports
project_root = Path(__file__).parent.parent
src_path = project_root / "src"
if src_path.exists() and str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

try:
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm
    import numpy as np
    import torch
    from torch_geometric.datasets import TUDataset, GNNBenchmarkDataset

    from graph_moes.experiments.track_avg_accuracy import (
        compute_average_per_graph,
        load_and_plot_average_per_graph,
    )
except ImportError as e:
    print(f"❌ Failed to import required modules: {e}")
    sys.exit(1)


def extract_labels_from_dataset(dataset) -> np.ndarray:
    """
    Extract labels from a dataset.

    Args:
        dataset: List of PyTorch Geometric Data objects or dataset object

    Returns:
        Numpy array of labels for each graph
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
                            # Multi-label: use first active label
                            non_zero = torch.nonzero(y, as_tuple=False)
                            if len(non_zero) > 0:
                                label = y[non_zero[0]].item()
                            else:
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

    # Handle NaN values
    nan_mask = np.isnan(labels_array)
    if np.any(nan_mask):
        labels_array[nan_mask] = 0

    return labels_array


def load_dataset(dataset_name: str, encoding: str | None = None) -> list | None:
    """
    Load a dataset by name.

    Args:
        dataset_name: Name of the dataset (e.g., "mutag", "cifar")
        encoding: Optional encoding name (e.g., "LAPE", "RWPE")

    Returns:
        List of Data objects or None if dataset not found
    """
    data_dir = project_root / "data"
    data_dir_cluster = project_root / "graph_datasets"
    cluster_data_dir = Path(
        "/n/holylabs/LABS/mweber_lab/Everyone/rpellegrin/graph_moes/graph_datasets"
    )

    # List of directories to check
    data_dirs = [data_dir]
    if data_dir_cluster.exists():
        data_dirs.append(data_dir_cluster)
    if cluster_data_dir.exists():
        data_dirs.append(cluster_data_dir)

    # Try to load from saved .pt file first (if encoding was used)
    if encoding:
        for data_dir_path in data_dirs:
            pt_file = data_dir_path / f"{dataset_name}_{encoding}.pt"
            if pt_file.exists():
                try:
                    dataset = torch.load(pt_file)
                    return dataset
                except Exception as e:
                    print(f"   ⚠️  Failed to load {pt_file}: {e}")

    # Try to load from data directory without encoding
    for data_dir_path in data_dirs:
        pt_file = data_dir_path / f"{dataset_name}_None.pt"
        if pt_file.exists():
            try:
                dataset = torch.load(pt_file)
                return dataset
            except Exception as e:
                print(f"   ⚠️  Failed to load {pt_file}: {e}")

    # Try to load from data directory with just dataset name (no encoding suffix)
    for data_dir_path in data_dirs:
        pt_file = data_dir_path / f"{dataset_name}.pt"
        if pt_file.exists():
            try:
                dataset = torch.load(pt_file)
                return dataset
            except Exception as e:
                print(f"   ⚠️  Failed to load {pt_file}: {e}")

    # If .pt files not found, try loading directly from PyTorch Geometric
    # Map dataset names to PyG dataset names
    dataset_name_mapping = {
        "mutag": ("MUTAG", TUDataset),
        "enzymes": ("ENZYMES", TUDataset),
        "proteins": ("PROTEINS", TUDataset),
        "imdb": ("IMDB-BINARY", TUDataset),
        "collab": ("COLLAB", TUDataset),
        "reddit": ("REDDIT-BINARY", TUDataset),
        "mnist": ("MNIST", GNNBenchmarkDataset),
        "cifar": ("CIFAR10", GNNBenchmarkDataset),
        "pattern": ("PATTERN", GNNBenchmarkDataset),
    }

    if dataset_name.lower() in dataset_name_mapping:
        pyg_name, dataset_class = dataset_name_mapping[dataset_name.lower()]
        # Try loading from different data directories
        for data_dir_path in data_dirs:
            if data_dir_path.exists():
                try:
                    dataset = list(
                        dataset_class(root=str(data_dir_path), name=pyg_name)
                    )
                    if len(dataset) > 0:
                        return dataset
                except Exception:
                    continue

    return None


def plot_rank_ordered_accuracy(
    rank_indices: np.ndarray,
    sorted_average_values: np.ndarray,
    dataset_name: str,
    layer_type: str,
    encoding: str | None,
    num_layers: int,
    task_type: str = "classification",
    output_dir: str = "results",
    save_filename: str | None = None,
    graph_labels: np.ndarray | None = None,
    sorted_label_positions: np.ndarray | None = None,
) -> str:
    """
    Create a plot showing accuracy ordered by rank (1 = highest accuracy, N = lowest accuracy).

    Args:
        rank_indices: Sequential rank numbers (1, 2, 3, ...)
        sorted_average_values: Accuracy values sorted from highest to lowest
        dataset_name: Name of the dataset
        layer_type: Type of layer/model used
        encoding: Encoding used (if any)
        num_layers: Number of layers in the model
        task_type: "classification" or "regression"
        output_dir: Directory to save the plot
        save_filename: Custom filename for the plot
        graph_labels: Optional array of labels for each graph (for coloring)
        sorted_label_positions: Optional array of positions in graph_labels corresponding to sorted order

    Returns:
        Path to the saved plot file
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    # Convert to percentage for classification
    if task_type == "classification":
        y_values_plot = sorted_average_values * 100
        ylabel = "Average Accuracy (%)"
        title_suffix = "Accuracy"
        ax.set_ylim(-5, 105)
    else:
        y_values_plot = sorted_average_values
        ylabel = "Average Error (MAE)"
        title_suffix = "Error"

    # Create the scatter plot with label coloring if available
    if graph_labels is not None and sorted_label_positions is not None:
        # sorted_label_positions contains the positions in graph_labels (0, 1, 2, ...)
        # that correspond to the sorted order
        sorted_labels = graph_labels[sorted_label_positions]

        # Get unique labels and create colormap
        unique_labels = np.unique(sorted_labels)
        num_classes = len(unique_labels)

        # Choose appropriate colormap
        if num_classes <= 10:
            cmap = plt.cm.tab10
        elif num_classes <= 20:
            cmap = plt.cm.tab20
        else:
            cmap = plt.cm.viridis

        # For discrete labels, use BoundaryNorm to map labels to colors
        # Create boundaries for discrete color mapping
        # Add small offsets to ensure each label gets a distinct color
        if num_classes == 1:
            boundaries = [unique_labels[0] - 0.5, unique_labels[0] + 0.5]
        else:
            # Create boundaries between labels
            boundaries = [unique_labels[0] - 0.5]
            for i in range(len(unique_labels) - 1):
                mid = (unique_labels[i] + unique_labels[i + 1]) / 2
                boundaries.append(mid)
            boundaries.append(unique_labels[-1] + 0.5)

        # Create norm for discrete color mapping
        norm = BoundaryNorm(boundaries, cmap.N, clip=True)

        # Create scatter plot with colors using 'x' markers
        scatter = ax.scatter(
            rank_indices,
            y_values_plot,
            c=sorted_labels,
            cmap=cmap,
            norm=norm,
            alpha=0.7,
            s=15,
            marker="x",
        )

        # Add colorbar with discrete ticks at actual label values
        cbar = plt.colorbar(scatter, ax=ax, ticks=unique_labels)
        cbar.set_label("Graph Label", fontsize=10)
        # Format ticks as integers if all labels are integers
        try:
            if all(
                isinstance(label, (int, np.integer))
                or (isinstance(label, (float, np.floating)) and label.is_integer())
                for label in unique_labels
            ):
                cbar.set_ticklabels([int(label) for label in unique_labels])
        except Exception:
            # If formatting fails, use default labels
            pass

    else:
        # Fallback to blue if no labels available
        ax.scatter(
            rank_indices, y_values_plot, alpha=0.7, s=15, marker="x", color="blue"
        )

    # Set labels and title
    ax.set_xlabel("Accuracy Rank (1 = Highest, N = Lowest)", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)

    # Create title
    title_parts = [
        f"Graphs Ranked by {title_suffix}",
        f"Dataset: {dataset_name}",
        f"Model: {layer_type} ({num_layers} layers)",
    ]
    if encoding:
        title_parts.append(f"Encoding: {encoding}")
    title = " | ".join(title_parts)
    ax.set_title(title, fontsize=14, fontweight="bold")

    # Add grid
    ax.grid(True, alpha=0.3, linestyle="--")

    # Add statistics
    mean_value = np.mean(y_values_plot).item()
    std_value = np.std(y_values_plot).item()
    min_value = np.min(y_values_plot).item()
    max_value = np.max(y_values_plot).item()

    stats_text = (
        f"Mean: {mean_value:.2f} | Std: {std_value:.2f} | "
        f"Min: {min_value:.2f} | Max: {max_value:.2f} | "
        f"Graphs: {len(rank_indices)}"
    )
    ax.text(
        0.5,
        -0.15,
        stats_text,
        transform=ax.transAxes,
        ha="center",
        fontsize=10,
        style="italic",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.3),
    )

    plt.tight_layout(rect=[0, 0.05, 1, 1])

    # Save the plot
    os.makedirs(f"{output_dir}/{num_layers}_layers", exist_ok=True)
    if save_filename is None:
        encoding_str = f"_{encoding}" if encoding else ""
        save_filename = f"{dataset_name}_{layer_type}{encoding_str}_rank_ordered.png"

    plot_path = f"{output_dir}/{num_layers}_layers/{save_filename}"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return plot_path


def parse_pickle_filename(filepath: str) -> tuple[str, str, str | None, int]:
    """
    Parse a pickle filename to extract dataset, layer_type, encoding, and num_layers.

    Expected format: results/{num_layers}_layers/{dataset}_{layer_type}_{encoding}_graph_dict.pickle

    Args:
        filepath: Path to the pickle file

    Returns:
        Tuple of (dataset_name, layer_type, encoding, num_layers)
    """
    path = Path(filepath)

    # Extract num_layers from directory name (e.g., "4_layers" -> 4)
    dir_name = path.parent.name
    if "_layers" in dir_name:
        num_layers = int(dir_name.split("_")[0])
    else:
        # Default fallback
        num_layers = 4

    # Extract components from filename
    # Format: {dataset}_{layer_type}_{encoding}_graph_dict.pickle
    filename = path.stem  # Remove .pickle extension
    if filename.endswith("_graph_dict"):
        filename = filename[:-11]  # Remove "_graph_dict" suffix

    # Split by underscores
    parts = filename.split("_")

    if len(parts) >= 3:
        dataset_name = parts[0]
        layer_type = parts[1]
        encoding_str = "_".join(parts[2:])  # Handle encodings with underscores
    else:
        raise ValueError(f"Could not parse filename: {filename}")

    # Convert "None" string to None
    encoding: str | None = None if encoding_str == "None" else encoding_str

    return dataset_name, layer_type, encoding, num_layers


def main():
    """Main function to generate sorted plots for all pickle files."""

    # Define results directory
    results_dir = project_root / "results_cluster" / "results_12_31_25" / "results"

    if not results_dir.exists():
        print(f"❌ Results directory not found: {results_dir}")
        sys.exit(1)

    print(f"🔍 Scanning for pickle files in: {results_dir}")

    # Find all pickle files
    pickle_files = list(results_dir.rglob("*_graph_dict.pickle"))

    if not pickle_files:
        print("❌ No pickle files found!")
        sys.exit(1)

    print(f"📁 Found {len(pickle_files)} pickle files to process")

    # Process each pickle file
    processed = 0
    skipped = 0

    for pickle_file in sorted(pickle_files):
        try:
            # Parse filename to get parameters
            dataset_name, layer_type, encoding, num_layers = parse_pickle_filename(
                pickle_file
            )

            print(
                f"📊 Processing: {dataset_name} | {layer_type} | {encoding} | {num_layers} layers"
            )

            # Load and process the data
            with open(pickle_file, "rb") as f:
                data = pickle.load(f)

            # Handle both old format (just graph_dict) and new format (dict with graph_dict key)
            if isinstance(data, dict) and "graph_dict" in data:
                graph_dict = data["graph_dict"]
            else:
                graph_dict = data

            # Compute averages
            graph_indices, average_values = compute_average_per_graph(graph_dict)

            if len(graph_indices) == 0:
                print("   ⚠️  No data found, skipping plot generation")
                skipped += 1
                continue

            # Generate original plot (by index)
            original_plot_path = load_and_plot_average_per_graph(
                pickle_filepath=str(pickle_file),
                dataset_name=dataset_name,
                layer_type=layer_type,
                encoding=encoding,
                num_layers=num_layers,
                task_type="classification",
                output_dir=str(results_dir),
            )[
                0
            ]  # Only get the original plot path

            # Generate rank-ordered plot (sorted by accuracy)
            sort_indices = np.argsort(average_values)[::-1]  # Sort descending
            sorted_average_values = average_values[sort_indices]
            rank_indices = np.arange(1, len(sorted_average_values) + 1)

            # Try to load dataset and extract labels for coloring
            graph_labels = None
            sorted_label_indices = None
            try:
                dataset = load_dataset(dataset_name, encoding)
                if dataset is not None:
                    all_labels = extract_labels_from_dataset(dataset)
                    # Map labels to the graphs that are in graph_dict
                    # graph_indices contains the indices of graphs that have data
                    if len(all_labels) > 0:
                        # Extract labels for graphs that have data
                        # graph_indices may not be sequential (e.g., [0, 1, 2, 5, 7])
                        # so we need to index all_labels with graph_indices
                        max_idx = max(graph_indices) if len(graph_indices) > 0 else -1
                        if max_idx < len(all_labels):
                            # Extract labels for the graphs we have data for
                            # graph_labels[i] corresponds to graph_indices[i]
                            graph_labels = all_labels[graph_indices]
                            # sorted_label_indices are the positions in graph_labels (0, 1, 2, ...)
                            # that correspond to the sorted order
                            sorted_label_indices = sort_indices
                            print(f"   ✅ Loaded labels for {len(graph_labels)} graphs")
                        else:
                            print(
                                f"   ⚠️  Dataset has {len(all_labels)} graphs but max index is {max_idx}"
                            )
            except Exception as e:
                print(f"   ⚠️  Could not load labels: {e}")

            sorted_plot_path = plot_rank_ordered_accuracy(
                rank_indices,
                sorted_average_values,
                dataset_name,
                layer_type,
                encoding,
                num_layers,
                task_type="classification",
                output_dir=str(results_dir),
                save_filename=f"{dataset_name}_{layer_type}_{encoding or 'None'}_by_accuracy.png",
                graph_labels=graph_labels,
                sorted_label_positions=sorted_label_indices,
            )

            print(f"   ✅ Original plot: {original_plot_path}")
            print(f"   ✅ Rank-ordered plot: {sorted_plot_path}")
            processed += 1

        except Exception as e:
            print(f"   ❌ Error processing {pickle_file}: {e}")
            skipped += 1

    print("\n📈 Summary:")
    print(f"   ✅ Successfully processed: {processed} files")
    print(f"   ⚠️  Skipped: {skipped} files")
    print("\n🎉 Done! All sorted plots have been generated.")


if __name__ == "__main__":
    main()
