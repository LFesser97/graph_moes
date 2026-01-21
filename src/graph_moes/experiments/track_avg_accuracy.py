"""Utilities for tracking and plotting average accuracy/error per graph across test appearances."""

import os
import pickle
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np


def compute_average_per_graph(
    graph_dict: Dict[int, List[Union[int, float]]],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute average accuracy (classification) or average error (regression) per graph.

    Args:
        graph_dict: Dictionary mapping graph indices to lists of correctness values (classification)
                    or error values (regression). Format: {graph_idx: [value1, value2, ...]}

    Returns:
        Tuple of (graph_indices, average_values) as numpy arrays
    """
    graph_indices = []
    average_values = []

    for graph_idx in sorted(graph_dict.keys()):
        values = graph_dict[graph_idx]
        if len(values) > 0:
            # For classification: values are 0/1, so mean gives accuracy proportion (0-1)
            # For regression: values are errors, so mean gives average error
            avg_value = np.mean(values)
            graph_indices.append(graph_idx)
            average_values.append(avg_value)

    return np.array(graph_indices), np.array(average_values)


def get_detailed_model_name(
    layer_type: str,
    layer_types: Optional[list] = None,
    router_type: str = "MLP",
    is_encoding_moe: bool = False,
    num_layers: Optional[int] = None,
) -> str:
    """
    Generate a detailed model name that includes MOE specifics and depth.

    Args:
        layer_type: Base layer type (MoE, MoE_E, GCN, etc.)
        layer_types: List of expert types for MOE models
        router_type: Router type for MOE models (MLP or GNN)
        is_encoding_moe: Whether this is an EncodingMoE model
        num_layers: Number of layers (depth) in the model

    Returns:
        Detailed model name string with depth included
    """
    # Add depth suffix if provided
    depth_suffix = f"_L{num_layers}" if num_layers is not None else ""

    if is_encoding_moe:
        # EncodingMoE model - include router type explicitly
        return f"{layer_type}_router_{router_type}{depth_suffix}"
    elif layer_types is not None:
        # Regular MOE model - include router type explicitly and expert combination
        expert_combo = "_".join(layer_types)
        return f"{layer_type}_router_{router_type}_{expert_combo}{depth_suffix}"
    else:
        # Non-MOE model - include depth if provided
        return f"{layer_type}{depth_suffix}" if num_layers is not None else layer_type


def plot_average_per_graph(
    graph_indices: np.ndarray,
    average_values: np.ndarray,
    dataset_name: str,
    layer_type: str,
    encoding: Optional[str],
    num_layers: int,
    task_type: str = "classification",
    output_dir: str = "results",
    save_filename: Optional[str] = None,
    layer_types: Optional[list] = None,
    router_type: str = "MLP",
    is_encoding_moe: bool = False,
) -> str:
    """
    Plot average accuracy (classification) or average error (regression) per graph.

    This creates a "heterogeneity profile" - a visualization showing how model performance
    varies across different graphs in the dataset. The profile reveals which graphs are
    consistently easy/hard for the model to predict, helping identify data heterogeneity.

    Args:
        graph_indices: Array of graph indices (x-axis)
        average_values: Array of average values (y-axis)
        dataset_name: Name of the dataset
        layer_type: Type of layer/model used
        encoding: Encoding used (if any)
        num_layers: Number of layers in the model
        task_type: "classification" or "regression"
        output_dir: Directory to save the plot
        save_filename: Optional custom filename. If None, auto-generated
        layer_types: List of expert types for MOE models
        router_type: Router type for MOE models
        is_encoding_moe: Whether this is an EncodingMoE model

    Returns:
        Path to saved plot file
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    # Create the plot
    ax.scatter(graph_indices, average_values, alpha=0.6, s=20)

    # Set labels and title
    if task_type == "classification":
        ylabel = "Average Accuracy (%)"
        title_suffix = "Accuracy"
        # Convert to percentage (values are 0-1)
        y_values_plot: np.ndarray = average_values * 100
        ax.set_ylim(-5, 105)  # Add some padding
    else:
        ylabel = "Average Error (MAE)"
        title_suffix = "Error"
        y_values_plot = average_values
        # Auto-scale y-axis for regression

    # Update scatter plot with correct y-values
    ax.clear()
    ax.scatter(graph_indices, y_values_plot, alpha=0.6, s=20)

    ax.set_xlabel("Graph Index", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)

    # Create title
    title_parts = [
        f"Average {title_suffix} per Graph",
        f"Dataset: {dataset_name}",
        f"Model: {layer_type} ({num_layers} layers)",
    ]
    if encoding:
        title_parts.append(f"Encoding: {encoding}")
    title = " | ".join(title_parts)
    ax.set_title(title, fontsize=14, fontweight="bold")

    # Add grid for better readability
    ax.grid(True, alpha=0.3, linestyle="--")

    # Add statistics text
    mean_value = np.mean(y_values_plot).item()  # Convert numpy scalar to Python float
    std_value = np.std(y_values_plot).item()
    min_value = np.min(y_values_plot).item()
    max_value = np.max(y_values_plot).item()

    stats_text = (
        f"Mean: {mean_value:.2f} | Std: {std_value:.2f} | "
        f"Min: {min_value:.2f} | Max: {max_value:.2f} | "
        f"Graphs: {len(graph_indices)}"
    )
    ax.text(
        0.5,
        -0.1,
        stats_text,
        transform=ax.transAxes,
        ha="center",
        fontsize=10,
        style="italic",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.3),
    )

    plt.tight_layout()

    # Save the plot
    os.makedirs(f"{output_dir}/{num_layers}_layers", exist_ok=True)
    if save_filename is None:
        encoding_str = f"_{encoding}" if encoding else ""
        detailed_model_name = get_detailed_model_name(
            layer_type, layer_types, router_type, is_encoding_moe, num_layers
        )
        save_filename = (
            f"{output_dir}/{num_layers}_layers/"
            f"{dataset_name}_{detailed_model_name}{encoding_str}_avg_{task_type}_per_graph.png"
        )
    else:
        # Ensure output_dir is in the path if it's a relative path
        if not os.path.isabs(save_filename):
            save_filename = os.path.join(output_dir, save_filename)

    fig.savefig(save_filename, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return save_filename


def load_and_plot_average_per_graph(
    pickle_filepath: str,
    dataset_name: str,
    layer_type: str,
    encoding: Union[str, None],
    num_layers: int,
    task_type: str = "classification",
    output_dir: str = "results",
    layer_types: Optional[list] = None,
    router_type: str = "MLP",
    skip_connection: bool = False,
    normalize_features: bool = False,
    is_encoding_moe: bool = False,
) -> Tuple[str, str]:
    """
    Load graph_dict from pickle file and create average accuracy/error plots.
    Creates two plots: one ordered by graph index, one ordered by highest average accuracy.

    Args:
        pickle_filepath: Path to the pickle file containing graph_dict
        dataset_name: Name of the dataset
        layer_type: Type of layer/model used
        encoding: Encoding used (if any)
        num_layers: Number of layers in the model
        task_type: "classification" or "regression"
        output_dir: Directory to save the plots
        layer_types: List of expert types for MOE models
        router_type: Router type for MOE models
        skip_connection: Whether skip connections were used
        normalize_features: Whether feature normalization was used
        is_encoding_moe: Whether this is an EncodingMoE model

    Returns:
        Tuple of (original_plot_path, sorted_plot_path)
    """
    # Load the pickle file
    with open(pickle_filepath, "rb") as f:
        data = pickle.load(f)

    # Handle both old format (just graph_dict) and new format (dict with graph_dict key)
    if isinstance(data, dict) and "graph_dict" in data:
        graph_dict = data["graph_dict"]
    else:
        graph_dict = data

    # Compute averages
    graph_indices, average_values = compute_average_per_graph(graph_dict)

    if len(graph_indices) == 0:
        print(f"⚠️  No data found in {pickle_filepath}, skipping plot generation")
        return "", ""

    # Create and save original plot (ordered by graph index)
    # This creates a "heterogeneity profile by index" - shows performance variation across
    # graphs in their original dataset order, revealing which graph indices are easier/harder.
    # Include encoding, skip connection, and normalize_features in filename
    # Use full detailed encoding name (e.g., "hg_rwpe_we_k20", "g_lape_k8", not abbreviated)
    skip_str = "skip_true" if skip_connection else "skip_false"
    norm_str = "norm_true" if normalize_features else "norm_false"
    encoding_str = (
        encoding if encoding else "none"
    )  # encoding should be full detailed name like "hg_rwpe_we_k20"
    encoding_suffix = f"_encodings_{encoding_str}"
    detailed_model_name = get_detailed_model_name(
        layer_type, layer_types, router_type, is_encoding_moe, num_layers
    )
    original_plot_path = plot_average_per_graph(
        graph_indices,
        average_values,
        dataset_name,
        layer_type,
        encoding,
        num_layers,
        task_type,
        output_dir,
        save_filename=f"{dataset_name}_{detailed_model_name}_{skip_str}_{norm_str}{encoding_suffix}_by_index.png",
        layer_types=layer_types,
        router_type=router_type,
        is_encoding_moe=is_encoding_moe,
    )

    # Create sorted plot (ordered by highest average accuracy)
    # This creates a "heterogeneity profile by accuracy" - shows performance distribution
    # sorted from highest to lowest accuracy, revealing the overall distribution of
    # model performance across the dataset.
    # Sort by average values in descending order (highest accuracy first)
    sort_indices = np.argsort(average_values)[::-1]  # Sort descending
    sorted_graph_indices = graph_indices[sort_indices]
    sorted_average_values = average_values[sort_indices]

    sorted_plot_path = plot_average_per_graph(
        sorted_graph_indices,
        sorted_average_values,
        dataset_name,
        layer_type,
        encoding,
        num_layers,
        task_type,
        output_dir,
        save_filename=f"{dataset_name}_{detailed_model_name}_{skip_str}_{norm_str}{encoding_suffix}_by_accuracy.png",
        layer_types=layer_types,
        router_type=router_type,
        is_encoding_moe=is_encoding_moe,
    )

    return original_plot_path, sorted_plot_path
