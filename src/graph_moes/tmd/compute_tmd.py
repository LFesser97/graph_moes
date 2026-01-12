"""Functions for computing pairwise TMD and class-distance ratios."""

import multiprocessing as mp
import os
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
from torch_geometric.data import Data

from graph_moes.tmd.tmd import TMD


def extract_labels(
    dataset: List[Data], dataset_name: Optional[str] = None  # noqa: ARG001
) -> Tuple[np.ndarray, int, dict]:
    """Extract labels from a dataset.

    Args:
        dataset: List of PyTorch Geometric Data objects
        dataset_name: Optional name of the dataset (for debugging)

    Returns:
        Tuple of (labels_array, num_classes, label_counts)
        - labels_array: numpy array of labels for each graph
        - num_classes: number of unique classes
        - label_counts: dictionary mapping label to count
    """
    labels: List[int] = []

    for i, graph in enumerate(dataset):
        if hasattr(graph, "y"):
            y = graph.y
            # Check if y is None
            if y is None:
                raise ValueError(f"Graph {i} has 'y' attribute but it is None")
            # Handle different label formats
            if isinstance(y, torch.Tensor):
                if y.dim() == 0:  # Scalar
                    label = int(y.item())
                elif y.dim() == 1 and len(y) == 1:  # 1D tensor with 1 element
                    label = int(y[0].item())
                elif y.dim() == 1:  # 1D tensor
                    if len(y) == 1:
                        label = int(y[0].item())
                    else:
                        # Multi-label: use first active label
                        non_zero = torch.nonzero(y, as_tuple=False)
                        if len(non_zero) > 0:
                            label = int(y[non_zero[0]].item())
                        else:
                            label = 0
                else:
                    # Multi-dimensional, use first element
                    label = int(y.flatten()[0].item())
            else:
                label = int(y)
            labels.append(label)
        else:
            raise ValueError(f"Graph {i} does not have a 'y' attribute (label)")

    labels_array = np.array(labels)
    unique_labels = np.unique(labels_array)
    num_classes = len(unique_labels)
    label_counts = {
        int(label): int(np.sum(labels_array == label)) for label in unique_labels
    }

    return labels_array, num_classes, label_counts


def _compute_single_tmd_pair(
    args: Tuple[int, int, List[Data], Union[float, List[float]], int],
) -> Tuple[int, int, float]:
    """Compute TMD for a single pair of graphs (for multiprocessing).

    Args:
        args: Tuple of (i, j, dataset, w, L) where:
            - i, j: indices of the two graphs
            - dataset: List of graphs (needed for indexing)
            - w: TMD weighting constant(s)
            - L: TMD computation tree depth

    Returns:
        Tuple of (i, j, tmd_value)
    """
    i, j, dataset, w, L = args
    tmd_value = TMD(dataset[i], dataset[j], w=w, L=L)
    return (i, j, tmd_value)


def compute_tmd_matrix(
    dataset: List[Data],
    w: Union[float, List[float]] = 1.0,
    L: int = 4,
    verbose: bool = True,
    cache_path: Optional[str] = None,
    n_jobs: Optional[int] = None,
) -> np.ndarray:
    """Compute pairwise TMD matrix for all graphs in a dataset.

    Computes TMD for all pairs (n choose 2) and returns a symmetric matrix.
    Can use multiprocessing to parallelize pairwise TMD computations.

    Args:
        dataset: List of PyTorch Geometric Data objects
        w: Weighting constant(s) for TMD computation. Can be a single float
           or a list of weights for each layer (must have length L-1)
        L: Depth of computation trees for TMD (default: 4)
        verbose: Whether to print progress
        cache_path: Optional path to save/load the TMD matrix. If file exists,
                   it will be loaded instead of recomputing.
        n_jobs: Number of parallel jobs to use. If None, uses all available CPUs.
                If 1, runs sequentially. Default: None (all CPUs)

    Returns:
        Symmetric numpy array of shape (n, n) where n is the number of graphs.
        tmd_matrix[i, j] = TMD(dataset[i], dataset[j])
    """
    n = len(dataset)

    # Check if cached version exists
    if cache_path is not None and os.path.exists(cache_path):
        if verbose:
            print(f"📂 Loading cached TMD matrix from {cache_path}")
        loaded_matrix: np.ndarray = np.load(cache_path)
        return loaded_matrix

    if verbose:
        print(f"🔄 Computing TMD matrix for {n} graphs...")
        print(f"   Total pairs to compute: {n * (n - 1) // 2}")

    tmd_matrix = np.zeros((n, n))

    # Set diagonal to 0 (distance to itself)
    for i in range(n):
        tmd_matrix[i, i] = 0.0

    # Prepare arguments for pairwise TMD computation
    total_pairs = n * (n - 1) // 2
    pair_args = [(i, j, dataset, w, L) for i in range(n) for j in range(i + 1, n)]

    # Determine number of jobs
    if n_jobs is None:
        n_jobs = mp.cpu_count()
    elif n_jobs < 1:
        n_jobs = 1

    if verbose:
        if n_jobs > 1:
            print(f"   Using {n_jobs} parallel workers...")
        else:
            print(f"   Running sequentially (n_jobs=1)...")

    # Compute pairwise TMD
    if n_jobs == 1:
        # Sequential computation
        computed = 0
        for i, j, dataset, w, L in pair_args:
            tmd_value = TMD(dataset[i], dataset[j], w=w, L=L)
            tmd_matrix[i, j] = tmd_value
            tmd_matrix[j, i] = tmd_value  # Symmetric

            computed += 1
            if verbose and computed % max(1, total_pairs // 100) == 0:
                print(
                    f"   Progress: {computed}/{total_pairs} pairs ({100*computed/total_pairs:.1f}%)"
                )
    else:
        # Parallel computation using multiprocessing
        # Use fork context (similar to OllivierRicci.py) for better compatibility
        try:
            ctx = mp.get_context("fork")
        except (ValueError, RuntimeError):
            # Fallback to default context if fork is not available
            ctx = mp

        computed = 0
        with ctx.Pool(processes=n_jobs) as pool:
            # Use imap_unordered for better progress tracking
            chunksize = max(1, total_pairs // (n_jobs * 4))
            if chunksize == 0:
                chunksize = 1

            results = pool.imap_unordered(
                _compute_single_tmd_pair, pair_args, chunksize=chunksize
            )

            for i, j, tmd_value in results:
                tmd_matrix[i, j] = tmd_value
                tmd_matrix[j, i] = tmd_value  # Symmetric

                computed += 1
                if verbose and computed % max(1, total_pairs // 100) == 0:
                    print(
                        f"   Progress: {computed}/{total_pairs} pairs ({100*computed/total_pairs:.1f}%)"
                    )

    if verbose:
        print("✅ TMD matrix computation complete!")

    # Save to cache if path provided
    if cache_path is not None:
        os.makedirs(
            os.path.dirname(cache_path) if os.path.dirname(cache_path) else ".",
            exist_ok=True,
        )
        np.save(cache_path, tmd_matrix)
        if verbose:
            print(f"💾 Saved TMD matrix to {cache_path}")

    return tmd_matrix


def compute_class_distance_ratios(
    tmd_matrix: np.ndarray,
    labels: np.ndarray,
    verbose: bool = True,
) -> Tuple[np.ndarray, dict]:
    """Compute class-distance ratio for each graph.

    For each graph G_i, the class-distance ratio is:
        ρ(G_i) = min{TMD(G_i, G_j) : Y_i = Y_j, j ≠ i}
                 / min{TMD(G_i, G_j) : Y_i ≠ Y_j}

    Args:
        tmd_matrix: Symmetric TMD matrix of shape (n, n)
        labels: Array of labels for each graph, shape (n,)
        verbose: Whether to print statistics

    Returns:
        Tuple of (ratios, stats_dict)
        - ratios: Array of class-distance ratios for each graph, shape (n,)
        - stats_dict: Dictionary with statistics:
            - 'mean': mean ratio
            - 'median': median ratio
            - 'std': standard deviation
            - 'min': minimum ratio
            - 'max': maximum ratio
            - 'num_hard': number of graphs with ρ > 1
            - 'num_easy': number of graphs with ρ < 1
            - 'num_ambiguous': number of graphs with ρ ≈ 1 (within 0.01)
    """
    n = len(labels)
    ratios = np.zeros(n)

    if verbose:
        print(f"🔄 Computing class-distance ratios for {n} graphs...")

    for i in range(n):
        # Find minimum TMD to same-class graphs
        same_class_mask = (labels == labels[i]) & (np.arange(n) != i)
        if np.any(same_class_mask):
            min_same_class = np.min(tmd_matrix[i, same_class_mask])
        else:
            # If no other graph has the same label, use a large value
            min_same_class = np.max(tmd_matrix[i, :]) + 1.0

        # Find minimum TMD to different-class graphs
        diff_class_mask = labels != labels[i]
        if np.any(diff_class_mask):
            min_diff_class = np.min(tmd_matrix[i, diff_class_mask])
        else:
            # If all graphs have the same label, use a small value
            min_diff_class = 0.0

        # Compute ratio
        if min_diff_class > 0:
            ratios[i] = min_same_class / min_diff_class
        else:
            # Edge case: all graphs have same label
            ratios[i] = np.inf if min_same_class > 0 else 1.0

    # Compute statistics
    finite_ratios = ratios[np.isfinite(ratios)]
    stats_dict = {
        "mean": float(np.mean(finite_ratios)) if len(finite_ratios) > 0 else np.nan,
        "median": float(np.median(finite_ratios)) if len(finite_ratios) > 0 else np.nan,
        "std": float(np.std(finite_ratios)) if len(finite_ratios) > 0 else np.nan,
        "min": float(np.min(finite_ratios)) if len(finite_ratios) > 0 else np.nan,
        "max": float(np.max(finite_ratios)) if len(finite_ratios) > 0 else np.nan,
        "num_hard": int(np.sum(ratios > 1.0)),
        "num_easy": int(np.sum(ratios < 1.0)),
        "num_ambiguous": int(np.sum(np.abs(ratios - 1.0) < 0.01)),
        "num_infinite": int(np.sum(np.isinf(ratios))),
    }

    if verbose:
        print("✅ Class-distance ratio computation complete!")
        print(f"   Mean ratio: {stats_dict['mean']:.4f}")
        print(f"   Median ratio: {stats_dict['median']:.4f}")
        print(f"   Std ratio: {stats_dict['std']:.4f}")
        print(f"   Easy graphs (ρ < 1): {stats_dict['num_easy']}")
        print(f"   Hard graphs (ρ > 1): {stats_dict['num_hard']}")
        print(f"   Ambiguous graphs (ρ ≈ 1): {stats_dict['num_ambiguous']}")

    return ratios, stats_dict


def save_tmd_results(
    dataset_name: str,
    tmd_matrix: np.ndarray,
    ratios: np.ndarray,
    labels: np.ndarray,
    stats_dict: dict,
    output_dir: str = "tmd_results",
) -> dict:
    """Save TMD computation results to files.

    Args:
        dataset_name: Name of the dataset
        tmd_matrix: TMD distance matrix of shape (n, n)
        ratios: Class-distance ratios array of shape (n,)
        labels: Graph labels array of shape (n,)
        stats_dict: Statistics dictionary
        output_dir: Output directory for results (default: "tmd_results")

    Returns:
        Dictionary mapping file type to saved file paths:
        - 'tmd_matrix': path to .npy file
        - 'class_ratios': path to .csv file
        - 'stats': path to .json file
    """
    import json

    os.makedirs(output_dir, exist_ok=True)

    # Save TMD matrix
    tmd_matrix_path = os.path.join(output_dir, f"{dataset_name.lower()}_tmd_matrix.npy")
    np.save(tmd_matrix_path, tmd_matrix)

    # Save class-distance ratios as CSV
    import pandas as pd

    ratios_df = pd.DataFrame(
        {
            "graph_index": np.arange(len(ratios)),
            "label": labels,
            "class_distance_ratio": ratios,
        }
    )
    ratios_path = os.path.join(output_dir, f"{dataset_name.lower()}_class_ratios.csv")
    ratios_df.to_csv(ratios_path, index=False)

    # Save statistics as JSON
    stats_path = os.path.join(output_dir, f"{dataset_name.lower()}_tmd_stats.json")
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats_dict, f, indent=2)

    return {
        "tmd_matrix": tmd_matrix_path,
        "class_ratios": ratios_path,
        "stats": stats_path,
    }
