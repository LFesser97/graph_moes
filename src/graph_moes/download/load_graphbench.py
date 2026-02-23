"""
Helper module to load GraphBench datasets and convert them to PyTorch Geometric format.

GraphBench datasets need to be converted to PyG Data objects for compatibility
with the existing codebase.
"""

from pathlib import Path
from typing import Any, Callable, List, Optional, Union

import torch
from torch_geometric.data import Data

try:
    from graphbench.loader import Loader  # type: ignore
except ImportError as exc:
    raise ImportError(
        "graphbench-lib is not installed. Install it with: pip install graphbench-lib"
    ) from exc


def load_graphbench_dataset(
    dataset_name: str,
    root: Union[str, Path],
    pre_filter: Optional[Callable[[Any], bool]] = None,
    pre_transform: Optional[Callable[[Any], Any]] = None,
    transform: Optional[Callable[[Any], Any]] = None,
) -> List[Data]:
    """
    Load a GraphBench dataset and convert it to a list of PyTorch Geometric Data objects.

    Args:
        dataset_name: Name of the GraphBench dataset
        root: Root directory where the dataset is stored
        pre_filter: Optional pre-filter function (PyG-style)
        pre_transform: Optional pre-transform function (PyG-style)
        transform: Optional transform function (PyG-style)

    Returns:
        List of PyTorch Geometric Data objects

    Raises:
        ImportError: If graphbench-lib is not installed
        ValueError: If dataset cannot be loaded or converted
    """
    # For certain datasets, try loading directly from .pt files first
    # This handles cases where the loader doesn't properly convert batched format
    root_path = Path(root)

    # Try direct loading for chipdesign and algorithmic_reasoning_easy
    if dataset_name in ["chipdesign", "algorithmic_reasoning_easy"]:
        direct_graphs = _load_graphbench_direct(dataset_name, root_path)
        if direct_graphs:
            return direct_graphs

    # Load the dataset using GraphBench Loader
    # Note: Loader expects (root, dataset_names, ...) based on actual signature
    loader = Loader(
        root=root,
        dataset_names=dataset_name,
        pre_filter=pre_filter,
        pre_transform=pre_transform,
        transform=transform,
    )

    dataset = loader.load()

    # Convert GraphBench dataset to list of PyG Data objects
    pyg_dataset = []

    # GraphBench datasets may come in different formats
    # We need to handle different return types from loader.load()

    # Check if dataset is a tuple of dictionaries (batched format from .pt files)
    # This happens when GraphBench loader returns raw data before conversion
    if isinstance(dataset, tuple):
        # Handle tuple of dictionaries (train/val/test splits or batched data)
        for split_idx, item in enumerate(dataset):
            if isinstance(item, dict):
                # Check if it's a batched dictionary that needs splitting
                if "num_nodes" in item or "_num_nodes" in item or "sample_idx" in item:
                    batched_graphs = _split_batched_dict(item)
                    if batched_graphs:
                        pyg_dataset.extend(batched_graphs)
                        continue
                # Otherwise try normal conversion
                pyg_data = _convert_to_pyg_data(item)
                if pyg_data is not None:
                    pyg_dataset.append(pyg_data)
    elif isinstance(dataset, list):
        # If it's already a list, iterate through it
        for item in dataset:
            # Check if item is a batched dictionary format
            if isinstance(item, dict) and "num_nodes" in item:
                batched_graphs = _split_batched_dict(item)
                pyg_dataset.extend(batched_graphs)
            else:
                pyg_data = _convert_to_pyg_data(item)
                if pyg_data is not None:
                    pyg_dataset.append(pyg_data)
    elif hasattr(dataset, "__iter__") and hasattr(dataset, "__len__"):
        # If it's an iterable with length (like PyG dataset)
        for i in range(len(dataset)):
            item = dataset[i]
            if isinstance(item, dict) and "num_nodes" in item:
                batched_graphs = _split_batched_dict(item)
                pyg_dataset.extend(batched_graphs)
            else:
                pyg_data = _convert_to_pyg_data(item)
                if pyg_data is not None:
                    pyg_dataset.append(pyg_data)
    elif hasattr(dataset, "__iter__"):
        # If it's just iterable
        for item in dataset:
            if isinstance(item, dict) and "num_nodes" in item:
                batched_graphs = _split_batched_dict(item)
                pyg_dataset.extend(batched_graphs)
            else:
                pyg_data = _convert_to_pyg_data(item)
                if pyg_data is not None:
                    pyg_dataset.append(pyg_data)
    else:
        # Try to treat as single graph or batched dict
        if isinstance(dataset, dict) and "num_nodes" in dataset:
            batched_graphs = _split_batched_dict(dataset)
            pyg_dataset.extend(batched_graphs)
        else:
            pyg_data = _convert_to_pyg_data(dataset)
            if pyg_data is not None:
                pyg_dataset.append(pyg_data)

    if not pyg_dataset:
        raise ValueError(
            f"Could not convert GraphBench dataset '{dataset_name}' to PyG format. "
            "The dataset may be empty or in an unsupported format."
        )

    return pyg_dataset


def _load_graphbench_direct(dataset_name: str, root_path: Path) -> List[Data]:
    """
    Load GraphBench dataset directly from .pt files, bypassing the loader.
    This is needed for datasets where the loader doesn't properly handle batched format.

    Args:
        dataset_name: Name of the dataset
        root_path: Root directory path

    Returns:
        List of Data objects, or empty list if loading fails
    """
    graphs = []

    try:
        if dataset_name == "chipdesign":
            # Chipdesign has train/val/test splits
            for split in ["train", "val", "test"]:
                pt_file = (
                    root_path
                    / "chipdesign"
                    / "chipdesign"
                    / "processed"
                    / f"chipdesign_{split}.pt"
                )
                if pt_file.exists():
                    data = torch.load(pt_file, map_location="cpu")
                    if isinstance(data, tuple) and len(data) > 0:
                        # Process each split in the tuple
                        for item in data:
                            if isinstance(item, dict) and (
                                "num_nodes" in item or "sample_idx" in item
                            ):
                                split_graphs = _split_batched_dict(item)
                                graphs.extend(split_graphs)

        elif dataset_name == "algorithmic_reasoning_easy":
            # Algorithmic reasoning has multiple subdirectories
            algoreas_dir = root_path / "algoreas"
            if algoreas_dir.exists():
                # Look for easy train/test files in subdirectories
                for subdir in algoreas_dir.iterdir():
                    if subdir.is_dir():
                        processed_dir = subdir / "processed"
                        if processed_dir.exists():
                            # Look for easy train/test files
                            for pt_file in processed_dir.glob("*_easy_*.pt"):
                                try:
                                    data = torch.load(pt_file, map_location="cpu")
                                    if isinstance(data, (list, tuple)):
                                        for item in data:
                                            if isinstance(item, dict) and (
                                                "num_nodes" in item
                                                or "sample_idx" in item
                                            ):
                                                split_graphs = _split_batched_dict(item)
                                                graphs.extend(split_graphs)
                                except Exception:
                                    continue

    except Exception:
        # If direct loading fails, return empty list and fall back to loader
        pass

    return graphs


def _split_batched_dict(batched_dict: dict) -> List[Data]:
    """
    Split a batched dictionary format into individual PyG Data objects.

    The batched format has tensors for multiple graphs, with 'num_nodes' indicating
    how many nodes each graph has. We need to split the batched tensors based on this.

    Args:
        batched_dict: Dictionary with batched graph data
            - 'num_nodes': tensor of shape [num_graphs] with node counts
            - 'x': batched node features
            - 'edge_index': batched edge indices
            - 'edge_attr': batched edge attributes (optional)
            - 'y': graph labels (optional)
            - Other optional attributes

    Returns:
        List of individual PyG Data objects
    """
    graphs = []

    try:
        # Try to get num_nodes per graph - could be tensor, list, or int
        num_nodes_list = None

        # First try _num_nodes (often a list)
        if "_num_nodes" in batched_dict:
            _nn = batched_dict["_num_nodes"]
            if isinstance(_nn, (list, tuple)):
                num_nodes_list = list(_nn)
            elif isinstance(_nn, torch.Tensor):
                num_nodes_list = _nn.cpu().tolist()

        # If not available, try to infer from sample_idx
        if num_nodes_list is None and "sample_idx" in batched_dict:
            sample_idx = batched_dict["sample_idx"]
            if isinstance(sample_idx, torch.Tensor):
                # Count nodes per graph using sample_idx
                unique_graphs, counts = torch.unique(sample_idx, return_counts=True)
                num_nodes_list = counts.cpu().tolist()

        # Last resort: try num_nodes as tensor
        if num_nodes_list is None and "num_nodes" in batched_dict:
            nn = batched_dict["num_nodes"]
            if isinstance(nn, torch.Tensor):
                num_nodes_list = nn.cpu().tolist()
            elif isinstance(nn, (list, tuple)):
                num_nodes_list = list(nn)

        if num_nodes_list is None or len(num_nodes_list) == 0:
            # Can't split without num_nodes information
            return []

        num_graphs = len(num_nodes_list)

        # Get batched tensors
        x_batched = batched_dict.get("x")
        edge_index_batched = batched_dict.get("edge_index")
        edge_attr_batched = batched_dict.get("edge_attr")
        y_batched = batched_dict.get("y")
        sample_idx = batched_dict.get("sample_idx")

        # Calculate cumulative node offsets
        node_offsets = [0]
        for i in range(num_graphs):
            node_offsets.append(node_offsets[-1] + num_nodes_list[i])

        # If sample_idx is available, use it to determine which nodes belong to which graph
        # This is more reliable than assuming sequential ordering
        use_sample_idx = sample_idx is not None and isinstance(sample_idx, torch.Tensor)

        # Split each graph
        for graph_idx in range(num_graphs):
            # Determine which nodes belong to this graph
            if use_sample_idx:
                # Use sample_idx to find nodes belonging to this graph
                node_mask = sample_idx == graph_idx
                node_indices = torch.where(node_mask)[0]
                if len(node_indices) == 0:
                    continue  # Skip if no nodes for this graph
                # Sort indices for consistent ordering
                node_indices = torch.sort(node_indices)[0]
            else:
                # Use sequential ordering
                node_start = node_offsets[graph_idx]
                node_end = node_offsets[graph_idx + 1]
                node_indices = torch.arange(node_start, node_end, dtype=torch.long)
                node_mask = None

            # Extract node features for this graph
            x = None
            if x_batched is not None and isinstance(x_batched, torch.Tensor):
                if len(x_batched.shape) >= 2:
                    if use_sample_idx:
                        x = x_batched[node_indices]
                    else:
                        x = x_batched[node_indices]
                elif len(x_batched.shape) == 1:
                    # Single feature per node
                    if use_sample_idx:
                        x = x_batched[node_indices].unsqueeze(1)
                    else:
                        x = x_batched[node_indices].unsqueeze(1)

            # Extract edges for this graph
            edge_index = None
            edge_attr = None
            if edge_index_batched is not None and isinstance(
                edge_index_batched, torch.Tensor
            ):
                # Find edges that belong to this graph
                # Edges are in [2, num_edges] format
                if edge_index_batched.shape[0] == 2:
                    if use_sample_idx:
                        # Filter edges where both nodes belong to this graph
                        mask = (sample_idx[edge_index_batched[0]] == graph_idx) & (
                            sample_idx[edge_index_batched[1]] == graph_idx
                        )
                    else:
                        # Filter edges where both nodes are in [node_start, node_end)
                        mask = (
                            (edge_index_batched[0] >= node_start)
                            & (edge_index_batched[0] < node_end)
                            & (edge_index_batched[1] >= node_start)
                            & (edge_index_batched[1] < node_end)
                        )

                    if mask.any():
                        edge_indices = edge_index_batched[:, mask]
                        # Adjust node indices to be relative to this graph
                        if use_sample_idx:
                            # Create mapping from global node index to local index
                            # Use searchsorted for efficient mapping (node_indices is sorted)
                            edge_index = torch.zeros_like(edge_indices)
                            for i in range(edge_indices.shape[1]):
                                src_global = edge_indices[0, i].item()
                                dst_global = edge_indices[1, i].item()
                                # Use searchsorted since node_indices is sorted
                                src_local = torch.searchsorted(
                                    node_indices, src_global, right=False
                                )
                                dst_local = torch.searchsorted(
                                    node_indices, dst_global, right=False
                                )
                                # Verify the mapping is correct
                                if (
                                    src_local < len(node_indices)
                                    and node_indices[src_local] == src_global
                                ):
                                    edge_index[0, i] = src_local
                                if (
                                    dst_local < len(node_indices)
                                    and node_indices[dst_local] == dst_global
                                ):
                                    edge_index[1, i] = dst_local
                        else:
                            node_start = node_offsets[graph_idx]
                            edge_index = edge_indices - node_start

                        # Extract corresponding edge attributes if available
                        if edge_attr_batched is not None and isinstance(
                            edge_attr_batched, torch.Tensor
                        ):
                            if len(edge_attr_batched.shape) == 1:
                                edge_attr = edge_attr_batched[mask]
                            elif len(edge_attr_batched.shape) == 2:
                                edge_attr = edge_attr_batched[mask]

            # Extract graph label if available
            y = None
            if y_batched is not None:
                if isinstance(y_batched, torch.Tensor):
                    if y_batched.shape[0] == num_graphs:
                        # One label per graph
                        y = y_batched[graph_idx]
                    elif len(y_batched.shape) > 1 and y_batched.shape[0] == num_graphs:
                        # Multi-dimensional labels
                        y = y_batched[graph_idx]
                elif (
                    isinstance(y_batched, (list, tuple))
                    and len(y_batched) == num_graphs
                ):
                    y = y_batched[graph_idx]

            # Create Data object
            data_dict = {}
            if x is not None:
                data_dict["x"] = x
            if edge_index is not None:
                data_dict["edge_index"] = edge_index
            if edge_attr is not None:
                data_dict["edge_attr"] = edge_attr
            if y is not None:
                data_dict["y"] = y

            # Add any other attributes that might be per-graph
            for key, value in batched_dict.items():
                if key not in [
                    "x",
                    "edge_index",
                    "edge_attr",
                    "y",
                    "num_nodes",
                    "_num_nodes",
                    "batch",
                ]:
                    # Check if it's a per-graph attribute
                    if isinstance(value, torch.Tensor) and value.shape[0] == num_graphs:
                        data_dict[key] = value[graph_idx]
                    elif isinstance(value, (list, tuple)) and len(value) == num_graphs:
                        data_dict[key] = value[graph_idx]

            if "x" in data_dict or "edge_index" in data_dict:
                graphs.append(Data(**data_dict))

    except Exception:
        # If splitting fails, return empty list
        # The caller will handle the error
        pass

    return graphs


def _convert_to_pyg_data(item: Any) -> Optional[Data]:
    """
    Convert a single item from GraphBench to PyTorch Geometric Data format.

    Args:
        item: Item from GraphBench dataset (could be PyG Data, dict, or custom object)

    Returns:
        PyG Data object or None if conversion fails
    """
    # If it's already a PyG Data object, return it
    if isinstance(item, Data):
        return item

    # If it's a dictionary, try to extract PyG attributes
    if isinstance(item, dict):
        try:
            return Data(
                x=item.get("x"),
                edge_index=item.get("edge_index"),
                edge_attr=item.get("edge_attr"),
                y=item.get("y"),
                pos=item.get("pos"),
                batch=item.get("batch"),
            )
        except (ValueError, TypeError, AttributeError):
            pass

    # If the item has PyG-like attributes, try to construct Data object
    if hasattr(item, "x") and hasattr(item, "edge_index"):
        try:
            data_dict: dict[str, Any] = {
                "x": item.x,
                "edge_index": item.edge_index,
            }

            # Add optional attributes
            if hasattr(item, "edge_attr"):
                data_dict["edge_attr"] = item.edge_attr
            if hasattr(item, "y"):
                data_dict["y"] = item.y
            if hasattr(item, "pos"):
                data_dict["pos"] = item.pos
            if hasattr(item, "batch"):
                data_dict["batch"] = item.batch

            return Data(**data_dict)
        except (ValueError, TypeError, AttributeError):
            pass

    # Last resort: try to access as if it's a PyG-style object
    try:
        # Some GraphBench datasets might return objects with .data attribute
        if hasattr(item, "data"):
            return _convert_to_pyg_data(item.data)

        # Try to get attributes directly
        attrs: dict[str, Any] = {}
        for attr in ["x", "edge_index", "edge_attr", "y", "pos", "batch"]:
            if hasattr(item, attr):
                attrs[attr] = getattr(item, attr)

        if "x" in attrs and "edge_index" in attrs:
            return Data(**attrs)
    except (ValueError, TypeError, AttributeError):
        pass

    return None


def get_graphbench_evaluator(dataset_name: str) -> Optional[Any]:
    """
    Get the GraphBench Evaluator for standardized metrics.

    Args:
        dataset_name: Name of the GraphBench dataset

    Returns:
        Evaluator instance or None if graphbench-lib is not installed

    Example:
        evaluator = get_graphbench_evaluator("socialnetwork")
        results = evaluator.evaluate(y_true, y_pred)
    """
    try:
        from graphbench.evaluator import Evaluator  # type: ignore

        # Get metric name for the dataset (you may need to look this up in master.csv)
        # For now, return the evaluator initialized with dataset_name
        return Evaluator(dataset_name)
    except ImportError:
        return None


def get_graphbench_dataset_info(dataset_name: str) -> dict:
    """
    Get information about a GraphBench dataset without loading it.

    Args:
        dataset_name: Name of the GraphBench dataset

    Returns:
        Dictionary with dataset information
    """
    # Domain to dataset_name mapping from GraphBench docs
    domain_mapping = {
        "socialnetwork": "Social media",
        "co": "Combinatorial Optimization",
        "sat": "SAT solving",
        "algorithmic_reasoning_easy": "Algorithmic reasoning (easy)",
        "algorithmic_reasoning_medium": "Algorithmic reasoning (medium)",
        "algorithmic_reasoning_hard": "Algorithmic reasoning (hard)",
        "electronic_circuits": "Electronic circuits",
        "chipdesign": "Chip design",
        "weather": "Weather forecasting",
    }

    domain = domain_mapping.get(dataset_name, "Unknown")

    return {
        "name": dataset_name,
        "domain": domain,
        "source": "GraphBench",
        "note": "Use load_graphbench_dataset() to load actual data",
        "evaluator_note": "Use get_graphbench_evaluator() for standardized metrics",
    }
