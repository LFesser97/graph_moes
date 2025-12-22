"""
Helper module to load GraphBench datasets and convert them to PyTorch Geometric format.

GraphBench datasets need to be converted to PyG Data objects for compatibility
with the existing codebase.
"""

from pathlib import Path
from typing import Any, Callable, List, Optional, Union

from torch_geometric.data import Data

try:
    from graphbench.loader import Loader  # type: ignore
    from graphbench.evaluator import Evaluator  # type: ignore
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
    if isinstance(dataset, list):
        # If it's already a list, iterate through it
        for item in dataset:
            pyg_data = _convert_to_pyg_data(item)
            if pyg_data is not None:
                pyg_dataset.append(pyg_data)
    elif hasattr(dataset, "__iter__") and hasattr(dataset, "__len__"):
        # If it's an iterable with length (like PyG dataset)
        for i in range(len(dataset)):
            item = dataset[i]
            pyg_data = _convert_to_pyg_data(item)
            if pyg_data is not None:
                pyg_dataset.append(pyg_data)
    elif hasattr(dataset, "__iter__"):
        # If it's just iterable
        for item in dataset:
            pyg_data = _convert_to_pyg_data(item)
            if pyg_data is not None:
                pyg_dataset.append(pyg_data)
    else:
        # Try to treat as single graph
        pyg_data = _convert_to_pyg_data(dataset)
        if pyg_data is not None:
            pyg_dataset.append(pyg_data)

    if not pyg_dataset:
        raise ValueError(
            f"Could not convert GraphBench dataset '{dataset_name}' to PyG format. "
            "The dataset may be empty or in an unsupported format."
        )

    return pyg_dataset


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
