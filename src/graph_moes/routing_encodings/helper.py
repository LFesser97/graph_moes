"""Helper functions for EncodingMoE (Encoding Mixture of Experts) model.

This module provides utility functions for initializing and managing EncodingMoE models,
which dynamically select and combine different structural encodings (e.g., LDP, RWPE, LAPE)
for graph neural networks using a routing mechanism.

The main functions include:
- _initialize_encoding_moe_model: Creates and configures an EncodingMoE model with proper
  input dimensions and encoding configurations
- _create_encoding_moe_loaders: Creates DataLoaders for encoded datasets used during training
- _get_encoding_moe_encoded_graphs_for_batch: Retrieves encoded graph batches for a specific
  encoding during training/evaluation
- _create_encoding_moe_loaders_for_split: Creates DataLoaders for encoded datasets for specific
  data splits (train/validation/test)

These functions handle the complexity of:
- Managing multiple precomputed encodings (graph-level and hypergraph-level)
- Ensuring proper dimension alignment between base features and encoded features
- Creating synchronized DataLoaders that maintain alignment between base and encoded graphs
- Handling encoding configurations (dimensions, whether encoding replaces or appends to base features)
"""

# ============================================================================
# EncodingMoE Helper Functions
# ============================================================================

import copy
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset
from torch_geometric.loader import DataLoader

try:
    from attrdict3 import AttrDict  # Python 3.10+ compatible
except ImportError:
    from attrdict import AttrDict  # Fallback for older Python

from graph_moes.architectures.graph_model import GNN, GPS, OrthogonalGCN, UnitaryGCN
from graph_moes.routing_encodings.encoding_moe import EncodingMoE
from graph_moes.routing_encodings.encoding_utils import create_encoding_config


def _initialize_encoding_moe_model(
    args: AttrDict,
    base_input_dim: int,
    dataset: Dataset,
    encoding_moe_encoded_datasets: Optional[Dict[str, Dict[str, List]]],
) -> Tuple[EncodingMoE, bool]:
    """
    Initialize EncodingMoE model with proper configuration.

    Args:
        args: Experiment arguments
        base_input_dim: Dimension of base input features
        dataset: Base dataset (for validation)
        encoding_moe_encoded_datasets: Dictionary of encoded datasets

    Returns:
        Tuple of (model, is_encoding_moe_flag)
    """
    # Create encoding configurations
    encoding_configs = []
    for encoding_name in args.encoding_moe_encodings:
        # Determine base input dim from dataset (for validation)
        base_dim = dataset[0].x.shape[1] if len(dataset) > 0 else base_input_dim

        # Try to get encoded dim from encoded_datasets if available
        encoded_dim = None
        if (
            encoding_moe_encoded_datasets
            and encoding_name in encoding_moe_encoded_datasets
        ):
            # Find dataset name (should match current dataset)
            for dataset_name, encoded_dataset in encoding_moe_encoded_datasets[
                encoding_name
            ].items():
                if len(encoded_dataset) > 0:
                    encoded_dim = encoded_dataset[0].x.shape[1]
                    break

        config = create_encoding_config(
            encoding_name,
            base_input_dim=base_dim,
            encoded_input_dim=encoded_dim,
        )
        encoding_configs.append(config)

    # Create GNN model that will be used by each encoding "expert"
    # The input_dim needs to handle both cases:
    # 1. If encoding replaces_base=True: use actual encoded graph dim (may be > encoding_dim)
    # 2. If encoding replaces_base=False: use base_input_dim + encoding_dim
    # We need the maximum input dimension across all encodings
    max_input_dims = []
    for config in encoding_configs:
        if config.get("replaces_base", False):
            # Encoding replaces base - use only encoding_dim (extracted from file)
            max_input_dims.append(config["encoding_dim"])
        else:
            # Encoding appends to base - input_dim = base + encoding_dim
            max_input_dims.append(base_input_dim + config["encoding_dim"])

    expert_input_dim = max(max_input_dims) if max_input_dims else base_input_dim
    max_encoding_dim = max(config["encoding_dim"] for config in encoding_configs)
    any_replaces_base = any(
        config.get("replaces_base", False) for config in encoding_configs
    )

    # Debug logging
    print(f"🔧 EncodingMoE GNN initialization:")
    print(f"   base_input_dim: {base_input_dim}")
    print(f"   max_encoding_dim: {max_encoding_dim}")
    print(f"   any_replaces_base: {any_replaces_base}")
    print(f"   expert_input_dim: {expert_input_dim}")
    print(
        f"   encoding_configs: {[(c['encoding_name'], c['encoding_dim']) for c in encoding_configs]}"
    )

    # Create args for GNN expert
    # Create a new AttrDict with updated input_dim to avoid deepcopy issues
    expert_args_dict = dict(args)  # Convert to regular dict
    expert_args_dict["input_dim"] = expert_input_dim
    expert_args = AttrDict(expert_args_dict)
    print(f"   expert_args.input_dim (set to): {expert_args.input_dim}")

    # Double-check the value is correct
    if expert_args.input_dim != expert_input_dim:
        raise ValueError(
            f"Failed to set expert_args.input_dim: got {expert_args.input_dim}, expected {expert_input_dim}"
        )

    # Initialize GNN expert model
    print(f"   Initializing GNN with input_dim: {expert_args.input_dim}")
    if expert_args.layer_type == "GPS":
        gnn_expert = GPS(expert_args).to(args.device)
    elif expert_args.layer_type == "Orthogonal":
        gnn_expert = OrthogonalGCN(expert_args).to(args.device)
    elif expert_args.layer_type == "Unitary":
        gnn_expert = UnitaryGCN(expert_args).to(args.device)
    else:
        gnn_expert = GNN(expert_args).to(args.device)

    # Verify the first layer has correct input dimension
    if hasattr(gnn_expert, "layers") and len(gnn_expert.layers) > 0:
        first_layer = gnn_expert.layers[0]
        if hasattr(first_layer, "lin"):
            first_layer_input_dim = first_layer.lin.weight.shape[1]
            print(
                f"   ✅ First layer input_dim: {first_layer_input_dim} (expected: {expert_input_dim})"
            )
            if first_layer_input_dim != expert_input_dim:
                print(
                    f"   ❌ Mismatch! Model initialized with wrong input_dim! Expected {expert_input_dim} but got {first_layer_input_dim}"
                )
                # This is a critical error - model won't work
                raise ValueError(
                    f"GNN model initialized with wrong input_dim: {first_layer_input_dim} "
                    f"but needs {expert_input_dim}. expert_args.input_dim was {expert_args.input_dim}"
                )

    # Initialize EncodingMoE
    encoding_moe_args = copy.deepcopy(args)
    encoding_moe_args.base_input_dim = base_input_dim
    encoding_moe_args.router_type = getattr(args, "encoding_moe_router_type", "MLP")
    model = EncodingMoE(encoding_moe_args, encoding_configs, gnn_expert).to(args.device)

    # Print initialization summary
    print(f"✅ Initialized EncodingMoE with {len(encoding_configs)} encodings")
    for config in encoding_configs:
        print(
            f"   - {config['encoding_name']}: dim={config['encoding_dim']}, replaces_base={config.get('replaces_base', False)}"
        )

    return model, True


def _create_encoding_moe_loaders(
    args: AttrDict,
    train_dataset: Dataset,
    train_indices: Optional[List[int]],
    encoding_moe_encoded_datasets: Optional[Dict[str, Dict[str, List]]],
    dataset_name: Optional[str],
    shuffle: bool = True,
) -> Dict[str, DataLoader]:
    """
    Create DataLoaders for encoded datasets for EncodingMoE training.

    Args:
        args: Experiment arguments
        train_dataset: Training dataset (for size reference)
        train_indices: Training indices (if available)
        encoding_moe_encoded_datasets: Dictionary of encoded datasets
        dataset_name: Name of the dataset
        shuffle: Whether to shuffle the loaders

    Returns:
        Dictionary mapping encoding names to DataLoaders
    """
    from torch_geometric.loader import DataLoader as PyGDataLoader

    encoded_loaders = {}

    if (
        encoding_moe_encoded_datasets
        and dataset_name
        and dataset_name
        in (train_dataset.dataset if hasattr(train_dataset, "dataset") else {})
    ):
        if dataset_name:
            for encoding_name in args.encoding_moe_encodings:
                if (
                    encoding_name in encoding_moe_encoded_datasets
                    and dataset_name in encoding_moe_encoded_datasets[encoding_name]
                ):
                    encoded_dataset = encoding_moe_encoded_datasets[encoding_name][
                        dataset_name
                    ]
                    # Split encoded dataset same way as base (use same indices)
                    encoded_train = (
                        [
                            encoded_dataset[i]
                            for i in range(len(encoded_dataset))
                            if i in train_indices
                        ]
                        if train_indices is not None
                        else encoded_dataset[: len(train_dataset)]
                    )
                    encoded_loaders[encoding_name] = PyGDataLoader(
                        encoded_train,
                        batch_size=args.batch_size,
                        shuffle=shuffle,
                    )

    return encoded_loaders


def _get_encoding_moe_encoded_graphs_for_batch(
    encoding_name: str,
    encoded_iterators: Dict[str, Any],
    encoded_loaders: Dict[str, DataLoader],
    device: torch.device,
) -> Optional[torch.Tensor]:
    """
    Get encoded graph batch for a specific encoding.

    Args:
        encoding_name: Name of the encoding
        encoded_iterators: Dictionary of iterators for encoded loaders
        encoded_loaders: Dictionary of encoded loaders
        device: Device to move data to

    Returns:
        Encoded graph batch or None if not available
    """
    if encoding_name in encoded_iterators:
        try:
            encoded_batch = next(encoded_iterators[encoding_name])
            encoded_batch = encoded_batch.to(device)
            return encoded_batch
        except StopIteration:
            # Reset iterator if exhausted (shouldn't happen with aligned datasets)
            encoded_iterators[encoding_name] = iter(encoded_loaders[encoding_name])
            encoded_batch = next(encoded_iterators[encoding_name])
            encoded_batch = encoded_batch.to(device)
            return encoded_batch
    return None


def _create_encoding_moe_loaders_for_split(
    args: AttrDict,
    loader: DataLoader,
    train_dataset: Dataset,
    validation_dataset: Dataset,
    categories: Optional[List[List[int]]],
    encoding_moe_encoded_datasets: Optional[Dict[str, Dict[str, List]]],
    dataset_name: Optional[str],
) -> Dict[str, DataLoader]:
    """
    Create DataLoaders for encoded datasets for a specific split (train/val/test).

    Args:
        args: Experiment arguments
        loader: The loader for the split (to determine which split)
        train_dataset: Training dataset
        validation_dataset: Validation dataset
        categories: List of indices for [train, val, test] splits
        encoding_moe_encoded_datasets: Dictionary of encoded datasets
        dataset_name: Name of the dataset

    Returns:
        Dictionary mapping encoding names to DataLoaders
    """
    from torch_geometric.loader import DataLoader as PyGDataLoader

    # Determine which split (train/val/test) based on dataset size
    loader_size = len(loader.dataset)
    train_size = len(train_dataset)
    val_size = len(validation_dataset)

    # Determine which indices to use
    if loader_size == train_size:
        split_indices = (
            categories[0] if categories is not None else list(range(train_size))
        )
    elif loader_size == val_size:
        split_indices = (
            categories[1]
            if categories is not None
            else list(range(train_size, train_size + val_size))
        )
    else:
        split_indices = (
            categories[2]
            if categories is not None
            else list(
                range(
                    train_size + val_size,
                    (
                        len(train_dataset.dataset)
                        if hasattr(train_dataset, "dataset")
                        else train_size + val_size
                    ),
                )
            )
        )

    # Create encoded loaders for this split
    encoded_loaders = {}

    if encoding_moe_encoded_datasets and dataset_name:
        for encoding_name in args.encoding_moe_encodings:
            if (
                encoding_name in encoding_moe_encoded_datasets
                and dataset_name in encoding_moe_encoded_datasets[encoding_name]
            ):
                encoded_dataset = encoding_moe_encoded_datasets[encoding_name][
                    dataset_name
                ]
                encoded_split = [
                    encoded_dataset[i]
                    for i in split_indices
                    if i < len(encoded_dataset)
                ]
                encoded_loaders[encoding_name] = PyGDataLoader(
                    encoded_split,
                    batch_size=args.batch_size,
                    shuffle=False,  # Don't shuffle for eval/test
                )

    return encoded_loaders
