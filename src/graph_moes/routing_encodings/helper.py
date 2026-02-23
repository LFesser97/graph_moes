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
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from torch.utils.data import Dataset
from torch_geometric.data import Batch, Data
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
    encoding_moe_encoded_datasets: Optional[Dict[str, List[Data]]],
) -> Tuple[EncodingMoE, bool]:
    """Initialize EncodingMoE model with proper configuration.

    Args:
        args: Experiment arguments.
        base_input_dim: Dimension of base input features.
        dataset: Base dataset (for validation).
        encoding_moe_encoded_datasets:
            Mapping from encoding name to the list of encoded ``Data`` objects
            for the current dataset (already extracted per-dataset upstream).

    Returns:
        Tuple of (model, is_encoding_moe_flag).
    """
    # Create encoding configurations
    encoding_configs: List[Dict[str, Any]] = []
    for encoding_name in args.encoding_moe_encodings:
        # Determine base input dim from dataset (for validation)
        base_dim: int = dataset[0].x.shape[1] if len(dataset) > 0 else base_input_dim

        # Try to get encoded dim from encoded_datasets if available
        encoded_dim: Optional[int] = None
        if (
            encoding_moe_encoded_datasets
            and encoding_name in encoding_moe_encoded_datasets
        ):
            encoded_dataset = encoding_moe_encoded_datasets[encoding_name]
            if len(encoded_dataset) > 0:
                encoded_dim = encoded_dataset[0].x.shape[1]

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
    encoding_moe_encoded_datasets: Optional[Dict[str, List[Data]]],
    dataset_name: Optional[str],
    shuffle: bool = True,
) -> Dict[str, DataLoader]:
    """Create DataLoaders for encoded datasets for EncodingMoE training.

    ``encoding_moe_encoded_datasets`` maps encoding names directly to the
    list of encoded ``Data`` objects for the current dataset (already
    extracted per-dataset by ``_extract_encoding_moe_datasets_for_key``).

    Args:
        args: Experiment arguments.
        train_dataset: Training dataset (for size reference).
        train_indices: Training indices (if available).
        encoding_moe_encoded_datasets:
            Mapping ``{encoding_name: List[Data]}`` for the current dataset.
        dataset_name: Name of the dataset (used only for logging).
        shuffle: Whether to shuffle the loaders.

    Returns:
        Dictionary mapping encoding names to DataLoaders.
    """
    _ = dataset_name  # kept for API compatibility; extraction is done upstream

    encoded_loaders: Dict[str, DataLoader] = {}

    if not encoding_moe_encoded_datasets:
        return encoded_loaders

    # Convert train_indices to a set for O(1) membership checks
    train_indices_set: Optional[frozenset[int]] = (
        frozenset(train_indices) if train_indices is not None else None
    )

    for encoding_name in args.encoding_moe_encodings:
        if encoding_name not in encoding_moe_encoded_datasets:
            continue

        encoded_dataset = encoding_moe_encoded_datasets[encoding_name]

        # Split encoded dataset the same way as the base dataset
        if train_indices_set is not None:
            encoded_train = [
                encoded_dataset[i]
                for i in range(len(encoded_dataset))
                if i in train_indices_set
            ]
        else:
            encoded_train = list(encoded_dataset[: len(train_dataset)])

        if len(encoded_train) == 0:
            print(f"  ⚠️  No training samples for encoding {encoding_name}, skipping")
            continue

        encoded_loaders[encoding_name] = DataLoader(
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
) -> Optional[Union[Data, Batch]]:
    """Get encoded graph batch for a specific encoding.

    Args:
        encoding_name: Name of the encoding.
        encoded_iterators: Dictionary of iterators for encoded loaders.
        encoded_loaders: Dictionary of encoded loaders.
        device: Device to move data to.

    Returns:
        Encoded graph ``Batch`` (or single ``Data``) moved to *device*,
        or ``None`` if the encoding is not present in the iterators.
    """
    if encoding_name not in encoded_iterators:
        return None

    try:
        encoded_batch = next(encoded_iterators[encoding_name])
    except StopIteration:
        # Reset iterator if exhausted (shouldn't happen with aligned datasets)
        encoded_iterators[encoding_name] = iter(encoded_loaders[encoding_name])
        encoded_batch = next(encoded_iterators[encoding_name])

    encoded_batch = encoded_batch.to(device)
    return encoded_batch


def _create_encoding_moe_loaders_for_split(
    args: AttrDict,
    loader: DataLoader,
    train_dataset: Dataset,
    validation_dataset: Dataset,
    categories: Optional[List[List[int]]],
    encoding_moe_encoded_datasets: Optional[Dict[str, List[Data]]],
    dataset_name: Optional[str],
) -> Dict[str, DataLoader]:
    """Create DataLoaders for encoded datasets for a specific split (train/val/test).

    The correct split is inferred by comparing *loader*'s dataset size against
    the training and validation dataset sizes.

    Args:
        args: Experiment arguments.
        loader: The loader for the split (used to infer which split via size).
        train_dataset: Training dataset.
        validation_dataset: Validation dataset.
        categories: List of index lists ``[train_idx, val_idx, test_idx]``.
        encoding_moe_encoded_datasets:
            Mapping ``{encoding_name: List[Data]}`` for the current dataset.
        dataset_name: Name of the dataset (used only for logging).

    Returns:
        Dictionary mapping encoding names to DataLoaders.
    """
    _ = dataset_name  # kept for API compatibility; extraction is done upstream

    if not encoding_moe_encoded_datasets:
        return {}

    # Determine which split (train/val/test) based on dataset size
    loader_size: int = len(loader.dataset)
    train_size: int = len(train_dataset)
    val_size: int = len(validation_dataset)

    # Determine which indices to use
    split_indices: List[int]
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
        # Test split — fall back to computing the range from the full dataset
        if categories is not None:
            split_indices = categories[2]
        else:
            full_size: int = (
                len(train_dataset.dataset)
                if hasattr(train_dataset, "dataset")
                else train_size + val_size + loader_size
            )
            split_indices = list(range(train_size + val_size, full_size))

    # Create encoded loaders for this split
    encoded_loaders: Dict[str, DataLoader] = {}

    for encoding_name in args.encoding_moe_encodings:
        if encoding_name not in encoding_moe_encoded_datasets:
            continue

        encoded_dataset = encoding_moe_encoded_datasets[encoding_name]
        encoded_split = [
            encoded_dataset[i] for i in split_indices if i < len(encoded_dataset)
        ]

        if len(encoded_split) == 0:
            continue

        encoded_loaders[encoding_name] = DataLoader(
            encoded_split,
            batch_size=args.batch_size,
            shuffle=False,  # Don't shuffle for eval/test
        )

    return encoded_loaders
