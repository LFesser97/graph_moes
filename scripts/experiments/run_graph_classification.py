"""Script to run graph classification experiments.

This script orchestrates comprehensive graph classification experiments across diverse datasets
including molecular graphs (MUTAG, ENZYMES, PROTEINS), social networks (IMDB, COLLAB, REDDIT),
and computer vision tasks (MNIST, CIFAR10 superpixels). It supports various GNN architectures.

The script handles dataset preprocessing, optional structural encodings (Laplacian eigenvectors,
random walk features, curvature profiles), multi-trial training with statistical analysis,
and comprehensive result logging for benchmarking different graph neural network approaches.

Desired behavior:
Skips the experiment if the encoding file doesn't exist: Instead of falling back to the
original dataset, it exits gracefully with sys.exit(0) when the precomputed encoding file is missing or fails to load.
"""

import ast
import os
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch_geometric.transforms as T

try:
    from attrdict3 import AttrDict  # Python 3.10+ compatible
except ImportError:
    from attrdict import AttrDict  # Fallback for older Python

from ogb.graphproppred import PygGraphPropPredDataset
from torch_geometric.data import Data
from torch_geometric.datasets import GNNBenchmarkDataset, TUDataset
from tqdm import tqdm

import wandb

# GraphBench loading disabled - comment out to re-enable
# from graph_moes.download.load_graphbench import load_graphbench_dataset
from graph_moes.encodings.custom_encodings import LocalCurvatureProfile
from graph_moes.experiments.graph_classification import Experiment

try:
    from graph_moes.experiments.track_avg_accuracy import (
        load_and_plot_average_per_graph,
    )
except ImportError:
    # Fallback: try adding src to path if package not properly installed
    import sys
    from pathlib import Path

    src_path = Path(__file__).parent.parent.parent / "src"
    if src_path.exists() and str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))
    from graph_moes.experiments.track_avg_accuracy import (
        load_and_plot_average_per_graph,
    )

from hyperparams import get_args_from_input


# ============================================================================
# EncodingMoE Helper Functions
# ============================================================================


def _parse_encoding_moe_encodings(encoding_list: list) -> list:
    """
    Parse encoding list from JSON string if needed.

    Args:
        encoding_list: List of encodings (may contain a single JSON string)

    Returns:
        Parsed list of encoding names
    """
    # Check if it's a single JSON string that needs parsing
    if len(encoding_list) == 1 and isinstance(encoding_list[0], str):
        encoding_str = encoding_list[0].strip()
        # Check if it looks like JSON (starts with [ and ends with ])
        if encoding_str.startswith("[") and encoding_str.endswith("]"):
            try:
                # Replace JSON null with Python None in string before parsing
                encoding_str_python = encoding_str.replace("null", "None")
                # Parse as Python literal (handles None correctly)
                parsed_list = ast.literal_eval(encoding_str_python)
                # Convert None back to string "None" for consistency
                encoding_list = [e if e is not None else "None" for e in parsed_list]
                print(
                    f"   📝 Parsed JSON encoding list: {encoding_str} -> {encoding_list}"
                )
            except (ValueError, SyntaxError) as e:
                print(f"   ⚠️  Failed to parse JSON encoding list: {e}, using as-is")

    return encoding_list


def _load_encoding_moe_datasets(
    args: AttrDict,
    datasets: dict,
    data_directory: str,
) -> tuple[dict, dict]:
    """
    Load base datasets and encoded datasets for EncodingMoE.

    Args:
        args: Experiment arguments
        datasets: Original datasets dictionary
        data_directory: Path to data directory

    Returns:
        Tuple of (encoding_moe_base_datasets, encoding_moe_encoded_datasets)
    """
    encoding_list = args.encoding_moe_encodings
    encoding_list = _parse_encoding_moe_encodings(encoding_list)
    args.encoding_moe_encodings = encoding_list

    print(
        f"\n🎯 EncodingMoE mode: Loading base dataset + {len(encoding_list)} encodings"
    )
    print(f"   Encodings: {encoding_list}")

    # Keep original datasets as base (no encoding)
    encoding_moe_base_datasets = datasets.copy()

    # Load each encoding separately
    encoding_moe_encoded_datasets = {}
    for encoding_name in encoding_list:
        # Handle "None" encoding (no encoding file to load, just use base dataset)
        if encoding_name == "None" or encoding_name is None:
            print(f"\n  📦 Encoding: None (using base dataset)")
            # Store empty dict to indicate no encoding file, will use base dataset
            encoding_moe_encoded_datasets["None"] = {}
            continue

        print(f"\n  📦 Loading encoding: {encoding_name}")

        # Determine encoding directory and file pattern (same logic as dataset_encoding)
        if encoding_name.startswith("hg_"):
            encoding_suffix = encoding_name[3:]
            encoded_data_dir = (
                Path(data_directory).parent / "graph_datasets_with_hg_encodings"
            )
            file_pattern = "{dataset_name}_hg_{encoding_suffix}.pt"
        elif encoding_name.startswith("g_"):
            encoding_suffix = encoding_name[2:]
            encoded_data_dir = (
                Path(data_directory).parent / "graph_datasets_with_g_encodings"
            )
            file_pattern = "{dataset_name}_g_{encoding_suffix}.pt"
        else:
            raise ValueError(
                f"Unknown encoding format: {encoding_name}. Must start with 'hg_' or 'g_'"
            )

        if not encoded_data_dir.exists():
            raise FileNotFoundError(
                f"Encoded datasets directory not found: {encoded_data_dir}"
            )

        # Load encoded datasets for this encoding
        encoded_datasets_for_encoding = {}
        datasets_to_check = [args.dataset] if args.dataset else list(datasets.keys())

        for dataset_name in datasets_to_check:
            filename = file_pattern.format(
                dataset_name=dataset_name, encoding_suffix=encoding_suffix
            )
            encoded_file_path = encoded_data_dir / filename

            if encoded_file_path.exists():
                try:
                    print(f"    ⏳ Loading {dataset_name} with {encoding_name}...")
                    encoded_dataset = torch.load(encoded_file_path, weights_only=False)
                    encoded_datasets_for_encoding[dataset_name] = encoded_dataset
                    print(
                        f"    ✅ {dataset_name} loaded: {len(encoded_dataset)} graphs"
                    )
                except Exception as e:
                    print(
                        f"    ❌ Failed to load {dataset_name} with {encoding_name}: {e}"
                    )
            else:
                print(f"    ⚠️  Encoded dataset file not found: {encoded_file_path}")

        encoding_moe_encoded_datasets[encoding_name] = encoded_datasets_for_encoding
        print(
            f"  ✅ Loaded {len(encoded_datasets_for_encoding)} datasets with {encoding_name}"
        )

    print(f"\n✅ EncodingMoE datasets loaded successfully!")
    print(f"   Base datasets: {len(encoding_moe_base_datasets)}")
    print(
        f"   Encoded datasets: {len(encoding_moe_encoded_datasets)} encodings × {len(encoding_moe_encoded_datasets.get(encoding_list[0], {}))} datasets"
    )

    return encoding_moe_base_datasets, encoding_moe_encoded_datasets


def _is_encoding_moe_enabled(args: AttrDict) -> bool:
    """
    Check if EncodingMoE is enabled.

    Args:
        args: Experiment arguments

    Returns:
        True if EncodingMoE is enabled, False otherwise
    """
    return (
        hasattr(args, "encoding_moe_encodings")
        and args.encoding_moe_encodings is not None
        and len(args.encoding_moe_encodings) > 0
    )


def _get_encoding_moe_wandb_suffix(args: AttrDict) -> str:
    """
    Generate WandB experiment ID suffix for EncodingMoE.

    Args:
        args: Experiment arguments

    Returns:
        Suffix string (empty if EncodingMoE not enabled)
    """
    if not _is_encoding_moe_enabled(args):
        return ""

    # Create compact encoding name (e.g., "g_ldp+g_orc")
    encoding_names_str = "+".join(args.encoding_moe_encodings)
    router_type = getattr(args, "encoding_moe_router_type", "MLP")
    return f"_EncMoE_{encoding_names_str}_r{router_type}"


def _extract_encoding_moe_datasets_for_key(
    args: AttrDict,
    key: str,
    encoding_moe_encoded_datasets: dict,
    encoding_moe_base_datasets: dict,
) -> tuple[dict, any]:
    """
    Extract encoded datasets for a specific dataset key and determine which dataset to use.

    Args:
        args: Experiment arguments
        key: Dataset key name
        encoding_moe_encoded_datasets: Dictionary of encoded datasets
        encoding_moe_base_datasets: Dictionary of base datasets

    Returns:
        Tuple of (encoding_moe_encoded_datasets_for_key, dataset_to_use)
    """
    encoding_moe_encoded_datasets_for_key = None
    if encoding_moe_encoded_datasets is not None and key in encoding_moe_base_datasets:
        # Extract encoded datasets for this specific dataset key
        encoding_moe_encoded_datasets_for_key = {}
        for encoding_name in args.encoding_moe_encodings:
            if (
                encoding_name in encoding_moe_encoded_datasets
                and key in encoding_moe_encoded_datasets[encoding_name]
            ):
                encoding_moe_encoded_datasets_for_key[encoding_name] = (
                    encoding_moe_encoded_datasets[encoding_name][key]
                )

    # Use base dataset if EncodingMoE is enabled, otherwise use regular dataset
    dataset_to_use = (
        encoding_moe_base_datasets[key]
        if (
            encoding_moe_base_datasets is not None and key in encoding_moe_base_datasets
        )
        else None
    )

    return encoding_moe_encoded_datasets_for_key, dataset_to_use


def _convert_lrgb(dataset: torch.Tensor) -> Data:
    """Convert LRGB dataset tuple format to PyTorch Geometric Data object.

    Args:
        dataset: Tuple containing (x, edge_attr, edge_index, y) tensors

    Returns:
        PyTorch Geometric Data object with node features, edges, and labels
    """
    x = dataset[0]
    edge_attr = dataset[1]
    edge_index = dataset[2]
    y = dataset[3]

    return Data(x=x, edge_index=edge_index, y=y, edge_attr=edge_attr)


# Simple approach - always use local directory
data_directory = (
    "/n/holylabs/LABS/mweber_lab/Everyone/rpellegrin/graph_moes/graph_datasets"
)
print("📁 Using project data directory")
os.makedirs(data_directory, exist_ok=True)

# Create data subdirectory for encodings
os.makedirs("data", exist_ok=True)
print(f"📁 Raw datasets: {data_directory}")
print("💾 Encoded datasets: ./data/")

# New datasets
print("📊 Loading NEW benchmark datasets...")
print("  ⏳ Loading MNIST superpixel graphs...")
mnist = list(GNNBenchmarkDataset(root=data_directory, name="MNIST"))
print(f"  ✅ MNIST loaded: {len(mnist)} graphs")

print("  ⏳ Loading CIFAR10 superpixel graphs...")
cifar = list(GNNBenchmarkDataset(root=data_directory, name="CIFAR10"))
print(f"  ✅ CIFAR10 loaded: {len(cifar)} graphs")

print("  ⏳ Loading PATTERN synthetic graphs...")
pattern = list(GNNBenchmarkDataset(root=data_directory, name="PATTERN"))
print(f"  ✅ PATTERN loaded: {len(pattern)} graphs")

print("📊 Loading existing TU datasets...")
# import TU datasets
print("  ⏳ Loading MUTAG...")
mutag = list(TUDataset(root=data_directory, name="MUTAG"))
print(f"  ✅ MUTAG loaded: {len(mutag)} graphs")

print("  ⏳ Loading ENZYMES...")
enzymes = list(TUDataset(root=data_directory, name="ENZYMES"))
print(f"  ✅ ENZYMES loaded: {len(enzymes)} graphs")

print("  ⏳ Loading PROTEINS...")
proteins = list(TUDataset(root=data_directory, name="PROTEINS"))
print(f"  ✅ PROTEINS loaded: {len(proteins)} graphs")

print("  ⏳ Loading IMDB-BINARY...")
imdb = list(TUDataset(root=data_directory, name="IMDB-BINARY"))
print(f"  ✅ IMDB-BINARY loaded: {len(imdb)} graphs")

print("  ⏳ Loading COLLAB...")
collab = list(TUDataset(root=data_directory, name="COLLAB"))
print(f"  ✅ COLLAB loaded: {len(collab)} graphs")

print("  ⏳ Loading REDDIT-BINARY...")
reddit = list(TUDataset(root=data_directory, name="REDDIT-BINARY"))
print(f"  ✅ REDDIT-BINARY loaded: {len(reddit)} graphs")

print("and yet more...")

# GraphBench datasets (graph classification tasks)
# ENABLED for additional data sweep - will attempt downloads
graphbench_datasets = {}

# GraphBench dataset names that are relevant for graph classification
# Based on GraphBench documentation: https://github.com/graphbench/package
# TEMPORARILY DISABLED - having loading issues
graphbench_classification_datasets = [
    # "socialnetwork",  # Social media datasets - failing to load
    # "co",  # Combinatorial optimization - file structure error
    # "sat",  # SAT solving - works but OOM issues
    # "electronic_circuits",  # Electronic circuits
    # "chipdesign",  # Chip design
    # Note: weather is for regression tasks, not included here
]

# Skip GraphBench loading - list is empty so this block won't execute
# GraphBench datasets are disabled to avoid download attempts
if len(graphbench_classification_datasets) > 0:
    print("\n📊 Loading GraphBench datasets...")
    try:
        from graph_moes.download.load_graphbench import load_graphbench_dataset
    except ImportError:
        print("  ⚠️  GraphBench import disabled, skipping GraphBench datasets")
        graphbench_classification_datasets = []

    if len(graphbench_classification_datasets) > 0:
        for dataset_name in graphbench_classification_datasets:
            try:
                print(f"  ⏳ Loading GraphBench: {dataset_name}...")
                graphbench_data = load_graphbench_dataset(
                    dataset_name=dataset_name, root=data_directory
                )
                graphbench_datasets[f"graphbench_{dataset_name}"] = graphbench_data
                print(
                    f"  ✅ GraphBench {dataset_name} loaded: {len(graphbench_data)} graphs"
                )
            except (
                ImportError,
                ValueError,
                RuntimeError,
                OSError,
                EOFError,
                Exception,
            ) as e:
                error_msg = str(e)
                error_type = type(e).__name__
                # Check if it's a rate limit error
                if (
                    "429" in error_msg
                    or "Too Many Requests" in error_msg
                    or "rate limit" in error_msg.lower()
                ):
                    print(
                        f"  ⚠️  Failed to load GraphBench {dataset_name}: Rate limited by server (HTTP 429). "
                        f"Skipping this dataset. Run download script separately to download datasets."
                    )
                # Check if it's a download/extraction error (corrupted file)
                elif (
                    "zlib.error" in error_msg
                    or "decompressing" in error_msg
                    or "invalid stored block" in error_msg
                    or "Error -3" in error_msg
                    or error_type == "EOFError"
                    or "Compressed file ended" in error_msg
                    or "end-of-stream marker" in error_msg
                    or "tarfile" in error_msg.lower()
                ):
                    print(
                        f"  ⚠️  Failed to load GraphBench {dataset_name}: Download/extraction error (file may be corrupted or incomplete). "
                        f"Skipping this dataset. To fix: delete {data_directory}/{dataset_name} and re-download."
                    )
                # Check if it's a file structure error
                elif (
                    "NotADirectoryError" in error_type or "Not a directory" in error_msg
                ):
                    print(
                        f"  ⚠️  Failed to load GraphBench {dataset_name}: File structure error. "
                        f"Skipping this dataset. Try deleting {data_directory}/{dataset_name} and re-downloading."
                    )
                else:
                    print(
                        f"  ⚠️  Failed to load GraphBench {dataset_name}: {error_type}: {e} (may not be installed or available)"
                    )
                # Continue with other datasets
                continue
else:
    print("  📊 Loading GraphBench datasets...")

print("  ⏭️  LRGB datasets disabled (commented out)")

# LRGB datasets (DISABLED: Commented out to avoid loading issues)
# print("\n📊 Loading LRGB datasets...")
# try:
#     print("  ⏳ Loading Cluster...")
#     cluster = list(LRGBDataset(root=data_directory, name="Cluster"))
#     print(f"  ✅ Cluster loaded: {len(cluster)} graphs")
# except (ImportError, ValueError, RuntimeError, OSError) as e:
#     print(f"  ⚠️  Failed to load Cluster: {e}")
#     cluster = []

# try:
#     print("  ⏳ Loading PascalVOC-SP...")
#     pascalvoc = list(LRGBDataset(root=data_directory, name="pascalvoc-sp"))
#     print(f"  ✅ PascalVOC-SP loaded: {len(pascalvoc)} graphs")
# except (ImportError, ValueError, RuntimeError, OSError) as e:
#     print(f"  ⚠️  Failed to load PascalVOC-SP: {e}")
#     pascalvoc = []

# try:
#     print("  ⏳ Loading COCO-SP...")
#     coco = list(LRGBDataset(root=data_directory, name="coco-sp"))
#     print(f"  ✅ COCO-SP loaded: {len(coco)} graphs")
# except (ImportError, ValueError, RuntimeError, OSError) as e:
#     print(f"  ⚠️  Failed to load COCO-SP: {e}")
#     coco = []

# Set LRGB datasets to empty (disabled)
cluster = []
pascalvoc = []
coco = []

# Peptides-func dataset
print("\n📊 Loading Peptides-func...")
try:
    peptides_func_path = os.path.join(data_directory, "peptidesfunc")
    if os.path.exists(peptides_func_path):
        peptides_train = torch.load(os.path.join(peptides_func_path, "train.pt"))
        peptides_val = torch.load(os.path.join(peptides_func_path, "val.pt"))
        peptides_test = torch.load(os.path.join(peptides_func_path, "test.pt"))
        peptides_func = (
            [_convert_lrgb(peptides_train[i]) for i in range(len(peptides_train))]
            + [_convert_lrgb(peptides_val[i]) for i in range(len(peptides_val))]
            + [_convert_lrgb(peptides_test[i]) for i in range(len(peptides_test))]
        )
        print(f"  ✅ Peptides-func loaded: {len(peptides_func)} graphs")
    else:
        print(f"  ⚠️  Peptides-func directory not found at {peptides_func_path}")
        peptides_func = []
except Exception as e:
    print(f"  ⚠️  Failed to load Peptides-func: {e}")
    peptides_func = []

# OGB datasets
print("\n📊 Loading OGB datasets...")
try:
    print("  ⏳ Loading ogbg-molhiv...")
    molhiv_dataset = PygGraphPropPredDataset(name="ogbg-molhiv", root=data_directory)
    molhiv = [molhiv_dataset[i] for i in range(len(molhiv_dataset))]
    print(f"  ✅ ogbg-molhiv loaded: {len(molhiv)} graphs")
except (ImportError, ValueError, RuntimeError, OSError) as e:
    print(f"  ⚠️  Failed to load ogbg-molhiv: {e}")
    molhiv = []

try:
    print("  ⏳ Loading ogbg-molpcba...")
    molpcba_dataset = PygGraphPropPredDataset(name="ogbg-molpcba", root=data_directory)
    molpcba = [molpcba_dataset[i] for i in range(len(molpcba_dataset))]
    print(f"  ✅ ogbg-molpcba loaded: {len(molpcba)} graphs")
except (ImportError, ValueError, RuntimeError, OSError) as e:
    print(f"  ⚠️  Failed to load ogbg-molpcba: {e}")
    molpcba = []

try:
    print("  ⏳ Loading ogbg-ppa...")
    ppa_dataset = PygGraphPropPredDataset(name="ogbg-ppa", root=data_directory)
    ppa = [ppa_dataset[i] for i in range(len(ppa_dataset))]
    print(f"  ✅ ogbg-ppa loaded: {len(ppa)} graphs")
except (ImportError, ValueError, RuntimeError, OSError, EOFError) as e:
    print(f"  ⚠️  Failed to load ogbg-ppa: {e}")
    ppa = []

try:
    print("  ⏳ Loading ogbg-code2...")
    code2_dataset = PygGraphPropPredDataset(name="ogbg-code2", root=data_directory)
    code2 = [code2_dataset[i] for i in range(len(code2_dataset))]
    print(f"  ✅ ogbg-code2 loaded: {len(code2)} graphs")
except (ImportError, ValueError, RuntimeError, OSError) as e:
    print(f"  ⚠️  Failed to load ogbg-code2: {e}")
    code2 = []

# MalNet-Tiny dataset
print("\n📊 Loading MalNet-Tiny...")
try:
    # MalNet-Tiny is available in PyG as TUDataset
    malnet = list(TUDataset(root=data_directory, name="MalNetTiny"))
    print(f"  ✅ MalNet-Tiny loaded: {len(malnet)} graphs")
except (ImportError, ValueError, RuntimeError, OSError) as e:
    print(f"  ⚠️  Failed to load MalNet-Tiny: {e}")
    malnet = []

print("🎉 All datasets loaded successfully!")


datasets = {
    "mutag": mutag,
    "enzymes": enzymes,
    "proteins": proteins,
    "imdb": imdb,
    "collab": collab,
    "reddit": reddit,
    # New datasets:
    "mnist": mnist,
    "cifar": cifar,
    "pattern": pattern,
    # LRGB datasets:
    "cluster": cluster if cluster else None,
    "pascalvoc": pascalvoc if pascalvoc else None,
    "coco": coco if coco else None,
    "peptides_func": peptides_func if peptides_func else None,
    # OGB datasets:
    "molhiv": molhiv if molhiv else None,
    "molpcba": molpcba if molpcba else None,
    "ppa": ppa if ppa else None,
    "code2": code2 if code2 else None,
    # Other datasets:
    "malnet": malnet if malnet else None,
    # GraphBench datasets:
    **graphbench_datasets,
}

# Remove None or empty datasets (failed to load)
datasets = {k: v for k, v in datasets.items() if v is not None and len(v) > 0}
# datasets = {"collab": collab, "imdb": imdb, "proteins": proteins, "reddit": reddit}


def log_to_file(
    message: str, filename: str = "results/graph_classification.txt"
) -> None:
    """Log a message to both console and file.

    Args:
        message: The message to log
        filename: Path to the log file (default: "results/graph_classification.txt")
    """
    print(message)
    file = open(filename, "a")
    file.write(message)
    file.close()


def get_encoding_category(encoding_dir: str | None) -> str:
    """Determine encoding category based on the directory used for loading encodings.

    Args:
        encoding_dir: Path to the directory where encodings were loaded from, or None if no encoding.

    Returns:
        Category string: "hypergraph", "graph", or "None"
    """
    if encoding_dir is None:
        return "None"
    elif "graph_datasets_with_hg_encodings" in encoding_dir:
        return "hypergraph"
    elif "graph_datasets_with_g_encodings" in encoding_dir:
        return "graph"
    else:
        return "None"


default_args = AttrDict(
    {
        "dropout": 0.1,
        "num_layers": 4,
        "hidden_dim": 64,
        "learning_rate": 1e-3,
        "layer_type": "MoE",
        "display": True,
        "num_trials": 200,  # Set high to allow stopping based on test appearances (default: 10)
        "eval_every": 1,
        "patience": 50,
        "output_dim": 2,
        "alpha": 0.1,
        "eps": 0.001,
        "dataset": None,
        "last_layer_fa": False,
        "encoding": None,
        "dataset_encoding": None,  # Pre-computed dataset encoding: None, hg_ldp, hg_frc, hg_rwpe_we_k20, hg_lape_normalized_k8,
        "encoding_moe_encodings": None,  # List of encodings for EncodingMoE (e.g., ["g_ldp", "g_orc"])
        "encoding_moe_router_type": "MLP",  # Router type for EncodingMoE: "MLP" or "GNN"
        "mlp": True,
        "layer_types": None,
        # WandB defaults
        "wandb_enabled": False,
        "wandb_project": "MOE_4",
        "wandb_entity": "weber-geoml-harvard-university",
        "wandb_name": None,
        "wandb_dir": "./wandb",
        "wandb_tags": None,
    }
)

hyperparams = {
    # TU datasets:
    "mutag": AttrDict({"output_dim": 2}),
    "enzymes": AttrDict({"output_dim": 6}),
    "proteins": AttrDict({"output_dim": 2}),
    "collab": AttrDict({"output_dim": 3}),
    "imdb": AttrDict({"output_dim": 2}),
    "reddit": AttrDict({"output_dim": 2}),
    # GNN Benchmark datasets:
    "mnist": AttrDict({"output_dim": 10}),
    "cifar": AttrDict({"output_dim": 10}),
    "pattern": AttrDict({"output_dim": 2}),  # Binary classification
    # LRGB datasets:
    "cluster": AttrDict({"output_dim": 6}),  # 6 clusters
    "pascalvoc": AttrDict({"output_dim": 21}),  # 21 object classes
    "coco": AttrDict({"output_dim": 81}),  # 81 object classes
    "peptides_func": AttrDict({"output_dim": 10}),  # 10 functional classes
    # OGB datasets:
    "molhiv": AttrDict(
        {"output_dim": 2}
    ),  # Binary classification (HIV active/inactive)
    "molpcba": AttrDict({"output_dim": 128}),  # Multi-label classification (128 assays)
    "ppa": AttrDict({"output_dim": 37}),  # 37 protein-protein association classes
    "code2": AttrDict({"output_dim": 1}),  # Regression task (single output)
    # Other datasets:
    "malnet": AttrDict({"output_dim": 5}),  # 5 malware categories
    # GraphBench datasets - output_dim will need to be determined based on actual dataset
    # These are placeholders and may need adjustment after loading the actual datasets
    # TODO TODO TODO
    "graphbench_socialnetwork": AttrDict(
        {"output_dim": 2}
    ),  # Placeholder - adjust based on actual task
    "graphbench_co": AttrDict(
        {"output_dim": 2}
    ),  # Placeholder - adjust based on actual task
    "graphbench_sat": AttrDict(
        {"output_dim": 2}
    ),  # Placeholder - adjust based on actual task
}

results = []
args = default_args
args += get_args_from_input()

# Track which encoding directory was used (for determining encoding_category later)
encoding_source_dir: str | None = None

# Load encoded datasets if dataset_encoding is specified
if hasattr(args, "dataset_encoding") and args.dataset_encoding is not None:
    dataset_encoding = args.dataset_encoding
    print(f"\n📦 Loading datasets with encoding: {dataset_encoding}")

    # Determine the encoding directory and file pattern
    if dataset_encoding.startswith("hg_"):
        # Hypergraph encodings: hg_ldp, hg_frc, hg_rwpe_we_k20, hg_lape_normalized_k8, etc.
        encoding_suffix = dataset_encoding[
            3:
        ]  # Remove "hg_" prefix to get suffix (e.g., "rwpe_we_k20", "lape_normalized_k8")
        encoded_data_dir = (
            Path(data_directory).parent / "graph_datasets_with_hg_encodings"
        )
        encoding_source_dir = str(encoded_data_dir)  # Track the directory used
        file_pattern = "{dataset_name}_hg_{encoding_suffix}.pt"
    elif dataset_encoding.startswith("g_"):
        # Graph encodings: g_ldp, g_rwpe_k16, g_lape_k8, g_orc, etc.
        encoding_suffix = dataset_encoding[
            2:
        ]  # Remove "g_" prefix to get suffix (e.g., "ldp", "rwpe_k16", "lape_k8")
        encoded_data_dir = (
            Path(data_directory).parent / "graph_datasets_with_g_encodings"
        )
        encoding_source_dir = str(encoded_data_dir)  # Track the directory used
        file_pattern = "{dataset_name}_g_{encoding_suffix}.pt"
    else:
        raise ValueError(
            f"Unknown dataset_encoding: {dataset_encoding}. Expected: None, hg_ldp, hg_frc, hg_rwpe_we_k20, hg_lape_normalized_k8, g_ldp, g_rwpe_k16, g_lape_k8, g_orc"
        )

    if not encoded_data_dir.exists():
        raise FileNotFoundError(
            f"Encoded datasets directory not found: {encoded_data_dir}"
        )

    print(f"📁 Encoded datasets directory: {encoded_data_dir}")
    print(f"📁 Encoded datasets directory exists: {encoded_data_dir.exists()}")
    print(f"📁 Encoded datasets directory (absolute): {encoded_data_dir.resolve()}")

    # List some example files in the directory for debugging
    try:
        existing_files = list(encoded_data_dir.glob("*.pt"))
        print(f"📋 Found {len(existing_files)} .pt files in directory")
        if len(existing_files) > 0:
            print(
                f"   Example files (first 10): {[f.name for f in existing_files[:10]]}"
            )
    except Exception as e:
        print(f"   ⚠️  Could not list files in directory: {e}")

    # Load encoded datasets
    # If --dataset is specified, only try to load encodings for that dataset
    # Otherwise, try to load encodings for all datasets (some may not have encodings)
    datasets_to_check = (
        [args.dataset]
        if hasattr(args, "dataset") and args.dataset
        else list(datasets.keys())
    )
    print(
        f"🔍 Checking encodings for {len(datasets_to_check)} dataset(s): {datasets_to_check}"
    )
    print(f"🔍 Encoding type: {dataset_encoding} (suffix: {encoding_suffix})")
    print(f"🔍 File pattern: {file_pattern}")

    encoded_datasets = {}
    skipped_datasets = []
    for dataset_name in datasets_to_check:
        if dataset_encoding.startswith("hg_"):
            filename = file_pattern.format(
                dataset_name=dataset_name, encoding_suffix=encoding_suffix
            )
        elif dataset_encoding.startswith("g_"):
            filename = file_pattern.format(
                dataset_name=dataset_name, encoding_suffix=encoding_suffix
            )
        else:
            filename = file_pattern.format(dataset_name=dataset_name)

        encoded_file_path = encoded_data_dir / filename
        print(f"  🔍 Dataset: {dataset_name}")
        print(f"     WHERE WE ARE LOOKING: {encoded_file_path}")
        print(f"     Absolute path: {encoded_file_path.resolve()}")
        print(f"     Filename constructed: {filename}")
        print(f"     File exists: {encoded_file_path.exists()}")

        if encoded_file_path.exists():
            try:
                print(f"  ⏳ Loading {dataset_name} with {dataset_encoding}...")
                encoded_dataset = torch.load(encoded_file_path)
                encoded_datasets[dataset_name] = encoded_dataset
                print(f"  ✅ {dataset_name} loaded: {len(encoded_dataset)} graphs")
            except Exception as e:
                print(
                    f"  ❌ Failed to load {dataset_name} with {dataset_encoding}: {e}"
                )
                print(
                    f"     Skipping {dataset_name} - encoding file exists but failed to load"
                )
                skipped_datasets.append(dataset_name)
        else:
            print(f"  ⚠️  Encoded dataset file not found: {encoded_file_path}")
            print(f"     Skipping {dataset_name} - encoding file not found")
            skipped_datasets.append(dataset_name)

    # Check if we have any encoded datasets
    if len(encoded_datasets) == 0:
        print(f"  ❌ No encoded datasets found for encoding: {dataset_encoding}")
        print(f"     Skipping experiment - at least one encoded dataset is required")
        import sys

        sys.exit(0)  # Exit gracefully, skip this experiment

    # Warn about skipped datasets
    if skipped_datasets:
        print(
            f"  ⚠️  Skipped {len(skipped_datasets)} dataset(s) without encodings: {', '.join(skipped_datasets)}"
        )

    # Replace datasets with encoded versions
    datasets = encoded_datasets
    print(f"✅ Loaded {len(datasets)} datasets with encoding: {dataset_encoding}\n")

# Load datasets for EncodingMoE if encoding_moe_encodings is specified
encoding_moe_base_datasets = None
encoding_moe_encoded_datasets = None

if _is_encoding_moe_enabled(args):
    encoding_moe_base_datasets, encoding_moe_encoded_datasets = (
        _load_encoding_moe_datasets(args, datasets, data_directory)
    )

if args.dataset:
    # restricts to just the given dataset if this mode is chosen
    name = args.dataset
    datasets = {name: datasets[name]}

for key in datasets:
    if key in ["reddit", "imdb", "collab"]:
        for graph in datasets[key]:
            n = graph.num_nodes
            graph.x = torch.ones((n, 1))
    # Handle GraphBench datasets that might not have node features
    elif key.startswith("graphbench_"):
        for graph in datasets[key]:
            if not hasattr(graph, "x") or graph.x is None:
                n = graph.num_nodes
                graph.x = torch.ones((n, 1))

for key in datasets:
    args += hyperparams[key]
    train_accuracies = []
    validation_accuracies = []
    test_accuracies = []
    energies = []
    dataset = datasets[key]

    # Determine which encoding method is being used
    dataset_encoding = getattr(args, "dataset_encoding", None)
    if dataset_encoding is not None:
        # New system: Pre-computed dataset encodings (preferred)
        encoding_info = f"dataset_encoding={dataset_encoding}"
        print(f"TESTING: {key} ({encoding_info}, layer={args.layer_type})")
        # Skip any on-the-fly encoding if dataset_encoding is specified
        pass
    elif args.encoding is not None:
        # DEPRECATED: On-the-fly encoding computation (args.encoding) is deprecated.
        # This approach has been replaced with pre-computed dataset encodings via args.dataset_encoding,
        # which loads pre-processed datasets from graph_datasets_with_hg_encodings/ or
        # graph_datasets_with_g_encodings/. The code below is kept for backwards compatibility
        # but should not be used in new experiments. Use --dataset_encoding instead.
        encoding_info = (
            f"encoding={args.encoding} (DEPRECATED - use --dataset_encoding instead)"
        )
        print(f"TESTING: {key} ({encoding_info}, layer={args.layer_type})")
        print(
            f"  ⚠️  WARNING: On-the-fly encoding ({args.encoding}) is deprecated. Use --dataset_encoding for pre-computed encodings."
        )
        if args.encoding in ["LAPE", "RWPE", "LCP", "LDP", "SUB", "EGO"]:

            if os.path.exists(f"data/{key}_{args.encoding}.pt"):
                print(f"✅ ENCODING ALREADY EXISTS: Loading {key}_{args.encoding}.pt")
                dataset = torch.load(f"data/{key}_{args.encoding}.pt")

            elif args.encoding == "LCP":
                print(f"🔄 ENCODING STARTED: {args.encoding} for {key.upper()}...")
                lcp = LocalCurvatureProfile()
                for i in tqdm(
                    range(len(dataset)),
                    desc=f"Encoding {key.upper()} with {args.encoding}",
                ):
                    dataset[i] = lcp.compute_orc(dataset[i])
                    print(f"Graph {i} of {len(dataset)} encoded with {args.encoding}")
                torch.save(dataset, f"data/{key}_{args.encoding}.pt")
                print(f"💾 Encoded dataset saved to: data/{key}_{args.encoding}.pt")

            else:
                print("ENCODING STARTED...")
                org_dataset_len = len(dataset)
                drop_datasets = []
                current_graph = 0

                for i in tqdm(
                    range(org_dataset_len),
                    desc=f"Encoding {key.upper()} with {args.encoding}",
                ):
                    if args.encoding == "LAPE":
                        num_nodes = dataset[i].num_nodes
                        eigvecs = np.min([num_nodes, 8]) - 2
                        transform = T.AddLaplacianEigenvectorPE(k=eigvecs)

                    elif args.encoding == "RWPE":
                        transform = T.AddRandomWalkPE(walk_length=16)

                    elif args.encoding == "LDP":
                        transform = T.LocalDegreeProfile()

                    elif args.encoding == "SUB":
                        transform = T.RootedRWSubgraph(walk_length=10)

                    elif args.encoding == "EGO":
                        transform = T.RootedEgoNets(num_hops=2)

                    try:
                        dataset[i] = transform(dataset[i])
                        print(
                            f"Graph {current_graph} of {org_dataset_len} encoded with {args.encoding}"
                        )
                        current_graph += 1

                    except Exception:
                        tqdm.write(
                            f"⚠️  Graph {current_graph} dropped due to encoding error"
                        )
                        drop_datasets.append(i)
                        current_graph += 1

                for i in sorted(drop_datasets, reverse=True):
                    dataset.pop(i)

                # save the dataset to a file in the data folder
                torch.save(dataset, f"data/{key}_{args.encoding}.pt")
                print(f"💾 Encoded dataset saved to: data/{key}_{args.encoding}.pt")
    else:
        # No encoding - using original dataset
        print(f"TESTING: {key} (no encoding, layer={args.layer_type})")

    # create a dictionary of the graphs in the dataset with the key being the graph index
    # graph_dict[graph_idx] = list of correctness values (1=correct, 0=incorrect) for each test appearance
    graph_dict = {}
    for i in range(len(dataset)):
        graph_dict[i] = []

    # Track how many times each graph has appeared in test sets
    test_appearances = {i: 0 for i in range(len(dataset))}
    required_test_appearances = 10  # Each graph should appear in test sets 10 times

    print("GRAPH DICTIONARY CREATED...")
    print(
        f"🎯 Goal: Each graph should appear in test sets {required_test_appearances} times"
    )

    print(f"🚀 TRAINING STARTED for {key.upper()} dataset...")
    print(
        f"🔧 Model: {args.layer_type} | Layers: {args.num_layers} | Hidden: {args.hidden_dim}"
    )
    start = time.time()

    # Add progress bar for trials
    experiment_id = None
    if args.wandb_enabled:
        # Create a unique experiment group ID (include skip_connection and normalize_features if relevant)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        skip_suffix = (
            "_skip"
            if getattr(args, "skip_connection", False)
            and args.layer_type in ["GCN", "GIN", "SAGE"]
            else ""
        )
        norm_suffix = "_norm" if getattr(args, "normalize_features", False) else ""
        dataset_encoding_str = getattr(args, "dataset_encoding", None) or "None"
        dataset_encoding_suffix = (
            f"_enc{dataset_encoding_str}" if dataset_encoding_str != "None" else ""
        )

        # Add EncodingMoE suffix if enabled
        encoding_moe_suffix = _get_encoding_moe_wandb_suffix(args)

        experiment_id = f"{key}_{args.layer_type}{skip_suffix}{norm_suffix}{dataset_encoding_suffix}{encoding_moe_suffix}_{timestamp}"
        if args.wandb_name:
            experiment_id = f"{args.wandb_name}{skip_suffix}{norm_suffix}{dataset_encoding_suffix}{encoding_moe_suffix}_{timestamp}"
        print(f"🔬 WandB Experiment Group: {experiment_id}")

    trial = 0
    while True:
        trial += 1

        # Check if we've satisfied the requirement (before running this trial)
        min_test_appearances = (
            min(test_appearances.values())
            if test_appearances and len(test_appearances) > 0
            else 0
        )
        if min_test_appearances >= required_test_appearances:
            print(
                f"\n✅ All graphs have appeared in test sets at least {required_test_appearances} times!"
            )
            print(f"   Stopping before trial {trial}")
            break

        # Also respect the num_trials limit if set (for backwards compatibility)
        # Note: num_trials should be set high (e.g., 200) to allow stopping based on test appearances
        if (
            hasattr(args, "num_trials")
            and args.num_trials is not None
            and trial > args.num_trials
        ):
            print(
                f"\n⚠️  Reached num_trials limit ({args.num_trials}), but not all graphs have appeared {required_test_appearances} times yet"
            )
            print(
                f"   Current min appearances: {min_test_appearances}/{required_test_appearances}"
            )
            print(
                "   Consider increasing num_trials to allow more trials for test appearance requirement"
            )
            break

        print(
            f"\n📊 TRIAL {trial} (Min test appearances before trial: {min_test_appearances}/{required_test_appearances})"
        )

        # Initialize wandb for this specific trial
        if args.wandb_enabled:
            trial_run_name = f"trial_{trial + 1:02d}"
            if args.wandb_name:
                trial_run_name = f"{args.wandb_name}_trial_{trial + 1:02d}"

            # Build config dictionary
            dataset_encoding_str = getattr(args, "dataset_encoding", None) or "None"
            # Determine encoding category based on the directory where encodings were loaded from:
            # - hypergraph encodings: graph_datasets_with_hg_encodings
            # - graph encodings: graph_datasets_with_g_encodings
            # - no encodings: None (using normal datasets)
            encoding_category = get_encoding_category(encoding_source_dir)

            wandb_config = {
                **dict(args),
                "trial_num": trial,
                "dataset": key,
                # Add grouping variables:
                "dataset_name": key,  # For grouping by dataset
                "dataset_encoding": dataset_encoding_str,  # Pre-computed dataset encoding
                "encoding_category": encoding_category,  # Category: None, hypergraph, or graph
                "is_moe": args.layer_types
                is not None,  # Boolean for MoE vs single layer
                "model_type": (
                    "MoE" if args.layer_types is not None else args.layer_type
                ),
                "num_layers": args.num_layers,
                "layer_combination": (
                    str(args.layer_types) if args.layer_types else args.layer_type
                ),
                "skip_connection": getattr(args, "skip_connection", False),
                "normalize_features": getattr(args, "normalize_features", False),
            }

            # Add EncodingMoE configuration if enabled
            is_encoding_moe = _is_encoding_moe_enabled(args)
            if is_encoding_moe:
                wandb_config["is_encoding_moe"] = True
                wandb_config["encoding_moe_encodings"] = args.encoding_moe_encodings
                wandb_config["encoding_moe_router_type"] = getattr(
                    args, "encoding_moe_router_type", "MLP"
                )
                wandb_config["model_type"] = "EncodingMoE"
            else:
                wandb_config["is_encoding_moe"] = False

            # Add router configuration ONLY for MoE models (not EncodingMoE, which has its own router)
            if args.layer_types is not None and not is_encoding_moe:
                router_type = getattr(args, "router_type", "MLP")
                wandb_config["router_type"] = router_type
                # If router_type is MLP, set router_layer_type to MLP for consistency
                # (router_layer_type is only used when router_type is GNN)
                if router_type == "MLP":
                    wandb_config["router_layer_type"] = "MLP"
                else:
                    wandb_config["router_layer_type"] = getattr(
                        args, "router_layer_type", "GIN"
                    )
                wandb_config["router_depth"] = getattr(args, "router_depth", 4)
                wandb_config["router_dropout"] = getattr(args, "router_dropout", 0.1)

            wandb.init(
                project=args.wandb_project,
                entity=args.wandb_entity,
                name=trial_run_name,
                group=experiment_id,  # Group all trials together
                config=wandb_config,
                dir=args.wandb_dir,
                tags=list(args.wandb_tags or []) + [f"trial_{trial}"],
                reinit=True,  # Allow multiple runs in same process
            )
            print(f"🚀 WandB Trial Run: {wandb.run.name}")

        try:
            # Pass encoded datasets to Experiment if EncodingMoE is enabled
            encoding_moe_encoded_datasets_for_key, dataset_to_use = (
                _extract_encoding_moe_datasets_for_key(
                    args,
                    key,
                    encoding_moe_encoded_datasets,
                    encoding_moe_base_datasets,
                )
            )
            # Fallback to regular dataset if EncodingMoE not enabled
            if dataset_to_use is None:
                dataset_to_use = dataset

            train_acc, validation_acc, test_acc, energy, dictionary, test_indices = (
                Experiment(
                    args=args,
                    dataset=dataset_to_use,
                    encoding_moe_encoded_datasets=encoding_moe_encoded_datasets_for_key,
                ).run()
            )

            train_accuracies.append(train_acc)
            validation_accuracies.append(validation_acc)
            test_accuracies.append(test_acc)
            energies.append(energy)

            # Track test set appearances and record correctness only for test set graphs
            for graph_idx in test_indices:
                test_appearances[graph_idx] += 1
                if graph_idx in dictionary and dictionary[graph_idx] != -1:
                    graph_dict[graph_idx].append(dictionary[graph_idx])

            # Show intermediate results
            min_appearances = min(test_appearances.values())
            max_appearances = max(test_appearances.values())
            print(
                f"   Train: {train_acc:.3f} | Val: {validation_acc:.3f} | Test: {test_acc:.3f}"
            )
            print(
                f"   Test appearances: min={min_appearances}, max={max_appearances}, graphs_in_test={len(test_indices)}"
            )

            # Log final trial results to this trial's wandb run
            if args.wandb_enabled:
                log_dict = {
                    "final/train_acc": train_acc,
                    "final/val_acc": validation_acc,
                    "final/test_acc": test_acc,
                    "final/energy": energy,
                    "final/min_test_appearances": min_appearances,
                    "final/max_test_appearances": max_appearances,
                    # Add grouping variables:
                    "groupby/dataset": key,
                    "groupby/dataset_encoding": getattr(args, "dataset_encoding", None)
                    or "None",
                    "groupby/encoding_category": encoding_category,
                    "groupby/model_type": (
                        "MoE" if args.layer_types else args.layer_type
                    ),
                    "groupby/is_moe": args.layer_types is not None,
                    "groupby/moe_layers": (
                        "+".join(args.layer_types)
                        if args.layer_types is not None
                        else None
                    ),
                    "groupby/num_layers": args.num_layers,
                    "groupby/skip_connection": getattr(args, "skip_connection", False),
                    "groupby/normalize_features": getattr(
                        args, "normalize_features", False
                    ),
                }
                # Add EncodingMoE info if enabled
                is_encoding_moe = _is_encoding_moe_enabled(args)
                if is_encoding_moe:
                    log_dict["groupby/is_encoding_moe"] = True
                    log_dict["groupby/encoding_moe_encodings"] = "+".join(
                        args.encoding_moe_encodings
                    )
                    log_dict["groupby/encoding_moe_router_type"] = getattr(
                        args, "encoding_moe_router_type", "MLP"
                    )
                else:
                    log_dict["groupby/is_encoding_moe"] = False

                # Add router info ONLY for MoE models (not EncodingMoE, which has its own router)
                if args.layer_types is not None and not is_encoding_moe:
                    router_type = getattr(args, "router_type", "MLP")
                    log_dict["groupby/router_type"] = router_type
                    # If router_type is MLP, set router_layer_type to MLP for consistency
                    # (router_layer_type is only used when router_type is GNN)
                    if router_type == "MLP":
                        log_dict["groupby/router_layer_type"] = "MLP"
                    else:
                        log_dict["groupby/router_layer_type"] = getattr(
                            args, "router_layer_type", "GIN"
                        )
                    log_dict["groupby/router_depth"] = getattr(args, "router_depth", 4)
                    log_dict["groupby/router_dropout"] = getattr(
                        args, "router_dropout", 0.1
                    )
                wandb.log(log_dict)

        finally:
            # Finish this trial's wandb run
            if args.wandb_enabled:
                wandb.finish()

    end = time.time()
    run_duration = end - start
    print(f"⏱️  Training completed in {run_duration:.2f} seconds")
    print(f"📊 Total trials run: {trial}")
    print(
        f"📈 Final test appearances: min={min(test_appearances.values())}, max={max(test_appearances.values())}"
    )

    # Calculate how many graphs have appeared the required number of times
    graphs_with_sufficient_appearances = sum(
        1 for count in test_appearances.values() if count >= required_test_appearances
    )
    print(
        f"✅ Graphs with ≥{required_test_appearances} test appearances: {graphs_with_sufficient_appearances}/{len(dataset)}"
    )

    # Save graph_dict (per-graph correctness tracking) and test_appearances
    import pickle

    os.makedirs(f"results/{args.num_layers}_layers", exist_ok=True)
    # Check if EncodingMoE is enabled
    is_encoding_moe = _is_encoding_moe_enabled(args)

    # Generate detailed model name for MOE models
    if is_encoding_moe:
        # EncodingMoE: use encoding names in model name
        encoding_names_str = "_".join(args.encoding_moe_encodings)
        router_type = getattr(args, "encoding_moe_router_type", "MLP")
        detailed_model_name = (
            f"EncodingMoE_{encoding_names_str}_r{router_type}_{args.layer_type}"
        )
    elif args.layer_types is not None:
        expert_combo = "_".join(args.layer_types)
        router_type = getattr(args, "router_type", "MLP")
        detailed_model_name = f"{args.layer_type}_{router_type}_{expert_combo}"
    else:
        detailed_model_name = args.layer_type

    # Add skip_connection suffix if applicable
    if getattr(args, "skip_connection", False) and args.layer_type in [
        "GCN",
        "GIN",
        "SAGE",
    ]:
        detailed_model_name = f"{detailed_model_name}_skip"

    # Add normalize_features suffix if applicable
    if getattr(args, "normalize_features", False):
        detailed_model_name = f"{detailed_model_name}_norm"

    # Use only pre-computed encoding for filename (legacy encoding is deprecated)
    # For EncodingMoE, don't add dataset_encoding to filename since encodings are in model name
    if is_encoding_moe:
        graph_dict_filename = f"results/{args.num_layers}_layers/{key}_{detailed_model_name}_graph_dict.pickle"
    else:
        dataset_encoding_str = getattr(args, "dataset_encoding", None) or "None"
        graph_dict_filename = f"results/{args.num_layers}_layers/{key}_{detailed_model_name}_enc{dataset_encoding_str}_graph_dict.pickle"
    with open(graph_dict_filename, "wb") as f:
        pickle.dump(
            {
                "graph_dict": graph_dict,
                "test_appearances": test_appearances,
                "required_test_appearances": required_test_appearances,
            },
            f,
        )
    print(f"💾 Graph correctness dictionary saved to: {graph_dict_filename}")

    # Generate and save heterogeneity profiles (average accuracy per graph plots)
    # These plots show how model performance varies across different graphs, revealing
    # data heterogeneity - which graphs are consistently easy/hard to predict.
    # Two plots are created: one ordered by graph index (by_index), one ordered by
    # accuracy (by_accuracy) to show the performance distribution.
    try:
        # Use full detailed encoding name (e.g., "hg_rwpe_we_k20", "g_lape_k8") for plot filenames
        dataset_encoding_for_plots = getattr(args, "dataset_encoding", None)
        skip_connection = getattr(args, "skip_connection", False)
        normalize_features = getattr(args, "normalize_features", False)

        # Check if EncodingMoE is enabled for plot filename
        is_encoding_moe_plots = _is_encoding_moe_enabled(args)

        original_plot_path, sorted_plot_path = load_and_plot_average_per_graph(
            graph_dict_filename,
            dataset_name=key,
            layer_type=args.layer_type,
            encoding=dataset_encoding_for_plots,
            num_layers=args.num_layers,
            task_type="classification",
            output_dir="results",
            layer_types=args.layer_types if args.layer_types else None,
            router_type=(
                getattr(args, "encoding_moe_router_type", "MLP")
                if is_encoding_moe_plots
                else getattr(args, "router_type", "MLP")
            ),
            skip_connection=skip_connection,
            normalize_features=normalize_features,
            is_encoding_moe=is_encoding_moe_plots,
        )
        if original_plot_path:
            print(f"📊 Average accuracy plot (by index) saved to: {original_plot_path}")
        if sorted_plot_path:
            print(
                f"📊 Average accuracy plot (by accuracy) saved to: {sorted_plot_path}"
            )
    except Exception as e:
        print(f"⚠️  Failed to generate average accuracy plots: {e}")

    # Calculate statistics
    num_trials_actual = len(train_accuracies)
    train_mean = 100 * np.mean(train_accuracies)
    val_mean = 100 * np.mean(validation_accuracies)
    test_mean = 100 * np.mean(test_accuracies)
    energy_mean = 100 * np.mean(energies)
    # Standard deviations (raw)
    train_std = 100 * np.std(train_accuracies) if num_trials_actual > 0 else 0
    val_std = 100 * np.std(validation_accuracies) if num_trials_actual > 0 else 0
    test_std = 100 * np.std(test_accuracies) if num_trials_actual > 0 else 0
    energy_std = 100 * np.std(energies) if num_trials_actual > 0 else 0
    # Confidence intervals (2 * std / sqrt(n) for 95% CI)
    train_ci = (
        2 * np.std(train_accuracies) / (num_trials_actual**0.5)
        if num_trials_actual > 0
        else 0
    )
    val_ci = (
        2 * np.std(validation_accuracies) / (num_trials_actual**0.5)
        if num_trials_actual > 0
        else 0
    )
    test_ci = (
        2 * np.std(test_accuracies) / (num_trials_actual**0.5)
        if num_trials_actual > 0
        else 0
    )
    energy_ci = (
        200 * np.std(energies) / (num_trials_actual**0.5)
        if num_trials_actual > 0
        else 0
    )
    dataset_encoding_str = getattr(args, "dataset_encoding", None) or "None"
    log_to_file(
        f"RESULTS FOR dataset: {key} (model: {args.layer_type}), dataset_encoding: {dataset_encoding_str}:\n"
    )
    log_to_file(f"average acc: {test_mean}\n")
    log_to_file(f"plus/minus (CI):  {test_ci}\n")
    log_to_file(f"std deviation: {test_std}\n\n")
    # Check if EncodingMoE is enabled for results tracking
    is_encoding_moe_results = _is_encoding_moe_enabled(args)

    # Check if regular MoE is enabled
    is_moe_results = args.layer_types is not None and not is_encoding_moe_results

    results.append(
        {
            "dataset": key,
            "dataset_encoding": getattr(args, "dataset_encoding", None),
            "encoding": args.encoding,
            "layer_type": args.layer_type,
            "layer_types": args.layer_types,
            "num_layers": args.num_layers,
            "alpha": args.alpha,
            "eps": args.eps,
            "test_mean": test_mean,
            "test_std": test_std,
            "test_ci": test_ci,
            "val_mean": val_mean,
            "val_std": val_std,
            "val_ci": val_ci,
            "train_mean": train_mean,
            "train_std": train_std,
            "train_ci": train_ci,
            "energy_mean": energy_mean,
            "energy_std": energy_std,
            "energy_ci": energy_ci,
            "last_layer_fa": args.last_layer_fa,
            "run_duration": run_duration,
            "num_trials_actual": num_trials_actual,
            "min_test_appearances": min(test_appearances.values()),
            "max_test_appearances": max(test_appearances.values()),
            "graphs_with_sufficient_appearances": graphs_with_sufficient_appearances,
            "is_moe": is_moe_results,
            "is_encoding_moe": is_encoding_moe_results,
            "encoding_moe_encodings": (
                "+".join(args.encoding_moe_encodings)
                if is_encoding_moe_results
                else None
            ),
            "encoding_moe_router_type": (
                getattr(args, "encoding_moe_router_type", "MLP")
                if is_encoding_moe_results
                else None
            ),
            "router_type": (
                getattr(args, "router_type", "MLP") if is_moe_results else None
            ),
            "router_layer_type": (
                getattr(args, "router_layer_type", "GIN")
                if is_moe_results and getattr(args, "router_type", "MLP") != "MLP"
                else (
                    "MLP"
                    if is_moe_results and getattr(args, "router_type", "MLP") == "MLP"
                    else None
                )
            ),
            "router_depth": (
                getattr(args, "router_depth", 4) if is_moe_results else None
            ),
            "router_dropout": (
                getattr(args, "router_dropout", 0.1) if is_moe_results else None
            ),
            "skip_connection": getattr(args, "skip_connection", False),
            "normalize_features": getattr(args, "normalize_features", False),
        }
    )

    # Create a summary run for the overall experiment
    if args.wandb_enabled:
        summary_run_name = "SUMMARY"
        if args.wandb_name:
            summary_run_name = f"{args.wandb_name}_SUMMARY"

        # Build config dictionary for summary run
        dataset_encoding_str = getattr(args, "dataset_encoding", None) or "None"
        # Determine encoding category based on the directory where encodings were loaded from:
        # - hypergraph encodings: graph_datasets_with_hg_encodings
        # - graph encodings: graph_datasets_with_g_encodings
        # - no encodings: None (using normal datasets)
        encoding_category = get_encoding_category(encoding_source_dir)

        # Check if EncodingMoE is enabled
        is_encoding_moe = (
            hasattr(args, "encoding_moe_encodings")
            and args.encoding_moe_encodings is not None
            and len(args.encoding_moe_encodings) > 0
        )

        # Determine layer combination string (for both MoE and single layer models)
        if is_encoding_moe:
            layer_combination = f"EncodingMoE({'+'.join(args.encoding_moe_encodings)})"
        elif args.layer_types is not None:
            layer_combination = str(args.layer_types)  # e.g., "['GCN', 'GIN']"
        else:
            layer_combination = args.layer_type  # e.g., "GCN"

        summary_config = {
            **dict(args),
            "dataset": key,
            "run_type": "summary",
            "num_trials": num_trials_actual,
            "required_test_appearances": required_test_appearances,
            "dataset_encoding": dataset_encoding_str,
            "encoding_category": encoding_category,
            "is_moe": args.layer_types is not None and not is_encoding_moe,
            "is_encoding_moe": is_encoding_moe,
            "layer_combination": layer_combination,
            "model_type": (
                "EncodingMoE"
                if is_encoding_moe
                else "MoE" if args.layer_types is not None else args.layer_type
            ),
            "skip_connection": getattr(args, "skip_connection", False),
            "normalize_features": getattr(args, "normalize_features", False),
        }

        # Add EncodingMoE configuration if enabled
        if is_encoding_moe:
            summary_config["encoding_moe_encodings"] = args.encoding_moe_encodings
            summary_config["encoding_moe_router_type"] = getattr(
                args, "encoding_moe_router_type", "MLP"
            )

        # Add router configuration ONLY for MoE models (not EncodingMoE, which has its own router)
        if args.layer_types is not None and not is_encoding_moe:
            router_type = getattr(args, "router_type", "MLP")
            summary_config["router_type"] = router_type
            # If router_type is MLP, set router_layer_type to MLP for consistency
            # (router_layer_type is only used when router_type is GNN)
            if router_type == "MLP":
                summary_config["router_layer_type"] = "MLP"
            else:
                summary_config["router_layer_type"] = getattr(
                    args, "router_layer_type", "GIN"
                )
            summary_config["router_depth"] = getattr(args, "router_depth", 4)
            summary_config["router_dropout"] = getattr(args, "router_dropout", 0.1)

        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=summary_run_name,
            group=experiment_id,
            config=summary_config,
            dir=args.wandb_dir,
            tags=list(args.wandb_tags or []) + ["summary"],
            reinit=True,
        )

        # Log aggregate statistics
        # Check if EncodingMoE is enabled (reuse from summary_config section)
        is_encoding_moe_summary = _is_encoding_moe_enabled(args)

        # Determine layer combination string (for both MoE and single layer models)
        if is_encoding_moe_summary:
            layer_combination_summary = (
                f"EncodingMoE({'+'.join(args.encoding_moe_encodings)})"
            )
        elif args.layer_types is not None:
            layer_combination_summary = str(args.layer_types)  # e.g., "['GCN', 'GIN']"
        else:
            layer_combination_summary = args.layer_type  # e.g., "GCN"

        summary_log_dict = {
            "summary/test_mean": test_mean,
            "summary/test_std": test_std,
            "summary/test_ci": test_ci,
            "summary/val_mean": val_mean,
            "summary/val_std": val_std,
            "summary/val_ci": val_ci,
            "summary/train_mean": train_mean,
            "summary/train_std": train_std,
            "summary/train_ci": train_ci,
            "summary/energy_mean": energy_mean,
            "summary/energy_std": energy_std,
            "summary/energy_ci": energy_ci,
            "summary/run_duration": run_duration,
            "summary/num_trials_actual": num_trials_actual,
            "summary/min_test_appearances": min(test_appearances.values()),
            "summary/max_test_appearances": max(test_appearances.values()),
            "summary/graphs_with_sufficient_appearances": graphs_with_sufficient_appearances,
            # Model metadata
            "summary/is_moe": args.layer_types is not None
            and not is_encoding_moe_summary,
            "summary/is_encoding_moe": is_encoding_moe_summary,
            "summary/layer_combination": layer_combination_summary,
            "summary/model_type": (
                "EncodingMoE"
                if is_encoding_moe_summary
                else "MoE" if args.layer_types is not None else args.layer_type
            ),
            # Log individual trial results for analysis
            "trials/train_accs": train_accuracies,
            "trials/val_accs": validation_accuracies,
            "trials/test_accs": test_accuracies,
            "trials/energies": energies,
            # Add grouping variables:
            "groupby/dataset": key,
            "groupby/dataset_encoding": getattr(args, "dataset_encoding", None)
            or "None",
            "groupby/encoding_category": encoding_category,
            "groupby/model_type": (
                "EncodingMoE"
                if is_encoding_moe_summary
                else "MoE" if args.layer_types else args.layer_type
            ),
            "groupby/is_moe": args.layer_types is not None
            and not is_encoding_moe_summary,
            "groupby/is_encoding_moe": is_encoding_moe_summary,
            "groupby/moe_layers": (
                "+".join(args.layer_types) if args.layer_types is not None else None
            ),
            "groupby/num_layers": args.num_layers,
            "groupby/skip_connection": getattr(args, "skip_connection", False),
            "groupby/normalize_features": getattr(args, "normalize_features", False),
        }

        # Add EncodingMoE grouping info if enabled
        if is_encoding_moe_summary:
            summary_log_dict["groupby/encoding_moe_encodings"] = "+".join(
                args.encoding_moe_encodings
            )
            summary_log_dict["groupby/encoding_moe_router_type"] = getattr(
                args, "encoding_moe_router_type", "MLP"
            )

        # Add router info ONLY for MoE models (not EncodingMoE, which has its own router)
        if args.layer_types is not None and not is_encoding_moe_summary:
            router_type = getattr(args, "router_type", "MLP")
            summary_log_dict["groupby/router_type"] = router_type
            # If router_type is MLP, set router_layer_type to MLP for consistency
            # (router_layer_type is only used when router_type is GNN)
            if router_type == "MLP":
                summary_log_dict["groupby/router_layer_type"] = "MLP"
            else:
                summary_log_dict["groupby/router_layer_type"] = getattr(
                    args, "router_layer_type", "GIN"
                )
            summary_log_dict["groupby/router_depth"] = getattr(args, "router_depth", 4)
            summary_log_dict["groupby/router_dropout"] = getattr(
                args, "router_dropout", 0.1
            )
        wandb.log(summary_log_dict)

        # Create a summary table
        trial_data = []
        for i, (train_acc, val_acc, test_acc, energy) in enumerate(
            zip(train_accuracies, validation_accuracies, test_accuracies, energies)
        ):
            trial_data.append([i + 1, train_acc, val_acc, test_acc, energy])

        wandb.log(
            {
                "trials_table": wandb.Table(
                    columns=["Trial", "Train_Acc", "Val_Acc", "Test_Acc", "Energy"],
                    data=trial_data,
                )
            }
        )

        wandb.finish()

    # Log every time a dataset is completed
    # Use full detailed encoding name (e.g., "hg_rwpe_we_k20", "g_lape_k8", not abbreviated)
    dataset_encoding_str = getattr(args, "dataset_encoding", None) or "None"
    skip_connection = getattr(args, "skip_connection", False)
    normalize_features = getattr(args, "normalize_features", False)
    skip_str = "true" if skip_connection else "false"
    norm_str = "true" if normalize_features else "false"

    # Check if EncodingMoE is enabled for CSV filename
    is_encoding_moe_csv = _is_encoding_moe_enabled(args)

    # Create more precise CSV filename with skip, normalize, and encoding info
    if is_encoding_moe_csv:
        # For EncodingMoE, include encoding names in filename
        encoding_names_csv = "_".join(args.encoding_moe_encodings)
        router_type_csv = getattr(args, "encoding_moe_router_type", "MLP")
        encoding_part = f"encmoe_{encoding_names_csv}_r{router_type_csv}"
        model_name_csv = f"EncodingMoE_{args.layer_type}"
    else:
        # For regular models, use dataset_encoding
        encoding_part = (
            f"encodings_{dataset_encoding_str}"
            if dataset_encoding_str != "None"
            else "encodings_none"
        )
        model_name_csv = args.layer_type

    csv_filename = f"results/graph_classification_{model_name_csv}_skip_{skip_str}_norm_{norm_str}_{encoding_part}.csv"
    df = pd.DataFrame(results)
    with open(csv_filename, "a") as f:
        df.to_csv(f, mode="a", header=f.tell() == 0)

    print(f"\n🎯 FINAL RESULTS for {key.upper()}:")
    print(
        f"   📈 Test Accuracy: {test_mean:.2f}% ± {test_std:.2f}% (std) / ± {test_ci:.2f}% (95% CI)"
    )
    print(
        f"   📊 Validation Accuracy: {val_mean:.2f}% ± {val_std:.2f}% (std) / ± {val_ci:.2f}% (95% CI)"
    )
    print(
        f"   🏃 Training Accuracy: {train_mean:.2f}% ± {train_std:.2f}% (std) / ± {train_ci:.2f}% (95% CI)"
    )
    print(
        f"   ⚡ Energy: {energy_mean:.2f}% ± {energy_std:.2f}% (std) / ± {energy_ci:.2f}% (95% CI)"
    )
    print(f"   ⏱️  Duration: {run_duration:.2f}s")
    print(f"   💾 Results saved to: {csv_filename}")
