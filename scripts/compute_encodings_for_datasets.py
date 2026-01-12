"""Script to compute hypergraph and graph encodings for all datasets.

This script:
1. Loads all datasets from graph_moes/graph_datasets
2. Computes hypergraph encodings (lifts graphs to hypergraphs, computes encodings)
3. Computes graph-level encodings (directly on graphs)
4. Saves augmented datasets to separate directories
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch_geometric.transforms as T
from ogb.graphproppred import PygGraphPropPredDataset
from torch_geometric.data import Data
from torch_geometric.datasets import GNNBenchmarkDataset, TUDataset
from tqdm import tqdm

# Add hypergraph encodings repo to path
hypergraph_encodings_path = (
    Path(__file__).parent.parent.parent
    / "Hypergraph_encodings_clean"
    / "Hypergraph_Encodings"
)
if hypergraph_encodings_path.exists():
    sys.path.insert(0, str(hypergraph_encodings_path / "src"))
    try:
        from encodings_hnns.encodings import HypergraphEncodings
        from encodings_hnns.expansions import compute_clique_expansion
        from encodings_hnns.liftings_and_expansions import lift_to_hypergraph

        HYPERGRAPH_ENCODINGS_AVAILABLE = True
    except ImportError as e:
        print(f"⚠️  Warning: Could not import hypergraph encodings: {e}")
        print("   Hypergraph encodings will be skipped.")
        HYPERGRAPH_ENCODINGS_AVAILABLE = False
else:
    print(
        f"⚠️  Warning: Hypergraph encodings repo not found at {hypergraph_encodings_path}"
    )
    print("   Hypergraph encodings will be skipped.")
    HYPERGRAPH_ENCODINGS_AVAILABLE = False

# Import graph-level encodings from graph_moes
try:
    from graph_moes.encodings.custom_encodings import LocalCurvatureProfile

    GRAPH_ENCODINGS_AVAILABLE = True
except ImportError:
    print("⚠️  Warning: Could not import graph encodings from graph_moes")
    GRAPH_ENCODINGS_AVAILABLE = False


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


def load_all_datasets(data_directory: str) -> Dict[str, List[Data]]:
    """Load all datasets from the data directory.

    Args:
        data_directory: Path to the directory containing datasets

    Returns:
        Dictionary mapping dataset names to lists of Data objects
    """
    datasets: Dict[str, List[Data]] = {}

    print("📊 Loading datasets...")

    # TU datasets
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
            dataset = list(TUDataset(root=data_directory, name=ds_name))
            short_name = ds_name.lower().replace("-", "_")
            datasets[short_name] = dataset
            print(f"  ✅ {ds_name} loaded: {len(dataset)} graphs")
        except Exception as e:
            print(f"  ⚠️  Failed to load {ds_name}: {e}")

    # GNN Benchmark datasets
    gnn_benchmark = ["MNIST", "CIFAR10", "PATTERN"]
    for ds_name in gnn_benchmark:
        try:
            print(f"  ⏳ Loading {ds_name}...")
            dataset = list(GNNBenchmarkDataset(root=data_directory, name=ds_name))
            datasets[ds_name.lower()] = dataset
            print(f"  ✅ {ds_name} loaded: {len(dataset)} graphs")
        except Exception as e:
            print(f"  ⚠️  Failed to load {ds_name}: {e}")

    # OGB datasets
    ogb_datasets = ["ogbg-molhiv", "ogbg-molpcba", "ogbg-ppa"]
    for ds_name in ogb_datasets:
        try:
            print(f"  ⏳ Loading {ds_name}...")
            dataset = PygGraphPropPredDataset(name=ds_name, root=data_directory)
            dataset_list = [dataset[i] for i in range(len(dataset))]
            short_name = ds_name.replace("ogbg-", "").replace("-", "_")
            datasets[short_name] = dataset_list
            print(f"  ✅ {ds_name} loaded: {len(dataset_list)} graphs")
        except Exception as e:
            print(f"  ⚠️  Failed to load {ds_name}: {e}")

    # Peptides-func
    try:
        peptides_func_path = os.path.join(data_directory, "peptidesfunc")
        if os.path.exists(peptides_func_path):
            print("  ⏳ Loading Peptides-func...")
            peptides_train = torch.load(os.path.join(peptides_func_path, "train.pt"))
            peptides_val = torch.load(os.path.join(peptides_func_path, "val.pt"))
            peptides_test = torch.load(os.path.join(peptides_func_path, "test.pt"))
            peptides_func = (
                [_convert_lrgb(peptides_train[i]) for i in range(len(peptides_train))]
                + [_convert_lrgb(peptides_val[i]) for i in range(len(peptides_val))]
                + [_convert_lrgb(peptides_test[i]) for i in range(len(peptides_test))]
            )
            datasets["peptides_func"] = peptides_func
            print(f"  ✅ Peptides-func loaded: {len(peptides_func)} graphs")
    except Exception as e:
        print(f"  ⚠️  Failed to load Peptides-func: {e}")

    # MalNet-Tiny
    try:
        print("  ⏳ Loading MalNetTiny...")
        malnet = list(TUDataset(root=data_directory, name="MalNetTiny"))
        datasets["malnet"] = malnet
        print(f"  ✅ MalNetTiny loaded: {len(malnet)} graphs")
    except Exception as e:
        print(f"  ⚠️  Failed to load MalNetTiny: {e}")

    print(f"\n✅ Loaded {len(datasets)} datasets")
    return datasets


def convert_hypergraph_dict_to_pyg_data(
    hypergraph_dict: Dict[str, Any], original_graph: Data
) -> Data:
    """Convert hypergraph dictionary back to PyG Data format.

    Uses clique expansion to preserve the graph structure while keeping
    the augmented features from hypergraph encodings.

    Args:
        hypergraph_dict: Dictionary with 'hypergraph', 'features', 'labels', 'n'
        original_graph: Original PyG Data object to extract edge structure

    Returns:
        PyG Data object with augmented features
    """
    # Use clique expansion to convert hypergraph back to graph
    if not HYPERGRAPH_ENCODINGS_AVAILABLE:
        return original_graph
    expanded_graph = compute_clique_expansion(hypergraph_dict)  # type: ignore

    # Ensure features are numpy arrays and convert to torch
    features = hypergraph_dict["features"]
    if isinstance(features, np.ndarray):
        expanded_graph.x = torch.tensor(features, dtype=torch.float32)
    else:
        expanded_graph.x = torch.tensor(np.array(features), dtype=torch.float32)

    # Copy labels from original graph (graph-level labels)
    if hasattr(original_graph, "y") and original_graph.y is not None:
        expanded_graph.y = original_graph.y
    elif "labels" in hypergraph_dict:
        labels = hypergraph_dict["labels"]
        if isinstance(labels, np.ndarray):
            if labels.ndim == 0:
                expanded_graph.y = torch.tensor([labels], dtype=torch.long)
            else:
                expanded_graph.y = torch.tensor(labels, dtype=torch.long)
        else:
            expanded_graph.y = torch.tensor(labels, dtype=torch.long)

    # Copy edge attributes if they exist
    if hasattr(original_graph, "edge_attr") and original_graph.edge_attr is not None:
        # Need to map edge attributes - for now, just copy structure
        # This is a simplification - in practice, edge attributes might need remapping
        pass

    return expanded_graph


def compute_single_hypergraph_encoding(
    graph: Data, encoding_type: str, encoding_params: dict, logger=None
) -> Data:
    """Compute a single hypergraph encoding for a graph.

    Each encoding is computed separately from the original hypergraph format.

    Args:
        graph: PyG Data object representing the graph
        encoding_type: Type of encoding ('ldp', 'frc', 'orc', 'rwpe', 'lape')
        encoding_params: Parameters for the encoding
        logger: Optional logger function for logging

    Returns:
        PyG Data object with augmented features from the encoding
    """
    log = logger if logger else (lambda x: None)

    if not HYPERGRAPH_ENCODINGS_AVAILABLE:
        return graph

    # Lift graph to hypergraph (always start fresh from original graph)
    hypergraph_dict = lift_to_hypergraph(graph, verbose=False)  # type: ignore

    # Ensure features are numpy arrays
    if isinstance(hypergraph_dict["features"], torch.Tensor):
        hypergraph_dict["features"] = hypergraph_dict["features"].cpu().numpy()
    elif not isinstance(hypergraph_dict["features"], np.ndarray):
        hypergraph_dict["features"] = np.array(hypergraph_dict["features"])

    # Reshape features if needed
    if hypergraph_dict["features"].ndim == 1:
        hypergraph_dict["features"] = hypergraph_dict["features"].reshape(-1, 1)

    log(
        f"    Computing {encoding_type.upper()} on hypergraph with {hypergraph_dict['n']} nodes, {len(hypergraph_dict['hypergraph'])} hyperedges"
    )
    log(f"    Original features shape: {hypergraph_dict['features'].shape}")

    # Compute the specific encoding (using a copy to avoid modifying original)
    try:
        encoder = HypergraphEncodings()  # type: ignore

        if encoding_type == "ldp":
            hypergraph_dict = encoder.add_degree_encodings(
                hypergraph_dict.copy(),
                verbose=False,
                normalized=True,
                dataset_name=None,
            )
        elif encoding_type == "frc":
            hypergraph_dict = encoder.add_curvature_encodings(
                hypergraph_dict.copy(),
                verbose=False,
                curvature_type="FRC",
                normalized=True,
                dataset_name=None,
            )
        elif encoding_type == "orc":
            hypergraph_dict = encoder.add_curvature_encodings(
                hypergraph_dict.copy(),
                verbose=False,
                curvature_type="ORC",
                normalized=True,
                dataset_name=None,
            )
        elif encoding_type == "rwpe":
            hypergraph_dict = encoder.add_randowm_walks_encodings(
                hypergraph_dict.copy(),
                verbose=False,
                rw_type=encoding_params.get("rw_type", "WE"),
                k=encoding_params.get("k", 20),
                normalized=True,
                dataset_name=None,
            )
        elif encoding_type == "lape":
            hypergraph_dict = encoder.add_laplacian_encodings(
                hypergraph_dict.copy(),
                verbose=False,
                laplacian_type=encoding_params.get("laplacian_type", "Normalized"),
                normalized=True,
                k=encoding_params.get("k", 8),
                dataset_name=None,
            )
        else:
            raise ValueError(f"Unknown encoding type: {encoding_type}")

        log(
            f"    After {encoding_type.upper()}: features shape={hypergraph_dict['features'].shape}"
        )

    except Exception as e:
        import traceback

        error_msg = f"    ⚠️  Error computing {encoding_type.upper()}: {e}"
        log(error_msg)
        log(f"    Traceback: {traceback.format_exc()}")
        # Return original graph if encoding fails
        return graph

    # Convert back to PyG Data format
    augmented_graph = convert_hypergraph_dict_to_pyg_data(hypergraph_dict, graph)  # type: ignore
    return augmented_graph


def compute_single_graph_encoding(
    graph: Data, encoding_type: str, encoding_params: dict, logger=None
) -> Data:
    """Compute a single graph-level encoding for a graph.

    Each encoding is computed separately from the original graph format.

    Args:
        graph: PyG Data object representing the graph
        encoding_type: Type of encoding ('ldp', 'rwpe', 'lape', 'orc')
        encoding_params: Parameters for the encoding
        logger: Optional logger function for logging

    Returns:
        PyG Data object with augmented features from the encoding
    """
    log = logger if logger else (lambda x: None)

    log(
        f"  Processing graph with {graph.num_nodes} nodes, {graph.edge_index.shape[1] if graph.edge_index is not None else 0} edges"
    )

    # Always start from original graph
    graph = graph.clone()

    # Initialize features if None
    log(
        f"    Original features: shape={graph.x.shape if graph.x is not None else 'None'}"
    )
    if graph.x is None:
        graph.x = torch.ones((graph.num_nodes, 1), dtype=torch.float32)
        log("    Initialized features to ones")

    # Ensure x is 2D
    if graph.x.dim() == 1:
        graph.x = graph.x.unsqueeze(1)
        log(f"    Reshaped 1D features to 2D: shape={graph.x.shape}")

    log(f"    Computing {encoding_type.upper()} on graph with {graph.num_nodes} nodes")
    log(f"    Original features shape: {graph.x.shape}")

    # Compute the specific encoding
    try:
        if encoding_type == "ldp":
            transform = T.LocalDegreeProfile()
            graph = transform(graph)
        elif encoding_type == "rwpe":
            walk_length = encoding_params.get("walk_length", 16)
            transform = T.AddRandomWalkPE(walk_length=walk_length)
            graph = transform(graph)
        elif encoding_type == "lape":
            k = encoding_params.get("k", 8)
            num_nodes = graph.num_nodes
            k = min(num_nodes - 1, k)
            if k > 0:
                transform = T.AddLaplacianEigenvectorPE(k=k)
                graph = transform(graph)
            else:
                log("    Skipping LAPE: k=0")
        elif encoding_type == "orc":
            if GRAPH_ENCODINGS_AVAILABLE:
                lcp = LocalCurvatureProfile()
                graph = lcp.compute_orc(graph)
            else:
                log("    Skipping ORC: GRAPH_ENCODINGS_AVAILABLE=False")
        else:
            raise ValueError(f"Unknown encoding type: {encoding_type}")

        log(f"    After {encoding_type.upper()}: features shape={graph.x.shape}")

    except Exception as e:
        import traceback

        error_msg = f"    ⚠️  Error computing {encoding_type.upper()}: {e}"
        log(error_msg)
        log(f"    Traceback: {traceback.format_exc()}")
        # Return original graph if encoding fails
        return graph.clone()

    return graph


def process_dataset_with_hypergraph_encodings(
    dataset: List[Data],
    dataset_name: str,
    output_dir: str,
    verbose: bool = False,
    encoding_type_filter: Optional[str] = None,
) -> None:
    """Process a dataset with hypergraph encodings.

    Computes each encoding separately and saves to separate files.

    Args:
        dataset: List of PyG Data objects
        dataset_name: Name of the dataset
        output_dir: Output directory for saved dataset
        verbose: Whether to print progress
    """
    if not HYPERGRAPH_ENCODINGS_AVAILABLE:
        print(f"⚠️  Skipping {dataset_name}: hypergraph encodings not available")
        return

    print(
        f"\n🔄 Processing {dataset_name} with hypergraph encodings ({len(dataset)} graphs)..."
    )

    # Log dataset statistics
    print(f"  Dataset info:")
    print(f"    Number of graphs: {len(dataset)}")
    if len(dataset) > 0:
        sample_graph = dataset[0]
        print(
            f"    Sample graph: {sample_graph.num_nodes} nodes, {sample_graph.edge_index.shape[1] if sample_graph.edge_index is not None else 0} edges"
        )
        if sample_graph.x is not None:
            print(f"    Sample features shape: {sample_graph.x.shape}")
        else:
            print(f"    Sample features: None")

    # Define encodings to compute (each separately)
    # NOTE: ORC is very slow (~17 min for 188 graphs), so skipping for now
    # Format: (encoding_type, params_dict, description, filename_suffix)
    all_encodings = [
        ("ldp", {}, "LDP (Local Degree Profile)", "ldp"),
        ("frc", {}, "FRC (Forman-Ricci Curvature)", "frc"),
        ("orc", {}, "ORC (Ollivier-Ricci Curvature)", "orc"),
        (
            "rwpe",
            {"rw_type": "WE", "k": 20},
            "RWPE (Random Walk Positional Encoding)",
            "rwpe_we_k20",
        ),
        (
            "lape",
            {"laplacian_type": "Normalized", "k": 8},
            "LAPE (Laplacian Positional Encoding)",
            "lape_normalized_k8",
        ),
    ]

    # Filter to only the requested encoding type if specified
    if encoding_type_filter:
        encodings_to_compute = [
            enc for enc in all_encodings if enc[0] == encoding_type_filter
        ]
        if not encodings_to_compute:
            print(
                f"⚠️  Warning: Encoding type '{encoding_type_filter}' not found for hypergraph encodings"
            )
            print(f"   Available types: {[enc[0] for enc in all_encodings]}")
            return
    else:
        # Default: skip ORC as it's very slow
        encodings_to_compute = [enc for enc in all_encodings if enc[0] != "orc"]

    # Setup logging function
    def log(msg: str) -> None:
        """Log message for first graph in verbose mode."""
        if verbose:
            print(msg)

    # Process each encoding type separately
    for (
        encoding_type,
        encoding_params,
        encoding_name,
        filename_suffix,
    ) in encodings_to_compute:
        print(f"\n  📊 Computing {encoding_name}...")
        augmented_dataset: List[Data] = []
        failed_count = 0

        for i, graph in enumerate(tqdm(dataset, desc=f"  {encoding_type.upper()}")):
            try:
                augmented_graph = compute_single_hypergraph_encoding(
                    graph,
                    encoding_type,
                    encoding_params,
                    logger=log if i == 0 else None,
                )
                augmented_dataset.append(augmented_graph)
                if i == 0 and verbose:
                    print(
                        f"  First graph after {encoding_type.upper()}: features shape={augmented_graph.x.shape if augmented_graph.x is not None else 'None'}"
                    )
            except Exception as e:
                failed_count += 1
                error_msg = (
                    f"\n⚠️  Error processing graph {i} with {encoding_type.upper()}: {e}"
                )
                if failed_count <= 5:  # Only print first few errors
                    print(error_msg)
                if verbose and i < 5:
                    import traceback

                    traceback.print_exc()
                # Keep original graph if encoding fails
                augmented_dataset.append(graph)

        if failed_count > 0:
            print(f"  ⚠️  {failed_count} graphs failed {encoding_type.upper()} encoding")

        # Save dataset with this encoding type (using specific filename suffix)
        output_path = os.path.join(
            output_dir, f"{dataset_name}_hg_{filename_suffix}.pt"
        )
        os.makedirs(output_dir, exist_ok=True)
        torch.save(augmented_dataset, output_path)
        print(f"  ✅ Saved {len(augmented_dataset)} graphs to {output_path}")

        # Log final statistics
        if len(augmented_dataset) > 0:
            sample = augmented_dataset[0]
            if sample.x is not None:
                print(
                    f"  Final feature dimensions: {sample.x.shape[1]} (original + {encoding_type.upper()})"
                )


def process_dataset_with_graph_encodings(
    dataset: List[Data],
    dataset_name: str,
    output_dir: str,
    verbose: bool = False,
    encoding_type_filter: Optional[str] = None,
) -> None:
    """Process a dataset with graph-level encodings.

    Computes each encoding separately and saves to separate files.

    Args:
        dataset: List of PyG Data objects
        dataset_name: Name of the dataset
        output_dir: Output directory for saved dataset
        verbose: Whether to print progress
    """
    print(
        f"\n🔄 Processing {dataset_name} with graph encodings ({len(dataset)} graphs)..."
    )

    # Log dataset statistics
    print(f"  Dataset info:")
    print(f"    Number of graphs: {len(dataset)}")
    if len(dataset) > 0:
        sample_graph = dataset[0]
        print(
            f"    Sample graph: {sample_graph.num_nodes} nodes, {sample_graph.edge_index.shape[1] if sample_graph.edge_index is not None else 0} edges"
        )
        if sample_graph.x is not None:
            print(f"    Sample features shape: {sample_graph.x.shape}")
        else:
            print(f"    Sample features: None")

    # Define encodings to compute (each separately)
    # Format: (encoding_type, params_dict, description, filename_suffix)
    all_encodings = [
        ("ldp", {}, "LDP (Local Degree Profile)", "ldp"),
        (
            "rwpe",
            {"walk_length": 16},
            "RWPE (Random Walk Positional Encoding)",
            "rwpe_k16",
        ),
        ("lape", {"k": 8}, "LAPE (Laplacian Positional Encoding)", "lape_k8"),
        ("orc", {}, "ORC (Ollivier-Ricci Curvature via LCP)", "orc"),
    ]

    # Filter to only the requested encoding type if specified
    if encoding_type_filter:
        encodings_to_compute = [
            enc for enc in all_encodings if enc[0] == encoding_type_filter
        ]
        if not encodings_to_compute:
            print(
                f"⚠️  Warning: Encoding type '{encoding_type_filter}' not found for graph encodings"
            )
            print(f"   Available types: {[enc[0] for enc in all_encodings]}")
            return
    else:
        encodings_to_compute = all_encodings

    # Setup logging function
    def log(msg: str) -> None:
        """Log message for first graph in verbose mode."""
        if verbose:
            print(msg)

    # Process each encoding type separately
    for (
        encoding_type,
        encoding_params,
        encoding_name,
        filename_suffix,
    ) in encodings_to_compute:
        print(f"\n  📊 Computing {encoding_name}...")
        augmented_dataset: List[Data] = []
        failed_count = 0

        for i, graph in enumerate(tqdm(dataset, desc=f"  {encoding_type.upper()}")):
            try:
                augmented_graph = compute_single_graph_encoding(
                    graph,
                    encoding_type,
                    encoding_params,
                    logger=log if i == 0 else None,
                )
                augmented_dataset.append(augmented_graph)
                if i == 0 and verbose:
                    print(
                        f"  First graph after {encoding_type.upper()}: features shape={augmented_graph.x.shape if augmented_graph.x is not None else 'None'}"
                    )
            except Exception as e:
                failed_count += 1
                error_msg = (
                    f"\n⚠️  Error processing graph {i} with {encoding_type.upper()}: {e}"
                )
                if failed_count <= 5:  # Only print first few errors
                    print(error_msg)
                if verbose and i < 5:
                    import traceback

                    traceback.print_exc()
                # Keep original graph if encoding fails
                augmented_dataset.append(graph)

        if failed_count > 0:
            print(f"  ⚠️  {failed_count} graphs failed {encoding_type.upper()} encoding")

        # Save dataset with this encoding type (using specific filename suffix)
        output_path = os.path.join(output_dir, f"{dataset_name}_g_{filename_suffix}.pt")
        os.makedirs(output_dir, exist_ok=True)
        torch.save(augmented_dataset, output_path)
        print(f"  ✅ Saved {len(augmented_dataset)} graphs to {output_path}")

        # Log final statistics
        if len(augmented_dataset) > 0:
            sample = augmented_dataset[0]
            if sample.x is not None:
                print(
                    f"  Final feature dimensions: {sample.x.shape[1]} (original + {encoding_type.upper()})"
                )


def main() -> None:
    """Main function to compute encodings for all datasets.

    Supports parallel execution by allowing selection of encoding level (hypergraph/graph)
    and encoding type (ldp, frc, orc, rwpe, lape).
    """
    parser = argparse.ArgumentParser(
        description="Compute encodings for graph datasets. "
        "Use --level and --encoding to run specific encodings in parallel."
    )
    parser.add_argument(
        "--level",
        type=str,
        choices=["hypergraph", "graph", "both"],
        default="both",
        help="Encoding level: 'hypergraph', 'graph', or 'both' (default: both)",
    )
    parser.add_argument(
        "--encoding",
        type=str,
        choices=["ldp", "frc", "orc", "rwpe", "lape"],
        default=None,
        help="Specific encoding type to compute (default: all except ORC for hypergraph)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output",
    )

    args = parser.parse_args()

    # Paths
    script_dir = Path(__file__).parent
    repo_root = script_dir.parent
    data_directory = str(repo_root / "graph_datasets")
    output_dir_hg = str(repo_root / "graph_datasets_with_hg_encodings")
    output_dir_g = str(repo_root / "graph_datasets_with_g_encodings")

    print(f"📁 Data directory: {data_directory}")
    print(f"📁 Output directory (hypergraph): {output_dir_hg}")
    print(f"📁 Output directory (graph): {output_dir_g}")
    if args.encoding:
        print(f"🔧 Computing only: {args.encoding}")
    if args.level != "both":
        print(f"🔧 Level: {args.level}")
    print("\n" + "=" * 80)
    print(
        "NOTE: This script computes each encoding separately and saves to separate files."
    )
    print("Hypergraph encodings: {dataset_name}_hg_{encoding_suffix}.pt")
    print("  - ldp: Local Degree Profile")
    print("  - frc: Forman-Ricci Curvature")
    print("  - orc: Ollivier-Ricci Curvature (very slow)")
    print("  - rwpe_we_k20: Random Walk Positional Encoding (WE type, k=20)")
    print("  - lape_normalized_k8: Laplacian Positional Encoding (Normalized, k=8)")
    print("Graph-level encodings: {dataset_name}_g_{encoding_suffix}.pt")
    print("  - ldp: Local Degree Profile")
    print("  - rwpe_k16: Random Walk Positional Encoding (walk_length=16)")
    print("  - lape_k8: Laplacian Positional Encoding (k=8)")
    print("  - orc: Ollivier-Ricci Curvature via LCP")
    print("=" * 80 + "\n")

    # Load all datasets
    datasets = load_all_datasets(data_directory)

    print(f"\n📊 Summary: Loaded {len(datasets)} datasets:")
    for name, dataset in datasets.items():
        if dataset is not None and len(dataset) > 0:
            print(f"  - {name}: {len(dataset)} graphs")
        else:
            print(f"  - {name}: (empty or failed)")

    # Process each dataset
    for dataset_name, dataset in datasets.items():
        if dataset is None or len(dataset) == 0:
            print(f"⚠️  Skipping {dataset_name}: empty dataset")
            continue

        # Process with hypergraph encodings
        if (args.level in ["hypergraph", "both"]) and HYPERGRAPH_ENCODINGS_AVAILABLE:
            process_dataset_with_hypergraph_encodings(
                dataset,
                dataset_name,
                output_dir_hg,
                verbose=args.verbose,
                encoding_type_filter=args.encoding,
            )

        # Process with graph encodings
        if args.level in ["graph", "both"]:
            process_dataset_with_graph_encodings(
                dataset,
                dataset_name,
                output_dir_g,
                verbose=args.verbose,
                encoding_type_filter=args.encoding,
            )

    print("\n🎉 Finished processing all datasets!")


if __name__ == "__main__":
    main()
