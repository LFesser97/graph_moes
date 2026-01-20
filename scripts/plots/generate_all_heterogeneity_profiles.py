#!/usr/bin/env python3
"""
Generate heterogeneity profiles for all pickle files in results directory.

This script scans for all graph_dict pickle files and generates heterogeneity profiles
(by_index and by_accuracy plots) for each model/encoding/skip/normalize combination.
"""

import os
import re
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

from graph_moes.experiments.track_avg_accuracy import load_and_plot_average_per_graph


def parse_pickle_filename(filepath: Path) -> dict:
    """
    Parse pickle filename to extract model configuration.

    Handles multiple filename formats:
    - Old: {dataset}_{model}_skip_enc{encoding} or {dataset}_{model}_enc{encoding}
    - New: {dataset}_{model}_norm_enc{encoding} or {dataset}_{model}_skip_enc{encoding}
    - Encoding formats: encNone, encg_ldp, encg_rwpe_k16, enchg_orc, enchg_lape_normalized_k8

    Examples:
    - enzymes_GCN_skip_enchg_orc_graph_dict.pickle
    - enzymes_MLP_encNone_graph_dict.pickle
    - enzymes_MoE_MLP_GCN_Unitary_encg_ldp_graph_dict.pickle
    - collab_GCN_norm_enchg_lape_normalized_k8_graph_dict.pickle

    Returns dict with: dataset, layer_type, encoding, num_layers, skip_connection, normalize_features, layer_types, router_type
    """
    filename = filepath.stem.replace("_graph_dict", "")
    parent_dir = filepath.parent.name

    # Extract num_layers from parent directory (e.g., "4_layers" -> 4)
    num_layers_match = re.match(r"(\d+)_layers", parent_dir)
    num_layers = int(num_layers_match.group(1)) if num_layers_match else None

    # Parse the filename - split on _enc to separate model and encoding
    # Format: {dataset}_{model}[_skip][_norm]_enc{encoding}
    parts = filename.split("_enc")

    if len(parts) < 2:
        # No encoding found - assume None
        encoding = "None"
        model_part_full = filename
    else:
        # Encoding part is after _enc (removed by split)
        encoding_part = parts[1] if len(parts) > 1 else "None"
        # Encoding formats: None, g_ldp, g_rwpe_k16, hg_orc, hg_lape_normalized_k8
        if encoding_part == "None":
            encoding = "None"
        else:
            encoding = encoding_part  # Already in correct format (g_ldp, hg_lape_normalized_k8, etc.)
        model_part_full = parts[0]

    # Extract dataset (first part)
    dataset = model_part_full.split("_")[0]
    model_part = model_part_full.replace(f"{dataset}_", "")

    # Parse model configuration flags
    skip_connection = False
    normalize_features = False
    layer_types = None
    router_type = "MLP"
    layer_type = None

    # Check for skip connection (format: _skip or _skip_)
    if "_skip" in model_part:
        skip_connection = True
        model_part = model_part.replace("_skip", "")

    # Check for normalization (format: _norm or _norm_)
    if "_norm" in model_part:
        normalize_features = True
        model_part = model_part.replace("_norm", "")

    # Clean up any double underscores or trailing underscores
    model_part = model_part.replace("__", "_").strip("_")

    # Check for MoE (has router type and expert types)
    if model_part.startswith("MoE_"):
        moe_parts = model_part.split("_")
        router_type = moe_parts[1]  # e.g., "MLP" or "GNN"
        # Expert types are everything after router_type
        if len(moe_parts) > 2:
            layer_types = moe_parts[2:]  # e.g., ["GCN", "Unitary"]
        layer_type = "MoE"
    else:
        layer_type = model_part

    return {
        "dataset": dataset,
        "layer_type": layer_type,
        "encoding": encoding if encoding != "None" else None,
        "num_layers": num_layers,
        "skip_connection": skip_connection,
        "normalize_features": normalize_features,
        "layer_types": layer_types,
        "router_type": router_type,
    }


def main():
    """Generate heterogeneity profiles for all pickle files."""
    # Define results directory
    results_dir = Path("results_cluster/results")

    if not results_dir.exists():
        print(f"❌ Results directory not found: {results_dir}")
        print(f"   Current directory: {os.getcwd()}")
        sys.exit(1)

    print(f"🔍 Scanning for pickle files in: {results_dir}")

    # Find all pickle files
    pickle_files = list(results_dir.rglob("*_graph_dict.pickle"))

    if not pickle_files:
        print("❌ No pickle files found!")
        sys.exit(1)

    print(f"📁 Found {len(pickle_files)} pickle files to process\n")

    # Process each pickle file
    processed = 0
    skipped = 0
    errors = 0

    for pickle_file in sorted(pickle_files):
        try:
            # Parse filename
            config = parse_pickle_filename(pickle_file)

            # Check if plots already exist
            dataset = config["dataset"]
            layer_type = config["layer_type"]
            encoding = config["encoding"]
            skip_str = "skip_true" if config["skip_connection"] else "skip_false"
            norm_str = "norm_true" if config["normalize_features"] else "norm_false"
            encoding_str = encoding if encoding else "none"
            encoding_suffix = f"_encodings_{encoding_str}"

            # Build model name
            if config["layer_types"]:
                expert_combo = "_".join(config["layer_types"])
                detailed_model_name = (
                    f"{layer_type}_{config['router_type']}_{expert_combo}"
                )
            else:
                detailed_model_name = layer_type

            plot_filename_base = f"{dataset}_{detailed_model_name}_{skip_str}_{norm_str}{encoding_suffix}"
            by_index_path = results_dir / f"{plot_filename_base}_by_index.png"
            by_accuracy_path = results_dir / f"{plot_filename_base}_by_accuracy.png"

            # Skip if both plots already exist
            if by_index_path.exists() and by_accuracy_path.exists():
                print(f"⏭️  Skipping {pickle_file.name} (plots already exist)")
                skipped += 1
                continue

            print(
                f"📊 Processing: {dataset} | {layer_type} | {encoding or 'None'} | {config['num_layers']} layers"
            )

            # Generate plots
            original_plot_path, sorted_plot_path = load_and_plot_average_per_graph(
                str(pickle_file),
                dataset_name=dataset,
                layer_type=layer_type,
                encoding=encoding,
                num_layers=config["num_layers"],
                task_type="classification",
                output_dir=str(results_dir),
                layer_types=config["layer_types"],
                router_type=config["router_type"],
                skip_connection=config["skip_connection"],
                normalize_features=config["normalize_features"],
            )

            if original_plot_path and sorted_plot_path:
                print(f"   ✅ Generated: {Path(original_plot_path).name}")
                print(f"   ✅ Generated: {Path(sorted_plot_path).name}")
                processed += 1
            else:
                print(f"   ⚠️  Failed to generate plots")
                errors += 1

        except Exception as e:
            print(f"   ❌ Error processing {pickle_file.name}: {e}")
            import traceback

            traceback.print_exc()
            errors += 1

    print("\n📈 Summary:")
    print(f"   ✅ Successfully processed: {processed} files")
    print(f"   ⏭️  Skipped (already exist): {skipped} files")
    print(f"   ❌ Errors: {errors} files")
    print("\n🎉 Done! All heterogeneity profiles have been generated.")


if __name__ == "__main__":
    main()
