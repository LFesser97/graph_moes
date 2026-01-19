#!/usr/bin/env python3
"""
Verify dataset consistency across all model configurations.

For the same dataset, all model/encoding/skip/norm combinations should have
the same number of graphs. This script checks for inconsistencies that might
indicate data processing errors or graph dropping issues.
"""

import pickle
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))


def parse_pickle_filename(filepath: Path) -> dict:
    """Parse pickle filename to extract model configuration."""
    filename = filepath.stem.replace("_graph_dict", "")
    parent_dir = filepath.parent.name

    # Extract num_layers from parent directory
    num_layers_match = re.match(r"(\d+)_layers", parent_dir)
    num_layers = int(num_layers_match.group(1)) if num_layers_match else None

    # Parse the filename
    parts = filename.split("_enc")
    dataset = parts[0].split("_")[0]

    model_part = parts[0].replace(f"{dataset}_", "")
    encoding = parts[1] if len(parts) > 1 else "None"

    # Parse model configuration
    skip_connection = False
    normalize_features = False
    layer_types = None
    router_type = "MLP"
    layer_type = None

    if "_skip" in model_part:
        skip_connection = True
        model_part = model_part.replace("_skip", "")

    if "_norm" in model_part:
        normalize_features = True
        model_part = model_part.replace("_norm", "")

    if model_part.startswith("MoE_"):
        moe_parts = model_part.split("_")
        router_type = moe_parts[1]
        if len(moe_parts) > 2:
            layer_types = moe_parts[2:]
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
        "filepath": str(filepath),
    }


def count_graphs_in_pickle(pickle_file: Path) -> int:
    """Load pickle file and count the number of graphs."""
    try:
        with open(pickle_file, "rb") as f:
            data = pickle.load(f)

        # Handle both old and new formats
        if isinstance(data, dict) and "graph_dict" in data:
            graph_dict = data["graph_dict"]
        else:
            graph_dict = data

        # Count unique graph indices
        return len(graph_dict)
    except Exception as e:
        print(f"   ⚠️  Error loading {pickle_file.name}: {e}")
        return -1


def main():
    """Main function to verify dataset consistency."""
    results_dir = Path("results_cluster/results")

    if not results_dir.exists():
        print(f"❌ Results directory not found: {results_dir}")
        print(f"   Current directory: {Path.cwd()}")
        sys.exit(1)

    print(f"🔍 Scanning for pickle files in: {results_dir}\n")

    # Find all pickle files
    pickle_files = list(results_dir.rglob("*_graph_dict.pickle"))

    if not pickle_files:
        print("❌ No pickle files found!")
        sys.exit(1)

    print(f"📁 Found {len(pickle_files)} pickle files\n")

    # Group by dataset
    dataset_groups = defaultdict(list)

    for pickle_file in sorted(pickle_files):
        try:
            config = parse_pickle_filename(pickle_file)
            dataset_groups[config["dataset"]].append((pickle_file, config))
        except Exception as e:
            print(f"⚠️  Error parsing {pickle_file.name}: {e}")

    print(f"📊 Found {len(dataset_groups)} datasets\n")
    print("=" * 80)

    # Check consistency for each dataset
    all_consistent = True
    inconsistent_datasets = []

    for dataset in sorted(dataset_groups.keys()):
        files_configs = dataset_groups[dataset]
        print(f"\n📦 Dataset: {dataset}")
        print(f"   Configurations: {len(files_configs)}")

        # Count graphs for each configuration
        graph_counts = {}
        config_labels = []

        for pickle_file, config in files_configs:
            num_graphs = count_graphs_in_pickle(pickle_file)
            if num_graphs < 0:
                continue

            # Create config label
            label_parts = [
                config["layer_type"],
                f"enc:{config['encoding'] or 'None'}",
            ]
            if config["skip_connection"]:
                label_parts.append("skip")
            if config["normalize_features"]:
                label_parts.append("norm")
            label = " | ".join(label_parts)

            graph_counts[label] = num_graphs
            config_labels.append((label, num_graphs, pickle_file))

        if not graph_counts:
            print(f"   ⚠️  No valid configurations found!")
            continue

        # Check if all counts are the same
        unique_counts = set(graph_counts.values())
        expected_count = max(graph_counts.values())  # Use the most common count

        if len(unique_counts) == 1:
            print(
                f"   ✅ All {len(config_labels)} configurations have {expected_count} graphs"
            )
        else:
            all_consistent = False
            inconsistent_datasets.append(dataset)
            print(f"   ❌ INCONSISTENCY DETECTED!")
            print(f"   Expected: {expected_count} graphs")
            print(
                f"   Found {len(unique_counts)} different graph counts: {sorted(unique_counts)}"
            )

            # Show details
            print("\n   Configuration details:")
            for label, count, filepath in sorted(config_labels):
                status = "✅" if count == expected_count else "❌"
                print(f"     {status} {label}: {count} graphs ({Path(filepath).name})")

    # Summary
    print("\n" + "=" * 80)
    print("\n📈 Summary:")
    print(f"   Total datasets checked: {len(dataset_groups)}")
    print(
        f"   ✅ Consistent datasets: {len(dataset_groups) - len(inconsistent_datasets)}"
    )
    print(f"   ❌ Inconsistent datasets: {len(inconsistent_datasets)}")

    if inconsistent_datasets:
        print(f"\n⚠️  Inconsistent datasets:")
        for dataset in inconsistent_datasets:
            print(f"   - {dataset}")
        print("\n💡 These may indicate:")
        print("   - Graphs dropped during encoding/processing")
        print("   - Data loading errors")
        print("   - Different test set splits")
        sys.exit(1)
    else:
        print("\n🎉 All datasets are consistent! Every model/encoding combination")
        print("   for the same dataset has the same number of graphs.")


if __name__ == "__main__":
    main()
