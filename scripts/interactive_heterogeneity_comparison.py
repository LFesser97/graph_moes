#!/usr/bin/env python3
"""
Interactive heterogeneity profile comparison tool.

Creates an interactive plotly dashboard with a single dropdown menu that shows
pairs of configurations to compare. Pairs are only created for configurations
that share the same base model (e.g., GCN variants, GIN variants, etc.).
"""

import os
import pickle
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import plotly.graph_objects as go

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

from graph_moes.experiments.track_avg_accuracy import compute_average_per_graph


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


def load_heterogeneity_data(pickle_file: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load graph_dict from pickle and compute average accuracy per graph."""
    with open(pickle_file, "rb") as f:
        data = pickle.load(f)

    # Handle both old and new formats
    if isinstance(data, dict) and "graph_dict" in data:
        graph_dict = data["graph_dict"]
    else:
        graph_dict = data

    graph_indices, average_values = compute_average_per_graph(graph_dict)
    return graph_indices, average_values * 100  # Convert to percentage


def create_config_label(config: dict) -> str:
    """Create a human-readable label for a configuration."""
    # Start with dataset name
    parts = [config["dataset"]]

    # Add layer type
    parts.append(config["layer_type"])

    if config["layer_types"]:
        parts.append(f"({'+'.join(config['layer_types'])})")

    if config["skip_connection"]:
        parts.append("skip")

    if config["normalize_features"]:
        parts.append("norm")

    if config["encoding"]:
        parts.append(f"enc:{config['encoding']}")
    else:
        parts.append("enc:None")

    return " | ".join(parts)


def create_interactive_comparison(
    results_dir: Path, output_file: str = "heterogeneity_comparison.html"
):
    """Create interactive comparison dashboard with two independent dropdown menus."""
    print(f"🔍 Scanning for pickle files in: {results_dir}")

    # Find all pickle files
    pickle_files = list(results_dir.rglob("*_graph_dict.pickle"))

    if not pickle_files:
        print("❌ No pickle files found!")
        sys.exit(1)

    print(f"📁 Found {len(pickle_files)} pickle files\n")

    # Parse all configurations
    configs = []
    for pickle_file in sorted(pickle_files):
        try:
            config = parse_pickle_filename(pickle_file)
            config["label"] = create_config_label(config)
            configs.append(config)
        except Exception as e:
            print(f"⚠️  Error parsing {pickle_file.name}: {e}")

    if not configs:
        print("❌ No valid configurations found!")
        sys.exit(1)

    datasets = sorted(set(c["dataset"] for c in configs))
    print(f"📊 Found {len(configs)} configurations across {len(datasets)} datasets")

    # Pre-load all data
    print("📥 Loading data for all configurations...")
    for config in configs:
        try:
            indices, values = load_heterogeneity_data(config["filepath"])
            config["indices"] = indices
            config["values"] = values
            config["sort_idx"] = np.argsort(values)[::-1]
            config["sorted_values"] = values[config["sort_idx"]]
        except Exception as e:
            print(f"⚠️  Error loading {config['label']}: {e}")
            config["indices"] = None

    # Filter out configs with no data
    configs = [c for c in configs if c.get("indices") is not None]

    if len(configs) < 2:
        print("❌ Need at least 2 valid configurations!")
        sys.exit(1)

    print(f"✅ Loaded {len(configs)} valid configurations\n")

    # Create figure - one trace per config
    fig = go.Figure()
    num_configs = len(configs)

    # Add one trace per config
    for i, config in enumerate(configs):
        sorted_values = config["sorted_values"]

        # Calculate overall average accuracy (averaged over all graphs)
        overall_avg = np.mean(config["values"])
        avg_str = f" (Avg: {overall_avg:.1f}%)"

        # Add average to trace name
        trace_name = config["label"] + avg_str

        fig.add_trace(
            go.Scatter(
                x=np.arange(len(sorted_values)),
                y=sorted_values,
                mode="markers",
                name=trace_name,
                marker=dict(size=5, opacity=0.7),
                visible=False,  # All traces initially hidden
            ),
        )

    # Create dropdown menu with pairs of same-dataset configurations (models can differ)
    # Group configs by dataset only
    from collections import defaultdict

    dataset_groups = defaultdict(list)

    for i, config in enumerate(configs):
        dataset = config["dataset"]
        dataset_groups[dataset].append((i, config))

    # Create dropdown buttons with pairs from same dataset
    dropdown_buttons = []

    # Add "Show All" option
    all_visible = [True] * num_configs
    dropdown_buttons.append(
        dict(
            label="Show All",
            method="update",
            args=[
                {"visible": all_visible},
                {"title": "Heterogeneity Profile Comparison - All Configurations"},
            ],
        )
    )

    # Add separator
    dropdown_buttons.append(
        dict(
            label="──────────",
            method="update",
            args=[{"visible": all_visible}, {}],
        )
    )

    # Create pairs within each dataset (all models within that dataset can be paired)
    for dataset_name in sorted(dataset_groups.keys()):
        group_configs = dataset_groups[dataset_name]

        # Create all pairs within this dataset (models can differ)
        for i in range(len(group_configs)):
            for j in range(i + 1, len(group_configs)):
                idx_i, config_i = group_configs[i]
                idx_j, config_j = group_configs[j]

                visibility = [False] * num_configs
                visibility[idx_i] = True
                visibility[idx_j] = True

                label = f"{config_i['label']} vs {config_j['label']}"

                dropdown_buttons.append(
                    dict(
                        label=label,
                        method="update",
                        args=[
                            {"visible": visibility},
                            {"title": f"Heterogeneity Profile Comparison: {label}"},
                        ],
                    )
                )

    print(
        f"📊 Created {len(dropdown_buttons) - 2} configuration pairs (within same dataset, models can differ)"
    )

    # Update layout
    fig.update_xaxes(title_text="Rank (sorted by accuracy)")
    fig.update_yaxes(title_text="Average Accuracy (%)", range=[0, 105])

    fig.update_layout(
        height=700,
        title_text="Interactive Heterogeneity Profile Comparison<br><sub>Select a pair to compare (same model variants)</sub>",
        showlegend=True,
        hovermode="closest",
        updatemenus=[
            dict(
                buttons=dropdown_buttons,
                direction="down",
                showactive=True,
                x=0.1,
                xanchor="left",
                y=1.02,
                yanchor="top",
                pad=dict(r=10, t=10),
                bgcolor="rgba(200,230,255,0.8)",
            ),
        ],
        annotations=[
            dict(
                text="Select Comparison:",
                x=0.05,
                xref="paper",
                y=1.05,
                yref="paper",
                showarrow=False,
                font=dict(size=12),
            ),
        ],
    )

    # Save to HTML (no custom JavaScript needed - Plotly handles single dropdown natively)
    output_path = results_dir / output_file
    fig.write_html(str(output_path))

    print(f"\n✅ Interactive comparison saved to: {output_path}")
    print(f"   Open in browser to use the dropdown menu")
    print(f"   Select a pair of configurations to compare (both from the same model)")


if __name__ == "__main__":
    results_dir = Path("results_cluster/results")

    if not results_dir.exists():
        print(f"❌ Results directory not found: {results_dir}")
        print(f"   Current directory: {os.getcwd()}")
        sys.exit(1)

    create_interactive_comparison(results_dir)
