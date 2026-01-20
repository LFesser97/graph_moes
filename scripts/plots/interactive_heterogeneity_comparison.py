#!/usr/bin/env python3
"""
Interactive heterogeneity profile comparison tool.

Creates interactive plotly dashboards where most parameters are fixed and one parameter
varies. Generates separate HTML files for:
1. Varying encodings (fix model, dataset, skip, norm)
2. Varying models (fix encoding, dataset, skip, norm)
3. Varying skip connections (fix model, encoding, dataset, norm)
4. Varying normalization (fix model, encoding, dataset, skip)
"""

import os
import pickle
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import plotly.graph_objects as go

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

from graph_moes.experiments.track_avg_accuracy import compute_average_per_graph


def parse_pickle_filename(filepath: Path) -> dict:
    """
    Parse pickle filename to extract model configuration.

    Handles multiple filename formats:
    - Old: {dataset}_{model}_skip_enc{encoding} or {dataset}_{model}_enc{encoding}
    - New: {dataset}_{model}_norm_enc{encoding} or {dataset}_{model}_skip_enc{encoding}
    - Encoding formats: encNone, encg_ldp, encg_rwpe_k16, enchg_orc, enchg_lape_normalized_k8

    Examples:
    - collab_GCN_skip_encg_ldp -> dataset=collab, layer_type=GCN, skip=True, encoding=g_ldp
    - collab_GCN_norm_enchg_lape_normalized_k8 -> dataset=collab, layer_type=GCN, norm=True, encoding=hg_lape_normalized_k8
    - enzymes_MLP_encNone -> dataset=enzymes, layer_type=MLP, encoding=None
    """
    filename = filepath.stem.replace("_graph_dict", "")
    parent_dir = filepath.parent.name

    # Extract num_layers from parent directory
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

    # Parse MoE models
    if model_part.startswith("MoE_"):
        moe_parts = model_part.split("_")
        router_type = moe_parts[1]  # e.g., "MLP" or "GNN"
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


def create_config_label(config: dict, highlight_var: Optional[str] = None) -> str:
    """
    Create a human-readable label for a configuration.

    Args:
        config: Configuration dictionary
        highlight_var: Optional parameter to highlight (e.g., "encoding", "layer_type")
    """
    # Start with dataset name
    parts = [config["dataset"]]

    # Add num_layers if available (helps distinguish duplicates)
    if config.get("num_layers"):
        parts.append(f"L{config['num_layers']}")

    # Add layer type
    layer_label = config["layer_type"]
    if config["layer_types"]:
        layer_label += f"({'+'.join(config['layer_types'])})"
    if highlight_var == "layer_type":
        layer_label = f"[{layer_label}]"
    parts.append(layer_label)

    # Skip connection
    skip_label = "skip" if config["skip_connection"] else "no-skip"
    if highlight_var == "skip_connection":
        skip_label = f"[{skip_label}]"
    parts.append(skip_label)

    # Normalization
    norm_label = "norm" if config["normalize_features"] else "no-norm"
    if highlight_var == "normalize_features":
        norm_label = f"[{norm_label}]"
    parts.append(norm_label)

    # Encoding
    enc_label = f"enc:{config['encoding']}" if config["encoding"] else "enc:None"
    if highlight_var == "encoding":
        enc_label = f"[{enc_label}]"
    parts.append(enc_label)

    return " | ".join(parts)


def get_model_key(config: dict) -> str:
    """Create a unique key for model type (including MoE details)."""
    if config["layer_type"] == "MoE" and config["layer_types"]:
        router = config.get("router_type", "MLP")
        experts = "+".join(sorted(config["layer_types"]))
        return f"MoE_{router}_{experts}"
    return config["layer_type"]


def get_grouping_key(config: dict, vary_param: str) -> tuple:
    """
    Get grouping key for configurations that should be compared together.
    All parameters except vary_param should match (including num_layers).

    Returns a tuple that can be used as a dictionary key.
    """
    key_parts = []

    # Always include dataset and num_layers (same model with different depths shouldn't be grouped)
    key_parts.append(("dataset", config["dataset"]))
    key_parts.append(("num_layers", config.get("num_layers")))

    # Include model details (layer_type, layer_types, router_type) unless varying model
    if vary_param != "layer_type":
        model_key = get_model_key(config)
        key_parts.append(("model", model_key))

    # Include skip connection unless varying it
    if vary_param != "skip_connection":
        key_parts.append(("skip", config["skip_connection"]))

    # Include normalization unless varying it
    if vary_param != "normalize_features":
        key_parts.append(("norm", config["normalize_features"]))

    # Include encoding unless varying it
    if vary_param != "encoding":
        enc = config["encoding"] if config["encoding"] else "None"
        key_parts.append(("encoding", enc))

    # Convert to tuple for hashing
    return tuple(sorted(key_parts))


def create_interactive_comparison_by_variable(
    configs: List[dict],
    vary_param: str,
    results_dir: Path,
    output_file: str,
) -> None:
    """
    Create interactive comparison dashboard where one parameter varies.

    Args:
        configs: List of configuration dictionaries with loaded data
        vary_param: Parameter to vary ("encoding", "layer_type", "skip_connection", "normalize_features")
        results_dir: Directory to save output
        output_file: Output filename
    """
    param_labels = {
        "encoding": "Encoding",
        "layer_type": "Model",
        "skip_connection": "Skip Connections",
        "normalize_features": "Normalization",
    }

    print(f"\n📊 Creating comparison by {param_labels[vary_param]}...")

    # Group configurations by all parameters except the varying one
    groups = defaultdict(list)
    for i, config in enumerate(configs):
        group_key = get_grouping_key(config, vary_param)
        groups[group_key].append((i, config))

    # Filter groups to only those with at least 2 variations
    valid_groups = {k: v for k, v in groups.items() if len(v) >= 2}

    if not valid_groups:
        print(
            f"   ⚠️  No groups with multiple {param_labels[vary_param]} variations found"
        )
        return

    print(f"   Found {len(valid_groups)} groups with multiple variations")

    # Create figure
    fig = go.Figure()
    num_configs = len(configs)

    # Add one trace per config
    for i, config in enumerate(configs):
        sorted_values = config["sorted_values"]
        overall_avg = np.mean(config["values"])
        overall_std = np.std(config["values"])
        avg_str = f" (Avg: {overall_avg:.1f}% ± {overall_std:.1f}%)"

        # Create label highlighting the varying parameter
        trace_name = create_config_label(config, highlight_var=vary_param) + avg_str

        fig.add_trace(
            go.Scatter(
                x=np.arange(len(sorted_values)),
                y=sorted_values,
                mode="lines+markers",
                name=trace_name,
                marker=dict(size=4, opacity=0.8),
                line=dict(width=1.5),
                visible=False,
            ),
        )

    # Create dropdown buttons
    dropdown_buttons = []

    # Add "Show All" option
    all_visible = [True] * num_configs
    dropdown_buttons.append(
        dict(
            label="Show All",
            method="update",
            args=[
                {"visible": all_visible},
                {
                    "title": f"Heterogeneity Profiles - All Configurations (varying {param_labels[vary_param]})"
                },
            ],
        )
    )

    dropdown_buttons.append(
        dict(
            label="──────────",
            method="update",
            args=[{"visible": all_visible}, {}],
        )
    )

    # Create buttons for each group
    for group_key, group_configs in sorted(valid_groups.items()):
        # Extract fixed parameters from group key for label
        fixed_params = dict(group_key)
        dataset = fixed_params.get("dataset", "unknown")

        # Create label showing fixed parameters
        label_parts = [f"{dataset}"]
        if "num_layers" in fixed_params and fixed_params["num_layers"]:
            label_parts.append(f"L{fixed_params['num_layers']}")
        if "model" in fixed_params:
            label_parts.append(f"Model: {fixed_params['model']}")
        if "skip" in fixed_params:
            skip_val = "skip" if fixed_params["skip"] else "no-skip"
            label_parts.append(skip_val)
        if "norm" in fixed_params:
            norm_val = "norm" if fixed_params["norm"] else "no-norm"
            label_parts.append(norm_val)
        if "encoding" in fixed_params:
            enc = fixed_params["encoding"]
            label_parts.append(f"enc:{enc}")

        # Show varying parameter values
        var_values = []
        visibility = [False] * num_configs
        for idx, config in group_configs:
            visibility[idx] = True
            if vary_param == "encoding":
                var_val = config["encoding"] if config["encoding"] else "None"
                var_values.append(var_val)
            elif vary_param == "layer_type":
                model_key = get_model_key(config)
                var_values.append(model_key)
            elif vary_param == "skip_connection":
                var_val = "skip" if config["skip_connection"] else "no-skip"
                var_values.append(var_val)
            elif vary_param == "normalize_features":
                var_val = "norm" if config["normalize_features"] else "no-norm"
                var_values.append(var_val)

        var_str = ", ".join(sorted(set(var_values)))
        label = (
            " | ".join(label_parts) + f" → Vary {param_labels[vary_param]}: [{var_str}]"
        )

        dropdown_buttons.append(
            dict(
                label=label,
                method="update",
                args=[
                    {"visible": visibility},
                    {"title": f"Heterogeneity Profiles: {label}"},
                ],
            )
        )

    print(f"   Created {len(dropdown_buttons) - 2} comparison groups")

    # Update layout
    fig.update_xaxes(title_text="Rank (sorted by accuracy)")
    fig.update_yaxes(title_text="Average Accuracy (%)", range=[0, 105])

    fig.update_layout(
        height=700,
        title_text=f"Heterogeneity Profile Comparison - Varying {param_labels[vary_param]}<br><sub>Select a group to compare variations</sub>",
        showlegend=True,
        hovermode="closest",
        updatemenus=[
            dict(
                buttons=dropdown_buttons,
                direction="down",
                showactive=True,
                x=1.0,
                xanchor="right",
                y=1.02,
                yanchor="top",
                pad=dict(r=10, t=10),
                bgcolor="rgba(200,230,255,0.8)",
            ),
        ],
        annotations=[
            dict(
                text=f"Select Group (varying {param_labels[vary_param]}):",
                x=0.95,
                xref="paper",
                y=1.05,
                yref="paper",
                showarrow=False,
                font=dict(size=12),
                align="right",
            ),
        ],
    )

    # Save to HTML
    output_path = results_dir / output_file
    fig.write_html(str(output_path))
    print(f"   ✅ Saved to: {output_path}")


def generate_all_comparisons(results_dir: Path) -> None:
    """Generate all comparison HTML files for different varying parameters."""
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
            configs.append(config)
        except Exception as e:
            print(f"⚠️  Error parsing {pickle_file.name}: {e}")

    if not configs:
        print("❌ No valid configurations found!")
        sys.exit(1)

    # Deduplicate: create unique key for each config and keep only one instance
    # Prefer files not in nested results/results directory
    # Sort configs so non-nested files are processed first
    configs.sort(
        key=lambda c: (
            "results/results" in str(Path(c["filepath"])),
            c["filepath"],
        )
    )

    seen_configs = {}
    skipped_nested = 0
    for config in configs:
        # Create unique key based on all configuration parameters
        config_key = (
            config["dataset"],
            config["layer_type"],
            config.get("encoding") or "None",
            config.get("num_layers"),
            config.get("skip_connection", False),
            config.get("normalize_features", False),
            config.get("router_type", "MLP"),
            (
                tuple(sorted(config.get("layer_types", [])))
                if config.get("layer_types")
                else None
            ),
        )

        # If we haven't seen this config, store it
        if config_key not in seen_configs:
            seen_configs[config_key] = config
        else:
            # Already have this config - skip if current is nested
            current_is_nested = "results/results" in str(Path(config["filepath"]))
            existing_is_nested = "results/results" in str(
                Path(seen_configs[config_key]["filepath"])
            )

            if current_is_nested and not existing_is_nested:
                # Current is nested, existing is not - skip current (keep existing)
                skipped_nested += 1
                continue
            elif not current_is_nested and existing_is_nested:
                # Current is not nested, existing is - replace
                seen_configs[config_key] = config
            # else: both nested or both not nested - keep first (which is non-nested due to sorting)

    configs = list(seen_configs.values())
    print(
        f"📊 After deduplication: {len(configs)} unique configurations (skipped {skipped_nested} nested duplicates)"
    )

    # Verify no duplicates by checking for duplicate keys
    verify_keys = set()
    duplicate_count = 0
    for config in configs:
        verify_key = (
            config["dataset"],
            config["layer_type"],
            config.get("encoding") or "None",
            config.get("num_layers"),
            config.get("skip_connection", False),
            config.get("normalize_features", False),
            config.get("router_type", "MLP"),
            (
                tuple(sorted(config.get("layer_types", [])))
                if config.get("layer_types")
                else None
            ),
        )
        if verify_key in verify_keys:
            duplicate_count += 1
        verify_keys.add(verify_key)

    if duplicate_count > 0:
        print(f"⚠️  Warning: Found {duplicate_count} duplicates after deduplication!")
    else:
        print(f"✅ Verified: No duplicates found")

    datasets = sorted(set(c["dataset"] for c in configs))
    print(f"📊 Found {len(configs)} configurations across {len(datasets)} datasets")

    # Pre-load all data
    print("\n📥 Loading data for all configurations...")
    for config in configs:
        try:
            indices, values = load_heterogeneity_data(config["filepath"])
            config["indices"] = indices
            config["values"] = values
            config["sort_idx"] = np.argsort(values)[::-1]
            config["sorted_values"] = values[config["sort_idx"]]
        except Exception as e:
            filepath = Path(config["filepath"])
            print(f"⚠️  Error loading {filepath.name}: {e}")
            config["indices"] = None

    # Filter out configs with no data
    configs = [c for c in configs if c.get("indices") is not None]

    if len(configs) < 2:
        print("❌ Need at least 2 valid configurations!")
        sys.exit(1)

    print(f"✅ Loaded {len(configs)} valid configurations\n")

    # Generate comparisons for each varying parameter
    vary_params = [
        ("encoding", "heterogeneity_by_encoding.html"),
        ("layer_type", "heterogeneity_by_model.html"),
        ("skip_connection", "heterogeneity_by_skip.html"),
        ("normalize_features", "heterogeneity_by_normalize.html"),
    ]

    for vary_param, output_file in vary_params:
        create_interactive_comparison_by_variable(
            configs, vary_param, results_dir, output_file
        )

    print("\n🎉 All comparison HTML files generated!")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate interactive heterogeneity profile comparisons"
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="results_cluster/results",
        help="Path to results directory containing pickle files",
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)

    if not results_dir.exists():
        print(f"❌ Results directory not found: {results_dir}")
        print(f"   Current directory: {os.getcwd()}")
        sys.exit(1)

    generate_all_comparisons(results_dir)
