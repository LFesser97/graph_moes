"""Script to plot violin plots of class-distance ratios from TMD analysis."""

import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def load_tmd_results(dataset_name: str, results_dir: str) -> tuple[pd.DataFrame, dict]:
    """Load TMD class-distance ratio results.

    Args:
        dataset_name: Name of the dataset (e.g., 'mutag', 'enzymes')
        results_dir: Directory containing TMD results

    Returns:
        Tuple of (ratios_df, stats_dict)
    """
    ratios_path = os.path.join(results_dir, f"{dataset_name.lower()}_class_ratios.csv")
    stats_path = os.path.join(results_dir, f"{dataset_name.lower()}_tmd_stats.json")

    if not os.path.exists(ratios_path):
        raise FileNotFoundError(f"Class ratios file not found: {ratios_path}")

    ratios_df = pd.read_csv(ratios_path)
    stats_dict = {}
    if os.path.exists(stats_path):
        with open(stats_path, "r", encoding="utf-8") as f:
            stats_dict = json.load(f)

    return ratios_df, stats_dict


def plot_class_distance_ratios_violin(
    ratios_df: pd.DataFrame,
    dataset_name: str,
    stats_dict: dict,
    output_dir: str = "visualizations/tmd",
    grouped_by_class: bool = False,
) -> str:
    """Plot violin plot of class-distance ratios.

    Args:
        ratios_df: DataFrame with columns: graph_index, label, class_distance_ratio
        dataset_name: Name of the dataset
        stats_dict: Statistics dictionary
        output_dir: Output directory for plots
        grouped_by_class: If True, create separate violin plots grouped by class label

    Returns:
        Path to saved plot file
    """
    os.makedirs(output_dir, exist_ok=True)

    # Filter out infinite ratios for plotting
    finite_ratios = ratios_df[np.isfinite(ratios_df["class_distance_ratio"])].copy()
    ratios = finite_ratios["class_distance_ratio"].values

    if grouped_by_class:
        # Grouped violin plot by class
        fig, ax = plt.subplots(figsize=(14, 8))

        # Get unique labels
        unique_labels = sorted(finite_ratios["label"].unique())
        num_classes = len(unique_labels)

        # Prepare data for grouped violin plot
        data_by_class: list[np.ndarray] = [
            finite_ratios[finite_ratios["label"] == label][
                "class_distance_ratio"
            ].values
            for label in unique_labels
        ]

        # Create violin plot
        parts = ax.violinplot(
            data_by_class,
            positions=range(num_classes),
            showmeans=True,
            showmedians=True,
            widths=0.7,
        )

        # Customize violins
        for pc in parts["bodies"]:  # type: ignore[attr-defined]
            pc.set_facecolor("lightblue")
            pc.set_alpha(0.7)

        # Set labels
        ax.set_xticks(range(num_classes))
        ax.set_xticklabels([f"Class {label}" for label in unique_labels])
        ax.set_xlabel("Class Label", fontsize=12)
        ax.set_ylabel("Class-Distance Ratio (ρ)", fontsize=12)
        ax.set_title(
            f"Class-Distance Ratios by Class: {dataset_name.upper()}",
            fontsize=14,
            fontweight="bold",
        )

        # Add horizontal line at ρ = 1
        ax.axhline(
            y=1.0, color="red", linestyle="--", linewidth=2, label="ρ = 1 (threshold)"
        )
        ax.legend()

        # Add grid
        ax.grid(True, alpha=0.3, linestyle="--", axis="y")

        # Add statistics text
        stats_text = (
            f"Mean: {stats_dict.get('mean', np.nan):.3f} | "
            f"Median: {stats_dict.get('median', np.nan):.3f} | "
            f"Std: {stats_dict.get('std', np.nan):.3f}\n"
            f"Easy (ρ < 1): {stats_dict.get('num_easy', 0)} | "
            f"Hard (ρ > 1): {stats_dict.get('num_hard', 0)} | "
            f"Ambiguous (ρ ≈ 1): {stats_dict.get('num_ambiguous', 0)}"
        )
        ax.text(
            0.5,
            -0.1,
            stats_text,
            transform=ax.transAxes,
            ha="center",
            fontsize=9,
            style="italic",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.3),
        )

        plot_path = os.path.join(
            output_dir, f"{dataset_name.lower()}_class_ratios_violin_grouped.png"
        )

    else:
        # Single violin plot (overall distribution)
        fig, ax = plt.subplots(figsize=(10, 6))

        # Create violin plot
        parts = ax.violinplot(
            [ratios],
            positions=[0],
            showmeans=True,
            showmedians=True,
            widths=0.5,
        )

        # Customize violin
        for pc in parts["bodies"]:  # type: ignore[attr-defined]
            pc.set_facecolor("lightblue")
            pc.set_alpha(0.7)

        # Set labels
        ax.set_xticks([0])
        ax.set_xticklabels([dataset_name.upper()])
        ax.set_ylabel("Class-Distance Ratio (ρ)", fontsize=12)
        ax.set_title(
            f"Class-Distance Ratio Distribution: {dataset_name.upper()}",
            fontsize=14,
            fontweight="bold",
        )

        # Add horizontal line at ρ = 1
        ax.axhline(
            y=1.0, color="red", linestyle="--", linewidth=2, label="ρ = 1 (threshold)"
        )
        ax.legend()

        # Add grid
        ax.grid(True, alpha=0.3, linestyle="--", axis="y")

        # Add statistics text
        stats_text = (
            f"Total graphs: {len(ratios)} | "
            f"Mean: {stats_dict.get('mean', np.nan):.3f} | "
            f"Median: {stats_dict.get('median', np.nan):.3f} | "
            f"Std: {stats_dict.get('std', np.nan):.3f}\n"
            f"Easy (ρ < 1): {stats_dict.get('num_easy', 0)} | "
            f"Hard (ρ > 1): {stats_dict.get('num_hard', 0)} | "
            f"Ambiguous (ρ ≈ 1): {stats_dict.get('num_ambiguous', 0)}"
        )
        ax.text(
            0.5,
            -0.1,
            stats_text,
            transform=ax.transAxes,
            ha="center",
            fontsize=9,
            style="italic",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.3),
        )

        plot_path = os.path.join(
            output_dir, f"{dataset_name.lower()}_class_ratios_violin.png"
        )

    plt.tight_layout()
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return plot_path


def main() -> None:
    """Main function to plot class-distance ratio violin plots."""
    parser = argparse.ArgumentParser(
        description="Plot violin plots of class-distance ratios from TMD analysis"
    )
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="+",
        default=["mutag", "enzymes"],
        help="Dataset names to plot (default: mutag enzymes)",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="tmd_results",
        help="Directory containing TMD results (default: tmd_results)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="visualizations/tmd",
        help="Output directory for plots (default: visualizations/tmd)",
    )
    parser.add_argument(
        "--grouped",
        action="store_true",
        help="Create grouped violin plots by class label",
    )

    args = parser.parse_args()

    print("🎨 Creating violin plots for class-distance ratios...\n")

    for dataset_name in args.datasets:
        print(f"📊 Processing {dataset_name.upper()}...")

        try:
            # Load results
            ratios_df, stats_dict = load_tmd_results(dataset_name, args.results_dir)

            print(f"   Loaded {len(ratios_df)} graphs")
            print(f"   Classes: {sorted(ratios_df['label'].unique())}")

            # Create single violin plot
            plot_path = plot_class_distance_ratios_violin(
                ratios_df,
                dataset_name,
                stats_dict,
                output_dir=args.output_dir,
                grouped_by_class=False,
            )
            print(f"   ✅ Created violin plot: {plot_path}")

            # Create grouped violin plot if requested
            if args.grouped:
                grouped_path = plot_class_distance_ratios_violin(
                    ratios_df,
                    dataset_name,
                    stats_dict,
                    output_dir=args.output_dir,
                    grouped_by_class=True,
                )
                print(f"   ✅ Created grouped violin plot: {grouped_path}")

            print()

        except FileNotFoundError as e:
            print(f"   ❌ Error: {e}")
            print()
            continue
        except (ValueError, KeyError, pd.errors.EmptyDataError, OSError) as e:
            print(f"   ❌ Error processing {dataset_name}: {e}")
            import traceback

            traceback.print_exc()
            print()
            continue

    print("🎉 All plots created!")


if __name__ == "__main__":
    main()
