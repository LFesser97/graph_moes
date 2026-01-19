#!/usr/bin/env python3
"""
Download Peptides-func dataset (LRGB dataset).

This script downloads the Peptides-func dataset from PyTorch Geometric's LRGBDataset
and saves it in the format expected by run_graph_classification.py:
- data/peptidesfunc/train.pt
- data/peptidesfunc/val.pt
- data/peptidesfunc/test.pt
"""

import os
import sys
from pathlib import Path

import torch
from torch_geometric.datasets import LRGBDataset

# Set project root
project_root = Path(__file__).parent.parent
data_directory = project_root / "data"

# Create data directory if it doesn't exist
os.makedirs(data_directory, exist_ok=True)

print(f"📦 Downloading Peptides-func dataset...")
print(f"   Data directory: {data_directory}")

try:
    # Download Peptides-func dataset using PyTorch Geometric
    # LRGBDataset will automatically download and cache the dataset
    print("   ⏳ Loading Peptides-func (this may download the dataset)...")
    dataset = LRGBDataset(root=data_directory, name="Peptides-func")

    print(f"   ✅ Dataset loaded: {len(dataset)} graphs")

    # LRGBDataset downloads split files to the raw directory
    # But processed directory has the correct splits
    # LRGBDataset downloads split files to the raw directory
    # The raw files contain the actual train/val/test splits in tuple format
    peptides_lrgb_raw = data_directory / "Peptides-func" / "raw"

    raw_train = peptides_lrgb_raw / "train.pt"
    raw_val = peptides_lrgb_raw / "val.pt"
    raw_test = peptides_lrgb_raw / "test.pt"

    # Create peptidesfunc directory (expected location for the code)
    peptides_func_dir = data_directory / "peptidesfunc"
    os.makedirs(peptides_func_dir, exist_ok=True)

    train_path = peptides_func_dir / "train.pt"
    val_path = peptides_func_dir / "val.pt"
    test_path = peptides_func_dir / "test.pt"

    # Load from raw files (weights_only=False for compatibility with PyTorch 2.6+)
    if raw_train.exists() and raw_val.exists() and raw_test.exists():
        print("   ✅ Found raw split files in Peptides-func/raw/")
        print("   ⏳ Loading split files...")

        # Load raw split files (they're in tuple format already)
        # Use weights_only=False to allow loading PyG Data objects if present
        train_data = torch.load(raw_train, weights_only=False)
        val_data = torch.load(raw_val, weights_only=False)
        test_data = torch.load(raw_test, weights_only=False)

        print(
            f"   📊 Raw splits: Train={len(train_data)}, Val={len(val_data)}, Test={len(test_data)}"
        )

    else:
        # Fallback: Extract from dataset object if raw files don't exist
        print("   ⚠️  Raw split files not found, extracting from dataset...")

        # Convert PyG Data objects to tuple format expected by _convert_lrgb
        # Format: (x, edge_attr, edge_index, y)
        def data_to_tuple(data):
            """Convert PyG Data object to tuple format (x, edge_attr, edge_index, y)."""
            x = data.x
            edge_attr = (
                data.edge_attr
                if hasattr(data, "edge_attr") and data.edge_attr is not None
                else None
            )
            edge_index = data.edge_index
            y = data.y
            return (x, edge_attr, edge_index, y)

        # Use standard split (shouldn't happen with LRGB but as fallback)
        print("   ⚠️  Using dataset indices (fallback - not ideal for LRGB)")
        n = len(dataset)
        train_idx = list(range(0, int(0.7 * n)))
        val_idx = list(range(int(0.7 * n), int(0.85 * n)))
        test_idx = list(range(int(0.85 * n), n))

        train_data = [data_to_tuple(dataset[i]) for i in train_idx]
        val_data = [data_to_tuple(dataset[i]) for i in val_idx]
        test_data = [data_to_tuple(dataset[i]) for i in test_idx]

    print(
        f"   📊 Final splits: Train={len(train_data)}, Val={len(val_data)}, Test={len(test_data)}"
    )

    # Save as .pt files in the expected location
    print(f"   💾 Saving train split to: {train_path}")
    torch.save(train_data, train_path)

    print(f"   💾 Saving val split to: {val_path}")
    torch.save(val_data, val_path)

    print(f"   💾 Saving test split to: {test_path}")
    torch.save(test_data, test_path)

    print(f"\n✅ Peptides-func dataset downloaded and saved successfully!")
    print(f"   Location: {peptides_func_dir}")
    print(
        f"   Files: train.pt ({len(train_data)} graphs), val.pt ({len(val_data)} graphs), test.pt ({len(test_data)} graphs)"
    )

except Exception as e:
    print(f"\n❌ Failed to download Peptides-func: {e}")
    import traceback

    traceback.print_exc()
    sys.exit(1)
