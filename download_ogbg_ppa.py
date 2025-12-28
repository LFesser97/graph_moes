#!/usr/bin/env python3
"""
Script to download ogbg-ppa dataset locally.
This will download the ~2.79GB dataset to avoid interactive prompts.
"""

import os
import sys
from pathlib import Path

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))

# Set up data directory
data_directory = str(Path(__file__).parent / "graph_datasets")
os.makedirs(data_directory, exist_ok=True)

print(f"📁 Using data directory: {data_directory}")

try:
    from ogb.graphproppred import PygGraphPropPredDataset
    from ogb.utils.url import decide_download
    import ogb.utils.url as ogb_url
    print("📦 OGB library loaded successfully")
except ImportError as e:
    print(f"❌ Failed to import OGB: {e}")
    sys.exit(1)

# Monkey patch the decide_download function to automatically approve large downloads
def auto_decide_download(url):
    """Automatically approve downloads without prompting."""
    d = ogb_url.ur.urlopen(url)
    size = int(d.info()["Content-Length"])/ogb_url.GBFACTOR
    print(f"This will download approximately {size:.2f}GB. Auto-approving...")
    return True

# Replace the original function
ogb_url.decide_download = auto_decide_download

print("🔄 Starting download of ogbg-ppa dataset...")
print("This will download approximately 2.79GB. Auto-approving download...")

try:
    # Download the dataset
    dataset = PygGraphPropPredDataset(name="ogbg-ppa", root=data_directory)
    print(f"✅ Successfully downloaded ogbg-ppa dataset with {len(dataset)} graphs")

    # Verify the data is accessible
    sample_graph = dataset[0]
    print(f"📊 Sample graph has {sample_graph.num_nodes} nodes and {sample_graph.num_edges} edges")

except Exception as e:
    print(f"❌ Failed to download ogbg-ppa dataset: {e}")
    sys.exit(1)

print("🎉 Download completed successfully!")
print(f"📁 Dataset saved to: {data_directory}/ogbg_ppa/")
