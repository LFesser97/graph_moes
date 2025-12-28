#!/bin/bash
# Script to upload the downloaded ogbg-ppa dataset to the Harvard cluster
# Run this after the download_ogbg_ppa.py script completes successfully

echo "🚀 Starting upload of ogbg-ppa dataset to Harvard cluster..."

# Check if the dataset exists locally
if [ ! -d "graph_datasets/ogbg_ppa" ]; then
    echo "❌ Error: ogbg_ppa dataset not found locally. Please run download_ogbg_ppa.py first."
    exit 1
fi

# Check the size of the dataset
DATASET_SIZE=$(du -sh graph_datasets/ogbg_ppa | cut -f1)
echo "📊 Local dataset size: $DATASET_SIZE"

echo "📤 Uploading to Harvard cluster..."
echo "   Target: rpellegrinext@login.rc.fas.harvard.edu:/n/holylabs/LABS/mweber_lab/Everyone/rpellegrin/graph_moes/graph_datasets/"

# Use rsync for efficient transfer (resumes if interrupted)
rsync -avz --progress graph_datasets/ogbg_ppa/ \
    rpellegrinext@login.rc.fas.harvard.edu:/n/holylabs/LABS/mweber_lab/Everyone/rpellegrin/graph_moes/graph_datasets/ogbg_ppa/

if [ $? -eq 0 ]; then
    echo "✅ Upload completed successfully!"
    echo "🎉 You can now run the comprehensive_sweep_parallel_additional_data.sh script on the cluster"
else
    echo "❌ Upload failed. Please check your connection and try again."
    exit 1
fi
