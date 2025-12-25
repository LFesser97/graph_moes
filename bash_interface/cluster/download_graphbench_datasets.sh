#!/bin/bash
# ============================================================================
# Download GraphBench Datasets on Cluster
# ============================================================================
# This script downloads GraphBench datasets once on the cluster so that
# experiment scripts don't need to download them repeatedly.
#
# It handles corrupted downloads by deleting and retrying, and checks if
# datasets already exist before downloading.
#
# Usage: sbatch download_graphbench_datasets.sh
#        or: bash download_graphbench_datasets.sh (interactive)
# ============================================================================

#SBATCH --job-name=download_graphbench
#SBATCH --ntasks=1
#SBATCH --time=4:00:00           # 4 hours should be enough for downloads
#SBATCH --mem=128GB              # Increased memory for large datasets (er_large, etc.)
#SBATCH --output=logs_comprehensive/download_graphbench_%j.log
#SBATCH --partition=mweber_gpu   # Can run on GPU partition or general
#SBATCH --gpus=0                 # No GPU needed for downloads

# Function to log messages with timestamp
log_message() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

log_message "🚀 Starting GraphBench dataset download on cluster..."

# Set environment path
export CONDA_ENVS_PATH=/n/holylabs/LABS/mweber_lab/Everyone/rpellegrin/conda/envs
export CONDA_PKGS_DIRS=/n/holylabs/LABS/mweber_lab/Everyone/rpellegrin/conda/pkgs

# Load Python module
log_message "📦 Loading Python module..."
module load python/3.10.12-fasrc01 || {
    log_message "⚠️  Failed to load python module, trying alternative..."
    module load python/3.10-fasrc01 || {
        log_message "❌ Failed to load Python module"
        exit 1
    }
}

# Initialize conda/mamba
log_message "🔧 Initializing conda/mamba..."
if [ -f "$(conda info --base 2>/dev/null)/etc/profile.d/conda.sh" ]; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
    log_message "✅ Conda initialized"
elif [ -f "/n/sw/Mambaforge-23.3.1-1/etc/profile.d/conda.sh" ]; then
    source "/n/sw/Mambaforge-23.3.1-1/etc/profile.d/conda.sh"
    log_message "✅ Conda initialized from Mambaforge path"
else
    log_message "⚠️  Could not find conda.sh"
fi

# Activate environment
ENV_NAME="moe_fresh"
log_message "🔧 Activating $ENV_NAME environment..."

# Use manual PATH setup (most reliable)
if [ -d "$CONDA_ENVS_PATH/$ENV_NAME/bin" ] && [ -f "$CONDA_ENVS_PATH/$ENV_NAME/bin/python" ]; then
    export PATH="$CONDA_ENVS_PATH/$ENV_NAME/bin:$PATH"
    export CONDA_DEFAULT_ENV=$ENV_NAME
    export PYTHONNOUSERSITE=1
    python_path=$(which python)
    if [[ "$python_path" == *"$ENV_NAME"* ]]; then
        log_message "✅ Environment activated: $python_path"
    else
        log_message "❌ Failed to activate environment"
        exit 1
    fi
else
    log_message "❌ Environment $ENV_NAME not found at $CONDA_ENVS_PATH/$ENV_NAME"
    exit 1
fi

# Navigate to project directory
cd /n/holylabs/LABS/mweber_lab/Everyone/rpellegrin/graph_moes || {
    log_message "❌ Failed to navigate to project directory"
    exit 1
}

log_message "📁 Project directory: $(pwd)"

# Add project root to PYTHONPATH
export PYTHONPATH="$(pwd):${PYTHONPATH}"

# Verify graphbench-lib is installed
if ! python -c "import graphbench" 2>/dev/null; then
    log_message "❌ graphbench-lib is not installed"
    log_message "   Installing graphbench-lib..."
    python -m pip install graphbench-lib --no-cache-dir --no-user || {
        log_message "❌ Failed to install graphbench-lib"
        exit 1
    }
else
    log_message "✅ graphbench-lib is installed"
fi

# Run the download script
log_message "📥 Starting dataset downloads..."
python scripts/download_graphbench_cluster.py

if [ $? -eq 0 ]; then
    log_message "✅ Download script completed successfully"
else
    log_message "❌ Download script failed with exit code $?"
    exit 1
fi

log_message "🎉 All done!"

