#!/bin/bash
# ============================================================================
# Download OGB PPA Dataset - Dedicated Download Job
# ============================================================================
# This script downloads the ogbg-ppa dataset which requires user confirmation
# and is too large to download during normal experiments.
#
# The script ensures pandas is available and sets up the full environment
# like the comprehensive sweep script.
#
# Usage: sbatch download_ppa.sh
# ============================================================================

#SBATCH --job-name=download_ppa
#SBATCH --time=10:00:00
#SBATCH --mem=32GB
#SBATCH --partition=mweber_gpu
#SBATCH --gpus=1
#SBATCH --output=logs/download_ppa_%j.log   # %j = job ID

# Create logs directory if it doesn't exist
mkdir -p logs

# Logging function
log_message() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') - $1" | tee -a "logs/download_ppa_${SLURM_JOB_ID}.log"
}

log_message "🚀 Starting ogbg-ppa download job (ID: ${SLURM_JOB_ID})"

cd /n/holylabs/LABS/mweber_lab/Everyone/rpellegrin/graph_moes
log_message "📂 Working directory: $(pwd)"

# Set PYTHONPATH
export PYTHONPATH="$(pwd):$(pwd)/src:${PYTHONPATH}"
log_message "🔧 PYTHONPATH set to: $PYTHONPATH"

# Disable user site-packages to prevent conflicts
export PYTHONNOUSERSITE=1

# Load CUDA modules if available (needed for torch CUDA support)
if command -v module &> /dev/null; then
    log_message "📦 Loading CUDA modules..."
    if module load cuda/12.9.1-fasrc01 2>/dev/null; then
        log_message "   ✅ Loaded cuda/12.9.1-fasrc01"
    elif module load cuda 2>/dev/null; then
        log_message "   ✅ Loaded cuda (default version)"
    else
        log_message "   ⚠️  CUDA module not found, continuing..."
    fi

    if module load cudnn/9.10.2.21_cuda12-fasrc01 2>/dev/null; then
        log_message "   ✅ Loaded cudnn/9.10.2.21_cuda12-fasrc01"
    elif module load cudnn 2>/dev/null; then
        log_message "   ✅ Loaded cudnn (default version)"
    else
        log_message "   ⚠️  cuDNN module not found, continuing..."
    fi

    # Set LD_LIBRARY_PATH to include CUDA libraries
    if [ -n "$CUDA_HOME" ]; then
        export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${CUDA_HOME}/lib:${CUDA_HOME}/targets/x86_64-linux/lib:${LD_LIBRARY_PATH}"
        log_message "   Using CUDA_HOME: $CUDA_HOME"
    elif [ -n "$CUDA_ROOT" ]; then
        export LD_LIBRARY_PATH="${CUDA_ROOT}/lib64:${CUDA_ROOT}/lib:${CUDA_ROOT}/targets/x86_64-linux/lib:${LD_LIBRARY_PATH}"
        log_message "   Using CUDA_ROOT: $CUDA_ROOT"
    fi
    log_message "   LD_LIBRARY_PATH: ${LD_LIBRARY_PATH}"
fi

# Initialize conda/mamba (required for source activate to work properly)
log_message "🔧 Initializing conda/mamba..."
if [ -f "$(conda info --base 2>/dev/null)/etc/profile.d/conda.sh" ]; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
    log_message "✅ Conda initialized from conda info --base"
elif [ -f "/n/sw/Mambaforge-23.3.1-1/etc/profile.d/conda.sh" ]; then
    source "/n/sw/Mambaforge-23.3.1-1/etc/profile.d/conda.sh"
    log_message "✅ Conda initialized from Mambaforge path"
else
    log_message "⚠️  Could not find conda.sh, activation may not work properly"
fi

# Activate environment (use moe_fresh - the fresh environment we created)
ENV_NAME="moe_fresh"
log_message "🔧 Activating $ENV_NAME environment..."
log_message "   CONDA_ENVS_PATH: $CONDA_ENVS_PATH"
log_message "   Python before activation: $(which python)"

# Use manual PATH setup (most reliable method on clusters)
activation_success=false
if [ -d "$CONDA_ENVS_PATH/$ENV_NAME/bin" ] && [ -f "$CONDA_ENVS_PATH/$ENV_NAME/bin/python" ]; then
    log_message "   Setting up environment PATH..."
    # Prepend environment bin to PATH
    export PATH="$CONDA_ENVS_PATH/$ENV_NAME/bin:$PATH"
    # Also set CONDA_DEFAULT_ENV for compatibility
    export CONDA_DEFAULT_ENV=$ENV_NAME
    activation_success=true
    log_message "✅ Environment activated via PATH (method 1)"
elif source activate "$ENV_NAME" 2>/dev/null; then
    activation_success=true
    log_message "✅ Environment activated via source activate (method 2)"
elif conda activate "$ENV_NAME" 2>/dev/null; then
    activation_success=true
    log_message "✅ Environment activated via conda activate (method 2)"
else
    log_message "⚠️  Environment activation failed, continuing with system Python..."
fi

log_message "   Python after activation: $(which python)"
log_message "   CONDA_DEFAULT_ENV: $CONDA_DEFAULT_ENV"

# Check if pandas is available, install if not
log_message "🔍 Checking pandas availability..."
pandas_installed=false
python -c "import pandas; print('✅ pandas available')" 2>/dev/null && pandas_installed=true

if [ "$pandas_installed" = false ]; then
    log_message "⚠️  pandas not found, installing..."

    # Try pip install first
    pip install pandas --quiet 2>&1 && {
        log_message "✅ pandas installed successfully via pip"
        pandas_installed=true
    } || {
        log_message "⚠️  pip install failed, trying conda install..."
        # Try conda install as fallback
        conda install -c conda-forge pandas -y --quiet 2>&1 && {
            log_message "✅ pandas installed successfully via conda"
            pandas_installed=true
        } || {
            log_message "❌ Failed to install pandas via both pip and conda"
            exit 1
        }
    }
else
    log_message "✅ pandas already available"
fi

# Quick verification of required packages
log_message "🔍 Verifying required packages..."
python -c "import pandas; print('✅ pandas available')" || {
    log_message "❌ pandas not available after installation"
    exit 1
}

log_message "📥 Starting ogbg-ppa dataset download..."
log_message "   Dataset: ppa"
log_message "   Expected size: ~2.8GB"

# Run the download (single trial to trigger download)
python scripts/run_graph_classification.py --dataset ppa --layer_type GCN --num_trials 1

# Check if download was successful
if [ -d "graph_datasets/ogbg_ppa" ]; then
    dataset_size=$(du -sh graph_datasets/ogbg_ppa/ | cut -f1)
    log_message "✅ Download completed successfully! Dataset size: $dataset_size"
    log_message "   Location: $(pwd)/graph_datasets/ogbg_ppa/"
else
    log_message "❌ Download failed - dataset directory not found"
    exit 1
fi

log_message "🎉 ogbg-ppa download job completed successfully!"
log_message "   Logs saved to: logs/download_ppa_${SLURM_JOB_ID}.log"
