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

cd /n/holylabs/LABS/mweber_lab/Everyone/rpellegrin/graph_moes_2/graph_moes || {
    log_message "❌ Failed to cd to project directory"
    exit 1
}
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

# Set CONDA_ENVS_PATH if not set
if [ -z "$CONDA_ENVS_PATH" ]; then
    export CONDA_ENVS_PATH="/n/holylabs/LABS/mweber_lab/Everyone/rpellegrin/conda/envs"
    log_message "   Set CONDA_ENVS_PATH to: $CONDA_ENVS_PATH"
fi

if [ -f "$(conda info --base 2>/dev/null)/etc/profile.d/conda.sh" ]; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
    log_message "✅ Conda initialized from conda info --base"
elif [ -f "/n/sw/Mambaforge-23.3.1-1/etc/profile.d/conda.sh" ]; then
    source "/n/sw/Mambaforge-23.3.1-1/etc/profile.d/conda.sh"
    log_message "✅ Conda initialized from Mambaforge path"
else
    log_message "⚠️  Could not find conda.sh, activation may not work properly"
fi

# Activate environment (use moe_fresh_2 - the fresh environment we created)
ENV_NAME="moe_fresh_2"
log_message "🔧 Activating $ENV_NAME environment..."
log_message "   CONDA_ENVS_PATH: $CONDA_ENVS_PATH"
log_message "   Python before activation: $(which python)"

# Use manual PATH setup (most reliable method on clusters)
activation_success=false

# First check what environments are actually available
log_message "   Checking available conda environments..."
conda env list 2>/dev/null | grep -E "(moe_fresh_2|base|root)" | head -5

# Build prioritized list of environment paths
possible_env_paths=()

# First priority: actual moe_fresh_2 environment if it exists
if [ -d "/n/holylabs/LABS/mweber_lab/Everyone/rpellegrin/conda/envs/$ENV_NAME" ]; then
    possible_env_paths+=("/n/holylabs/LABS/mweber_lab/Everyone/rpellegrin/conda/envs/$ENV_NAME")
    log_message "   moe_fresh_2 environment found at expected location"
fi

# Other possible locations
possible_env_paths+=(
    "$CONDA_ENVS_PATH/$ENV_NAME"
    "$HOME/.conda/envs/$ENV_NAME"
    "/n/sw/Mambaforge-23.3.1-1/envs/$ENV_NAME"
    "$(conda info --base 2>/dev/null)/envs/$ENV_NAME"
    "$CONDA_PREFIX"  # If we're already in the environment
)

# Last resort: base environment
if [ -d "$(conda info --base 2>/dev/null)" ]; then
    possible_env_paths+=("$(conda info --base 2>/dev/null)")
fi

log_message "   Will check $((${#possible_env_paths[@]})) possible environment paths"

for env_path in "${possible_env_paths[@]}"; do
    if [ -d "$env_path/bin" ] && [ -f "$env_path/bin/python" ]; then
        log_message "   Found environment at: $env_path"
        log_message "   Setting up environment PATH..."

        # Check if this is actually moe_fresh_2 or just base
        if [[ "$env_path" == *"/envs/$ENV_NAME" ]] && [[ "$env_path" == *"rpellegrin"* ]]; then
            log_message "   ✅ Confirmed this is the actual moe_fresh_2 environment"
        elif [[ "$env_path" == *"Miniforge"* ]] || [[ "$env_path" == *"Mambaforge"* ]]; then
            log_message "   ⚠️  This appears to be the base conda environment, not moe_fresh_2"
            log_message "   Continuing anyway - packages should still be available"
        else
            log_message "   Using environment at: $env_path"
        fi

        # Prepend environment bin to PATH
        export PATH="$env_path/bin:$PATH"
        # Also set PYTHONPATH to include user packages
        export PYTHONPATH="$HOME/.local/lib/python3.*/site-packages:$PYTHONPATH"
        # Set CONDA_DEFAULT_ENV for compatibility
        export CONDA_DEFAULT_ENV=$ENV_NAME
        export CONDA_PREFIX="$env_path"
        activation_success=true
        log_message "✅ Environment activated via PATH (method 1)"
        break
    fi
done

if [ "$activation_success" = false ]; then
    # Check if moe_fresh_2 environment exists at all
    if conda env list 2>/dev/null | grep -q "$ENV_NAME"; then
        log_message "   moe_fresh_2 environment exists, trying activation methods..."

        # Try conda activation methods as fallback
        if source activate "$ENV_NAME" 2>/dev/null; then
            activation_success=true
            log_message "✅ Environment activated via source activate (method 2)"
        elif conda activate "$ENV_NAME" 2>/dev/null; then
            activation_success=true
            log_message "✅ Environment activated via conda activate (method 3)"
        else
            log_message "⚠️  Environment activation failed despite environment existing"
            log_message "   Continuing with current Python environment..."
        fi
    else
        log_message "⚠️  $ENV_NAME environment not found in conda"
        log_message "   Using base conda environment instead..."

        # Use base conda environment
        base_env=$(conda info --base 2>/dev/null)
        if [ -n "$base_env" ] && [ -d "$base_env/bin" ]; then
            export PATH="$base_env/bin:$PATH"
            export CONDA_DEFAULT_ENV="base"
            export CONDA_PREFIX="$base_env"
            activation_success=true
            log_message "✅ Using base conda environment"
        else
            log_message "⚠️  Could not find base conda environment either"
            log_message "   Continuing with system Python..."
        fi
    fi
fi

log_message "   Python after activation: $(which python)"
log_message "   CONDA_DEFAULT_ENV: $CONDA_DEFAULT_ENV"
log_message "   Active environment path: $CONDA_PREFIX"

# Ensure user-installed packages are in PYTHONPATH
USER_PYTHON_PATH="$HOME/.local/lib/python3.*/site-packages"
if [ -d "$USER_PYTHON_PATH" ]; then
    export PYTHONPATH="$USER_PYTHON_PATH:$PYTHONPATH"
    log_message "   Added user packages to PYTHONPATH: $USER_PYTHON_PATH"
fi

# Check if pandas is available, install if not
log_message "🔍 Checking pandas availability..."
pandas_installed=false
python -c "import pandas; print('✅ pandas available')" 2>/dev/null && pandas_installed=true

if [ "$pandas_installed" = false ]; then
    log_message "⚠️  pandas not found, installing..."

    # Try pip install with user flag (avoids permission issues)
    pip install --user pandas --quiet 2>&1 && {
        log_message "✅ pandas installed successfully via pip --user"
        pandas_installed=true
    } || {
        log_message "⚠️  pip --user failed, trying conda install..."

        # Try conda install with user-specific location
        conda install -c conda-forge pandas -y --quiet --prefix="$HOME/.conda/envs/${ENV_NAME}_pandas" 2>&1 && {
            log_message "✅ pandas installed successfully via conda (user env)"
            # Add to PATH if conda created a new environment
            if [ -d "$HOME/.conda/envs/${ENV_NAME}_pandas/bin" ]; then
                export PATH="$HOME/.conda/envs/${ENV_NAME}_pandas/bin:$PATH"
                log_message "   Added pandas env to PATH"
            fi
            pandas_installed=true
        } || {
            log_message "❌ Failed to install pandas via pip and conda"
            log_message "   Continuing anyway - pandas is optional for download"
            # Don't exit - pandas is only needed for CSV saving, not for download itself
        }
    }
else
    log_message "✅ pandas already available"
fi

# Quick verification of required packages
log_message "🔍 Verifying required packages..."
python -c "import numpy; print('✅ numpy available')" || {
    log_message "❌ numpy not available - this is required"
    exit 1
}

if [ "$pandas_installed" = true ]; then
    python -c "import pandas; print('✅ pandas available')" || {
        log_message "⚠️  pandas not available after installation (continuing anyway)"
    }
else
    log_message "⚠️  pandas not available (continuing anyway - only needed for CSV saving)"
fi

log_message "📥 Starting ogbg-ppa dataset download..."
log_message "   Dataset: ppa"
log_message "   Expected size: ~2.8GB"

# Use the dedicated download script that handles prompts automatically
python download_ogbg_ppa.py

# Check if download was successful
if [ -d "graph_datasets/ogbg_ppa" ]; then
    dataset_size=$(du -sh graph_datasets/ogbg_ppa/ | cut -f1)
    log_message "✅ Download completed successfully! Dataset size: $dataset_size"
    log_message "   Location: $(pwd)/graph_datasets/ogbg_ppa/"

    # Count the number of graphs if possible
    if [ -f "graph_datasets/ogbg_ppa/processed/data.pt" ]; then
        log_message "   Dataset appears to be properly processed"
    else
        log_message "   ⚠️  Dataset downloaded but may need processing"
    fi
else
    log_message "❌ Download failed - dataset directory not found"
    exit 1
fi

log_message "🎉 ogbg-ppa download job completed successfully!"
log_message "   Logs saved to: logs/download_ppa_${SLURM_JOB_ID}.log"
