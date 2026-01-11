#!/bin/bash
# ============================================================================
# Compute Hypergraph and Graph Encodings for All Datasets
# ============================================================================
# This script computes hypergraph and graph encodings for all datasets in the
# graph_datasets directory. It handles:
#   - Hypergraph encodings: LDP, FRC, RWPE, LAPE (ORC skipped - very slow)
#   - Graph encodings: LDP, FRC, ORC, RWPE, LAPE (all combined)
#
# The script:
#   1. Sets up the conda environment
#   2. Installs/clones the Hypergraph_Encodings repo if needed
#   3. Runs the encoding computation script
#
# Output files are saved to:
#   - graph_datasets_with_hg_encodings/{dataset}_hg_{encoding_suffix}.pt
#     - e.g., mutag_hg_ldp.pt, mutag_hg_frc.pt, mutag_hg_rwpe_we_k20.pt, mutag_hg_lape_normalized_k8.pt
#   - graph_datasets_with_g_encodings/{dataset}_g_{encoding_suffix}.pt
#     - e.g., mutag_g_ldp.pt, mutag_g_rwpe_k16.pt, mutag_g_lape_k8.pt, mutag_g_orc.pt
#
# Usage: sbatch compute_encodings.sh
# ============================================================================

#SBATCH --job-name=compute_encodings
#SBATCH --ntasks=1
#SBATCH --time=72:00:00           # Long time - encoding computation can take hours
#SBATCH --mem=128GB               # Large memory for large datasets
#SBATCH --output=logs_encodings/compute_encodings_%j.log
#SBATCH --partition=mweber_gpu    # Use GPU partition (but no GPU needed for encodings)
#SBATCH --gpus=0                  # No GPU needed - encodings are CPU-based
#SBATCH --cpus-per-task=8         # Multiple CPUs for parallel operations

# Function to log messages with timestamp
log_message() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

log_message "Starting encoding computation job"

# Set environment paths
export CONDA_ENVS_PATH=/n/holylabs/LABS/mweber_lab/Everyone/rpellegrin/conda/envs
export CONDA_PKGS_DIRS=/n/holylabs/LABS/mweber_lab/Everyone/rpellegrin/conda/pkgs

# Load Python module
log_message "📦 Loading Python module..."
module load python/3.10.12-fasrc01 || {
    log_message "⚠️  Failed to load python module, trying alternative..."
    module load python/3.10-fasrc01 || {
        log_message "❌ Failed to load Python module"
        module avail python 2>&1 | head -5
        exit 1
    }
}

log_message "✅ Python module loaded: $(which python)"

# Initialize conda/mamba
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

# Activate environment
ENV_NAME="moe_fresh"
log_message "🔧 Activating $ENV_NAME environment..."
log_message "   CONDA_ENVS_PATH: $CONDA_ENVS_PATH"
log_message "   Python before activation: $(which python)"

# Use manual PATH setup
activation_success=false
if [ -d "$CONDA_ENVS_PATH/$ENV_NAME/bin" ] && [ -f "$CONDA_ENVS_PATH/$ENV_NAME/bin/python" ]; then
    log_message "   Setting up environment PATH..."
    export PATH="$CONDA_ENVS_PATH/$ENV_NAME/bin:$PATH"
    export CONDA_DEFAULT_ENV=$ENV_NAME
    export PYTHONNOUSERSITE=1
    python_path=$(which python)
    if [[ "$python_path" == *"$ENV_NAME"* ]]; then
        log_message "✅ Environment activated"
        log_message "   Python after activation: $python_path"
        activation_success=true
    else
        log_message "⚠️  PATH setup didn't work, Python still: $python_path"
    fi
else
    log_message "⚠️  $ENV_NAME/bin directory or python not found at: $CONDA_ENVS_PATH/$ENV_NAME/bin"
fi

if [ "$activation_success" = false ]; then
    log_message "❌ Failed to activate $ENV_NAME environment"
    log_message "   Current Python: $(which python)"
    exit 1
fi

# Get project root directory
# Use SLURM_SUBMIT_DIR if available (directory where sbatch was called), otherwise use absolute path
if [ -n "$SLURM_SUBMIT_DIR" ]; then
    # SLURM sets this to the directory where sbatch was called
    PROJECT_ROOT="$SLURM_SUBMIT_DIR"
    log_message "📁 Using SLURM_SUBMIT_DIR: $PROJECT_ROOT"
else
    # Fallback: use absolute path
    PROJECT_ROOT="/n/holylabs/LABS/mweber_lab/Everyone/rpellegrin/graph_moes"
    log_message "📁 Using absolute path: $PROJECT_ROOT"
fi

# Verify project root exists
if [ ! -d "$PROJECT_ROOT" ]; then
    log_message "❌ Project root directory not found: $PROJECT_ROOT"
    log_message "   Current directory: $(pwd)"
    exit 1
fi

# Change to project root
cd "$PROJECT_ROOT" || {
    log_message "❌ Failed to change to project root: $PROJECT_ROOT"
    exit 1
}

log_message "📁 Current directory: $(pwd)"
log_message "📁 Project root: $PROJECT_ROOT"

# Set up Hypergraph_Encodings repo path
# The Python script looks for: Repos_GNN/Hypergraph_encodings_clean/Hypergraph_Encodings
# But it might also be directly in the parent directory: Repos_GNN/Hypergraph_Encodings
HG_ENCODINGS_PARENT_DIR="$(dirname "$PROJECT_ROOT")"
HG_ENCODINGS_REPO_DIR_ALT1="$HG_ENCODINGS_PARENT_DIR/Hypergraph_encodings_clean/Hypergraph_Encodings"
HG_ENCODINGS_REPO_DIR_ALT2="$HG_ENCODINGS_PARENT_DIR/Hypergraph_Encodings"
HG_ENCODINGS_REPO_DIR=""
HG_ENCODINGS_SRC_DIR=""

log_message "🔍 Checking for Hypergraph_Encodings repo..."

# Check multiple possible locations
if [ -d "$HG_ENCODINGS_REPO_DIR_ALT1" ] && [ -d "$HG_ENCODINGS_REPO_DIR_ALT1/src" ]; then
    HG_ENCODINGS_REPO_DIR="$HG_ENCODINGS_REPO_DIR_ALT1"
    HG_ENCODINGS_SRC_DIR="$HG_ENCODINGS_REPO_DIR/src"
    log_message "✅ Hypergraph_Encodings repo found at: $HG_ENCODINGS_REPO_DIR"
elif [ -d "$HG_ENCODINGS_REPO_DIR_ALT2" ] && [ -d "$HG_ENCODINGS_REPO_DIR_ALT2/src" ]; then
    HG_ENCODINGS_REPO_DIR="$HG_ENCODINGS_REPO_DIR_ALT2"
    HG_ENCODINGS_SRC_DIR="$HG_ENCODINGS_REPO_DIR/src"
    log_message "✅ Hypergraph_Encodings repo found at: $HG_ENCODINGS_REPO_DIR"
else
    log_message "⚠️  Hypergraph_Encodings repo not found at either location:"
    log_message "   Option 1: $HG_ENCODINGS_REPO_DIR_ALT1"
    log_message "   Option 2: $HG_ENCODINGS_REPO_DIR_ALT2"
    log_message "   Attempting to clone from GitHub to Option 1..."
    
    # Create parent directory structure for Option 1
    HG_ENCODINGS_ALT1_PARENT="$HG_ENCODINGS_PARENT_DIR/Hypergraph_encodings_clean"
    mkdir -p "$HG_ENCODINGS_ALT1_PARENT" || {
        log_message "❌ Failed to create directory: $HG_ENCODINGS_ALT1_PARENT"
        log_message "   Trying Option 2 instead..."
        HG_ENCODINGS_ALT1_PARENT="$HG_ENCODINGS_PARENT_DIR"
    }
    
    # Try to clone the repo
    cd "$HG_ENCODINGS_ALT1_PARENT" || {
        log_message "❌ Failed to change to directory: $HG_ENCODINGS_ALT1_PARENT"
        exit 1
    }
    
    if git clone https://github.com/Weber-GeoML/Hypergraph_Encodings.git 2>&1; then
        log_message "✅ Successfully cloned Hypergraph_Encodings repo"
        HG_ENCODINGS_REPO_DIR="$HG_ENCODINGS_ALT1_PARENT/Hypergraph_Encodings"
        HG_ENCODINGS_SRC_DIR="$HG_ENCODINGS_REPO_DIR/src"
    else
        log_message "❌ Failed to clone Hypergraph_Encodings repo"
        log_message "   Please clone it manually to one of these locations:"
        log_message "   - $HG_ENCODINGS_REPO_DIR_ALT1"
        log_message "   - $HG_ENCODINGS_REPO_DIR_ALT2"
        exit 1
    fi
    
    # Reset to project root
    cd "$PROJECT_ROOT" || exit 1
fi

# Install the Hypergraph_Encodings package
log_message "📦 Installing Hypergraph_Encodings package..."
cd "$HG_ENCODINGS_REPO_DIR" || {
    log_message "❌ Failed to change to Hypergraph_Encodings directory"
    exit 1
}

# Check if already installed, otherwise install
if python -c "import sys; sys.path.insert(0, '$HG_ENCODINGS_SRC_DIR'); from encodings_hnns.encodings import HypergraphEncodings" 2>/dev/null; then
    log_message "✅ Hypergraph_Encodings already accessible"
else
    log_message "📦 Installing Hypergraph_Encodings..."
    if pip install -e . 2>&1; then
        log_message "✅ Successfully installed Hypergraph_Encodings"
    else
        log_message "⚠️  pip install -e . failed, but continuing (may already be installed)"
    fi
fi

# Go back to project root
cd "$PROJECT_ROOT" || exit 1

# Verify the script exists
ENCODING_SCRIPT="$PROJECT_ROOT/scripts/compute_encodings_for_datasets.py"
if [ ! -f "$ENCODING_SCRIPT" ]; then
    log_message "❌ Encoding script not found: $ENCODING_SCRIPT"
    exit 1
fi

log_message "✅ Encoding script found: $ENCODING_SCRIPT"

# Create output directories
mkdir -p logs_encodings
mkdir -p graph_datasets_with_hg_encodings
mkdir -p graph_datasets_with_g_encodings

# Check if data directory exists
if [ ! -d "graph_datasets" ]; then
    log_message "⚠️  graph_datasets directory not found"
    log_message "   The script will attempt to create it, but datasets may need to be downloaded first"
fi

# Print Python and package versions
log_message "🐍 Python version: $(python --version)"
log_message "📦 Checking key packages..."
python -c "import torch; print(f'   PyTorch: {torch.__version__}')" 2>/dev/null || log_message "   ⚠️  PyTorch not found"
python -c "import torch_geometric; print(f'   PyG: {torch_geometric.__version__}')" 2>/dev/null || log_message "   ⚠️  PyTorch Geometric not found"
python -c "import numpy; print(f'   NumPy: {numpy.__version__}')" 2>/dev/null || log_message "   ⚠️  NumPy not found"
python -c "import sys; sys.path.insert(0, '$HG_ENCODINGS_SRC_DIR'); from encodings_hnns.encodings import HypergraphEncodings; print('   ✅ HypergraphEncodings importable')" 2>/dev/null || log_message "   ⚠️  HypergraphEncodings not importable"

# Run the encoding computation script
log_message "🚀 Starting encoding computation..."
log_message "   This may take several hours depending on dataset sizes"

python scripts/compute_encodings_for_datasets.py

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    log_message "✅ Encoding computation completed successfully!"
else
    log_message "❌ Encoding computation failed with exit code: $EXIT_CODE"
    exit $EXIT_CODE
fi

log_message "🎉 Job finished!"