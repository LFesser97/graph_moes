#!/bin/bash
# ============================================================================
# Tree Mover's Distance (TMD) Analysis on Harvard Cluster
# ============================================================================
# This script computes TMD (Tree Mover's Distance) and class-distance ratios
# for graph datasets on the Harvard cluster. It processes datasets like MUTAG
# and ENZYMES, computing pairwise TMD distances and class-distance ratios.
#
# Usage: sbatch compute_tmd_analysis.sh
#        or: bash compute_tmd_analysis.sh (for testing)
# ============================================================================

#SBATCH --job-name=tmd_analysis
#SBATCH --ntasks=1
#SBATCH --time=48:00:00           # 24 hours should be enough for TMD computation
#SBATCH --mem=32GB                # TMD computation can be memory-intensive
#SBATCH --output=logs_tmd/tmd_analysis_%j.log
#SBATCH --partition=mweber_gpu    # Use GPU partition (matches other scripts, TMD doesn't need GPU but can run here)
#SBATCH --cpus-per-task=4         # Can use multiple CPUs for parallel computation

# Function to log messages with timestamp
log_message() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "${SLURM_OUTFILE:-/dev/stdout}"
}

log_message "🚀 Starting TMD Analysis Job"

# Create log directory
mkdir -p logs_tmd tmd_results

# Set environment path
export CONDA_ENVS_PATH=/n/holylabs/LABS/mweber_lab/Everyone/rpellegrin/conda/envs
export CONDA_PKGS_DIRS=/n/holylabs/LABS/mweber_lab/Everyone/rpellegrin/conda/pkgs

# Load Python module (provides mamba/conda infrastructure)
log_message "📦 Loading Python module..."
module load python/3.10.12-fasrc01 || {
    log_message "⚠️  Failed to load python module, trying alternative..."
    module load python/3.10-fasrc01 || {
        log_message "❌ Failed to load Python module"
        log_message "   Available modules:"
        module avail python 2>&1 | head -5
        exit 1
    }
}

# Verify mamba/conda is available
if ! command -v mamba &> /dev/null && ! command -v conda &> /dev/null; then
    log_message "❌ Neither mamba nor conda available after loading Python module"
    exit 1
fi

log_message "✅ Python module loaded: $(which python)"

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

# Activate environment (use moe_fresh - the fresh environment)
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
    # Prevent user site-packages
    export PYTHONNOUSERSITE=1
    python_path=$(which python)
    if [[ "$python_path" == *"$ENV_NAME"* ]]; then
        log_message "✅ Environment activated"
        log_message "   Python after activation: $python_path"
        activation_success=true
    else
        log_message "⚠️  PATH setup didn't work, Python still: $python_path"
        log_message "   Checking if $ENV_NAME/bin/python exists..."
        ls -la "$CONDA_ENVS_PATH/$ENV_NAME/bin/python" 2>&1 || log_message "   ❌ $ENV_NAME/bin/python does not exist!"
    fi
else
    log_message "⚠️  $ENV_NAME/bin directory or python not found at: $CONDA_ENVS_PATH/$ENV_NAME/bin"
fi

# Final verification
if [ "$activation_success" = false ]; then
    log_message "❌ Failed to activate $ENV_NAME environment"
    log_message "   Current Python: $(which python)"
    log_message "   Expected path should contain: $ENV_NAME"
    log_message "   CONDA_ENVS_PATH: $CONDA_ENVS_PATH"
    if [ -d "$CONDA_ENVS_PATH" ]; then
        log_message "   Available environments:"
        ls -la "$CONDA_ENVS_PATH/" 2>&1 | head -10
    else
        log_message "   CONDA_ENVS_PATH directory does not exist!"
    fi
    exit 1
fi

# Double-check Python is from the correct environment
python_path=$(which python)
if [[ "$python_path" != *"$ENV_NAME"* ]]; then
    log_message "❌ CRITICAL: Python not from $ENV_NAME environment after activation!"
    log_message "   Python path: $python_path"
    log_message "   This will cause import errors!"
    exit 1
fi

log_message "✅ Verified $ENV_NAME environment active: $python_path"

# Navigate to project directory
cd /n/holylabs/LABS/mweber_lab/Everyone/rpellegrin/graph_moes || {
    log_message "❌ Failed to navigate to project directory"
    exit 1
}

log_message "📁 Project directory: $(pwd)"

# Add project root and src to PYTHONPATH
export PYTHONPATH="$(pwd):$(pwd)/src:${PYTHONPATH}"

# Install required packages that might be missing
log_message "📦 Checking and installing required packages..."

# Fix NumPy version (downgrade if needed - project requires <2.0)
NUMPY_VERSION=$(python -c "import numpy; print(numpy.__version__)" 2>/dev/null | cut -d. -f1)
if [ "$NUMPY_VERSION" = "2" ]; then
    log_message "   ⚠️  NumPy 2.x detected, downgrading to <2.0..."
    python -m pip install "numpy>=1.23.1,<2.0" --no-cache-dir --no-user --quiet 2>&1 || log_message "⚠️  Failed to downgrade numpy"
fi

# Install attrdict3 for Python 3.10+ compatibility
if ! python -c "import attrdict3" 2>/dev/null; then
    log_message "   Installing attrdict3 (Python 3.10+ compatible)..."
    python -m pip install attrdict3 --no-cache-dir --no-user --quiet 2>&1 || log_message "⚠️  Failed to install attrdict3"
fi

# Check and install python-louvain (imported as 'community')
if ! python -c "import community" 2>/dev/null; then
    log_message "     Installing python-louvain (provides 'community' module)..."
    python -m pip install "python-louvain>=0.16" --no-cache-dir --no-user --quiet 2>&1 || log_message "     ⚠️  Failed to install python-louvain"
fi

# Install pot (Python Optimal Transport) - CRITICAL for TMD computation
log_message "   Installing pot (Python Optimal Transport) for TMD..."
python -m pip install "pot>=0.9.0" --no-cache-dir --no-user --quiet 2>&1 || {
    log_message "   ⚠️  Failed to install pot, will check again later..."
}

# Check and install other dependencies
MISSING_DEPS=(
    "graphriccicurvature>=0.5.3.1:GraphRicciCurvature"
    "numba>=0.56.4:numba"
    "networkit>=10.1:networkit"
    "cvxpy>=1.4.1:cvxpy"
    "pot>=0.9.0:ot"
    "torcheval>=0.0.7:torcheval"
    "wget>=3.2:wget"
    "ipdb>=0.13.13:ipdb"
    "asgl>=1.0.5:asgl"
)
for dep_spec in "${MISSING_DEPS[@]}"; do
    DEP=$(echo "$dep_spec" | cut -d':' -f1)
    IMPORT_NAME=$(echo "$dep_spec" | cut -d':' -f2)
    if ! python -c "import $IMPORT_NAME" 2>/dev/null; then
        PACKAGE_NAME=$(echo "$DEP" | cut -d'>' -f1 | cut -d'=' -f1)
        log_message "     Installing $PACKAGE_NAME..."
        python -m pip install "$DEP" --no-cache-dir --no-user --quiet 2>&1 || log_message "     ⚠️  Failed to install $PACKAGE_NAME"
    fi
done

# Verify pot (Python Optimal Transport) is installed - critical for TMD
if ! python -c "import ot" 2>/dev/null; then
    log_message "❌ CRITICAL: pot (Python Optimal Transport) not installed!"
    log_message "   Installing pot..."
    python -m pip install "pot>=0.9.0" --no-cache-dir --no-user 2>&1 || {
        log_message "❌ Failed to install pot. TMD computation will fail!"
        exit 1
    }
    log_message "✅ pot installed successfully"
else
    log_message "✅ pot (Python Optimal Transport) is installed"
fi

# Install project in development mode (if not already installed)
if ! python -c "import graph_moes" 2>/dev/null; then
    log_message "📦 Installing graph_moes project (without dependencies)..."
    if pip install -e . --no-deps --quiet 2>/dev/null; then
        log_message "✅ Project installed (no-deps mode)"
    elif [ -d "src" ]; then
        log_message "⚠️  pip install failed, adding src to PYTHONPATH..."
        export PYTHONPATH="$(pwd)/src:$PYTHONPATH"
        if python -c "import graph_moes" 2>/dev/null; then
            log_message "✅ Project accessible via PYTHONPATH"
        else
            log_message "❌ Failed to make graph_moes importable"
            exit 1
        fi
    else
        log_message "❌ Failed to install graph_moes project and src directory not found"
        exit 1
    fi
else
    log_message "✅ graph_moes already installed"
    # Verify TMD module can be imported
    if ! python -c "from graph_moes.tmd import TMD, compute_tmd_matrix" 2>/dev/null; then
        log_message "⚠️  graph_moes installed but TMD module not importable, adding src to PYTHONPATH..."
        export PYTHONPATH="$(pwd)/src:$PYTHONPATH"
        if ! python -c "from graph_moes.tmd import TMD, compute_tmd_matrix" 2>/dev/null; then
            log_message "❌ Failed to import TMD module even with PYTHONPATH"
            log_message "   Attempting to reinstall package..."
            pip install -e . --no-deps --force-reinstall --quiet 2>&1 || log_message "⚠️  Reinstall failed, continuing anyway..."
        else
            log_message "✅ TMD module now accessible via PYTHONPATH"
        fi
    fi
fi

# Quick verification
log_message "🔍 Verifying TMD module and dependencies..."
python -c "import numpy, pandas, torch, ot; from graph_moes.tmd import TMD, compute_tmd_matrix, compute_class_distance_ratios; print('✅ All TMD dependencies available')" || {
    log_message "❌ TMD dependencies not available"
    log_message "   Testing individual imports..."
    python -c "import numpy" 2>&1 || log_message "   ❌ numpy failed"
    python -c "import pandas" 2>&1 || log_message "   ❌ pandas failed"
    python -c "import torch" 2>&1 || log_message "   ❌ torch failed"
    python -c "import ot" 2>&1 || log_message "   ❌ ot (pot) failed"
    python -c "from graph_moes.tmd import TMD" 2>&1 || log_message "   ❌ graph_moes.tmd failed"
    exit 1
}

log_message "✅ All dependencies verified"

# Set data directory (cluster path)
DATA_DIR="/n/holylabs/LABS/mweber_lab/Everyone/rpellegrin/graph_moes/graph_datasets"
if [ ! -d "$DATA_DIR" ]; then
    log_message "⚠️  Data directory not found at $DATA_DIR"
    log_message "   Trying local directory..."
    DATA_DIR="$(pwd)/graph_datasets"
    if [ ! -d "$DATA_DIR" ]; then
        log_message "❌ Data directory not found. Please check the path."
        exit 1
    fi
fi

log_message "📁 Using data directory: $DATA_DIR"

# Set output directory
OUTPUT_DIR="$(pwd)/tmd_results"
log_message "📁 Output directory: $OUTPUT_DIR"

# Define datasets to process
# Available TU datasets for TMD analysis:
# - MUTAG: Small (188 graphs, 2 classes) - ~35k pairwise distances
# - ENZYMES: Medium (600 graphs, 6 classes) - ~360k pairwise distances
# - PROTEINS: Large (1113 graphs, 2 classes) - ~1.2M pairwise distances
# - IMDB-BINARY: Large (1000 graphs, 2 classes) - ~1M pairwise distances
# - COLLAB: Very Large (5000 graphs, 3 classes) - ~25M pairwise distances - WARNING: May take days!
# - REDDIT-BINARY: Very Large (2000 graphs, 2 classes) - ~4M pairwise distances - WARNING: May take days!
#
# Available GNN Benchmark datasets for TMD analysis:
# - MNIST: Very Large (~70k graphs, 10 classes) - ~4.9B pairwise distances - WARNING: Extremely long!
# - CIFAR10: Very Large (~60k graphs, 10 classes) - ~3.6B pairwise distances - WARNING: Extremely long!
# - PATTERN: Large (~14k graphs, 2 classes) - ~196M pairwise distances - WARNING: May take days!
#
# To process only specific datasets, modify the DATASETS variable below.
# Large datasets (COLLAB, REDDIT-BINARY, PATTERN) and very large datasets (MNIST, CIFAR10)
# are excluded by default due to computation time.

# Default datasets (small to medium size)
DATASETS="PROTEINS IMDB-BINARY COLLAB"

# Uncomment the line below to include large datasets (expect days of computation)
# DATASETS="MUTAG ENZYMES PROTEINS IMDB-BINARY COLLAB REDDIT-BINARY"

# Or uncomment to include GNN Benchmark datasets (WARNING: Very long computation time!)
# DATASETS="MUTAG ENZYMES PATTERN"
# DATASETS="MUTAG ENZYMES PROTEINS IMDB-BINARY PATTERN"

# Or uncomment to process only specific datasets:
# DATASETS="MUTAG ENZYMES"
# DATASETS="PROTEINS IMDB-BINARY"

log_message "🧪 Starting TMD analysis for multiple datasets..."
log_message "   Datasets to process: $DATASETS"
log_message "   This may take several hours depending on dataset size..."
log_message "   Computation time scales roughly as O(n^2) where n is the number of graphs"

# TMD parameters
TMD_W=1.0    # Weighting constant
TMD_L=4      # Computation tree depth

# Run the TMD analysis script with specified datasets
python scripts/compute_tmd_analysis.py \
    --datasets $DATASETS \
    --data-dir "$DATA_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --w "$TMD_W" \
    --L "$TMD_L" \
    --cache

# Check exit status
EXIT_CODE=$?
if [ $EXIT_CODE -eq 0 ]; then
    log_message "✅ TMD analysis completed successfully!"
    log_message "   Results saved to: $OUTPUT_DIR"
    log_message "   Files created:"
    ls -lh "$OUTPUT_DIR"/*.npy "$OUTPUT_DIR"/*.csv "$OUTPUT_DIR"/*.json 2>/dev/null | while read line; do
        log_message "     $line"
    done
else
    log_message "❌ TMD analysis failed with exit code $EXIT_CODE"
    exit $EXIT_CODE
fi

log_message "🎉 TMD Analysis Job Complete!"

