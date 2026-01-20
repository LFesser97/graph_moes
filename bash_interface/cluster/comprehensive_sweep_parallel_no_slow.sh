#!/bin/bash
# ============================================================================
# Comprehensive Graph MoE Sweep - Parallel Array Job Version (Excluding Slow Datasets)
# ============================================================================
# This script runs a comprehensive hyperparameter sweep for graph neural network
# experiments using SLURM array jobs for parallel execution. It tests both single
# layer architectures (GCN, GIN, SAGE, MLP, Unitary, GPS) and MoE (Mixture of Experts)
# combinations across multiple datasets.
#
# This version EXCLUDES mnist and cifar datasets (which are slow to run).
#
# The script uses optimal hyperparameters from research papers for each dataset
# and model combination, loaded from hyperparams_lookup.sh.
#
# Total experiments: 600
#   - Base experiments per encoding variant: 120 (60 × 2 normalization variants)
#     - 12 single layer experiments (6 × 2 normalization variants):
#       - GPS: 6 datasets × 1 (no skip) × 2 (norm/no-norm) = 12
#     - 108 MoE experiments (54 × 2 normalization variants): 9 combinations × 6 datasets × 1 router type (GNN) × 2 (norm/no-norm)
#   - Encoding variants: 5 (hg_lape_normalized_k8, hg_rwpe_we_k20, g_rwpe_k16, g_lape_k8, None)
#   - Total: 120 × 5 = 600 experiments
#     Encoding types: hg_ldp, hg_frc, hg_rwpe_we_k20, hg_lape_normalized_k8, g_ldp, g_rwpe_k16, g_lape_k8, g_orc
# Note: GraphBench/PATTERN/cluster excluded (node classification or disabled)
# Note: mnist and cifar excluded (slow datasets)
# Each experiment runs 200 trials to ensure proper test set coverage
# Skip connections are only applied to GCN, GIN, and SAGE
# Feature normalization is applied to all models as a variant
#
# Usage: sbatch comprehensive_sweep_parallel_no_slow.sh
# ============================================================================

#SBATCH --job-name=comprehensive_sweep_no_slow
#SBATCH --array=1-600             # Total experiments: 120 base × 5 encoding variants = 600
#SBATCH --ntasks=1
#SBATCH --time=192:00:00           # Long time for comprehensive sweep
#SBATCH --mem=128GB               # Sufficient memory
#SBATCH --output=logs_comprehensive/Parallel_comprehensive_sweep_no_slow_%A_%a.log  # %A = array job ID, %a = task ID
#SBATCH --partition=mweber_gpu
#SBATCH --gpus=1
#SBATCH --nice=0                  # Higher priority (lower nice value = higher priority)

# WandB Environment Setup
echo "🚀 Setting up WandB environment for Comprehensive Graph MoE experiments..."

export WANDB_API_KEY="ea7c6eeb5a095b531ef60cc784bfeb87d47ea0b0"
export WANDB_ENTITY="weber-geoml-harvard-university"
export WANDB_PROJECT="MOE_4"
# Use temp directory for WandB files (gets cleaned up automatically on cluster)
# This avoids filling up home directory with wandb files
WANDB_TMP_DIR="${TMPDIR:-/tmp}/wandb_${SLURM_JOB_ID:-$$}"
export WANDB_DIR="${WANDB_TMP_DIR}"
export WANDB_CACHE_DIR="${WANDB_TMP_DIR}/.cache"
# Disable code saving to reduce disk usage
export WANDB_DISABLE_CODE=true
# Sync immediately instead of caching (minimizes local storage)
export WANDB_SYNC_MODE="now"

mkdir -p "${WANDB_TMP_DIR}" logs logs_comprehensive

# Disable user site-packages to prevent conflicts (also set in activation, but set here too)
export PYTHONNOUSERSITE=1

# Load CUDA modules if available (needed for torch CUDA support)
if command -v module &> /dev/null; then
    echo "📦 Loading CUDA modules..."
    if module load cuda/12.9.1-fasrc01 2>/dev/null; then
        echo "   ✅ Loaded cuda/12.9.1-fasrc01"
    elif module load cuda 2>/dev/null; then
        echo "   ✅ Loaded cuda (default version)"
    else
        echo "   ⚠️  CUDA module not found, continuing..."
    fi
    
    if module load cudnn/9.10.2.21_cuda12-fasrc01 2>/dev/null; then
        echo "   ✅ Loaded cudnn/9.10.2.21_cuda12-fasrc01"
    elif module load cudnn 2>/dev/null; then
        echo "   ✅ Loaded cudnn (default version)"
    else
        echo "   ⚠️  cuDNN module not found, continuing..."
    fi
    
    # Set LD_LIBRARY_PATH to include CUDA libraries (including targets directory for libcusparseLt.so.0)
    if [ -n "$CUDA_HOME" ]; then
        export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${CUDA_HOME}/lib:${CUDA_HOME}/targets/x86_64-linux/lib:${LD_LIBRARY_PATH}"
        echo "   Using CUDA_HOME: $CUDA_HOME"
    elif [ -n "$CUDA_ROOT" ]; then
        export LD_LIBRARY_PATH="${CUDA_ROOT}/lib64:${CUDA_ROOT}/lib:${CUDA_ROOT}/targets/x86_64-linux/lib:${LD_LIBRARY_PATH}"
        echo "   Using CUDA_ROOT: $CUDA_ROOT"
    fi
    echo "   LD_LIBRARY_PATH: ${LD_LIBRARY_PATH}"
fi

echo "✅ WandB environment configured"
echo "   Entity: $WANDB_ENTITY"
echo "   Project: $WANDB_PROJECT" 
echo "   API Key: ${WANDB_API_KEY:0:10}..."
echo "   Directory: $WANDB_DIR"

# Install wandb if not already installed
if ! python -c "import wandb" &> /dev/null; then
    echo "📦 Installing wandb..."
    pip install wandb
else
    echo "✅ wandb already installed"
fi

echo "🎉 WandB setup complete!"
echo "🎉 WandB setup complete!"

# Function to log messages with timestamp
log_message() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [Task $SLURM_ARRAY_TASK_ID] $1"
}

log_message "Starting Comprehensive MoE Sweep Task $SLURM_ARRAY_TASK_ID (Excluding Slow Datasets)"

# Set environment path
export CONDA_ENVS_PATH=/n/holylabs/LABS/mweber_lab/Everyone/rpellegrin/conda/envs
export CONDA_PKGS_DIRS=/n/holylabs/LABS/mweber_lab/Everyone/rpellegrin/conda/pkgs

# Load Python module (provides mamba/conda infrastructure)
log_message "📦 Loading Python module..."
module load python/3.10.12-fasrc01 || {
    log_message "⚠️  Failed to load python module, trying alternative..."
    # Try alternative module name
    module load python/3.10-fasrc01 || {
        log_message "❌ Failed to load Python module"
        log_message "   Available modules:"
        module avail python 2>&1 | head -5
        exit 1
    }
}

# Verify mamba is available
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

# Add project root and src to PYTHONPATH so hyperparams.py and graph_moes can be imported
export PYTHONPATH="$(pwd):$(pwd)/src:${PYTHONPATH}"

# Check if scipy is working, if not, try to reinstall
log_message "🔍 Verifying scipy installation..."
if ! python -c "import scipy.signal" 2>/dev/null; then
    log_message "❌ Scipy import failed, attempting to reinstall scipy..."
    log_message "   Uninstalling scipy..."
    python -m pip uninstall scipy -y --quiet 2>/dev/null || log_message "⚠️  Failed to uninstall scipy (may not be installed)."
    log_message "   Installing scipy (using python -m pip to target environment)..."
    # Use python -m pip to ensure we use the environment's pip
    if ! python -m pip install scipy --no-cache-dir --quiet 2>&1; then
        log_message "❌ Failed to reinstall scipy. Trying numpy + scipy reinstall..."
        python -m pip uninstall numpy scipy -y --quiet 2>/dev/null
        python -m pip install numpy --no-cache-dir --quiet && python -m pip install scipy --no-cache-dir --quiet || {
            log_message "❌ Critical: Failed to reinstall numpy/scipy. GraphBench may not work."
        }
    else
        log_message "✅ Scipy reinstalled successfully."
    fi
else
    log_message "✅ Scipy import successful."
fi

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

# Install missing dependencies
log_message "   Installing required dependencies..."
# Check and install python-louvain (imported as 'community')
if ! python -c "import community" 2>/dev/null; then
    log_message "     Installing python-louvain (provides 'community' module)..."
    python -m pip install "python-louvain>=0.16" --no-cache-dir --no-user --quiet 2>&1 || log_message "     ⚠️  Failed to install python-louvain"
fi

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

if ! python -c "import graphbench" 2>/dev/null; then
    log_message "   Installing graphbench-lib..."
    if ! python -m pip install graphbench-lib --no-cache-dir --no-user 2>&1; then
        log_message "⚠️  Failed to install graphbench-lib, continuing anyway..."
    else
        log_message "✅ graphbench-lib installed successfully."
    fi
else
    log_message "   ✅ graphbench-lib already installed"
fi

# Install project in development mode (if not already installed)
# Use --no-deps to avoid rebuilding torch-cluster and other compiled packages
if ! python -c "import graph_moes" 2>/dev/null; then
    log_message "📦 Installing graph_moes project (without dependencies)..."
    # Try installing without dependencies first (dependencies should already be in environment)
    if pip install -e . --no-deps --quiet 2>/dev/null; then
        log_message "✅ Project installed (no-deps mode)"
    # If that fails, try adding src to PYTHONPATH as fallback
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
    # Verify submodules can be imported (sometimes package installs but submodules don't work)
    if ! python -c "from graph_moes.experiments.track_avg_accuracy import load_and_plot_average_per_graph" 2>/dev/null; then
        log_message "⚠️  graph_moes installed but submodules not importable, adding src to PYTHONPATH..."
        export PYTHONPATH="$(pwd)/src:$PYTHONPATH"
        # Verify it works now
        if ! python -c "from graph_moes.experiments.track_avg_accuracy import load_and_plot_average_per_graph" 2>/dev/null; then
            log_message "❌ Failed to import graph_moes submodules even with PYTHONPATH"
            log_message "   Attempting to reinstall package..."
            pip install -e . --no-deps --force-reinstall --quiet 2>&1 || log_message "⚠️  Reinstall failed, continuing anyway..."
        else
            log_message "✅ Submodules now accessible via PYTHONPATH"
        fi
    fi
fi

# Quick verification
python -c "import numpy, pandas, torch, graph_moes; print('✅ Core packages available')" || {
    log_message "❌ Core packages not available"
    exit 1
}

# Define all layer type combinations
declare -a single_layer_types=(
    # "GCN"
    # "GIN" 
    # "SAGE"
    # "MLP"
    # "Unitary"
    "GPS"
)

declare -a moe_combinations=(
    '["GCN", "GIN"]'
    '["GCN", "SAGE"]'
    '["GCN", "Unitary"]'
    '["GIN", "SAGE"]'
    '["GIN", "Unitary"]'
    '["SAGE", "Unitary"]'
    '["SAGE", "GPS"]'
    '["Unitary", "GPS"]'
    '["GCN", "GPS"]'
    '["GIN", "GPS"]'
)

# Load hyperparameter lookup function
source /n/holylabs/LABS/mweber_lab/Everyone/rpellegrin/graph_moes/bash_interface/cluster/hyperparams_lookup.sh

# Define datasets - each will use optimal hyperparameters from research paper
# GraphBench datasets excluded for now (to avoid download attempts)
# EXCLUDED: mnist and cifar (slow datasets)
datasets=(enzymes proteins mutag imdb collab reddit)

# Calculate which experiment this array task should run
task_id=${SLURM_ARRAY_TASK_ID:-1}

# Encoding variants: None, hg_ldp, hg_frc, hg_rwpe_we_k20, hg_lape_normalized_k8, g_ldp, g_rwpe_k16, g_lape_k8, g_orc
declare -a dataset_encodings=("hg_lape_normalized_k8" "hg_rwpe_we_k20" "g_rwpe_k16" "g_lape_k8" "None")
num_encoding_variants=${#dataset_encodings[@]}

# Base experiments per encoding variant: 120 (60 × 2 normalization variants)
#   - 12 single layer experiments (6 × 2 normalization variants):
#     - GPS: 6 datasets × 1 (no skip) × 2 (norm/no-norm) = 12
#   - 108 MoE experiments (54 × 2 normalization variants): 9 combinations × 6 datasets × 1 router type (GNN) × 2 (norm/no-norm)
base_experiments_per_variant=120

# Calculate which encoding variant and base experiment this task corresponds to
encoding_variant_idx=$(((task_id - 1) / base_experiments_per_variant))
base_experiment_id=$(((task_id - 1) % base_experiments_per_variant + 1))

dataset_encoding=${dataset_encodings[$encoding_variant_idx]}

log_message "📦 Task $task_id: Using dataset_encoding=$dataset_encoding (variant $((encoding_variant_idx + 1))/$num_encoding_variants), base_experiment=$base_experiment_id"

# Calculate normalization variant (0 = no norm, 1 = norm) and adjust base experiment ID
# Normalization variant cycles through base experiments
normalize_variant=$(( (base_experiment_id - 1) % 2 ))
actual_base_experiment_id=$(( (base_experiment_id - 1) / 2 + 1 ))

use_normalize=$([ "$normalize_variant" -eq 1 ] && echo "true" || echo "false")
log_message "📦 Normalization variant: $normalize_variant (normalize=$use_normalize), actual_base_experiment=$actual_base_experiment_id"

if [ "$actual_base_experiment_id" -le 6 ]; then
    # Single layer experiment (GPS only)
    # GPS: 6 datasets (normalization handled separately)
    experiment_type="single"
    adjusted_id=$((actual_base_experiment_id - 1))
    
    # Calculate dataset
    num_datasets=${#datasets[@]}
    
    # Only GPS: 6 datasets
    layer_type="GPS"
    dataset_idx=$adjusted_id  # Should be 0-5
    skip_variant=0  # GPS doesn't support skip connections
    
    dataset=${datasets[$dataset_idx]}
    use_skip="false"  # GPS doesn't support skip connections
    
    log_message "🧪 Single Layer Experiment $task_id (base=$base_experiment_id, actual_base=$actual_base_experiment_id): ${dataset}_${layer_type} (skip=${use_skip}, normalize=${use_normalize}, encoding=${dataset_encoding})"
    
    # Get optimal hyperparameters
    get_hyperparams "$dataset" "$layer_type"
    
    # Extract hyperparameters
    learning_rate=$HYPERPARAM_LEARNING_RATE
    hidden_dim=$HYPERPARAM_HIDDEN_DIM
    num_layer=$HYPERPARAM_NUM_LAYERS
    dropout=$HYPERPARAM_DROPOUT
    patience=$HYPERPARAM_PATIENCE
    
    skip_suffix=$([ "$use_skip" = "true" ] && echo "_skip" || echo "")
    norm_suffix=$([ "$use_normalize" = "true" ] && echo "_norm" || echo "")
    encoding_suffix=$([ "$dataset_encoding" != "None" ] && echo "_${dataset_encoding}" || echo "")
    wandb_run_name="${dataset}_${layer_type}${skip_suffix}${norm_suffix}${encoding_suffix}_L${num_layer}_H${hidden_dim}_lr${learning_rate}_d${dropout}_task${task_id}"
    
    # Build command arguments
    cmd_args=(
        --num_trials 200
        --dataset "$dataset"
        --layer_type "$layer_type"
        --learning_rate "$learning_rate"
        --hidden_dim "$hidden_dim"
        --num_layers "$num_layer"
        --dropout "$dropout"
        --patience "$patience"
        --wandb_enabled
        --wandb_name "$wandb_run_name"
        --wandb_tags '["cluster", "comprehensive", "single_layer", "research_hyperparams", "dataset_encoding_'${dataset_encoding}'"]'
    )
    
    # Add dataset_encoding if not None
    if [ "$dataset_encoding" != "None" ]; then
        cmd_args+=(--dataset_encoding "$dataset_encoding")
    fi
    
    # Add skip_connection flag if applicable
    if [ "$use_skip" = "true" ]; then
        cmd_args+=(--skip_connection)
    fi
    
    # Add normalize_features flag if applicable
    if [ "$use_normalize" = "true" ]; then
        cmd_args+=(--normalize_features)
    fi
    
    # Run single layer experiment
    python scripts/experiments/run_graph_classification.py "${cmd_args[@]}"

else
    # MoE experiment
    experiment_type="moe"
    moe_id=$((actual_base_experiment_id - 7))  # Adjust for 6 GPS experiments (GPS only, norm handled separately)
    
    # Calculate dataset and MoE combination
    # MoE experiments are organized as: 9 combinations × 6 datasets × 1 router type (GNN)
    num_datasets=${#datasets[@]}
    num_moe_combinations=${#moe_combinations[@]}
    
    # Calculate dataset and MoE combination (no router cycling, only GNN)
    dataset_idx=$((moe_id % num_datasets))
    combo_idx=$((moe_id / num_datasets))
    
    dataset=${datasets[$dataset_idx]}
    layer_combo=${moe_combinations[$combo_idx]}
    
    # Set router type to GNN (fixed)
    router_type="GNN"
    router_layer_type="GIN"  # Default for GNN router
    
    log_message "🧪 MoE Experiment $task_id (base=$base_experiment_id, actual_base=$actual_base_experiment_id): ${dataset}_MoE_${layer_combo} (router=${router_type}, normalize=${use_normalize}, encoding=${dataset_encoding})"
    
    # For MoE, use the first layer type in combination as base for hyperparameters
    first_layer=$(echo "$layer_combo" | grep -o '"[^"]*"' | head -1 | tr -d '"')
    get_hyperparams "$dataset" "$first_layer"
    
    # Extract hyperparameters
    learning_rate=$HYPERPARAM_LEARNING_RATE
    hidden_dim=$HYPERPARAM_HIDDEN_DIM
    num_layer=$HYPERPARAM_NUM_LAYERS
    dropout=$HYPERPARAM_DROPOUT
    patience=$HYPERPARAM_PATIENCE
    
    # Create a clean name from layer combination
    clean_combo=$(echo "$layer_combo" | tr -d '[]",' | tr ' ' '_')
    encoding_suffix=$([ "$dataset_encoding" != "None" ] && echo "_${dataset_encoding}" || echo "")
    router_suffix="_${router_type}"
    norm_suffix=$([ "$use_normalize" = "true" ] && echo "_norm" || echo "")
    wandb_run_name="${dataset}_MoE_${clean_combo}${router_suffix}${norm_suffix}${encoding_suffix}_L${num_layer}_H${hidden_dim}_lr${learning_rate}_d${dropout}_task${task_id}"
    
    # Build command arguments for MoE
    moe_cmd_args=(
        --num_trials 200
        --dataset "$dataset"
        --layer_types "$layer_combo"
        --router_type "$router_type"
        --router_layer_type "$router_layer_type"
        --learning_rate "$learning_rate"
        --hidden_dim "$hidden_dim"
        --num_layers "$num_layer"
        --dropout "$dropout"
        --patience "$patience"
        --wandb_enabled
        --wandb_name "$wandb_run_name"
        --wandb_tags '["cluster", "comprehensive", "moe", "research_hyperparams", "dataset_encoding_'${dataset_encoding}'", "router_'${router_type}'"]'
    )
    
    # Add dataset_encoding if not None
    if [ "$dataset_encoding" != "None" ]; then
        moe_cmd_args+=(--dataset_encoding "$dataset_encoding")
    fi
    
    # Add normalize_features flag if applicable
    if [ "$use_normalize" = "true" ]; then
        moe_cmd_args+=(--normalize_features)
    fi
    
    # Run MoE experiment
    python scripts/experiments/run_graph_classification.py "${moe_cmd_args[@]}"
fi

# Check exit status
if [ $? -eq 0 ]; then
    log_message "✅ Task $task_id ($experiment_type) completed successfully"
else
    log_message "❌ Task $task_id ($experiment_type) failed with exit code $?"
    exit 1
fi

log_message "🎉 Task $task_id completed!"
