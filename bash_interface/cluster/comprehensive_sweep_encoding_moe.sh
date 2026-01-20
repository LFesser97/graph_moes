#!/bin/bash
# ============================================================================
# Comprehensive EncodingMoE Sweep - Parallel Array Job Version
# ============================================================================
# This script runs a comprehensive hyperparameter sweep for EncodingMoE
# (Encoding Mixture of Experts) experiments using SLURM array jobs for parallel
# execution. EncodingMoE uses a router to dynamically select and combine
# different structural encodings before passing them to a GNN model.
#
# This version EXCLUDES mnist and cifar datasets (which are slow to run).
#
# The script uses optimal hyperparameters from research papers for each dataset
# and model combination, loaded from hyperparams_lookup.sh.
#
# Total experiments: TBD (calculated based on combinations)
#   - Base model types: GCN, GIN, SAGE, MLP, Unitary, GPS (6 types)
#   - Encoding combinations: multiple pairs of encodings
#   - Router types: MLP, GNN (2 types)
#   - Skip connections: yes/no (for GCN, GIN, SAGE only)
#   - Normalization: true/false (2 variants)
#   - Datasets: 6 (enzymes, proteins, mutag, imdb, collab, reddit)
#
# Usage: sbatch comprehensive_sweep_encoding_moe.sh
# ============================================================================

#SBATCH --job-name=encoding_moe_sweep
#SBATCH --array=1-2376            # Total: 2376 experiments (see calculation below)
#SBATCH --ntasks=1
#SBATCH --time=192:00:00           # Long time for comprehensive sweep
#SBATCH --mem=128GB               # Sufficient memory
#SBATCH --output=logs_comprehensive/Parallel_encoding_moe_sweep_%A_%a.log
#SBATCH --partition=mweber_gpu
#SBATCH --gpus=1
#SBATCH --nice=0                  # Higher priority

# WandB Environment Setup
echo "🚀 Setting up WandB environment for EncodingMoE experiments..."

export WANDB_API_KEY="ea7c6eeb5a095b531ef60cc784bfeb87d47ea0b0"
export WANDB_ENTITY="weber-geoml-harvard-university"
export WANDB_PROJECT="MOE_4"
WANDB_TMP_DIR="${TMPDIR:-/tmp}/wandb_${SLURM_JOB_ID:-$$}"
export WANDB_DIR="${WANDB_TMP_DIR}"
export WANDB_CACHE_DIR="${WANDB_TMP_DIR}/.cache"
export WANDB_DISABLE_CODE=true
export WANDB_SYNC_MODE="now"

mkdir -p "${WANDB_TMP_DIR}" logs logs_comprehensive

export PYTHONNOUSERSITE=1

# Load CUDA modules if available
if command -v module &> /dev/null; then
    echo "📦 Loading CUDA modules..."
    if module load cuda/12.9.1-fasrc01 2>/dev/null; then
        echo "   ✅ Loaded cuda/12.9.1-fasrc01"
    elif module load cuda 2>/dev/null; then
        echo "   ✅ Loaded cuda (default version)"
    fi
    
    if module load cudnn/9.10.2.21_cuda12-fasrc01 2>/dev/null; then
        echo "   ✅ Loaded cudnn/9.10.2.21_cuda12-fasrc01"
    elif module load cudnn 2>/dev/null; then
        echo "   ✅ Loaded cudnn (default version)"
    fi
    
    if [ -n "$CUDA_HOME" ]; then
        export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${CUDA_HOME}/lib:${CUDA_HOME}/targets/x86_64-linux/lib:${LD_LIBRARY_PATH}"
    elif [ -n "$CUDA_ROOT" ]; then
        export LD_LIBRARY_PATH="${CUDA_ROOT}/lib64:${CUDA_ROOT}/lib:${CUDA_ROOT}/targets/x86_64-linux/lib:${LD_LIBRARY_PATH}"
    fi
fi

echo "✅ WandB environment configured"

# Install wandb if not already installed
if ! python -c "import wandb" &> /dev/null; then
    echo "📦 Installing wandb..."
    pip install wandb
else
    echo "✅ wandb already installed"
fi

# Function to log messages with timestamp
log_message() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [Task $SLURM_ARRAY_TASK_ID] $1"
}

log_message "Starting EncodingMoE Sweep Task $SLURM_ARRAY_TASK_ID"

# Set environment path
export CONDA_ENVS_PATH=/n/holylabs/LABS/mweber_lab/Everyone/rpellegrin/conda/envs
export CONDA_PKGS_DIRS=/n/holylabs/LABS/mweber_lab/Everyone/rpellegrin/conda/pkgs

# Load Python module
log_message "📦 Loading Python module..."
module load python/3.10.12-fasrc01 || {
    module load python/3.10-fasrc01 || {
        log_message "❌ Failed to load Python module"
        exit 1
    }
}

# Verify mamba/conda is available
if ! command -v mamba &> /dev/null && ! command -v conda &> /dev/null; then
    log_message "❌ Neither mamba nor conda available"
    exit 1
fi

log_message "✅ Python module loaded: $(which python)"

# Initialize conda/mamba
log_message "🔧 Initializing conda/mamba..."
if [ -f "$(conda info --base 2>/dev/null)/etc/profile.d/conda.sh" ]; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
elif [ -f "/n/sw/Mambaforge-23.3.1-1/etc/profile.d/conda.sh" ]; then
    source "/n/sw/Mambaforge-23.3.1-1/etc/profile.d/conda.sh"
fi

# Activate environment
ENV_NAME="moe_fresh"
log_message "🔧 Activating $ENV_NAME environment..."

activation_success=false
if [ -d "$CONDA_ENVS_PATH/$ENV_NAME/bin" ] && [ -f "$CONDA_ENVS_PATH/$ENV_NAME/bin/python" ]; then
    export PATH="$CONDA_ENVS_PATH/$ENV_NAME/bin:$PATH"
    export CONDA_DEFAULT_ENV=$ENV_NAME
    export PYTHONNOUSERSITE=1
    python_path=$(which python)
    if [[ "$python_path" == *"$ENV_NAME"* ]]; then
        log_message "✅ Environment activated"
        activation_success=true
    fi
fi

if [ "$activation_success" = false ]; then
    log_message "❌ Failed to activate $ENV_NAME environment"
    exit 1
fi

log_message "✅ Verified $ENV_NAME environment active: $(which python)"

# Navigate to project directory
cd /n/holylabs/LABS/mweber_lab/Everyone/rpellegrin/graph_moes || {
    log_message "❌ Failed to navigate to project directory"
    exit 1
}

log_message "📁 Project directory: $(pwd)"

# Add project root and src to PYTHONPATH
export PYTHONPATH="$(pwd):$(pwd)/src:${PYTHONPATH}"

# Check if scipy is working
log_message "🔍 Verifying scipy installation..."
if ! python -c "import scipy.signal" 2>/dev/null; then
    log_message "❌ Scipy import failed, attempting to reinstall..."
    python -m pip uninstall scipy -y --quiet 2>/dev/null || true
    python -m pip install scipy --no-cache-dir --quiet 2>&1 || {
        log_message "⚠️  Failed to reinstall scipy"
    }
fi

# Install required packages
log_message "📦 Checking and installing required packages..."

NUMPY_VERSION=$(python -c "import numpy; print(numpy.__version__)" 2>/dev/null | cut -d. -f1)
if [ "$NUMPY_VERSION" = "2" ]; then
    log_message "   ⚠️  NumPy 2.x detected, downgrading to <2.0..."
    python -m pip install "numpy>=1.23.1,<2.0" --no-cache-dir --no-user --quiet 2>&1 || true
fi

if ! python -c "import attrdict3" 2>/dev/null; then
    log_message "   Installing attrdict3..."
    python -m pip install attrdict3 --no-cache-dir --no-user --quiet 2>&1 || true
fi

# Install missing dependencies
MISSING_DEPS=(
    "graphriccicurvature>=0.5.3.1:GraphRicciCurvature"
    "numba>=0.56.4:numba"
    "networkit>=10.1:networkit"
    "cvxpy>=1.4.1:cvxpy"
    "pot>=0.9.0:ot"
    "torcheval>=0.0.7:torcheval"
)
for dep_spec in "${MISSING_DEPS[@]}"; do
    DEP=$(echo "$dep_spec" | cut -d':' -f1)
    IMPORT_NAME=$(echo "$dep_spec" | cut -d':' -f2)
    if ! python -c "import $IMPORT_NAME" 2>/dev/null; then
        PACKAGE_NAME=$(echo "$DEP" | cut -d'>' -f1 | cut -d'=' -f1)
        log_message "     Installing $PACKAGE_NAME..."
        python -m pip install "$DEP" --no-cache-dir --no-user --quiet 2>&1 || true
    fi
done

# Install project in development mode
if ! python -c "import graph_moes" 2>/dev/null; then
    log_message "📦 Installing graph_moes project..."
    if pip install -e . --no-deps --quiet 2>/dev/null; then
        log_message "✅ Project installed"
    elif [ -d "src" ]; then
        export PYTHONPATH="$(pwd)/src:$PYTHONPATH"
        log_message "✅ Project accessible via PYTHONPATH"
    else
        log_message "❌ Failed to install graph_moes project"
        exit 1
    fi
else
    log_message "✅ graph_moes already installed"
    if ! python -c "from graph_moes.experiments.track_avg_accuracy import load_and_plot_average_per_graph" 2>/dev/null; then
        export PYTHONPATH="$(pwd)/src:$PYTHONPATH"
    fi
fi

# Quick verification
python -c "import numpy, pandas, torch, graph_moes; print('✅ Core packages available')" || {
    log_message "❌ Core packages not available"
    exit 1
}

# Load hyperparameter lookup function
source /n/holylabs/LABS/mweber_lab/Everyone/rpellegrin/graph_moes/bash_interface/cluster/hyperparams_lookup.sh

# Define datasets (excluding slow datasets)
datasets=(enzymes proteins mutag imdb collab reddit)

# Define base layer types
declare -a layer_types=("GCN" "GIN" "SAGE" "MLP" "Unitary" "GPS")

# Define encoding combinations for EncodingMoE
# Format: "encoding1,encoding2" (will be converted to JSON array)
declare -a encoding_combinations=(
    "g_ldp,g_orc"
    "g_ldp,None"
    "g_orc,None"
    "g_ldp,g_rwpe_k16"
    "g_orc,g_rwpe_k16"
    "g_rwpe_k16,g_lape_k8"
    "hg_ldp,hg_orc"
    "hg_ldp,None"
    "hg_orc,None"
    "hg_ldp,hg_frc"
    "hg_frc,hg_orc"
)

# Define router types
declare -a router_types=("MLP" "GNN")

# Calculate experiment dimensions
num_datasets=${#datasets[@]}
num_layer_types=${#layer_types[@]}
num_encoding_combos=${#encoding_combinations[@]}
num_router_types=${#router_types[@]}
num_skip_variants=2  # true/false (but only for GCN, GIN, SAGE)
num_norm_variants=2  # true/false

# Calculate total experiments
# Structure: dataset × layer_type × encoding_combo × router_type × skip × norm
# But skip is only applicable to GCN, GIN, SAGE
# For those: 6 datasets × 3 layer_types × 2 skip variants × 11 encoding_combos × 2 router_types × 2 norm variants = 6 × 3 × 2 × 11 × 2 × 2 = 1584
# For MLP, Unitary, GPS: 6 datasets × 3 layer_types × 1 skip variant × 11 encoding_combos × 2 router_types × 2 norm variants = 6 × 3 × 1 × 11 × 2 × 2 = 792
# Total = 1584 + 792 = 2376

# But let's use a simpler structure: cycle through all combinations
# Order: dataset, layer_type, encoding_combo, router_type, skip, norm
task_id=${SLURM_ARRAY_TASK_ID:-1}

# Calculate indices for each dimension
task_idx=$((task_id - 1))

# Normalization (innermost, cycles fastest): 0 = false, 1 = true
norm_idx=$((task_idx % num_norm_variants))
task_idx=$((task_idx / num_norm_variants))

# Skip connection: 0 = false, 1 = true (only used for GCN, GIN, SAGE)
skip_idx=$((task_idx % num_skip_variants))
task_idx=$((task_idx / num_skip_variants))

# Router type: 0 = MLP, 1 = GNN
router_idx=$((task_idx % num_router_types))
task_idx=$((task_idx / num_router_types))

# Encoding combination
encoding_combo_idx=$((task_idx % num_encoding_combos))
task_idx=$((task_idx / num_encoding_combos))

# Layer type
layer_type_idx=$((task_idx % num_layer_types))
task_idx=$((task_idx / num_layer_types))

# Dataset (outermost)
dataset_idx=$((task_idx % num_datasets))

# Get values
dataset=${datasets[$dataset_idx]}
layer_type=${layer_types[$layer_type_idx]}
encoding_combo_str=${encoding_combinations[$encoding_combo_idx]}
router_type=${router_types[$router_idx]}

# Skip connections only apply to GCN, GIN, SAGE
# For layer types that don't support skip, force skip_idx to 0
if [ "$layer_type" != "GCN" ] && [ "$layer_type" != "GIN" ] && [ "$layer_type" != "SAGE" ]; then
    skip_idx=0
fi

use_skip=$([ "$skip_idx" -eq 1 ] && echo "true" || echo "false")
use_norm=$([ "$norm_idx" -eq 1 ] && echo "true" || echo "false")

# Convert encoding combination string to JSON array
# "g_ldp,g_orc" -> ["g_ldp", "g_orc"]
IFS=',' read -ra ENC_PARTS <<< "$encoding_combo_str"
encoding_json="["
for i in "${!ENC_PARTS[@]}"; do
    if [ $i -gt 0 ]; then
        encoding_json+=", "
    fi
    enc="${ENC_PARTS[$i]}"
    # Trim whitespace
    enc=$(echo "$enc" | xargs)
    if [ "$enc" = "None" ]; then
        encoding_json+="null"
    else
        encoding_json+="\"$enc\""
    fi
done
encoding_json+="]"

log_message "🧪 EncodingMoE Experiment $task_id:"
log_message "   Dataset: $dataset"
log_message "   Layer Type: $layer_type"
log_message "   Encoding Combo: $encoding_combo_str ($encoding_json)"
log_message "   Router Type: $router_type"
log_message "   Skip: $use_skip"
log_message "   Normalize: $use_norm"

# Get optimal hyperparameters
get_hyperparams "$dataset" "$layer_type"

# Extract hyperparameters
learning_rate=$HYPERPARAM_LEARNING_RATE
hidden_dim=$HYPERPARAM_HIDDEN_DIM
num_layer=$HYPERPARAM_NUM_LAYERS
dropout=$HYPERPARAM_DROPOUT
patience=$HYPERPARAM_PATIENCE

# Build command arguments for EncodingMoE
skip_suffix=$([ "$use_skip" = "true" ] && echo "_skip" || echo "")
norm_suffix=$([ "$use_norm" = "true" ] && echo "_norm" || echo "")
encoding_label=$(echo "$encoding_combo_str" | tr ',' '+' | tr '_' '-')
router_suffix="_${router_type}"
wandb_run_name="${dataset}_EncodingMoE_${layer_type}${skip_suffix}${norm_suffix}_enc${encoding_label}${router_suffix}_L${num_layer}_H${hidden_dim}_lr${learning_rate}_d${dropout}_task${task_id}"

cmd_args=(
    --num_trials 200
    --dataset "$dataset"
    --layer_type "$layer_type"
    --encoding_moe_encodings "$encoding_json"
    --encoding_moe_router_type "$router_type"
    --learning_rate "$learning_rate"
    --hidden_dim "$hidden_dim"
    --num_layers "$num_layer"
    --dropout "$dropout"
    --patience "$patience"
    --wandb_enabled
    --wandb_name "$wandb_run_name"
    --wandb_tags "[\"cluster\", \"encoding_moe\", \"research_hyperparams\", \"router_${router_type}\"]"
)

# Add skip_connection flag if applicable
if [ "$use_skip" = "true" ]; then
    cmd_args+=(--skip_connection)
fi

# Add normalize_features flag if applicable
if [ "$use_norm" = "true" ]; then
    cmd_args+=(--normalize_features)
fi

# Run EncodingMoE experiment
log_message "🚀 Running EncodingMoE experiment..."
python scripts/experiments/run_graph_classification.py "${cmd_args[@]}"

# Check exit status
if [ $? -eq 0 ]; then
    log_message "✅ Task $task_id completed successfully"
else
    log_message "❌ Task $task_id failed with exit code $?"
    exit 1
fi

log_message "🎉 Task $task_id completed!"
