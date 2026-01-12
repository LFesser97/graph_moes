#!/bin/bash
# ============================================================================
# Compute Encodings - SLURM Array Job Version
# ============================================================================
# This script computes graph and hypergraph encodings using SLURM array jobs
# for maximum parallelization. Each array task computes a single (dataset, level, encoding)
# combination.
#
# Total tasks = num_datasets × num_levels × num_encodings_per_level
#   - Datasets: mutag, enzymes, proteins, imdb_binary, collab, reddit_binary,
#               mnist, cifar10, pattern, molhiv, molpcba, ppa (11-12 datasets)
#   - Graph encodings: ldp, rwpe, lape, orc (4 types)
#   - Hypergraph encodings: ldp, frc, rwpe, lape, orc (5 types)
#
# This allows full parallel execution - each encoding computation runs independently.
#
# Usage: sbatch compute_encodings_array.sh
# ============================================================================

#SBATCH --job-name=compute_encodings
#SBATCH --array=1-99              # Total: 11 datasets × (4 graph + 5 hypergraph) = 99
#SBATCH --ntasks=1
#SBATCH --time=72:00:00           # Long time for slow encodings (ORC especially)
#SBATCH --mem=128GB               # Increased memory for large datasets (COLLAB ORC needs this)
#SBATCH --output=logs_encodings/compute_encodings_array_%A_%a.log  # %A = job ID, %a = task ID
#SBATCH --partition=mweber_gpu
#SBATCH --cpus-per-task=8

# Change to script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/../.." || exit 1

# Function to log messages with timestamp
log_message() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [Task $SLURM_ARRAY_TASK_ID] $1"
}

log_message "Starting encoding computation task $SLURM_ARRAY_TASK_ID"

# Set environment path
export CONDA_ENVS_PATH=/n/holylabs/LABS/mweber_lab/Everyone/rpellegrin/conda/envs
export CONDA_PKGS_DIRS=/n/holylabs/LABS/mweber_lab/Everyone/rpellegrin/conda/pkgs

# Load Python module (provides mamba/conda infrastructure)
log_message "📦 Loading Python module..."
module load python/3.10.12-fasrc01 || {
    log_message "⚠️  Failed to load python module, trying alternative..."
    module load python/3.10.12-fasrc02 || {
        log_message "❌ Failed to load python module"
        exit 1
    }
}

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

# Activate conda environment
ENV_NAME="moe_fresh"
log_message "🔧 Activating $ENV_NAME environment..."
if [ -d "$CONDA_ENVS_PATH/$ENV_NAME/bin" ] && [ -f "$CONDA_ENVS_PATH/$ENV_NAME/bin/python" ]; then
    export PATH="$CONDA_ENVS_PATH/$ENV_NAME/bin:$PATH"
    export CONDA_DEFAULT_ENV=$ENV_NAME
    export PYTHONNOUSERSITE=1
    log_message "✅ Environment activated"
else
    log_message "❌ Failed to activate conda environment"
    exit 1
fi

# Set PYTHONPATH (project root is now current directory)
export PYTHONPATH="$(pwd)/src:${PYTHONPATH}"

# Define datasets (matching what load_all_datasets loads)
# TU datasets (short names)
datasets=("mutag" "enzymes" "proteins" "imdb_binary" "collab" "reddit_binary")
# GNN Benchmark datasets
datasets+=("mnist" "cifar10" "pattern")
# OGB datasets (short names)
datasets+=("molhiv" "molpcba" "ppa")

num_datasets=${#datasets[@]}

# Define encoding levels and their encodings
# Graph encodings: ldp, rwpe, lape, orc
graph_encodings=("ldp" "rwpe" "lape" "orc")
num_graph_encodings=${#graph_encodings[@]}

# Hypergraph encodings: ldp, frc, rwpe, lape, orc
hypergraph_encodings=("ldp" "frc" "rwpe" "lape" "orc")
num_hypergraph_encodings=${#hypergraph_encodings[@]}

# Calculate which (dataset, level, encoding) combination this task should compute
task_id=${SLURM_ARRAY_TASK_ID:-1}

# Total combinations per dataset: graph encodings (4) + hypergraph encodings (5) = 9
encodings_per_dataset=$((num_graph_encodings + num_hypergraph_encodings))

# Calculate dataset index and encoding index
dataset_idx=$(((task_id - 1) / encodings_per_dataset))
encoding_idx=$(((task_id - 1) % encodings_per_dataset))

# Check if dataset index is valid
if [ "$dataset_idx" -ge "$num_datasets" ]; then
    log_message "⚠️  Task $task_id: Dataset index $dataset_idx out of range (max: $((num_datasets - 1)))"
    log_message "   Skipping this task"
    exit 0
fi

dataset=${datasets[$dataset_idx]}

# Determine level and encoding
if [ "$encoding_idx" -lt "$num_graph_encodings" ]; then
    # Graph encoding
    level="graph"
    encoding=${graph_encodings[$encoding_idx]}
else
    # Hypergraph encoding
    level="hypergraph"
    encoding_idx_adjusted=$((encoding_idx - num_graph_encodings))
    encoding=${hypergraph_encodings[$encoding_idx_adjusted]}
fi

log_message "📦 Task $task_id: Computing $level encoding '$encoding' for dataset '$dataset'"
log_message "   Dataset index: $dataset_idx/$((num_datasets - 1)), Encoding index: $encoding_idx/$((encodings_per_dataset - 1))"

# Check for Hypergraph_Encodings repo if needed
if [ "$level" = "hypergraph" ]; then
    HYPERGRAPH_REPO="/n/holylabs/mweber_lab/Everyone/rpellegrin/Hypergraph_Encodings"
    if [ ! -d "$HYPERGRAPH_REPO" ]; then
        log_message "⚠️  Hypergraph_Encodings repo not found at $HYPERGRAPH_REPO"
        log_message "   Skipping hypergraph encoding computation"
        exit 0
    fi
    
    # Add to PYTHONPATH
    export PYTHONPATH="${HYPERGRAPH_REPO}/src:${PYTHONPATH}"
fi

# Run the encoding computation
log_message "🚀 Running encoding computation..."

python scripts/compute_encodings_for_datasets.py \
    --level "$level" \
    --encoding "$encoding" \
    --dataset "$dataset"

exit_code=$?

if [ $exit_code -eq 0 ]; then
    log_message "✅ Task $task_id completed successfully"
else
    log_message "❌ Task $task_id failed with exit code: $exit_code"
    exit $exit_code
fi
