#!/bin/bash
#SBATCH --job-name=moe_gcn_gin_array
#SBATCH --array=1-96              # Total combinations: 2 datasets × 2 lr × 2 hidden × 4 layers × 3 dropout = 96
#SBATCH --ntasks=1
#SBATCH --time=8:00:00           # Shorter time per individual job
#SBATCH --mem=16GB               # Less memory per job
#SBATCH --output=logs/moe_gcn_gin_%A_%a.log  # %A = array job ID, %a = task ID
#SBATCH --partition=mweber_gpu
#SBATCH --gpus=1

# WandB Environment Setup for Graph MoE Experiments
echo "🚀 Setting up WandB environment for Graph MoE experiments..."

# Set WandB environment variables
export WANDB_API_KEY="ea7c6eeb5a095b531ef60cc784bfeb87d47ea0b0"
export WANDB_ENTITY="weber-geoml-harvard-university"
export WANDB_PROJECT="MOE"

# Optional: Set other WandB configurations
export WANDB_DIR="./wandb"
export WANDB_CACHE_DIR="./wandb/.cache"

# Create wandb directory if it doesn't exist
mkdir -p ./wandb

echo "✅ WandB environment configured:"
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

# Create logs directory
mkdir -p logs

# Function to log messages with timestamp
log_message() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [Task $SLURM_ARRAY_TASK_ID] $1"
}

log_message "Starting MoE GCN+GIN task $SLURM_ARRAY_TASK_ID"

# Load Python module and activate mamba environment
module load python/3.10.12-fasrc01

# Set environment path and activate moe environment
export CONDA_ENVS_PATH=/n/holylabs/LABS/mweber_lab/Everyone/rpellegrin/conda/envs
source activate moe

log_message "Using moe mamba environment"

# Navigate to project directory
cd /n/netscratch/mweber_lab/Lab/graph_moes

# Define hyperparameter combinations
datasets=(enzymes proteins)
learning_rates=(0.001 0.0001)
hidden_dims=(64 128)
num_layers_list=(4 5 6 7)
dropouts=(0.0 0.1 0.2)

# Calculate which combination this task should run
# Total combinations per dataset: 2 × 2 × 4 × 3 = 48
# Task 1-48: enzymes, Task 49-96: proteins

task_id=$SLURM_ARRAY_TASK_ID
if [ $task_id -le 48 ]; then
    dataset="enzymes"
    combo_id=$((task_id - 1))
else
    dataset="proteins"
    combo_id=$((task_id - 49))
fi

# Convert combo_id to specific hyperparameters
total_lr=${#learning_rates[@]}
total_hd=${#hidden_dims[@]}
total_nl=${#num_layers_list[@]}
total_do=${#dropouts[@]}

# Calculate indices
do_idx=$((combo_id % total_do))
combo_id=$((combo_id / total_do))
nl_idx=$((combo_id % total_nl))
combo_id=$((combo_id / total_nl))
hd_idx=$((combo_id % total_hd))
lr_idx=$((combo_id / total_hd))

# Get actual values
learning_rate=${learning_rates[$lr_idx]}
hidden_dim=${hidden_dims[$hd_idx]}
num_layer=${num_layers_list[$nl_idx]}
dropout=${dropouts[$do_idx]}

log_message "Configuration: dataset=$dataset, lr=$learning_rate, hidden_dim=$hidden_dim, num_layers=$num_layer, dropout=$dropout"

# Generate wandb run name for this specific task
wandb_run_name="${dataset}_GCN_GIN_L${num_layer}_H${hidden_dim}_lr${learning_rate}_d${dropout}_task${task_id}"

log_message "WandB run name: $wandb_run_name"

# Run the experiment with wandb enabled
python run_graph_classification.py \
    --num_trials 10 \
    --dataset "$dataset" \
    --learning_rate "$learning_rate" \
    --hidden_dim "$hidden_dim" \
    --num_layers "$num_layer" \
    --dropout "$dropout" \
    --patience 50 \
    --layer_types '["GCN", "GIN"]' \
    --wandb_enabled \
    --wandb_name "$wandb_run_name" \
    --wandb_tags '["cluster", "sweep", "gcn_gin"]'

# Check exit status
if [ $? -eq 0 ]; then
    log_message "✅ Task completed successfully"
else
    log_message "❌ Task failed with exit code $?"
    exit 1
fi