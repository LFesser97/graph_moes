#!/bin/bash
#SBATCH --job-name=moe_uni_gin_array
#SBATCH --array=1-48              # Total combinations: 2 datasets × 2 lr × 2 hidden × 2 layers × 3 dropout = 48
#SBATCH --ntasks=1
#SBATCH --time=8:00:00
#SBATCH --mem=16GB
#SBATCH --output=logs/moe_uni_gin_%A_%a.log  # %A = array job ID, %a = task ID
#SBATCH --partition=mweber_gpu
#SBATCH --gpus=1

# Create logs directory
mkdir -p logs

# Function to log messages with timestamp
log_message() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [Task $SLURM_ARRAY_TASK_ID] $1"
}

log_message "Starting MoE GIN+Unitary task $SLURM_ARRAY_TASK_ID"

# Load environment
module load anaconda/2023.07
source activate borf

# Navigate to project directory
cd /n/netscratch/mweber_lab/Lab/graph_moes

# Define hyperparameter combinations
datasets=(proteins mutag)
learning_rates=(0.001 0.0001)
hidden_dims=(64 128)
num_layers_list=(4 6)
dropouts=(0.0 0.1 0.2)

# Calculate which combination this task should run
# Total combinations per dataset: 2 × 2 × 2 × 3 = 24
# Task 1-24: proteins, Task 25-48: mutag

task_id=$SLURM_ARRAY_TASK_ID
if [ $task_id -le 24 ]; then
    dataset="proteins"
    combo_id=$((task_id - 1))
else
    dataset="mutag"
    combo_id=$((task_id - 25))
fi

# Convert combo_id to specific hyperparameters
total_lr=${#learning_rates[@]}
total_hd=${#hidden_dims[@]}
total_nl=${#num_layers_list[@]}
total_do=${#dropouts[@]}

# Calculate indices using modular arithmetic
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

# Run the experiment
python run_graph_classification.py \
    --num_trials 5 \
    --dataset "$dataset" \
    --learning_rate "$learning_rate" \
    --hidden_dim "$hidden_dim" \
    --num_layers "$num_layer" \
    --dropout "$dropout" \
    --patience 50 \
    --layer_types '["GIN", "Unitary"]'

# Check exit status
if [ $? -eq 0 ]; then
    log_message "✅ Task completed successfully"
else
    log_message "❌ Task failed with exit code $?"
    exit 1
fi