#!/bin/bash
#SBATCH --job-name=download_ppa
#SBATCH --time=10:00:00
#SBATCH --mem=32GB
#SBATCH --partition=mweber_gpu
#SBATCH --gpus=1

cd /n/holylabs/LABS/mweber_lab/Everyone/rpellegrin/graph_moes
export PYTHONPATH="$(pwd):$(pwd)/src:${PYTHONPATH}"

python scripts/run_graph_classification.py --dataset ppa --layer_type GCN --num_trials 1
