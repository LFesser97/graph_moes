# GeoML lab:

## Lab spaces/scratch spaces

scratch space:

```
/n/netscratch/mweber_lab
```

lab space:

```
/n/holylabs/LABS/mweber_lab
```

## Checking space:

```
df -h /n/home04/rpellegrinext/
```

```
df -h /n/netscratch/mweber_lab/
```

## Exporting the environment

We should remember to run:

# Export environment to YAML file (recommended)
```
mamba env export > environment.yml
```
# or
```
conda env export > environment.yml
```


# From FASRC Documentation - Official Guidelines

# ========================================
# FASRC PYTHON PACKAGE INSTALLATION GUIDE
# ========================================

# 1. MAMBA is the recommended package manager (faster than conda)
#    - Drop-in replacement for conda
#    - Uses conda-forge channel by default (free)
#    - Better dependency resolution

# 2. INTERACTIVE SESSION REQUIRED for environment creation
salloc --partition test --nodes=1 --cpus-per-task=2 --mem=4GB --time=0-02:00:00

# 3. LOAD PYTHON MODULE (not anaconda)
module load python/{PYTHON_VERS}-fasrc01
# Example: module load python/3.11.0-fasrc01

# 4. RECOMMENDED: Set package/environment locations for Weber Lab
# Use lab directory for better performance and sharing:
export CONDA_PKGS_DIRS=/n/holylabs/LABS/mweber_lab/Everyone/rpellegrin/conda/pkgs
export CONDA_ENVS_PATH=/n/holylabs/LABS/mweber_lab/Everyone/rpellegrin/conda/envs

# Alternative: Use scratch space for temporary environments
# export CONDA_PKGS_DIRS=/n/netscratch/mweber_lab/Lab/conda/pkgs
# export CONDA_ENVS_PATH=/n/netscratch/mweber_lab/Lab/conda/envs

# 5. CREATE ENVIRONMENT with mamba
# Option A: Named environment in personal lab space
mamba create -n moe python=3.10 pip wheel

# Option B: Shared environment using prefix (for lab-wide access)
mamba create --prefix /n/holylabs/LABS/mweber_lab/Everyone/rpellegrin/envs/moe python=3.10

# 6. ACTIVATE ENVIRONMENT
# For named environment:
source activate moe

# For prefix environment:
source activate /n/holylabs/LABS/mweber_lab/Everyone/rpellegrin/envs/moe

# 7. INSTALL PACKAGES - Use mamba for most packages (faster than pip)
mamba install -y numpy pandas tqdm
mamba install -y pytorch pytorch-geometric -c pytorch -c pyg

# 8. PIP INSTALLS - Only for packages not available in conda-forge
# NEVER use pip outside of mamba environment
pip install wandb attrdict

# 9. WEBER LAB SPECIFIC PATHS
# Personal environment: /n/holylabs/LABS/mweber_lab/Everyone/rpellegrin/envs/moe
# Shared lab environment: /n/holylabs/LABS/mweber_lab/shared_envs/moe
# Scratch space: /n/netscratch/mweber_lab/Lab/envs/moe

# 10. JUPYTER INTEGRATION - Install these for notebook support
mamba install ipykernel nb_conda_kernels

# ========================================
# TROUBLESHOOTING (from FASRC)
# ========================================

# Problem: Works in interactive but fails in batch jobs
# Solution: Don't submit jobs from within mamba environment
# - Either deactivate environment before sbatch
# - Or open new terminal to submit jobs

# Problem: conda initialize in ~/.bashrc
# Solution: Remove conda initialize section from ~/.bashrc

# ========================================
# BEST PRACTICES (from FASRC)
# ========================================

# - Use interactive sessions for environment creation
# - Use shared lab directories (not home) for better performance
# - Use mamba instead of conda
# - Use conda-forge channel (default in mamba)
# - Never use pip outside mamba environments
# - Don't submit jobs from within activated environments
