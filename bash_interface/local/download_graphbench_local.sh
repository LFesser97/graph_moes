#!/bin/bash
# ============================================================================
# Download GraphBench Datasets Locally
# ============================================================================
# This script downloads GraphBench datasets sequentially on your local machine.
# It processes datasets one by one with comprehensive logging and verification.
#
# Usage: bash bash_interface/local/download_graphbench_local.sh
#        or: ./bash_interface/local/download_graphbench_local.sh
# ============================================================================

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to log messages with timestamp
log_message() {
    echo -e "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

log_success() {
    echo -e "${GREEN}[$(date '+%Y-%m-%d %H:%M:%S')] ✅ $1${NC}"
}

log_error() {
    echo -e "${RED}[$(date '+%Y-%m-%d %H:%M:%S')] ❌ $1${NC}"
}

log_warning() {
    echo -e "${YELLOW}[$(date '+%Y-%m-%d %H:%M:%S')] ⚠️  $1${NC}"
}

log_info() {
    echo -e "${BLUE}[$(date '+%Y-%m-%d %H:%M:%S')] ℹ️  $1${NC}"
}

# Get script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
DATA_DIRECTORY="$PROJECT_ROOT/graph_datasets"

log_message "🚀 Starting GraphBench dataset downloads (local)..."
log_info "Project root: $PROJECT_ROOT"
log_info "Download directory: $DATA_DIRECTORY"

# Create data directory if it doesn't exist
mkdir -p "$DATA_DIRECTORY"
log_success "Data directory ready: $DATA_DIRECTORY"

# Activate conda environment
ENV_NAME="moe"
log_info "Activating conda environment: $ENV_NAME..."

# Try to initialize conda
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/anaconda3/etc/profile.d/conda.sh"
elif [ -f "$(conda info --base 2>/dev/null)/etc/profile.d/conda.sh" ]; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
else
    log_warning "Could not find conda.sh, trying to use conda directly..."
fi

# Activate the environment
if command -v conda &> /dev/null; then
    if conda activate "$ENV_NAME" 2>/dev/null || source activate "$ENV_NAME" 2>/dev/null; then
        log_success "Activated conda environment: $ENV_NAME"
    else
        log_warning "Failed to activate conda environment '$ENV_NAME', continuing with system Python..."
        log_warning "This may cause dependency issues. Make sure $ENV_NAME environment exists."
    fi
else
    log_warning "conda command not found, using system Python..."
fi

# Check if Python is available
if ! command -v python3 &> /dev/null && ! command -v python &> /dev/null; then
    log_error "Python is not installed or not in PATH"
    exit 1
fi

PYTHON_CMD=""
if command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
elif command -v python &> /dev/null; then
    PYTHON_CMD="python"
fi

log_success "Using Python: $PYTHON_CMD ($($PYTHON_CMD --version))"
log_info "Python path: $(which $PYTHON_CMD)"

# Check if graphbench-lib is installed
log_info "Checking for graphbench-lib..."
if ! $PYTHON_CMD -c "import graphbench" 2>/dev/null; then
    log_warning "graphbench-lib is not installed"
    log_info "Installing graphbench-lib..."
    if ! $PYTHON_CMD -m pip install graphbench-lib --no-cache-dir; then
        log_error "Failed to install graphbench-lib"
        exit 1
    fi
    log_success "graphbench-lib installed successfully"
else
    log_success "graphbench-lib is already installed"
fi

# Navigate to project directory
cd "$PROJECT_ROOT" || {
    log_error "Failed to navigate to project directory"
    exit 1
}

# Add project root to PYTHONPATH
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH}"

# Create log file
LOG_FILE="$PROJECT_ROOT/graphbench_download_local.log"
log_info "Log file: $LOG_FILE"

# Datasets to download (same as cluster version)
DATASETS=(
    "socialnetwork"              # Social media datasets
    "co"                         # Combinatorial optimization
    "sat"                        # SAT solving
    "algorithmic_reasoning_easy" # Algorithmic reasoning (easy)
    "algorithmic_reasoning_medium" # Algorithmic reasoning (medium)
    "algorithmic_reasoning_hard"  # Algorithmic reasoning (hard)
    "electronic_circuits"        # Electronic circuits
    "chipdesign"                  # Chip design
    # Note: weather is for regression, not included here
)

TOTAL_DATASETS=${#DATASETS[@]}
log_info "Will download $TOTAL_DATASETS dataset(s):"
for name in "${DATASETS[@]}"; do
    log_info "   - $name"
done

echo "" | tee -a "$LOG_FILE"
echo "============================================================================" | tee -a "$LOG_FILE"
echo "GraphBench Download Session Started: $(date)" | tee -a "$LOG_FILE"
echo "============================================================================" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# Track results
SUCCESSFUL_DOWNLOADS=()
FAILED_DOWNLOADS=()
START_TIME=$(date +%s)

# Download each dataset sequentially
for i in "${!DATASETS[@]}"; do
    DATASET_NAME="${DATASETS[$i]}"
    DATASET_NUM=$((i + 1))
    
    echo "" | tee -a "$LOG_FILE"
    log_message "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    log_message "📊 Processing Dataset $DATASET_NUM/$TOTAL_DATASETS: $DATASET_NAME"
    log_message "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Processing: $DATASET_NAME" | tee -a "$LOG_FILE"
    
    DATASET_START_TIME=$(date +%s)
    
    # Run the Python download script for this dataset
    if $PYTHON_CMD scripts/download_data/download_graphbench_single.py "$DATASET_NAME" "$DATA_DIRECTORY" 2>&1 | tee -a "$LOG_FILE"; then
        DATASET_END_TIME=$(date +%s)
        DATASET_DURATION=$((DATASET_END_TIME - DATASET_START_TIME))
        
        # Verify the dataset was downloaded
        DATASET_PATH="$DATA_DIRECTORY/$DATASET_NAME"
        if [ -d "$DATASET_PATH" ] && [ "$(ls -A "$DATASET_PATH" 2>/dev/null)" ]; then
            log_success "$DATASET_NAME downloaded successfully (${DATASET_DURATION}s)"
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] ✅ SUCCESS: $DATASET_NAME (${DATASET_DURATION}s)" | tee -a "$LOG_FILE"
            
            # Check for processed directory (indicates successful processing)
            if [ -d "$DATASET_PATH/processed" ] || [ -d "$DATA_DIRECTORY/algoreas" ]; then
                log_success "$DATASET_NAME appears to be fully processed"
            fi
            
            SUCCESSFUL_DOWNLOADS+=("$DATASET_NAME")
        else
            log_warning "$DATASET_NAME download completed but directory is empty or missing"
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] ⚠️  WARNING: $DATASET_NAME directory empty" | tee -a "$LOG_FILE"
            FAILED_DOWNLOADS+=("$DATASET_NAME")
        fi
    else
        DATASET_END_TIME=$(date +%s)
        DATASET_DURATION=$((DATASET_END_TIME - DATASET_START_TIME))
        log_error "$DATASET_NAME download failed (${DATASET_DURATION}s)"
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] ❌ FAILED: $DATASET_NAME (${DATASET_DURATION}s)" | tee -a "$LOG_FILE"
        FAILED_DOWNLOADS+=("$DATASET_NAME")
    fi
    
    # Force Python garbage collection between downloads
    log_info "Cleaning up memory before next download..."
    $PYTHON_CMD -c "import gc; gc.collect()" 2>/dev/null || true
    
    # Small delay between downloads to avoid overwhelming the system
    sleep 2
done

# Calculate total time
END_TIME=$(date +%s)
TOTAL_DURATION=$((END_TIME - START_TIME))
MINUTES=$((TOTAL_DURATION / 60))
SECONDS=$((TOTAL_DURATION % 60))

# Print summary
echo "" | tee -a "$LOG_FILE"
log_message "============================================================================"
log_message "🎉 Download process complete!"
log_message "============================================================================"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Total duration: ${MINUTES}m ${SECONDS}s" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

if [ ${#SUCCESSFUL_DOWNLOADS[@]} -gt 0 ]; then
    log_success "Successfully downloaded ${#SUCCESSFUL_DOWNLOADS[@]} dataset(s):"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Successful downloads:" | tee -a "$LOG_FILE"
    for name in "${SUCCESSFUL_DOWNLOADS[@]}"; do
        log_success "   ✅ $name"
        echo "   - $name" | tee -a "$LOG_FILE"
    done
fi

if [ ${#FAILED_DOWNLOADS[@]} -gt 0 ]; then
    log_error "Failed to download ${#FAILED_DOWNLOADS[@]} dataset(s):"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Failed downloads:" | tee -a "$LOG_FILE"
    for name in "${FAILED_DOWNLOADS[@]}"; do
        log_error "   ❌ $name"
        echo "   - $name" | tee -a "$LOG_FILE"
    done
    echo "" | tee -a "$LOG_FILE"
    log_warning "💡 Tip: You can re-run this script to retry failed downloads."
    log_warning "   Corrupted datasets will be automatically deleted and re-downloaded."
fi

log_info "📁 Datasets location: $DATA_DIRECTORY"
log_info "📝 Full log saved to: $LOG_FILE"
log_success "✅ Experiments can now run without downloading datasets!"

echo "" | tee -a "$LOG_FILE"
echo "============================================================================" | tee -a "$LOG_FILE"
echo "GraphBench Download Session Ended: $(date)" | tee -a "$LOG_FILE"
echo "============================================================================" | tee -a "$LOG_FILE"

