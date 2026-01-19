#!/bin/bash
# ============================================================================
# Check Progress of Specific Job Array
# ============================================================================
# Usage: ./check_job_progress.sh <JOB_ID>
# Example: ./check_job_progress.sh 55737607
# ============================================================================

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

JOB_ID=$1

if [ -z "$JOB_ID" ]; then
    echo "Usage: $0 <JOB_ID>"
    echo "Example: $0 55737607"
    echo ""
    echo "Your active jobs:"
    squeue -u $USER -o "%i %j %t %M" --noheader | grep -i comprehe | head -5
    exit 1
fi

LOG_DIR="logs_comprehensive"
echo "📊 Checking Progress for Job Array: $JOB_ID"
echo "=========================================="
echo ""

# Check if job is still running
JOB_INFO=$(squeue -j $JOB_ID --noheader 2>/dev/null)
if [ ! -z "$JOB_INFO" ]; then
    echo "Status: Still in queue/running"
    echo "$JOB_INFO"
    echo ""
else
    echo "Status: Job array completed or not found"
    echo ""
fi

# Find all log files for this job
LOG_FILES=$(find "$LOG_DIR" -name "Parallel_comprehensive_sweep_${JOB_ID}_*.log" 2>/dev/null | sort)

if [ -z "$LOG_FILES" ]; then
    echo "⚠️  No log files found for job $JOB_ID in $LOG_DIR"
    exit 1
fi

TOTAL_LOGS=$(echo "$LOG_FILES" | wc -l)
echo "Found $TOTAL_LOGS log files"
echo ""

# Categorize logs
COMPLETED=0
FAILED=0
RUNNING=0
EMPTY=0
LONG_RUNNING=0

declare -a FAILED_TASKS=()
declare -a LONG_TASKS=()

for log in $LOG_FILES; do
    TASK_ID=$(basename "$log" .log | sed "s/Parallel_comprehensive_sweep_${JOB_ID}_//")
    
    if [ ! -s "$log" ]; then
        EMPTY=$((EMPTY + 1))
        continue
    fi
    
    # Check if completed
    if grep -q "✅ Task.*completed successfully" "$log" 2>/dev/null; then
        COMPLETED=$((COMPLETED + 1))
    # Check if failed
    elif grep -q "❌.*failed" "$log" 2>/dev/null; then
        FAILED=$((FAILED + 1))
        FAILED_TASKS+=("$TASK_ID")
    # Check if running (has content but no completion message)
    else
        RUNNING=$((RUNNING + 1))
        
        # Check runtime from log
        FIRST_LINE=$(head -1 "$log" 2>/dev/null)
        LAST_LINE=$(tail -1 "$log" 2>/dev/null)
        
        # Try to extract time info
        LINES=$(wc -l < "$log" 2>/dev/null || echo "0")
        if [ "$LINES" -gt 1000 ]; then
            LONG_RUNNING=$((LONG_RUNNING + 1))
            LONG_TASKS+=("$TASK_ID")
        fi
    fi
done

# Summary
echo "📈 Progress Summary"
echo "------------------"
echo -e "   ${GREEN}✓ Completed:${NC} $COMPLETED"
echo -e "   ${YELLOW}⏳ Running:${NC} $RUNNING"
echo -e "   ${RED}✗ Failed:${NC} $FAILED"
echo -e "   ${BLUE}○ Empty/Pending:${NC} $EMPTY"
echo ""

# Show failed tasks
if [ $FAILED -gt 0 ]; then
    echo "❌ Failed Tasks ($FAILED):"
    for task in "${FAILED_TASKS[@]}"; do
        echo "   - Task $task"
        # Show last few lines of error
        LOG_FILE="$LOG_DIR/Parallel_comprehensive_sweep_${JOB_ID}_${task}.log"
        if [ -f "$LOG_FILE" ]; then
            echo "     Last lines:"
            tail -3 "$LOG_FILE" | sed 's/^/       /'
        fi
    done
    echo ""
fi

# Show long-running tasks
if [ $LONG_RUNNING -gt 0 ] && [ ${#LONG_TASKS[@]} -gt 0 ]; then
    echo "⏱️  Long-Running Tasks (${#LONG_TASKS[@]}):"
    for task in "${LONG_TASKS[@]:0:10}"; do  # Show first 10
        LOG_FILE="$LOG_DIR/Parallel_comprehensive_sweep_${JOB_ID}_${task}.log"
        if [ -f "$LOG_FILE" ]; then
            LINES=$(wc -l < "$LOG_FILE" 2>/dev/null || echo "0")
            LAST_LINE=$(tail -1 "$LOG_FILE" 2>/dev/null | cut -c1-80)
            echo "   - Task $task ($LINES lines): $LAST_LINE"
        fi
    done
    if [ ${#LONG_TASKS[@]} -gt 10 ]; then
        echo "   ... and $((${#LONG_TASKS[@]} - 10)) more"
    fi
    echo ""
fi

# Progress percentage
if [ $TOTAL_LOGS -gt 0 ]; then
    PROGRESS=$(( (COMPLETED * 100) / TOTAL_LOGS) ))
    echo "📊 Progress: $PROGRESS% ($COMPLETED/$TOTAL_LOGS tasks completed)"
    
    # Progress bar (simple)
    BAR_LENGTH=50
    FILLED=$(( (COMPLETED * BAR_LENGTH) / TOTAL_LOGS ))
    BAR=""
    for i in $(seq 1 $BAR_LENGTH); do
        if [ $i -le $FILLED ]; then
            BAR="${BAR}█"
        else
            BAR="${BAR}░"
        fi
    done
    echo "   [$BAR]"
fi

echo ""
echo "💡 Quick Commands:"
echo "   View specific task log: tail -f $LOG_DIR/Parallel_comprehensive_sweep_${JOB_ID}_<TASK>.log"
echo "   Cancel failed task: scancel ${JOB_ID}_<TASK>"
echo "   Rerun failed task: sbatch --dependency=afterany:${JOB_ID}_<TASK> <script.sh>"
