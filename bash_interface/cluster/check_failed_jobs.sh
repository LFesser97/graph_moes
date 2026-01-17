#!/bin/bash
# ============================================================================
# Quick Check for Failed/Unfinished Jobs
# ============================================================================
# Usage: ./check_failed_jobs.sh [JOB_ID]
# If JOB_ID is provided, checks only that job array
# Otherwise checks all recent job arrays
# ============================================================================

LOG_DIR="logs_comprehensive"
JOB_FILTER=$1

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo "🔍 Checking for Failed/Unfinished Jobs..."
echo "=========================================="
echo ""

if [ -z "$JOB_FILTER" ]; then
    # Find all log files
    LOG_FILES=$(find "$LOG_DIR" -name "Parallel_comprehensive_sweep_*.log" -type f 2>/dev/null | sort -V | tail -200)
    echo "Checking recent log files (all jobs)..."
else
    # Filter by job ID
    LOG_FILES=$(find "$LOG_DIR" -name "Parallel_comprehensive_sweep_${JOB_FILTER}_*.log" -type f 2>/dev/null | sort -V)
    echo "Checking job array: $JOB_FILTER"
fi

if [ -z "$LOG_FILES" ]; then
    echo "   No log files found"
    exit 1
fi

# Categorize
FAILED=()
UNFINISHED=()
EMPTY=()
COMPLETED_COUNT=0

for log in $LOG_FILES; do
    TASK_ID=$(basename "$log" .log | sed 's/Parallel_comprehensive_sweep_[0-9]*_//')
    JOB_ID=$(basename "$log" .log | sed 's/Parallel_comprehensive_sweep_//' | sed 's/_[0-9]*$//')
    
    if [ ! -s "$log" ]; then
        EMPTY+=("${JOB_ID}_${TASK_ID}")
        continue
    fi
    
    if grep -q "✅ Task.*completed successfully" "$log" 2>/dev/null; then
        COMPLETED_COUNT=$((COMPLETED_COUNT + 1))
    elif grep -q "❌.*failed\|Error\|Exception\|Traceback\|CRITICAL" "$log" 2>/dev/null; then
        FAILED+=("${JOB_ID}_${TASK_ID}")
    else
        # Has content but no completion - likely still running or died
        UNFINISHED+=("${JOB_ID}_${TASK_ID}")
    fi
done

TOTAL=$(echo "$LOG_FILES" | wc -l)

# Print summary
echo ""
echo -e "${GREEN}✓ Completed:${NC} $COMPLETED_COUNT/$TOTAL"
echo -e "${RED}✗ Failed:${NC} ${#FAILED[@]}"
echo -e "${YELLOW}⏳ Unfinished/Running:${NC} ${#UNFINISHED[@]}"
echo -e "${BLUE}○ Empty/Pending:${NC} ${#EMPTY[@]}"
echo ""

# Show failed tasks
if [ ${#FAILED[@]} -gt 0 ]; then
    echo -e "${RED}❌ FAILED TASKS (${#FAILED[@]}):${NC}"
    for task in "${FAILED[@]:0:20}"; do  # Show first 20
        JOB_TASK=$(echo "$task" | sed 's/_/ /')
        JOB=$(echo "$JOB_TASK" | awk '{print $1}')
        TASK=$(echo "$JOB_TASK" | awk '{print $2}')
        LOG_FILE="$LOG_DIR/Parallel_comprehensive_sweep_${JOB}_${TASK}.log"
        
        # Get last error line
        ERROR_LINE=$(grep -iE "error|failed|exception|❌" "$LOG_FILE" 2>/dev/null | tail -1 | cut -c1-100)
        if [ ! -z "$ERROR_LINE" ]; then
            echo -e "   ${RED}✗${NC} ${JOB}_${TASK}: ${ERROR_LINE}"
        else
            echo -e "   ${RED}✗${NC} ${JOB}_${TASK}"
        fi
    done
    if [ ${#FAILED[@]} -gt 20 ]; then
        echo "   ... and $((${#FAILED[@]} - 20)) more"
    fi
    echo ""
fi

# Show unfinished tasks
if [ ${#UNFINISHED[@]} -gt 0 ]; then
    echo -e "${YELLOW}⏳ UNFINISHED/RUNNING TASKS (${#UNFINISHED[@]}):${NC}"
    for task in "${UNFINISHED[@]:0:30}"; do  # Show first 30
        JOB_TASK=$(echo "$task" | sed 's/_/ /')
        JOB=$(echo "$JOB_TASK" | awk '{print $1}')
        TASK=$(echo "$JOB_TASK" | awk '{print $2}')
        LOG_FILE="$LOG_DIR/Parallel_comprehensive_sweep_${JOB}_${TASK}.log"
        
        if [ -f "$LOG_FILE" ]; then
            LINES=$(wc -l < "$LOG_FILE" 2>/dev/null || echo "0")
            LAST_LINE=$(tail -1 "$LOG_FILE" 2>/dev/null | cut -c1-80)
            MOD_TIME=$(stat -c %y "$LOG_FILE" 2>/dev/null | cut -d' ' -f1,2 | cut -d':' -f1,2 || stat -f "%Sm" "$LOG_FILE" 2>/dev/null | cut -d' ' -f1,2,3 || echo "unknown")
            
            echo -e "   ${YELLOW}⏳${NC} ${JOB}_${TASK} ($LINES lines, updated: $MOD_TIME)"
            if [ ! -z "$LAST_LINE" ]; then
                echo "      Last: ${LAST_LINE}..."
            fi
        fi
    done
    if [ ${#UNFINISHED[@]} -gt 30 ]; then
        echo "   ... and $((${#UNFINISHED[@]} - 30)) more"
    fi
    echo ""
fi

# Show empty tasks
if [ ${#EMPTY[@]} -gt 0 ] && [ ${#EMPTY[@]} -lt 50 ]; then
    echo -e "${BLUE}○ EMPTY/PENDING TASKS (${#EMPTY[@]}):${NC}"
    for task in "${EMPTY[@]}"; do
        echo "   ○ $task"
    done
elif [ ${#EMPTY[@]} -ge 50 ]; then
    echo -e "${BLUE}○ EMPTY/PENDING TASKS:${NC} ${#EMPTY[@]} (too many to list)"
fi

echo ""
echo "💡 To see details: tail -f $LOG_DIR/Parallel_comprehensive_sweep_<JOB>_<TASK>.log"
echo "💡 To check specific job: $0 <JOB_ID>"
