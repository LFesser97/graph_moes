#!/bin/bash
# ============================================================================
# Monitor Comprehensive Sweep Jobs
# ============================================================================
# This script helps monitor SLURM array jobs for the comprehensive sweep
# - Shows which jobs are running/completed/failed
# - Identifies jobs running longer than expected
# - Checks logs for errors
# - Shows progress summary
# ============================================================================

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo "📊 Comprehensive Sweep Job Monitor"
echo "=================================="
echo ""

# Get user's jobs
echo "🔍 Checking active jobs..."
JOBS=$(squeue -u $USER -o "%i %j %t %M %N" --noheader | grep -i comprehe | head -5)

if [ -z "$JOBS" ]; then
    echo "   No active comprehensive sweep jobs found"
else
    echo "$JOBS" | while IFS= read -r line; do
        JOB_ID=$(echo "$line" | awk '{print $1}')
        JOB_NAME=$(echo "$line" | awk '{print $2}')
        STATE=$(echo "$line" | awk '{print $3}')
        TIME=$(echo "$line" | awk '{print $4}')
        NODE=$(echo "$line" | awk '{print $5}')
        
        if [ "$STATE" = "R" ]; then
            echo -e "   ${GREEN}✓${NC} $JOB_ID ($JOB_NAME) - Running for $TIME on $NODE"
        elif [ "$STATE" = "PD" ]; then
            REASON=$(squeue -j $JOB_ID -o "%R" --noheader)
            echo -e "   ${YELLOW}⏳${NC} $JOB_ID ($JOB_NAME) - Pending: $REASON"
        else
            echo -e "   ${BLUE}ℹ${NC}  $JOB_ID ($JOB_NAME) - State: $STATE"
        fi
    done
fi

echo ""
echo "📋 Checking Recent Job Completions..."
echo "-----------------------------------"

# Find log directory
LOG_DIR="logs_comprehensive"
if [ ! -d "$LOG_DIR" ]; then
    echo "   ⚠️  Log directory '$LOG_DIR' not found"
    exit 1
fi

# Get recent log files (last 24 hours)
RECENT_LOGS=$(find "$LOG_DIR" -name "Parallel_comprehensive_sweep_*.log" -mtime -1 2>/dev/null | head -20)

if [ -z "$RECENT_LOGS" ]; then
    echo "   No recent log files found"
else
    echo "   Recent log files found:"
    for log in $RECENT_LOGS; do
        JOB_ARR=$(basename "$log" .log | sed 's/Parallel_comprehensive_sweep_//')
        if grep -q "✅ Task.*completed successfully" "$log" 2>/dev/null; then
            echo -e "   ${GREEN}✓${NC} $JOB_ARR - Completed"
        elif grep -q "❌.*failed" "$log" 2>/dev/null; then
            echo -e "   ${RED}✗${NC} $JOB_ARR - Failed"
        elif [ -f "$log" ]; then
            SIZE=$(wc -l < "$log" 2>/dev/null || echo "0")
            if [ "$SIZE" -gt 0 ]; then
                echo -e "   ${YELLOW}⏳${NC} $JOB_ARR - Running ($SIZE lines)"
            fi
        fi
    done
fi

echo ""
echo "⚠️  Checking for Errors in Recent Logs..."
echo "----------------------------------------"

ERROR_COUNT=0
for log in $RECENT_LOGS; do
    JOB_ARR=$(basename "$log" .log | sed 's/Parallel_comprehensive_sweep_//')
    
    # Check for common error patterns
    ERRORS=$(grep -iE "(error|failed|exception|traceback|critical)" "$log" 2>/dev/null | tail -3)
    
    if [ ! -z "$ERRORS" ]; then
        ERROR_COUNT=$((ERROR_COUNT + 1))
        echo -e "   ${RED}✗${NC} $JOB_ARR has errors:"
        echo "$ERRORS" | sed 's/^/      /'
        echo ""
    fi
done

if [ $ERROR_COUNT -eq 0 ]; then
    echo -e "   ${GREEN}✓${NC} No errors found in recent logs"
fi

echo ""
echo "⏱️  Long-Running Tasks (>2 hours)..."
echo "-----------------------------------"

# Check for jobs running longer than 2 hours
LONG_RUNNING=$(squeue -u $USER -o "%i %j %M %N" --noheader | grep -i comprehe | awk '$3 ~ /[0-9]+:[0-9]+:[0-9]+/ || $3 ~ /[0-9]+-[0-9]+:[0-9]+:[0-9]+/' | awk '{
    time=$3
    gsub(/-.*/, "", time)
    split(time, parts, ":")
    hours = parts[1]
    if (hours >= 2) print $1, $2, $3, $4
}')

if [ -z "$LONG_RUNNING" ]; then
    echo -e "   ${GREEN}✓${NC} No tasks running longer than 2 hours"
else
    echo "$LONG_RUNNING" | while IFS= read -r line; do
        JOB_ID=$(echo "$line" | awk '{print $1}')
        TIME=$(echo "$line" | awk '{print $3}')
        echo -e "   ${YELLOW}⚠${NC}  $JOB_ID - Running for $TIME"
    done
fi

echo ""
echo "📈 Quick Stats..."
echo "---------------"

# Count running jobs
RUNNING=$(squeue -u $USER --noheader | grep -i comprehe | grep " R " | wc -l)
PENDING=$(squeue -u $USER --noheader | grep -i comprehe | grep " PD " | wc -l)

echo "   Running: $RUNNING"
echo "   Pending: $PENDING"

# Show most recent job array
RECENT_JOB=$(squeue -u $USER -o "%i" --noheader | grep -E "[0-9]+_\[" | head -1)
if [ ! -z "$RECENT_JOB" ]; then
    echo "   Most recent job: $RECENT_JOB"
fi

echo ""
echo "💡 Tips:"
echo "   - Check specific log: tail -f logs_comprehensive/Parallel_comprehensive_sweep_<JOB_ID>_<TASK_ID>.log"
echo "   - Cancel a job array: scancel <JOB_ID>"
echo "   - Cancel specific task: scancel <JOB_ID>_<TASK_ID>"
