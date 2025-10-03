#!/bin/bash
# run.sh - Three-step PANOSETI L0 -> L1 pipeline with separated cluster setup

set -e
set -u
set -o pipefail

#==============================================================================
# CONFIGURATION
#==============================================================================

CONFIG_FILE="${CONFIG_FILE:-config.toml}"
SCHEDULER_FILE=""
SCHEDULER_ADDRESS=""
CLUSTER_PID=""
USE_CLUSTER=false
CLEANUP_DONE=0

#==============================================================================
# CLEANUP - GUARANTEED TO RUN
#==============================================================================

cleanup() {
    if [ "$CLEANUP_DONE" -eq 1 ]; then
        return
    fi
    CLEANUP_DONE=1
    
    local exit_code=$?
    
    echo ""
    echo "================================================"
    echo "Pipeline Cleanup"
    echo "================================================"
    
    # Stop cluster if we started it
    if [ ! -z "${CLUSTER_PID:-}" ] && [ "$CLUSTER_PID" != "0" ]; then
        echo "Stopping cluster (PID: $CLUSTER_PID)..."
        if kill -0 "$CLUSTER_PID" 2>/dev/null; then
            kill -TERM "$CLUSTER_PID" 2>/dev/null || true
            # Wait for graceful shutdown
            for i in {1..10}; do
                if ! kill -0 "$CLUSTER_PID" 2>/dev/null; then
                    echo "  ✓ Cluster stopped"
                    break
                fi
                sleep 1
            done
            # Force kill if still running
            if kill -0 "$CLUSTER_PID" 2>/dev/null; then
                echo "  Force killing cluster..."
                kill -9 "$CLUSTER_PID" 2>/dev/null || true
            fi
        fi
    fi
    
    # Cleanup scheduler file
    if [ ! -z "${SCHEDULER_FILE:-}" ] && [ -f "${SCHEDULER_FILE}" ]; then
        rm -f "${SCHEDULER_FILE}"
    fi
    
    echo "✓ Cleanup complete"
    echo "================================================"
    
    if [ "$exit_code" -ne 0 ]; then
        echo "Pipeline exited with error code: $exit_code"
        exit $exit_code
    fi
}

trap cleanup EXIT INT TERM ERR

#==============================================================================
# USAGE
#==============================================================================

usage() {
    cat << EOF
Usage: $0 [input_pff_file2 ...] <output_l1_dir>

Three-step pipeline:
  Step 0: Setup Dask cluster (optional, based on config)
  Step 1: Convert PFF to Zarr (L0)
  Step 2: Apply baseline subtraction (L0 -> L1)

Arguments:
  input_pff_file  One or more PFF files to process
  output_l1_dir   Directory for L1 output

Configuration:
  CONFIG_FILE     Path to TOML config (default: config.toml)

Examples:
  $0 file1.pff file2.pff /scratch/output/L1
  CONFIG_FILE=custom.toml $0 data/*.pff /scratch/output/L1
EOF
    exit 1
}

#==============================================================================
# VALIDATION
#==============================================================================

check_required_files() {
    echo "Checking required files..."
    local required=(
        "step0_setup_cluster.py"
        "step1_pff_to_zarr.py"
        "step2_dask_baseline.py"
        "cluster_manager.py"
        "pff.py"
    )
    
    for script in "${required[@]}"; do
        if [ ! -f "$script" ]; then
            echo "ERROR: $script not found!"
            exit 1
        fi
        echo "  [OK] $script"
    done
    
    if [ ! -f "$CONFIG_FILE" ]; then
        echo "Warning: $CONFIG_FILE not found - using defaults"
    else
        echo "  [OK] $CONFIG_FILE"
    fi
}

#==============================================================================
# ARGUMENT PARSING
#==============================================================================

if [ $# -lt 2 ]; then
    echo "Error: Insufficient arguments"
    echo ""
    usage
fi

OUTPUT_L1_DIR="${!#}"
INPUT_FILES=()
for ((i=1; i<$#; i++)); do
    INPUT_FILES+=("${!i}")
done

#==============================================================================
# STARTUP
#==============================================================================

echo "================================================"
echo "PANOSETI L0 -> L1 Processing Pipeline"
echo "THREE-STEP ARCHITECTURE"
echo "================================================"
echo ""
echo "Configuration:"
echo "  Input files: ${#INPUT_FILES[@]}"
for file in "${INPUT_FILES[@]}"; do
    echo "    - $file"
done
echo "  Output L1 dir: $OUTPUT_L1_DIR"
echo "  Config file: $CONFIG_FILE"
echo ""

# Create directories
L0_TEMP_DIR="${OUTPUT_L1_DIR}/../L0_temp"
mkdir -p "$OUTPUT_L1_DIR" || { echo "ERROR: Cannot create $OUTPUT_L1_DIR"; exit 1; }
mkdir -p "$L0_TEMP_DIR" || { echo "ERROR: Cannot create $L0_TEMP_DIR"; exit 1; }

# Validate inputs
echo "Verifying input files..."
for pff_file in "${INPUT_FILES[@]}"; do
    if [ ! -f "$pff_file" ]; then
        echo "ERROR: Input file not found: $pff_file"
        exit 1
    fi
    echo "  [OK] $pff_file"
done
echo ""

check_required_files
echo ""

#==============================================================================
# STEP 0: SETUP DASK CLUSTER
#==============================================================================

# Check if Dask is enabled in config
USE_DASK_CONFIG=$(python3 - "$CONFIG_FILE" <<'PY'
import sys
config_path = sys.argv[1]
use_dask = False
try:
    import tomllib
except ImportError:
    try:
        import tomli as tomllib
    except ImportError:
        tomllib = None
if tomllib:
    try:
        with open(config_path, "rb") as f:
            data = tomllib.load(f)
        use_dask = bool(data.get("cluster", {}).get("use_dask", False))
    except:
        pass
print("true" if use_dask else "false")
PY
)

if [ "$USE_DASK_CONFIG" != "true" ]; then
    echo "================================================"
    echo "Step 0: Dask Cluster Setup - SKIPPED"
    echo "================================================"
    echo "Dask disabled in $CONFIG_FILE"
    echo "Pipeline will use local processing"
    echo ""
    USE_CLUSTER=false
else
    echo "================================================"
    echo "Step 0: Dask Cluster Setup"
    echo "================================================"
    echo ""
    
    SCHEDULER_FILE=$(mktemp /tmp/dask_scheduler_XXXXXX.txt)
    
    echo "Starting cluster..."
    python3 step0_setup_cluster.py "$CONFIG_FILE" "$SCHEDULER_FILE" 2>&1 &
    CLUSTER_PID=$!
    
    echo "  Cluster PID: $CLUSTER_PID"
    echo "  Waiting for initialization..."
    
    # Wait for scheduler file with timeout
    MAX_WAIT=60
    for i in $(seq 1 $MAX_WAIT); do
        if [ -f "$SCHEDULER_FILE" ]; then
            SCHEDULER_ADDRESS=$(cat "$SCHEDULER_FILE")
            if [ ! -z "$SCHEDULER_ADDRESS" ]; then
                break
            fi
        fi
        
        # Check if cluster manager died
        if ! kill -0 "$CLUSTER_PID" 2>/dev/null; then
            echo "ERROR: Cluster manager died during startup"
            exit 1
        fi
        
        sleep 1
        
        if [ "$i" -eq "$MAX_WAIT" ]; then
            echo "ERROR: Cluster failed to start within ${MAX_WAIT}s"
            exit 1
        fi
    done
    
    if [ -z "$SCHEDULER_ADDRESS" ]; then
        echo "WARNING: No scheduler address - using local processing"
        USE_CLUSTER=false
    else
        echo "  ✓ Cluster ready!"
        echo "  Scheduler: $SCHEDULER_ADDRESS"
        USE_CLUSTER=true
    fi
    echo ""
fi

#==============================================================================
# STEP 1 & 2: PROCESS FILES
#==============================================================================

file_count=1
total_files=${#INPUT_FILES[@]}
PROCESSING_FAILED=0

for pff_file in "${INPUT_FILES[@]}"; do
    basename=$(basename "$pff_file" .pff)
    
    echo "================================================"
    echo "Processing file ${file_count}/${total_files}: $basename"
    echo "================================================"
    echo ""
    
    L0_ZARR="${L0_TEMP_DIR}/${basename}_L0.zarr"
    L1_ZARR="${OUTPUT_L1_DIR}/${basename}_L1.zarr"
    
    # STEP 1: PFF to Zarr
    echo "------------------------------------------------"
    echo "Step 1: PFF to Zarr Conversion"
    echo "------------------------------------------------"
    echo "  Input:  $pff_file"
    echo "  Output: $L0_ZARR"
    echo ""
    
    if [ "$USE_CLUSTER" = true ]; then
        if ! python3 step1_pff_to_zarr.py "$pff_file" "$L0_ZARR" "$CONFIG_FILE" "$SCHEDULER_ADDRESS"; then
            echo "ERROR: Step 1 failed"
            PROCESSING_FAILED=1
            break
        fi
    else
        if ! python3 step1_pff_to_zarr.py "$pff_file" "$L0_ZARR" "$CONFIG_FILE"; then
            echo "ERROR: Step 1 failed"
            PROCESSING_FAILED=1
            break
        fi
    fi
    echo ""
    
    # STEP 2: Baseline Subtraction
    echo "------------------------------------------------"
    echo "Step 2: Baseline Subtraction"
    echo "------------------------------------------------"
    echo "  Input:  $L0_ZARR"
    echo "  Output: $L1_ZARR"
    echo ""
    
    if [ "$USE_CLUSTER" = true ]; then
        if ! python3 step2_dask_baseline.py "$L0_ZARR" "$L1_ZARR" --config "$CONFIG_FILE" --dask-scheduler "$SCHEDULER_ADDRESS"; then
            echo "ERROR: Step 2 failed"
            PROCESSING_FAILED=1
            break
        fi
    else
        if ! python3 step2_dask_baseline.py "$L0_ZARR" "$L1_ZARR" --config "$CONFIG_FILE"; then
            echo "ERROR: Step 2 failed"
            PROCESSING_FAILED=1
            break
        fi
    fi
    echo ""
    
    # Cleanup L0 temp
    echo "Cleaning up temporary L0 file..."
    rm -rf "$L0_ZARR"
    
    echo "[DONE] Completed $basename"
    echo ""
    
    file_count=$((file_count + 1))
done

# Cleanup empty temp directory
if [ -d "$L0_TEMP_DIR" ] && [ -z "$(ls -A $L0_TEMP_DIR)" ]; then
    rmdir "$L0_TEMP_DIR"
fi

#==============================================================================
# SUMMARY
#==============================================================================

if [ "$PROCESSING_FAILED" -eq 1 ]; then
    echo "================================================"
    echo "Pipeline Failed"
    echo "================================================"
    exit 1
fi

echo "================================================"
echo "Pipeline Complete!"
echo "================================================"
echo ""
echo "Processed ${total_files} file(s)"
echo "L1 products: $OUTPUT_L1_DIR"
echo ""
echo "Output files:"
for pff_file in "${INPUT_FILES[@]}"; do
    basename=$(basename "$pff_file" .pff)
    L1_ZARR="${OUTPUT_L1_DIR}/${basename}_L1.zarr"
    if [ -d "$L1_ZARR" ]; then
        echo "  ✓ ${basename}_L1.zarr"
    else
        echo "  ✗ ${basename}_L1.zarr (MISSING)"
    fi
done
echo ""

