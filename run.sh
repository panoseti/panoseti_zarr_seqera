#!/bin/bash

# run.sh - Execute the PANOSETI L0 to L1 data pipeline with persistent Dask cluster
# 
# IMPROVED: Robust cluster lifecycle management with guaranteed cleanup

set -e  # Exit on error
set -u  # Exit on undefined variable
set -o pipefail  # Pipe failures propagate

#==============================================================================
# CONFIGURATION
#==============================================================================

CONFIG_FILE="${CONFIG_FILE:-config.toml}"

# Cluster management
CLUSTER_PID=""
SCHEDULER_FILE=""
SCHEDULER_ADDRESS=""
USE_CLUSTER=false
CLEANUP_DONE=0

#==============================================================================
# CLEANUP FUNCTION - GUARANTEED TO RUN
#==============================================================================

cleanup() {
    if [ "$CLEANUP_DONE" -eq 1 ]; then
        return  # Prevent double cleanup
    fi
    CLEANUP_DONE=1

    local exit_code=$?

    echo ""
    echo "================================================"
    echo "Shutting Down Pipeline"
    echo "================================================"

    # Kill cluster manager if running
    if [ ! -z "${CLUSTER_PID:-}" ] && [ "$CLUSTER_PID" != "0" ]; then
        echo "Stopping cluster manager (PID: $CLUSTER_PID)..."

        # Try graceful shutdown first
        if kill -0 "$CLUSTER_PID" 2>/dev/null; then
            kill -TERM "$CLUSTER_PID" 2>/dev/null || true

            # Wait up to 10 seconds for graceful shutdown
            for i in {1..10}; do
                if ! kill -0 "$CLUSTER_PID" 2>/dev/null; then
                    echo "  ✓ Cluster manager stopped gracefully"
                    break
                fi
                sleep 1
            done

            # Force kill if still running
            if kill -0 "$CLUSTER_PID" 2>/dev/null; then
                echo "  Force killing cluster manager..."
                kill -9 "$CLUSTER_PID" 2>/dev/null || true
                sleep 2
            fi
        fi
    fi

    # Additional safety: kill any orphaned dask processes
    echo "Checking for orphaned Dask processes..."
    pkill -f "distributed.cli.dask_scheduler" 2>/dev/null || true
    pkill -f "distributed.cli.dask_worker" 2>/dev/null || true

    # Clean up scheduler file
    if [ ! -z "${SCHEDULER_FILE:-}" ] && [ -f "${SCHEDULER_FILE}" ]; then
        rm -f "${SCHEDULER_FILE}"
    fi

    echo "✓ Cleanup complete"
    echo "================================================"

    # Exit with the original exit code if non-zero
    if [ "$exit_code" -ne 0 ]; then
        echo "Pipeline exited with error code: $exit_code"
        exit $exit_code
    fi
}

# Register cleanup on ALL exit conditions
trap cleanup EXIT INT TERM ERR

#==============================================================================
# FUNCTIONS
#==============================================================================

usage() {
    echo "Usage: $0 <input_pff_file1> [input_pff_file2 ...] <output_l1_dir>"
    echo ""
    echo "Arguments:"
    echo "  input_pff_file  One or more PFF files to process"
    echo "  output_l1_dir   Directory where L1 products will be written"
    echo ""
    echo "Configuration:"
    echo "  CONFIG_FILE     Path to TOML config (default: config.toml)"
    echo ""
    echo "Examples:"
    echo "  $0 file1.pff file2.pff /scratch/user/output/L1"
    echo "  CONFIG_FILE=custom.toml $0 file1.pff /scratch/user/output/L1"
    exit 1
}

check_required_files() {
    echo "Checking required scripts..."
    local required_files=(
        "step1_pff_to_zarr.py"
        "step2_dask_baseline.py"
        "cluster_manager.py"
        "cluster_lifecycle_manager.py"
        "pff.py"
    )

    for script in "${required_files[@]}"; do
        if [ ! -f "${script}" ]; then
            echo "ERROR: ${script} not found in current directory!"
            exit 1
        fi
        echo "  [OK] ${script}"
    done

    if [ ! -f "${CONFIG_FILE}" ]; then
        echo "Warning: Config file not found: ${CONFIG_FILE}"
        echo "  Will use default settings"
    else
        echo "  [OK] ${CONFIG_FILE}"
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
# STARTUP INFORMATION
#==============================================================================

echo "================================================"
echo "PANOSETI L0 -> L1 Processing Pipeline"
echo "WITH PERSISTENT DASK CLUSTER"
echo "================================================"
echo ""
echo "Configuration:"
echo "  Input files: ${#INPUT_FILES[@]}"
for file in "${INPUT_FILES[@]}"; do
    echo "    - $file"
done
echo "  Output L1 dir: ${OUTPUT_L1_DIR}"
echo "  Config file: ${CONFIG_FILE}"
echo ""

# Create output directories
L0_TEMP_DIR="${OUTPUT_L1_DIR}/../L0_temp"
echo "Creating directories..."
mkdir -p "${OUTPUT_L1_DIR}" || {
    echo "Error: Failed to create output directory: ${OUTPUT_L1_DIR}"
    exit 1
}
mkdir -p "${L0_TEMP_DIR}" || {
    echo "Error: Failed to create L0 temp directory: ${L0_TEMP_DIR}"
    exit 1
}
echo "  L0 temp dir: ${L0_TEMP_DIR}"
echo ""

#==============================================================================
# VALIDATION
#==============================================================================

echo "Verifying input files..."
for pff_file in "${INPUT_FILES[@]}"; do
    if [ ! -f "${pff_file}" ]; then
        echo "Error: Input file not found: ${pff_file}"
        exit 1
    fi
    echo "  [OK] ${pff_file}"
done
echo ""

check_required_files
echo ""

#==============================================================================
# START PERSISTENT DASK CLUSTER
#==============================================================================

echo "================================================"
echo "Starting Persistent Dask Cluster"
echo "================================================"
echo ""

# Create temporary scheduler file
SCHEDULER_FILE=$(mktemp /tmp/dask_scheduler_XXXXXX.txt)

echo "Starting cluster lifecycle manager..."
echo "  Config: ${CONFIG_FILE}"
echo "  Scheduler file: ${SCHEDULER_FILE}"
echo ""

# Start cluster manager with robust error handling
python3 cluster_lifecycle_manager.py "${CONFIG_FILE}" "${SCHEDULER_FILE}" 2>&1 &
CLUSTER_PID=$!

echo "Cluster manager started (PID: $CLUSTER_PID)"

# Wait for scheduler file to be created with timeout
echo "Waiting for cluster to initialize..."
MAX_WAIT=3
for i in $(seq 1 $MAX_WAIT); do
    if [ -f "${SCHEDULER_FILE}" ]; then
        SCHEDULER_ADDRESS=$(cat "${SCHEDULER_FILE}")
        if [ ! -z "${SCHEDULER_ADDRESS}" ]; then
            break
        fi
        break
    fi

    # Check if cluster manager is still running
    if ! kill -0 "$CLUSTER_PID" 2>/dev/null; then
        echo "ERROR: Cluster manager process died during startup"
        echo "Check logs above for error details"
        exit 1
    fi

    sleep 1

    if [ "$i" -eq "$MAX_WAIT" ]; then
        echo "ERROR: Cluster failed to start within ${MAX_WAIT} seconds"
        exit 1
    fi
done

# Read scheduler address
if [ ! -f "${SCHEDULER_FILE}" ]; then
    echo "ERROR: Scheduler file not created"
    exit 1
fi

SCHEDULER_ADDRESS=$(cat "${SCHEDULER_FILE}")

if [ -z "${SCHEDULER_ADDRESS}" ]; then
    echo "No Dask cluster - using local processing"
    USE_CLUSTER=false
else
    echo "✓ Cluster ready!"
    echo "  Scheduler address: ${SCHEDULER_ADDRESS}"
    USE_CLUSTER=true
fi
echo ""

# Verify cluster manager is still running
if ! kill -0 "$CLUSTER_PID" 2>/dev/null; then
    echo "ERROR: Cluster manager not running after startup"
    exit 1
fi

#==============================================================================
# PROCESSING
#==============================================================================

file_count=1
total_files=${#INPUT_FILES[@]}
PROCESSING_FAILED=0

for pff_file in "${INPUT_FILES[@]}"; do
    basename=$(basename "${pff_file}" .pff)

    echo "================================================"
    echo "Processing file ${file_count}/${total_files}: ${basename}"
    echo "================================================"
    echo ""

    L0_ZARR="${L0_TEMP_DIR}/${basename}_L0.zarr"
    L1_ZARR="${OUTPUT_L1_DIR}/${basename}_L1.zarr"

    # Step 1: PFF to Zarr
    echo "Step 1/2: Converting PFF to Zarr (L0)..."
    echo "  Input: ${pff_file}"
    echo "  Output: ${L0_ZARR}"
    echo ""

    if [ "$USE_CLUSTER" = true ]; then
        if ! python3 step1_pff_to_zarr.py "${pff_file}" "${L0_ZARR}" "${CONFIG_FILE}" "${SCHEDULER_ADDRESS}"; then
            echo "ERROR: Step 1 failed for ${pff_file}"
            PROCESSING_FAILED=1
            break
        fi
    else
        if ! python3 step1_pff_to_zarr.py "${pff_file}" "${L0_ZARR}" "${CONFIG_FILE}"; then
            echo "ERROR: Step 1 failed for ${pff_file}"
            PROCESSING_FAILED=1
            break
        fi
    fi

    echo ""

    # Step 2: Baseline subtraction
    echo "Step 2/2: Applying baseline subtraction (L0 -> L1)..."
    echo "  Input: ${L0_ZARR}"
    echo "  Output: ${L1_ZARR}"
    echo ""

    if [ "$USE_CLUSTER" = true ]; then
        if ! python3 step2_dask_baseline.py "${L0_ZARR}" "${L1_ZARR}" --config "${CONFIG_FILE}" --dask-scheduler "${SCHEDULER_ADDRESS}"; then
            echo "ERROR: Step 2 failed for ${basename}"
            PROCESSING_FAILED=1
            break
        fi
    else
        if ! python3 step2_dask_baseline.py "${L0_ZARR}" "${L1_ZARR}" --config "${CONFIG_FILE}"; then
            echo "ERROR: Step 2 failed for ${basename}"
            PROCESSING_FAILED=1
            break
        fi
    fi

    echo ""
    echo "Cleaning up L0 temporary file..."
    rm -rf "${L0_ZARR}"

    echo ""
    echo "[DONE] Completed ${basename}"
    echo ""

    file_count=$((file_count + 1))
done

# Clean up L0 temp directory if empty
if [ -d "${L0_TEMP_DIR}" ] && [ -z "$(ls -A ${L0_TEMP_DIR})" ]; then
    rmdir "${L0_TEMP_DIR}"
fi

#==============================================================================
# SUMMARY
#==============================================================================

if [ "$PROCESSING_FAILED" -eq 1 ]; then
    echo "================================================"
    echo "Pipeline Failed!"
    echo "================================================"
    echo ""
    echo "Processing stopped due to errors"
    echo "Check logs above for details"
    echo ""
    exit 1
fi

echo "================================================"
echo "Pipeline Complete!"
echo "================================================"
echo ""
echo "Processed ${total_files} file(s)"
echo "L1 products written to: ${OUTPUT_L1_DIR}"
echo ""
echo "Output files:"
for pff_file in "${INPUT_FILES[@]}"; do
    basename=$(basename "${pff_file}" .pff)
    L1_ZARR="${OUTPUT_L1_DIR}/${basename}_L1.zarr"
    if [ -d "${L1_ZARR}" ]; then
        echo "  ✓ ${basename}_L1.zarr"
    else
        echo "  ✗ ${basename}_L1.zarr (MISSING)"
    fi
done
echo ""

# cleanup() will be called automatically via trap
