#!/bin/bash

# run.sh - Execute the PANOSETI L0 to L1 data pipeline with persistent Dask cluster
# 
# UPDATED: Creates Dask cluster ONCE at startup, reuses for all operations

set -e
set -u

#==============================================================================
# CONFIGURATION
#==============================================================================

CONFIG_FILE="${CONFIG_FILE:-config.toml}"

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

echo "Checking required scripts..."
for script in step1_pff_to_zarr.py step2_dask_baseline.py cluster_manager.py pff.py; do
    if [ ! -f "${script}" ]; then
        echo "Error: ${script} not found in current directory!"
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
echo ""

#==============================================================================
# START PERSISTENT DASK CLUSTER
#==============================================================================

echo "================================================"
echo "Starting Persistent Dask Cluster"
echo "================================================"
echo ""

# Start cluster in background and capture scheduler address
SCHEDULER_FILE="/tmp/dask_scheduler_$$.txt"
python3 -c "
import asyncio
import sys
from cluster_manager import load_cluster_config, create_dask_cluster

async def start_cluster():
    config = load_cluster_config('${CONFIG_FILE}')
    client, cluster = await create_dask_cluster(config)
    
    if client:
        scheduler_address = client.scheduler.address
        print(f'SCHEDULER_ADDRESS={scheduler_address}', file=sys.stderr)
        with open('${SCHEDULER_FILE}', 'w') as f:
            f.write(scheduler_address)
        
        # Keep cluster alive by waiting indefinitely
        print('Cluster is running. Press Ctrl+C to shutdown.', file=sys.stderr)
        try:
            await asyncio.Event().wait()
        except KeyboardInterrupt:
            print('\\nShutting down cluster...', file=sys.stderr)
            await client.close()
            if cluster:
                await cluster.close()
    else:
        # No Dask - write empty file
        with open('${SCHEDULER_FILE}', 'w') as f:
            f.write('')

asyncio.run(start_cluster())
" &

CLUSTER_PID=$!

# Wait for scheduler file to be created
echo "Waiting for cluster to initialize..."
for i in {1..30}; do
    if [ -f "${SCHEDULER_FILE}" ]; then
        break
    fi
    sleep 1
done

if [ ! -f "${SCHEDULER_FILE}" ]; then
    echo "Error: Cluster failed to start"
    kill $CLUSTER_PID 2>/dev/null || true
    exit 1
fi

SCHEDULER_ADDRESS=$(cat "${SCHEDULER_FILE}")

if [ -z "${SCHEDULER_ADDRESS}" ]; then
    echo "No Dask cluster - using local processing"
    SCHEDULER_ARG=""
else
    echo "✓ Cluster ready!"
    echo "  Scheduler address: ${SCHEDULER_ADDRESS}"
    SCHEDULER_ARG="${SCHEDULER_ADDRESS}"
fi
echo ""

# Cleanup function
cleanup() {
    echo ""
    echo "================================================"
    echo "Shutting Down Cluster"
    echo "================================================"
    if [ ! -z "${CLUSTER_PID:-}" ]; then
        kill $CLUSTER_PID 2>/dev/null || true
        wait $CLUSTER_PID 2>/dev/null || true
    fi
    rm -f "${SCHEDULER_FILE}"
    echo "✓ Cleanup complete"
}

trap cleanup EXIT INT TERM

#==============================================================================
# PROCESSING
#==============================================================================

file_count=1
total_files=${#INPUT_FILES[@]}

for pff_file in "${INPUT_FILES[@]}"; do
    basename=$(basename "${pff_file}" .pff)
    
    echo "================================================"
    echo "Processing file ${file_count}/${total_files}: ${basename}"
    echo "================================================"
    echo ""
    
    L0_ZARR="${L0_TEMP_DIR}/${basename}_L0.zarr"
    L1_ZARR="${OUTPUT_L1_DIR}/${basename}_L1.zarr"
    
    echo "Step 1/2: Converting PFF to Zarr (L0)..."
    echo "  Input: ${pff_file}"
    echo "  Output: ${L0_ZARR}"
    echo ""
    
    if [ -z "${SCHEDULER_ARG}" ]; then
        python3 step1_pff_to_zarr.py "${pff_file}" "${L0_ZARR}" "${CONFIG_FILE}" || {
            echo "ERROR: Step 1 failed for ${pff_file}"
            exit 1
        }
    else
        python3 step1_pff_to_zarr.py "${pff_file}" "${L0_ZARR}" "${CONFIG_FILE}" "${SCHEDULER_ARG}" || {
            echo "ERROR: Step 1 failed for ${pff_file}"
            exit 1
        }
    fi
    
    echo ""
    echo "Step 2/2: Applying baseline subtraction (L0 -> L1)..."
    echo "  Input: ${L0_ZARR}"
    echo "  Output: ${L1_ZARR}"
    echo ""
    
    if [ -z "${SCHEDULER_ARG}" ]; then
        python3 step2_dask_baseline.py "${L0_ZARR}" "${L1_ZARR}" --config "${CONFIG_FILE}" || {
            echo "ERROR: Step 2 failed for ${basename}"
            exit 1
        }
    else
        python3 step2_dask_baseline.py "${L0_ZARR}" "${L1_ZARR}" --config "${CONFIG_FILE}" --dask-scheduler "${SCHEDULER_ARG}" || {
            echo "ERROR: Step 2 failed for ${basename}"
            exit 1
        }
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
    echo "  - ${OUTPUT_L1_DIR}/${basename}_L1.zarr"
done
echo ""
