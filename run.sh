#!/bin/bash

# run.sh - PANOSETI L0 -> L1 pipeline with stream-based architecture
# Updated to process entire observation directories with PFF grouping

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
Usage: $0 <observation_directory> <output_l0_dir> <output_l1_dir>

Stream-based pipeline:
  Step 0: Setup Dask cluster (optional, based on config)
  Step 1: Convert all PFF files to Zarr (grouped by dp/module -> L0)
  Step 2: Apply baseline subtraction to all L0 Zarr files (L0 -> L1)

Arguments:
  observation_directory   Directory containing PFF files (*.pff)
  output_l0_dir          Directory for L0 Zarr output
  output_l1_dir          Directory for L1 Zarr output

Configuration:
  CONFIG_FILE            Path to TOML config (default: config.toml)

Examples:
  # Process observation directory
  $0 /mnt/beegfs/data/L0/obs_Lick.start_2024-07-25T04:34:06Z.runtype_sci-data.pffd \
     /mnt/beegfs/zarr/L0 \
     /mnt/beegfs/zarr/L1

  # Custom config
  CONFIG_FILE=custom.toml $0 /path/to/obs.pffd /path/to/L0 /path/to/L1

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

discover_pff_streams() {
    local obs_dir="$1"

    echo "Discovering PFF data streams..."
    python3 - "$obs_dir" <<'PYEOF'
import sys
import os
from pathlib import Path
from collections import defaultdict
import pff

obs_dir = sys.argv[1]
obs_path = Path(obs_dir)

if not obs_path.exists():
    print(f"ERROR: Observation directory not found: {obs_dir}")
    sys.exit(1)

# Group files by (dp, module)
streams = defaultdict(list)
pff_files = list(obs_path.glob("*.pff"))

if not pff_files:
    print(f"ERROR: No PFF files found in {obs_dir}")
    sys.exit(1)

print(f"Found {len(pff_files)} PFF files")
print()

for pff_file in pff_files:
    try:
        parts = pff.parse_name(pff_file.name)
        if not parts or 'dp' not in parts or 'module' not in parts:
            continue

        dp = parts['dp']
        module = parts['module']
        seqno = int(parts.get('seqno', 0))

        stream_key = (dp, module)
        streams[stream_key].append((str(pff_file), seqno))
    except Exception as e:
        print(f"Warning: Could not parse {pff_file.name}: {e}")
        continue

# Sort each stream by seqno
for key in streams:
    streams[key].sort(key=lambda x: x[1])

print(f"Discovered {len(streams)} data streams:")
print()

for (dp, module), files in sorted(streams.items()):
    print(f"Stream: dp={dp}, module={module}")
    print(f"  Files: {len(files)}")
    for fpath, seqno in files[:3]:  # Show first 3
        print(f"    - {os.path.basename(fpath)} (seqno={seqno})")
    if len(files) > 3:
        print(f"    ... and {len(files) - 3} more")
    print()

print(f"Total streams: {len(streams)}")
PYEOF

    if [ $? -ne 0 ]; then
        echo "ERROR: Failed to discover PFF streams"
        exit 1
    fi
}

#==============================================================================
# ARGUMENT PARSING
#==============================================================================

if [ $# -lt 3 ]; then
    echo "Error: Insufficient arguments"
    echo ""
    usage
fi

OBS_DIR="$1"
OUTPUT_L0_DIR="$2"
OUTPUT_L1_DIR="$3"

# Validate observation directory
if [ ! -d "$OBS_DIR" ]; then
    echo "ERROR: Observation directory not found: $OBS_DIR"
    exit 1
fi

#==============================================================================
# STARTUP
#==============================================================================

echo "================================================"
echo "PANOSETI L0 -> L1 Processing Pipeline"
echo "STREAM-BASED ARCHITECTURE"
echo "================================================"
echo ""
echo "Configuration:"
echo "  Observation dir: $OBS_DIR"
echo "  L0 output dir: $OUTPUT_L0_DIR"
echo "  L1 output dir: $OUTPUT_L1_DIR"
echo "  Config file: $CONFIG_FILE"
echo ""

# Create output directories
mkdir -p "$OUTPUT_L0_DIR" || { echo "ERROR: Cannot create $OUTPUT_L0_DIR"; exit 1; }
mkdir -p "$OUTPUT_L1_DIR" || { echo "ERROR: Cannot create $OUTPUT_L1_DIR"; exit 1; }

check_required_files
echo ""

# Discover PFF streams
discover_pff_streams "$OBS_DIR"
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
            # Check both old and new key names
            use_dask = bool(data.get("cluster", {}).get("use_dask", False) or
                          data.get("cluster", {}).get("use_cluster", False))
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
# STEP 1: CONVERT ALL PFF FILES TO ZARR (GROUPED BY STREAM)
#==============================================================================

echo "================================================"
echo "Step 1: PFF to Zarr Conversion (All Streams)"
echo "================================================"
echo ""
echo "This step will convert all PFF files to Zarr format."
echo "Files with the same data product and module will be"
echo "combined into a single Zarr file."
echo ""

STEP1_START=$(date +%s)

if [ "$USE_CLUSTER" = true ]; then
    echo "Running with Dask cluster..."
    if ! python3 step1_pff_to_zarr.py "$OBS_DIR" "$OUTPUT_L0_DIR" "$CONFIG_FILE" "$SCHEDULER_ADDRESS"; then
        echo "ERROR: Step 1 (PFF to Zarr) failed"
        exit 1
    fi
else
    echo "Running with local processing..."
    if ! python3 step1_pff_to_zarr.py "$OBS_DIR" "$OUTPUT_L0_DIR" "$CONFIG_FILE"; then
        echo "ERROR: Step 1 (PFF to Zarr) failed"
        exit 1
    fi
fi

STEP1_END=$(date +%s)
STEP1_DURATION=$((STEP1_END - STEP1_START))

echo ""
echo "✓ Step 1 complete (${STEP1_DURATION}s)"
echo ""

#==============================================================================
# STEP 2: BASELINE SUBTRACTION FOR ALL L0 ZARR FILES
#==============================================================================

echo "================================================"
echo "Step 2: Baseline Subtraction (All L0 -> L1)"
echo "================================================"
echo ""

# Discover all L0 Zarr files
L0_ZARR_FILES=($(find "$OUTPUT_L0_DIR" -maxdepth 1 -type d -name "*.zarr" | sort))

if [ ${#L0_ZARR_FILES[@]} -eq 0 ]; then
    echo "ERROR: No L0 Zarr files found in $OUTPUT_L0_DIR"
    exit 1
fi

echo "Found ${#L0_ZARR_FILES[@]} L0 Zarr file(s) to process:"
for zarr_file in "${L0_ZARR_FILES[@]}"; do
    echo "  - $(basename "$zarr_file")"
done
echo ""

STEP2_START=$(date +%s)
STEP2_FAILED=0
file_count=1
total_files=${#L0_ZARR_FILES[@]}

for L0_ZARR in "${L0_ZARR_FILES[@]}"; do
    basename_zarr=$(basename "$L0_ZARR" .zarr)
    L1_ZARR="${OUTPUT_L1_DIR}/${basename_zarr}_L1.zarr"

    echo "------------------------------------------------"
    echo "Processing ${file_count}/${total_files}: $basename_zarr"
    echo "------------------------------------------------"
    echo "  Input (L0):  $L0_ZARR"
    echo "  Output (L1): $L1_ZARR"
    echo ""

    if [ "$USE_CLUSTER" = true ]; then
        if ! python3 step2_dask_baseline.py "$L0_ZARR" "$L1_ZARR" --config "$CONFIG_FILE" --dask-scheduler "$SCHEDULER_ADDRESS"; then
            echo "ERROR: Step 2 failed for $basename_zarr"
            STEP2_FAILED=1
            break
        fi
    else
        if ! python3 step2_dask_baseline.py "$L0_ZARR" "$L1_ZARR" --config "$CONFIG_FILE"; then
            echo "ERROR: Step 2 failed for $basename_zarr"
            STEP2_FAILED=1
            break
        fi
    fi

    echo ""
    echo "✓ Completed $basename_zarr"
    echo ""

    file_count=$((file_count + 1))
done

STEP2_END=$(date +%s)
STEP2_DURATION=$((STEP2_END - STEP2_START))

if [ "$STEP2_FAILED" -eq 1 ]; then
    echo "================================================"
    echo "Pipeline Failed at Step 2"
    echo "================================================"
    exit 1
fi

echo "✓ Step 2 complete (${STEP2_DURATION}s)"
echo ""

#==============================================================================
# SUMMARY
#==============================================================================

TOTAL_END=$(date +%s)
TOTAL_DURATION=$((TOTAL_END - STEP1_START))

echo "================================================"
echo "Pipeline Complete!"
echo "================================================"
echo ""
echo "Timing Summary:"
echo "  Step 1 (PFF->Zarr):      ${STEP1_DURATION}s"
echo "  Step 2 (Baseline):       ${STEP2_DURATION}s"
echo "  Total time:              ${TOTAL_DURATION}s"
echo ""
echo "Output directories:"
echo "  L0 Zarr files: $OUTPUT_L0_DIR"
echo "  L1 Zarr files: $OUTPUT_L1_DIR"
echo ""

echo "L0 files created:"
for zarr_file in "${L0_ZARR_FILES[@]}"; do
    if [ -d "$zarr_file" ]; then
        size=$(du -sh "$zarr_file" | cut -f1)
        echo "  ✓ $(basename "$zarr_file") [$size]"
    fi
done
echo ""

echo "L1 files created:"
for L0_ZARR in "${L0_ZARR_FILES[@]}"; do
    basename_zarr=$(basename "$L0_ZARR" .zarr)
    L1_ZARR="${OUTPUT_L1_DIR}/${basename_zarr}_L1.zarr"
    if [ -d "$L1_ZARR" ]; then
        size=$(du -sh "$L1_ZARR" | cut -f1)
        echo "  ✓ $(basename "$L1_ZARR") [$size]"
    else
        echo "  ✗ $(basename "$L1_ZARR") (MISSING)"
    fi
done
echo ""

echo "================================================"
echo "Pipeline execution complete!"
echo "================================================"
