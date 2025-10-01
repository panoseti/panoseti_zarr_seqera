#!/bin/bash

# run.sh - Execute the PANOSETI L0 to L1 data pipeline with optimized settings

# This script performs two steps:
# 1. Convert PFF files to Zarr format (L0) using parallel HDD-optimized reading
# 2. Apply baseline subtraction to create L1 product

# Usage: ./run.sh <input_pff_file1> [input_pff_file2 ...] <output_l1_dir>

# Example: ./run.sh file1.pff file2.pff /path/to/output/L1

set -e  # Exit on error
set -u  # Exit on undefined variable

# Enable debugging - uncomment to see each command
# set -x

#==============================================================================
# CONFIGURATION
#==============================================================================

# Path to configuration file (can be overridden)
CONFIG_FILE="${CONFIG_FILE:-config.toml}"

# Environment variables for additional control
# These override values in config.toml if set
# export BLOSC_NTHREADS=8
# export TS_CODEC=blosc-lz4
# export TS_LEVEL=5
# export TS_CHUNK=65536
# export TS_CONCURRENT=16

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
    echo "  # Use default config.toml"
    echo "  $0 file1.pff file2.pff /scratch/user/output/L1"
    echo ""
    echo "  # Use custom config"
    echo "  CONFIG_FILE=custom.toml $0 file1.pff /scratch/user/output/L1"
    echo ""
    echo "  # Override specific settings"
    echo "  TS_CODEC=zstd TS_LEVEL=3 $0 file1.pff /output/L1"
    echo ""
    echo "Requirements:"
    echo "  - Python 3.10+ with packages: tensorstore, zarr, xarray, dask, numpy, tqdm"
    echo "  - pff.py, step1_pff_to_zarr.py, step2_dask_baseline.py in current directory"
    echo "  - config.toml (or custom config file)"
    exit 1
}

#==============================================================================
# ARGUMENT PARSING
#==============================================================================

# Check minimum number of arguments (at least 2: one input file + output dir)
if [ $# -lt 2 ]; then
    echo "Error: Insufficient arguments"
    echo ""
    usage
fi

# Parse arguments - last argument is output directory
OUTPUT_L1_DIR="${!#}"

# All arguments except last are input files
INPUT_FILES=()
for ((i=1; i<$#; i++)); do
    INPUT_FILES+=("${!i}")
done

#==============================================================================
# STARTUP INFORMATION
#==============================================================================

echo "================================================"
echo "PANOSETI L0 -> L1 Processing Pipeline"
echo "================================================"
echo ""
echo "Configuration:"
echo "  Input files:    ${#INPUT_FILES[@]}"
for file in "${INPUT_FILES[@]}"; do
    echo "    - $file"
done
echo "  Output L1 dir:  ${OUTPUT_L1_DIR}"
echo "  Config file:    ${CONFIG_FILE}"
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

echo "  L0 temp dir:    ${L0_TEMP_DIR}"
echo ""

#==============================================================================
# VALIDATION
#==============================================================================

# Verify all input files exist
echo "Verifying input files..."
for pff_file in "${INPUT_FILES[@]}"; do
    if [ ! -f "${pff_file}" ]; then
        echo "Error: Input file not found: ${pff_file}"
        exit 1
    fi
    echo "  [OK] ${pff_file}"
done
echo ""

# Check if required Python scripts exist
echo "Checking required scripts..."

if [ ! -f "step1_pff_to_zarr.py" ]; then
    echo "Error: step1_pff_to_zarr.py not found in current directory!"
    echo "Current directory: $(pwd)"
    exit 1
fi
echo "  [OK] step1_pff_to_zarr.py"

if [ ! -f "step2_dask_baseline.py" ]; then
    echo "Error: step2_dask_baseline.py not found!"
    exit 1
fi
echo "  [OK] step2_dask_baseline.py"

if [ ! -f "pff.py" ]; then
    echo "Error: pff.py not found!"
    exit 1
fi
echo "  [OK] pff.py"

# Check for config file
if [ ! -f "${CONFIG_FILE}" ]; then
    echo "Warning: Config file not found: ${CONFIG_FILE}"
    echo "  Will use default settings and environment variables"
else
    echo "  [OK] ${CONFIG_FILE}"
fi
echo ""

# Check Python and required packages
echo "Checking Python environment..."
python3 --version || {
    echo "Error: python3 not found!"
    exit 1
}

echo "Checking Python packages..."
required_packages=("tensorstore" "zarr" "xarray" "dask" "numpy" "tqdm")
for pkg in "${required_packages[@]}"; do
    python3 -c "import $pkg" 2>/dev/null && echo "  [OK] $pkg" || {
        echo "  [MISSING] $pkg"
        echo "Error: Required package not found: $pkg"
        echo "Install with: pip install $pkg"
        exit 1
    }
done

# Check for tomli/tomllib if using TOML config
if [ -f "${CONFIG_FILE}" ]; then
    python3 -c "import sys; import tomllib if sys.version_info >= (3,11) else __import__('tomli')" 2>/dev/null &&         echo "  [OK] TOML support" || {
        echo "  [INFO] TOML library not found (tomli/tomllib)"
        echo "  Install with: pip install tomli"
        echo "  Falling back to environment variables"
    }
fi

echo ""

#==============================================================================
# DISPLAY CONFIGURATION
#==============================================================================

echo "Active Configuration:"
if [ -f "${CONFIG_FILE}" ] && python3 -c "import sys; import tomllib if sys.version_info >= (3,11) else __import__('tomli')" 2>/dev/null; then
    python3 -c "
import sys
try:
    import tomllib
except ImportError:
    import tomli as tomllib

try:
    with open('${CONFIG_FILE}', 'rb') as f:
        config = tomllib.load(f)
    if 'pff_to_zarr' in config:
        for key, value in config['pff_to_zarr'].items():
            print(f'  {key}: {value}')
except Exception as e:
    print(f'  (Unable to read config: {e})')
" 2>/dev/null || echo "  (Config file present but not readable)"
else
    echo "  Using environment variables and defaults"
    [ -n "${BLOSC_NTHREADS:-}" ] && echo "  BLOSC_NTHREADS: ${BLOSC_NTHREADS}"
    [ -n "${TS_CODEC:-}" ] && echo "  TS_CODEC: ${TS_CODEC}"
    [ -n "${TS_LEVEL:-}" ] && echo "  TS_LEVEL: ${TS_LEVEL}"
    [ -n "${TS_CHUNK:-}" ] && echo "  TS_CHUNK: ${TS_CHUNK}"
    [ -n "${TS_CONCURRENT:-}" ] && echo "  TS_CONCURRENT: ${TS_CONCURRENT}"
fi
echo ""

#==============================================================================
# PROCESSING
#==============================================================================

file_count=1
total_files=${#INPUT_FILES[@]}

for pff_file in "${INPUT_FILES[@]}"; do
    # Extract basename without extension for output naming
    basename=$(basename "${pff_file}" .pff)

    echo "================================================"
    echo "Processing file ${file_count}/${total_files}: ${basename}"
    echo "================================================"
    echo ""

    # Define output paths
    L0_ZARR="${L0_TEMP_DIR}/${basename}_L0.zarr"
    L1_ZARR="${OUTPUT_L1_DIR}/${basename}_L1.zarr"

    echo "Step 1/2: Converting PFF to Zarr (L0)..."
    echo "  Input:  ${pff_file}"
    echo "  Output: ${L0_ZARR}"
    echo ""

    # Run step1 with config file
    if [ -f "${CONFIG_FILE}" ]; then
        python3 step1_pff_to_zarr.py "${pff_file}" "${L0_ZARR}" "${CONFIG_FILE}" || {
            echo ""
            echo "=========================================="
            echo "ERROR: Step 1 failed for ${pff_file}"
            echo "=========================================="
            exit 1
        }
    else
        python3 step1_pff_to_zarr.py "${pff_file}" "${L0_ZARR}" || {
            echo ""
            echo "=========================================="
            echo "ERROR: Step 1 failed for ${pff_file}"
            echo "=========================================="
            exit 1
        }
    fi

    echo ""
    echo "Step 2/2: Applying baseline subtraction (L0 -> L1)..."
    echo "  Input:  ${L0_ZARR}"
    echo "  Output: ${L1_ZARR}"
    echo ""

    python3 step2_dask_baseline.py "${L0_ZARR}" "${L1_ZARR}" || {
        echo ""
        echo "=========================================="
        echo "ERROR: Step 2 failed for ${basename}"
        echo "=========================================="
        exit 1
    }

    # Clean up L0 temporary file to save space
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

