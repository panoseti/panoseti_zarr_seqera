#!/bin/bash
# run.sh - Execute the PANOSETI L0 to L1 data pipeline
#
# This script performs two steps:
# 1. Convert PFF files to Zarr format (L0)
# 2. Apply baseline subtraction to create L1 product
#
# Usage: ./run.sh <input_pff_file1> [input_pff_file2 ...] <output_l1_directory>
#
# Example: ./run.sh file1.pff file2.pff /path/to/output/L1

set -e  # Exit on error
set -u  # Exit on undefined variable

# Enable debugging - uncomment to see each command
# set -x

# Function to display usage
usage() {
    echo "Usage: $0 <input_pff_file1> [input_pff_file2 ...] <output_l1_directory>"
    echo ""
    echo "Arguments:"
    echo "  input_pff_file    One or more PFF files to process"
    echo "  output_l1_dir     Directory where L1 products will be written"
    echo ""
    echo "Example:"
    echo "  $0 file1.pff file2.pff /scratch/user/output/L1"
    echo ""
    echo "Requirements:"
    echo "  - pff.py must be in the same directory or in PYTHONPATH"
    echo "  - step1_pff_to_zarr.py and step2_dask_baseline.py must be available"
    exit 1
}

# Check minimum number of arguments (at least 2: one input file + output dir)
if [ $# -lt 2 ]; then
    echo "Error: Insufficient arguments"
    echo ""
    usage
fi

# Parse arguments - last argument is output directory
OUTPUT_L1_DIR="${@: -1}"
# All arguments except last are input files
INPUT_FILES=("${@:1:$#-1}")

echo "================================================"
echo "PANOSETI L0 -> L1 Processing Pipeline"
echo "================================================"
echo ""
echo "Configuration:"
echo "  Input files: ${#INPUT_FILES[@]}"
for file in "${INPUT_FILES[@]}"; do
    echo "    - $file"
done
echo "  Output L1 directory: ${OUTPUT_L1_DIR}"
echo ""

# Create output directories
L0_TEMP_DIR="${OUTPUT_L1_DIR}/../L0_temp"
mkdir -p "${OUTPUT_L1_DIR}" || {
    echo "Error: Failed to create output directory: ${OUTPUT_L1_DIR}"
    exit 1
}
mkdir -p "${L0_TEMP_DIR}" || {
    echo "Error: Failed to create L0 temp directory: ${L0_TEMP_DIR}"
    exit 1
}

echo "  L0 temp directory: ${L0_TEMP_DIR}"
echo ""

# Verify all input files exist
echo "Verifying input files..."
for pff_file in "${INPUT_FILES[@]}"; do
    if [ ! -f "${pff_file}" ]; then
        echo "Error: Input file not found: ${pff_file}"
        exit 1
    fi
    echo "  [OK] Found: ${pff_file}"
done
echo ""

# Check if required Python scripts exist
echo "Checking required scripts..."
if [ ! -f "step1_pff_to_zarr.py" ]; then
    echo "Error: step1_pff_to_zarr.py not found in current directory!"
    echo "Current directory: $(pwd)"
    ls -la step*.py 2>/dev/null || echo "No step*.py files found"
    exit 1
fi
echo "  [OK] Found: step1_pff_to_zarr.py"

if [ ! -f "step2_dask_baseline.py" ]; then
    echo "Error: step2_dask_baseline.py not found in current directory!"
    exit 1
fi
echo "  [OK] Found: step2_dask_baseline.py"

if [ ! -f "pff.py" ]; then
    echo "Error: pff.py not found in current directory!"
    exit 1
fi
echo "  [OK] Found: pff.py"
echo ""

# Check Python and required packages
echo "Checking Python environment..."
python3 --version || {
    echo "Error: python3 not found!"
    exit 1
}

echo "Checking Python packages..."
python3 -c "import tensorstore; print('  [OK] tensorstore')" || {
    echo "Error: tensorstore package not found!"
    echo "Install with: pip install tensorstore"
    exit 1
}

python3 -c "import zarr; print('  [OK] zarr')" || {
    echo "Error: zarr package not found!"
    exit 1
}

python3 -c "import xarray; print('  [OK] xarray')" || {
    echo "Error: xarray package not found!"
    exit 1
}

python3 -c "import dask; print('  [OK] dask')" || {
    echo "Error: dask package not found!"
    exit 1
}

python3 -c "import numpy; print('  [OK] numpy')" || {
    echo "Error: numpy package not found!"
    exit 1
}

python3 -c "import tqdm; print('  [OK] tqdm')" || {
    echo "Error: tqdm package not found!"
    exit 1
}
echo ""

# Process each input file
file_count=1  # Start at 1 to avoid ((0++)) issue with set -e
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

    python3 step1_pff_to_zarr.py "${pff_file}" "${L0_ZARR}" || {
        echo ""
        echo "=========================================="
        echo "ERROR: Step 1 failed for ${pff_file}"
        echo "=========================================="
        exit 1
    }

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

    # Increment counter (now safe because it won't be 0)
    file_count=$((file_count + 1))
done

# Clean up L0 temp directory if empty
if [ -d "${L0_TEMP_DIR}" ] && [ -z "$(ls -A ${L0_TEMP_DIR})" ]; then
    rmdir "${L0_TEMP_DIR}"
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
    echo "  - ${OUTPUT_L1_DIR}/${basename}_L1.zarr"
done
echo ""

