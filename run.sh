#!/bin/bash
# run.sh - Execute the PANOSETI L0 to L1 data pipeline
#
# This script performs two steps:
# 1. Convert PFF files to Zarr format (L0)
# 2. Apply baseline subtraction to create L1 product
#
# Usage: ./run.sh <input_pff_files...> <output_l1_directory>
#
# Example: ./run.sh file1.pff file2.pff /path/to/output/L1

set -e  # Exit on error
set -u  # Exit on undefined variable

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
mkdir -p "${OUTPUT_L1_DIR}"
mkdir -p "${L0_TEMP_DIR}"

echo "  L0 temp directory: ${L0_TEMP_DIR}"
echo ""

# Verify all input files exist
for pff_file in "${INPUT_FILES[@]}"; do
    if [ ! -f "${pff_file}" ]; then
        echo "Error: Input file not found: ${pff_file}"
        exit 1
    fi
done

# Check if required Python scripts exist
if [ ! -f "step1_pff_to_zarr.py" ]; then
    echo "Error: step1_pff_to_zarr.py not found!"
    exit 1
fi

if [ ! -f "step2_dask_baseline.py" ]; then
    echo "Error: step2_dask_baseline.py not found!"
    exit 1
fi

# Process each input file
file_count=0
total_files=${#INPUT_FILES[@]}

for pff_file in "${INPUT_FILES[@]}"; do
    ((file_count++))

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
    python3 step1_pff_to_zarr.py "${pff_file}" "${L0_ZARR}"

    echo ""
    echo "Step 2/2: Applying baseline subtraction (L0 -> L1)..."
    echo "  Input:  ${L0_ZARR}"
    echo "  Output: ${L1_ZARR}"
    python3 step2_dask_baseline.py "${L0_ZARR}" "${L1_ZARR}"

    # Clean up L0 temporary file to save space
    echo ""
    echo "Cleaning up L0 temporary file..."
    rm -rf "${L0_ZARR}"

    echo ""
    echo "✓ Completed ${basename}"
    echo ""
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
