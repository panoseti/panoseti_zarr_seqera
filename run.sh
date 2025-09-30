#!/bin/bash
# run.sh - Execute the PANOSETI L0 to L1 data pipeline
#
# This script performs two steps:
# 1. Convert PFF files to Zarr format (L0)
# 2. Apply baseline subtraction to create L1 product

set -e  # Exit on error

# Configuration
IMG16_FILE="start_2024-07-25T04_34_46Z.dp_img16.bpp_2.module_1.seqno_0.debug_TRUNCATED.pff"
PH256_FILE="start_2024-07-25T04_34_46Z.dp_ph256.bpp_2.module_1.seqno_0.debug_TRUNCATED.pff"

# Output directories
OUTPUT_DIR="output"
L0_DIR="${OUTPUT_DIR}/L0"
L1_DIR="${OUTPUT_DIR}/L1"

# Create output directories
mkdir -p "${L0_DIR}"
mkdir -p "${L1_DIR}"

echo "================================================"
echo "PANOSETI L0 -> L1 Processing Pipeline"
echo "================================================"
echo ""

# Check if input files exist
if [ ! -f "${IMG16_FILE}" ]; then
    echo "Error: ${IMG16_FILE} not found!"
    echo "Please ensure the PFF files are in the current directory."
    exit 1
fi

if [ ! -f "${PH256_FILE}" ]; then
    echo "Error: ${PH256_FILE} not found!"
    echo "Please ensure the PFF files are in the current directory."
    exit 1
fi

# Check if pff.py exists
if [ ! -f "pff.py" ]; then
    echo "Error: pff.py not found!"
    echo "This file is required for parsing PFF format."
    exit 1
fi

# Process IMG16 file
echo "================================================"
echo "Processing IMG16 file: ${IMG16_FILE}"
echo "================================================"
echo ""

IMG16_L0_ZARR="${L0_DIR}/img16_L0.zarr"
IMG16_L1_ZARR="${L1_DIR}/img16_L1.zarr"

echo "Step 1: Converting PFF to Zarr (L0)..."
python3 step1_pff_to_zarr.py "${IMG16_FILE}" "${IMG16_L0_ZARR}"

echo ""
echo "Step 2: Applying baseline subtraction (L0 -> L1)..."
python3 step2_dask_baseline.py "${IMG16_L0_ZARR}" "${IMG16_L1_ZARR}"

echo ""
echo "================================================"
echo "Processing PH256 file: ${PH256_FILE}"
echo "================================================"
echo ""

PH256_L0_ZARR="${L0_DIR}/ph256_L0.zarr"
PH256_L1_ZARR="${L1_DIR}/ph256_L1.zarr"

echo "Step 1: Converting PFF to Zarr (L0)..."
python3 step1_pff_to_zarr.py "${PH256_FILE}" "${PH256_L0_ZARR}"

echo ""
echo "Step 2: Applying baseline subtraction (L0 -> L1)..."
python3 step2_dask_baseline.py "${PH256_L0_ZARR}" "${PH256_L1_ZARR}"

echo ""
echo "================================================"
echo "Pipeline Complete!"
echo "================================================"
echo ""
echo "Output files:"
echo "  L0 products:"
echo "    - ${IMG16_L0_ZARR}"
echo "    - ${PH256_L0_ZARR}"
echo "  L1 products:"
echo "    - ${IMG16_L1_ZARR}"
echo "    - ${PH256_L1_ZARR}"
echo ""
