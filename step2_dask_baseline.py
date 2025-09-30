#!/usr/bin/env python3
"""
Step 2: Take L0 Zarr data product as input, perform baseline subtraction with Dask,
and output L1 data product
"""

import os
import sys
import argparse
import numpy as np
import xarray as xr
import dask
import dask.array as da
from pathlib import Path
import time

def baseline_subtract(zarr_input: str, zarr_output: str, baseline_window: int = 100):
    """
    Perform baseline subtraction on L0 Zarr data to produce L1 data product.

    Parameters:
    -----------
    zarr_input : str
        Path to input L0 Zarr directory
    zarr_output : str
        Path to output L1 Zarr directory
    baseline_window : int
        Number of frames to use for rolling baseline calculation
    """

    print(f"Loading L0 data from: {zarr_input}")
    start = time.time()

    # Open the L0 Zarr dataset
    ds = xr.open_dataset(
        zarr_input,
        engine='zarr',
        chunks={}  # Use existing chunks
    )

    print(f"Dataset loaded in {time.time() - start:.2f}s")
    print(f"Dataset shape: {ds.images.shape}")
    print(f"Dataset chunks: {ds.images.chunks}")

    # Perform baseline subtraction
    # Calculate rolling mean as baseline
    print(f"\nCalculating baseline (window={baseline_window})...")

    images_da = ds.images.data  # Get dask array

    # Calculate baseline - mean of first N frames
    baseline = images_da[:baseline_window].mean(axis=0, keepdims=True)

    # Subtract baseline
    print("Subtracting baseline...")
    images_corrected = images_da - baseline

    # Create output dataset
    ds_out = xr.Dataset(
        {
            'images': (['time', 'y', 'x'], images_corrected),
            'timestamps': ds.timestamps,
        },
        attrs={
            **ds.attrs,
            'processing_level': 'L1',
            'baseline_window': baseline_window,
            'baseline_method': 'mean_of_first_n_frames',
        }
    )

    # Save to Zarr
    print(f"\nWriting L1 data to: {zarr_output}")

    # Remove existing output if it exists
    if os.path.exists(zarr_output):
        import shutil
        shutil.rmtree(zarr_output)

    # Write with progress
    write_start = time.time()

    ds_out.to_zarr(
        zarr_output,
        mode='w',
        compute=True,
        consolidated=True
    )

    write_time = time.time() - write_start
    total_time = time.time() - start

    print(f"\nProcessing complete!")
    print(f"Write time: {write_time:.2f}s")
    print(f"Total time: {total_time:.2f}s")
    print(f"Output: {zarr_output}")

    # Print output size
    def get_dir_size(path):
        total = 0
        for dirpath, dirnames, filenames in os.walk(path):
            for f in filenames:
                fp = os.path.join(dirpath, f)
                if os.path.exists(fp):
                    total += os.path.getsize(fp)
        return total

    output_size = get_dir_size(zarr_output)
    print(f"Output size: {output_size / 1e9:.2f} GB")

def main():
    parser = argparse.ArgumentParser(
        description='Apply baseline subtraction to L0 Zarr data to create L1 product',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        'input_zarr',
        type=str,
        help='Path to input L0 Zarr directory'
    )

    parser.add_argument(
        'output_zarr',
        type=str,
        help='Path to output L1 Zarr directory'
    )

    parser.add_argument(
        '--baseline-window',
        type=int,
        default=100,
        help='Number of frames to use for baseline calculation (default: 100)'
    )

    args = parser.parse_args()

    if not os.path.exists(args.input_zarr):
        print(f"Error: Input path does not exist: {args.input_zarr}")
        sys.exit(1)

    baseline_subtract(args.input_zarr, args.output_zarr, baseline_window=args.baseline_window)

if __name__ == "__main__":
    main()
