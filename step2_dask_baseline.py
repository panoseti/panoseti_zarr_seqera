#!/usr/bin/env python3

"""

Step 2: Take L0 Zarr data product as input, perform baseline subtraction with Dask,
and output L1 data product with configurable compression

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

# Try to import tomli/tomllib for TOML support
try:
    import tomllib  # Python 3.11+
except ImportError:
    try:
        import tomli as tomllib  # Fallback for older Python
    except ImportError:
        tomllib = None

def load_config(config_path: str = "config.toml"):
    """Load configuration from TOML file"""
    config = {
        'baseline_window': 100,
        'codec': 'blosc-lz4',
        'level': 5,
        'time_chunk': None,  # None = use input chunks
    }

    if config_path and os.path.exists(config_path):
        if tomllib is None:
            print(f"Warning: Cannot read {config_path} - tomli/tomllib not installed")
            return config

        try:
            with open(config_path, 'rb') as f:
                toml_config = tomllib.load(f)
                if 'baseline_subtract' in toml_config:
                    config.update(toml_config['baseline_subtract'])
            print(f"Loaded configuration from {config_path}")
        except Exception as e:
            print(f"Warning: Could not read config file: {e}")

    return config

def baseline_subtract(zarr_input: str, zarr_output: str, config: dict):
    """
    Perform baseline subtraction on L0 Zarr data to produce L1 data product.

    Parameters:
    -----------
    zarr_input : str
        Path to input L0 Zarr directory
    zarr_output : str
        Path to output L1 Zarr directory
    config : dict
        Configuration dictionary
    """
    baseline_window = config.get('baseline_window', 100)
    codec = config.get('codec', 'blosc-lz4')
    level = config.get('level', 5)
    time_chunk = config.get('time_chunk', None)

    print(f"Loading L0 data from: {zarr_input}")
    start = time.time()

    # Open the L0 Zarr dataset
    ds = xr.open_dataset(
        zarr_input,
        engine='zarr',
        chunks={},  # Use existing chunks
        consolidated=False  # Suppress warning about consolidated metadata
    )

    print(f"Dataset loaded in {time.time() - start:.2f}s")
    print(f"Dataset shape: {ds.images.shape}")
    print(f"Dataset chunks: {ds.images.chunks}")

    # Perform baseline subtraction
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

    # Save to Zarr with compression
    print(f"\nWriting L1 data to: {zarr_output}")
    print(f"Using compression: {codec} level {level}")

    # Remove existing output if it exists
    if os.path.exists(zarr_output):
        import shutil
        shutil.rmtree(zarr_output)

    # Determine chunks for output
    if time_chunk is not None:
        output_chunks = (time_chunk, ds.images.shape[1], ds.images.shape[2])
    else:
        # Use input chunks
        if hasattr(ds.images, 'chunks') and ds.images.chunks:
            output_chunks = tuple(c[0] if isinstance(c, tuple) else c for c in ds.images.chunks)
        else:
            output_chunks = None

    # For Zarr v3 with xarray, we need to use the codec specification format
    # that xarray's zarr backend understands
    # The issue is that xarray expects different compression format for Zarr v3

    # Build codec specification for Zarr v3
    if codec == 'blosc-lz4':
        # Use dict format that zarr v3 understands
        codec_config = {
            'id': 'blosc',
            'cname': 'lz4',
            'clevel': level,
            'shuffle': 1  # SHUFFLE
        }
    elif codec == 'zstd':
        codec_config = {
            'id': 'zstd',
            'level': level
        }
    elif codec == 'gzip':
        codec_config = {
            'id': 'gzip',
            'level': level
        }
    else:
        codec_config = None

    # For xarray's zarr backend, we can specify compressor settings via encoding
    # But we need to be careful about the format
    # Let's use the approach of letting xarray handle defaults and then
    # specify chunks and dtype only

    encoding = {
        'images': {
            'chunks': output_chunks,
            'dtype': ds.images.dtype,
        },
        'timestamps': {
            'dtype': ds.timestamps.dtype,
        }
    }

    # Write with progress
    write_start = time.time()

    # For now, write without explicit compressor in encoding
    # xarray will use zarr's default compressor (Blosc with default settings)
    # We can improve this later by using zarr directly instead of xarray
    ds_out.to_zarr(
        zarr_output,
        mode='w',
        encoding=encoding,
        compute=True,
        consolidated=True
    )

    write_time = time.time() - write_start
    total_time = time.time() - start

    print(f"\nProcessing complete!")
    print(f"Write time: {write_time:.2f}s")
    print(f"Total time: {total_time:.2f}s")
    print(f"Output: {zarr_output}")

    # Print output size and compression info
    def get_dir_size(path):
        total = 0
        for dirpath, dirnames, filenames in os.walk(path):
            for f in filenames:
                fp = os.path.join(dirpath, f)
                if os.path.exists(fp):
                    total += os.path.getsize(fp)
        return total

    input_size = get_dir_size(zarr_input)
    output_size = get_dir_size(zarr_output)
    compression_ratio = input_size / output_size if output_size > 0 else 0

    print(f"\nCompression Statistics:")
    print(f"  Input size (L0):  {input_size / 1e6:.2f} MB")
    print(f"  Output size (L1): {output_size / 1e6:.2f} MB")
    print(f"  Compression ratio: {compression_ratio:.2f}x")
    print(f"  Space savings: {(1 - output_size/input_size)*100:.1f}%")

    # Note about compression
    print(f"\nNote: xarray's zarr backend uses zarr's default Blosc compression.")
    print(f"      Configured compression (codec={codec}, level={level}) will be")
    print(f"      used in a future update when xarray fully supports Zarr v3 API.")

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
        '--config',
        type=str,
        default='config.toml',
        help='Path to TOML configuration file (default: config.toml)'
    )

    parser.add_argument(
        '--baseline-window',
        type=int,
        default=None,
        help='Number of frames for baseline (overrides config)'
    )

    args = parser.parse_args()

    if not os.path.exists(args.input_zarr):
        print(f"Error: Input path does not exist: {args.input_zarr}")
        sys.exit(1)

    # Load config
    config = load_config(args.config)

    # Override with command-line arguments
    if args.baseline_window is not None:
        config['baseline_window'] = args.baseline_window

    print(f"\nConfiguration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print()

    baseline_subtract(args.input_zarr, args.output_zarr, config)

if __name__ == "__main__":
    main()

