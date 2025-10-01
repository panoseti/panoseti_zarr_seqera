#!/usr/bin/env python3

"""
Step 2: Baseline subtraction using Dask distributed computing

UPDATED: Now accepts existing Dask scheduler address instead of creating cluster
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
from dask.distributed import Client, wait
from dask.diagnostics import ProgressBar
import cluster_manager

# Try to import tomli/tomllib
try:
    import tomllib
except ImportError:
    try:
        import tomli as tomllib
    except ImportError:
        tomllib = None

def load_config(config_path: str = "config.toml"):
    """Load configuration from TOML file"""
    config = {
        'baseline_window': 100,
        'codec': 'blosc-lz4',
        'level': 5,
        'compute_chunk_size': 8192,
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

async def connect_to_dask(scheduler_address: str):
    """Connect to existing Dask scheduler"""
    if not scheduler_address:
        print("No Dask scheduler address provided - using local threading")
        return None
    
    try:
        print(f"Connecting to Dask scheduler: {scheduler_address}")
        client = await Client(scheduler_address, asynchronous=True)
        print(f"  ✓ Connected! Workers: {len(client.scheduler_info()['workers'])}")
        return client
    except Exception as e:
        print(f"Warning: Could not connect to Dask scheduler: {e}")
        print("Falling back to local threading")
        return None

def baseline_subtract_dask(zarr_input: str, zarr_output: str, config: dict, client=None):
    """
    Perform baseline subtraction using Dask distributed computing.
    """
    baseline_window = config.get('baseline_window', 100)
    codec = config.get('codec', 'blosc-lz4')
    level = config.get('level', 5)
    compute_chunk_size = config.get('compute_chunk_size', 8192)
    
    print(f"Loading L0 data from: {zarr_input}")
    start = time.time()
    
    # Open the L0 Zarr dataset with Dask arrays
    ds = xr.open_dataset(
        zarr_input,
        engine='zarr',
        chunks={},  # Use existing chunks
        consolidated=False
    )
    
    print(f"Dataset loaded in {time.time() - start:.2f}s")
    print(f"Dataset shape: {ds.images.shape}")
    print(f"Dataset chunks: {ds.images.chunks}")
    
    # Check if we have a Dask client
    use_distributed = client is not None
    if use_distributed:
        print(f"\nUsing Dask distributed computing")
        print(f"Workers: {len(client.scheduler_info()['workers'])}")
        print(f"Total cores: {sum(w['nthreads'] for w in client.scheduler_info()['workers'].values())}")
    else:
        print(f"\nUsing local Dask threading")
    
    # Perform baseline subtraction with Dask
    print(f"\nCalculating baseline (window={baseline_window})...")
    images_da = ds.images.data  # Get dask array
    
    # Calculate baseline - mean of first N frames
    baseline = images_da[:baseline_window].mean(axis=0, keepdims=True)
    
    # Subtract baseline (lazy)
    print("Subtracting baseline (lazy computation)...")
    images_corrected = images_da - baseline
    
    # Rechunk for better performance if needed
    if compute_chunk_size != ds.images.chunks[0][0]:
        print(f"Rechunking from {ds.images.chunks[0][0]} to {compute_chunk_size} frames per chunk...")
        images_corrected = images_corrected.rechunk({0: compute_chunk_size, 1: -1, 2: -1})
    
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
    
    # Build encoding dict
    encoding = {
        'images': {
            'chunks': tuple([images_corrected.chunks[0][0] if isinstance(images_corrected.chunks[0], tuple)
                           else images_corrected.chunks[0]] + [ds.images.shape[1], ds.images.shape[2]]),
            'dtype': ds.images.dtype,
        },
        'timestamps': {
            'dtype': ds.timestamps.dtype,
        }
    }
    
    # Write with progress
    write_start = time.time()
    if use_distributed:
        # Use Dask distributed compute
        print("Computing with Dask cluster...")
        future = ds_out.to_zarr(
            zarr_output,
            mode='w',
            encoding=encoding,
            compute=False,
            consolidated=True
        )
        result = client.compute(future, sync=True)
    else:
        # Use local Dask with progress bar
        print("Computing locally with Dask...")
        with ProgressBar():
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
    print(f"  Input size (L0): {input_size / 1e6:.2f} MB")
    print(f"  Output size (L1): {output_size / 1e6:.2f} MB")
    print(f"  Compression ratio: {compression_ratio:.2f}x")
    print(f"  Space savings: {(1 - output_size/input_size)*100:.1f}%")

async def main_async():
    parser = argparse.ArgumentParser(
        description='Apply baseline subtraction with Dask distributed computing',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('input_zarr', type=str, help='Path to input L0 Zarr directory')
    parser.add_argument('output_zarr', type=str, help='Path to output L1 Zarr directory')
    parser.add_argument('--config', type=str, default='config.toml',
                       help='Path to TOML configuration file (default: config.toml)')
    parser.add_argument('--baseline-window', type=int, default=None,
                       help='Number of frames for baseline (overrides config)')
    parser.add_argument('--dask-scheduler', type=str, default=None,
                       help='Dask scheduler address')
    
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
    
    # Connect to Dask cluster if scheduler provided
    client = None
    if args.dask_scheduler:
        client = await connect_to_dask(args.dask_scheduler)
    
    try:
        baseline_subtract_dask(args.input_zarr, args.output_zarr, config, client)
    finally:
        # Close client connection (but don't shutdown cluster)
        if client:
            print("  Keeping cluster alive for next task...")

def main():
    """Synchronous wrapper for async main"""
    import asyncio
    asyncio.run(main_async())

if __name__ == "__main__":
    main()
