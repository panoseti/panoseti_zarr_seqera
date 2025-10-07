#!/usr/bin/env python3

"""
Step 2: Advanced baseline/median subtraction using Dask distributed computing

This script implements two processing modes:
1. Pulse-height data (ph*): Pedestal subtraction with 5-sigma thresholding
2. Image data (img*): 8x8 block median + temporal median subtraction

UPDATED: Now accepts existing Dask scheduler address instead of creating cluster
UPDATED: Implements the exact workflow from the Jupyter notebook
"""

import os
import sys
import argparse
import numpy as np
import xarray as xr
import zarr
import dask
import dask.array as da
from pathlib import Path
import time
from dask.distributed import Client, wait, LocalCluster
from dask.diagnostics import ProgressBar

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
        # Pulse-height data specific
        'ph_baseline_offset': 800,
        'ph_sigma_threshold': 5,
        # Image data specific
        'img_frame_step': 200,
        'img_block_size': 8,
        'img_adc_to_pe': 1.5,
        'img_zstd_level': 7,
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
    """Connect to existing Dask scheduler or start a local one"""
    client = None
    cluster = None
    if not scheduler_address:
        print("No Dask scheduler address provided - starting a local Dask cluster")
        try:
            cluster = LocalCluster(n_workers=os.cpu_count() or 4, processes=True, threads_per_worker=1)
            client = await Client(cluster, asynchronous=True)
            print(f"  ✓ Local Dask cluster started at: {client.scheduler_info()['address']}")
            return client, cluster
        except Exception as e:
            print(f"Error starting local Dask cluster: {e}")
            print("Falling back to local threading")
            return None, None

    try:
        print(f"Connecting to Dask scheduler: {scheduler_address}")
        client = await Client(scheduler_address, asynchronous=True)
        worker_info = client.scheduler_info()
        print(f"  ✓ Connected! Workers: {len(worker_info['workers'])}")
        return client, None # No local cluster to manage if connecting to existing
    except Exception as e:
        print(f"Warning: Could not connect to Dask scheduler: {e}")
        print("Falling back to local threading")
        return None, None


def detect_data_product(zarr_path: str) -> str:
    """Detect data product type from path"""
    zarr_path_lower = str(zarr_path).lower()
    if 'ph' in zarr_path_lower or 'pulse-height' in zarr_path_lower:
        return 'pulse-height'
    elif 'img' in zarr_path_lower or 'image' in zarr_path_lower:
        return 'image'
    else:
        raise ValueError(f"Cannot detect data product type from path: {zarr_path}")


async def process_pulse_height_data(ds: xr.Dataset, config: dict) -> xr.DataArray:
    """
    Process pulse-height data (ph*) with pedestal subtraction and sigma thresholding.

    Workflow:
    1. Add baseline offset (800) to handle negative values
    2. Compute median pedestal across time
    3. Subtract pedestal
    4. Calculate 5-sigma threshold
    5. Mask values below threshold
    """
    print("\nProcessing pulse-height data...")

    baseline_offset = config.get('ph_baseline_offset', 800)
    sigma_threshold = config.get('ph_sigma_threshold', 5)

    print(f"  Baseline offset: {baseline_offset}")
    print(f"  Sigma threshold: {sigma_threshold}")

    # Check if data needs swapping (for 32x32 ph1024)
    if ds.images.shape[1] == 32:
        print("  Detected ph1024 (32x32) - swapping quabo positions")
        # Swap quabo positions: [0,1,2,3] -> [0,2,1,3]
        images_swapped = ds.images.copy()
        # Top-left (0,0:16) stays, top-right and bottom-left swap
        temp = images_swapped[:, :16, 16:].copy()
        images_swapped[:, :16, 16:] = images_swapped[:, 16:, :16]
        images_swapped[:, 16:, :16] = temp
        pre_preprocessed = images_swapped
    else:
        print("  Detected ph256 (16x16) - no swapping needed")
        pre_preprocessed = ds.images

    # Add baseline offset
    add_baselines = pre_preprocessed + baseline_offset

    # Compute median pedestal across time
    print("  Computing median pedestal (lazy)...")
    pedestal = add_baselines.median(dim='time')

    # Subtract pedestal
    print("  Subtracting pedestal (lazy)...")
    pedestal_sub = add_baselines - pedestal

    # Define threshold and mask
    print(f"  Computing {sigma_threshold}-sigma threshold (lazy)...")
    sigma_n = pedestal_sub.std('time') * sigma_threshold
    sigma_n_above_mask = (pedestal_sub > sigma_n)

    # Create masked array (values below threshold become NaN)
    print("  Applying mask (lazy)...")
    ph_sigma_above = pedestal_sub.where(sigma_n_above_mask)

    ph_sigma_above.name = 'pedestal_subtracted_data'

    return ph_sigma_above


async def process_image_data(ds: xr.Dataset, config: dict) -> xr.DataArray:
    """
    Process image data (img*) with 8x8 block median + temporal median subtraction.

    Workflow:
    1. Convert to int32 to avoid overflow
    2. Compute 8x8 block medians on strided subset
    3. Upsample back to 32x32
    4. Subtract spatial medians
    5. Compute temporal supermedian
    6. Subtract supermedian
    7. Convert to photoelectrons
    """
    print("\nProcessing image data...")

    frame_step = config.get('img_frame_step', 200)
    block_size = config.get('img_block_size', 8)
    adc_to_pe = config.get('img_adc_to_pe', 1.5)

    print(f"  Frame step: {frame_step}")
    print(f"  Block size: {block_size}x{block_size}")
    print(f"  ADC to PE conversion: {adc_to_pe}")

    # Convert to int32 to avoid uint16 overflow during subtraction
    print("  Converting to int32 (lazy)...")
    img_int32 = ds.images.astype('int32')

    # Compute 8x8 block medians on strided subset, then median over time
    print(f"  Computing {block_size}x{block_size} block medians (lazy)...")
    block_medians = (
        img_int32[::frame_step]
        .coarsen(y=block_size, x=block_size, boundary="trim")
        .median()
        .median('time')
    )  # Result: (4, 4) for 32x32 with block_size=8

    # Upsample 4x4 medians to 32x32 using da.repeat
    print(f"  Upsampling to full resolution (lazy)...")
    upsampled_medians_da = da.repeat(block_medians.data, block_size, axis=0)
    upsampled_medians_da = da.repeat(upsampled_medians_da, block_size, axis=1)

    # Wrap in xarray.DataArray to restore coordinates for broadcasting
    upsampled_medians = xr.DataArray(
        upsampled_medians_da,
        dims=('y', 'x'),
        coords={'y': ds.images.y, 'x': ds.images.x},
    )

    # Subtract spatial medians via broadcasting
    print("  Subtracting spatial medians (lazy)...")
    median_subtraction_8x8 = img_int32 - upsampled_medians

    # Compute temporal supermedian and subtract
    print("  Computing temporal supermedian (lazy)...")
    supermedian_img = median_subtraction_8x8[::frame_step].median('time')

    print("  Subtracting temporal supermedian (lazy)...")
    median_subtraction_final = median_subtraction_8x8 - supermedian_img

    # Convert to photoelectrons
    print(f"  Converting to photoelectrons (lazy)...")
    median_subtraction_final_pe = median_subtraction_final / adc_to_pe

    median_subtraction_final_pe.name = 'median_subtracted_data'

    return median_subtraction_final_pe


async def baseline_subtract_dask(zarr_input: str, zarr_output: str, config: dict, client=None):
    """
    Perform baseline/median subtraction using Dask distributed computing.

    Automatically detects data product type (pulse-height vs image) and applies
    appropriate processing workflow.
    """
    print(f"Loading L0 data from: {zarr_input}")
    start = time.time()

    # Detect data product type
    data_product = detect_data_product(zarr_input)
    print(f"Detected data product: {data_product}")

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
        worker_info = client.scheduler_info()
        print(f"\nUsing Dask distributed computing")
        print(f"Workers: {len(worker_info['workers'])}")
        print(f"Total cores: {sum(w['nthreads'] for w in worker_info['workers'].values())}")
    else:
        print(f"\nUsing local Dask threading")

    # Process based on data product type
    if data_product == 'pulse-height':
        processed_data = await process_pulse_height_data(ds, config)
    else:  # image
        processed_data = await process_image_data(ds, config)

    # Save to Zarr with compression
    print(f"\nWriting L1 data to: {zarr_output}")

    # Remove existing output if it exists
    if os.path.exists(zarr_output):
        import shutil
        shutil.rmtree(zarr_output)

    # Set umask for group write permissions
    original_umask = os.umask(0o000)

    try:
        # Choose compression based on data product type
        if data_product == 'pulse-height':
            # For pulse-height data, use standard compression
            print(f"Using compression: {config.get('codec', 'blosc-lz4')} level {config.get('level', 5)}")

            store_operation = processed_data.to_zarr(
                zarr_output,
                mode='w',
                consolidated=True,
                compute=False,
                zarr_format=3
            )
        else:
            # For image data, use Zstd compression
            zstd_level = config.get('img_zstd_level', 7)
            print(f"Using compression: Zstd level {zstd_level}")

            compressor = zarr.codecs.ZstdCodec(level=zstd_level)
            encoding = {
                processed_data.name: {"compressors": [compressor]}
            }

            store_operation = processed_data.to_zarr(
                zarr_output,
                mode='w',
                consolidated=True,
                encoding=encoding,
                compute=False,
                zarr_format=3
            )

        # Compute the result
        write_start = time.time()

        if use_distributed:
            print("Computing with Dask cluster...")
            futures = client.compute(store_operation)
            result = await client.gather(futures)
        else:
            print("Computing locally with Dask...")
            with ProgressBar():
                result = store_operation.compute()

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
        print(f"  Space savings: {(1 - output_size / input_size) * 100:.1f}%")

    finally:
        os.umask(original_umask)


async def main_async():
    parser = argparse.ArgumentParser(
        description='Apply baseline/median subtraction with Dask distributed computing',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Data Product Processing:
  pulse-height data (ph*): Pedestal subtraction with 5-sigma thresholding
  Image data (img*): 8x8 block median + temporal median subtraction

Examples:
  # Process with local threading
  python step2_dask_baseline.py input.zarr output.zarr

  # Process with Dask cluster
  python step2_dask_baseline.py input.zarr output.zarr --dask-scheduler tcp://10.0.1.2:8786
        """
    )

    parser.add_argument('input_zarr', type=str, help='Path to input L0 Zarr directory')
    parser.add_argument('output_zarr', type=str, help='Path to output L1 Zarr directory')
    parser.add_argument('--config', type=str, default='config.toml',
                        help='Path to TOML configuration file (default: config.toml)')
    parser.add_argument('--dask-scheduler', type=str, default=None,
                        help='Dask scheduler address (e.g., tcp://10.0.1.2:8786)')

    args = parser.parse_args()

    if not os.path.exists(args.input_zarr):
        print(f"Error: Input path does not exist: {args.input_zarr}")
        sys.exit(1)

    # Load config
    config = load_config(args.config)

    print(f"\nConfiguration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print()

    # Connect to Dask cluster if scheduler provided
    client = None
    local_cluster = None
    if args.dask_scheduler:
        client, local_cluster = await connect_to_dask(args.dask_scheduler)
    else:
        client, local_cluster = await connect_to_dask(None) # Call with None to trigger local cluster creation

    try:
        await baseline_subtract_dask(args.input_zarr, args.output_zarr, config, client)
    finally:
        # Close client connection and shutdown local cluster if it was started by this script
        if client:
            print("\n  ✓ Closing Dask client...")
            await client.close()
        if local_cluster:
            print("  ✓ Shutting down local Dask cluster...")
            await local_cluster.close()


def main():
    """Synchronous wrapper for async main"""
    import asyncio
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
