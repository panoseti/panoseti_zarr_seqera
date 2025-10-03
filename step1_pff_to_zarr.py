#!/usr/bin/env python3

"""
Step 1: Convert L0 PFF-formatted data to L0 Zarr-formatted data using Dask distributed computing

UPDATED: Groups all PFF files with same dp_[data product] and module_[id] into single L0 Zarr file
         Files are concatenated based on seqno (sequence number) along the time dimension

Usage:
    python step1_pff_to_zarr.py <observation_dir> <output_zarr_dir> [config.toml] [scheduler_address]

Example:
    python step1_pff_to_zarr.py /mnt/beegfs/data/L0/obs_Lick.start_2024-07-25T04:34:06Z.runtype_sci-data.pffd \
                                /mnt/beegfs/zarr/L0 \
                                config.toml \
                                tcp://10.0.1.2:8786
"""

import os
import sys
import json
import asyncio
import shutil
import zarr
import numpy as np
import tensorstore as ts
from tqdm import tqdm
import time
from pathlib import Path
from collections import defaultdict, deque
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import cpu_count
import pff
from dask.distributed import Client
from typing import List, Tuple, Dict

# Try to import tomli/tomllib for TOML support
try:
    import tomllib
except ImportError:
    try:
        import tomli as tomllib
    except ImportError:
        tomllib = None

# Configure Blosc
try:
    import blosc
    num_cores = os.cpu_count() or 4
    blosc.set_nthreads(num_cores)
except ImportError:
    pass


def _infer_dp_from_name(pff_path: str):
    """Parse data product type from PFF filename"""
    parts = pff.parse_name(os.path.basename(pff_path))
    if not parts or 'dp' not in parts:
        raise ValueError("Could not parse data product from filename.")

    dp = parts['dp']
    bpp = int(parts.get('bpp', 2))

    if dp == 'img16':
        img_shape = (32, 32); dtype = np.uint16; bytes_per_image = 32*32*2
        header_kind = 'img_two_level'
    elif dp == 'img8':
        img_shape = (32, 32); dtype = np.uint8; bytes_per_image = 32*32*1
        header_kind = 'img_two_level'
    elif dp == 'ph256':
        img_shape = (16, 16); dtype = np.int16; bytes_per_image = 16*16*2
        header_kind = 'ph256_one_level'
    elif dp == 'ph1024':
        img_shape = (32, 32); dtype = np.int16; bytes_per_image = 32*32*2
        header_kind = 'img_two_level'
    else:
        raise ValueError(f"Unsupported data product type: {dp}")

    return parts, dp, bpp, img_shape, dtype, bytes_per_image, header_kind


def _zarr3_codec_chain(codec: str, level: int):
    """Create Zarr v3 codec chain"""
    codecs = [{"name": "bytes", "configuration": {"endian": "little"}}]

    if codec == "zstd":
        codecs.append({"name": "zstd", "configuration": {"level": int(level)}})
    elif codec == "gzip":
        codecs.append({"name": "gzip", "configuration": {"level": int(level)}})
    elif codec == "blosc-lz4":
        codecs.append({"name": "blosc", "configuration": {
            "cname": "lz4",
            "clevel": int(level),
            "shuffle": "shuffle"
        }})
    elif codec == "none":
        pass
    else:
        raise ValueError(f"Unknown codec '{codec}'")

    return codecs


async def _open_ts_array(root_path, name, shape, chunks, np_dtype, codec_chain,
                         attributes=None, create=True, context=None):
    """Open TensorStore array"""
    root_path_abs = str(Path(root_path).resolve())

    spec = {
        "driver": "zarr3",
        "kvstore": {"driver": "file", "path": root_path_abs},
        "metadata": {
            "shape": list(shape),
            "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": list(chunks)}},
            "data_type": np.dtype(np_dtype).name,
            "codecs": codec_chain,
            "attributes": attributes or {},
            "dimension_names": attributes['_ARRAY_DIMENSIONS']
        },
        "path": name,
        "create": create,
        "open": True,
    }

    if context:
        return await ts.open(spec, context=context)
    else:
        return await ts.open(spec)


def discover_frame_structure(pff_path: str, W: int, bpp: int):
    """Discover the fixed frame structure from the first frame"""
    with open(pff_path, 'rb') as f:
        header_with_sep = b''
        while True:
            chunk = f.read(1)
            if not chunk:
                raise ValueError("Unexpected EOF while reading first header")
            header_with_sep += chunk
            if header_with_sep.endswith(b'\n\n'):
                break

        header_size = len(header_with_sep)
        img_data_size = 1 + W * W * bpp
        img_data = f.read(img_data_size)

        if len(img_data) != img_data_size:
            raise ValueError(f"Expected {img_data_size} bytes of image data, got {len(img_data)}")

        frame_size = header_size + img_data_size
        return header_size, img_data_size, frame_size


def parse_frames_from_buffer(buffer: bytes, header_size: int, img_data_size: int,
                             frame_size: int, W: int, np_img_dtype, header_kind: str,
                             max_frames: int = None):
    """Parse multiple frames from a memory buffer"""
    H = W
    imgs = []
    timestamps = []
    headers = []
    pos = 0
    frame_count = 0

    while pos + frame_size <= len(buffer):
        if max_frames and frame_count >= max_frames:
            break

        header_bytes = buffer[pos:pos + header_size]
        try:
            header = json.loads(header_bytes[:-2].decode('utf-8'))
        except (json.JSONDecodeError, UnicodeDecodeError):
            pos += frame_size
            continue

        img_start = pos + header_size + 1
        img_end = img_start + (W * W * np_img_dtype().itemsize)
        img_bytes = buffer[img_start:img_end]

        if len(img_bytes) != W * W * np_img_dtype().itemsize:
            break

        img = np.frombuffer(img_bytes, dtype=np_img_dtype).reshape(H, W)
        imgs.append(img)

        # Extract timestamp
        if header_kind == "ph256_one_level":
            if "pkt_tai" in header and "pkt_nsec" in header:
                ts = float(header["pkt_tai"]) + float(header["pkt_nsec"])*1e-9
            elif "tv_sec" in header and "tv_usec" in header:
                ts = float(header["tv_sec"]) + float(header["tv_usec"])*1e-6
            else:
                ts = np.nan
        else:
            if "tv_sec" in header and "tv_usec" in header:
                ts = float(header["tv_sec"]) + float(header["tv_usec"])*1e-6
            else:
                q0 = header.get("quabo_0", {})
                if "pkt_tai" in q0 and "pkt_nsec" in q0:
                    ts = float(q0["pkt_tai"]) + float(q0["pkt_nsec"])*1e-9
                else:
                    ts = np.nan

        timestamps.append(ts)
        headers.append(header)
        pos += frame_size
        frame_count += 1

    if imgs:
        imgs = np.stack(imgs, axis=0)
        timestamps = np.array(timestamps, dtype=np.float64)
    else:
        imgs = np.empty((0, H, W), dtype=np_img_dtype)
        timestamps = np.empty(0, dtype=np.float64)

    return imgs, timestamps, headers


def read_sequential_chunk_worker(args):
    """Worker function for reading sequential chunks (HDD-optimized)"""
    pff_path, byte_start, byte_end, header_size, img_data_size, frame_size, \
        W, np_img_dtype, header_kind, start_frame_hint = args

    aligned_start = (byte_start // frame_size) * frame_size
    aligned_end = ((byte_end + frame_size - 1) // frame_size) * frame_size
    chunk_size = aligned_end - aligned_start

    with open(pff_path, 'rb') as f:
        f.seek(aligned_start)
        buffer = f.read(chunk_size)

    imgs, timestamps, headers = parse_frames_from_buffer(
        buffer, header_size, img_data_size, frame_size,
        W, np_img_dtype, header_kind
    )

    return imgs, timestamps, headers, start_frame_hint


def group_pff_files_by_stream(obs_dir: str) -> Dict[Tuple[str, str], List[Tuple[str, int]]]:
    """
    Group PFF files by (data_product, module) into streams.

    Returns:
        Dictionary mapping (dp, module) -> [(filepath, seqno), ...]
        Each list is sorted by seqno.
    """
    streams = defaultdict(list)

    obs_dir_path = Path(obs_dir)
    if not obs_dir_path.exists():
        raise ValueError(f"Observation directory not found: {obs_dir}")

    for pff_file in obs_dir_path.glob("*.pff"):
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

    return dict(streams)


async def connect_to_dask(scheduler_address: str):
    """Connect to existing Dask scheduler"""
    if not scheduler_address:
        print("No Dask scheduler address provided - using local multiprocessing")
        return None

    try:
        print(f"Connecting to Dask scheduler: {scheduler_address}")
        client = await Client(scheduler_address, asynchronous=True)
        worker_info = client.scheduler_info()
        print(f"  ✓ Connected! Workers: {len(worker_info['workers'])}")
        return client
    except Exception as e:
        print(f"Warning: Could not connect to Dask scheduler: {e}")
        print("Falling back to local multiprocessing")
        return None


async def convert_pff_stream_to_zarr(
    pff_files: List[Tuple[str, int]],  # [(filepath, seqno), ...]
    zarr_root: str,
    dp: str,
    module: str,
    config: dict,
    client = None
):
    """
    Convert multiple PFF files (same dp/module, different seqno) into single Zarr file.
    Files are concatenated along the time dimension.
    """
    zarr_root = str(Path(zarr_root).resolve())

    # Remove existing zarr if present
    if os.path.exists(zarr_root):
        shutil.rmtree(zarr_root)

    root_group = zarr.open_group(zarr_root, mode='w')

    print(f"\n{'='*80}")
    print(f"Processing stream: dp={dp}, module={module}")
    print(f"Input files: {len(pff_files)}")
    for fpath, seqno in pff_files:
        print(f"  - {os.path.basename(fpath)} (seqno={seqno})")
    print(f"Output: {zarr_root}")
    print(f"{'='*80}\n")

    # Parse metadata from first file
    first_file = pff_files[0][0]
    name_parts, dp_type, bpp, img_hw, np_img_dtype, bytes_per_image, header_kind = _infer_dp_from_name(first_file)
    H, W = img_hw

    # Discover frame structure
    print(f"Discovering frame structure from first file...")
    header_size, img_data_size, frame_size = discover_frame_structure(first_file, W, bpp)
    print(f"  Header size: {header_size} bytes")
    print(f"  Image data: {img_data_size} bytes")
    print(f"  Frame size: {frame_size} bytes")

    # Count total frames across all files
    total_frames = 0
    file_frame_counts = []
    file_sizes = []

    for pff_path, _ in pff_files:
        with open(pff_path, "rb") as f:
            i0, nframes, t0, t1 = pff.img_info(f, bytes_per_image)
        total_frames += nframes
        file_frame_counts.append(nframes)
        file_sizes.append(os.path.getsize(pff_path))
        print(f"  {os.path.basename(pff_path)}: {nframes:,} frames")

    print(f"\nTotal frames across all files: {total_frames:,}")

    # Configuration
    codec = config.get('codec', 'blosc-lz4')
    level = config.get('level', 5)
    time_chunk = config.get('time_chunk', 65536)
    max_concurrent_writes = config.get('max_concurrent_writes', 12)

    print(f"\nZarr configuration:")
    print(f"  Codec: {codec}, Level: {level}")
    print(f"  Time chunk: {time_chunk}")
    print(f"  Max concurrent writes: {max_concurrent_writes}")

    codecs = _zarr3_codec_chain(codec, level)

    # Create TensorStore arrays with total frame count
    T = total_frames
    img_shape = (T, H, W)
    img_chunks = (time_chunk, H, W)
    ts_shape_only = (T,)
    ts_chunks_scalar = (max(1024, time_chunk*2),)

    num_cores = os.cpu_count() or 4
    context = ts.Context({
        'file_io_concurrency': {'limit': max(num_cores, max_concurrent_writes * 2)},
        'cache_pool': {'total_bytes_limit': 1_000_000_000}
    })

    print(f"\nCreating Zarr arrays...")
    images = await _open_ts_array(
        zarr_root, "images",
        img_shape, img_chunks, np_img_dtype, codecs,
        attributes={
            "_ARRAY_DIMENSIONS": ["time", "y", "x"],
            "data_product": dp,
            "module": module,
            "source_pff_files": [os.path.basename(f) for f, _ in pff_files],
            "pff_metadata": name_parts,
        }, create=True, context=context)

    timestamps = await _open_ts_array(
        zarr_root, "timestamps",
        ts_shape_only, ts_chunks_scalar, np.float64, codecs,
        attributes={"_ARRAY_DIMENSIONS": ["time"]}, create=True, context=context)

    # Create header arrays
    header_arrays = {}
    root_group.create_group('headers')

    if header_kind == "ph256_one_level":
        header_fields = {
            "quabo_num": np.uint8,
            "pkt_num": np.int64,
            "pkt_tai": np.int64,
            "pkt_nsec": np.int64,
            "tv_sec": np.int64,
            "tv_usec": np.int64,
        }
        for k, dt in header_fields.items():
            header_arrays[k] = await _open_ts_array(
                zarr_root, f"headers/{k}", ts_shape_only, ts_chunks_scalar, dt, codecs,
                attributes={"_ARRAY_DIMENSIONS": ["time"]}, create=True, context=context)
    else:
        per_quabo_fields = {
            "pkt_num": np.int64, "pkt_tai": np.int64, "pkt_nsec": np.int64,
            "tv_sec": np.int64, "tv_usec": np.int64,
        }
        for qi in range(4):
            root_group.create_group(f'headers/quabo_{qi}')
            for k, dt in per_quabo_fields.items():
                name = f"headers/quabo_{qi}/{k}"
                header_arrays[name] = await _open_ts_array(
                    zarr_root, name, ts_shape_only, ts_chunks_scalar, dt, codecs,
                    attributes={"_ARRAY_DIMENSIONS": ["time"]}, create=True, context=context)

    # Processing
    start = time.monotonic()
    pbar = tqdm(total=T, desc="Converting PFF -> Zarr", unit="frames",
                smoothing=0.1, dynamic_ncols=True)

    frames_written = 0
    pending_writes = deque()
    semaphore = asyncio.Semaphore(max_concurrent_writes)

    async def flush_batch(img_batch, ts_batch, header_list, n, start_frame):
        """Flush batch with concurrency control"""
        nonlocal frames_written
        if n == 0:
            return

        async with semaphore:
            s = slice(start_frame, start_frame + n)
            write_futures = []

            fut = images[s, :, :].write(img_batch[:n])
            if fut: write_futures.append(fut)

            fut = timestamps[s].write(ts_batch[:n])
            if fut: write_futures.append(fut)

            # Write headers
            if header_kind == "ph256_one_level":
                for k, arr in header_arrays.items():
                    data = np.array([h.get(k, 0) for h in header_list[:n]], dtype=arr.dtype.numpy_dtype)
                    fut = arr[s].write(data)
                    if fut: write_futures.append(fut)
            else:
                for qi in range(4):
                    for fld in ("pkt_num", "pkt_tai", "pkt_nsec", "tv_sec", "tv_usec"):
                        name = f"headers/quabo_{qi}/{fld}"
                        if name in header_arrays:
                            data = np.array([h.get(f"quabo_{qi}", {}).get(fld, h.get(fld, 0))
                                           for h in header_list[:n]], dtype=np.int64)
                            fut = header_arrays[name][s].write(data)
                            if fut: write_futures.append(fut)

            if write_futures:
                await asyncio.gather(*write_futures)

            pbar.update(n)
            frames_written += n

    use_dask = client is not None
    global_frame_offset = 0

    # Process each file in sequence (by seqno)
    for file_idx, (pff_path, seqno) in enumerate(pff_files):
        file_size = file_sizes[file_idx]
        nframes_in_file = file_frame_counts[file_idx]

        print(f"\nProcessing file {file_idx+1}/{len(pff_files)}: {os.path.basename(pff_path)}")
        print(f"  Frames: {nframes_in_file:,}")
        print(f"  Global frame offset: {global_frame_offset:,}")

        if use_dask:
            # DASK DISTRIBUTED APPROACH
            worker_info = client.scheduler_info()
            num_workers_available = len(worker_info['workers'])

            work_chunks = []
            for worker_id in range(num_workers_available):
                byte_start = (file_size * worker_id) // num_workers_available
                byte_end = (file_size * (worker_id + 1)) // num_workers_available if worker_id < num_workers_available - 1 else file_size
                start_frame_hint = (byte_start // frame_size)

                work_chunks.append((
                    os.path.abspath(pff_path), byte_start, byte_end, header_size, img_data_size, frame_size,
                    W, np_img_dtype, header_kind, start_frame_hint
                ))

            # Submit work to cluster
            futures = client.map(read_sequential_chunk_worker, work_chunks)
            gathered = await client.gather(futures)

            results = []
            for result in gathered:
                imgs, ts_batch, header_list, start_frame = result
                n = len(imgs)
                if n > 0:
                    results.append((imgs, ts_batch, header_list, start_frame))

            # Sort by local frame position within this file
            results.sort(key=lambda x: x[3])

            # Write with global offset
            for imgs, ts_batch, header_list, local_start_frame in results:
                n = len(imgs)
                global_start_frame = global_frame_offset + local_start_frame
                task = asyncio.create_task(flush_batch(imgs, ts_batch, header_list, n, global_start_frame))
                pending_writes.append(task)

                while len(pending_writes) >= max_concurrent_writes:
                    await pending_writes.popleft()

        else:
            # LOCAL MULTIPROCESSING APPROACH
            num_workers = config.get('num_workers', min(cpu_count(), 8))

            work_chunks = []
            for worker_id in range(num_workers):
                byte_start = (file_size * worker_id) // num_workers
                byte_end = (file_size * (worker_id + 1)) // num_workers if worker_id < num_workers - 1 else file_size
                start_frame_hint = (byte_start // frame_size)

                work_chunks.append((
                    pff_path, byte_start, byte_end, header_size, img_data_size, frame_size,
                    W, np_img_dtype, header_kind, start_frame_hint
                ))

            with ProcessPoolExecutor(max_workers=num_workers) as executor:
                futures = [executor.submit(read_sequential_chunk_worker, chunk) for chunk in work_chunks]

                results = []
                for future in futures:
                    imgs, ts_batch, header_list, start_frame = future.result()
                    n = len(imgs)
                    if n > 0:
                        results.append((imgs, ts_batch, header_list, start_frame))

                # Sort by local frame position within this file
                results.sort(key=lambda x: x[3])

                # Write with global offset
                for imgs, ts_batch, header_list, local_start_frame in results:
                    n = len(imgs)
                    global_start_frame = global_frame_offset + local_start_frame
                    task = asyncio.create_task(flush_batch(imgs, ts_batch, header_list, n, global_start_frame))
                    pending_writes.append(task)

                    while len(pending_writes) >= max_concurrent_writes:
                        await pending_writes.popleft()

        # Update global offset for next file
        global_frame_offset += nframes_in_file

    # Wait for all pending writes
    while pending_writes:
        await pending_writes.popleft()

    pbar.close()
    end = time.monotonic()
    elapsed_s = end - start

    # Calculate statistics
    def _dir_size(path):
        total = 0
        for dirpath, _, filenames in os.walk(path):
            for fn in filenames:
                fp = os.path.join(dirpath, fn)
                if os.path.exists(fp):
                    total += os.path.getsize(fp)
        return total

    total_pff_size = sum(file_sizes)
    zarr_size = _dir_size(zarr_root)
    compression_ratio = (total_pff_size / zarr_size) if zarr_size > 0 else float("inf")
    throughput_mbps = (total_pff_size / (1024**2)) / elapsed_s if elapsed_s > 0 else 0

    report = {
        "data_product": dp,
        "module": module,
        "source_pff_files": [os.path.basename(f) for f, _ in pff_files],
        "num_files": len(pff_files),
        "frames": T,
        "frames_written": frames_written,
        "elapsed_seconds": round(elapsed_s, 2),
        "throughput_MB_per_sec": round(throughput_mbps, 2),
        "total_pff_size_MB": round(total_pff_size / (1024**2), 2),
        "zarr_size_MB": round(zarr_size / (1024**2), 2),
        "compression_ratio": round(compression_ratio, 2),
        "optimization": "DASK_DISTRIBUTED" if use_dask else "LOCAL_MULTIPROCESSING",
    }

    print("\n" + "="*80)
    print("Conversion Report:")
    print("="*80)
    print(json.dumps(report, indent=2))
    print("="*80)

    return report


def load_config(config_path: str = None):
    """Load configuration from TOML file or environment variables"""
    config = {
        'codec': 'blosc-lz4',
        'level': 1,
        'time_chunk': 32768,
        'max_concurrent_writes': 12,
        'num_workers': None,
        'chunk_size_mb': 50,
        'blosc_threads': None,
    }

    if config_path and os.path.exists(config_path):
        if tomllib is None:
            print(f"Warning: Cannot read {config_path} - tomli/tomllib not installed")
        else:
            with open(config_path, 'rb') as f:
                toml_config = tomllib.load(f)
                if 'pff_to_zarr' in toml_config:
                    config.update(toml_config['pff_to_zarr'])
            print(f"Loaded configuration from {config_path}")

    # Override with environment variables
    if os.environ.get('TS_CODEC'):
        config['codec'] = os.environ['TS_CODEC']
    if os.environ.get('TS_LEVEL'):
        config['level'] = int(os.environ['TS_LEVEL'])
    if os.environ.get('TS_CHUNK'):
        config['time_chunk'] = int(os.environ['TS_CHUNK'])
    if os.environ.get('TS_CONCURRENT'):
        config['max_concurrent_writes'] = int(os.environ['TS_CONCURRENT'])
    if os.environ.get('BLOSC_NTHREADS'):
        config['blosc_threads'] = int(os.environ['BLOSC_NTHREADS'])

    return config


async def main():
    original_umask = os.umask(0o000)
    if len(sys.argv) < 3:
        print("Usage: python step1_pff_to_zarr.py <obs_dir> <output_zarr_dir> [config.toml] [scheduler_address]")
        print()
        print("Arguments:")
        print("  obs_dir           - Input observation directory containing PFF files")
        print("  output_zarr_dir   - Output directory for Zarr files")
        print("  config.toml       - Configuration file (optional)")
        print("  scheduler_address - Dask scheduler address (optional, e.g. tcp://10.0.1.2:8786)")
        print()
        print("Example:")
        print("  python step1_pff_to_zarr.py \\")
        print("    /mnt/beegfs/data/L0/obs_Lick.start_2024-07-25T04:34:06Z.runtype_sci-data.pffd \\")
        print("    /mnt/beegfs/zarr/L0 \\")
        print("    config.toml \\")
        print("    tcp://10.0.1.2:8786")
        sys.exit(1)

    obs_dir = sys.argv[1]
    output_zarr_dir = sys.argv[2]
    config_path = sys.argv[3] if len(sys.argv) > 3 else "config.toml"
    scheduler_address = sys.argv[4] if len(sys.argv) > 4 else ""

    config = load_config(config_path)

    print("Configuration:")
    for key, value in config.items():
        if value is not None:
            print(f"  {key}: {value}")
    print()

    # Group PFF files by stream (dp, module)
    print(f"Scanning observation directory: {obs_dir}")
    streams = group_pff_files_by_stream(obs_dir)

    if not streams:
        print("ERROR: No valid PFF files found in observation directory")
        sys.exit(1)

    print(f"\nFound {len(streams)} data streams:")
    for (dp, module), files in streams.items():
        print(f"  dp={dp}, module={module}: {len(files)} file(s)")
    print()

    # Create output directory
    os.makedirs(output_zarr_dir, exist_ok=True)

    # Connect to Dask cluster if address provided
    client = None
    if scheduler_address:
        client = await connect_to_dask(scheduler_address)

    try:
        # Process each stream
        all_reports = []
        for stream_idx, ((dp, module), pff_files) in enumerate(streams.items(), 1):
            print(f"\n{'#'*80}")
            print(f"# Processing stream {stream_idx}/{len(streams)}")
            print(f"{'#'*80}")

            # Generate output zarr path
            obs_basename = os.path.basename(obs_dir.rstrip('/'))
            zarr_name = f"{obs_basename}.dp_{dp}.module_{module}.zarr"
            zarr_root = os.path.join(output_zarr_dir, zarr_name)

            report = await convert_pff_stream_to_zarr(
                pff_files, zarr_root, dp, module, config, client
            )
            all_reports.append(report)

        # Summary
        print(f"\n{'='*80}")
        print("OVERALL SUMMARY")
        print(f"{'='*80}")
        print(f"Total streams processed: {len(all_reports)}")
        total_frames = sum(r['frames'] for r in all_reports)
        total_time = sum(r['elapsed_seconds'] for r in all_reports)
        print(f"Total frames: {total_frames:,}")
        print(f"Total time: {total_time:.2f}s")
        print(f"Overall throughput: {total_frames/total_time:.1f} frames/sec")
        print(f"{'='*80}")

    finally:
        # Close client connection (but don't shutdown cluster)
        if client:
            print("\n✓ Keeping cluster alive for next task...")
            await client.close()
        os.umask(original_umask)


if __name__ == "__main__":
    asyncio.run(main())
