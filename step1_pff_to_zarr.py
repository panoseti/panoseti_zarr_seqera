#!/usr/bin/env python3

"""

Step 1: Convert L0 PFF-formatted data to L0 Zarr-formatted data using TensorStore

PARALLEL PFF READING - HDD/BeeGFS OPTIMIZED VERSION:

Key optimization for HDDs and network filesystems (BeeGFS):
- Each worker reads ONE LARGE SEQUENTIAL CHUNK (10-100 MB)
- Avoids many small seeks (extremely slow on HDDs)
- Optimal for BeeGFS stripe performance
- Workers read different parts of file independently

Strategy:
1. Discover frame size from first frame
2. Split file into N large byte-range chunks (not frame-by-frame)
3. Each worker reads its entire chunk sequentially (one large read)
4. Parse frames from in-memory buffer
5. Write results asynchronously

For HDDs: Sequential reads are 100-200x faster than random seeks
For BeeGFS: Large sequential I/O utilizes stripe parallelism

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
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import cpu_count
import pff

# Configure Blosc for multi-threading
try:
    import blosc
    num_cores = os.cpu_count() or 4
    blosc.set_nthreads(num_cores)
    print(f"Blosc configured to use {num_cores} threads for compression")
except ImportError:
    print("Warning: python-blosc not installed, using tensorstore's blosc")

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
    """
    Discover the fixed frame structure from the first frame.
    """
    with open(pff_path, 'rb') as f:
        # Read first header
        header_with_sep = b''
        while True:
            chunk = f.read(1)
            if not chunk:
                raise ValueError("Unexpected EOF while reading first header")
            header_with_sep += chunk
            if header_with_sep.endswith(b'\n\n'):
                break

        header_size = len(header_with_sep)
        img_data_size = 1 + W * W * bpp  # 1 byte for '*' prefix + image data

        # Verify by reading image data
        img_data = f.read(img_data_size)
        if len(img_data) != img_data_size:
            raise ValueError(f"Expected {img_data_size} bytes of image data, got {len(img_data)}")

        frame_size = header_size + img_data_size

    return header_size, img_data_size, frame_size

def parse_frames_from_buffer(buffer: bytes, header_size: int, img_data_size: int,
                            frame_size: int, W: int, np_img_dtype, header_kind: str,
                            max_frames: int = None):
    """
    Parse multiple frames from a memory buffer.

    This is the key function that enables sequential reading:
    - Read one large chunk into memory
    - Parse all frames from the buffer (no disk I/O)
    - Much faster than seek + read per frame
    """
    H = W
    imgs = []
    timestamps = []
    headers = []

    pos = 0
    frame_count = 0

    while pos + frame_size <= len(buffer):
        if max_frames and frame_count >= max_frames:
            break

        # Parse header
        header_bytes = buffer[pos:pos + header_size]
        try:
            header = json.loads(header_bytes[:-2].decode('utf-8'))
        except (json.JSONDecodeError, UnicodeDecodeError):
            # Skip invalid frame
            pos += frame_size
            continue

        # Parse image (skip '*' prefix)
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

    # Convert to numpy arrays
    if imgs:
        imgs = np.stack(imgs, axis=0)
        timestamps = np.array(timestamps, dtype=np.float64)
    else:
        imgs = np.empty((0, H, W), dtype=np_img_dtype)
        timestamps = np.empty(0, dtype=np.float64)

    return imgs, timestamps, headers

def read_sequential_chunk_worker(args):
    """
    Worker function optimized for HDDs and BeeGFS.

    KEY OPTIMIZATION:
    - Reads ONE LARGE SEQUENTIAL CHUNK (10-100 MB)
    - Single f.read() call - no seeks
    - Parses all frames from memory buffer

    For HDDs:
    - Sequential read: ~100-200 MB/s
    - Random seeks: ~1-5 MB/s (100x slower!)

    For BeeGFS:
    - Large sequential reads utilize stripe parallelism
    - Reduced metadata operations
    """
    pff_path, byte_start, byte_end, header_size, img_data_size, frame_size, \
        W, np_img_dtype, header_kind, start_frame_hint = args

    # Calculate exact frame boundaries
    # Round down to nearest frame boundary
    aligned_start = (byte_start // frame_size) * frame_size
    # Round up to nearest frame boundary (with small overlap for safety)
    aligned_end = ((byte_end + frame_size - 1) // frame_size) * frame_size

    chunk_size = aligned_end - aligned_start

    # SINGLE LARGE SEQUENTIAL READ (KEY OPTIMIZATION)
    with open(pff_path, 'rb') as f:
        f.seek(aligned_start)
        buffer = f.read(chunk_size)

    # Parse all frames from buffer (in memory, very fast)
    imgs, timestamps, headers = parse_frames_from_buffer(
        buffer, header_size, img_data_size, frame_size,
        W, np_img_dtype, header_kind
    )

    return imgs, timestamps, headers, start_frame_hint

async def convert_pff_to_tensorstore(
    pff_path: str,
    zarr_root: str,
    codec: str = "blosc-lz4",
    level: int = 1,
    time_chunk: int = 32768,
    max_concurrent_writes: int = 12,
    num_workers: int = None,
    chunk_size_mb: int = 50,  # Size of sequential chunk per worker
):
    """
    Convert PFF to Zarr using PARALLEL SEQUENTIAL READING.

    Optimized for HDDs and BeeGFS:
    - Each worker reads a large sequential chunk (default 50 MB)
    - No random seeks (critical for HDD performance)
    - Large reads utilize BeeGFS stripe parallelism

    Parameters:
    -----------
    num_workers : int
        Number of parallel workers (default: CPU count)
    chunk_size_mb : int
        Size of sequential chunk per worker in MB (default: 50)
        Larger = better for HDDs (fewer seeks)
        Recommended: 20-100 MB for HDDs, 50-200 MB for BeeGFS
    """
    if not os.path.exists(pff_path):
        raise FileNotFoundError(pff_path)

    zarr_root = str(Path(zarr_root).resolve())

    if os.path.exists(zarr_root):
        shutil.rmtree(zarr_root)

    root_group = zarr.open_group(zarr_root, mode='w')

    name_parts, dp, bpp, img_hw, np_img_dtype, bytes_per_image, header_kind = _infer_dp_from_name(pff_path)

    # Get frame count
    with open(pff_path, "rb") as f:
        i0, nframes, t0, t1 = pff.img_info(f, bytes_per_image)

    H, W = img_hw

    print(f"\nDiscovering frame structure from first frame...")
    header_size, img_data_size, frame_size = discover_frame_structure(pff_path, W, bpp)
    print(f"  Header size:    {header_size} bytes")
    print(f"  Image data:     {img_data_size} bytes (includes '*' prefix)")
    print(f"  Frame size:     {frame_size} bytes (fixed)")
    print(f"  Total frames:   {nframes}")

    file_size = os.path.getsize(pff_path)
    chunk_size_bytes = chunk_size_mb * 1024 * 1024
    frames_per_chunk = chunk_size_bytes // frame_size

    print(f"\nOptimized for HDD/BeeGFS:")
    print(f"  Sequential chunk size: {chunk_size_mb} MB ({frames_per_chunk} frames)")
    print(f"  Each worker: 1 large sequential read (no seeks)")

    print(f"\nCreating Zarr group at: {zarr_root}")
    print(f"Settings: codec={codec}, level={level}, time_chunk={time_chunk}")

    codecs = _zarr3_codec_chain(codec, level)
    T = nframes
    img_shape = (T, H, W)
    img_chunks = (time_chunk, H, W)
    ts_shape_only = (T,)
    ts_chunks_scalar = (max(1024, time_chunk*2),)

    # Create TensorStore context
    num_cores = os.cpu_count() or 4
    context = ts.Context({
        'file_io_concurrency': {'limit': max(num_cores, max_concurrent_writes * 2)},
        'cache_pool': {'total_bytes_limit': 1_000_000_000}
    })

    # Create TensorStore arrays
    images = await _open_ts_array(
        zarr_root, "images",
        img_shape, img_chunks, np_img_dtype, codecs,
        attributes={
            "_ARRAY_DIMENSIONS": ["time", "y", "x"],
            "source_pff_file": os.path.basename(pff_path),
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

    pff_file_size = os.path.getsize(pff_path)
    start = time.monotonic()

    pbar = tqdm(total=T, desc="Converting PFF -> Zarr (HDD OPTIMIZED)", unit="frames")

    written = 0
    pending_writes = deque()
    semaphore = asyncio.Semaphore(max_concurrent_writes)

    async def flush_batch(img_batch, ts_batch, header_list, n, start_frame):
        """Flush batch with concurrency control"""
        nonlocal written
        if n == 0:
            return

        async with semaphore:
            s = slice(start_frame, start_frame + n)
            write_futures = []

            # Write images
            fut = images[s, :, :].write(img_batch[:n])
            if fut: write_futures.append(fut)

            # Write timestamps
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

    # PARALLEL SEQUENTIAL READING
    if num_workers is None:
        num_workers = min(cpu_count(), 8)  # Cap at 8 for HDD to avoid thrashing

    print(f"\nReading with {num_workers} parallel workers...")
    print(f"Each worker reads {chunk_size_mb} MB sequentially (HDD-optimized)")

    # Create work chunks based on byte ranges (not frame counts)
    work_chunks = []
    for worker_id in range(num_workers):
        byte_start = (file_size * worker_id) // num_workers
        byte_end = (file_size * (worker_id + 1)) // num_workers if worker_id < num_workers - 1 else file_size

        # Estimate starting frame for this chunk
        start_frame_hint = (byte_start // frame_size)

        work_chunks.append((
            pff_path, byte_start, byte_end, header_size, img_data_size, frame_size,
            W, np_img_dtype, header_kind, start_frame_hint
        ))

    print(f"Created {len(work_chunks)} work chunks (avg {file_size / num_workers / (1024**2):.1f} MB each)")

    # Process chunks in parallel
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        # Submit all work
        futures = [executor.submit(read_sequential_chunk_worker, chunk) for chunk in work_chunks]

        # Collect results as they complete
        results = []
        for future in futures:
            imgs, ts_batch, header_list, start_frame = future.result()
            n = len(imgs)
            if n > 0:
                results.append((imgs, ts_batch, header_list, start_frame))

        # Sort by start_frame to maintain order
        results.sort(key=lambda x: x[3])

        # Write in order
        for imgs, ts_batch, header_list, start_frame in results:
            n = len(imgs)
            task = asyncio.create_task(flush_batch(imgs, ts_batch, header_list, n, start_frame))
            pending_writes.append(task)

            # Limit concurrent writes
            while len(pending_writes) >= max_concurrent_writes:
                await pending_writes.popleft()

    # Wait for all pending writes
    while pending_writes:
        await pending_writes.popleft()

    pbar.close()
    end = time.monotonic()
    elapsed_s = end - start

    # Compute size
    def _dir_size(path):
        total = 0
        for dirpath, _, filenames in os.walk(path):
            for fn in filenames:
                fp = os.path.join(dirpath, fn)
                if os.path.exists(fp):
                    total += os.path.getsize(fp)
        return total

    zarr_size = _dir_size(zarr_root)
    compression_ratio = (pff_file_size / zarr_size) if zarr_size > 0 else float("inf")
    throughput_mbps = (pff_file_size / (1024**2)) / elapsed_s if elapsed_s > 0 else 0

    report = {
        "source_pff": os.path.basename(pff_path),
        "frames": T,
        "elapsed_seconds": round(elapsed_s, 2),
        "throughput_MB_per_sec": round(throughput_mbps, 2),
        "pff_size_MB": round(pff_file_size / (1024**2), 2),
        "zarr_size_MB": round(zarr_size / (1024**2), 2),
        "compression_ratio": round(compression_ratio, 2),
        "optimization": "HDD_SEQUENTIAL_PARALLEL",
        "num_workers": num_workers,
        "chunk_size_mb": chunk_size_mb,
    }
    print("\n" + "="*60)
    print("Conversion Report:")
    print("="*60)
    print(json.dumps(report, indent=2))
    print("="*60)
    return report

async def main():
    if len(sys.argv) not in [3, 4, 5]:
        print("Usage: python step1_pff_to_zarr.py <input.pff> <output.zarr> [num_workers] [chunk_size_mb]")
        print()
        print("Arguments:")
        print("  num_workers: Number of parallel workers (default: CPU count, max 8)")
        print("  chunk_size_mb: MB per sequential read (default: 50)")
        print()
        print("Recommended for HDDs: num_workers=4-8, chunk_size_mb=50-100")
        print("Recommended for BeeGFS: num_workers=8-16, chunk_size_mb=100-200")
        print()
        print("Environment variables:")
        print("  TS_CODEC=blosc-lz4|zstd|gzip|none (default: blosc-lz4)")
        print("  TS_LEVEL=1-9 (default: 1)")
        print("  TS_CHUNK=32768 (default: 32768)")
        print("  TS_CONCURRENT=12 (default: 12)")
        print()
        print("HDD/BeeGFS OPTIMIZATION:")
        print("  - Each worker reads ONE large sequential chunk")
        print("  - No random seeks (critical for HDD performance)")
        print("  - Large I/O utilizes BeeGFS stripes")
        sys.exit(1)

    pff_path = sys.argv[1]
    zarr_root = sys.argv[2]
    num_workers = int(sys.argv[3]) if len(sys.argv) >= 4 else None
    chunk_size_mb = int(sys.argv[4]) if len(sys.argv) == 5 else 50

    codec = os.environ.get("TS_CODEC", "blosc-lz4")
    level = int(os.environ.get("TS_LEVEL", "1"))
    time_chunk = int(os.environ.get("TS_CHUNK", "32768"))
    max_concurrent = int(os.environ.get("TS_CONCURRENT", "12"))

    await convert_pff_to_tensorstore(
        pff_path,
        zarr_root,
        codec=codec,
        level=level,
        time_chunk=time_chunk,
        max_concurrent_writes=max_concurrent,
        num_workers=num_workers,
        chunk_size_mb=chunk_size_mb
    )

if __name__ == "__main__":
    asyncio.run(main())

