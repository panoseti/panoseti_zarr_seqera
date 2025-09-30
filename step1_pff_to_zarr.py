#!/usr/bin/env python3
"""
Step 1: Convert L0 PFF-formatted data to L0 Zarr-formatted data using TensorStore
"""

import os
import sys
import json
import math
import asyncio
import shutil
import zarr
import numpy as np
import tensorstore as ts
from tqdm import tqdm
import time
from pathlib import Path

import pff

def _infer_dp_from_name(pff_path: str):
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
    allowed_codecs = ["zstd", "gzip", "blosc-lz4"]
    codecs = [{"name": "bytes", "configuration": {"endian": "little"}}]
    if codec == "zstd":
        codecs.append({"name": "zstd", "configuration": {"level": int(level)}})

    if codec == "gzip":
        codecs.append({"name": "gzip", "configuration": {"level": int(level)}})

    if codec == "blosc-lz4":
        codecs.append({"name": "blosc", "configuration": {"cname": "lz4", "clevel": int(level), "shuffle": "shuffle"}})

    if codec not in allowed_codecs:
        raise ValueError("Unknown codec; use 'zstd', 'gzip', or 'blosc-lz4'.")
    else:
        return codecs

async def _open_ts_array(root_path, name, shape, chunks, np_dtype, codec_chain, attributes=None, create=True, delete_existing=True):
    # Convert to absolute path to avoid TensorStore path issues
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
    return await ts.open(spec)

async def convert_pff_to_tensorstore(
    pff_path: str,
    zarr_root: str,
    codec: str = "zstd",
    level: int = 3,
    time_chunk: int = 8192,
):
    if not os.path.exists(pff_path):
        raise FileNotFoundError(pff_path)

    # Convert zarr_root to absolute path
    zarr_root = str(Path(zarr_root).resolve())

    # ensure target directory is clean
    if os.path.exists(zarr_root):
        shutil.rmtree(zarr_root)

    # create the zarr group using the zarr library.
    root_group = zarr.open_group(zarr_root, mode='w')
    print(f"Created Zarr group at: {zarr_root}")

    name_parts, dp, bpp, img_hw, np_img_dtype, bytes_per_image, header_kind = _infer_dp_from_name(pff_path)

    with open(pff_path, "rb") as f:
        i0, nframes, t0, t1 = pff.img_info(f, bytes_per_image)

    codecs = _zarr3_codec_chain(codec, level)

    T = nframes
    H, W = img_hw
    img_shape = (T, H, W)
    img_chunks = (time_chunk, H, W)
    ts_shape_only = (T,)
    ts_chunks_scalar = (max(1024, time_chunk*2),)

    # make arrays within the group using tensorstore.
    images = await _open_ts_array(
        zarr_root, "images",
        img_shape, img_chunks, np_img_dtype, codecs,
        attributes={
            "_ARRAY_DIMENSIONS": ["time", "y", "x"],
            "source_pff_file": os.path.basename(pff_path),
            "pff_metadata": name_parts,
        }, create=True)

    timestamps = await _open_ts_array(
        zarr_root, "timestamps",
        ts_shape_only, (max(1024, time_chunk*2),), np.float64, codecs,
        attributes={"_ARRAY_DIMENSIONS": ["time"]}, create=True)

    # header schema arrays
    header_arrays = {}
    root_group.create_group('headers')
    if header_kind == "ph256_one_level":

        # one-level fields
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
                attributes={"_ARRAY_DIMENSIONS": ["time"]}, create=True)
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
                    attributes={"_ARRAY_DIMENSIONS": ["time"]}, create=True)

    # batched write buffers
    batch_T = time_chunk
    img_batch = np.empty((batch_T, H, W), dtype=np_img_dtype)
    ts_batch = np.empty((batch_T,), dtype=np.float64)
    header_batches = {k: np.empty((batch_T,), dtype=arr.dtype.numpy_dtype) for k, arr in header_arrays.items()}

    pff_file_size = os.path.getsize(pff_path)

    start = time.monotonic()

    # progress bar setup
    pbar = tqdm(total=T, desc="Converting PFF -> Zarr (TensorStore)", unit="frames")

    # streaming parse/write
    written = 0
    futures = []

    def flush_batch(n):
        nonlocal written, futures
        if n == 0:
            return
        s = slice(written, written + n)
        f = images[s, :, :].write(img_batch[:n])
        if f is not None:
            futures.append(f)
        f = timestamps[s].write(ts_batch[:n])
        if f is not None:
            futures.append(f)
        for k, arr in header_arrays.items():
            f = arr[s].write(header_batches[k][:n])
            if f is not None:
                futures.append(f)
        written += n
        pbar.update(n)

    with open(pff_path, "rb") as f:
        bi = 0
        for i in range(T):
            hdr_str = pff.read_json(f)
            if hdr_str is None:
                break
            header = json.loads(hdr_str)

            # read image
            flat = pff.read_image(f, W, bpp)
            if flat is None:
                continue
            img_batch[bi, :, :] = np.asarray(flat).reshape(H, W).astype(np_img_dtype)

            # timestamp
            if header_kind == "ph256_one_level":
                if "pkt_tai" in header and "pkt_nsec" in header:
                    ts_batch[bi] = float(header["pkt_tai"]) + float(header["pkt_nsec"])*1e-9
                elif "tv_sec" in header and "tv_usec" in header:
                    ts_batch[bi] = float(header["tv_sec"]) + float(header["tv_usec"])*1e-6
                else:
                    ts_batch[bi] = np.nan
                for fld in ("pkt_num", "pkt_tai", "pkt_nsec", "tv_sec", "tv_usec", "quabo_num"):
                    if fld in header_arrays:
                        v = header.get(fld, 0)
                        header_batches[fld][bi] = header_fields[fld](v)
            else:  # two-level (32x32)
                if "tv_sec" in header and "tv_usec" in header:
                    ts_batch[bi] = float(header["tv_sec"]) + float(header["tv_usec"])*1e-6
                else:
                    q0 = header.get("quabo_0", {})
                    if "pkt_tai" in q0 and "pkt_nsec" in q0:
                        ts_batch[bi] = float(q0["pkt_tai"]) + float(q0["pkt_nsec"])*1e-9
                    else:
                        ts_batch[bi] = np.nan

                for qi in range(4):
                    q = header.get(f"quabo_{qi}", {})
                    for fld in ("pkt_num", "pkt_tai", "pkt_nsec", "tv_sec", "tv_usec", "quabo_num"):
                        name = f"headers/quabo_{qi}/{fld}"
                        if name in header_arrays:
                            header_batches[name][bi] = np.int64(q.get(fld, header.get(fld, 0)))

            bi += 1
            if bi == batch_T:
                flush_batch(bi)
                bi = 0

        if bi > 0:
            flush_batch(bi)

    # flush all writes
    for fut in futures:
        await fut

    pbar.close()

    end = time.monotonic()
    elapsed_s = end - start

    # compute Zarr store size
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

    # print compression report
    report = {
        "source_pff": os.path.basename(pff_path),
        "frames": T,
        "elapsed_seconds": elapsed_s,
        "pff_size_bytes": pff_file_size,
        "zarr_size_bytes": zarr_size,
        "compression_ratio_pff_over_zarr": compression_ratio,
        "images": {
            "shape": (T, H, W),
            "dtype": str(np.dtype(np_img_dtype)),
            "chunks": (time_chunk, H, W),
        },
        "timestamps": {
            "shape": (T,),
            "dtype": "float64",
            "chunks": (max(1024, time_chunk*2),),
        },
        "header_arrays": sorted(list(header_arrays.keys())),
        "codec": codec,
        "level": level,
    }

    print("\nConversion Report:")
    print(json.dumps(report, indent=2))

    return report

async def main():
    if len(sys.argv) != 3:
        print("Usage: python step1_pff_to_zarr.py <input_pff_file> <output_zarr_directory>")
        sys.exit(1)

    pff_path = sys.argv[1]
    zarr_root = sys.argv[2]

    await convert_pff_to_tensorstore(
        pff_path,
        zarr_root,
        time_chunk=8192,
        codec="zstd",
        level=3
    )

if __name__ == "__main__":
    asyncio.run(main())
