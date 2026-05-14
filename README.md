# PANOSETI Zarr Nextflow Pipeline

Converts PanoSETI PFF observation data to calibrated Zarr v3 stores using
**Nextflow 26.04 strict DSL2**. Runs identically on a laptop (local executor)
and on SDSC Expanse (SLURM + Singularity).

## Pipeline stages

```
.pffd  ──►  PFF_TO_ZARR  ──►  L0/ (one .zarr per product × module)
                                    │
                       ┌────────────┴─────────────┐
                       ▼                          ▼
               CALIBRATE_PH                CALIBRATE_IMG
           (ph256, ph1024)              (img8, img16)
           pedestal subtraction         block-median subtraction
           n-σ thresholding             + temporal supermedian
                       │                          │
                       └────────────┬─────────────┘
                                    ▼
                             L1/ (calibrated .zarr + summary.json + preview.png)
```

**Key design choices (v0.3.0):**
- Per-product parallelism is Nextflow-native — no Dask cluster required.
- Output publishing uses the Nextflow 26 `output {}` block (no `publishDir`).
- Calibration runs synchronously in a single Python process per store; xarray + zstd
  writes ~GB/s on a laptop and scales to HPC via SLURM without code changes.
- `ph` and `img` products are calibrated differently:
  - `ph256`/`ph1024` (`int16` ADC intensities): pedestal subtracted, n-σ masked.
  - `img8`/`img16` (`uint8`/`uint16` counts above threshold): spatial block-median
    + temporal supermedian subtracted, then ADC→PE scaled.

---

## Quick start — laptop

```bash
# 1. Install Python dependencies (requires uv)
uv sync

# 2. Run against the bundled test data (~seconds)
nextflow run . -profile laptop

# Results in results/L0/ and results/L1/
ls results/L0/*.zarr results/L1/*.zarr
```

Inspect a calibrated store:
```python
import xarray as xr
ds = xr.open_zarr("results/L1/dp_ph256.bpp_2.module_1_L1.zarr", consolidated=False)
print(ds)                         # shows pedestal_subtracted, unix_t_ns, pkt_num, …
print(ds.attrs["calibration"])    # params used
```

---

## Repository layout

```
panoseti_zarr_seqera/
├── main.nf                      ← entry workflow (strict DSL2)
├── nextflow.config              ← params + profiles + report settings
├── conf/
│   ├── laptop.config            ← local executor, no container
│   └── hpc_slurm.config         ← SLURM + singularity
├── modules/
│   ├── pff_to_zarr.nf
│   ├── calibrate_ph.nf
│   └── calibrate_img.nf
├── subworkflows/
│   └── calibrate.nf             ← routes ph vs img products
├── bin/                         ← executable entry-points (auto on $PATH in processes)
│   ├── pff2zarr
│   ├── calibrate_ph
│   └── calibrate_img
├── src/panoseti_zarr_pipeline/  ← importable Python package
│   ├── _common.py               ← shared: open_l0, write_l1, Stats, preview PNG
│   ├── calibrate_ph.py
│   └── calibrate_img.py
├── tests/
│   ├── test_calibrate_ph.py
│   └── test_calibrate_img.py
├── hpc/
│   ├── run_expanse.sh           ← SDSC Expanse launcher (updated for v0.3)
│   └── legacy_dask/             ← archived Dask cluster scripts (Andrea Zonca)
├── obs_TEST.pffd/               ← bundled test observation (img16 + ph256, truncated)
├── scripts/
│   └── bench_convert.py         ← PFF→Zarr codec/chunk benchmark harness
└── pyproject.toml
```

---

## Parameters

All defaults live in `nextflow.config` under `params { … }`. Override on the CLI or via `-params-file`.

| Parameter | Default | Description |
|---|---|---|
| `input_obs_dir` | `obs_TEST.pffd` | Input `.pffd` observation directory |
| `outdir` | `results/` | Base output directory (L0/, L1/ published here) |
| `codec` | `zstd` | Zarr compression codec |
| `level` | `5` | Compression level |
| `time_chunk` | `0` (auto) | Time chunk size; 0 = auto-sized by pypff |
| `ph_sigma` | `5.0` | n-σ threshold for pulse-height masking |
| `ph_offset` | `800` | ADC offset added before pedestal estimation |
| `ph_stride` | `200` | Frame stride for pedestal sampling |
| `img_stride` | `200` | Frame stride for block-median sampling |
| `img_block` | `8` | Spatial block size (pixels) |
| `img_adc_to_pe` | `1.5` | ADC counts per photoelectron |
| `slurm_queue` | `debug` | SLURM queue (HPC profile only) |
| `slurm_account` | `''` | SLURM account (HPC profile only) |

---

## Running on SDSC Expanse

Transfer data with Globus:
```bash
globus login
globus transfer <src_endpoint>:/path/to/obs.pffd \
    <expanse_endpoint>:/expanse/lustre/scratch/$USER/panoseti/inputs/obs.pffd \
    --recursive
```

Submit pipeline:
```bash
bash hpc/run_expanse.sh \
    /expanse/lustre/scratch/$USER/panoseti/inputs/obs.pffd \
    /expanse/lustre/scratch/$USER/panoseti/results
```

Or via Seqera Tower:
```bash
source ~/.bashrc   # loads TOWER_ACCESS_TOKEN
tw launch panoseti_zarr \
    --workspace sdsc/panoseti \
    --profile hpc_slurm \
    --revision main \
    --params-file params.json
```

Example `params.json`:
```json
{
  "input_obs_dir": "/expanse/lustre/scratch/user/panoseti/inputs/obs.pffd",
  "outdir":        "/expanse/lustre/scratch/user/panoseti/results",
  "slurm_account": "sds166",
  "slurm_queue":   "debug"
}
```

---

## Tower CLI cheatsheet

```bash
# Inspect runs
tw runs list --workspace sdsc/panoseti --max 5
tw runs view --id <runId> --workspace sdsc/panoseti --status
tw runs view --id <runId> --workspace sdsc/panoseti download --type log | tail -n 80
tw runs view --id <runId> --workspace sdsc/panoseti tasks

# Relaunch
tw launch panoseti_zarr --workspace sdsc/panoseti --profile hpc_slurm --revision main
```

---

## Testing

Unit tests (no Nextflow required):
```bash
uv sync
uv run pytest tests/ -v
```

End-to-end smoke (requires Nextflow ≥ 26.04):
```bash
nextflow run . -profile laptop --input_obs_dir obs_TEST.pffd --outdir results_smoke
```

---

## Troubleshooting

- **Container not found on HPC**: the `hpc_slurm` profile references `panoseti-zarr-pipeline:0.3.0`.
  Build it with `docker build -t panoseti-zarr-pipeline:0.3.0 .` and convert to SIF for Singularity,
  or pull from a registry if one has been pushed. The `laptop` profile needs no container.
- **`ZarrUserWarning` about `summary.json` / `preview.png`**: zarr-python warns about non-Zarr
  files inside a store directory. These files are intentional (per-store summaries) and the warning
  is harmless.
- **SLURM `debug` queue timeout**: the `debug` queue on Expanse has a 30-minute wall limit.
  Increase `process.time` in `conf/hpc_slurm.config` or switch to `--slurm_queue shared`.
- **Tower run stuck at `SUBMITTED`**: check that the Expanse compute environment is healthy
  and the queue has available slots. Use `tw runs view ... download --type log`.

---

## Relationship to `pypff`

This pipeline depends on `pypff[zarr]` (installed in editable mode from `../pypff`).
The L0 conversion step (`bin/pff2zarr`) calls `pypff.zarr.convert_run` directly — no
hand-written PFF parsing. The L0 stores follow the [pypff Zarr v3 spec](../pypff/docs/zarr_v3_spec.md).
