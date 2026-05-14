# AGENTS.md — Maintainer notes

## Pipeline overview (v0.3.0)

- Nextflow 26.04 strict DSL2 with two stages:
  1. `PFF_TO_ZARR` (module `modules/pff_to_zarr.nf`): calls `bin/pff2zarr`, writes L0 Zarr stores + `manifest.tsv`.
  2. `CALIBRATE` (subworkflow `subworkflows/calibrate.nf`): fans out per (product, kind) to either `CALIBRATE_PH` or `CALIBRATE_IMG`.
- Output publishing uses the new NF 26 `output {}` block (no `publishDir`).
- Per-product parallelism is Nextflow-native; no Dask cluster needed.
- Python logic lives in `src/panoseti_zarr_pipeline/`; `bin/` contains thin typer wrappers.
- `obs_TEST.pffd/` is the local smoke-test fixture (two truncated PFF files: img16 + ph256).

## Profiles

| Profile | Executor | When to use |
|---|---|---|
| `laptop` | local | development, smoke tests |
| `hpc_slurm` | SLURM | Expanse / any SLURM cluster |

## Running locally

```bash
uv sync
nextflow run . -profile laptop
# or with a custom input:
nextflow run . -profile laptop --input_obs_dir /path/to/obs.pffd --outdir results/
```

## Running on Expanse

```bash
bash hpc/run_expanse.sh /expanse/lustre/scratch/$USER/panoseti/inputs/obs.pffd \
                        /expanse/lustre/scratch/$USER/panoseti/results
# Override account/queue:
SLURM_ACCOUNT=abc123 SLURM_QUEUE=shared bash hpc/run_expanse.sh ...
```

## Tower CLI cheatsheet

Tower authentication: source `~/.bashrc` (loads `TOWER_ACCESS_TOKEN`); CLI binary is `tw`.

```bash
# Context
tw organizations list
tw workspaces list --organization sdsc
tw pipelines list --workspace sdsc/panoseti

# Inspect runs
tw runs list --workspace sdsc/panoseti --max 5
tw runs view --id <runId> --workspace sdsc/panoseti --status
tw runs view --id <runId> --workspace sdsc/panoseti download --type log | tail -n 80
tw runs view --id <runId> --workspace sdsc/panoseti tasks

# Launch
tw launch panoseti_zarr \
  --workspace sdsc/panoseti \
  --profile hpc_slurm \
  --revision main \
  --params-file params.json   # optional overrides
```

Key params to override via `params.json` on Tower:
```json
{
  "input_obs_dir": "/expanse/lustre/scratch/user/panoseti/inputs/obs.pffd",
  "outdir":        "/expanse/lustre/scratch/user/panoseti/results",
  "slurm_account": "sds166",
  "slurm_queue":   "debug"
}
```

## Transferring data with Globus

```bash
# Install Globus CLI: pip install globus-cli
globus login
globus endpoint search "SDSC Expanse"     # find endpoint ID
globus transfer <src_endpoint>:<src_path> <dst_endpoint>:/expanse/lustre/scratch/$USER/panoseti/inputs/
```

## Legacy Dask cluster scripts

See `hpc/legacy_dask/` for the original SSH/local Dask cluster management scripts written
by Andrea Zonca. These are no longer used by the main pipeline but are preserved as reference.
The `hpc/legacy_dask/README.md` explains the difference and when they might still be useful.

## Known issues / gotchas

- `nextflow.enable.strict = true` is set in `nextflow.config`. Any implicit channel binding
  or unnamed workflow block will raise an error — this is intentional.
- The container for `hpc_slurm` (`panoseti-zarr-pipeline:0.3.0`) must be built/pulled manually
  before the first HPC run. See `Dockerfile` (to be added) or build with:
  `docker build -t panoseti-zarr-pipeline:0.3.0 .` then convert to SIF for Singularity.
- L1 stores contain `summary.json` and `preview.png` alongside the Zarr arrays; zarr-python
  will emit a `ZarrUserWarning` about these non-Zarr files when listing the group — this is
  expected and harmless.
