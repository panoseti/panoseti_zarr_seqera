# PANOSETI Zarr Nextflow Pipeline

This repository orchestrates the two-stage conversion of PANOSETI observation data into hierarchical Zarr stores using Nextflow DSL2 and Seqera Platform (Tower).

- **Stage 1 – `pff_to_zarr`**: runs `step1_pff_to_zarr.py` to convert `.pffd` observations into L0 Zarr datasets.
- **Stage 2 – `dask_baseline`**: runs `step2_dask_baseline.py` to aggregate each L0 Zarr into a corresponding L1 Zarr baseline.
- Both stages execute inside the container image `oras://ghcr.io/zonca/singularity_dask_zarr:latest`.

The pipeline is typically launched via Tower against SDSC Expanse resources, using a dedicated SLURM debug profile defined in `nextflow.config`.

## Repository Layout

| Path | Purpose |
| ---- | ------- |
| `main.nf` | Nextflow DSL2 workflow wiring the two processes. Each process stages its scripts plus `config.toml` through the `file` directive. |
| `nextflow.config` | Tower/Expanse configuration profile (`slurm_debug`) that specifies SLURM options, staging directories under `/expanse/lustre/scratch/$USER/temp_project`, and enables timeline/report/trace outputs under `logs/`. |
| `config.toml` | Runtime configuration consumed by both processes. |
| `step1_pff_to_zarr.py`, `step2_dask_baseline.py`, `pff.py` | Python utilities used by the workflow stages. |
| `obs_TEST.pffd` | Example observation bundle for quick smoke tests. |
| `run.sh`, `run_expanse.sh`, `run_old.sh` | Helper launch scripts (not used by Tower). |
| `AGENTS.md` | Maintainer cheatsheet with process notes and Tower CLI quick commands. |

## Requirements

- Nextflow `25.04` or newer with DSL2 enabled (Tower launches supply this via the configured compute environment).
- Access to `oras://ghcr.io/zonca/singularity_dask_zarr:latest` (requires Singularity/Apptainer runtime on the compute nodes).
- Seqera Platform (Tower) CLI `tw`, authenticated via `TOWER_ACCESS_TOKEN` (sourced from `~/.bashrc` in the SDSC environment).
- SDSC Expanse workspace `sdsc/panoseti` with the `expanse-compute-3` environment.
- Input observation directory (`params.input_obs_dir`) and shared filesystem locations referenced in `nextflow.config`.

## Parameters

Default values are defined both in `main.nf` and `nextflow.config`. Override them via a params file or CLI flags.

| Parameter | Default | Description |
| --------- | ------- | ----------- |
| `input_obs_dir` | `obs_TEST.pffd` | Path (file or directory) holding the source PFFD data. |
| `output_l0_dir` | `L0_zarr` | Name of the directory where L0 Zarr outputs are published. |
| `output_l1_dir` | `L1_zarr` | Name of the directory where the stage-two L1 Zarr outputs are published. |
| `config_file` | `config.toml` | Configuration file staged into both processes. |
| `outdir` | `.` (overridden to Expanse scratch in `slurm_debug`) | Base directory for published outputs. |

These values are referenced through `params.<name>` inside the processes; adjust them in a Tower launch or with `-params-file` when running locally.

## Running the Pipeline with Tower CLI

1. **Authenticate**
   ```bash
   source ~/.bashrc      # exports TOWER_ACCESS_TOKEN and endpoint
   ```

2. **Confirm workspace and pipeline**
   ```bash
   tw organizations list
   tw workspaces list --organization sdsc
   tw pipelines list --workspace sdsc/panoseti
   ```

3. **Inspect recent runs**
   ```bash
   tw runs list --workspace sdsc/panoseti --max 5
   tw runs view --id <runId> --workspace sdsc/panoseti --status
   tw runs view --id <runId> --workspace sdsc/panoseti download --type log | tail -n 80
   ```

4. **Launch a new execution**
   ```bash
   tw launch panoseti_zarr \
     --workspace sdsc/panoseti \
     --profile slurm_debug \
     --revision main \
     --name <optional_run_name> \
     --params-file <optional_params.json>
   ```

5. **Monitor progress**
   ```bash
   tw runs view --id <newRunId> --workspace sdsc/panoseti --status
   tw runs view --id <newRunId> --workspace sdsc/panoseti tasks
   ```

The Tower web UI linked in the launch output provides real-time dashboards, timeline, and trace artifacts.

## Running Locally (development)

Local execution is possible for smoke testing, though SLURM-specific settings in `slurm_debug` may not apply. Example:

```bash
nextflow run . \
  -profile slurm_debug \          # or a custom profile that matches your environment
  -params-file local-params.json
```

If you do not have SLURM or Expanse paths available, create a development profile in `nextflow.config` that sets `process.executor = 'local'`, `workDir`, and `params.outdir` to local directories.

## Troubleshooting

- **`DuplicateProcessInvocation` errors:** Nextflow flags this when the same process instance is invoked multiple times with shared channel consumers. The pipeline now stages `config.toml` directly via each process’ `file` directive, so avoid reintroducing shared config channels unless you split them with dedicated clones (e.g., `config_file_ch.into { a; b }` with distinct process bindings).
- **Tower run stuck at `SUBMITTED`:** Check that the `expanse-compute-3` environment is healthy and that the SLURM `debug` queue has available slots. Review the workflow log via `tw runs view ... download --type log`.
- **Container resolution issues:** Ensure Singularity/Apptainer can reach `ghcr.io`; if running locally without ORAS support, pull the image manually or update the container reference to a format supported in your environment.
- **Output directories missing:** Both stages publish results under `params.outdir`. Confirm `params.outdir` points to a writable path and that the compute environment grants write access.

## Contributing

- Keep process scripts idempotent and parameterized; the Nextflow stages rely on deterministic naming (`<basename>_L1.zarr`) for downstream processing.
- Update `AGENTS.md` when new operational details or Tower instructions arise so on-call teammates can ramp quickly.
- Run `tw runs view ... download --type log` after each launch to verify the workflow reaches the intended tasks before concluding a change is stable.
