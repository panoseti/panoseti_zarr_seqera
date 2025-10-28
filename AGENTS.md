Project insights
----------------
- Nextflow pipeline `main.nf` defines two DSL2 processes: `pff_to_zarr` converts `.pffd` observations into L0 Zarr datasets, and `dask_baseline` turns each L0 directory into an L1 Zarr. Both run inside `oras://ghcr.io/zonca/singularity_dask_zarr:latest`.
- `dask_baseline` expects L0 directories ending in `.zarr`; it now derives L1 names as `<basename>_L1.zarr` to keep outputs unique.
- Workflow parameters default to local test artifacts (`obs_TEST.pffd`, `config.toml`) and emit results under `params.outdir`.
- `nextflow.config` exposes a single `slurm_debug` profile that submits to Expanse via SLURM (`debug` queue), stages work under `/expanse/lustre/scratch/$USER/temp_project`, and enables timeline/report/trace artifacts (written into `logs/`).
- Pipeline fetches dependencies (`step1_pff_to_zarr.py`, `step2_dask_baseline.py`, `pff.py`) from the repo and uses Singularity with node-local scratch exported through `TMPDIR`/`SCRATCH_DIR`.
- Recent production issue: reusing the single `config_file_ch` channel across two processes triggered `Process 'pff_to_zarr' has been already used`; provisioning independent value channels with `Channel.value(file(params.config_file))` for each consumer avoids the duplicate invocation.

Tower CLI cheatsheet
--------------------
- Authenticate by sourcing `~/.bashrc` (loads `TOWER_ACCESS_TOKEN`); the CLI binary is `tw`.
- Discover context  
  - `tw organizations list` → show accessible org IDs (e.g., `sdsc`).  
  - `tw workspaces list --organization sdsc` → list workspaces (e.g., `sdsc/panoseti`).  
  - `tw pipelines list --workspace sdsc/panoseti` → list registered pipelines (IDs and Git repos).
- Inspect runs  
  - `tw runs list --workspace sdsc/panoseti --max 5` → recent executions.  
  - `tw runs view --id <runId> --workspace sdsc/panoseti --status` → high-level status.  
  - `tw runs view --id <runId> --workspace sdsc/panoseti download --type log` → workflow log (pipe to `tail -n 80` for summary).  
  - `tw runs view --id <runId> --workspace sdsc/panoseti tasks` → task-level statuses.
- Launching  
  - `tw launch panoseti_zarr --workspace sdsc/panoseti --profile slurm_debug --revision main` → submit latest `main` revision with workspace defaults.  
  - Optional flags: `--params-file <json/yaml>`, `--name <run_name>`, `--compute-env <envName>`, `--wait SUCCEEDED`.
- Common workflow: source credentials → check `tw runs list` for last status → inspect logs → push fixes → relaunch with `tw launch`.
