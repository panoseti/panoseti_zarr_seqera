#!/usr/bin/env nextflow

// Define parameters
params.input_obs_dir = "obs_TEST.pffd"
params.output_l0_dir = "L0_zarr"
params.output_l1_dir = "L1_zarr"
params.config_file = "config.toml"

// Process for step1_pff_to_zarr.py
process pff_to_zarr {
    publishDir "${params.outdir}/${params.output_l0_dir}", mode: 'copy'

    input:
        path obs_dir
        path config_file

    output:
        path "${params.output_l0_dir}" // Output L0 Zarr directory

    script:
    """
    uv run python step1_pff_to_zarr.py \
        ${obs_dir} \
        ${params.output_l0_dir} \
        ${config_file}
    """
}

// Process for step2_dask_baseline.py
process dask_baseline {
    publishDir "${params.outdir}/${params.output_l1_dir}", mode: 'copy'

    input:
        path l0_zarr_base_dir // This will be the base directory for L0 Zarrs
        path config_file

    output:
        path "${params.output_l1_dir}" // Output L1 Zarr directory

    script:
    """
    # Find all L0 Zarr directories within the input l0_zarr_base_dir
    find ${l0_zarr_base_dir} -maxdepth 1 -type d -name "*.zarr" | while read L0_ZARR; do
        BASENAME_ZARR=\$(basename "\$L0_ZARR" .zarr)
        L1_ZARR="${params.output_l1_dir}/${BASENAME_ZARR}_L1.zarr"
        uv run python step2_dask_baseline.py \
            "$L0_ZARR" \
            "$L1_ZARR" \
            --config "${config_file}"
    done
    """
}

workflow {
    // Create channels for inputs
    input_obs_ch = Channel.fromPath(params.input_obs_dir)
    config_file_ch = Channel.fromPath(params.config_file)

    // Run pff_to_zarr process
    pff_to_zarr(input_obs_ch, config_file_ch)

    // Run dask_baseline process, taking output from pff_to_zarr
    dask_baseline(pff_to_zarr.out, config_file_ch)
}
