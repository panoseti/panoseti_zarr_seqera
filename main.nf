#!/usr/bin/env nextflow

// Define parameters
params.input_obs_dir = "obs_TEST.pffd"
params.output_l0_dir = "L0_zarr"
params.output_l1_dir = "L1_zarr"
params.config_file = "config.toml"
params.outdir = "."

// Process for step1_pff_to_zarr.py
process pff_to_zarr {
    publishDir "${params.outdir}/${params.output_l0_dir}", mode: 'copy'
    container 'oras://ghcr.io/zonca/singularity_dask_zarr:latest'
    file 'step1_pff_to_zarr.py', 'pff.py'

    input:
        path obs_dir
        path config_file

    output:
        path "${params.output_l0_dir}" // Output L0 Zarr directory

    script:
    """
    python step1_pff_to_zarr.py \
        ${obs_dir} \
        ${params.output_l0_dir} \
        ${config_file}
    """
}

// Process for step2_dask_baseline.py
process dask_baseline {
    publishDir "${params.outdir}/${params.output_l1_dir}", mode: 'copy'

    container 'oras://ghcr.io/zonca/singularity_dask_zarr:latest'
    file 'step2_dask_baseline.py', 'pff.py'
    input:
        path l0_zarr_base_dir // This will be the base directory for L0 Zarrs
        path config_file

    output:
        path "${params.output_l1_dir}" // Output L1 Zarr directory

    script:
    """
    # Create the output L1 Zarr directory if it doesn't exist
    mkdir -p ${params.output_l1_dir}

    # Find all L0 Zarr directories within the input l0_zarr_base_dir
    find ${l0_zarr_base_dir} -maxdepth 1 -type d -name "*.zarr" | while read L0_ZARR; do
        BASENAME_ZARR=\$(basename "\$L0_ZARR" .zarr)
        BASENAME_ZARR_L1="\${BASENAME_ZARR}_L1"
        L1_ZARR="${params.output_l1_dir}/\${BASENAME_ZARR_L1}.zarr"
        python step2_dask_baseline.py \
            "\$L0_ZARR" \
            "\$L1_ZARR" \
            --config "${config_file}"
    done
    """
}

workflow {
    main:
        input_obs_ch = Channel.fromPath(params.input_obs_dir)
        config_for_step1 = Channel.value(file(params.config_file))
        config_for_step2 = Channel.value(file(params.config_file))

        l0_results = pff_to_zarr(input_obs_ch, config_for_step1)
        dask_baseline(l0_results.out, config_for_step2)

    emit:
        dask_baseline.out
}
