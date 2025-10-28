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

    input:
        path obs_dir
        val config_file_path

    output:
        path "${params.output_l0_dir}" // Output L0 Zarr directory

    script:
    """
    python ${projectDir}/step1_pff_to_zarr.py \
        ${obs_dir} \
        ${params.output_l0_dir} \
        ${config_file_path}
    """
}

// Process for step2_dask_baseline.py
process dask_baseline {
    publishDir "${params.outdir}/${params.output_l1_dir}", mode: 'copy'

    container 'oras://ghcr.io/zonca/singularity_dask_zarr:latest'
    input:
        path l0_zarr_base_dir // This will be the base directory for L0 Zarrs
        val config_file_path

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
        python ${projectDir}/step2_dask_baseline.py \
            "\$L0_ZARR" \
            "\$L1_ZARR" \
            --config "${config_file_path}"
    done
    """
}

workflow {
    main:
        def obs_path = params.input_obs_dir.startsWith('/') ? params.input_obs_dir : "${projectDir}/${params.input_obs_dir}"
        def config_file_path = params.config_file.startsWith('/') ? params.config_file : "${projectDir}/${params.config_file}"

        input_obs_ch = Channel.fromPath(obs_path)

        pff_to_zarr(input_obs_ch, config_file_path)
        dask_baseline(pff_to_zarr.out, config_file_path)

    emit:
        dask_baseline.out
}
