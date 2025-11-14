#!/usr/bin/env nextflow

nextflow.enable.dsl = 2

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
process dask_baseline_conditional_A {
    publishDir "${params.outdir}/${params.output_l1_dir}/A", mode: 'copy'

    container 'oras://ghcr.io/zonca/singularity_dask_zarr:latest'
    input:
        path l0_zarr_base_dir // This will be the base directory for L0 Zarrs
        val config_file_path

    output:
        path "${params.output_l1_dir}/A" // Output L1 Zarr directory

    script:
    """
    # Create the output L1 Zarr directory if it doesn't exist
    mkdir -p ${params.output_l1_dir}/A

    # Find all L0 Zarr directories within the input l0_zarr_base_dir
    find ${l0_zarr_base_dir} -maxdepth 1 -type d -name "*.zarr" | while read L0_ZARR; do
        BASENAME_ZARR=\$(basename "\$L0_ZARR" .zarr)
        BASENAME_ZARR_L1="\${BASENAME_ZARR}_L1"
        L1_ZARR="${params.output_l1_dir}/A/\${BASENAME_ZARR_L1}.zarr"
        python ${projectDir}/step2_dask_baseline.py \
            "\$L0_ZARR" \
            "\$L1_ZARR" \
            --config "${config_file_path}"
    done
    """
}

// Process for step2_dask_baseline.py (duplicated for conditional B)
process dask_baseline_conditional_B {
    publishDir "${params.outdir}/${params.output_l1_dir}/B", mode: 'copy'

    container 'oras://ghcr.io/zonca/singularity_dask_zarr:latest'
    input:
        path l0_zarr_base_dir // This will be the base directory for L0 Zarrs
        val config_file_path

    output:
        path "${params.output_l1_dir}/B" // Output L1 Zarr directory

    script:
    """
    # Create the output L1 Zarr directory if it doesn't exist
    mkdir -p ${params.output_l1_dir}/B

    # Find all L0 Zarr directories within the input l0_zarr_base_dir
    find ${l0_zarr_base_dir} -maxdepth 1 -type d -name "*.zarr" | while read L0_ZARR; do
        BASENAME_ZARR=\$(basename "\$L0_ZARR" .zarr)
        BASENAME_ZARR_L1="\${BASENAME_ZARR}_L1"
        L1_ZARR="${params.output_l1_dir}/B/\${BASENAME_ZARR_L1}.zarr"
        python ${projectDir}/step2_dask_baseline.py \
            "\$L0_ZARR" \
            "\$L1_ZARR" \
            --config "${config_file_path}"
    done
    """
}

// Process to generate a random decision
process random_decision {
    output:
        val decision

    script:
    """
    echo "0"
    """
}

workflow {
    main:
        def obs_path = params.input_obs_dir.startsWith('/') ? params.input_obs_dir : "${projectDir}/${params.input_obs_dir}"
        def config_file_path = params.config_file.startsWith('/') ? params.config_file : "${projectDir}/${params.config_file}"

        input_obs_ch = Channel.fromPath(obs_path)

        // First step: pff_to_zarr
        pff_to_zarr(input_obs_ch, config_file_path)

        // Generate a random decision
        random_decision()

        // Combine pff_to_zarr output with the random decision
        pff_to_zarr.out
            .combine(random_decision.out.map { it.trim() })
            .set { combined_ch }

        // Branch based on the random decision
        combined_ch
            .branch { l0_zarr_base_dir, decision ->
                if (decision == "0") emit: path_A: l0_zarr_base_dir
                else emit: path_B: l0_zarr_base_dir
            }
            .set { branched_paths }

        // Execute conditional processes
        dask_baseline_conditional_A(branched_paths.path_A, config_file_path)
        dask_baseline_conditional_B(branched_paths.path_B, config_file_path)

    emit:
        dask_baseline_conditional_A.out
        dask_baseline_conditional_B.out
}
