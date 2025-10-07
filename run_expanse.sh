#!/bin/bash

# Run the Nextflow pipeline on Expanse using the slurm_debug profile

echo "Starting Nextflow pipeline on Expanse..."

# You can override parameters by adding them to the command, e.g.:
# --input_obs_dir my_custom_obs.pffd \
# --output_l0_dir my_l0_output \
# --output_l1_dir my_l1_output \
# --config_file my_custom_config.toml

nextflow run pipeline.nf -profile slurm_debug

echo "Nextflow pipeline submitted."

