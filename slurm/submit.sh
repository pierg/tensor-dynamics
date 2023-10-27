#!/bin/bash

#SBATCH -A pc_nnkinetics -p lr6 -q lr_normal -t 0-42:0:0 -N 1
#SBATCH --job-name=nn_cpu

# Ensure any error stops the script execution
set -e 


# Optional: Accept input argument for CONFIG_ID
CONFIG_ID=$1

# Check if the secrets file exists before sourcing
if [ ! -f "../.secrets" ]; then
    echo "Error: Missing secrets file."
    exit 1
fi

# Source the secrets file to export GITHUB_TOKEN and other variables
source ../.secrets

# Prevent the script from logging the following commands to protect the confidentiality of the token.
set +x 

# Ensure required variables are set up.
if [ -z "$GITHUB_RESULTS_REPO" ] || [ -z "$GITHUB_TOKEN" ]; then
    echo "Error: Required environment variables GITHUB_RESULTS_REPO or GITHUB_TOKEN are not set."
    exit 1
fi

# Check if CONFIG_ID was provided as an argument
if [ -n "$CONFIG_ID" ]; then
    echo "Running Config ID: $CONFIG_ID"
    
    # Run the container, passing in the CONFIG_ID
    apptainer run \
      --fakeroot \
      --writable-tmpfs \
      --bind /global/home/groups/pc_nnkinetics/app/data:/data \
      --bind /global/home/groups/pc_nnkinetics/app/results:/results \
      --env GITHUB_RESULTS_REPO="$GITHUB_RESULTS_REPO" \
      --env GITHUB_TOKEN="$GITHUB_TOKEN" \
      ../../neural_networks_latest_gpu.sif \
      CONFIGS="$CONFIG_ID"
else
    echo "No config ID. Running all configurations."

    # Run the container, running all configurations
    apptainer run \
      --fakeroot \
      --writable-tmpfs \
      --bind /global/home/groups/pc_nnkinetics/app/data:/data \
      --bind /global/home/groups/pc_nnkinetics/app/results:/results \
      --env GITHUB_RESULTS_REPO="$GITHUB_RESULTS_REPO" \
      --env GITHUB_TOKEN="$GITHUB_TOKEN" \
      ../../neural_networks_latest_gpu.sif
fi