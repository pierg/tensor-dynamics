#!/bin/bash

#SBATCH -A pc_nnkinetics -p lr6 -q lr_normal -t 0-42:0:0 -N 1
#SBATCH --job-name=my_neural_network_job
#SBATCH --output=result-%j.out  # %j inserts the job number
#SBATCH --error=result-%j.err

# Ensure any error stops the script execution
set -e 

# Check if configuration id was passed to the script, if not, exit with error.
if [ -z "$1" ]; then
    echo "Error: No configuration ID provided."
    exit 1
fi

CONFIG_ID=$1

# Source the secrets file to export GITHUB_TOKEN and other variables
source ../.secrets

# Prevent the script from logging the following commands to protect the confidentiality of the token.
set +x 

# Ensure required variables are set up.
if [ -z "$GITHUB_RESULTS_REPO" ] || [ -z "$GITHUB_TOKEN" ]; then
    echo "Error: Required environment variables GITHUB_RESULTS_REPO or GITHUB_TOKEN are not set."
    exit 1
fi

# Run the container with the neural network application, passing in the CONFIG_ID
apptainer run \
  --fakeroot \
  --writable-tmpfs \
  --bind /global/scratch/users/edkinigstein/Dataset2/F1:/data \
  --env GITHUB_RESULTS_REPO="$GITHUB_RESULTS_REPO" \
  --env GITHUB_TOKEN="$GITHUB_TOKEN" \
  --env CONFIG_ID="$CONFIG_ID" \  # Pass the CONFIG_ID to the container
  neural_networks_latest.sif
