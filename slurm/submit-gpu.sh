#!/bin/bash

#SBATCH -A pc_nnkinetics -p es1 -q es_normal -t 0-72:0:0 -N 1
#SBATCH --gres=gpu:1 --ntasks 1 --cpus-per-task=8
#SBATCH --job-name=nn_gpu

# Script to submit individual GPU job for a given neural network configuration

# Ensure any error stops the script execution
set -e

# Accept configuration ID as an input argument
CONFIG_ID=$1

# Check for the existence of the secrets file
if [ ! -f "../.secrets" ]; then
    echo "Error: Missing secrets file."
    exit 1
fi

# Load environment variables from the secrets file
source ../.secrets

# Check if required environment variables are set
if [ -z "$GITHUB_RESULTS_REPO" ] || [ -z "$GITHUB_TOKEN" ]; then
    echo "Error: Required environment variables are not set."
    exit 1
fi

# Disable command logging for security
set +x 

# Conditional execution based on provided CONFIG_ID
if [ -n "$CONFIG_ID" ]; then
    echo "Running Config ID: $CONFIG_ID"
    apptainer run \
      --nv \
      --fakeroot \
      --writable-tmpfs \
      --bind /global/home/groups/pc_nnkinetics/results:/app/results \
      --env GITHUB_RESULTS_REPO="$GITHUB_RESULTS_REPO" \
      --env GITHUB_TOKEN="$GITHUB_TOKEN" \
      ../../neural_networks_latest_gpu.sif \
      CONFIGS="$CONFIG_ID"
else
    echo "No config ID provided. Running default configurations."
    apptainer run \
      --nv \
      --fakeroot \
      --writable-tmpfs \
      --bind /global/home/groups/pc_nnkinetics/results:/app/results \
      --env GITHUB_RESULTS_REPO="$GITHUB_RESULTS_REPO" \
      --env GITHUB_TOKEN="$GITHUB_TOKEN" \
      ../../neural_networks_latest_gpu.sif
fi
