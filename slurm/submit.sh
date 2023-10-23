#!/bin/bash

#SBATCH -A pc_nnkinetics -p lr6 -q lr_normal -t 0-42:0:0 -N1


# Source the secrets file to export GITHUB_TOKEN
source ../.secrets

# The 'set +x' command is used to prevent the script from logging the following commands, 
# protecting the confidentiality of the token.
set +x 


apptainer run \
  --fakeroot \
  --writable-tmpfs \
  --bind /global/scratch/users/edkinigstein/Dataset2/F1:/data \
  --env GITHUB_RESULTS_REPO=$GITHUB_RESULTS_REPO \
  --env GITHUB_TOKEN=$GITHUB_TOKEN \
  ../../neural_networks_latest.sif