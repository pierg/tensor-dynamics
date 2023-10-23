#!/bin/bash

#SBATCH -A pc_nnkinetics -p lr6 -q lr_normal -t 0-42:0:0 -N1

apptainer run \
  --fakeroot \
  --writable-tmpfs \
  --bind /global/scratch/users/edkinigstein/Dataset2/F1:/data \
  --bind /global/home/groups-sw/pc_nnkinetics/results:/app/results \
  --bind /global/home/groups-sw/pc_nnkinetics/logs:/app/logs \
  ../neural_networks_latest_v2.sif

