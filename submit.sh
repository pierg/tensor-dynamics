#!/bin/bash

#SBATCH -A pc_nnkinetics -p lr6 -q lr_normal -t 0-0:30:0 -N1


#CONFIGS='config_A config_P'
apptainer run \
  --fakeroot \
  --writable-tmpfs \
  --bind /global/scratch/users/edkinigstein/Dataset_2/F1:/data \
  --bind /global/home/groups-sw/pc_nnkinetics/results:/app/results \
  --bind /global/home/groups-sw/pc_nnkinetics/logs:/app/logs \
  neural_networks_latest_v2.sif