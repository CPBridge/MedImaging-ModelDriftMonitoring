#!/usr/bin/env bash

#SBATCH -o /homes/5/fd881/slurm_log/resnet_features-%j.out
#SBATCH -J resnet_features
#SBATCH -p rtx6000,rtx8000
#SBATCH -A qtim
#SBATCH -n 1
#SBATCH --cpus-per-task=12
#SBATCH --gpus 1
#SBATCH --mem=256G
#SBATCH -t 36:00:00

export PYENV_VERSION="med-image-drift"
export PYTHONPATH="${HOME}/repos/MedImaging-ModelDriftMonitoring/src":${PYTHONPATH}

python ${HOME}/repos/MedImaging-ModelDriftMonitoring/src/scripts/resnet_features/score_resnet_features.py \
    --output_dir /autofs/cluster/qtim/projects/xray_drift/models/mgb/resnet_features \
    --default_root_dir /autofs/cluster/qtim/projects/xray_drift/models/mgb/resnet_features \
    --data_folder /autofs/cluster/qtim/datasets/private/xray_drift/dicom/ \
    --dataset mgb \
    --num_workers 12 \
    