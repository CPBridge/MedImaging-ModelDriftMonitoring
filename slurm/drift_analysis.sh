#!/usr/bin/env bash

#SBATCH -o /homes/5/fd881/slurm_log/drift_analysis_05_resnet-%j.out
#SBATCH -J drift_analysis 
#SBATCH -p basic 
#SBATCH -A qtim
#SBATCH -n 1
#SBATCH --cpus-per-task=24
#SBATCH --mem=96G
#SBATCH -t 36:00:00

export PYENV_VERSION="med-image-drift"
export PYTHONPATH="${HOME}/repos/MedImaging-ModelDriftMonitoring/src":${PYTHONPATH}

python ${HOME}/repos/MedImaging-ModelDriftMonitoring/src/scripts/drift/generate-drift-csv-mgb.py \
    -v /autofs/cluster/qtim/projects/xray_drift/models/mgb/resnet_features\
    -i /autofs/cluster/qtim/projects/xray_drift/inferences/mgb_data_from_chexpert_retrain_frontal_only_lr1e-4_frozen_step25 \
    -o /autofs/cluster/qtim/projects/xray_drift/drift_analyses/bad_good_q_05_resnet \
    --bad_q 0.05 \
    --bad_start_date "2019-10-01" \
    --bad_end_date "2020-01-01" \
    --bad_sample_start_date "2019-10-01" \
    --bad_sample_end_date "2020-01-01" \
    --good_q 0.05 \
    --good_start_date "2019-10-01" \
    --good_end_date "2020-01-01" \
    --good_sample_start_date "2019-10-01" \
    --good_sample_end_date "2020-01-01" \
    --num_vae_features 512 \