#!/usr/bin/env bash

#SBATCH -o /homes/5/fd881/slurm_log/drift_analysis_flapjack-%j.out
#SBATCH -J drift_analysis 
#SBATCH -p pubcpu,basic
#SBATCH -A qtim
#SBATCH -n 1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH -t 36:00:00

export PYENV_VERSION="med-image-drift"
export PYTHONPATH="${HOME}/repos/MedImaging-ModelDriftMonitoring/src":${PYTHONPATH}

python ${HOME}/repos/MedImaging-ModelDriftMonitoring/src/scripts/drift/generate-drift-csv-mgb.py \
    -v /autofs/cluster/qtim/projects/xray_drift/inferences/mgb_with_chexpert_model_vae_take2 \
    -i /autofs/cluster/qtim/projects/xray_drift/inferences/mgb_data_from_chexpert_retrain_frontal_only_lr1e-4_frozen_step25_qwen_labels_correct \
    -o /autofs/cluster/qtim/projects/xray_drift/drift_analyses/all_flapjack \
    #--bad_q 0.25 \
    #--bad_start_date "2019-10-01" \
    #--bad_end_date "2020-01-01" \
    #--bad_sample_start_date "2019-10-01" \
    #--bad_sample_end_date "2020-01-01" \
    #--good_q 0.25 \
    #--good_start_date "2019-10-01" \
    #--good_end_date "2020-01-01" \
    #--good_sample_start_date "2019-10-01" \
    #--good_sample_end_date "2020-01-01" \