#!/usr/bin/env bash

#SBATCH -o /homes/5/fd881/slurm_log/drift_analysis_allpoc_mmd_500_noresamp_florence-%j.out
#SBATCH -J drift_analysis_allpoc_mmd
#SBATCH -p pubgpu
#SBATCH -A qtim
#SBATCH -n 1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gpus=1
#SBATCH -t 48:00:00

export PYENV_VERSION="alibi_test"
export PYTHONPATH="${HOME}/repos/MedImaging-ModelDriftMonitoring/src":${PYTHONPATH}

python ${HOME}/repos/MedImaging-ModelDriftMonitoring/src/scripts/drift/generate-drift-csv-mgb.py \
    -v /autofs/space/crater_001/datasets/private/xray_drift/mgb_florence_embeddings \
    -i /autofs/cluster/qtim/projects/xray_drift/inferences/classification_final_allpoc_inference_woconsolidation \
    -o /autofs/cluster/qtim/projects/xray_drift/drift_analyses/drift_analysis_allpoc_mmd_500_noresamp_florence \
    --window "30D" \
    --combine_vae_classifier 1 \
    --sample_size 500 \
    --n_samples 1 \
    --replacement 0 \
    --num_vae_features 1024 \
    #--point_of_care "MGH IMG XR NS" \
    #--ref_window_start "2020-10-01" \
    #--ref_window_end "2021-01-01" \
    #--point_of_care "MGH IMG XR ER MG WHT1"
    #--stride "1D" \ 
    #--sample_size 500 \
    #--n_samples 1 \
    #--replacement 1 \
    #--point_of_care "MGH IMG XR NS"
    #--good_q 0.05 \
    #--good_start_date "2019-10-01" \
    #--good_end_date "2020-01-01" \
    #--good_sample_start_date "2019-10-01" \
    #--good_sample_end_date "2020-01-01" \
    #--bad_q 0.05 \
    #--bad_start_date "2019-10-01" \
    #--bad_end_date "2020-01-01" \
    #--bad_sample_start_date "2019-10-01" \
    #--bad_sample_end_date "2020-01-01" \
    #--ref_window_start "2021-03-01" \
    #--ref_window_end "2021-06-01" \
    #--n_samples 100 \
    #--bad_sample_start_date "2019-10-01" \
    #--bad_sample_end_date "2020-01-01" \
    #--good_q 0.25 \
    #--good_start_date "2019-10-01" \
    #--good_end_date "2020-01-01" \
    #--good_sample_start_date "2019-10-01" \
    #--good_sample_end_date "2020-01-01" \
