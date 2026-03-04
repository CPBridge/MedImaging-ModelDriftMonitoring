#!/usr/bin/env bash

export PYENV_VERSION="med-image-drift"
export PYTHONPATH="${HOME}/repos/MedImaging-ModelDriftMonitoring/src":${PYTHONPATH}


# All POC
python basic_performance_plots.py \
    "/autofs/cluster/qtim/projects/xray_drift/drift_analyses/drift_analysis_allpoc_emd_jackknife_helllinger_final_florence/output.csv" \
    "/autofs/cluster/qtim/projects/xray_drift/drift_analyses/PLOTS/paper/drift_analysis_allpoc_emd_jackknife_helllinger_final_florence_PLOTS/"

# ER
#python basic_performance_plots.py \
#    "/autofs/cluster/qtim/projects/xray_drift/drift_analyses/drift_analysis_allpoc_emd_jackknife_helllinger_final_er_florence/output.csv" \
#    "/autofs/cluster/qtim/projects/xray_drift/drift_analyses/PLOTS/paper/drift_analysis_allpoc_emd_jackknife_helllinger_final_er_florence_PLOTS/"

# WAC2
#python basic_performance_plots.py \
#    "/autofs/cluster/qtim/projects/xray_drift/drift_analyses/drift_analysis_allpoc_emd_jackknife_helllinger_final_wac2_florence/output.csv" \
#    "/autofs/cluster/qtim/projects/xray_drift/drift_analyses/PLOTS/paper/drift_analysis_allpoc_emd_jackknife_helllinger_final_wac2_florence_PLOTS/"

# Recalibration Reference Window
#python basic_performance_plots.py \
#    "/autofs/cluster/qtim/projects/xray_drift/drift_analyses/drift_analysis_allpoc_emd_jackknife_helllinger_final_ref10_2020_florence/output.csv" \
#    "/autofs/cluster/qtim/projects/xray_drift/drift_analyses/PLOTS/paper/drift_analysis_allpoc_emd_jackknife_helllinger_final_ref10_2020_florence_PLOTS/"

# Run Recalibration Experiment
#python recalibration_experiment.py \
#    "/autofs/cluster/qtim/projects/xray_drift/drift_analyses/PLOTS/paper/drift_analysis_allpoc_emd_jackknife_helllinger_final_florence_PLOTS/weighted_mmc+_with_range.csv" \
#    "/autofs/cluster/qtim/projects/xray_drift/drift_analyses/PLOTS/paper/drift_analysis_allpoc_emd_jackknife_helllinger_final_ref10_2020_florence_PLOTS/weighted_mmc+_with_range.csv" \
#    "/autofs/cluster/qtim/projects/xray_drift/drift_analyses/PLOTS/paper/drift_analysis_allpoc_emd_jackknife_helllinger_final_ref10_2020_florence_PLOTS/"

