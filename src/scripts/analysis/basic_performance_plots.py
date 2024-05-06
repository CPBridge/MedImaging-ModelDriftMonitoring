from email import utils
from pathlib import Path
import pandas as pd
import sys
import os
sys.path.append('/autofs/homes/005/fd881/repos/MedImaging-ModelDriftMonitoring/')
sys.path.append('/home/fd881/repos/MedImaging-ModelDriftMonitoring/')
from datetime import datetime


import click
import json
from pycrumbs import tracked

from src.model_drift.data import mgb_data
from src.scripts.analysis import analysis_utils
import numpy as np
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from tqdm import tqdm
date_format = mdates.DateFormatter('%Y-%m')
month_locator = mdates.MonthLocator(interval=3)
plt.rcParams['svg.fonttype'] = 'none'



@click.command()
@click.argument('drift-csv-path', type=Path)
@click.argument('output-dir', type=Path)
@click.option('--ref-window-start', default='', help='Start of reference window for MMC calculation')
@click.option('--ref-window-end', default='', help='End of reference window for MMC calculation')
@tracked(directory_parameter='output_dir')
def basic_performance_plots(
        drift_csv_path: Path,
        output_dir: Path,
        ref_window_start: str = '', 
        ref_window_end: str = '', 
):
    """Makes some basic performance against time plots from a drift CSV."""
    df = pd.read_csv(drift_csv_path, header=[0, 1, 2, 3])

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # The date column gets read in with a stupid name
    date_col = tuple(f'Unnamed: 0_level_{i}' for i in range(4))
    performance_col = ('performance', 'micro avg', 'auroc', 'mean')

    if ref_window_start:
        ref_window_start = datetime.strptime(ref_window_start, "%Y-%m-%d")
        ref_window_end = datetime.strptime(ref_window_end, "%Y-%m-%d")
    else:
        ref_window_start = mgb_data.TRAIN_DATE_END
        ref_window_end = mgb_data.VAL_DATE_END

    df[date_col] = pd.to_datetime(df[date_col])


    analysis_utils.create_performance_plots(df, output_dir)

    # Unweighted MMC
    mmc_cols = [
        col for col in df.columns
        if not col[0].startswith('performance')
        and col[2] == 'distance'
        and col[3] == 'mean'
    ]
    mmc_cols_min = [
        col for col in df.columns
        if not col[0].startswith('performance')
        and col[2] == 'distance'
        and col[3] == 'min'
    ]
    mmc_cols_max = [
        col for col in df.columns
        if not col[0].startswith('performance')
        and col[2] == 'distance'
        and col[3] == 'max'
    ]

    vae_cols = [
        col for col in df.columns
        if col[0].startswith('mu') | col[0].startswith('full_mu')
        and col[2] == 'distance'
        and col[3] == 'mean'
    ]

    score_cols = [
        col for col in df.columns
        if  col[0].startswith('activation')
        and col[2] == 'distance'
        and col[3] == 'mean'
    ]

    metadata_cols = [
        col for col in df.columns
        if not col[0].startswith('performance')
        if not col[0].startswith('mu')
        if not col[0].startswith('activation')
        and col[2] == 'distance'
        and col[3] == 'mean'
    ]

    
    mmc_df = df[mmc_cols + [date_col]].copy()
    mmc_df_min = df[mmc_cols_min + [date_col]].copy()
    mmc_df_max = df[mmc_cols_max + [date_col]].copy()

    ref_df = mmc_df[(mmc_df[date_col] >= ref_window_start) & (mmc_df[date_col] < ref_window_end)].copy()

    mmc_df_weights = df[mmc_cols + [date_col]+ [performance_col]].copy()
    ref_df_weights = mmc_df_weights[(mmc_df_weights[date_col] >= ref_window_start) & (mmc_df_weights[date_col] < ref_window_end)].copy()


    # Normalize columns by mean and std of reference data
    for c in mmc_cols:
        mmc_df[c] = (mmc_df[c] - ref_df[c].mean()) / (ref_df[c].std() + 1e-6)
        # replace mean word in c with min
        c_list = list(c)
        c_list[-1] = 'min'
        c_min = tuple(c_list)
        mmc_df_min[c_min] = (mmc_df_min[c_min] - ref_df[c].mean()) / (ref_df[c].std() + 1e-6)
        # replace mean word in c with max
        c_list = list(c)
        c_list[-1] = 'max'
        c_max = tuple(c_list)
        mmc_df_max[c_max] = (mmc_df_max[c_max] - ref_df[c].mean()) / (ref_df[c].std() + 1e-6)

    mmc_df['mmc'] = mmc_df.mean(axis=1)
    mmc_df_min['mmc'] = mmc_df_min.mean(axis=1)
    mmc_df_max['mmc'] = mmc_df_max.mean(axis=1)


    analysis_utils.create_mmc_plot(mmc_df, date_col, output_dir, title='Unweighted MMC with Range', mmc_min=mmc_df_min, mmc_max=mmc_df_max)
    analysis_utils.create_mmc_plot(mmc_df, date_col, output_dir, title='Unweighted MMC')

    # Weighted MMC
    correlation_matrix = ref_df_weights.corr()

    # To get correlation with the performance column specifically
    performance_correlation = correlation_matrix[performance_col]
    performance_correlation_df = pd.DataFrame(performance_correlation)
    plot_df = performance_correlation_df.reset_index()
    plot_df.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col for col in plot_df.columns]

    plot_df.drop(columns=["level_1___", "level_2___", "level_3___"], inplace=True)
    plot_df.columns = ["Metric", "Avg_AUROC_mean"]

    # drop row where Metric is performance
    plot_df = plot_df[plot_df["Metric"] != "performance"]

    weights_raw = pd.Series(plot_df.Avg_AUROC_mean.values, index=plot_df.Metric).to_dict()
    weights = {metric: abs(weight) for metric, weight in weights_raw.items()}

    #replace nan values with 0 for next step
    weights = {metric: (0 if np.isnan(weight) else weight) for metric, weight in weights.items()}

    # normalize and take negative value -> should that be applied before?
    weights = {metric: (-1) * weight / sum(weights.values()) for metric, weight in weights.items()}

    # save weights for future reference
    with open(os.path.join(output_dir, 'correlation_weights.json'), 'w') as f:
        json.dump(weights, f)

    mmc_df_weighted = mmc_df.copy()
    mmc_df_min_weighted = mmc_df_min.copy()
    mmc_df_max_weighted = mmc_df_max.copy()

    mmc_df_weighted.drop(columns=["mmc"], inplace=True)
    mmc_df_min_weighted.drop(columns=["mmc"], inplace=True)
    mmc_df_max_weighted.drop(columns=["mmc"], inplace=True)


        
    for col in mmc_df_weighted.columns:
        metric_name = [metric for metric in col if metric in weights]
        if metric_name:
            mmc_df_weighted[col] = mmc_df_weighted[col] * weights[metric_name[0]]

        else:
            print(f"Column {col} does not match any metric name in the weights dictionary")
    mmc_df_weighted['mmc'] = mmc_df_weighted.sum(axis=1)

    for col in mmc_df_min_weighted.columns:
        metric_name = [metric for metric in col if metric in weights]
        if metric_name:
            mmc_df_min_weighted[col] = mmc_df_min_weighted[col] * weights[metric_name[0]]

        else:
            print(f"Column {col} does not match any metric name in the weights dictionary")
    mmc_df_min_weighted['mmc'] = mmc_df_min_weighted.sum(axis=1)

    for col in mmc_df_max_weighted.columns:
        metric_name = [metric for metric in col if metric in weights]
        if metric_name:
            mmc_df_max_weighted[col] = mmc_df_max_weighted[col] * weights[metric_name[0]]

        else:
            print(f"Column {col} does not match any metric name in the weights dictionary")
    mmc_df_max_weighted['mmc'] = mmc_df_max_weighted.sum(axis=1)

    analysis_utils.create_mmc_plot(mmc_df_weighted, date_col, output_dir, title='Weighted MMC with Range', mmc_min=mmc_df_min_weighted, mmc_max=mmc_df_max_weighted)
    analysis_utils.create_mmc_plot(mmc_df_weighted, date_col, output_dir, title='Weighted MMC')

    # VAE features alone
    vae_df = df[vae_cols + [date_col]].copy()
    vae_ref_df = vae_df[(vae_df[date_col] >= ref_window_start) & (vae_df[date_col] < ref_window_end)].copy()

    for c in vae_cols:
        vae_df[c] = (vae_df[c] - vae_ref_df[c].mean()) / vae_ref_df[c].std()

    vae_df['mean_vae_distance'] = vae_df[vae_cols].mean(axis=1)

    analysis_utils.create_mmc_plot(vae_df, date_col, output_dir, title='VAE', col_plot='mean_vae_distance')

    # Score features alone
    score_df = df[score_cols + [date_col]].copy()
    score_ref_df = score_df[(score_df[date_col] >= ref_window_start) & (score_df[date_col] < ref_window_end)].copy()

    for c in score_cols:
        score_df[c] = (score_df[c] - score_ref_df[c].mean()) / score_ref_df[c].std()

    score_df['mean_activation_distance'] = score_df[score_cols].mean(axis=1)

    analysis_utils.create_mmc_plot(score_df, date_col, output_dir, title='Score', col_plot='mean_activation_distance')

    # Metadata features alone
    metadata_df = df[metadata_cols + [date_col]].copy()
    metadata_ref_df = metadata_df[(metadata_df[date_col] >= ref_window_start) & (metadata_df[date_col] < ref_window_end)].copy()

    for c in metadata_cols:
        metadata_df[c] = (metadata_df[c] - metadata_ref_df[c].mean()) / metadata_ref_df[c].std()

    metadata_df['mean_metadata_distance'] = metadata_df[metadata_cols].mean(axis=1)

    analysis_utils.create_mmc_plot(metadata_df, date_col, output_dir, title='Metadata', col_plot='mean_metadata_distance')


    # Create Histograms for the drilldown features if present

    dirname = os.path.dirname(drift_csv_path)
    base_path_drilldown = os.path.join(dirname, 'history')

    date_json = os.path.join(base_path_drilldown, '2019-10-10.json')
    with open(date_json, 'r') as f:
        data = json.load(f)

    keys = data['drilldowns'].keys()

    if keys:
        output_dir_hist = Path(os.path.join(output_dir, 'histograms'))
        output_dir_hist.mkdir(parents=True, exist_ok=True)
        for feature in tqdm(keys, desc="Creating Histograms"):
            analysis_utils.plot_hist_feature(feature, basepath = base_path_drilldown, output_dir=output_dir_hist)
    else: 
        print("There is no drilldown data present so no histograms could be created")


if __name__ == "__main__":
    basic_performance_plots()
