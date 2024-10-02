import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

import click
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pycrumbs import tracked
from tqdm import tqdm

from scripts.analysis import analysis_utils

date_format = mdates.DateFormatter('%Y-%m')
month_locator = mdates.MonthLocator(interval=3)
plt.rcParams['svg.fonttype'] = 'none'



@click.command()
@click.argument('drift-csv-path', type=Path)
@click.argument('output-dir', type=Path)
@click.option('--window-length', type=str, default='30D')
@click.option(
    '--equal-weights', 
    type=bool, 
    default=True, 
    help=(
        'If true, the VAE, score, and metadata metrics are weighted equally. '
        'The correlation weights are only used to weight within the different '
        'metadata values.'
    )
)
@tracked(directory_parameter='output_dir')
def basic_performance_plots(
        drift_csv_path: Path,
        output_dir: Path,
        window_length: str = '30D',
        equal_weights: bool = True
):
    """Makes some basic performance against time plots from a drift CSV."""

    # Setup output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Setup logging
    log_file_path = os.path.join(output_dir, 'plotting_log.log')
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', handlers=[logging.FileHandler(log_file_path), logging.StreamHandler(sys.stdout)])
    logging.info(f"Saving plots to {output_dir}")


    df = pd.read_csv(drift_csv_path, header=[0, 1, 2, 3])
    logging.info(f"Loaded drift CSV from: {drift_csv_path}")

    # load the raw file with all exams in reference window to get the start and end dates
    ref_csv = pd.read_csv(str(drift_csv_path).replace('output.csv', 'ref.csv'))
    logging.info(f"Loaded reference CSV from: {str(drift_csv_path).replace('output.csv', 'ref.csv')}")
    ref_window_start_str = ref_csv["StudyDate"].min()
    ref_window_end_str = ref_csv["StudyDate"].max()
    ref_window_start = datetime.strptime(ref_window_start_str, "%Y-%m-%d")
    ref_window_end = datetime.strptime(ref_window_end_str, "%Y-%m-%d")

    # Convert window length string in days to month float
    try:
        window_length_days = int(window_length.rstrip('D'))
    except ValueError as e:
        raise ValueError(f"{e} Note: only integer days are supported for window length.")
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")
        raise

    # add window_length to the ref_window_start to account for the overlap with the period before (1 window length into
    # the reference window is the first day where only days that are actually within the reference window are included)
    ref_window_start = ref_window_start + pd.DateOffset(days=window_length_days)


    # The date column gets read in with a strange name
    date_col = tuple(f'Unnamed: 0_level_{i}' for i in range(4))
    performance_col = ('performance', 'micro avg', 'auroc', 'mean')


    df[date_col] = pd.to_datetime(df[date_col])


    analysis_utils.create_performance_plots(df, output_dir, ref_window_start, ref_window_end)
    analysis_utils.create_normalized_performance_plots(df, output_dir, ref_window_start, ref_window_end)


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
        if not col[0].startswith('mu') | col[0].startswith('full_mu')
        if not col[0].startswith('activation')
        and col[2] == 'distance'
        and col[3] == 'mean'
    ]

    
    mmc_df = df[mmc_cols + [date_col]].copy()
    mmc_df_min = df[mmc_cols_min + [date_col]].copy()
    mmc_df_max = df[mmc_cols_max + [date_col]].copy()

    ref_df = mmc_df[(mmc_df[date_col] >= ref_window_start) & (mmc_df[date_col] <= ref_window_end)].copy()

    mmc_df_weights = df[mmc_cols + [date_col]+ [performance_col]].copy()
    ref_df_weights = mmc_df_weights[(mmc_df_weights[date_col] >= ref_window_start) & (mmc_df_weights[date_col] <= ref_window_end)].copy()


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

    mmc_df['mmc'] = mmc_df.mean(axis=1, numeric_only=True)
    mmc_df_min['mmc'] = mmc_df_min.mean(axis=1, numeric_only=True)
    mmc_df_max['mmc'] = mmc_df_max.mean(axis=1, numeric_only=True)


    analysis_utils.create_mmc_plot(mmc_df, date_col, output_dir, title='Unweighted MMC with Range', mmc_min=mmc_df_min, mmc_max=mmc_df_max)
    analysis_utils.create_mmc_plot(mmc_df, date_col, output_dir, title='Unweighted MMC')

    #TODO: These plots currently use the unweighted MMC, but are not used in the paper
    analysis_utils.create_normalized_performance_plots_w_mmc(df, output_dir, ref_window_start, ref_window_end, mmc_df)

    if equal_weights:
        #  For runs in the paper, we will weight each drift compoment (metadata, vae, activations) as 1/3. Within the metadata we are still weighting according to correlation with performance
        correlation_matrix = ref_df_weights[metadata_cols + [performance_col]].corr()
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

        # normalize and take negative value. Note: Here the weights should sum to 1/3, as we will be adding the vae and scores each with 1/3 as well
        weights = {metric: (1/3) * weight / (sum(weights.values())) for metric, weight in weights.items()}

        # add weight for vae and score
        weights['full_mu'] = 1/3
        weights['activation'] = 1/3

        logging.info(f'Equal weighting was used. Weights: {weights}')

    else:
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
        weights = {metric: (1) * weight / sum(weights.values()) for metric, weight in weights.items()}

        logging.info(f'Correlation weighting was used for all metrics. Weights: {weights}')


    # save weights for future reference
    with open(os.path.join(output_dir, 'correlation_weights.json'), 'w') as f:
        json.dump(weights, f)

    # Create weighted MMC dataframes
    mmc_df_weighted = mmc_df.copy()
    mmc_df_min_weighted = mmc_df_min.copy()
    mmc_df_max_weighted = mmc_df_max.copy()

    mmc_df_weighted.drop(columns=["mmc"], inplace=True)
    mmc_df_min_weighted.drop(columns=["mmc"], inplace=True)
    mmc_df_max_weighted.drop(columns=["mmc"], inplace=True)


    # Apply weights to the MMC dataframes
    for col in mmc_df_weighted.columns:
        metric_name = [metric for metric in col if metric in weights]
        if metric_name:
            mmc_df_weighted[col] = mmc_df_weighted[col] * weights[metric_name[0]]

        else:
            logging.warning(f"Column {col} does not match any metric name in the weights dictionary")
    mmc_df_weighted['mmc'] = mmc_df_weighted.sum(axis=1)

    for col in mmc_df_min_weighted.columns:
        metric_name = [metric for metric in col if metric in weights]
        if metric_name:
            mmc_df_min_weighted[col] = mmc_df_min_weighted[col] * weights[metric_name[0]]

        else:
            logging.warning(f"Column {col} does not match any metric name in the weights dictionary")
    mmc_df_min_weighted['mmc'] = mmc_df_min_weighted.sum(axis=1)

    for col in mmc_df_max_weighted.columns:
        metric_name = [metric for metric in col if metric in weights]
        if metric_name:
            mmc_df_max_weighted[col] = mmc_df_max_weighted[col] * weights[metric_name[0]]

        else:
            logging.warning(f"Column {col} does not match any metric name in the weights dictionary")
    mmc_df_max_weighted['mmc'] = mmc_df_max_weighted.sum(axis=1)

    # Create plots for weighted MMC
    analysis_utils.create_mmc_plot(mmc_df_weighted, date_col, output_dir, title='Weighted MMC with Range', mmc_min=mmc_df_min_weighted, mmc_max=mmc_df_max_weighted)
    analysis_utils.create_mmc_plot(mmc_df_weighted, date_col, output_dir, title='Weighted MMC')
    analysis_utils.create_joint_scatter_density_plots(df, output_dir, ref_window_start, ref_window_end, mmc_df_weighted)

    # Create plots for VAE features alone, using the weighted values
    vae_df = mmc_df_weighted[vae_cols + [date_col]].copy()
    vae_df['mean_vae_distance'] = vae_df[vae_cols].sum(axis=1)

    analysis_utils.create_mmc_plot(vae_df, date_col, output_dir, title='VAE', col_plot='mean_vae_distance')

    # Create plots for activation features alone, using the weighted values
    score_df = mmc_df_weighted[score_cols + [date_col]].copy()
    score_df['mean_activation_distance'] = score_df[score_cols].sum(axis=1)

    analysis_utils.create_mmc_plot(score_df, date_col, output_dir, title='Score', col_plot='mean_activation_distance')

    # Create plots for Metadata features alone, using the weighted values
    metadata_df = mmc_df_weighted[metadata_cols + [date_col]].copy()
    metadata_df['mean_metadata_distance'] = metadata_df[metadata_cols].sum(axis=1)

    analysis_utils.create_mmc_plot(metadata_df, date_col, output_dir, title='Metadata', col_plot='mean_metadata_distance')


    # Create Histograms for the drilldown features if present
    dirname = os.path.dirname(drift_csv_path)
    base_path_drilldown = os.path.join(dirname, 'history')

    #Load one example date json to get the keys, needs to be adjusted if using a different dataset
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
        logging.info("There is no drilldown data present so no histograms could be created")


if __name__ == "__main__":
    basic_performance_plots()
