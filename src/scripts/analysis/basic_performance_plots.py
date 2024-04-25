from pathlib import Path
import pandas as pd

import sys
import os
sys.path.append('/autofs/homes/005/fd881/repos/MedImaging-ModelDriftMonitoring/')

import click
import json
from pycrumbs import tracked

from src.model_drift.data import mgb_data
import numpy as np
import matplotlib.dates as mdates
import matplotlib.pyplot as plt

date_format = mdates.DateFormatter('%Y-%m')
month_locator = mdates.MonthLocator(interval=3)
plt.rcParams['svg.fonttype'] = 'none'

def create_performance_plots(df: pd.DataFrame, output_dir: Path):

    plt.style.use('ggplot')
    date_col = tuple(f'Unnamed: 0_level_{i}' for i in range(4))
    target_names = tuple(mgb_data.LABEL_GROUPINGS)
    target_names = target_names + ('micro avg', 'macro avg')
    num_cols = 2
    num_rows = (len(target_names) + num_cols - 1) // num_cols  # This ensures you have enough rows

    # Create a figure and an array of axes
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(10, num_rows * 5))
    axs = axs.flatten()  # Flatten the array of axes for easier indexing

    # Loop over each label name and plot data
    for i, name in enumerate(target_names):
        # Extract the required series from the DataFrame
        auroc_series = df[('performance', name, 'auroc', 'mean')]
        f1_score_series = df[('performance', name, 'f1-score', 'mean')]
        date_series = df[date_col]

        # Plot each metric on its subplot
        axs[i].plot(date_series, auroc_series, label='AUROC')
        axs[i].plot(date_series, f1_score_series, label='F1-Score')
        axs[i].set_title(name) 
        axs[i].legend()
        axs[i].grid(True)
        axs[i].set_xlim(pd.to_datetime('2019-11-01').date(), pd.to_datetime('2021-07-01').date())
        axs[i].tick_params(axis='x', rotation=45)
        axs[i].set_ylim(0, 1)
        axs[i].yaxis.set_major_locator(plt.MultipleLocator(0.1))

        # Save each plot individually
        fig2, ax2 = plt.subplots(figsize=(10, 5))
        ax2.plot(date_series, auroc_series, label='AUROC')
        ax2.plot(date_series, f1_score_series, label='F1-Score')
        ax2.set_title(name)
        ax2.legend()
        ax2.grid(True)
        ax2.set_xlim(pd.to_datetime('2019-11-01').date(), pd.to_datetime('2021-07-01').date())
        ax2.tick_params(axis='x', rotation=45)
        ax2.set_ylim(0, 1)
        ax2.yaxis.set_major_locator(plt.MultipleLocator(0.1))
        plt.tight_layout()
        

        fig2.savefig(os.path.join(output_dir, f'performance_{name}.png'))
        fig2.savefig(os.path.join(output_dir, f'performance_{name}.svg'), format='svg', bbox_inches='tight')

        plt.close(fig2) 

    # Hide unused axes if there are any
    for j in range(i + 1, num_cols * num_rows):
        axs[j].axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'performance_combined.png'))
    plt.savefig(os.path.join(output_dir, 'performance_combined.svg'), format='svg', bbox_inches='tight')  
    plt.close()

def create_mmc_plot(df, date_col, output_dir, title, col_plot = 'MMC'):
    plt.style.use('ggplot')


    # Plot setup
    fig, ax = plt.subplots(figsize=(10, 6))  
    ax.plot(df[date_col], df[col_plot.lower()], label=col_plot)  
    ax.set_title(title)  
    ax.set_xlabel('Date')  
    ax.set_ylabel(col_plot)  
    ax.grid(True)  
    ax.legend()  
    ax.set_xlim(pd.to_datetime('2019-11-01').date(), pd.to_datetime('2021-07-01').date())  
    ax.tick_params(axis='x', rotation=45)  

    # Save the plot
    plt.savefig(output_dir / f'{title.lower().replace(" ", "_")}.png')
    plt.savefig(output_dir / f'{title.lower().replace(" ", "_")}.svg', format='svg', bbox_inches='tight')
    plt.close()  



@click.command()
@click.argument('drift-csv-path', type=Path)
@click.argument('output-dir', type=Path)
@tracked(directory_parameter='output_dir')
def basic_performance_plots(
        drift_csv_path: Path,
        output_dir: Path,
):
    """Makes some basic performance against time plots from a drift CSV."""
    df = pd.read_csv(drift_csv_path, header=[0, 1, 2, 3])

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # The date column gets read in with a stupid name
    date_col = tuple(f'Unnamed: 0_level_{i}' for i in range(4))
    performance_col = ('performance', 'micro avg', 'auroc', 'mean')

    df[date_col] = pd.to_datetime(df[date_col])


    create_performance_plots(df, output_dir)

    # Unweighted MMC
    mmc_cols = [
        col for col in df.columns
        if not col[0].startswith('performance')
        and col[2] == 'distance'
        and col[3] == 'mean'
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
    ref_df = mmc_df[mmc_df[date_col] < mgb_data.VAL_DATE_END].copy()

    mmc_df_weights = df[mmc_cols + [date_col]+ [performance_col]].copy()
    ref_df_weights = mmc_df_weights[mmc_df_weights[date_col] < mgb_data.VAL_DATE_END].copy()



    # Normalize columns by mean and std of reference data
    for c in mmc_cols:
        mmc_df[c] = (mmc_df[c] - ref_df[c].mean()) / ref_df[c].std()

    mmc_df['mmc'] = mmc_df.mean(axis=1)


    create_mmc_plot(mmc_df, date_col, output_dir, title='Unweighted MMC')

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

    mmc_df_weighted.drop(columns=["mmc"], inplace=True)


    for col in mmc_df_weighted.columns:
        metric_name = [metric for metric in col if metric in weights]
        if metric_name:
            mmc_df_weighted[col] = mmc_df_weighted[col] * weights[metric_name[0]]

        else:
            print(f"Column {col} does not match any metric name in the weights dictionary")
    mmc_df_weighted['mmc'] = mmc_df_weighted.sum(axis=1)

    create_mmc_plot(mmc_df_weighted, date_col, output_dir, title='Weighted MMC')

    # VAE features alone
    vae_df = df[vae_cols + [date_col]].copy()
    vae_df['mean_vae_distance'] = vae_df[vae_cols].mean(axis=1)

    create_mmc_plot(vae_df, date_col, output_dir, title='VAE', col_plot='mean_vae_distance')

    # Score features alone
    score_df = df[score_cols + [date_col]].copy()
    score_df['mean_activation_distance'] = score_df[score_cols].mean(axis=1)

    create_mmc_plot(score_df, date_col, output_dir, title='Score', col_plot='mean_activation_distance')

    # Metadata features alone
    metadata_df = df[metadata_cols + [date_col]].copy()
    metadata_ref_df = metadata_df[metadata_df[date_col] < mgb_data.VAL_DATE_END].copy()

    for c in metadata_cols:
        metadata_df[c] = (metadata_df[c] - metadata_ref_df[c].mean()) / metadata_ref_df[c].std()

    metadata_df['mean_metadata_distance'] = metadata_df[metadata_cols].mean(axis=1)

    create_mmc_plot(metadata_df, date_col, output_dir, title='Metadata', col_plot='mean_metadata_distance')

if __name__ == "__main__":
    basic_performance_plots()
