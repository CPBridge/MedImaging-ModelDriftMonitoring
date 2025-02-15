from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from pycrumbs import tracked
import click

# Plotting parameters
# Plotting parameters
plt.rcParams['svg.fonttype'] = 'none'
mpl.rcParams['svg.fonttype'] = 'none'
mpl.rcParams['font.size'] = 12
mpl.rcParams['xtick.labelsize'] = 10
mpl.rcParams['ytick.labelsize'] = 10
plt.rcParams.update({
    'svg.fonttype': 'none',
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 16,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.color': '#cccccc'
})

@click.command()
@click.argument('original-mmc-path', type=Path)
@click.argument('recalibrated-mmc-path', type=Path)
@click.argument('output-dir', type=Path)
@click.option('--initial-reference-start', default='2019-10-01', type=str, help='Start date of the initial reference period in YYYY-MM-DD format')
@click.option('--window-length', type=str, default='30D')
@click.option('--recalibration-reference-start', default='2020-10-01', type=str, help='Start date of the recalibration reference period in YYYY-MM-DD format')
@click.option('--recalibration-reference-end', default='2021-01-01', type=str, help='End date of the recalibration reference period in YYYY-MM-DD format')
@tracked(directory_parameter='output_dir')
def main(
    original_mmc_path: Path,
    recalibrated_mmc_path: Path,
    output_dir: Path,
    initial_reference_start: str,
    window_length: str,
    recalibration_reference_start: str,
    recalibration_reference_end: str,
):

    # Convert window length string in days to month float
    try:
        window_length_days = int(window_length.rstrip('D'))
    except ValueError as e:
        raise ValueError(f"{e} Note: only integer days are supported for window length.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        raise

    # add window_length to the ref_window_start to account for the overlap with the period before (1 window length into
    # the reference window is the first day where only days that are actually within the reference window are included)
    initial_reference_start = pd.to_datetime(initial_reference_start) + pd.DateOffset(days=window_length_days)

    df_mmc = pd.read_csv(original_mmc_path)
    df_mmc_recal = pd.read_csv(recalibrated_mmc_path)

    df_mmc['date'] = pd.to_datetime(df_mmc['date'])
    df_mmc_recal['date'] = pd.to_datetime(df_mmc_recal['date'])
    
    df_mmc_recal.set_index('date', inplace=True)
    df_mmc.set_index('date', inplace=True)

    df_mmc_recal_crop = df_mmc_recal.loc[recalibration_reference_end:]
    df_mmc_crop = df_mmc.loc[:recalibration_reference_end]

    df_mmc_combined = pd.concat([df_mmc_crop, df_mmc_recal_crop])

    # Split the data into before and after the recalibration event
    before_recalibration = df_mmc_combined.loc[:recalibration_reference_end]
    after_recalibration = df_mmc_combined.loc[recalibration_reference_end:]

    fig, ax = plt.subplots(figsize=(4.8, 2.4), facecolor='white')

    # Plot the MMC line before and after the recalibration event with different colors
    ax.plot(before_recalibration.index, before_recalibration['mmc'], label='MMC+', color='r')
    ax.plot(after_recalibration.index, after_recalibration['mmc'], label='MMC+ after recalibration', color='purple')

    ax.fill_between(df_mmc_combined.index, df_mmc_combined['mmc_min'], df_mmc_combined['mmc_max'], alpha=0.5, label='MMC Range', color='gray')
    ax.axvline(pd.to_datetime(recalibration_reference_end), color='lightgreen', label='Recalibration Event')  # Ensure color is set to green
    ax.axvspan(pd.to_datetime(recalibration_reference_start), pd.to_datetime(recalibration_reference_end), color='lightgreen', alpha=0.3, label='Recalibration Reference Period')

    # Add vertical line on Junary 1st, 2020 and March 10th
    ax.axvline(x=pd.to_datetime('2020-01-01'), color='darkblue', linestyle='--', linewidth=1)
    ax.axvline(x=pd.to_datetime('2020-03-10'), color='#5088A1', linestyle='--', linewidth=1)
    ax.legend(fontsize=10)  
    ax.set_title('MMC+ Recalibration', fontsize=8)
    ax.set_xlabel('Date', fontsize=8)
    ax.set_ylabel('MMC+', fontsize=8)
    ax.set_xlim(initial_reference_start.date(), df_mmc_combined.index.max().date())
    ax.tick_params(axis='both', which='major', labelsize=6)
    ax.tick_params(axis='x', rotation=45)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # Adjust layout to prevent cut-off labels
    plt.tight_layout()

    fig = plt.gcf()
    fig_width, fig_height = fig.get_size_inches()

    # save the plot in the output directory
    plt.savefig(output_dir / 'weighted_mmc+_with_range_recalibration.png', dpi=600)
    fig.set_size_inches(fig_width, fig_height)

    plt.savefig(output_dir / 'weighted_mmc+_with_range_recalibration.svg', format='svg', bbox_inches='tight')

    fig.set_size_inches(10, 6)  # New size
    
    # Save the plot in the new size
    plt.savefig(output_dir / 'weighted_mmc+_with_range_recalibration_large.png', dpi=600, bbox_inches='tight')
    plt.savefig(output_dir / 'weighted_mmc+_with_range_recalibration_large.svg', format='svg', bbox_inches='tight')
    plt.close()

if __name__ == '__main__':
    main()