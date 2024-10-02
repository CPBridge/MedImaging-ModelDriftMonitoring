from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from pycrumbs import tracked
import click


@click.command()
@click.argument('original-mmc-path', type=Path)
@click.argument('recalibrated-mmc-path', type=Path)
@click.argument('output-dir', type=Path)
@click.option('--recalibration-reference-start', default='2020-10-01', type=str, help='Start date of the recalibration reference period in YYYY-MM-DD format')
@click.option('--recalibration-reference-end', default='2021-01-01', type=str, help='End date of the recalibration reference period in YYYY-MM-DD format')
@tracked(directory_parameter='output-dir')
def main(
    original_mmc_path: Path,
    recalibrated_mmc_path: Path,
    output_dir: Path,
    recalibration_reference_start: str,
    recalibration_reference_end: str,
):
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

    fig, ax = plt.subplots(figsize=(10, 6), facecolor='white')

    # Plot the MMC line before and after the recalibration event with different colors
    ax.plot(before_recalibration.index, before_recalibration['mmc'], label='MMC', color='r')
    ax.plot(after_recalibration.index, after_recalibration['mmc'], label='MMC after recalibration', color='purple')

    ax.fill_between(df_mmc_combined.index, df_mmc_combined['mmc_min'], df_mmc_combined['mmc_max'], alpha=0.5, label='MMC Range', color='gray')
    ax.axvline(pd.to_datetime(recalibration_reference_end), color='lightgreen', label='Recalibration Event')  # Ensure color is set to green
    ax.axvspan(pd.to_datetime(recalibration_reference_start), pd.to_datetime(recalibration_reference_end), color='lightgreen', alpha=0.3, label='Recalibration Reference Period')

    ax.legend()
    ax.set_title('MMC Recalibration')
    ax.set_xlabel('Date')
    ax.set_ylabel('MMC')
    ax.set_xlim(pd.to_datetime(recalibration_reference_start).date(), pd.to_datetime(recalibration_reference_end).date())
    ax.tick_params(axis='x', rotation=45)

    # save the plot in the output directory
    plt.savefig(output_dir / 'weighted_mmc_with_range_recalibration.png', dpi=600)
    plt.close()

if __name__ == '__main__':
    main()