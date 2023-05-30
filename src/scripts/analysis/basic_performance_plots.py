from pathlib import Path
import pandas as pd

import click
import plotnine
from pycrumbs import tracked

from model_drift.data import mgb_data


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

    # The date column gets read in with a stupid name
    date_col = tuple(f'Unnamed: 0_level_{i}' for i in range(4))
    df[date_col] = pd.to_datetime(df[date_col])

    for finding in mgb_data.LABEL_GROUPINGS.keys():
        for metric in ['auroc', 'f1-score']:
            y_col = ('performance', finding, metric, 'mean')
            p = (
                plotnine.ggplot(df, plotnine.aes(df[date_col], y=df[y_col])) +
                plotnine.geom_line() +
                plotnine.ggtitle(f'{finding} ({metric})') +
                plotnine.theme(figure_size=(10, 6)) +
                plotnine.labs(x='Date', y=metric)
            )
            finding_str = finding.lower().replace(' ', '_')
            p.save(output_dir / f'{finding_str}_{metric}.png')

    # Unweighted MMC
    mmc_cols = [
        col for col in df.columns
        if not col[0].startswith('performance')
        and col[2] == 'distance'
        and col[3] == 'mean'
    ]

    mmc_df = df[mmc_cols + [date_col]].copy()
    ref_df = mmc_df[mmc_df[date_col] < mgb_data.VAL_DATE_END].copy()

    # Normalize columns by mean and std of reference data
    for c in mmc_cols:
        mmc_df[c] = (mmc_df[c] - ref_df[c].mean()) / ref_df[c].std()

    mmc_df['mmc'] = mmc_df.mean(axis=1)

    p = (
        plotnine.ggplot(
            mmc_df,
            plotnine.aes(mmc_df[date_col], y=mmc_df['mmc'])
        )
        + plotnine.geom_line()
        + plotnine.ggtitle(f'Unweighted MMC')
        + plotnine.theme(figure_size=(10, 6))
        + plotnine.labs(x='Date', y='mmc')
    )
    finding_str = finding.lower().replace(' ', '_')
    p.save(output_dir / f'unweighted_mmc.png')


if __name__ == "__main__":
    basic_performance_plots()
