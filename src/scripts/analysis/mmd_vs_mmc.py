import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import click
from pathlib import Path

# Setup plotting parameters
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
@click.argument('mmd_input_path', type=click.Path(exists=True))
@click.argument('mmc_input_path', type=click.Path(exists=True))
@click.argument('output_dir', type=click.Path())
@click.option('--start_date', default='2019-11-01', help='Start date for x-axis')
@click.option('--end_date', default='2021-07-01', help='End date for x-axis')
def main(mmd_input_path, mmc_input_path, output_dir, start_date, end_date):
    save_path = Path(output_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    # MMD setup
    df_mmd = pd.read_csv(mmd_input_path, header=[0, 1, 2])
    date_col = tuple(f'Unnamed: 0_level_{i}' for i in range(3))
    performance_col = ('performance', 'micro avg', 'auroc', 'mean')

    df_mmd[date_col] = pd.to_datetime(df_mmd[date_col])

    plt.figure(figsize=(12, 6))
    plt.plot(df_mmd[date_col], df_mmd[('mu_activation_combined', 'mmd', 'pval')], label='MMD p-value')
    plt.xticks(rotation=45)
    plt.title('MMD p-value over time')
    plt.xlim(pd.Timestamp(start_date), pd.Timestamp(end_date))
    plt.axhline(0.05, color='r', linestyle='--')
    plt.yticks([0, 0.05, 0.1, 0.2, 0.5, 1])
    plt.ylabel('p-value')
    plt.tight_layout()

    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)
    plt.legend()
    
    plt.savefig(save_path / "mmd_pvalue_over_time.png", dpi=600, bbox_inches='tight')
    plt.savefig(save_path / "mmd_pvalue_over_time.svg", format='svg', bbox_inches='tight')

    plot_data_mmd = pd.DataFrame({
        'date': df_mmd[date_col].dt.strftime('%Y-%m-%d'),
        'mmd_pvalue': df_mmd[('mu_activation_combined', 'mmd', 'pval')]
    })
    plot_data_mmd.to_csv(save_path / "mmd_pvalue_over_time.csv", index=False)

    # MMC setup
    plot_data_mmc = pd.read_csv(mmc_input_path)
    plot_data_mmc['date'] = pd.to_datetime(plot_data_mmc['date'])
    plot_data_mmd['date'] = pd.to_datetime(plot_data_mmd['date'])

    # Plot MMD p-value and MMC
    fig, ax1 = plt.subplots(figsize=(12, 6))

    color = 'tab:blue'
    ax1.set_xlabel('Date')
    ax1.set_ylabel('MMD p-value', color=color)
    ax1.plot(plot_data_mmd['date'], plot_data_mmd['mmd_pvalue'], color=color, label='MMD p-value')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_ylim(-0.05, 1)
    ax1.set_yticks([0, 0.05, 0.1, 0.2, 0.5, 1])

    ax2 = ax1.twinx()
    color = 'gray'
    ax2.set_ylabel('MMC', color=color)
    ax2.plot(plot_data_mmc['date'], plot_data_mmc['mmc'], color=color, label='MMC+')
    # Add vertical line on February 1st, 2020
    ax2.axvline(x=pd.to_datetime('2020-02-01'), color='darkblue', linestyle='--', linewidth=1, label='First window containing only test data')
    ax2.tick_params(axis='y', labelcolor=color)
    ax1.tick_params(axis='x', rotation=45)

    plt.xlim(pd.Timestamp(start_date), pd.Timestamp(end_date))

    ax1.axhline(0.05, color='r', linestyle='--', label='p-value threshold')

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

    ax1.spines['top'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    plt.title('MMD p-value and MMC over time')
    plt.tight_layout()

    plt.savefig(save_path / "mmd_vs_mmc_over_time.png", dpi=600, bbox_inches='tight')
    plt.savefig(save_path / "mmd_vs_mmc_over_time.svg", format='svg', bbox_inches='tight')
    plt.close()

    plot_data_combined = pd.merge(plot_data_mmd, plot_data_mmc[['date', 'mmc']], on='date', how='outer')
    plot_data_combined.columns = ['date', 'mmd_pvalue', 'mmc']
    plot_data_combined.to_csv(save_path / "mmd_vs_mmc_over_time.csv", index=False)

if __name__ == '__main__':
    main()