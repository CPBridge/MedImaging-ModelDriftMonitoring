import json 
import os
from pathlib import Path
from datetime import datetime
import logging

import pandas as pd
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib as mpl
import plotly.graph_objs as go
import seaborn as sns

from model_drift.data import mgb_data

date_format = mdates.DateFormatter('%Y-%m')
month_locator = mdates.MonthLocator(interval=3)

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

logger = logging.getLogger(__name__)

def create_performance_plots(df: pd.DataFrame, output_dir: Path, ref_start: str, ref_end: str):

    date_col = tuple(f'Unnamed: 0_level_{i}' for i in range(4))
    target_names = tuple(mgb_data.LABEL_GROUPINGS)
    target_names = target_names + ('micro avg', 'macro avg')
    num_cols = 2
    num_rows = (len(target_names) + num_cols - 1) // num_cols 

    # Create a figure and flatten axs for indexing
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(10, num_rows * 5))
    axs = axs.flatten() 

    # Define the date range to limit to reference and monitoring period
    start_date = pd.to_datetime('2019-11-01')
    end_date = pd.to_datetime('2021-07-01')

    # Convert date_col to datetime and filter the DataFrame
    df['date'] = pd.to_datetime(df[date_col])
    df_filtered = df[(df['date'] >= start_date) & (df['date'] <= end_date)]

    # Loop over each label name and plot data
    for i, name in enumerate(target_names):
        # Extract the required series from the DataFrame
        if not isinstance(df_filtered.index, pd.DatetimeIndex):
            df_filtered.set_index(pd.to_datetime(df_filtered[date_col]), inplace=True)
        auroc_series = df_filtered[('performance', name, 'auroc', 'mean')]
        f1_score_series = df_filtered[('performance', name, 'f1-score', 'mean')]
        date_series = df_filtered[date_col]

        # Calculate the average AUROC during the monitoring period
        avg_auroc = df_filtered[('performance', name, 'auroc', 'mean')].mean()        

        _normalized_auroc, avg_auroc_ref, std_auroc_ref = normalize_series(auroc_series, ref_start, ref_end, return_mean_std=True)
        
        std_3_upper = avg_auroc_ref + 3 * std_auroc_ref
        std_3_lower = avg_auroc_ref - 3 * std_auroc_ref


        # Plot each metric on its subplot
        mask = (date_series >= ref_start) & (date_series <= ref_end)
        axs[i].plot(date_series[~mask], auroc_series[~mask], label='AUROC', color='blue')
        axs[i].plot(date_series[mask], auroc_series[mask], label='AUROC during reference period', color='cornflowerblue')
        axs[i].axhline(y=avg_auroc_ref, color='gray', linestyle='--', label='Avg AUROC during Reference Window')
        axs[i].fill_between(date_series, std_3_upper, std_3_lower, color='gray', alpha=0.35, label='3 Std Range of Reference Window')
        #axs[i].plot(date_series, f1_score_series, label='F1-Score')
        axs[i].set_title(name) 
        axs[i].legend()
        #axs[i].grid(False)
        axs[i].set_xlim(pd.to_datetime('2019-11-01').date(), pd.to_datetime('2021-07-01').date())
        axs[i].tick_params(axis='x', rotation=45)
        axs[i].set_ylim(0.7, 1)
        axs[i].spines['top'].set_visible(False)
        axs[i].spines['right'].set_visible(False)
        axs[i].yaxis.set_major_locator(plt.MultipleLocator(0.1))

        # Save each plot individually
        fig2, ax2 = plt.subplots(figsize=(10, 5))
        mask = (date_series >= ref_start) & (date_series <= ref_end)
        ax2.plot(date_series[~mask], auroc_series[~mask], label='AUROC', color='blue')
        ax2.plot(date_series[mask], auroc_series[mask], label='AUROC during reference period', color='cornflowerblue')
        ax2.axhline(y=avg_auroc_ref, color='gray', linestyle='--', label='Avg AUROC during Reference Window')
        ax2.fill_between(date_series, std_3_upper, std_3_lower, color='gray', alpha=0.35, label='3 Std Range of Reference Window')   
        # Add vertical line on Junary 1st, 2020 and March 10th
        ax2.axvline(x=pd.to_datetime('2020-01-01'), color='darkblue', linestyle='--', linewidth=1)
        ax2.axvline(x=pd.to_datetime('2020-03-10'), color='#5088A1', linestyle='--', linewidth=1)     
        ax2.set_title(name)
        ax2.legend()
        #ax2.grid(False)
        ax2.set_xlim(pd.to_datetime('2019-11-01').date(), pd.to_datetime('2021-07-01').date())
        ax2.tick_params(axis='x', rotation=45)
        ax2.set_ylim(0.7, 1)
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        ax2.yaxis.set_major_locator(plt.MultipleLocator(0.1))
        plt.tight_layout()
        

        fig2.savefig(os.path.join(output_dir, f'performance_{name}.png'), dpi=600)
        fig2.savefig(os.path.join(output_dir, f'performance_{name}.svg'), format='svg', bbox_inches='tight')
        # Save performance data as CSV
        performance_data = pd.DataFrame({
            'date': date_series,
            'auroc': auroc_series,
            'avg_auroc_ref': avg_auroc_ref,
            'std_3_upper': std_3_upper,
            'std_3_lower': std_3_lower
        })
        performance_data.to_csv(os.path.join(output_dir, f'performance_{name}.csv'), index=False)

        plt.close(fig2) 

    # Hide unused axes if there are any
    for j in range(i + 1, num_cols * num_rows):
        axs[j].axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'performance_combined.png'))
    plt.savefig(os.path.join(output_dir, 'performance_combined.svg'), format='svg', bbox_inches='tight')  
    plt.close()

def normalize_series(data, ref_start, ref_end, return_mean_std=False):
    """ Normalize the series data based on reference period mean and std """
    reference_period = data.loc[ref_start:ref_end]
    mean = reference_period.mean()
    std = reference_period.std()
    norm_result = (data - mean) / std

    if return_mean_std:
        return norm_result, mean, std
    else:
        return norm_result

def create_normalized_performance_plots(
    df: pd.DataFrame, output_dir: Path, ref_start: str, ref_end: str, 
    plot_start_date=pd.to_datetime('2019-11-01'), plot_end_date=pd.to_datetime('2021-07-01')
):
    date_col = tuple(f'Unnamed: 0_level_{i}' for i in range(4))
    target_names = tuple(mgb_data.LABEL_GROUPINGS) + ('micro avg', 'macro avg')
    num_cols = 2
    num_rows = (len(target_names) + num_cols - 1) // num_cols

    fig, axs = plt.subplots(num_rows, num_cols, figsize=(10, num_rows * 5))
    axs = axs.flatten()

    for i, name in enumerate(target_names):
        # Extract and normalize the series data
        if not isinstance(df.index, pd.DatetimeIndex):
            df.set_index(pd.to_datetime(df[date_col]), inplace=True)
        auroc_series = df[('performance', name, 'auroc', 'mean')]
        #f1_score_series = df[('performance', name, 'f1-score', 'mean')]
        date_series = df[date_col]

        normalized_auroc = normalize_series(auroc_series, ref_start, ref_end)
        #normalized_f1_score = normalize_series(f1_score_series, ref_start, ref_end)

        axs[i].plot(date_series, normalized_auroc, label='Normalized AUROC')
        #axs[i].plot(date_series, normalized_f1_score, label='Normalized F1-Score')
        axs[i].axhline(y=0, color='grey', linestyle='--', linewidth=1)  #
        axs[i].set_title(name)
        axs[i].legend()
        #axs[i].grid(True)
        axs[i].tick_params(axis='x', rotation=45)
        axs[i].set_xlim(plot_start_date.date(), plot_end_date.date())
        axs[i].set_ylim(-3, 3)  # Adjust the y-axis limits for normalized data
        axs[i].yaxis.set_major_locator(plt.MultipleLocator(0.5))

    for j in range(i + 1, num_cols * num_rows):
        axs[j].axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'normalized_performance_combined.png'))
    plt.savefig(os.path.join(output_dir, 'normalized_performance_combined.svg'), format='svg', bbox_inches='tight')
    plt.close()

def create_normalized_performance_plots_w_mmc(
    df: pd.DataFrame, output_dir: Path, ref_start: str, ref_end: str, 
    mmc_df: pd.DataFrame, plot_start_date=pd.to_datetime('2019-11-01'), 
    plot_end_date=pd.to_datetime('2021-07-01')
):
    date_col = tuple(f'Unnamed: 0_level_{i}' for i in range(4))
    target_names = tuple(mgb_data.LABEL_GROUPINGS) + ('micro avg', 'macro avg')
    num_cols = 2
    num_rows = (len(target_names) + num_cols - 1) // num_cols

    fig, axs = plt.subplots(num_rows, num_cols, figsize=(10, num_rows * 5))
    axs = axs.flatten()

    for i, name in enumerate(target_names):
        # Extract and normalize the series data
        if not isinstance(df.index, pd.DatetimeIndex):
            df.set_index(pd.to_datetime(df[date_col]), inplace=True)
        auroc_series = df[('performance', name, 'auroc', 'mean')]
        date_series = df.index

        normalized_auroc = normalize_series(auroc_series, ref_start, ref_end)

        ax1 = axs[i]
        ax2 = ax1.twinx()  # Create a second y-axis

        ax1.plot(date_series, normalized_auroc, label='Normalized AUROC', color='blue')
        ax1.axhline(y=0, color='grey', linestyle='--', linewidth=1)
        ax1.set_title(name)
        #ax1.grid(True)
        ax1.tick_params(axis='x', rotation=45)
        ax1.set_xlim(plot_start_date.date(), plot_end_date.date())
        ax1.set_ylim(-3, 3)
        ax1.yaxis.set_major_locator(plt.MultipleLocator(0.5))
        ax1.set_ylabel('Normalized AUROC')

        ax2.plot(date_series, mmc_df['mmc'], label='MMC', linestyle='--', color='orange')
        ax2.set_ylim(min(mmc_df['mmc']), max(mmc_df['mmc']))
        ax2.set_ylabel('MMC')

        # Combine legends
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

    for j in range(i + 1, num_cols * num_rows):
        axs[j].axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'normalized_performance_with_mmc_combined.png'))
    plt.savefig(os.path.join(output_dir, 'normalized_performance_with_mmc_combined.svg'), format='svg', bbox_inches='tight')
    plt.close()

def create_joint_scatter_density_plots(df: pd.DataFrame, output_dir: Path, ref_start: str, ref_end: str, mmc_df: pd.DataFrame):
    date_col = tuple(f'Unnamed: 0_level_{i}' for i in range(4))
    target_names = tuple(mgb_data.LABEL_GROUPINGS) + ('micro avg', 'macro avg')

    plt.rc('axes', titlesize=16)     # Title font size
    plt.rc('axes', labelsize=12)      # Axis label font size
    plt.rc('xtick', labelsize=5)     # X-tick label font size
    plt.rc('ytick', labelsize=5)     # Y-tick label font size

    num_plots = len(target_names)
    num_cols = 2
    num_rows = (num_plots + num_cols - 1) // num_cols

    fig, axs = plt.subplots(num_rows, num_cols, figsize=(10, num_rows * 5))
    axs = axs.flatten()

    # Initialize the final combined data with the MMC column
    final_combined_data = mmc_df.loc[ref_end:, ['mmc']]

    for i, name in enumerate(target_names):
        if not isinstance(df.index, pd.DatetimeIndex):
            df.set_index(pd.to_datetime(df[date_col]), inplace=True)
        
        auroc_series = df[('performance', name, 'auroc', 'mean')]
        normalized_auroc = normalize_series(auroc_series, ref_start, ref_end)

        # Exclude all the dates before ref_end
        normalized_auroc = normalized_auroc.loc[ref_end:]

        # Add the normalized AUROC for this target to the final combined data
        final_combined_data[f'normalized_auroc_{name}'] = normalized_auroc

        # Create combined_data for this iteration (for plotting)
        combined_data = pd.concat([final_combined_data['mmc'], normalized_auroc], axis=1)
        combined_data.columns = ['mmc', f'normalized_auroc_{name}']

        combined_data.index = pd.to_datetime(combined_data.index)
        combined_data['date_category'] = combined_data.index >= pd.Timestamp('2020-03-10')

        g = sns.jointplot(
            data=combined_data,
            x='mmc', y=f'normalized_auroc_{name}', hue='date_category',
            kind="scatter",
            legend=False,
            joint_kws={'edgecolor':'w', 'linewidth':0.2},
        )
        #g.ax_joint.set_title(name, loc='left')
        g.ax_joint.set_xlabel('MMC')
        g.ax_joint.set_ylabel('Normalized AUROC')

        # Add horizontal lines at +3 and -3
        g.ax_joint.axhline(y=3, color='gray', linestyle='--')
        g.ax_joint.axhline(y=-3, color='gray', linestyle='--')
 
        # Add vertical line at x=10
        g.ax_joint.axvline(x=10, color='gray', linestyle='-.')

        if False: #name == 'cardiomegaly':
            y_min, y_max = df[auroc_col].min(), df[auroc_col].max()
            max_limit = max(abs(y_min), abs(y_max))
            g.ax_joint.set_ylim([-max_limit, max_limit])
        else:
            g.ax_joint.set_ylim([-16, 16])

        handles, labels = g.ax_joint.get_legend_handles_labels()
        labels = ['Before March 10, 2020', 'After March 10, 2020']

        if i == 0:
            g.ax_joint.legend(labels, title='Date', loc='lower right')

        # set grid lines false
        g.ax_joint.grid(False)

        g.fig.suptitle(name, x=0.5, y=0.95, ha='center', fontsize=16)
        g.fig.subplots_adjust(top=0.9)  # Adjust to make room for the title

        g.savefig(os.path.join(output_dir, f'{name}_KDE.svg'))
        g.savefig(os.path.join(output_dir, f'{name}_KDE.png'), dpi=600)


    # Save the final combined data to CSV
    final_combined_data.to_csv(os.path.join(output_dir, 'weighted_mmc_vs_performance_combined.csv'))

    # Create Table for the out of spec MMC values
    proportions = {}
    for name in target_names:
        df_prop = final_combined_data[[f'normalized_auroc_{name}', 'mmc']]
        
        # select rows where mmc is smaller than 10
        df_prop_smaller = df_prop[df_prop['mmc'] < 10]
        # calculate proportion where normalized_auroc is within [-3, 3] for mmc < 10
        if len(df_prop_smaller) > 0:
            proportion_smaller = len(df_prop_smaller[(df_prop_smaller[f'normalized_auroc_{name}'] > -3) & (df_prop_smaller[f'normalized_auroc_{name}'] < 3)]) / len(df_prop_smaller)
        else:
            proportion_smaller = 0

        # select rows where mmc is larger than or equal to 10
        df_prop_larger = df_prop[df_prop['mmc'] >= 10]
        # calculate proportion where normalized_auroc is within [-3, 3] for mmc >= 10
        if len(df_prop_larger) > 0:
            proportion_larger = len(df_prop_larger[(df_prop_larger[f'normalized_auroc_{name}'] > -3) & (df_prop_larger[f'normalized_auroc_{name}'] < 3)]) / len(df_prop_larger)
        else:
            proportion_larger = 0
        proportions[name] = (proportion_smaller, proportion_larger)

    df_proportions = pd.DataFrame(proportions).T

    # rename columns
    df_proportions.columns = ['Proportion MMC < 10', 'Proportion MMC >= 10']
    df_proportions = df_proportions.round(3)
    df_proportions.to_csv(os.path.join(output_dir, 'weighted_mmc_vs_performance_combined_proportions.csv'))



def create_mmc_plot(df, date_col, output_dir, title, col_plot='MMC', mmc_min=None, 
                    mmc_max=None, plot_start_date=pd.to_datetime('2019-11-01'), 
                    plot_end_date=pd.to_datetime('2021-07-01')):
        
    col_plot_display = 'MMC+' if col_plot.lower() == 'mmc' else col_plot

    # Create complete date range and merge to introduce NaN values for missing dates
    date_range = pd.date_range(start=plot_start_date, end=plot_end_date, freq='D')
    date_df = pd.DataFrame({'date': date_range})

    df['date'] = df[date_col]
    df = df.merge(date_df, on='date', how='right')
    df.sort_values(by='date', inplace=True)

    # Check if there are NaN values in the 'mmc' columns and count them
    nan_count = df[col_plot.lower()].isna().sum()
    if nan_count > 0:
        logger.warning(f"Warning: There are {nan_count} NaN values in the '{col_plot.lower()}' column.")
    
        # Interpolate NaN values only if there are less than 3 days of gap
        df[col_plot.lower()] = df[col_plot.lower()].interpolate(method='linear', limit=2, limit_direction='both')
        
        # Check if there are still NaN values after interpolation
        remaining_nan = df[col_plot.lower()].isna().sum()
        if remaining_nan > 0:
            logger.warning(f"Warning: There are still {remaining_nan} NaN values in the '{col_plot.lower()}' column after interpolation.")
            logger.warning("These NaN values represent gaps of 3 or more days and were not interpolated.")

    # Create the figure with the original size
    fig, ax = plt.subplots(figsize=(4.8, 2.4), facecolor='white')  

    ax.plot(df['date'], df[col_plot.lower()], label=col_plot_display, color='r')  
    
    if mmc_min is not None and mmc_max is not None:

        mmc_min['date'] = mmc_min[date_col]
        mmc_max['date'] = mmc_max[date_col]
        
        mmc_min = mmc_min.merge(date_df, on='date', how='right')
        mmc_max = mmc_max.merge(date_df, on='date', how='right')

        mmc_min.sort_values(by='date', inplace=True)
        mmc_max.sort_values(by='date', inplace=True)

        # Interpolate NaN values for mmc_min and mmc_max if provided
        nan_count_min = mmc_min['mmc'].isna().sum()
        if nan_count_min > 0:
            logger.warning(f"Warning: There are {nan_count_min} NaN values in the 'mmc' column of mmc_min.")
            mmc_min['mmc'] = mmc_min['mmc'].interpolate(method='linear', limit=2, limit_direction='both')
            remaining_nan_min = mmc_min['mmc'].isna().sum()
            if remaining_nan_min > 0:
                logger.warning(f"Warning: There are still {remaining_nan_min} NaN values in the 'mmc' column of mmc_min after interpolation.")
                logger.warning("These NaN values represent gaps of 3 or more days and were not interpolated.")

        nan_count_max = mmc_max['mmc'].isna().sum()
        if nan_count_max > 0:
            logger.warning(f"Warning: There are {nan_count_max} NaN values in the 'mmc' column of mmc_max.")
            mmc_max['mmc'] = mmc_max['mmc'].interpolate(method='linear', limit=2, limit_direction='both')
            remaining_nan_max = mmc_max['mmc'].isna().sum()
            if remaining_nan_max > 0:
                logger.warning(f"Warning: There are still {remaining_nan_max} NaN values in the 'mmc' column of mmc_max after interpolation.")
                logger.warning("These NaN values represent gaps of 3 or more days and were not interpolated.")


        ax.fill_between(df['date'], mmc_min['mmc'], mmc_max['mmc'], 
                        alpha=0.5, label='MMC+ Range', color='gray')
                    
    # Add vertical line on Junary 1st, 2020 and March 10th
    ax.axvline(x=pd.to_datetime('2020-01-01'), color='darkblue', linestyle='--', linewidth=1)
    ax.axvline(x=pd.to_datetime('2020-03-10'), color='#5088A1', linestyle='--', linewidth=1)

    ax.set_title(title, fontsize=8)  
    ax.set_xlabel('Date', fontsize=8) 
    ax.set_ylabel(col_plot_display, fontsize=8)  
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(fontsize=10)  
    ax.set_xlim(plot_start_date.date(), plot_end_date.date())  

    # TODO: Only for ER and WAC2
    #ax.set_ylim(-10, 85)
    ## Set y-axis ticks
    #ax.set_yticks([0, 10, 20, 30, 40, 50, 60, 70, 80])
    #ax.set_yticklabels(['0', '10', '20', '30', '40', '50', '60', '70', '80'], fontsize=6)

    # Standardize tick sizes and rotation
    ax.tick_params(axis='both', which='major', labelsize=6)
    ax.tick_params(axis='x', rotation=45)

    # Adjust layout to prevent cut-off labels
    plt.tight_layout()

    fig = plt.gcf()
    fig_width, fig_height = fig.get_size_inches()

    # Save the plot
    plt.savefig(output_dir / f'{title.lower().replace(" ", "_")}.png', dpi=600, bbox_inches='tight')
    fig.set_size_inches(fig_width, fig_height)
    plt.savefig(output_dir / f'{title.lower().replace(" ", "_")}.svg', format='svg', bbox_inches='tight')
    
    # Resize the figure
    fig.set_size_inches(10, 6)  # New size
    
    # Save the plot in the new size
    plt.savefig(output_dir / f'{title.lower().replace(" ", "_")}_large.png', dpi=600, bbox_inches='tight')
    plt.savefig(output_dir / f'{title.lower().replace(" ", "_")}_large.svg', format='svg', bbox_inches='tight')

    # Save the plot data as a CSV
    plot_data = pd.DataFrame({
        'date': df[date_col],
        col_plot.lower(): df[col_plot.lower()]
    })

    if mmc_min is not None and mmc_max is not None:
        plot_data['mmc_min'] = mmc_min['mmc']
        plot_data['mmc_max'] = mmc_max['mmc']

    plot_data.to_csv(output_dir / f'{title.lower().replace(" ", "_")}.csv', index=False)
    plt.close()

def parse_date(filename):
    try:
        return datetime.strptime(filename.split('.')[0], '%Y-%m-%d')
    except ValueError as e:
        pass

def plot_hist_feature(feature, basepath, output_dir):
    dates = os.listdir(basepath)

    #select only the 1st and 15th of each month
    dates_parsed = [(date, parse_date(date)) for date in dates if parse_date(date) is not None]
    sorted_dates_parsed = sorted(dates_parsed, key=lambda x: x[1])
    sorted_dates_filtered = [date for date, date_obj in sorted_dates_parsed if date_obj.day in {1, 15}]

    all_categories = set()
    for date in sorted_dates_filtered:
        date_json = os.path.join(basepath, date)
        with open(date_json, 'r') as f:
            data = json.load(f)
        if "histogram" in data["drilldowns"][feature]:
            all_categories.update(data["drilldowns"][feature]["histogram"]["x"])

        else:
            break
    # Convert set to sorted list to maintain order
    all_categories = sorted(all_categories)
    

    # Make a figure and add the traces for histogram and kde
    fig = go.Figure()
    max_y_value = 0
    trace_counter = 0
    for i, date in enumerate(sorted_dates_filtered):
        date_json = os.path.join(basepath, date)

        with open(date_json, 'r') as f:
            data = json.load(f)

        if "kdehistplot" in data["drilldowns"][feature] and "kde_x" in data["drilldowns"][feature]["kdehistplot"]:
            # Extract the necessary data
            hist = data["drilldowns"][feature]["kdehistplot"]["hist"]
            edges = data["drilldowns"][feature]["kdehistplot"]["plot_edges"]
            centers = data["drilldowns"][feature]["kdehistplot"]["plot_centers"]
            kde_x = data["drilldowns"][feature]["kdehistplot"]["kde_x"]
            kde = data["drilldowns"][feature]["kdehistplot"]["kde"]

            # Add traces for each date
            fig.add_trace(
                go.Bar(x=centers, y=hist, marker=dict(color='blue'), name=f'Histogram', opacity=0.75, visible=(i == 0))
            )
            fig.add_trace(
                go.Scatter(x=kde_x, y=kde, mode='lines', line=dict(color='red'), name=f'KDE', visible=(i == 0))
            )
            trace_counter += 2
            max_hist_value = max(hist)
            max_kde_value = max(kde)
            max_y_value = max(max_y_value, max_hist_value, max_kde_value)

        elif "kdehistplot" in data["drilldowns"][feature]:
            hist = data["drilldowns"][feature]["kdehistplot"]["hist"]
            edges = data["drilldowns"][feature]["kdehistplot"]["plot_edges"]
            centers = data["drilldowns"][feature]["kdehistplot"]["plot_centers"]

            # Add traces for each date
            fig.add_trace(
                go.Bar(x=centers, y=hist, marker=dict(color='blue'), name=f'Histogram', opacity=0.75, visible=(i == 0))
            )

            trace_counter += 1
            max_hist_value = max(hist)
            max_y_value = max(max_y_value, max_hist_value)

        else:
            # Categorical data processing
            probability = data["drilldowns"][feature]["histogram"]["probability"]
            category_data = {cat: 0 for cat in all_categories}  # Initialize all categories with 0
            for cat, prob in zip(data["drilldowns"][feature]["histogram"]["x"], probability):
                category_data[cat] = prob

            fig.add_trace(
                go.Bar(x=list(category_data.keys()), y=list(category_data.values()), marker=dict(color='blue'), name=f'Category Probability {date}', visible=(i == 0))
            )
            trace_counter += 1
            max_y_value = max(max_y_value, max(probability))


    # Create steps for the slider
    steps = []
    visibility_array = [False] * trace_counter

    current_trace_index = 0
    for i, date in enumerate(sorted_dates_filtered):
        visible = visibility_array[:]
        data = json.load(open(os.path.join(basepath, sorted_dates_filtered[i]), 'r'))
        if "kdehistplot" in data["drilldowns"][feature] and "kde_x" in data["drilldowns"][feature]["kdehistplot"]:
            visible[current_trace_index] = True
            visible[current_trace_index + 1] = True
            current_trace_index += 2

        elif "kdehistplot" in data["drilldowns"][feature]:
            visible[current_trace_index] = True
            current_trace_index += 1
        else:
            visible[current_trace_index] = True
            current_trace_index += 1

        steps.append({
            'method': 'update',
            'args': [{'visible': visible}, {'title': f"Histogram for {feature} on {date.split('.')[0]}"}],
            'label': date.split('.')[0]
        })
    # Create and add slider
    sliders = [dict(
        active=0,
        currentvalue={"prefix": "Date: "},
        pad={"t": 80},
        steps=steps
    )]

    fig.update_layout(
        sliders=sliders,
        title_text=f"Histogram for {feature} on " + sorted_dates_filtered[0],
        height=600,
        width=1000,
        title_x=0.5, 
        title_y=0.9,
    )
    fig.update_yaxes(range=[0, max_y_value])

    fig.write_html(os.path.join(output_dir,f'{feature}_histogram_interactive.html'))
    #fig.show()