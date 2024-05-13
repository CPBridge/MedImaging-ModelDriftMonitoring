from pathlib import Path
import pandas as pd

import sys
import os
sys.path.append('/autofs/homes/005/fd881/repos/MedImaging-ModelDriftMonitoring/')


from src.model_drift.data import mgb_data
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import json
import plotly.graph_objs as go
from datetime import datetime
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

def normalize_series(data, ref_start, ref_end):
    """ Normalize the series data based on reference period mean and std """
    # ref_start = datetime(2019, 11, 1, 0, 0)
    reference_period = data.loc[ref_start:ref_end]
    mean = reference_period.mean()
    std = reference_period.std()
    norm_result = (data - mean) / std
    return norm_result

def create_normalized_performance_plots(df: pd.DataFrame, output_dir: Path, ref_start: str, ref_end: str):
    plt.style.use('ggplot')
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
        axs[i].grid(True)
        axs[i].tick_params(axis='x', rotation=45)
        axs[i].set_xlim(pd.to_datetime('2019-10-01').date(), pd.to_datetime('2021-07-01').date())
        axs[i].set_ylim(-3, 3)  # Adjust the y-axis limits for normalized data
        axs[i].yaxis.set_major_locator(plt.MultipleLocator(0.5))

    for j in range(i + 1, num_cols * num_rows):
        axs[j].axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'normalized_performance_combined.png'))
    plt.savefig(os.path.join(output_dir, 'normalized_performance_combined.svg'), format='svg', bbox_inches='tight')
    plt.close()

def create_mmc_plot(df, date_col, output_dir, title, col_plot = 'MMC', mmc_min = None, mmc_max = None):
    plt.style.use('ggplot')


    # Plot setup
    if mmc_min is None:
        fig, ax = plt.subplots(figsize=(10, 6))  
        ax.plot(df[date_col], df[col_plot.lower()], label=col_plot)  
        
    else:
        fig, ax = plt.subplots(figsize=(10, 6))  
        ax.plot(df[date_col], df[col_plot.lower()], label=col_plot)  
        ax.fill_between(df[date_col], mmc_min['mmc'], mmc_max['mmc'], alpha=0.5, label='MMC Range', color='gray')

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
    

    #make a figure and add the traces for histogram and kde
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