import os
import subprocess
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

def process_folder(input_dir, skip_existing=True):
    """
    Process each directory by creating a corresponding output directory and running a plotting script.

    Args:
    input_dir (Path): The input directory to process.
    """
    output_dir = input_dir.parent / f"{input_dir.name}_PLOTS"
    if skip_existing and output_dir.exists() and any(output_dir.iterdir()):
        print(f"Skipping {input_dir} as output directory {output_dir} already exists and is not empty.")
        return

    # If not exists or empty, create the directory
    output_dir.mkdir(exist_ok=True)
    
    drift_csv = os.path.join(input_dir, 'output.csv')
    # run plotting
    command = [
        "python", "basic_performance_plots.py",  
        str(drift_csv),
        str(output_dir)
    ]

    # Execute the command
    print(f"Running plotting on {drift_csv} outputting to {output_dir}")
    subprocess.run(command, check=True)

def run_plotting_on_all_folders(root_dir, skip_existing=True):
    """
    Iterates over all folders in the specified root directory and processes each in parallel.

    Args:
    root_dir (str): The root directory containing folders to process.
    """
    root_path = Path(root_dir)
    # Filter directories to process
    directories = [
        d for d in root_path.iterdir()
        if d.is_dir() and d.name != 'old_analysis' and 'PLOTS' not in d.name
    ]
    skip_existing_list = [skip_existing] * len(directories)
    # Process each directory in parallel
    with ProcessPoolExecutor(max_workers=12) as executor:
        executor.map(process_folder, directories, skip_existing_list)

if __name__ == "__main__":
    directory_to_process = '/autofs/cluster/qtim/projects/xray_drift/drift_analyses'  # Replace with your directory path
    skip_existing = False
    run_plotting_on_all_folders(directory_to_process, skip_existing=skip_existing)
