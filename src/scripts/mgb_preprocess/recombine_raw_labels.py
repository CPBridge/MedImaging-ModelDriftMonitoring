import pandas as pd

from model_drift import mgb_locations


def recombine_raw_labels():
    """Recombine the multiple raw label files.

    After splitting the findings into sections for the NLP tool, this function
    recombines the outputs of the NLP back into the full raw labels CSV.

    """
    # read in and combine all the raw_labels files
    raw_labels_df = pd.DataFrame()
    splits_dir = mgb_locations.csv_dir / "raw_labels_splits"
    for i in range(10):
        filepath = splits_dir / f"raw_labels{i + 1}.csv"
        raw_labels_df_tem = pd.read_csv(filepath, dtype=str)
        raw_labels_df_tem = raw_labels_df_tem.loc[(raw_labels_df_tem.Reports != '0'), :]
        raw_labels_df = pd.concat([raw_labels_df, raw_labels_df_tem], ignore_index=True)

    # output and store the combined raw labels file
    raw_labels_df.to_csv(mgb_locations.raw_labels_csv, index=False)


if __name__ == "__main__":
    recombine_raw_labels()
