import pandas as pd

from model_drift import mgb_locations


def split_findings():
    """Split the findings dataframe into 10 smaller dataframes.

    This is done to parallelize the NLP processing.

    """
    splits_dir = mgb_locations.reports_dir / "splits"
    splits_dir.mkdir(exist_ok=True)
    findings_df = pd.read_csv(mgb_locations.reports_dir / 'findings.csv', header=None)
    split_number = 1

    for i in range(findings_df.shape[0]):
        if i % 10000 == 0:
            df = findings_df.iloc[i:i+10000, :]
            df.to_csv(splits_dir / "findings{split_number}.csv", header=None)
            split_number += 1


if __name__ == "__main__":
    split_findings()
