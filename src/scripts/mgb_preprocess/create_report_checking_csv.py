from pathlib import Path

import click
import pandas as pd

from pycrumbs import tracked

from model_drift import mgb_locations
from model_drift.data import mgb_data


@click.command()
@click.argument("n", type=int)
@click.option(
    "--output-dir",
    "-o",
    type=click.Path(path_type=Path, file_okay=False, writable=True),
    default=mgb_locations.report_checking_dir,
    help="Output directory",
    show_default=True,
)
@tracked(directory_parameter="output_dir")
def create_report_checking_csv(n: int, output_dir: Path):
    """Create a CSV file of a random selection of reports to check.

    The output contains N reports sampled from each day of the dataset.

    """
    # Load in raw findings file
    findings = pd.read_csv(
        mgb_locations.findings_csv,
        header=None,
        names=["Findings"],
    )
    reports = pd.read_csv(mgb_locations.reports_csv, dtype=str)
    reports = reports[["Patient MRN", "Accession Number", "Report Text"]].copy()
    reports = pd.concat([reports, findings], axis=1)

    # Load in labels file
    labels = pd.read_csv(
        mgb_locations.preprocessed_labels_csv,
        dtype={"PatientID": str, "AccessionNumber": str, "StudyDate": str},
        index_col=0,
    )
    labels = labels[labels["AccessionNumber"].notnull()].copy()

    # Load in crosswalk table
    crosswalk = pd.read_csv(mgb_locations.crosswalk_csv, dtype=str)

    merged = (
        labels.merge(
            crosswalk,
            how="left",
            left_on="AccessionNumber",
            right_on="ANON_AccNumber",
            validate="one_to_one",
        )
        .merge(
            reports,
            how="left",
            left_on="ORIG_AccNumber",
            right_on="Accession Number",
            validate="one_to_one",
        )
    )
    merged["StudyDate"] = pd.to_datetime(merged.StudyDate)

    random_state = 12345
    sample = (
        merged.groupby(
            [
                merged.StudyDate.dt.year,
                merged.StudyDate.dt.month,
                merged.StudyDate.dt.day,
            ]
        )
        .sample(n=n, random_state=random_state)  # Sample n per group
        .sample(frac=1, random_state=random_state)  # Shuffle full result
    )
    sample["FindingsExtractionError"] = ""
    label_cols = list(mgb_data.LABEL_GROUPINGS.keys())
    sample = sample[
        [
            "Findings",
            *mgb_data.LABEL_GROUPINGS.keys(),
            "ANON_MRN",
            "ANON_AccNumber",
            "Report Text",
            "FindingsExtractionError",
        ]
    ].copy()

    sample.to_csv(
        output_dir / "reports_to_check.csv",
        index=False
    )


if __name__ == "__main__":
    create_report_checking_csv()
