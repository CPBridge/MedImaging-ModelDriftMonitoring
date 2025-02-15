#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import os
from pathlib import Path
from datetime import datetime

library_path = str(Path(__file__).parent.parent.parent)
PYPATH = os.environ.get("PYTHONPATH", "").split(":")
if library_path not in PYPATH:
    PYPATH.append(library_path)
    os.environ["PYTHONPATH"] = ":".join(PYPATH)

from model_drift.data.dataset import MGBCXRDataset
from model_drift.data.utils import split_on_date
from model_drift.data import mgb_data
from model_drift.drift.config import mgb_default_config
from model_drift.drift.sampler import Sampler, DummySampler
from model_drift.drift.performance import ClassificationReportCalculator
from model_drift import settings, helpers, mgb_locations
from model_drift.helpers import create_score_based_ood_frame
from pycrumbs import tracked
import warnings
import pandas as pd

import argparse


def make_index(row: pd.Series):
    return f"{row.PatientID}_{row.AccessionNumber}_{row.SOPInstanceUID}"


@tracked(directory_parameter="output_dir")
def main(output_dir: Path, args: argparse.Namespace) -> None:

    logger = helpers.basic_logging()

    if args.dataset != "mgb":
        raise NotImplementedError("unrecognized dataset")

    output_dir.mkdir(exist_ok=True)
    name = "output"
    fname = output_dir.joinpath(name + ".csv")

    num_cpus = os.cpu_count()
    if args.num_workers < 0:
        args.num_workers = num_cpus

    print("loading dataset predicted probabilities")
    label_cols = list(MGBCXRDataset.LABEL_COLUMNS)
    scores_pred_file = args.input_dir.joinpath("preds.jsonl")
    scores_df = helpers.jsonl_files2dataframe([scores_pred_file], desc="reading classifier results", refresh_rate=.1)
    scores_df = pd.concat(
        [
            scores_df,
            pd.DataFrame(scores_df['activation'].values.tolist(), columns=[f"activation.{c}" for c in label_cols])
        ],
        axis=1
    )

    print("loading dicom metadata")
    meta_df = pd.read_csv(
        args.metadata_csv,
        index_col=0,
    )
    meta_df.drop(columns=["StudyDate"], inplace=True)  # anonymized dates
    labels_df = pd.read_csv(
        mgb_locations.labels_csv,
        index_col=0,
    )  # need real dates from this file
    meta_df = meta_df.merge(
        labels_df,
        how="left",
        on=("StudyInstanceUID", "PatientID", "AccessionNumber"),
    )

    # Some metadata is from the RIS and is in the reports CSV
    reports = pd.read_csv(mgb_locations.reports_csv, dtype=str)
    reports = reports[
        [
            "Accession Number",
            "Point of Care",
            "Patient Sex",
            "Patient Age",
            "Is Stat",
            "Exam Code",
        ]
    ].copy()
    crosswalk = pd.read_csv(mgb_locations.crosswalk_csv, dtype={"ANON_AccNumber": int})
    crosswalk = crosswalk[["ANON_AccNumber", "ORIG_AccNumber"]]
    # meta_df.assign(AccessionNumber=lambda x: x.AccessionNumber.str.lstrip("0"))

    meta_df = meta_df.merge(
        crosswalk,
        how="left",
        left_on="AccessionNumber",
        right_on="ANON_AccNumber",
        validate="many_to_one",
    )
    meta_df = meta_df.merge(
        reports,
        how="left",
        left_on="ORIG_AccNumber",
        right_on="Accession Number",
        validate="many_to_one",
    )

    meta_df["StudyDate"] = pd.to_datetime(meta_df["StudyDate"], format='%m/%d/%Y')
    meta_df["index"] = meta_df.apply(make_index, axis=1)

    print("loading dataset vae results")
    vae_pred_file = args.vae_input_dir.joinpath('preds.jsonl')
    vae_df = helpers.jsonl_files2dataframe([vae_pred_file], desc="reading VAE results", refresh_rate=.1)
    vae_df = pd.concat(
        [
            vae_df,
            pd.DataFrame(vae_df['mu'].values.tolist(), columns=[f"mu.{c:0>3}" for c in range(args.num_vae_features)])
        ],
        axis=1
    )
    vae_df.drop_duplicates(subset="index", inplace=True)

    # rename the mu column to full_mu, to ensure avoid confusion when regex matching
    vae_df['full_mu'] = vae_df['mu']

    merged_df = scores_df.merge(vae_df, on="index", how="left")
    merged_df = merged_df.merge(meta_df, on="index", how="left")

    # option to only evaluate drift on single location
    if args.point_of_care:
        merged_df = merged_df[merged_df["Point of Care"] == args.point_of_care].copy()

    # only use frontal images
    print("Only using frontal images")
    print(f"Number of samples before filtering: {len(merged_df)}")
    merged_df = merged_df[merged_df["ViewPosition"].isin(('AP', 'PA'))].copy()
    print(f"Number of samples after filtering: {len(merged_df)}")

    if args.ref_window_start and args.ref_window_end:
        ref_window_start = datetime.strptime(args.ref_window_start, "%Y-%m-%d")
        ref_window_end = datetime.strptime(args.ref_window_end, "%Y-%m-%d")
        print('Using non default reference window: from', ref_window_start, ' to ', ref_window_end)

        train_df, val_df, test_df = split_on_date(
            merged_df,
            [ref_window_start, ref_window_end],
            col="StudyDate",
        )

    elif args.ref_window_start or args.ref_window_end:
        raise ValueError("Both reference window start and end dates must be provided.")

    else:
        train_df, val_df, test_df = split_on_date(
            merged_df,
            [mgb_data.TRAIN_DATE_END, mgb_data.VAL_DATE_END],
            col="StudyDate",
        )

    #sampler = Sampler(args.sample_size, replacement=args.replacement)
    sampler = DummySampler(args.sample_size, replacement=args.replacement)


    ref_df = val_df.copy().assign(in_distro=True)

    # here is the hard data injection
    target_df = merged_df.set_index('StudyDate')
    #target_df.index = pd.to_datetime(target_df.index)
    indistro_data = target_df.copy().assign(in_distro=False)

    targets = {}
    targets["indistro"] = indistro_data

    if args.bad_q:
        targets['bad_sample_data'] = create_score_based_ood_frame(indistro_data, label_cols, q=args.bad_q,
                                                                  sample_start_date=args.bad_sample_start_date,
                                                                  sample_end_date=args.bad_sample_end_date,
                                                                  ood_start_date=args.bad_start_date,
                                                                  ood_end_date=args.bad_end_date, bottom=True
                                                                  ).assign(in_distro=False)

    if args.good_q:
        targets['good_sample_data'] = create_score_based_ood_frame(indistro_data, label_cols, q=args.good_q,
                                                                   sample_start_date=args.good_sample_start_date,
                                                                   sample_end_date=args.good_sample_end_date,
                                                                   ood_start_date=args.good_start_date,
                                                                   ood_end_date=args.good_end_date, bottom=True
                                                                   ).assign(in_distro=False)

    for name, target in targets.items():
        target["source"]= name
    
    target_df = pd.concat(targets.values(), sort=True)
    # end of hard data injection code
    dwc = mgb_default_config(ref_df, args.point_of_care, vae_cols=r"^full_mu$", score_cols= r"^activation$")

    dwc.add_drift_stat(
        'performance',
        ClassificationReportCalculator(
            target_names=tuple(mgb_data.LABEL_GROUPINGS)
        ),
        col=("activation", "label"),
        include_stat_name=False
    )

    dwc.prepare(ref_df)

    #target_df = merged_df.set_index('StudyDate')

    ref_df.to_csv(output_dir.joinpath('ref.csv'))
    target_df.to_csv(output_dir.joinpath('target.csv'))

    print("starting drift experiment!")

    print("Currently all warnings are being supressed, this is not safe!")
    warnings.filterwarnings("ignore")

    output = dwc.rolling_window_predict(
        target_df,
        sampler=sampler,
        n_samples=args.n_samples,
        stride=args.stride,
        window=args.window,
        min_periods=args.min_periods,
        n_jobs=args.num_workers,
        backend="threading",
        refresh_rate=.01,
        output_dir=output_dir,
        agg = ('min', 'max', 'mean', 'std', 'median')
    )
    output.to_csv(fname)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--run_azure", type=int, dest="run_azure", help="run in AzureML", default=0)

    parser.add_argument("--input_dir", "-i", type=Path)
    parser.add_argument("--vae_input_dir", "-v", type=Path)
    parser.add_argument("--output_dir", "-o", type=Path)
    parser.add_argument("--metadata_csv", "-m", type=Path, default=mgb_locations.dicom_inventory_csv)

    parser.add_argument("--dataset", type=str, default='mgb')
    parser.add_argument("--vae_dataset", type=str, default='padchest-trained')
    parser.add_argument("--classifier_dataset", type=str, default='padchest-finetuned')
    parser.add_argument("--vae_filter", type=str, default='all-data')
    parser.add_argument("--classifier_filter", type=str, default='frontal_only')
    parser.add_argument("--window", "-w", type=str, default="14D")
    parser.add_argument("--stride", type=str)
    parser.add_argument("--min_periods", type=int, default=150)
    parser.add_argument("--ref_frontal_only", type=int, default=1)

    parser.add_argument("--lateral_add_date", type=str, default=None)
    parser.add_argument("--indist_remove_date", type=str, default=None)

    parser.add_argument("--peds_weight", type=float, default=0)
    parser.add_argument("--peds_start_date", type=str, default=None)
    parser.add_argument("--peds_end_date", type=str, default=None)

    parser.add_argument("--replacement", type=int, default=1)
    parser.add_argument("--sample_size", type=int, default=1000)
    parser.add_argument("--n_samples", type=int, default=20)

    parser.add_argument("--generate_name", type=int, default=0)

    parser.add_argument("--num_workers", type=int, default=-1)
    parser.add_argument("--dbg", type=int, default=0)

    parser.add_argument("--bad_q", type=float, default=0)
    parser.add_argument("--bad_start_date", type=str, default=None)
    parser.add_argument("--bad_end_date", type=str, default=None)
    parser.add_argument("--bad_sample_start_date", type=str, default=None)
    parser.add_argument("--bad_sample_end_date", type=str, default=None)

    parser.add_argument("--good_q", type=float, default=0)
    parser.add_argument("--good_start_date", type=str, default=None)
    parser.add_argument("--good_end_date", type=str, default=None)
    parser.add_argument("--good_sample_start_date", type=str, default=None)
    parser.add_argument("--good_sample_end_date", type=str, default=None)

    parser.add_argument("--start_date", type=str, default=None)
    parser.add_argument("--end_date", type=str, default=None)

    parser.add_argument("--num_vae_features", type=int, default=128)
    parser.add_argument("--point_of_care", type=str, default=None)

    parser.add_argument("--ref_window_start", type=str, default=None)
    parser.add_argument("--ref_window_end", type=str, default=None)

    args = parser.parse_args()

    main(args.output_dir, args)
