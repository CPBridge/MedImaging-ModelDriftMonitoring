import sys
sys.path.append('/autofs/homes/005/fd881/repos/MedImaging-ModelDriftMonitoring/src')

import json
import pandas as pd
from pathlib import Path
from model_drift import mgb_locations
import os
import six
from datetime import timedelta, datetime
from model_drift.data import mgb_data

TRAIN_DATE_END = datetime(year=2019, month=10, day=1)
VAL_DATE_END = datetime(year=2020, month=1, day=1)

def split_on_date(df, splits, col=None):
    splits = pd.to_datetime(splits).sort_values()

    rem = df

    for split in splits:
        if col is None:
            curr, rem = rem[rem.index < split], rem[rem.index >= split]
        else:
            curr, rem = rem[rem[col] < split], rem[rem[col] >= split]
        yield curr
    yield rem

def make_index(row: pd.Series):
    return f"{row.PatientID}_{row.AccessionNumber}_{row.SOPInstanceUID}"

def jsonl_files2dataframe(jsonl_files, converter=None, refresh_rate=None, **kwargs):
    if isinstance(jsonl_files, six.string_types):
        jsonl_files = [jsonl_files]

    if converter is None:
        def converter(x): return x

    df = []
    for fn in jsonl_files:
        with open(fn, 'r') as f:
            lines = f.readlines()
            if refresh_rate is not None:
                kwargs['miniters'] = int(len(lines) * refresh_rate)
            for line in lines:
                df.append(converter(json.loads(line)))
    return pd.json_normalize(df)

meta_df = pd.read_csv(
    mgb_locations.dicom_inventory_csv,
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

label_cols = list(mgb_data.LABEL_GROUPINGS.keys()) 
input_dir = Path('/autofs/cluster/qtim/projects/xray_drift/inferences/classification_final_allpoc_inference_woconsolidation')

scores_pred_file = input_dir.joinpath("preds.jsonl")
scores_df = jsonl_files2dataframe([scores_pred_file], desc="reading classifier results", refresh_rate=.1)
scores_df = pd.concat(
    [
        scores_df,
        pd.DataFrame(scores_df['activation'].values.tolist(), columns=[f"activation.{c}" for c in label_cols])
    ],
    axis=1
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

vae_input_dir = Path('/autofs/space/crater_001/datasets/private/xray_drift/mgb_florence_embeddings')

print("loading dataset vae results")
vae_pred_file = vae_input_dir.joinpath('preds.jsonl')
vae_df = jsonl_files2dataframe([vae_pred_file], desc="reading VAE results", refresh_rate=.1)
vae_df = pd.concat(
    [
        vae_df,
        pd.DataFrame(vae_df['mu'].values.tolist(), columns=[f"mu.{c:0>3}" for c in range(1024)])
    ],
    axis=1
)
vae_df.drop_duplicates(subset="index", inplace=True)

# rename the mu column to full_mu, to ensure avoid confusion when regex matching
vae_df['full_mu'] = vae_df['mu']

merged_df = scores_df.merge(vae_df, on="index", how="left")
merged_df = merged_df.merge(meta_df, on="index", how="left")

merged_df = merged_df[merged_df["ViewPosition"].isin(('AP', 'PA'))].copy()

train_df, val_df, test_df = split_on_date(
    merged_df,
    [TRAIN_DATE_END, VAL_DATE_END],
    col="StudyDate",
)

output_path = Path('/autofs/cluster/qtim/projects/xray_drift/drift_analyses/PLOTS/paper/drift_analysis_allpoc_emd_jackknife_helllinger_final_florence_PLOTS/label_prevalence.csv')

# Generate label counts for each dataframe
label_counts_data = []

for df_name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
    for col in label_cols:
        counts = df[col].value_counts()
        for value, count in counts.items():
            label_counts_data.append({
                'dataset': df_name,
                'label': col,
                'value': value,
                'count': count
            })

# Create DataFrame and save to CSV
label_counts_df = pd.DataFrame(label_counts_data)
label_counts_df.to_csv(output_path, index=False)

print(f"Label counts saved to: {output_path}")
