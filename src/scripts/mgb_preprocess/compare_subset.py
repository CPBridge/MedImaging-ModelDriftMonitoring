import pandas as pd

from model_drift import mgb_locations


def compare_subset():
    """Creates a subset of labels that were previously checked by a radiologist
    to enable comparisons.

    """
    # compare radiologist labeling with findings labels
    # select unique identifiers from the radiologist labeling file
    rad_labeling = pd.read_excel(mgb_locations.csv_dir / "radiologist_checked_labels.xlsx")
    rad_labeling = rad_labeling.loc[(rad_labeling.Impression.notna()), :]
    rad_labeling_unique_id = rad_labeling.loc[:, ['ANON_MRN', 'ANON_AccNumber']]

    # read in labels.csv and filter reports
    labels_df = pd.read_csv(mgb_locations.labels_csv)
    labels_df_subset = labels_df.merge(
        rad_labeling_unique_id,
        how="inner",
        left_on=['PatientID', 'AccessionNumber'],
        right_on=['ANON_MRN', 'ANON_AccNumber']
    )
    labels_df_subset.to_csv(mgb_locations.csv_dir / "labels_subset.csv")


if __name__ == '__main__':
    compare_subset()
