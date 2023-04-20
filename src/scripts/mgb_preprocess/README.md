# Steps in the Report Labelling Pipeline:

- Start with reports/combined_reports.csv
- Use `extract_findings.py` to extract the findings sections from each report
  and place in a single-column CSV at `reports/findings.csv` (previously
  `extract_impressions.py` was used but this didn't work very well).
- (Optional): Use `split_findings.py` to split the findings into partitions so
  that the next step can be parallelized. These go into `csv/findings_splits`.
- Run the chexpert labelling tool with the `reports/findings.csv` file as
  the input and `csv/raw_labels.csv` as the output. The raw labels file contains
  just the labels and no other study information. If this is done on splits, 
  each split goes into `csv/raw_labels_splits`.
- (Optional): Use `recombine_raw_labels.py` to recombine the raw labels again
  if they were previously split so that the next step can be parallelized.
- Run `merge_labels.py` to merge the raw labels back with the rest of the
  anonymized information. This creates `csv/labels.csv`.
- Run `preprocess_labels.py`, which applies project-specific preprocessing to
  the labels by combining labels, removing labels that are not used and
  converting missing values to negatives and uncertain values to positives. The
  output is placed into the project directory (rather than the dataset
  directory) in `preprocessed_labels/labels.csv`
- Run `create_report_checking_csv.py` to create a stratified sample of the
  preprocessed labels in a format suitable for checking by a radiologist.
