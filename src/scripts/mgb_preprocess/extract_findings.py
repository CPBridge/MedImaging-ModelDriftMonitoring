from os import linesep
from pathlib import Path

import click
import pandas as pd
from pycrumbs import tracked

from model_drift import mgb_locations


def get_finding(report: str) -> str:
    """Get the impression and finding sections of a single report.

    Parameters
    ----------
    report: str
        The report string, raw.

    Returns
    -------
    str:
        The report's impression and finding sections.

    """
    start_markers = [
        "FINDINGS:",
        "IMPRESSION:",
        "IMPRESSION.",
        "IMPRESSION",
        "RECOMMENDATION:",
    ]
    for marker in start_markers:
        if marker in report:
            if len(report.split(marker)[1]) > 5:  # some reports have two FINDINGS titles.
                findings = report.split(marker)[1]
            else:
                findings = report.split(marker)[2]
            break
    else:
        print("FAILED CASE:")
        print(report)
        print()
        findings = report

    end_markers = [
        "RECOMMENDATION:",
        "ATTESTATION:",
        "ATTESTATION.",
        "ATTESTATION",
    ]
    for marker in end_markers:
        if marker in report:
            findings = findings.split(marker)[0]
            break

    # findings = " ".join(findings.splitlines()).strip() + linesep (version 0)

    # remove headings and retain line space ('\n') in findings
    ## the code is slow (~3 minutes)
    findings = " ".join(findings.splitlines()).strip()
    # check whether there is only one finding in the report
    # (e.g., there is only Impressions section and no Findings section)
    if len(findings.split(':')) != 1:
        findings = findings.split(':')[1:]
        for i, finding in enumerate(findings):
            # print(i, "Finding:\n", finding)
            # if we reach the last finding, add a line space at the end
            if i + 1 == len(findings):
                findings[i] = finding.strip() + linesep
                break
            # for all the other findings, remove the last word/phrase
            # (i.e., the original heading such as "Lungs" and "Heart and mediastinum")
            else:
                # if len(finding.split(". ")) != 1: (version 2)
                # findings[i] = '. '.join(finding.split('. ')[:-1]).strip() (version 1)
                finding = ' '.join(finding.split(' ')[:-1]).strip()
                # some headings have length > 1, we have to further remove some words/phrases
                if finding.endswith(' Heart and'):  # (i.e., Heart and mediastinum)
                    findings[i] = ' '.join(finding.split(' ')[:-2]).strip()
                elif finding.endswith(' Bones/Soft'):  # (i.e., Bones/Soft tissues)
                    findings[i] = ' '.join(finding.split(' ')[:-1]).strip()
                elif finding.endswith(' Bones and soft'):  # (i.e., Bones and soft tissues)
                    findings[i] = ' '.join(finding.split(' ')[:-3]).strip()
                else:
                    findings[i] = finding
        # join all the findings in a list back together
        # findings = '.\n'.join(findings) (version 1)
        findings = '\n'.join(findings)

    return findings


@click.command()
@click.argument(
    "output-dir",
    type=click.Path(file_okay=False, writable=True, path_type=Path),
    default=mgb_locations.reports_dir,
)
@tracked(directory_parameter="output_dir")
def extract_findings(output_dir: Path) -> None:
    """Extract impressions and findings from the reports CSV and stored in findings.csv."""
    reports_df = pd.read_csv(mgb_locations.reports_csv, dtype=str)
    findings = reports_df["Report Text"].apply(get_finding)
    findings_df = pd.DataFrame(
        {
            "finding": findings,
        }
    )
    findings_df.to_csv(
        output_dir / "findings.csv",
        header=False,
        index=False,
    )


# split the findings dataframe into 10 smaller dataframes and store them
def split_findings():
    findings_df = pd.read_csv(mgb_locations.reports_dir / 'findings.csv', header=None)
    filename = 1
    filepath = "/Volumes/qtim/datasets/private/xray_drift/reports/"
    for i in range(findings_df.shape[0]):
        if i % 10000 == 0:
            df = findings_df.iloc[i:i+10000, :]
            df.to_csv(filepath + "findings" + str(filename) + ".csv", header=None)
            filename += 1


# combine and output the raw_labels.csv file (based on findings)
def output_raw_labels_df():
    # read in and combine all the raw_labels files
    raw_labels_df = pd.DataFrame()
    for i in range(10):
        filepath = str(mgb_locations.reports_dir) + "/raw_labels" + str(i + 1) + ".csv"
        raw_labels_df_tem = pd.read_csv(filepath)
        raw_labels_df_tem = raw_labels_df_tem.loc[(raw_labels_df_tem.Reports != '0'), :]
        raw_labels_df = pd.concat([raw_labels_df, raw_labels_df_tem], ignore_index=True)

    # output and store the combined raw labels file
    raw_labels_df.to_csv(mgb_locations.raw_labels_csv)


if __name__ == "__main__":
    # extract_findings()
    # split_findings()
    # output_raw_labels_df()

    # compare radiologist labeling with findings labels
    # select unique identifiers from the radiologist labeling file
    rad_labeling = pd.read_excel(mgb_locations.csv_dir / "radiologist_checked_labels.xlsx")
    rad_labeling = rad_labeling.loc[(rad_labeling.Impression.notna()), :]
    rad_labeling_unique_id = rad_labeling.loc[:, ['ANON_MRN', 'ANON_AccNumber']]

    # read in labels.csv and filter reports
    labels_df = pd.read_csv(mgb_locations.labels_csv)
    labels_df_subset = labels_df.merge(rad_labeling_unique_id,
                                       how="inner",
                                       left_on=['PatientID', 'AccessionNumber'],
                                       right_on=['ANON_MRN', 'ANON_AccNumber'])
    labels_df_subset.to_csv(mgb_locations.csv_dir / "labels_subset.csv")


##################################To BE DELETED#########################################
    # check docker running issue
    # reports_path = mgb_locations.reports_dir / 'findings10.csv'
    # reports = pd.read_csv(reports_path,
    #                       header=None,
    #                       names=[0])[0].tolist()
    #
    # for i, report in enumerate(reports):
    #     print(i, 'Reports:\n', report)
    #     lower_report = report.lower()

    # check the results of our get_finding() function
    # report 1: There is only the IMPRESSION: section.
    report1 = "Order Reason for Exam:" \
             "\nFever" \
             "\nNarrative:" \
             "\nXR ABDOMEN 1 VIEW 7/15/2019, XR CHEST 1 VIEW 7/15/2019, XR SKULL LESS THAN 4" \
             "\nVIEWS 7/15/2019." \
             "\nCOMPARISON: XR CHEST 1 VIEW 3/25/2019." \
             "\n" \
             "\n" \
             "IMPRESSION:" \
             "\nA ventriculoperitoneal shunt is present originating in the lateral ventricle and" \
             "\nterminating in the abdomen. The visualized portions appear intact." \
             "\nLeft basilar opacity likely represents atelectasis, though superimposed" \
             "\naspiration or pneumonia cannot be excluded in the appropriate clinical setting. " \
             "\nNonobstructive bowel gas pattern. Degenerative changes of the visualized" \
             "\nskeleton." \
             "\nATTESTATION: I, Dr. James N Lawrason as teaching physician, have reviewed the" \
             "\nimages for this case and if necessary edited the report originally created by" \
             "\nDr. Sarah A Ebert."

    # report 2: This is the most common case.
    report2 = "Order Reason for Exam:" \
              "\n*Shortness of breath" \
              "\nNarrative:" \
              "\nTECHNIQUE: XR CHEST PA AND LATERAL 2 VIEWS" \
              "\nCOMPARISON: 8/17/2016" \
              "\n" \
              "\nFINDINGS:" \
              "\nLines/tubes: There has been a prior sternotomy." \
              "\nLungs: Chain sutures remain in the RIGHT upper lobe, there is no evidence of" \
              "\nsuperimposed pneumonia or edema" \
              "\nPleura: There is no pleural effusion or pneumothorax." \
              "\nHeart and mediastinum: The cardiac silhouette remains stable status post valve" \
              "\nrepair" \
              "\nBones: Unchanged" \
              "\n" \
              "\n" \
              "\nIMPRESSION:" \
              "\nPrior sternotomy postsurgical changes in the RIGHT lung, no definite evidence of" \
              "\npneumonia or edema"

    # report 3: There are some special headings like Heart/Mediastinum, and Bones/Soft Tissues.
    report3 = "Order Reason for Exam:" \
              "\nFever" \
              "\nNarrative:" \
              "\nXR CHEST PA AND LATERAL 2 VIEWS" \
              "\n" \
              "\nCOMPARISON: XRCH2 2013-Dec-21" \
              "\n " \
              "\nFINDINGS:" \
              "\n " \
              "\nDevices/Tubes/Lines: There is an atrial septum occlusion device. There is an arrhythmia monitor projecting over the left chest.." \
              "\n " \
              "\nLungs: The lungs are well inflated. No focal consolidation or pulmonary edema." \
              "\n" \
              "\nPleura: No pleural effusion or pneumothorax." \
              "\nHeart/Mediastinum: The cardiomediastinal silhouette is stable in appearance." \
              "\nBones/Soft Tissues: No significant skeletal abnormality." \
              "\n" \
              "\nIMPRESSION:" \
              "\n No radiographic evidence of pneumonia. [Cov19Neg]." \
              "\n " \
              "\nATTESTATION: I, Dr. Parul Penkar as teaching physician, have reviewed " \
              "the images for this case and if necessary edited the report originally created by Dr. Samuel Cartmell."

    # report 4: Two FINDINGS titles (55925 in combined_reports)
    # other examples: 75037, 81784, 90477
    report4 = "FINDINGS:" \
              "\n" \
              "\nFINDINGS:" \
              "\nLines/tubes: Tracheostomy tube terminates in mid trachea.. The tip of a left PICC projects over the SVC. External EKG leads are again present." \
              "\nLungs: There is complete opacification of the right hemithorax, status post right pneumonectomy. Mild left basilar opacity is present likely atelectasis." \
              "\nPleura: S" \
              "\natus post right pneumonectomy, with expected complete opacification of the pneumonectomy space. There is no pneumothorax or pleural effusion." \
              "\nHeart and mediastinum: The mediastinal contours are unchanged." \
              "\nBones: The visualized bones are unchanged." \
              "\nIMPRESSION:" \
              "\nStatus post right pneumonectomy with opacification of the right hemithorax." \
              "\nLeft basilar opacity improved from prior exam most consistent with atelectasis. In the proper clinical setting, superimposed aspiration or pneumonia the left base is possible."

    clean_report1 = get_finding(report1)
    clean_report2 = get_finding(report2)
    clean_report3 = get_finding(report3)
    clean_report4 = get_finding(report4)










