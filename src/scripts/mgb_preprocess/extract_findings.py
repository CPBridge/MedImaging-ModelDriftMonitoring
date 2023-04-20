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


if __name__ == "__main__":
    extract_findings()
