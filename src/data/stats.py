"""Basic dataset statistics — label prevalence, uncertain-label counts, split sizes."""
from typing import Dict, List

import pandas as pd

from src.data.constants import NEGATIVE, POSITIVE, UNCERTAIN
from src.data.labels import extract_patient_id


def compute_label_prevalence(df: pd.DataFrame, target_findings: List[str]) -> pd.DataFrame:
    """Per-finding counts of positive / negative / uncertain / missing."""
    rows = []
    for finding in target_findings:
        col = df[finding]
        rows.append(
            {
                "finding": finding,
                "positive": int((col == POSITIVE).sum()),
                "negative": int((col == NEGATIVE).sum()),
                "uncertain": int((col == UNCERTAIN).sum()),
                "missing": int(col.isna().sum()),
                "prevalence_%": round(100 * (col == POSITIVE).sum() / len(df), 2) if len(df) else 0.0,
            }
        )
    return pd.DataFrame(rows)


def dataset_summary(df: pd.DataFrame, target_findings: List[str]) -> Dict:
    n_patients = df["Path"].map(extract_patient_id).nunique()
    return {
        "num_images": len(df),
        "num_patients": n_patients,
        "images_per_patient_avg": round(len(df) / n_patients, 2) if n_patients else 0.0,
        "label_prevalence": compute_label_prevalence(df, target_findings).to_dict(orient="records"),
    }


def format_summary(summary: Dict) -> str:
    lines = [
        f"Images: {summary['num_images']}",
        f"Patients: {summary['num_patients']}",
        f"Images/patient (avg): {summary['images_per_patient_avg']}",
        "Label prevalence:",
    ]
    for row in summary["label_prevalence"]:
        lines.append(
            f"  {row['finding']:<20} pos={row['positive']:<6} neg={row['negative']:<6} "
            f"uncertain={row['uncertain']:<6} missing={row['missing']:<6} ({row['prevalence_%']}% positive)"
        )
    return "\n".join(lines)
