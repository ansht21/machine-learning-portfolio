"""Data-drift reporting for the deployed churn gateway.

Compares the training distribution (reference) against a recent batch of
customers (current) with Evidently's DataDriftPreset, and writes:

    outputs/churn_drift_report.html   (interactive report)
    outputs/churn_drift_summary.json (machine-readable summary)

Run from 02_subscription-retention-prediction/:
    python src/drift_report.py

Honest note: with no live traffic archive, the "current"-batch default is the
most recent segment of the Telco dataset (last 2,034 rows), which mimics a
monitoring batch. When real Space traffic is logged, pass that CSV instead:
    python src/drift_report.py --current path/to/live_batch.csv
"""
import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from src.config import DATA, OUTPUTS_DIR

NUMERICAL = ["tenure", "MonthlyCharges", "TotalCharges"]
CATEGORICAL = ["Contract", "InternetService", "PaymentMethod",
               "OnlineSecurity", "TechSupport", "PaperlessBilling"]


def load_frames(current_path: str | None):
    """Time-ordered split: reference = earliest rows, current = latest rows."""
    import pandas as pd
    df = pd.read_csv(DATA)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df = df.dropna(subset=["TotalCharges"]).reset_index(drop=True)
    cols = NUMERICAL + CATEGORICAL
    if current_path:
        cur = pd.read_csv(current_path)
        cur["TotalCharges"] = pd.to_numeric(cur["TotalCharges"], errors="coerce")
        cur = cur.dropna(subset=["TotalCharges"])
        ref = df.sample(n=min(5000, len(df)), random_state=42)
    else:
        cut = int(len(df) * 0.71)  # earliest ~5,000 as training window
        ref, cur = df.iloc[:cut], df.iloc[cut:]
    return ref[cols], cur[cols]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--current", default=None, help="CSV of a recent live batch")
    args = ap.parse_args()

    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    from evidently import Report
    from evidently.presets import DataDriftPreset

    ref, cur = load_frames(args.current)

    report = Report([DataDriftPreset()])
    result = report.run(reference_data=ref, current_data=cur)

    OUTPUTS_DIR.mkdir(exist_ok=True)
    html_path = OUTPUTS_DIR / "churn_drift_report.html"
    result.save_html(str(html_path))

    content = None
    for getter in ("get_dict", "to_dict", "load_dict"):
        try:
            content = getattr(result, getter)()
            break
        except (AttributeError, TypeError):
            continue
    if content is None:  # fall back to widget-JSON if API shifts again
        try:
            content = result.dict()
        except Exception as e:
            content = {"metrics": [], "parse_note": f"snapshot extraction failed: {e}"}
    summary = {"drifted_columns": [], "share_drifted": None}
    try:
        # Evidently 0.7 format: each metric entry is {id, metric_name, config, value}
        # DriftedColumnsCount -> {"count": n, "share": s}
        # ValueDrift(column=X) -> per-column drift score (p-value; drifted if < drift_share=0.5)
        total_cols = 0
        for m in content.get("metrics", []):
            name = m.get("metric_name", "")
            val = m.get("value", {})
            if name.startswith("ValueDrift"):
                total_cols += 1
                score = val if isinstance(val, (int, float)) else val.get("drift_score")
                if isinstance(score, (int, float)) and score < 0.5:
                    # e.g. "ValueDrift(column=tenure)"
                    col = name.split("column=")[-1].rstrip(")")
                    summary["drifted_columns"].append({"column": col, "score": round(float(score), 4)})
            elif name.startswith("DriftedColumnsCount"):
                if summary["share_drifted"] is None and isinstance(val, dict):
                    summary["share_drifted"] = val.get("share")
        if total_cols:
            summary["share_drifted"] = round(len(summary["drifted_columns"]) / total_cols, 3)
    except Exception as e:  # keep summary best-effort; HTML is the source of truth
        summary["parse_note"] = f"summary extraction failed: {e}"

    with open(OUTPUTS_DIR / "churn_drift_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print("drift report ->", html_path)
    print("summary:", json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
