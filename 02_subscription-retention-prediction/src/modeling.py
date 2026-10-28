"""Model training, CV, threshold + revenue-at-risk (matches notebook + tests)."""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix,
                             precision_recall_curve)
try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
from src.config import TARGET, RANDOM_STATE
from src.data_processing import load_and_clean
from src.feature_engineering import build_features


def make_splits(df: pd.DataFrame, val_size=0.2, test_size=0.2):
    """Stratified 3-way split: train (60%) / validation (20%) / test (20%).

    The threshold is later selected on VALIDATION only; test stays untouched
    until final reporting. Feature encoding runs on the full table (so all
    splits share identical dummy columns), while the only aggregate feature
    (`monthly_above_avg`) uses the TRAINING mean — no leakage.
    """
    y = df[TARGET]
    train_idx, tmp_idx = train_test_split(
        df.index, test_size=val_size + test_size,
        random_state=RANDOM_STATE, stratify=y)
    val_idx, test_idx = train_test_split(
        tmp_idx, test_size=test_size / (val_size + test_size),
        random_state=RANDOM_STATE, stratify=y.loc[tmp_idx])
    avg_monthly = float(df.loc[train_idx, "MonthlyCharges"].mean())
    X = build_features(df, avg_monthly=avg_monthly)
    splits = {
        "X_train": X.loc[train_idx], "y_train": y.loc[train_idx],
        "X_val": X.loc[val_idx], "y_val": y.loc[val_idx],
        "X_test": X.loc[test_idx], "y_test": y.loc[test_idx],
        "columns": list(X.columns),
        "monthly_charges_avg": avg_monthly,
    }
    return splits


def fit_models(splits: dict):
    Xtr, ytr = splits["X_train"], splits["y_train"]
    scaler = StandardScaler().fit(Xtr)
    Xtr_s = scaler.transform(Xtr)
    logreg = LogisticRegression(max_iter=5000, random_state=RANDOM_STATE).fit(Xtr_s, ytr)
    rf = RandomForestClassifier(n_estimators=400, min_samples_leaf=5,
                                random_state=RANDOM_STATE, n_jobs=-1).fit(Xtr, ytr)
    result = {"Logistic Regression": logreg, "Random Forest": rf, "scaler": scaler}
    if HAS_XGB:
        xgb = XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            use_label_encoder=False,
            eval_metric="logloss",
            random_state=RANDOM_STATE,
            n_jobs=-1
        )
        xgb.fit(Xtr, ytr)
        result["XGBoost"] = xgb
    return result


def evaluate(name, model, X_test, y_test, scaled=False, scaler=None):
    Xt = scaler.transform(X_test) if scaled else X_test
    pred = model.predict(Xt)
    prob = model.predict_proba(Xt)[:, 1]
    return {"model": name, "accuracy": accuracy_score(y_test, pred),
            "precision": precision_score(y_test, pred),
            "recall": recall_score(y_test, pred),
            "f1": f1_score(y_test, pred),
            "roc_auc": roc_auc_score(y_test, prob),
            "confusion_matrix": confusion_matrix(y_test, pred).tolist()}


def cross_validate(splits: dict, cv_folds=5):
    X, y = splits["X_train"], splits["y_train"]
    rows = []
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_STATE)
    estimators = [
        ("Logistic Regression", LogisticRegression(max_iter=5000)),
        ("Random Forest", RandomForestClassifier(n_estimators=200,
         min_samples_leaf=5, random_state=RANDOM_STATE, n_jobs=-1)),
    ]
    if HAS_XGB:
        estimators.append(("XGBoost", XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            use_label_encoder=False, eval_metric="logloss",
            random_state=RANDOM_STATE, n_jobs=-1
        )))
    for name, est in estimators:
        scores = cross_val_score(est, X, y, cv=skf, scoring="roc_auc")
        rows.append({"model": name, "cv_mean_roc_auc": float(scores.mean()),
                     "cv_std": float(scores.std()), "folds": cv_folds})
    return pd.DataFrame(rows)


def optimize_threshold(y_true, proba):
    best, best_cost, thrs = 0.5, float("inf"), np.arange(0.1, 0.9, 0.01)
    for t in thrs:
        pred = (proba >= t).astype(int)
        fp = int(((pred == 1) & (y_true == 0)).sum())
        fn = int(((pred == 0) & (y_true == 1)).sum())
        cost = fp * 1.0 + fn * 5.0  # missed churner costs 5x a false alarm
        if cost < best_cost:
            best, best_cost = t, cost
    return {"cost_optimal": float(best), "min_cost": float(best_cost)}


def plot_threshold_analysis(y_true, proba, output_path):
    """Plot precision, recall, F1 vs threshold and the PR curve.

    Args:
        y_true: array-like of true binary labels.
        proba: array-like of predicted probabilities for the positive class.
        output_path: path (str or Path) where the PNG will be saved.

    Returns:
        str: absolute path of the saved figure.
    """
    from pathlib import Path
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    COST_OPTIMAL_THR = 0.17

    # --- compute metrics at each threshold ---
    thresholds = np.arange(0.01, 0.99, 0.01)
    precisions, recalls, f1s = [], [], []
    for t in thresholds:
        pred = (proba >= t).astype(int)
        tp = int(((pred == 1) & (y_true == 1)).sum())
        fp = int(((pred == 1) & (y_true == 0)).sum())
        fn = int(((pred == 0) & (y_true == 1)).sum())
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        precisions.append(p)
        recalls.append(r)
        f1s.append(f)

    # PR curve from sklearn (has its own threshold grid)
    sk_precision, sk_recall, sk_thresholds = precision_recall_curve(y_true, proba)

    # --- find the point on the sklearn PR curve closest to COST_OPTIMAL_THR ---
    if len(sk_thresholds) > 0:
        idx_pr = int(np.argmin(np.abs(sk_thresholds - COST_OPTIMAL_THR)))
        pr_point = (sk_recall[idx_pr], sk_precision[idx_pr])
    else:
        pr_point = None

    # --- find the point on the threshold grid closest to COST_OPTIMAL_THR ---
    idx_thr = int(np.argmin(np.abs(thresholds - COST_OPTIMAL_THR)))

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Threshold Analysis — Random Forest", fontsize=13, fontweight="bold")

    # Panel 1: P / R / F1 vs Threshold
    ax = axes[0]
    ax.plot(thresholds, precisions, label="Precision", color="steelblue", linewidth=1.8)
    ax.plot(thresholds, recalls, label="Recall", color="darkorange", linewidth=1.8)
    ax.plot(thresholds, f1s, label="F1", color="green", linewidth=1.8)
    ax.axvline(COST_OPTIMAL_THR, color="red", linestyle="--", linewidth=1.5,
               label=f"Cost-optimal thr ({COST_OPTIMAL_THR})")
    ax.scatter([thresholds[idx_thr]], [f1s[idx_thr]], color="red", zorder=5, s=60)
    ax.set_xlabel("Threshold", fontsize=11)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_title("Precision / Recall / F1 vs Threshold", fontsize=11)
    ax.legend(fontsize=9)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)

    # Panel 2: Precision-Recall curve
    ax2 = axes[1]
    ax2.plot(sk_recall, sk_precision, color="steelblue", linewidth=2,
             label="PR curve (Random Forest)")
    ax2.set_xlabel("Recall", fontsize=11)
    ax2.set_ylabel("Precision", fontsize=11)
    ax2.set_title("Precision-Recall Curve", fontsize=11)
    if pr_point is not None:
        ax2.scatter([pr_point[0]], [pr_point[1]], color="red", zorder=5, s=80,
                    label=f"thr={COST_OPTIMAL_THR}")
        ax2.annotate(f"thr={COST_OPTIMAL_THR}\n(R={pr_point[0]:.2f}, P={pr_point[1]:.2f})",
                     xy=pr_point, xytext=(pr_point[0] - 0.25, pr_point[1] - 0.12),
                     fontsize=8, arrowprops=dict(arrowstyle="->", color="red"),
                     color="red")
    ax2.legend(fontsize=9)
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1.05)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return str(output_path)


def risk_band(p: float) -> str:
    if p >= 0.7:
        return "High"
    if p >= 0.4:
        return "Medium"
    return "Low"


def revenue_at_risk(df: pd.DataFrame):
    flagged = df[df["risk_level"] != "Low"] if "risk_level" in df.columns else df
    exp_risk = float((flagged["churn_probability"] * flagged["MonthlyCharges"]).sum()
                     if "churn_probability" in flagged.columns else 0.0)
    return {"flagged_customers": int(len(flagged)),
            "expected_monthly_revenue_at_risk": round(exp_risk, 2)}
