"""Model Training & Evaluation Module for Leakage Prediction and Anomaly Detection.

Trains:
1. Leakage Risk Classifier (GradientBoosting / RandomForest)
2. Unsupervised Transaction Outlier Detector (Isolation Forest)
3. Weekly Revenue Forecasting Baseline
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, IsolationForest, RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

logger = logging.getLogger(__name__)


def train_leakage_models(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    output_dir: Path | str,
) -> Dict[str, Any]:
    """Trains Leakage Risk Classification models and selects the top performer."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    models = {
        'RandomForest': RandomForestClassifier(
            n_estimators=120, max_depth=12, min_samples_split=10, random_state=42, n_jobs=-1
        ),
        'GradientBoosting': GradientBoostingClassifier(
            n_estimators=100, learning_rate=0.08, max_depth=5, random_state=42
        ),
    }

    results = {}
    best_model_name = None
    best_auc = -1.0
    best_model_obj = None

    for name, model in models.items():
        logger.info(f"Training {name}...")
        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        probs = model.predict_proba(X_test)[:, 1]

        auc = roc_auc_score(y_test, probs)
        pr_auc = average_precision_score(y_test, probs)
        acc = accuracy_score(y_test, preds)
        prec = precision_score(y_test, preds, zero_division=0)
        rec = recall_score(y_test, preds, zero_division=0)
        f1 = f1_score(y_test, preds, zero_division=0)

        results[name] = {
            'roc_auc': round(float(auc), 4),
            'pr_auc': round(float(pr_auc), 4),
            'accuracy': round(float(acc), 4),
            'precision': round(float(prec), 4),
            'recall': round(float(rec), 4),
            'f1_score': round(float(f1), 4),
        }

        if auc > best_auc:
            best_auc = auc
            best_model_name = name
            best_model_obj = model

    # Save best model
    model_file = out_path / "leakage_risk_model.joblib"
    joblib.dump(best_model_obj, model_file)

    # Save feature names and metadata
    metadata = {
        'best_model': best_model_name,
        'metrics': results,
        'feature_names': X_train.columns.tolist(),
    }
    with open(out_path / "model_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    return metadata


def train_isolation_forest_anomaly_detector(
    df: pd.DataFrame, output_dir: Path | str, contamination: float = 0.03
) -> Dict[str, Any]:
    """Trains Isolation Forest on transactional numeric features to flag multi-dimensional anomalies."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    features = ['Amount', 'Qty', 'unit_price', 'order_dayofweek']
    clean_df = df[features].dropna().copy()

    iso_forest = IsolationForest(
        n_estimators=100, contamination=contamination, random_state=42, n_jobs=-1
    )
    iso_forest.fit(clean_df)

    # Save model
    joblib.dump(iso_forest, out_path / "isolation_forest.joblib")

    return {
        'contamination': contamination,
        'features': features,
        'n_samples_trained': len(clean_df),
    }
