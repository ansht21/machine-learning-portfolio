"""Inference, Scoring, and Cash Clearance Forecasting.

Provides:
1. LeakageRiskScorer: Real-time risk probability score (0-100%) and tier (Low/Medium/High).
2. OutlierDetector: Isolation Forest scoring for suspicious or abnormal transactions.
3. CashLiquidationForecaster: Estimates the conversion of In-Transit inventory to realized bank cash.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd


class LeakageRiskScorer:
    """Inference scorer using the trained Leakage Risk Classifier."""

    def __init__(self, model_dir: Path | str):
        self.model_dir = Path(model_dir)
        self.model = joblib.load(self.model_dir / "leakage_risk_model.joblib")
        with open(self.model_dir / "model_metadata.json", "r") as f:
            self.metadata = json.load(f)
        self.feature_names = self.metadata['feature_names']

    def score_df(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        """Takes a dataframe of orders, aligns features, and returns risk scores."""
        from src.pipeline.feature_pipeline import engineer_features

        feat_df = engineer_features(raw_df)

        # One-hot encode categoricals
        cat_cols = ['state_grouped', 'category_grouped', 'size_grouped']
        encoded = pd.get_dummies(feat_df, columns=cat_cols, drop_first=True)

        # Align with training features
        for col in self.feature_names:
            if col not in encoded.columns:
                encoded[col] = 0
        X_aligned = encoded[self.feature_names]

        probs = self.model.predict_proba(X_aligned)[:, 1]
        
        result_df = raw_df.copy()
        result_df['leakage_risk_score'] = (probs * 100).round(1)
        result_df['risk_tier'] = pd.cut(
            result_df['leakage_risk_score'],
            bins=[-1, 30, 60, 100],
            labels=['Low Risk', 'Moderate Risk', 'High Risk'],
        )
        return result_df


class AnomalyDetector:
    """Inference detector using the trained Isolation Forest."""

    def __init__(self, model_dir: Path | str):
        self.model_dir = Path(model_dir)
        self.model = joblib.load(self.model_dir / "isolation_forest.joblib")

    def score_df(self, df: pd.DataFrame) -> pd.DataFrame:
        clean = df.copy()
        if 'unit_price' not in clean.columns:
            clean['unit_price'] = (clean['Amount'].fillna(0) / clean['Qty'].clip(lower=1)).round(2)
        if 'order_dayofweek' not in clean.columns:
            clean['order_dayofweek'] = pd.to_datetime(clean['Date']).dt.dayofweek

        features = ['Amount', 'Qty', 'unit_price', 'order_dayofweek']
        preds = self.model.predict(clean[features].fillna(0))
        scores = self.model.decision_function(clean[features].fillna(0))

        result = df.copy()
        result['is_anomaly'] = (preds == -1).astype(int)
        result['anomaly_score'] = (-scores).round(4)
        return result


def forecast_cash_liquidation(
    df: pd.DataFrame, historical_clearance_rate: float = 0.6918
) -> Dict[str, Any]:
    """Calculates liquidation dynamics for currently In-Transit inventory."""
    in_transit = df[df['financial_bucket'] == 'In-Transit']
    transit_gross = float(in_transit['Amount'].sum())
    transit_orders = int(len(in_transit))

    expected_realized_cash = transit_gross * historical_clearance_rate
    expected_leakage = transit_gross * (1.0 - historical_clearance_rate)

    # 3-week projected clearance breakdown
    week_projections = [
        {
            'horizon': 'Week +1 (Fast Settle)',
            'pct_settling': 0.50,
            'projected_inflow': round(expected_realized_cash * 0.50, 2),
            'projected_leakage': round(expected_leakage * 0.50, 2),
        },
        {
            'horizon': 'Week +2 (Standard Settle)',
            'pct_settling': 0.35,
            'projected_inflow': round(expected_realized_cash * 0.35, 2),
            'projected_leakage': round(expected_leakage * 0.35, 2),
        },
        {
            'horizon': 'Week +3 (Long Tail Settle)',
            'pct_settling': 0.15,
            'projected_inflow': round(expected_realized_cash * 0.15, 2),
            'projected_leakage': round(expected_leakage * 0.15, 2),
        },
    ]

    return {
        'in_transit_gross': transit_gross,
        'in_transit_orders': transit_orders,
        'clearance_rate_assumed': historical_clearance_rate,
        'total_expected_realized': round(expected_realized_cash, 2),
        'total_expected_leakage': round(expected_leakage, 2),
        'horizon_breakdown': week_projections,
    }
