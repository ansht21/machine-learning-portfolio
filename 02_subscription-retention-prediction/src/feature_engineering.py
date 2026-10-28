"""Leakage-safe feature engineering (same as notebook).

The `avg_monthly` aggregate uses the TRAINING mean only (passed by
make_splits/train.py) so that batch statistics never leak from evaluation
data, and the API can reproduce the exact same feature from the value
persisted in the model bundle.
"""
import pandas as pd
from src.config import CATEGORICALS, NUMERICS


def build_features(df: pd.DataFrame, avg_monthly: float | None = None) -> pd.DataFrame:
    X = df.copy()
    X["charge_ratio"] = X["TotalCharges"] / (X["tenure"] + 1)
    X["tenure_x_monthly"] = X["tenure"] * X["MonthlyCharges"]
    if avg_monthly is None:
        avg_monthly = X["MonthlyCharges"].mean()
    X["monthly_above_avg"] = (X["MonthlyCharges"] > avg_monthly).astype(int)
    feat_cols = CATEGORICALS + NUMERICS + ["charge_ratio",
                                           "tenure_x_monthly",
                                           "monthly_above_avg"]
    X_enc = pd.get_dummies(X[feat_cols], drop_first=True)
    return X_enc


def feature_columns():
    return None  # resolved at train time; columns persisted with model
