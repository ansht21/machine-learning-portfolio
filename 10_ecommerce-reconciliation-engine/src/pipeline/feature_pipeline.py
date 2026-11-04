"""Feature Engineering & Dataset Pipeline for E-Commerce Reconciliation Engine.

Extracts operational, geographical, transactional, and time-based features
free from target leakage to train predictive risk models and anomaly detectors.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


def load_raw_reconciliation_data(parquet_path: Path | str) -> pd.DataFrame:
    """Loads cleaned parquet data and ensures standard types."""
    df = pd.read_parquet(parquet_path)
    if not np.issubdtype(df['parsed_date'].dtype, np.datetime64):
        df['parsed_date'] = pd.to_datetime(df['parsed_date'])
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extracts features available at order intake time (leak-free)."""
    feat_df = df.copy()

    # Time-based features
    feat_df['order_dayofweek'] = feat_df['parsed_date'].dt.dayofweek
    feat_df['is_weekend'] = feat_df['order_dayofweek'].isin([5, 6]).astype(int)
    feat_df['order_day'] = feat_df['parsed_date'].dt.day
    feat_df['order_month'] = feat_df['parsed_date'].dt.month

    # Geographical encoding (top tiers + other)
    top_states = feat_df['ship-state'].value_counts().nlargest(12).index.tolist()
    feat_df['state_grouped'] = feat_df['ship-state'].apply(
        lambda s: s if s in top_states else 'OTHER'
    )

    # Category and Size cleaning
    top_categories = feat_df['Category'].value_counts().nlargest(8).index.tolist()
    feat_df['category_grouped'] = feat_df['Category'].apply(
        lambda c: c if c in top_categories else 'OTHER'
    )

    top_sizes = ['S', 'M', 'L', 'XL', 'XXL', '3XL', 'XS']
    feat_df['size_grouped'] = feat_df['Size'].apply(
        lambda sz: sz if sz in top_sizes else 'OTHER'
    )

    # Financial and Order metrics
    feat_df['amount_clean'] = feat_df['Amount'].fillna(0.0)
    feat_df['qty_clean'] = feat_df['Qty'].clip(lower=1)
    feat_df['unit_price'] = (feat_df['amount_clean'] / feat_df['qty_clean']).round(2)
    
    # Flags
    feat_df['is_b2b'] = feat_df['B2B'].astype(int)
    feat_df['is_expedited'] = (feat_df['ship-service-level'] == 'Expedited').astype(int)
    feat_df['is_amazon_fulfilled'] = (feat_df['Fulfilment'] == 'Amazon').astype(int)

    return feat_df


def prepare_leakage_training_data(
    df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, list[str]]:
    """Filters to settled orders and splits into train/test sets."""
    # Settle criteria: delivered or leaked (cancelled/returned/lost)
    settled_mask = df['financial_bucket'].isin(
        ['Net Realized', 'Cancellation Leakage', 'Return Leakage', 'Loss Leakage']
    )
    settled_df = df[settled_mask].copy()

    # Binary Target: 1 if Leaked (cancellation/return/loss), 0 if Net Realized
    settled_df['is_leakage'] = (settled_df['financial_bucket'] != 'Net Realized').astype(int)

    features = [
        'order_dayofweek',
        'is_weekend',
        'order_day',
        'amount_clean',
        'qty_clean',
        'unit_price',
        'is_b2b',
        'is_expedited',
        'state_grouped',
        'category_grouped',
        'size_grouped',
    ]

    X = settled_df[features].copy()
    y = settled_df['is_leakage'].copy()

    # One-hot encode categorical features
    cat_cols = ['state_grouped', 'category_grouped', 'size_grouped']
    X_encoded = pd.get_dummies(X, columns=cat_cols, drop_first=True)

    feature_names = X_encoded.columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y, test_size=test_size, random_state=random_state, stratify=y
    )

    return X_train, X_test, y_train, y_test, feature_names
