"""Main training and model build pipeline runner."""

import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline.feature_pipeline import (
    engineer_features,
    load_raw_reconciliation_data,
    prepare_leakage_training_data,
)
from src.models.train import (
    train_isolation_forest_anomaly_detector,
    train_leakage_models,
)

def run():
    data_path = PROJECT_ROOT / "data" / "cleaned_transactions.parquet"
    models_dir = PROJECT_ROOT / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    print("Step 1: Loading reconciliation data...")
    df = load_raw_reconciliation_data(data_path)
    print(f"Loaded {len(df):,} transactions.")

    print("Step 2: Engineering features...")
    feat_df = engineer_features(df)

    print("Step 3: Preparing leakage training dataset...")
    X_train, X_test, y_train, y_test, feat_names = prepare_leakage_training_data(feat_df)
    print(f"Train samples: {len(X_train):,}, Test samples: {len(X_test):,}")

    print("Step 4: Training ML risk models...")
    meta = train_leakage_models(X_train, X_test, y_train, y_test, models_dir)
    print("Model Training Complete! Metadata:", meta)

    print("Step 5: Training Isolation Forest Outlier Detector...")
    iso_meta = train_isolation_forest_anomaly_detector(feat_df, models_dir)
    print("Isolation Forest trained:", iso_meta)
    print("All ML artifacts saved to:", models_dir)

if __name__ == "__main__":
    run()
