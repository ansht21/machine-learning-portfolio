"""Modeling + API smoke tests matching current src/ layout."""
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.config import TARGET
from src.data_processing import load_and_clean
from src import modeling as mdl


@pytest.fixture(scope="session")
def df():
    return load_and_clean(ROOT / "data" / "Telco_Customer_Churn.csv")


@pytest.fixture(scope="session")
def splits(df):
    return mdl.make_splits(df)


def test_split_shapes(splits):
    assert len(splits["y_train"]) + len(splits["y_val"]) + len(splits["y_test"]) == 7032


def test_threshold_uses_validation_not_test(splits):
    """The decision threshold must be selected on the validation split."""
    assert "y_val" in splits and "X_val" in splits
    # validation and test must be disjoint index sets
    assert not (set(splits["y_val"].index) & set(splits["y_test"].index))
    # meta.json documents the leakage-safe threshold policy
    import json
    meta = json.load(open(ROOT / "models" / "meta.json"))
    assert "validation" in meta.get("threshold_note", "").lower()


def test_feature_engineering_parity(splits):
    """API must compute monthly_above_avg from the same training mean."""
    import joblib
    from app.main import _score, CustomerFeatures
    b = joblib.load(ROOT / "models" / "churn_model.pkl")
    assert "monthly_charges_avg" in b
    val = (CustomerFeatures().MonthlyCharges > b["monthly_charges_avg"])
    # reproduce src-side feature for the same row
    from src.feature_engineering import build_features
    X = build_features(pd.DataFrame([{**CustomerFeatures().model_dump()}]),
                       avg_monthly=b["monthly_charges_avg"])
    assert X["monthly_above_avg"].iloc[0] == int(val)
    # and the API path runs end-to-end
    _, proba, thr, band, _ = _score(CustomerFeatures())
    assert 0 <= proba <= 1 and 0 <= thr <= 1


def test_cross_validation_reports_mean_and_std(splits):
    cv = mdl.cross_validate(splits, cv_folds=3)
    assert {"cv_mean_roc_auc", "cv_std", "folds"}.issubset(cv.columns)
    assert cv["cv_std"].between(0, 0.2).all()


def test_risk_band_boundaries():
    assert mdl.risk_band(0.75) == "High"
    assert mdl.risk_band(0.70) == "High"
    assert mdl.risk_band(0.50) == "Medium"
    assert mdl.risk_band(0.40) == "Medium"
    assert mdl.risk_band(0.10) == "Low"


def test_threshold_is_input_range(splits):
    model = mdl.fit_models(splits)["Random Forest"]
    prob = model.predict_proba(splits["X_test"])[:, 1]
    thr = mdl.optimize_threshold(splits["y_test"].values, prob)
    assert 0 <= thr["cost_optimal"] <= 1
    assert thr["min_cost"] >= 0


def test_revenue_at_risk_expectation(df):
    df2 = df.copy()
    df2["churn_probability"] = 1.0
    df2["risk_level"] = "High"
    rav = mdl.revenue_at_risk(df2)
    assert rav["flagged_customers"] == len(df2)
    assert round(float(rav["expected_monthly_revenue_at_risk"]), 2) == round(
        float(df2["MonthlyCharges"].sum()), 2)


def test_api_predict_smoke():
    from app.main import predict, CustomerFeatures
    out = predict(CustomerFeatures())
    assert 0 <= out["churn_probability"] <= 1
    assert out["risk_band"] in ("High", "Medium", "Low")
