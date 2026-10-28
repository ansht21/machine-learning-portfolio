"""Train churn models with MLflow tracking; save model + metrics.
Run:  python src/train.py   (from 02_subscription-retention-prediction/)
"""
import json
import sys
from pathlib import Path

import joblib
import pandas as pd

try:
    import mlflow
    import mlflow.sklearn
    HAS_MLFLOW = True
except ImportError:
    HAS_MLFLOW = False


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import DATA, MODELS_DIR, OUTPUTS_DIR, RANDOM_STATE
from src.data_processing import load_and_clean
from src.modeling import (make_splits, fit_models, evaluate, cross_validate,
                          optimize_threshold, plot_threshold_analysis, HAS_XGB)

MODELS_DIR.mkdir(exist_ok=True)
OUTPUTS_DIR.mkdir(exist_ok=True)
if HAS_MLFLOW:
    mlflow.set_tracking_uri("file:" + str(Path("mlruns").resolve()))
    mlflow.set_experiment("churn-prediction")


def main():
    df = load_and_clean(DATA)
    splits = make_splits(df)
    run = mlflow.start_run(run_name="logreg-vs-rf-vs-xgb") if HAS_MLFLOW else None
    try:
        if HAS_MLFLOW:
            mlflow.log_param("test_size", 0.2)
            mlflow.log_param("random_state", RANDOM_STATE)
        models = fit_models(splits)
        rows = []
        eval_targets = [
            ("Logistic Regression", "Logistic Regression", True),
            ("Random Forest", "Random Forest", False),
        ]
        if HAS_XGB and "XGBoost" in models:
            eval_targets.append(("XGBoost", "XGBoost", False))
        for name, key, scaled in eval_targets:
            m = evaluate(name, models[key], splits["X_test"], splits["y_test"],
                         scaled=scaled, scaler=models["scaler"])
            if HAS_MLFLOW:
                mlflow.log_metrics({f"{name}/accuracy": m["accuracy"],
                                    f"{name}/roc_auc": m["roc_auc"],
                                    f"{name}/f1": m["f1"]})
            rows.append({k: (round(v, 4) if isinstance(v, float) else v)
                         for k, v in m.items() if k != "confusion_matrix"})
        cv = cross_validate(splits, cv_folds=5)
        pd.DataFrame(rows).to_csv(OUTPUTS_DIR / "churn_model_metrics.csv", index=False)
        # Threshold selection happens on VALIDATION data only (cost-based:
        # fp*1 + fn*5). Test data stays untouched for final reporting.
        prob_val = models["Random Forest"].predict_proba(splits["X_val"])[:, 1]
        thr = optimize_threshold(splits["y_val"].values, prob_val)

        # Threshold analysis plot — also computed on VALIDATION (leak-free)
        image_dir = OUTPUTS_DIR.parent / "image"
        image_dir.mkdir(exist_ok=True)
        plot_threshold_analysis(
            splits["y_val"].values,
            prob_val,
            image_dir / "threshold_analysis.png"
        )

        bundle = {"model": models["Random Forest"], "columns": splits["columns"],
                  "threshold": thr["cost_optimal"], "high_risk": 0.7, "medium_risk": 0.4,
                  "monthly_charges_avg": splits["monthly_charges_avg"]}
        joblib.dump(bundle, MODELS_DIR / "churn_model.pkl")
        with open(MODELS_DIR / "meta.json", "w") as f:
            json.dump({"final_model": "Random Forest",
                       "threshold": thr["cost_optimal"],
                       "threshold_note": ("decision threshold selected on the VALIDATION "
                                          "split by cost optimization (fp=1, fn=5); the test "
                                          "split is only used for final reported metrics"),
                       "risk_bands": {"note": ("bands are descriptive severity tiers used "
                                               "by the API/UI and the AI offer trigger, "
                                               "independent of the binary decision threshold"),
                                      "high_risk": 0.7, "medium_risk": 0.4},
                       "monthly_charges_avg": splits["monthly_charges_avg"],
                       "splits": "60/20/20 train/val/test (stratified)"},
                      f, indent=2)
        if HAS_MLFLOW:
            mlflow.sklearn.log_model(models["Random Forest"], name="random-forest",
                                     registered_model_name="churn-rf",
                                     skops_trusted_types=["sklearn.tree._tree.Tree"])
            mlflow.log_artifact(str(MODELS_DIR / "meta.json"))
        print(f"saved model + metrics | thr={thr['cost_optimal']:.3f}")
        print(pd.DataFrame(rows).to_string(index=False))
    finally:
        if run is not None:
            mlflow.end_run()


if __name__ == "__main__":
    main()
