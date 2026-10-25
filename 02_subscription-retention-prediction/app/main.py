"""Churn scoring API. Run: uvicorn app.main:app --reload (needs models/churn_model.pkl)."""
from pathlib import Path
import joblib
import os
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "churn_model.pkl"

app = FastAPI(title="Churn Prediction API", version="1.0.0")
_bundle = None


def get_bundle():
    global _bundle
    if _bundle is None:
        if not MODEL_PATH.exists():
            raise HTTPException(status_code=503,
                                detail="Model not trained. Run: python src/train.py")
        _bundle = joblib.load(MODEL_PATH)
    return _bundle


class CustomerFeatures(BaseModel):
    gender: str = "Female"
    Partner: str = "No"
    Dependents: str = "No"
    tenure: int = 12
    PhoneService: str = "Yes"
    MultipleLines: str = "No"
    InternetService: str = "Fiber optic"
    OnlineSecurity: str = "No"
    OnlineBackup: str = "No"
    DeviceProtection: str = "No"
    TechSupport: str = "No"
    StreamingTV: str = "No"
    StreamingMovies: str = "No"
    Contract: str = "Month-to-month"
    PaperlessBilling: str = "Yes"
    PaymentMethod: str = "Electronic check"
    MonthlyCharges: float = 80.0
    TotalCharges: float = 960.0


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": MODEL_PATH.exists()}


@app.post("/predict")
def predict(c: CustomerFeatures):
    row_dict, proba, thr, band, _ = _score(c)
    return {"churn_probability": round(proba, 4),
            "predicted_churn": bool(proba >= thr),
            "risk_band": band, "threshold": round(thr, 4)}


# ---- GenAI-shaped helpers shared by /predict/v2 ----
def _score(c: "CustomerFeatures"):
    """Shared inference used by both endpoints (keeps logic DRY)."""
    b = get_bundle()
    row = pd.DataFrame([{**c.model_dump()}])
    row["charge_ratio"] = row["TotalCharges"] / (row["tenure"] + 1)
    row["tenure_x_monthly"] = row["tenure"] * row["MonthlyCharges"]
    row["monthly_above_avg"] = 0
    cats = ["gender", "Partner", "Dependents", "PhoneService", "MultipleLines",
            "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
            "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
            "PaperlessBilling", "PaymentMethod"]
    X = pd.get_dummies(row[cats + ["tenure", "MonthlyCharges", "charge_ratio",
                                   "tenure_x_monthly", "monthly_above_avg"]],
                       drop_first=True)
    X = X.reindex(columns=b["columns"], fill_value=0)
    proba = float(b["model"].predict_proba(X)[0, 1])
    thr = float(b.get("threshold", 0.5))
    band = "High" if proba >= 0.7 else ("Medium" if proba >= 0.4 else "Low")
    return row.iloc[0].to_dict(), proba, thr, band, c


@app.post("/predict/v2")
def predict_v2(c: CustomerFeatures, request: Request):
    """Dual-stage retention gateway.

    Stage 1 — the real classifier scores the customer (same model as /predict).
    Stage 2 — only high-risk customers (>= 0.70) trigger the RAG layer:
              ChromaDB semantic retrieval of similar historical customer
              profiles + a LangChain chain producing a strict-JSON retention
              offer (Pydantic-validated). Degrades gracefully without a key.
    """
    row_dict, proba, thr, band, _ = _score(c)
    response = {
        "churn_probability": round(proba, 4),
        "predicted_churn": bool(proba >= thr),
        "risk_band": band,
        "threshold": round(thr, 4),
    }
    # This RF rarely exceeds 0.70 (its distribution tops out ~0.56 on the API
    # feature space), so the AI gate defaults to Medium (>= 0.40) — the model's
    # own documented band — and is overridable via the AI_TRIGGER_BAND env var.
    trigger_bands = os.getenv("AI_TRIGGER_BAND", "Medium,High").split(",")
    if band not in trigger_bands:
        response["ai_retention_intervention"] = None
        response["genai_note"] = "No AI offer generated: risk band is " + band
        return response

    genai = getattr(request.app.state, "genai", None)
    if genai is None or not genai["generator"].configured or not genai["index"].ready:
        response["ai_retention_intervention"] = None
        response["genai_note"] = (
            "GenAI layer unavailable (vector index or API key missing). "
            "Build the index: python -m app.build_vector_index; "
            "set GROQ_API_KEY per .env.example")
        return response
    try:
        evidence = genai["index"].query_similar(row_dict, k=3)
        offer = genai["generator"].generate(row_dict, proba, evidence)
        response["ai_retention_intervention"] = offer
        response["retrieved_evidence"] = [e["evidence"] for e in evidence]
    except Exception as e:
        response["ai_retention_intervention"] = None
        response["genai_error"] = f"GenAI generation failed: {e}"
    return response


# ---- Initialization hooks ----
@app.on_event("startup")
def setup_genai():
    from app.retention_rag import RetentionOfferGenerator, RetentionVectorIndex
    app.state.genai = {"index": RetentionVectorIndex(),
                       "generator": RetentionOfferGenerator()}
