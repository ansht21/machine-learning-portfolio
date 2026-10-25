"""Streamlit retention dashboard.

Talks to the /predict/v2 gateway. Run:
    streamlit run app/frontend.py --server.port 8501
Set API_SERVER_URL to the FastAPI base URL when not running locally.
"""
import os

import requests
import streamlit as st

st.set_page_config(page_title="AI Retention Gateway", page_icon="🤖", layout="wide")
st.title("🤖 Customer Retention Gateway — ML + GenAI")
st.caption("LogReg/RF churn classifier (ROC-AUC 0.839) + RAG-grounded retention offers")

API_BASE = os.getenv("API_SERVER_URL", "http://localhost:8000")
API_URL = f"{API_BASE}/predict/v2"

with st.sidebar:
    st.header("Customer profile")
    tenure = st.slider("Tenure (months)", 0, 72, 5)
    MonthlyCharges = st.number_input("Monthly charge ($)", 10.0, 250.0, 92.0)
    TotalCharges = st.number_input("Total charged ($)", 0.0, 20000.0, tenure * MonthlyCharges)
    Contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    InternetService = st.selectbox("Internet", ["DSL", "Fiber optic", "No"])
    PaymentMethod = st.selectbox(
        "Payment method",
        ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
    TechSupport = st.selectbox("Tech support add-on", ["No", "Yes"])
    OnlineSecurity = st.selectbox("Online security add-on", ["No", "Yes"])
    PaperlessBilling = st.selectbox("Paperless billing", ["Yes", "No"])
    run = st.button("Run gateway", type="primary", use_container_width=True)

payload = {
    "gender": "Female", "Partner": "No", "Dependents": "No", "PhoneService": "Yes",
    "MultipleLines": "No", "StreamingTV": "No", "StreamingMovies": "No",
    "tenure": int(tenure), "MonthlyCharges": float(MonthlyCharges),
    "TotalCharges": float(TotalCharges), "Contract": Contract,
    "InternetService": InternetService, "PaymentMethod": PaymentMethod,
    "TechSupport": TechSupport, "OnlineSecurity": OnlineSecurity,
    "PaperlessBilling": PaperlessBilling,
}

if run:
    with st.spinner("Stage 1: classifier inference..."):
        try:
            r = requests.post(API_URL, json=payload, timeout=60)
            r.raise_for_status()
            result = r.json()
        except requests.exceptions.RequestException as e:
            st.error(f"Gateway unreachable at {API_BASE} — is the API running? ({e})")
            st.stop()

    proba = result.get("churn_probability", 0)
    band = result.get("risk_band", "Low")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("1️⃣ Predictive ML layer")
        st.metric("Churn probability", f"{proba * 100:.1f}%")
        if band == "High":
            st.error(f"Risk band: {band} 🚨")
        elif band == "Medium":
            st.warning(f"Risk band: {band}")
        else:
            st.success(f"Risk band: {band}")
        st.caption("Model: balanced ensemble pipeline · held-out ROC-AUC 0.839 · "
                   "threshold 0.17")

    with col2:
        st.subheader("2️⃣ GenAI retention layer")
        offer = result.get("ai_retention_intervention")
        if offer:
            st.success("RAG intervention executed")
            st.markdown(f"**🩺 Primary grievance:** {offer.get('primary_grievance')}")
            st.markdown(f"**🎁 Incentive:** `{offer.get('incentive_category')}`")
            st.info(f"💬 {offer.get('personalized_offer_text')}")
            st.metric("Offer cost impact",
                      f"${float(offer.get('dollar_cost_impact', 0)):.2f}/mo")
        else:
            st.info(result.get("genai_note", "No AI intervention needed."))

    with st.expander("🔎 Retrieved historical evidence (ChromaDB)"):
        ev = result.get("retrieved_evidence") or []
        if ev:
            for i, e in enumerate(ev, 1):
                st.markdown(f"**{i}.** {e}")
        else:
            st.write("No evidence returned (AI layer not triggered).")

    with st.expander("🧾 Raw API response"):
        st.json(result)
