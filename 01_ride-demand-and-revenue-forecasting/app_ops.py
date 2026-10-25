"""Ride demand operations portal (Streamlit).

Run:  streamlit run app_ops.py --server.port 8502
Left panel: observed demand vs SARIMA expectation + 95% upper bound.
Right side: automated LLM tactical brief whenever the bound is breached.
"""
from datetime import datetime

import streamlit as st

from forecast_guard import ForecastAnomalyAnalyzer

st.set_page_config(page_title="Ride Demand Operations", page_icon="🚖", layout="wide")
st.title("🚖 Ride Demand Forecast Guard — SARIMA + LLM Ops Brief")
st.caption("SARIMA forecast (MAPE 8.3%) detects the anomaly · the LLM writes the response")

with st.sidebar:
    st.header("Real-time operations intake")
    zone = st.selectbox("Pickup zone (cluster)",
                        ["Cluster 0 — Airport long-haul hub",
                         "Cluster 1 — High-volume commuter cell"])
    actual = st.slider("Observed trips this hour", 10, 500, 220)
    expected = st.number_input("SARIMA expected demand", 10, 400, 100)
    upper = st.number_input("SARIMA 95% upper bound", 10, 450, 140)
    audit = st.button("Audit network volatility", type="primary", use_container_width=True)

if audit:
    st.subheader(f"Hour analysis — {zone}")
    var_pct = ((actual - expected) / expected) * 100
    c1, c2, c3 = st.columns(3)
    c1.metric("Observed", f"{actual} trips")
    c2.metric("SARIMA expected", f"{expected:.0f} trips")
    c3.metric("Variance", f"+{var_pct:.1f}%")

    analyzer = ForecastAnomalyAnalyzer()
    breached = analyzer.is_anomaly(actual, float(expected), float(upper))

    if breached is not None:
        st.warning("🚨 ALERT — demand breached the 95% upper confidence bound.")
        st.subheader("🤖 Automated tactical brief")
        st.markdown(analyzer.evaluate_variance_event(
            datetime.now().strftime("%Y-%m-%d %H:00"),
            float(expected), float(actual), float(upper)))
    else:
        st.success("✅ Demand sits inside the 95% forecast band — no agent action.")

st.divider()
st.markdown("**How this works:** the statistical gate is the SARIMA model's own "
            "confidence interval — the LLM only ever fires when the math says "
            "something is structurally abnormal (decision is math; communication is AI).")
