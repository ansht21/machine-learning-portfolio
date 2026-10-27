"""LLM/RAG reliability eval for the Retention Gateway.
Tests schema validation, customer description formatting, vector index state,
and graceful degradation — all without requiring an API key or running server.
Run:  python eval_llm.py
"""
import sys
import os
from pathlib import Path

# Make sure the project root is on the path so `app` and `src` are importable.
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

passed = 0
total = 5
results = []


def run_test(number: int, name: str, fn):
    global passed
    try:
        fn()
        passed += 1
        results.append(f"  [PASS] Test {number}: {name}")
    except Exception as exc:
        results.append(f"  [FAIL] Test {number}: {name} — {exc}")


# ---------------------------------------------------------------------------
# Test 1 — RetentionStrategy schema: valid data accepted, invalid rejected
# ---------------------------------------------------------------------------
def test_retention_strategy_schema():
    from app.retention_rag import RetentionStrategy

    # Valid data should be accepted
    valid = RetentionStrategy(
        primary_grievance="Price shock on month-to-month fibre plan.",
        incentive_category="Rate Waiver",
        personalized_offer_text="10% off for 6 months.",
        dollar_cost_impact=9.50,
    )
    assert valid.incentive_category == "Rate Waiver", "Valid category should be stored"

    # Invalid incentive_category must be rejected by Pydantic
    rejected = False
    try:
        from pydantic import ValidationError
        RetentionStrategy(
            primary_grievance="test",
            incentive_category="Free Pizza",          # not in Literal
            personalized_offer_text="test offer",
            dollar_cost_impact=0.0,
        )
    except Exception:
        rejected = True
    assert rejected, "Pydantic must reject an invalid incentive_category"


run_test(1, "RetentionStrategy schema validation", test_retention_strategy_schema)


# ---------------------------------------------------------------------------
# Test 2 — _describe_customer: output contains expected key fields
# ---------------------------------------------------------------------------
def test_describe_customer():
    from app.retention_rag import _describe_customer

    customer = {
        "tenure": 6,
        "Contract": "Month-to-month",
        "InternetService": "Fiber optic",
        "PhoneService": "Yes",
        "OnlineSecurity": "No",
        "TechSupport": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 89.95,
        "TotalCharges": 539.70,
        "Churn": "Yes",
    }
    description = _describe_customer(customer)
    assert isinstance(description, str), "_describe_customer must return a string"
    for field in ("6", "Month-to-month", "Fiber optic", "89.95"):
        assert field in description, f"Expected '{field}' to appear in description"
    assert "left" in description or "stayed" in description, \
        "Description must include outcome (left/stayed)"


run_test(2, "_describe_customer formatting", test_describe_customer)


# ---------------------------------------------------------------------------
# Test 3 — RetentionVectorIndex.ready: returns False when DB_PATH absent
# ---------------------------------------------------------------------------
def test_vector_index_ready_without_db():
    from app.retention_rag import RetentionVectorIndex, DB_PATH

    # Temporarily point DB_PATH to a path that definitely doesn't exist
    import app.retention_rag as rag_module
    original = rag_module.DB_PATH
    try:
        rag_module.DB_PATH = Path("/nonexistent_path_eval_llm_test/churn_rag_db")
        idx = RetentionVectorIndex()
        assert idx.ready is False, \
            "RetentionVectorIndex.ready must be False when DB_PATH does not exist"
    finally:
        rag_module.DB_PATH = original


run_test(3, "RetentionVectorIndex.ready without DB", test_vector_index_ready_without_db)


# ---------------------------------------------------------------------------
# Test 4 — RetentionOfferGenerator.configured: False without env vars
# ---------------------------------------------------------------------------
def test_generator_not_configured_without_key():
    from app.retention_rag import RetentionOfferGenerator

    # Temporarily remove known API-key env vars
    saved = {}
    for key in ("GROQ_API_KEY", "OPENAI_API_KEY"):
        saved[key] = os.environ.pop(key, None)
    try:
        gen = RetentionOfferGenerator()
        assert gen.configured is False, \
            "RetentionOfferGenerator.configured must be False without API keys"
    finally:
        for key, val in saved.items():
            if val is not None:
                os.environ[key] = val


run_test(4, "RetentionOfferGenerator.configured without API key", test_generator_not_configured_without_key)


# ---------------------------------------------------------------------------
# Test 5 — Graceful degradation: FastAPI app importable and has expected routes
# ---------------------------------------------------------------------------
def test_fastapi_app_importable():
    import app.main as main_module

    app_obj = main_module.app
    assert app_obj is not None, "app.main.app must exist"

    # Collect all registered route paths
    routes = {route.path for route in app_obj.routes if hasattr(route, "path")}
    for expected in ("/health", "/predict", "/predict/v2"):
        assert expected in routes, f"Route '{expected}' must be registered in the FastAPI app"


run_test(5, "FastAPI app importable with /health /predict /predict/v2", test_fastapi_app_importable)


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print("\nLLM/RAG Reliability Eval")
print("=" * 40)
for line in results:
    print(line)
print("=" * 40)
print(f"{passed}/{total} tests passed")

if passed < total:
    sys.exit(1)
