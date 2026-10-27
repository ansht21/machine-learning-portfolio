"""LLM Reliability Eval for ForecastAnomalyAnalyzer.

What it tests
-------------
1. Anomaly detection logic (is_anomaly) — pure unit tests, no LLM needed.
2. Graceful degradation — evaluate_variance_event output when no API key is set:
   checks that the no-key fallback path returns a correctly formatted message
   containing expected keywords and the actual demand number.
3. Edge cases — exact-boundary and invalid-input behaviour.

How to run
----------
    python eval_llm.py

No API key is required.  The script covers both the statistical gate (unit tests
on is_anomaly) and the no-key fallback path of evaluate_variance_event.
Expected output: 5/5 tests passed.
"""

import os
import sys

# ---------------------------------------------------------------------------
# Ensure no API key is active so evaluate_variance_event uses the fallback path
# ---------------------------------------------------------------------------
for _k in ("GROQ_API_KEY", "OPENAI_API_KEY"):
    os.environ.pop(_k, None)

from forecast_guard import ForecastAnomalyAnalyzer  # noqa: E402 (after env clear)

# ---------------------------------------------------------------------------
# Golden test-case definitions
# ---------------------------------------------------------------------------
GOLDEN_CASES = [
    {
        "id": 1,
        "name": "Anomaly detected (actual > upper)",
        "description": (
            "is_anomaly should return a positive float when actual breaches upper."
        ),
    },
    {
        "id": 2,
        "name": "Normal reading (actual < upper)",
        "description": "is_anomaly should return None when actual is below upper.",
    },
    {
        "id": 3,
        "name": "Edge case — actual exactly equals upper",
        "description": (
            "is_anomaly should return None when actual == upper "
            "(breach is strict: actual > upper only)."
        ),
    },
    {
        "id": 4,
        "name": "LLM output format — no-API-key fallback",
        "description": (
            "evaluate_variance_event should return a message containing "
            "'Structural alert' and the actual demand number when no API key is set."
        ),
        "timestamp": "2026-10-25 18:00",
        "expected_val": 100.0,
        "actual_val": 300.0,
        "upper": 140.0,
        "expected_keywords": ["Structural alert", "300"],
    },
    {
        "id": 5,
        "name": "Invalid input — upper bound zero",
        "description": "is_anomaly with upper=0 should raise ValueError.",
    },
]

# ---------------------------------------------------------------------------
# Test runner helpers
# ---------------------------------------------------------------------------
analyzer = ForecastAnomalyAnalyzer()
passed = 0
failed = 0
results = []


def record(test_id: int, name: str, ok: bool, detail: str = "") -> None:
    global passed, failed
    status = "PASS" if ok else "FAIL"
    if ok:
        passed += 1
    else:
        failed += 1
    msg = f"[{status}] Test {test_id}: {name}"
    if detail:
        msg += f"\n       Detail: {detail}"
    results.append(msg)
    print(msg)


# ---------------------------------------------------------------------------
# Test 1 — Anomaly detected (actual > upper)
# ---------------------------------------------------------------------------
tc = GOLDEN_CASES[0]
try:
    result = analyzer.is_anomaly(actual=220, expected=100, upper=140)
    ok = result is not None and result > 0
    record(
        tc["id"],
        tc["name"],
        ok,
        f"is_anomaly returned {result!r} (expected non-None positive float)",
    )
except Exception as exc:
    record(tc["id"], tc["name"], False, f"Raised unexpected exception: {exc}")

# ---------------------------------------------------------------------------
# Test 2 — Normal reading (actual < upper)
# ---------------------------------------------------------------------------
tc = GOLDEN_CASES[1]
try:
    result = analyzer.is_anomaly(actual=120, expected=100, upper=140)
    ok = result is None
    record(
        tc["id"],
        tc["name"],
        ok,
        f"is_anomaly returned {result!r} (expected None)",
    )
except Exception as exc:
    record(tc["id"], tc["name"], False, f"Raised unexpected exception: {exc}")

# ---------------------------------------------------------------------------
# Test 3 — Edge case: actual == upper (should NOT trigger, strict >)
# ---------------------------------------------------------------------------
tc = GOLDEN_CASES[2]
try:
    result = analyzer.is_anomaly(actual=140, expected=100, upper=140)
    ok = result is None
    record(
        tc["id"],
        tc["name"],
        ok,
        f"is_anomaly returned {result!r} (expected None — boundary is not breached)",
    )
except Exception as exc:
    record(tc["id"], tc["name"], False, f"Raised unexpected exception: {exc}")

# ---------------------------------------------------------------------------
# Test 4 — LLM fallback output format (no API key)
# ---------------------------------------------------------------------------
tc = GOLDEN_CASES[3]
try:
    output = analyzer.evaluate_variance_event(
        timestamp=tc["timestamp"],
        expected=tc["expected_val"],
        actual=tc["actual_val"],
        upper=tc["upper"],
    )
    missing = [
        kw for kw in tc["expected_keywords"] if kw.lower() not in output.lower()
    ]
    ok = len(missing) == 0
    detail = (
        f"output contains all expected keywords: {tc['expected_keywords']}"
        if ok
        else f"missing keywords: {missing}\n       Output snippet: {output[:200]!r}"
    )
    record(tc["id"], tc["name"], ok, detail)
except Exception as exc:
    record(tc["id"], tc["name"], False, f"Raised unexpected exception: {exc}")

# ---------------------------------------------------------------------------
# Test 5 — Invalid input: upper=0 must raise ValueError
# ---------------------------------------------------------------------------
tc = GOLDEN_CASES[4]
try:
    analyzer.is_anomaly(actual=100, expected=80, upper=0)
    # If we reach here, the expected exception was NOT raised — test fails
    record(tc["id"], tc["name"], False, "Expected ValueError was not raised")
except ValueError as exc:
    record(tc["id"], tc["name"], True, f"Correctly raised ValueError: {exc}")
except Exception as exc:
    record(
        tc["id"],
        tc["name"],
        False,
        f"Expected ValueError but got {type(exc).__name__}: {exc}",
    )

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
total = passed + failed
print()
print("=" * 50)
print(f"  Summary: {passed}/{total} tests passed")
print("=" * 50)

if failed:
    sys.exit(1)
