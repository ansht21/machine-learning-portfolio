"""Forecast anomaly guard: statistical trigger -> LLM operations brief.

Monitors a SARIMA forecast's 95% confidence interval: when actual observed
demand breaches the upper bound, a zero-shot LLM chain converts the breach
numbers into a Markdown tactical action brief. Works with any OpenAI-
compatible endpoint — the documented free option is Groq (see .env.example).

Stand-alone usage:
    python forecast_guard.py            # demo with hardcoded sanity-check case
    from forecast_guard import ForecastAnomalyAnalyzer
"""
import os
import textwrap


class ForecastAnomalyAnalyzer:
    """Zero-shot LLM brief writer, triggered only by the statistical gate."""

    def __init__(self):
        self._chain = None

    @staticmethod
    def is_anomaly(actual: float, expected: float, upper: float) -> float | None:
        """Return the variance % if the upper CI is breached, else None."""
        if upper <= 0:
            raise ValueError("confidence upper bound must be positive")
        return ((actual - expected) / expected * 100.0
                if actual > upper else None)

    def _llm(self):
        model = os.getenv("GENAI_MODEL", "openai/gpt-oss-120b")
        base = os.getenv("GENAI_BASE_URL")
        if os.getenv("GROQ_API_KEY"):
            return {"model": model, "api_key": os.environ["GROQ_API_KEY"],
                    "base_url": base or "https://api.groq.com/openai/v1"}
        if os.getenv("OPENAI_API_KEY"):
            return {"model": "gpt-4o-mini", "api_key": os.environ["OPENAI_API_KEY"],
                    "base_url": base}
        return None

    @property
    def configured(self) -> bool:
        return self._llm() is not None

    def evaluate_variance_event(self, timestamp: str, expected: float,
                                actual: float, upper: float) -> str:
        """Return a Markdown operations brief, or a safe-status message."""
        var_pct = self.is_anomaly(actual, expected, upper)
        if var_pct is None:
            return "Metrics remain safely within the expected 95% forecast bands."

        if not self.configured:
            return textwrap.dedent("""\
                ## Structural alert — LLM disabled (no API key)

                Observed demand **{a:.0f} trips/hr** vs SARIMA expectation of
                **{e:.0f}** (+{v:.1f}%), breaching the 95% upper bound of {u:.0f}.

                *Manual baseline guidance:* verify surge multiplier, review
                driver supply in the affected zone, and recompute the hourly
                forecast band.

                Set GROQ_API_KEY to enable automated brief generation.""".format(
                    a=actual, e=expected, v=var_pct, u=upper))

        from langchain_core.prompts import PromptTemplate
        from langchain_openai import ChatOpenAI

        prompt = PromptTemplate(
            template=textwrap.dedent("""\
                You are a senior ride-hailing operations analyst writing for the on-shift city manager.
                A SARIMA demand model has breached its 95% upper confidence interval.

                - Hour: {time}
                - Predicted demand: {expected_val:.0f} trips
                - Actual observed demand: {actual_val:.0f} trips
                - Variance: +{var_pct:.1f}% above expectation
                - 95% upper bound that was breached: {upper_lim:.0f} trips
                - Forecast quality context: the model holds an 8.3% MAPE, so this
                  breach is outside normal error ranges
                - Surge-pricing and premium-vehicle levers are available.

                Write a Markdown tactical brief (max ~150 words) with:
                ## Surge strategy
                ## Vehicle tier allocation
                ## Fleet deployment
                Reference the numbers above directly. No preamble, no closing chat."""),
            input_variables=["time", "expected_val", "actual_val",
                             "var_pct", "upper_lim"])
        chain = prompt | ChatOpenAI(model=self._llm()["model"],
                                    api_key=self._llm()["api_key"],
                                    base_url=self._llm()["base_url"],
                                    temperature=0.0, max_retries=1)
        out = chain.invoke({"time": timestamp, "expected_val": expected,
                            "actual_val": actual, "var_pct": var_pct,
                            "upper_lim": upper})
        return out.content


if __name__ == "__main__":
    test = ForecastAnomalyAnalyzer()
    print("anomaly test:", test.is_anomaly(220, 100, 140))
    print("normal test :", test.is_anomaly(120, 100, 140))
    print(test.evaluate_variance_event("2026-10-25 18:00", 100, 220, 140))
