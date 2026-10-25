"""RAG retention layer.

Retrieves the most semantically similar historical customers from the Telco
dataset (ChromaDB + all-MiniLM-L6-v2 embeddings) and asks the LLM to turn the
evidence + fresh model prediction into a structured retention offer.

The LLM runs on any OpenAI-compatible endpoint configured in the environment:
GROQ_API_KEY is the documented free option (https://console.groq.com); an
OPENAI_API_KEY works as well. Everything degrades gracefully: without a key the
API still returns the plain ML prediction.

Run the one-off indexer first:  python -m app.build_vector_index
"""
from pathlib import Path
from typing import Any, Dict, List, Literal

from pydantic import BaseModel, Field

# NOTE: langchain imports are deferred (see _get_llm_deps) so that /predict and
# /health keep working in a minimal install without the GenAI stack.

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "Telco_Customer_Churn.csv"
DB_PATH = ROOT / "models" / "churn_rag_db"

EMBED_MODEL = "all-MiniLM-L6-v2"

COVERAGE_CATEGORIES = {
    "Rate Waiver": "price-sensitive customer, month-to-month contract, electronic check",
    "Technical Audit": "TechSupport / OnlineSecurity gaps, fibre-optic reliability issues",
    "Plan Upgrade": "low tenure, single-service customer with unused add-on services",
    "Competitor Price Match": "high monthly charges relative to similar historical churners",
}


class RetentionStrategy(BaseModel):
    """Strict contract for everything the LLM is allowed to say."""
    primary_grievance: str = Field(
        description="The most likely underlying issue driving churn, grounded in the retrieved historical evidence.")
    incentive_category: Literal["Rate Waiver", "Technical Audit",
                                "Plan Upgrade", "Competitor Price Match"]
    personalized_offer_text: str = Field(
        description="A short, specific retention pitch referencing that evidence. No invented facts.")
    dollar_cost_impact: float = Field(
        description="Estimated monthly revenue cost of the offer in USD.")


def _describe_customer(r: Dict[str, Any]) -> str:
    """Turn one raw Telco row into a natural-language evidence document."""
    churned = "left" if r.get("Churn", "No") == "Yes" else "stayed"
    return (
        f"{r.get('tenure', '?')}-month customer: {r.get('Contract', '?')} contract, "
        f"{r.get('InternetService', '?')} internet, PhoneService={r.get('PhoneService', '?')}, "
        f"OnlineSecurity={r.get('OnlineSecurity', '?')}, TechSupport={r.get('TechSupport', '?')}, "
        f"OnlineBackup={r.get('OnlineBackup', '?')}, DeviceProtection={r.get('DeviceProtection', '?')}, "
        f"PaperlessBilling={r.get('PaperlessBilling', '?')}, "
        f"payment via {r.get('PaymentMethod', '?')}, "
        f"charges ${r.get('MonthlyCharges', '?')}/mo (total ${r.get('TotalCharges', '?')}); "
        f"outcome: customer {churned}."
    )


class RetentionVectorIndex:
    """Thin wrapper around a persistent Chroma collection Telco rows."""

    def __init__(self):
        self._db = None

    @property
    def ready(self) -> bool:
        return DB_PATH.exists() and DATA_PATH.exists()

    def _open(self):
        if self._db is None:
            _, _, _ = self._deps()
            from langchain_community.embeddings import HuggingFaceEmbeddings
            from langchain_community.vectorstores import Chroma
            self._db = Chroma(
                persist_directory=str(DB_PATH),
                embedding_function=HuggingFaceEmbeddings(model_name=EMBED_MODEL),
            )
        return self._db

    def _deps(self):
        try:
            from langchain_core.output_parsers import JsonOutputParser  # noqa: F401
            return True, None, None
        except ImportError as e:  # pragma: no cover
            return False, str(e), None

    def query_similar(self, features: Dict[str, Any], k: int = 3) -> List[Dict[str, Any]]:
        """Retrieve historical customers whose profile is most similar to the
        incoming one — this is the 'grounding context' for the offer."""
        if not self.ready:
            return []
        db = self._open()
        docs = db.similarity_search(_describe_customer(features), k=k)
        out = []
        for d in docs:
            out.append({"evidence": d.page_content, "meta": d.metadata})
        return out


class RetentionOfferGenerator:
    """LangChain chain: prediction + retrieved evidence -> strict JSON offer."""

    def __init__(self):
        self._chain = None

    def _llm(self):
        import os
        model = os.getenv("GENAI_MODEL", "openai/gpt-oss-120b")
        base = os.getenv("GENAI_BASE_URL")
        if os.getenv("GROQ_API_KEY"):
            return {"model": model,
                    "api_key": os.environ["GROQ_API_KEY"],
                    "base_url": base or "https://api.groq.com/openai/v1"}
        if os.getenv("OPENAI_API_KEY"):
            return {"model": base and model or "gpt-4o-mini",
                    "api_key": os.environ["OPENAI_API_KEY"],
                    "base_url": base}
        return None

    @property
    def configured(self) -> bool:
        return self._llm() is not None

    def generate(self, customer: Dict[str, Any], proba: float,
                 evidence: List[Dict[str, Any]]) -> Dict[str, Any]:
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_core.output_parsers import JsonOutputParser

        llm_cfg = self._llm()
        parser = JsonOutputParser(pydantic_object=RetentionStrategy)
        evidence_text = "\n".join(
            f"- {e['evidence']}" for e in evidence) or "No similar historical profile retrieved."

        prompt = ChatPromptTemplate.from_messages([
            ("system",
             "You are a customer-retention analyst. Use ONLY the retrieved historical "
             "evidence and the given prediction to design a retention offer. Never "
             "invent facts about the customer. Follow exactly these fields:\n"
             "{format_instructions}\n"
             "Allowed incentive_category values: "
             + ", ".join(COVERAGE_CATEGORIES)
             + ".\nMatching heuristics:\n" + "\n".join(
                 f"- {k}: {v}" for k, v in COVERAGE_CATEGORIES.items())),
            ("human",
             "Churn probability from the classifier: {proba:.2f}\n"
             "Customer profile: {profile}\n\n"
             "Retrieved historical evidence:\n{evidence}\n\n"
             "Produce the retention package as strict JSON."),
        ])
        from langchain_openai import ChatOpenAI
        chain = (
            prompt.partial(format_instructions=parser.get_format_instructions())
            | ChatOpenAI(model=llm_cfg["model"], api_key=llm_cfg["api_key"],
                         base_url=llm_cfg["base_url"], temperature=0.1,
                         max_retries=1)
            | parser
        )
        raw = chain.invoke({"proba": proba,
                            "profile": _describe_customer(customer),
                            "evidence": evidence_text})
        # Clamp to what we actually know
        raw["churn_probability"] = round(float(proba), 4)
        return raw
