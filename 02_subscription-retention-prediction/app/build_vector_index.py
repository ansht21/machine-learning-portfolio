"""One-off indexer: builds models/churn_rag_db from the Telco CSV.

Run once:  python -m app.build_vector_index
Needs the GenAI deps: pip install -r requirements-api.txt (GenAI extras included).
"""
from pathlib import Path

import pandas as pd
from langchain_core.documents import Document

from app.retention_rag import DATA_PATH, DB_PATH, EMBED_MODEL, _describe_customer


def main() -> None:
    if not DATA_PATH.exists():
        raise SystemExit(f"Dataset not found: {DATA_PATH}")

    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_community.vectorstores import Chroma

    df = pd.read_csv(DATA_PATH)
    documents = []
    for _, r in df.iterrows():
        row = r.to_dict()
        documents.append(Document(
            page_content=_describe_customer(row),
            metadata={
                "customer_id": str(row.get("customerID", "")),
                "churned": row.get("Churn", "No") == "Yes",
                "tenure": int(row.get("tenure", 0) or 0),
                "contract": str(row.get("Contract", "")),
                "monthly_charges": float(row.get("MonthlyCharges", 0) or 0),
            },
        ))

    DB_PATH.mkdir(parents=True, exist_ok=True)
    db = Chroma.from_documents(
        documents,
        embedding=HuggingFaceEmbeddings(model_name=EMBED_MODEL),
        persist_directory=str(DB_PATH),
    )
    count = getattr(db, "_collection", None)
    n = count.count() if count is not None else len(documents)
    print(f"Indexed {n} customer records into {DB_PATH}")


if __name__ == "__main__":
    main()
