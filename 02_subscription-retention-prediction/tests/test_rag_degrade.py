"""RAG-layer degradation tests (item-5 fix: predictable behaviour without index)."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def test_vector_index_not_ready_when_missing(monkeypatch, tmp_path):
    """An empty/missing index directory must read as not-ready, not ready."""
    import app.retention_rag as rag

    fake_db = tmp_path / "churn_rag_db"          # exists but empty (no chroma.sqlite3)
    fake_db.mkdir()
    monkeypatch.setattr(rag, "DB_PATH", fake_db)
    monkeypatch.setattr(rag, "DATA_PATH", ROOT / "data" / "Telco_Customer_Churn.csv")
    idx = rag.RetentionVectorIndex()
    assert idx.ready is False


def test_vector_index_ready_reports_true_without_deps(monkeypatch, tmp_path):
    import app.retention_rag as rag
    # missing folder entirely -> not ready (no exceptions escape)
    monkeypatch.setattr(rag, "DB_PATH", tmp_path / "nope")
    monkeypatch.setattr(rag, "DATA_PATH", tmp_path / "nope.csv")
    idx = rag.RetentionVectorIndex()
    assert idx.ready is False
    assert idx.count() == 0
    assert idx.query_similar({"tenure": 1, "MonthlyCharges": 85}) == []
