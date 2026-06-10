from src.agentrag.agent.graph_service import _chain_query


def test_chain_query_prepends_prior_snippet():
    prior = {"hits": [{"content": "Aspirin ức chế kết tập tiểu cầu, dùng trong NMCT."}]}
    chained = _chain_query("Liều dùng là bao nhiêu?", prior)
    assert "Aspirin" in chained
    assert "Liều dùng" in chained


def test_chain_query_no_prior_returns_original():
    assert _chain_query("Câu hỏi gốc", None) == "Câu hỏi gốc"
    assert _chain_query("Câu hỏi gốc", {"hits": []}) == "Câu hỏi gốc"
