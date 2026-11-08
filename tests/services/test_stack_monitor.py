from __future__ import annotations

from scripts.ops.stack_monitor import (
    FAIL,
    OK,
    WARN,
    classify_chat_payload,
    classify_deepseek_balance,
    classify_es_health,
    classify_ollama_models,
    classify_rerank_scores,
    classify_tei_device,
    required_ollama_models,
)


def test_chat_error_in_200_body_is_fail():
    # The live /chat returns {"error": ...} with HTTP 200.
    status, detail = classify_chat_payload({"error": "APIConnectionError"})
    assert status == FAIL
    assert "APIConnectionError" in detail


def test_chat_empty_answer_is_fail():
    assert classify_chat_payload({"answer": "  ", "citations": [{"page": 1}]})[0] == FAIL


def test_chat_answer_without_citations_is_warn():
    assert classify_chat_payload({"answer": "Xin chào", "citations": []})[0] == WARN


def test_chat_grounded_answer_is_ok():
    assert classify_chat_payload({"answer": "Nguyên nhân...", "citations": [{"page": "22-24"}]})[0] == OK


def test_chat_non_object_is_fail():
    assert classify_chat_payload("Internal Server Error")[0] == FAIL


def test_es_single_node_yellow_is_ok():
    assert classify_es_health({"status": "yellow", "number_of_nodes": 1})[0] == OK


def test_es_multi_node_yellow_is_warn_and_red_fails():
    assert classify_es_health({"status": "yellow", "number_of_nodes": 3})[0] == WARN
    assert classify_es_health({"status": "red", "number_of_nodes": 1})[0] == FAIL


def test_deepseek_unavailable_is_fail():
    assert classify_deepseek_balance({"is_available": False, "balance_infos": []})[0] == FAIL


def test_deepseek_low_balance_is_warn():
    payload = {"is_available": True, "balance_infos": [{"currency": "USD", "total_balance": "1.20"}]}
    assert classify_deepseek_balance(payload)[0] == WARN


def test_deepseek_healthy_balance_is_ok():
    payload = {"is_available": True, "balance_infos": [{"currency": "USD", "total_balance": "18.40"}]}
    assert classify_deepseek_balance(payload) == (OK, "18.40 USD")


def test_rerank_logits_fail():
    assert classify_rerank_scores([{"index": 0, "score": 4.2}, {"index": 1, "score": -3.1}])[0] == FAIL


def test_rerank_probabilities_ok():
    assert classify_rerank_scores([{"index": 0, "score": 0.97}, {"index": 1, "score": 0.01}])[0] == OK


def test_rerank_error_body_fails():
    assert classify_rerank_scores({"error": "model loading"})[0] == FAIL


def test_required_ollama_models_skip_cloud_models():
    task_map = {"answer": "deepseek-chat", "decide": "llama3.2:3b", "eval_judge": "gemini-2.5-pro"}
    assert required_ollama_models(task_map) == {"llama3.2:3b"}


def test_deepseek_non_usd_balance_is_not_compared_to_usd_threshold():
    # 10 CNY ~ 1.4 USD: must not be reported OK as "10.00 CNY".
    payload = {"is_available": True, "balance_infos": [{"currency": "CNY", "total_balance": "10"}]}
    status, detail = classify_deepseek_balance(payload)
    assert status == WARN and "no USD balance" in detail


def test_required_ollama_models_include_ollama_providers():
    models = required_ollama_models(
        {},
        provider_models=(("ollama", "qwen2.5-coder:14b-instruct"), ("deepseek", "deepseek-chat"), (None, None)),
    )
    assert models == {"qwen2.5-coder:14b-instruct"}


def test_task_map_ignored_when_routing_disabled():
    assert required_ollama_models({"decide": "llama3.2:3b"}, routing_enabled=False) == set()


def test_ollama_missing_model_fails():
    status, detail = classify_ollama_models({"qwen2.5:7b"}, {"llama3.2:3b"})
    assert status == FAIL
    assert "llama3.2:3b" in detail


def test_ollama_latest_tag_matches_bare_name():
    assert classify_ollama_models({"nomic-embed-text:latest"}, {"nomic-embed-text"})[0] == OK


def test_tei_cpu_fallback_is_warn():
    log = "Could not find a compatible CUDA device on host\nUsing CPU instead\nStarting Bert model on Cpu"
    assert classify_tei_device("agentrag-tei-rerank", log)[0] == WARN


def test_tei_on_gpu_is_ok():
    assert classify_tei_device("agentrag-tei", "Starting Bert model on Cuda(CudaDevice(DeviceId(1)))")[0] == OK
