"""Slice C Phase 3 — synthesize an image-DEPENDENT eval-Q set.

For a stratified sample of augmented image segments:
  1. gemini (vision) drafts a Q + gold answer that requires READING THE IMAGE.
  2. dependency filter: give the project's TEXT model the image's neighbouring
     text (same document, nearby positions) WITHOUT the image + the question; if it
     answers correctly (score_correctness.mean >= 0.6) the Q is text-answerable ->
     DISCARD. Keep only Qs the text model fails -> those measure vision's lift.

Model diversity (anti-circularity): gemini generates, deepseek answers the filter,
score_correctness (task=eval_judge) judges.
"""
import asyncio, base64, json, os, sys
from collections import defaultdict
sys.path.insert(0, ".")
from elasticsearch import AsyncElasticsearch
from openai import AsyncOpenAI
from src.agentrag.config import settings
from src.agentrag.services.llm_gateway import LLMGateway
from src.agentrag.eval.correctness_judge import score_correctness

N_IMAGES = 60          # candidates to draft from (kept set will be smaller after filter)
KEEP_TARGET = 40
TEXT_ANSWERABLE_THRESHOLD = 0.6
OUT = "data/eval/vision_evalset_2026-07-19.jsonl"

GEN_SYS = (
  "You write ONE exam question that can ONLY be answered by LOOKING AT this medical image "
  "(a finding, count, structure, modality, or label visible in the image), NOT from general "
  "knowledge or surrounding prose. Return JSON: {\"question\": \"...\", \"answer\": \"<concise gold>\"}."
)


def _b64(p):
    with open(p, "rb") as f:
        return base64.b64encode(f.read()).decode()


async def _neighbour_text(es, title, position):
    # nearby text segments from the same document = the 'surrounding text' a reader has.
    res = await es.search(index=settings.ELASTICSEARCH_INDEX_NAME, size=6,
        query={"bool": {"must": [{"term": {"document_title.keyword": title}}],
                        "must_not": [{"term": {"segment_type": "image"}}]}},
        _source=["content"])
    return "\n".join(h["_source"].get("content", "") for h in res["hits"]["hits"])[:4000]


async def main():
    es = AsyncElasticsearch(settings.ELASTICSEARCH_URL)
    res = await es.search(index=settings.ELASTICSEARCH_INDEX_NAME, size=500,
        query={"term": {"segment_type": "image"}},
        _source=["content", "metadata", "document_title", "position"])
    hits = res["hits"]["hits"]
    # stratify across docs
    by_doc = defaultdict(list)
    for h in hits:
        by_doc[h["_source"].get("document_title")].append(h)
    docs = sorted(by_doc); sample = []
    i = 0
    while len(sample) < N_IMAGES and any(by_doc[d] for d in docs):
        d = docs[i % len(docs)]
        if by_doc[d]:
            sample.append(by_doc[d].pop(0))
        i += 1

    gem = AsyncOpenAI(api_key=os.environ["GEMINI_API_KEY"],
                      base_url="https://generativelanguage.googleapis.com/v1beta/openai/")
    gateway = LLMGateway()
    kept = []
    for h in sample:
        if len(kept) >= KEEP_TARGET:
            break
        s = h["_source"]; meta = s.get("metadata") or {}
        path = meta.get("image_path"); title = s.get("document_title")
        if not path or not os.path.exists(path):
            continue
        # 1. draft Q + gold from the image
        try:
            r = await gem.chat.completions.create(
                model="gemini-2.5-flash", temperature=0.2, max_tokens=2048,
                response_format={"type": "json_object"},
                messages=[{"role": "system", "content": GEN_SYS},
                          {"role": "user", "content": [
                              {"type": "text", "text": "Draft the question now."},
                              {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{_b64(path)}"}}]}])
            j = json.loads(r.choices[0].message.content or "{}")
            q, gold = j.get("question", "").strip(), j.get("answer", "").strip()
        except Exception:
            continue
        if not q or not gold:
            continue
        # 2. dependency filter — text model, no image
        ctx = await _neighbour_text(es, title, s.get("position"))
        try:
            # json_response returns (payload_dict, latency_ms)
            payload, _ = await gateway.json_response(
                system_prompt="Answer the question from the provided context only. Return JSON {\"answer\":\"...\"}.",
                user_prompt=f"CONTEXT:\n{ctx}\n\nQUESTION: {q}", task="eval_judge2")
            txt_answer = (payload or {}).get("answer", "") if isinstance(payload, dict) else str(payload)
        except Exception:
            txt_answer = ""
        e = await score_correctness(q, txt_answer, gold, ctx, gateway, task="eval_judge")
        if e.mean >= TEXT_ANSWERABLE_THRESHOLD:
            continue  # text-answerable → not image-dependent → discard
        kept.append({"id": f"vis-{len(kept)}", "question": q, "reference_answer": gold,
                     "gold_contexts": [s.get("content", "")], "lang": "vi",
                     "source": title, "image_path": path})
        print(f"  kept {len(kept)}/{KEEP_TARGET} (text_score={e.mean:.2f}) {title[:30]}")
    await es.close()
    with open(OUT, "w") as f:
        for row in kept:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"WROTE {len(kept)} image-dependent Qs -> {OUT}")


asyncio.run(main())
