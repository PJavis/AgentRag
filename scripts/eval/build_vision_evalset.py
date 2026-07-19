"""Slice C Phase 3 — synthesize an image-DEPENDENT eval-Q set.

For a stratified sample of augmented image segments:
  1. gemini (vision) drafts a Q + gold answer that requires READING THE IMAGE.
  2. dependency filter: give the project's TEXT model the image's neighbouring
     text (same document, nearby positions) WITHOUT the image + the question; if it
     answers correctly (score_correctness.mean >= 0.6) the Q is text-answerable ->
     DISCARD. Keep only Qs the text model fails -> those measure vision's lift.

Model diversity (anti-circularity): gemini generates, deepseek answers the filter,
score_correctness (task=eval_judge) judges.

Post-review fixes (v2, first review found the filter too weak / fail-open):
  - `_neighbour_text` now actually uses the image's `position`: it fetches a
    larger candidate pool of the doc's non-image segments and sorts by
    |seg.position - image.position| in Python, keeping the 6 CLOSEST — not an
    arbitrary ES-ordered 6 from anywhere in the document. v1 ignored
    `position` entirely, so the "surrounding text" was sometimes unrelated
    text far from the image, making it artificially easy for the text model
    to "fail" and the filter to wrongly keep the question.
  - The filter's text-answerer call now DISCARDS the candidate (after one
    retry) if the deepseek call raises, instead of silently treating the
    exception as `txt_answer=""` (which scores ~0 and therefore PASSES the
    keep bar). Exception -> discard, never exception -> keep.
  - Light de-dup: a candidate is skipped if its normalized gold answer has
    already appeared twice among kept rows (caps repeated facts like
    "Scoliosis" x4).
"""
import asyncio, base64, json, os, sys
from collections import defaultdict
sys.path.insert(0, ".")
from elasticsearch import AsyncElasticsearch
from openai import AsyncOpenAI
from src.agentrag.config import settings
from src.agentrag.services.llm_gateway import LLMGateway
from src.agentrag.eval.correctness_judge import score_correctness

N_IMAGES = 150         # candidates to draft from (kept set will be smaller after filter;
                       # raised from 60 since the fixed, stricter filter discards more)
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
    # Fetch a larger candidate pool of the doc's non-image segments, then pick the 6
    # actually CLOSEST to the image's own position (not an arbitrary ES-ordered 6 from
    # anywhere in the doc — that made the filter too weak, testing against unrelated text).
    res = await es.search(index=settings.ELASTICSEARCH_INDEX_NAME, size=50,
        query={"bool": {"must": [{"term": {"document_title.keyword": title}}],
                        "must_not": [{"term": {"segment_type": "image"}}]}},
        _source=["content", "position"])
    target = position if position is not None else 0
    cands = res["hits"]["hits"]
    cands.sort(key=lambda h: abs((h["_source"].get("position") or 0) - target))
    nearest = cands[:6]
    return "\n".join(h["_source"].get("content", "") for h in nearest)[:4000]


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
    gold_counts: dict[str, int] = defaultdict(int)  # de-dup: cap repeats of same gold answer
    n_drafted = n_filter_failed = n_dedup_skipped = n_text_answerable = 0
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
        n_drafted += 1
        # 2. dependency filter — text model, no image
        ctx = await _neighbour_text(es, title, s.get("position"))
        txt_answer = None
        for attempt in range(2):  # one retry before giving up
            try:
                # json_response returns (payload_dict, latency_ms)
                payload, _ = await gateway.json_response(
                    system_prompt="Answer the question from the provided context only. Return JSON {\"answer\":\"...\"}.",
                    user_prompt=f"CONTEXT:\n{ctx}\n\nQUESTION: {q}", task="eval_judge2")
                txt_answer = (payload or {}).get("answer", "") if isinstance(payload, dict) else str(payload)
                break
            except Exception:
                continue
        if txt_answer is None:
            # The filter call itself failed (both attempts) — we cannot determine
            # text-answerability, so DISCARD rather than defaulting to "keep" (an
            # exception must never silently masquerade as "text model failed to
            # answer" — that would inflate the keep rate with unverified questions).
            print(f"  DISCARD (filter-call failed x2) {title[:30]}")
            n_filter_failed += 1
            continue
        norm_gold = gold.strip().lower()
        if gold_counts[norm_gold] >= 2:
            n_dedup_skipped += 1
            continue  # de-dup: cap repeats of the same gold answer at 2
        e = await score_correctness(q, txt_answer, gold, ctx, gateway, task="eval_judge")
        if e.mean >= TEXT_ANSWERABLE_THRESHOLD:
            n_text_answerable += 1
            continue  # text-answerable → not image-dependent → discard
        gold_counts[norm_gold] += 1
        kept.append({"id": f"vis-{len(kept)}", "question": q, "reference_answer": gold,
                     "gold_contexts": [s.get("content", "")], "lang": "vi",
                     "source": title, "image_path": path})
        print(f"  kept {len(kept)}/{KEEP_TARGET} (text_score={e.mean:.2f}) {title[:30]}")
    await es.close()
    with open(OUT, "w") as f:
        for row in kept:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"WROTE {len(kept)} image-dependent Qs -> {OUT}")
    print(f"SUMMARY drafted={n_drafted} kept={len(kept)} "
          f"discarded_text_answerable={n_text_answerable} "
          f"discarded_filter_call_failed={n_filter_failed} "
          f"discarded_dedup={n_dedup_skipped}")


asyncio.run(main())
