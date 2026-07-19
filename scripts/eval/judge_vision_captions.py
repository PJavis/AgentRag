"""Judge sandbox image captions (qwen2.5vl:7b) against the actual images via Gemini.

Reads segment_type=image segments from the scratch ES index, then for each sampled
image sends the ACTUAL image bytes + the qwen caption to gemini-2.5-flash for a 1-5
medical-accuracy/usefulness score + a failure-mode label. Deliverable of vision Slice B
(caption-quality gate for the full-corpus re-ingest, Slice C).

Sampling: up to MAX_IMAGES, stratified round-robin across documents so scanned vs
text+embedded docs are both represented (not just whatever ES returns first).

PHI note: this is an assessment-only judge (not a prod path). It sends the bounded
sample of images to Gemini (cloud). Ingest captioning itself is 100% local (Ollama).

Run:
  source .env
  ELASTICSEARCH_INDEX_NAME=agentrag_segments_scratch GEMINI_API_KEY="$GEMINI_API_KEY" \
    PYTHONPATH=. uv run python scripts/eval/judge_vision_captions.py
"""
import asyncio, base64, json, os, sys
from collections import Counter, defaultdict

sys.path.insert(0, ".")
from elasticsearch import AsyncElasticsearch
from openai import AsyncOpenAI
from src.agentrag.config import settings

MAX_IMAGES = 40
FETCH_CAP = 1000  # pull all image segments (bounded) then stratify-sample locally
OUT_PATH = "/tmp/vis_scores.json"

JUDGE_SYS = (
    "You grade how accurately and usefully a CAPTION describes a MEDICAL image for retrieval. "
    "Look at the image, then judge the caption. "
    'Return ONLY JSON: {"score": 1-5 (1=wrong/hallucinated, 5=accurate+specific+useful), '
    '"failure_mode": one of [accurate, generic-uninformative, missed-key-finding, '
    'hallucinated-finding, ocr-of-text-slide, unreadable], "note": "<short reason>"}.'
)


def _b64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def _resolve_path(meta):
    """image_path is the local disk path written at ingest (/tmp/vis_images/...).
    Fall back to mapping a /images/<..> serving URL under IMAGE_STORAGE_DIR."""
    path = meta.get("image_path")
    if path and os.path.exists(path):
        return path
    url = meta.get("image_url") or ""
    if url.startswith("/images/"):
        cand = os.path.join(getattr(settings, "IMAGE_STORAGE_DIR", "data/images"),
                            url[len("/images/"):])
        if os.path.exists(cand):
            return cand
    return path  # may not exist → caller records image-file-missing


def _stratified(hits, k):
    """Round-robin across documents so each doc contributes before any doc repeats."""
    by_doc = defaultdict(list)
    for h in hits:
        by_doc[h["_source"].get("document_title") or "?"].append(h)
    docs = sorted(by_doc)  # deterministic order
    out = []
    i = 0
    while len(out) < k and any(by_doc[d] for d in docs):
        d = docs[i % len(docs)]
        if by_doc[d]:
            out.append(by_doc[d].pop(0))
        i += 1
    return out[:k]


async def main():
    es = AsyncElasticsearch(settings.ELASTICSEARCH_URL)
    assert settings.ELASTICSEARCH_INDEX_NAME == "agentrag_segments_scratch", \
        f"refuse to judge non-scratch index: {settings.ELASTICSEARCH_INDEX_NAME}"
    res = await es.search(
        index="agentrag_segments_scratch", size=FETCH_CAP,
        query={"term": {"segment_type": "image"}},
        _source=["content", "metadata", "document_title"],
    )
    await es.close()
    hits = res["hits"]["hits"]
    print(f"TOTAL image segments in scratch index: {len(hits)}")
    sample = _stratified(hits, MAX_IMAGES)
    print(f"SAMPLING {len(sample)} (stratified across {len({h['_source'].get('document_title') for h in hits})} docs)")

    gem = AsyncOpenAI(api_key=os.environ["GEMINI_API_KEY"],
                      base_url="https://generativelanguage.googleapis.com/v1beta/openai/")
    out = []
    for h in sample:
        s = h["_source"]
        meta = s.get("metadata") or s.get("extra_metadata") or {}
        caption = s.get("content") or ""
        doc = s.get("document_title")
        path = _resolve_path(meta)
        if not path or not os.path.exists(path):
            out.append({"doc": doc, "image_path": path, "caption": caption[:200],
                        "score": None, "failure_mode": "image-file-missing", "note": "no local image"})
            continue
        msgs = [{"role": "system", "content": JUDGE_SYS},
                {"role": "user", "content": [
                    {"type": "text", "text": f"CAPTION: {caption[:1500]}"},
                    {"type": "image_url",
                     "image_url": {"url": f"data:image/jpeg;base64,{_b64(path)}"}}]}]
        j = {"score": None, "failure_mode": "judge-error", "note": "unset"}
        # gemini-2.5-flash is a thinking model — a small max_tokens gets consumed by
        # reasoning and returns empty content. Give a generous budget + retry once.
        for attempt in range(2):
            try:
                r = await gem.chat.completions.create(
                    model="gemini-2.5-flash", temperature=0.0, max_tokens=2048,
                    response_format={"type": "json_object"}, messages=msgs)
                content = (r.choices[0].message.content or "").strip()
                if not content:
                    fr = getattr(r.choices[0], "finish_reason", "?")
                    j = {"score": None, "failure_mode": "judge-error",
                         "note": f"empty content (finish_reason={fr}, attempt={attempt})"}
                    continue
                j = json.loads(content)
                break
            except Exception as e:  # noqa: BLE001
                j = {"score": None, "failure_mode": "judge-error", "note": str(e)[:150]}
        out.append({"doc": doc, "image_path": path, "caption": caption[:200], **j})
        print(f"  [{len(out)}/{len(sample)}] {str(doc)[:24]:24} score={j.get('score')} {j.get('failure_mode')}")

    json.dump(out, open(OUT_PATH, "w"), ensure_ascii=False, indent=2)
    scored = [o for o in out if isinstance(o.get("score"), (int, float))]
    fm = Counter(o.get("failure_mode") for o in out)
    mean = sum(o["score"] for o in scored) / len(scored) if scored else 0
    halluc = sum(1 for o in out if o.get("failure_mode") == "hallucinated-finding") / (len(out) or 1)

    # Per-doc mean split (scanned vs text+embedded reported in the results doc).
    per_doc = defaultdict(list)
    for o in scored:
        per_doc[o["doc"]].append(o["score"])

    print("\n==== SLICE B CAPTION-QUALITY RESULT ====")
    print(f"JUDGED={len(out)} SCORED={len(scored)} MEAN={mean:.2f} HALLUC_RATE={halluc:.2f}")
    print("FAILURE_MODES:", dict(fm))
    print("PER_DOC_MEAN:", {d: round(sum(v) / len(v), 2) for d, v in sorted(per_doc.items())})
    verdict = "GO" if (mean >= 3.5 and halluc < 0.15) else "NO-GO"
    print(f"PRE-REGISTERED VERDICT (GO iff MEAN>=3.5 AND HALLUC<0.15): {verdict}")
    print(f"scores written -> {OUT_PATH}")


asyncio.run(main())
