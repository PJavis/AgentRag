---
license: mit
base_model: intfloat/multilingual-e5-base
language:
  - vi
library_name: sentence-transformers
tags:
  - sentence-transformers
  - sentence-similarity
  - feature-extraction
  - text-embeddings-inference
  - medical
pipeline_tag: sentence-similarity
---

# agentrag-embed-v1

Vietnamese-medical fine-tune of [intfloat/multilingual-e5-base](https://huggingface.co/intfloat/multilingual-e5-base) for RAG retrieval in the AgentRag project.

- **Pooling:** mean (+ L2 normalize)
- **Dimensions:** 768
- **Max sequence length:** 512
- **Domain:** Vietnamese medical documents
- **Training:** 5.3k (query, positive, negative) triplets, MultipleNegativesRankingLoss, 2 epochs
- **Eval:** recall@10 +0.20 over base e5 on the project's retrieval benchmark (C1)

## Serving with TEI

```yaml
services:
  tei:
    image: ghcr.io/huggingface/text-embeddings-inference:cuda-latest
    command:
      - --model-id=dung6903/agentrag-embed-v1
      - --pooling=mean
```

Query prefix conventions follow e5: `query: ...` / `passage: ...`.
