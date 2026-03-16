# src/pam/ingestion/embedders/hf_inference_embedder.py
from huggingface_hub import InferenceClient
from cachetools import LRUCache
from typing import List
from src.pam.config import settings
import asyncio

class HFInferenceEmbedder:
    def __init__(self, model: str = "intfloat/multilingual-e5-large-instruct"):
        token = settings.HF_TOKEN
        if not token:
            raise ValueError("HF_TOKEN not found in .env")

        self.client = InferenceClient(model=model, token=token)
        self.model = model
        self.cache = LRUCache(maxsize=10000)

    async def embed(self, texts: List[str]) -> List[List[float]]:
        embeddings = []
        to_embed = []
        indices = []

        for i, text in enumerate(texts):
            if text in self.cache:
                embeddings.append(self.cache[text])
            else:
                to_embed.append(text)
                indices.append(i)

        if to_embed:
            batch_size = 32
            for start in range(0, len(to_embed), batch_size):
                batch = to_embed[start:start + batch_size]
                try:
                    # Sửa tham số: dùng 'text' thay vì 'text_inputs'
                    response = await asyncio.to_thread(
                        self.client.feature_extraction,
                        text=batch,  # <-- sửa thành text= (hỗ trợ list string)
                        normalize=True,
                        truncate=True,
                    )
                    # response là list of list[float] (embedding vectors)
                    new_embeds = response

                    for j, embed in enumerate(new_embeds):
                        orig_idx = start + j
                        self.cache[texts[indices[orig_idx]]] = embed
                        embeddings.insert(indices[orig_idx], embed)

                except Exception as e:
                    raise RuntimeError(f"HF Inference error: {str(e)}")

        return embeddings
