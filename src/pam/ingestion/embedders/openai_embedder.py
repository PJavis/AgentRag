# src/pam/ingestion/embedders/openai_embedder.py
from openai import OpenAI
from cachetools import LRUCache
from typing import List

class OpenAIEmbedder:
    def __init__(self, model: str = "text-embedding-3-large"):
        self.client = OpenAI()
        self.model = model
        self.cache = LRUCache(maxsize=10000)  # 10K entries

    def embed(self, texts: List[str]) -> List[List[float]]:
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
            response = self.client.embeddings.create(
                model=self.model,
                input=to_embed
            )
            new_embeds = [item.embedding for item in response.data]

            for i, embed in zip(indices, new_embeds):
                self.cache[texts[i]] = embed
                embeddings.insert(i, embed)

        return embeddings
