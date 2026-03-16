# src/pam/ingestion/chunkers/hybrid_chunker.py
from typing import List, Dict
import tiktoken  # dùng để đếm token chính xác (OpenAI style)

tokenizer = tiktoken.encoding_for_model("text-embedding-3-large")

class HybridChunker:
    def __init__(self, max_tokens: int = 512):
        self.max_tokens = max_tokens

    def chunk(self, content: str, metadata: Dict = None) -> List[Dict]:
        chunks = []
        tokens = tokenizer.encode(content)
        position = 0
        chunk_id = 0

        while position < len(tokens):
            end = min(position + self.max_tokens, len(tokens))
            chunk_tokens = tokens[position:end]
            chunk_text = tokenizer.decode(chunk_tokens)

            # Ước lượng section_path đơn giản (sau này dùng Docling structure)
            section_path = f"chunk_{chunk_id}"

            chunks.append({
                "content": chunk_text,
                "content_hash": hashlib.sha256(chunk_text.encode()).hexdigest(),
                "segment_type": "text",
                "section_path": section_path,
                "position": chunk_id,
                "metadata": metadata or {},
            })
            position = end
            chunk_id += 1

        return chunks
