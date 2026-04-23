import json
from dataclasses import dataclass

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


@dataclass
class SearchResult:
    text: str
    score: float  # cosine similarity, higher is better
    index: int


class FAISSStore:
    def __init__(self, dim: int):
        # IndexFlatIP = dot product on normalized vecs = cosine similarity
        # Exact search - no approximation, no training needed
        self.index = faiss.IndexFlatIP(dim)
        self.chunks: list[str] = []

    def add(self, texts: list[str], embeddings: np.ndarray) -> None:
        # FAISS requires float32 - normalize first for cosine similarity
        vecs = embeddings.astype("float32")
        faiss.normalize_L2(vecs)  # in-place, required for IndexFlatIP = cosine

        self.index.add(vecs)
        self.chunks.extend(texts)

    def search(self, query_vec: np.ndarray, k: int = 3) -> list[SearchResult]:
        q = query_vec.astype("float32").reshape(1, -1)
        faiss.normalize_L2(q)

        scores, indices = self.index.search(q, k)  # returns (1, k) arrays

        return [
            SearchResult(text=self.chunks[i], score=float(s), index=int(i))
            for s, i in zip(scores[0], indices[0])
            if i != -1  # FAISS returns -1 for unfilled slots
        ]

    def save(self, index_path: str, meta_path: str) -> None:
        faiss.write_index(self.index, index_path)
        with open(meta_path, "w") as f:
            json.dump(self.chunks, f)

    @classmethod
    def load(cls, index_path: str, meta_path: str) -> "FAISSStore":
        idx = faiss.read_index(index_path)
        with open(meta_path) as f:
            chunks = json.load(f)

        store = cls.__new__(cls)
        store.index = idx
        store.chunks = chunks
        return store


# Saving vectors
model = SentenceTransformer("all-MiniLM-L6-v2")
chunks = [
    "Refund policy is 30 days from purchase.",
    "Office hours are 9am to 5pm, Monday to Friday.",
    "Shipping takes 3 to 5 business days.",
    "To cancel your subscription, go to account settings.",
    "We support Visa, Mastercard, and PayPal.",
]
embeddings = model.encode(chunks, normalize_embeddings=True)  # (5, 384)
store = FAISSStore(dim=embeddings.shape[1])
store.add(chunks, embeddings)
store.save("store.faiss", "store.json")
print(f"Indexed {store.index.ntotal} vectors")

# Searching vectors
store = FAISSStore.load("store.faiss", "store.json")
query = "How long do I have to return something?"
q_vec = model.encode(query, normalize_embeddings=True)
results = store.search(q_vec, k=3)

for r in results:
    print(f"Score: {r.score:.3f} | {r.text}")
