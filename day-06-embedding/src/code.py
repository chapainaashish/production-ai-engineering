import time
from dataclasses import dataclass

import numpy as np
from sentence_transformers import SentenceTransformer

DEFAULT_MODEL = "all-MiniLM-L6-v2"


@dataclass
class EmbeddingResult:
    embeddings: list[list[float]]
    model: str
    total_texts: int
    latency_ms: float
    dims: int


class EmbeddingClient:
    def __init__(self, model_name: str = DEFAULT_MODEL):
        print(f"[EmbeddingClient] Loading model: {model_name} ...")
        self.model = SentenceTransformer(model_name)
        self.model_name = model_name
        self.dims = self.model.get_embedding_dimension()

    def embed_batch(self, texts: list[str], batch_size: int = 64) -> EmbeddingResult:
        """
        Embed texts in batches. sentence-transformers handles batching internally
        but explicit batch_size controls memory usage on large corpora.
        """
        start = time.time()

        # normalize_embeddings=True applies L2 normalization so cosine similarity
        # becomes a simple dot product — faster at search time (Day 11 uses this)
        vectors = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=len(texts) > 100,  # progress bar only for large batches
            normalize_embeddings=True,
        ).tolist()

        latency = (time.time() - start) * 1000

        result = EmbeddingResult(
            embeddings=vectors,
            model=self.model_name,
            total_texts=len(texts),
            latency_ms=latency,
            dims=self.dims,
        )

        print(f"\n[embed_batch] {len(texts)} texts → {self.dims}-dim vectors")
        print(f"  Model:   {self.model_name}")
        print(f"  Latency: {latency:.0f}ms")

        return result

    def embed_chunks(self, chunks) -> tuple[list[list[float]], list[dict]]:
        """
        Embed LangChain Document chunks.
        Returns (vectors, metadata_list) - keep them paired by index.
        """
        texts = [c.page_content for c in chunks]
        metadata = [c.metadata for c in chunks]
        result = self.embed_batch(texts)
        return result.embeddings, metadata


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """
    Cosine similarity: measures angle between vectors, ignoring magnitude.
    Range: -1 (opposite) to 1 (identical).

    Why not euclidean?
    Longer text → larger magnitude embedding. Two semantically identical chunks
    could appear "far apart" just because one has more words. Cosine eliminates
    this - only the direction matters, not the length.

    Note: since embed_batch uses normalize_embeddings=True, vectors are already
    unit-length. Cosine similarity on unit vectors = dot product. Same result,
    slightly faster. Both approaches shown below.
    """
    a, b = np.array(a), np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def demo_similarity():
    """
    Demonstrates why cosine works for semantic similarity.
    """
    client = EmbeddingClient()  # uses all-MiniLM-L6-v2 by default

    sentence_pairs = [
        ("I want a refund", "I'd like my money back"),
        ("I want a refund", "How do I return a product?"),
        ("I want a refund", "What are your business hours?"),
        ("I want a refund", "Quantum entanglement explained"),
    ]

    # Embed all unique sentences in one batch - never embed one-by-one
    all_texts = list({s for pair in sentence_pairs for s in pair})
    result = client.embed_batch(all_texts)
    vec_map = {t: v for t, v in zip(all_texts, result.embeddings)}

    print("COSINE SIMILARITY DEMO")
    print(f"{'Pair':<60} {'Score':>6}")
    print("-" * 70)

    for a, b in sentence_pairs:
        score = cosine_similarity(vec_map[a], vec_map[b])
        label = "high" if score > 0.8 else "mid" if score > 0.5 else "low"
        print(f"{a!r:>25} ↔ {b!r:<30} {score:.4f}  {label}")


if __name__ == "__main__":
    demo_similarity()
