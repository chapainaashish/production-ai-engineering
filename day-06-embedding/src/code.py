import random
import time
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.manifold import TSNE
from sklearn.preprocessing import normalize

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
        Embed LangChain Document chunks
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
        (
            "I want a refund",
            "What are your business hours?",
        ),
        (
            "I want a refund",
            "Quantum entanglement explained",
        ),
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


def visualize_embeddings(
    vectors: list[list[float]],
    labels: list[str],
    title: str = "Embedding Clusters (t-SNE)",
    max_points: int = 500,
):
    """
    t-SNE plot of chunk embeddings reduced to 2D.

    Not production code
    Use this to sanity-check chunking before building the FAISS index
    If chunks that should be semantically close aren't clustering together,
    chunking or cleaning has a problem. then, need to fix it here, not in retrieval.
    """

    if len(vectors) > max_points:
        idx = random.sample(range(len(vectors)), max_points)
        vectors = [vectors[i] for i in idx]
        labels = [labels[i] for i in idx]
        print(f"[t-SNE] Sampled {max_points} points for visualization")

    print(f"[t-SNE] Reducing {len(vectors)} vectors ({len(vectors[0])} dims) → 2D ...")
    arr = normalize(np.array(vectors))  # L2 normalize before t-SNE
    reduced = TSNE(
        n_components=2,
        perplexity=min(30, len(vectors) - 1),
        random_state=42,
        n_iter=1000,
    ).fit_transform(arr)

    plt.figure(figsize=(12, 8))
    plt.scatter(reduced[:, 0], reduced[:, 1], alpha=0.6, s=15)

    # Annotate every ~5% of points so clusters are readable without overlap
    for i in range(0, len(reduced), max(1, len(reduced) // 20)):
        plt.annotate(
            labels[i][:30], (reduced[i, 0], reduced[i, 1]), fontsize=6, alpha=0.7
        )

    plt.title(title)
    plt.tight_layout()
    plt.savefig("embedding_clusters.png", dpi=150)
    plt.show()
    print("[t-SNE] Saved: embedding_clusters.png")


if __name__ == "__main__":
    print("=== SIMILARITY DEMO ===")
    demo_similarity()

    # 2. To visualize chunks
    # from day08_document_loading import build_chunks
    # chunks = build_chunks(["docs/sample.pdf"])
    # client = EmbeddingClient()
    # vectors, metadata = client.embed_chunks(chunks)
    # labels = [m.get("file", "?") for m in metadata]
    # visualize_embeddings(vectors, labels)
