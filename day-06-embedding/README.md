# Embeddings in RAG

Retrieval-Augmented Generation (RAG) systems rely on the core idea of converting text into vectors so we can search by meaning instead of keywords. These vectors are called embeddings, and they are the backbone of semantic search. But turning raw text into embeddings is only the beginning. A production RAG system depends on how you chunk data, choose models, store vectors, and perform retrieval efficiently.


## One Chunk → One Vector

Before anything reaches a vector database, text must be split into chunks. Then, each chunk goes into the embedding model and comes out as a single vector(a list of floating point numbers)

```
Chunk 1: "Refund policy is 30 days."
→ [0.21, 0.82, -0.33, ...]   (1536 dims)

Chunk 2: "Office hours are 9am to 5pm."
→ [0.52, -0.11, 0.91, ...]   (1536 dims)
```

A vector represents the entire meaning of a chunk. If you mix unrelated concepts:

> "Refund policy is 30 days. Office hours are 9am–5pm."

The embedding must compress both meanings into a single representation. That means if a chunk mixes two unrelated topics, like refund policy and office hours in the same chunk, that single vector has to average both meanings. This reduces retrieval accuracy because the vector no longer strongly represents either concept. This is why proper chunking is crucial in a RAG system. To achieve that, you should follow:

- One topic per chunk
- Keep semantic boundaries clean (headers, sections, paragraphs)
- Use overlap (important, covered later)


## Inside the Embedding Model

Embedding models are transformer-based architectures that output one vector per token.

Example: "Refund policy" → 2 tokens → 2 vectors:

```
"Refund"  →  [0.2,  0.8, -0.3]
"policy"  →  [0.5, -0.1,  0.9]
```

But now you have 2 vectors. The vector DB needs one. Collapsing multiple token vectors into a single vector is called **pooling**. Three ways to do it:

**Mean Pooling (Most Common in Modern Models)** - Average all token vectors dimension by dimension. Most modern models use this as it has better traits for semantic similarity.

```
v_sentence = (v1 + v2 + ... + vn) / n
```

**CLS Pooling** - BERT adds a special `[CLS]` token at the start of every input. After the transformer processes everything, take that token's vector as the sentence representation. It is designed for classification tasks and not ideal for similarity.

**Max Pooling** - Take the maximum value at each dimension and preserve the strongest signal from any token. It is less common for text similarity.

Most of the time, you don't choose the pooling strategy, it's already implemented into the embedding model you are going to use. However, it's necessary to know why `all-MiniLM-L6-v2` and `text-embedding-3-small` behave differently on the same text even at similar dimension counts. Each model might have different pooling, different training, and different vectors.


## Choosing the Embedding Model

When selecting the embedding model for your project, you can use either API-based or local models.

**API-based (OpenAI or other models):**

| Model | Dimensions | When to use |
|---|---|---|---|
| `text-embedding-3-small` | 1536 | Development, most production use cases |
| `text-embedding-3-large` | 3072 | When retrieval accuracy is critical |

**Local (free, no API):**

| Model | Dimensions | When to use |
|---|---|---|---|
| `all-MiniLM-L6-v2` | 384 | Prototyping, privacy-sensitive data |

Before choosing a model, go through the Massive Text Embedding Benchmark (MTEB). MTEB is a standardized leaderboard that tests embedding models across 8 tasks and 181 datasets. For RAG, look at the **Retrieval tab** and the **NDCG@10 score**.

- **NDCG@10** measures the quality of the top 10 retrieved results. It scores from 0 to 1 and cares about position — a document at rank 1 scores higher than one at rank 8. Because in real usage, if the answer is buried at position 8, your RAG might as well have missed it.

- When analyzing this data, don't fall into the trap of always selecting the top-ranked model. MTEB averages scores across all 181 datasets — climate science, biomedical, debate arguments, Wikipedia, everything. Your RAG is probably for one specific domain. Choose wisely according to your need.

> **Note:** MTEB ranks **embedding models only**, not GPT-4, not Claude. Those are response generation models and never appear here.


## Vector Similarity in Retrieval

Once embeddings are stored, we need to compute similarity between the query vector and document vectors. There are three ways to measure closeness between two vectors:

**Euclidean distance** - Straight-line distance. Problem: a long document produces a larger-magnitude embedding than a short chunk. Two semantically identical chunks could appear "far apart" just because one has more words.

**Dot product** - Sensitive to the magnitude problem. Works best when vectors are normalized.

**Cosine similarity** - Measures the angle between vectors, ignoring magnitude. Two vectors pointing the same direction score 1.0 regardless of length. This is what you want for text.

```python
import numpy as np

def cosine_similarity(a: list[float], b: list[float]) -> float:
    a, b = np.array(a), np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

# Scores: -1 (opposite) to 1 (identical)
# In practice, text embeddings rarely go below 0
```

Intuition for scores: above 0.9 is near-identical meaning. 0.7–0.9 is strongly related. Below 0.5 is weak or unrelated.

You do not compare the query vector against every stored vector, that would be O(N) and too slow. Instead, use approximate nearest neighbor (ANN) search, which:

- builds an index
- searches only a subset of candidates
- trades tiny accuracy loss for massive speed gain

Common libraries that facilitate ANN:

- FAISS (Meta)
- HNSW (Hierarchical Navigable Small World graphs)
- ScaNN (Google)

These embeddings are stored in vector databases that provide indexing, similarity searching, and metadata filtering, so you don't have to worry about all of that yourself.

## What Not to Do

- **Don't embed entire documents.** Embed chunks. A 50-page PDF as one embedding loses all granularity.
- **Don't mix embedding models.** Index with `text-embedding-3-small`, query with `text-embedding-3-small`. Different models produce incompatible vector spaces - mixing them silently breaks retrieval.
- **Don't embed during the request lifecycle.** Embed at index time. At query time, embed only the query string (~10ms), not your knowledge base.
- **Use overlap to preserve context while chunking:**

```
Chunk 1: sentences 1–5
Chunk 2: sentences 4–8
```

- **Normalize your vectors.** Many embedding models produce normalized vectors internally. If not normalized, cosine similarity behaves inconsistently and dot product becomes biased.
