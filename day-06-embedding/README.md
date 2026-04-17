# Embeddings in RAG

You've chunked your documents. Now what? You can't store text in a vector database. You can't compute "similarity" between two sentences. You need something call "embeddings".

## One Chunk, One Vector

This is the foundation everything else builds on. When, you have chunks. Each chunk goes into the embedding model and comes out as a single vector(a list of floating point numbers). 

```
Chunk 1: "Refund policy is 30 days."     →  [0.2, 0.8, -0.3, ...]  (1536 numbers)
Chunk 2: "Office hours are 9am to 5pm."  →  [0.5, -0.1, 0.9, ...]  (1536 numbers)
```

Your vector DB stores all three. At query time, you embed the user's question and compare it against every stored vector. The closest one wins This is why chunk quality matters so much. Each chunk gets exactly **one** vector to represent everything inside it. If a chunk mixes two unrelated topics like, refund policy and office hours in the same chunk, that single vector has to average both meanings. It ends up representing neither well. 


## Inside the Embedding Model

A embedding transformer doesn't output one vector. It outputs one vector per token.

"Refund policy" → 2 tokens → 2 vectors:

```
"Refund"  →  [0.2,  0.8, -0.3]
"policy"  →  [0.5, -0.1,  0.9]
```

Now you have 2 vectors. The vector DB needs one. Collapsing multiple token vectors into a single vector is called **pooling**. Three ways to do it:

**CLS pooling** - BERT adds a special `[CLS]` token at the start of every input. After the transformer processes everything, take that token's vector as the sentence representation. It is designed for classification tasks and not ideal for similarity.

**Mean pooling** - It average all token vectors dimension by dimension. Most modern models use this as it has better traits for semantic similarity.

```
"Refund"  →  [0.2,  0.8, -0.3]
"policy"  →  [0.5, -0.1,  0.9]
average   →  [0.35, 0.35, 0.3]  ← sentence embedding
```

**Max pooling** - It take the maximum value at each dimension and preserves the strongest signal from any token. It is less common for text similarity. 

Most of the time, you don't choose the pooling strategy but it's already implemented into the embedding model you are going to use. However, it's necessary to know why `all-MiniLM-L6-v2` and `text-embedding-3-small` behave differently on the same text even at similar dimension counts. So, each model might have different pooling, different training and different vectors.


## Choosing the Embedding Model

When selecting the embeeding model in your project, You can use either API-based or local model.

**API-based (OpenAI or other models)**

| Model | Dimensions | Cost per 1M tokens | When to use |
|---|---|---|---|
| `text-embedding-3-small` | 1536 | $0.02 | Development, most production use cases |
| `text-embedding-3-large` | 3072 | $0.13 | When retrieval accuracy is critical |

**Local (free, no API):**

| Model | Dimensions | Cost | When to use |
|---|---|---|---|
| `all-MiniLM-L6-v2` | 384 | Free | Prototyping, privacy-sensitive data |


Before choosing the model, you should go through Massive Text Embedding Benchmark (MTEB). MTEB is a standardized leaderboard that tests embedding models across 8 tasks and 181 datasets. For RAG, you should look at the **Retrieval tab** and the **NDCG@10 score**.

* NDCG@10 score measure the quality of the top 10 retrieved results of the model. It scores from 0 to 1 and cares about position, which means document at rank 1 scores higher than at rank 8. Because in real usage, if the answer is buried at position 8, your RAG might as well have missed it. 

* When analyzing this data, don't fall into the trap by always selecting the ranked model.MTEB averages scores across all 181 datasets, climate science, biomedical, debate arguments, Wikipedia, everything. Your RAG is probably for one specific domain. So, choose wisely according to your need. 


* NOTE:  MTEB ranks **embedding models only**, not GPT-4, not Claude. Those are response generation models and never appear here.

```
Embedding model  →  converts text to vectors  (MTEB ranks these)
LLM              →  generates the answer      (different benchmarks)
```


## Finding Similarity of Vecotrs

There are three ways to measure closeness between two vectors:

**Euclidean distance** - straight-line distance. Problem: a long document produces a larger-magnitude embedding than a short chunk. Two semantically identical chunks could appear "far apart" just because one has more words.

**Dot product** - same magnitude problem. Biased toward longer text.

**Cosine similarity** - measures the angle between vectors, ignoring magnitude. Two vectors pointing the same direction score 1.0 regardless of length. This is what you want for text.

```python
import numpy as np

def cosine_similarity(a: list[float], b: list[float]) -> float:
    a, b = np.array(a), np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

# Scores: -1 (opposite) to 1 (identical)
# In practice, text embeddings rarely go below 0
```

Intuition for scores: above 0.9 is near-identical meaning. 0.7–0.9 is strongly related. Below 0.5 is weak or unrelated.


## What the t-SNE Visualization Shows You

Reduce 1536 dimensions down to 2 with t-SNE and plot your chunks. Similar topics cluster together spatially.

Two uses in practice:

**Sanity check your chunking** - if chunks that should be semantically close aren't clustering together, your chunking or cleaning broke something.

**Understand your data** - you can literally see which topics are "close" in your knowledge base and spot gaps in coverage.


## What Not to Do

- **Don't embed entire documents.** Embed chunks. A 50-page PDF as one embedding loses all granularity.
- **Don't mix embedding models.** Index with `text-embedding-3-small`, query with `text-embedding-3-small`. Different models produce incompatible vector spaces — mixing them silently breaks retrieval.
- **Don't skip cost tracking.** 10,000 chunks × 500 tokens avg = 5M tokens = $0.10 on `3-small`. Cheap until you reindex daily.
- **Don't embed during the request lifecycle.** Embed at index time. At query time, embed only the query string (~10ms), not your knowledge base.


## Key Takeaway

The surprising thing isn't that embeddings work — it's how well they work. "I want a refund" and "I'd like my money back" score above 0.92 with `text-embedding-3-small`. Zero keyword overlap. Pure semantic capture. But that only works if your chunks are clean and focused. One topic per chunk. The embedding model can't fix bad chunking - it just faithfully converts whatever you give it into a vector.
