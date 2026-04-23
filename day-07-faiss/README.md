# FAISS: Store and Search Your Embeddings
After the embedding is completed, now the question is: where do vectors live, and how do you find the closest ones fast? FAISS (Facebook AI Similarity Search) is the standard answer for local, high-performance vector search. It's a C++ library with a Python wrapper. It does one thing: given a query vector, return the k most similar vectors from your index.


## The Metadata Problem 

FAISS stores vectors only. It doesn't store text and metadata. When you search, you get back indices like `[42, 7, 156]`. FAISS can't tell you what chunk 42 says. You need to maintain a parallel list yourself, a metadata sidecar which is indexed in the same order as your FAISS index.

```
FAISS index:   [vec_0,  vec_1,  vec_2, ..., vec_N]
chunks list:   [text_0, text_1, text_2, ..., text_N]
```
FAISS index position 42 → `chunks[42]`. Always keep them in sync and never shuffle one without the other.


## Example 1: Custom FAISS Store

### Implementation

```python
import json
import faiss
import numpy as np
from dataclasses import dataclass
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
```

### Using the Store

```python
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

# Save it - don't re-embed every startup
store.save("store.faiss", "store.json")
print(f"Indexed {store.index.ntotal} vectors")
```

### Searching the Store

```python
from sentence_transformers import SentenceTransformer
from faiss_store import FAISSStore

model = SentenceTransformer("all-MiniLM-L6-v2")
store = FAISSStore.load("store.faiss", "store.json")

query = "How long do I have to return something?"
q_vec = model.encode(query, normalize_embeddings=True)

results = store.search(q_vec, k=3)

for r in results:
    print(f"Score: {r.score:.3f} | {r.text}")
```

Output:
```
Score: 0.821 | Refund policy is 30 days from purchase.
Score: 0.412 | To cancel your subscription, go to account settings.
Score: 0.381 | Shipping takes 3 to 5 business days.
```

The top result is right. Score 0.82 is strong. The others are low and under 0.5 means weak match, which is expected.

Three things to notice in this implementation:

1. **`IndexFlatIP` + `faiss.normalize_L2`** - This is cosine similarity. Use `IndexFlatL2` and you get euclidean distance, which is biased by vector magnitude. Cosine is what you want for text.
2. **Persist both files** - `store.faiss` has the vectors. `store.json` has the text. Lose either one and the store is useless.
3. **`i != -1` guard** - FAISS returns `-1` index when `k` is larger than the number of stored vectors. Always handle it.


## Example 2: LangChain FAISS

LangChain wraps FAISS and handles the metadata sidecar, embedding calls, and persistence automatically. This is what you'd use in most projects.

```python
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# LangChain embedding wrapper around SentenceTransformers
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

chunks = [
    "Refund policy is 30 days from purchase.",
    "Office hours are 9am to 5pm, Monday to Friday.",
    "Shipping takes 3 to 5 business days.",
    "To cancel your subscription, go to account settings.",
    "We support Visa, Mastercard, and PayPal.",
]

# Embeds + indexes in one call
store = FAISS.from_texts(chunks, embeddings)

# Persist to disk
store.save_local("faiss_index")

# Load back
store = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

# Search
query = "How long do I have to return something?"
results = store.similarity_search_with_score(query, k=3)

for doc, score in results:
    # LangChain uses L2 distance by default - lower score = more similar
    print(f"Score: {score:.3f} | {doc.page_content}")
```

Output:
```
Score: 0.312 | Refund policy is 30 days from purchase.
Score: 1.247 | To cancel your subscription, go to account settings.
Score: 1.389 | Shipping takes 3 to 5 business days.
```

Note the score direction: LangChain's default is **L2 distance**, so lower = more similar - the opposite of cosine similarity. The right result is still first, but don't compare scores across implementations.

If you want cosine similarity scores in LangChain:

```python
results = store.similarity_search_with_relevance_scores(query, k=3)
# Returns (doc, score) where score is cosine similarity 0→1
```

## Index Types (Quick Reference)

You'll see three types in FAISS docs:

| Index | When to use | Notes |
|---|---|---|
| `IndexFlatIP` / `IndexFlatL2` | < 100K vectors | Exact search, no training |
| `IndexIVFFlat` | 100K–10M vectors | Approximate, requires training step |
| `IndexHNSWFlat` | 1M+ vectors | Graph-based, no training, high RAM |

For a RAG system over documents - a company wiki, a codebase, a support knowledge base - you're almost never past 100K chunks. `IndexFlatIP` is the right choice. Don't add complexity you don't need.
