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
store = FAISS.load_local(
    "faiss_index", embeddings, allow_dangerous_deserialization=True
)

# Search
query = "How long do I have to return something?"
results = store.similarity_search_with_score(query, k=3)

for doc, score in results:
    # LangChain uses L2 distance by default - lower score = more similar
    print(f"Score: {score:.3f} | {doc.page_content}")
