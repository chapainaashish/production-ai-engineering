# pip install langchain langchain-community pypdf docx2txt beautifulsoup4 unstructured langchain-text-splitters pypdf

import re
from pathlib import Path

from langchain_community.document_loaders import (
    BSHTMLLoader,
    Docx2txtLoader,
    PyPDFLoader,
    UnstructuredMarkdownLoader,
)
from langchain_text_splitters import (
    CharacterTextSplitter,
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)


# 1. LOAD
def load_document(file_path: str):
    path = Path(file_path)
    ext = path.suffix.lower()

    loaders = {
        ".pdf": PyPDFLoader,
        ".docx": Docx2txtLoader,
        ".html": BSHTMLLoader,
        ".htm": BSHTMLLoader,
        ".md": UnstructuredMarkdownLoader,
    }

    if ext not in loaders:
        raise ValueError(f"Unsupported format: {ext}")

    loader = loaders[ext](file_path)
    docs = loader.load()
    print(f"Loaded {len(docs)} document(s) from {path.name}")
    return docs


# 2. CLEAN
def clean_text(text: str) -> str:
    text = re.sub(r"<[^>]+>", " ", text)  # stray HTML tags
    text = re.sub(r"\s+", " ", text)  # collapse whitespace
    text = re.sub(r"[-–]\s*\d+\s*[-–]", "", text)  # "- 3 -" page markers
    text = re.sub(r"page\s+\d+\s+of\s+\d+", "", text, flags=re.IGNORECASE)
    return text.strip()


def clean_documents(docs):
    cleaned = []
    for doc in docs:
        doc.page_content = clean_text(doc.page_content)
        if len(doc.page_content) >= 50:  # drop near-empty docs
            cleaned.append(doc)
    return cleaned


# 3. SCANNED PDF DETECTION
def is_scanned_pdf(file_path: str) -> bool:
    import pypdf

    reader = pypdf.PdfReader(file_path)
    total_chars = sum(len(page.extract_text() or "") for page in reader.pages[:3])
    return total_chars < 100 * min(3, len(reader.pages))


def load_pdf_smart(file_path: str):
    if is_scanned_pdf(file_path):
        # requires: pip install pytesseract pdf2image + tesseract-ocr system package
        from langchain_community.document_loaders import UnstructuredPDFLoader

        loader = UnstructuredPDFLoader(file_path, strategy="ocr_only")
        print(f"[OCR]  Scanned PDF detected: {file_path}")
    else:
        loader = PyPDFLoader(file_path)
        print(f"[TEXT] Normal PDF detected: {file_path}")
    return loader.load()


# 4. CHUNK STRATEGIES
def chunk_fixed(docs):
    """Fast, simple. Good for uniform docs (tickets, product descriptions)."""
    splitter = CharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separator="\n\n",
    )
    return splitter.split_documents(docs)


def chunk_recursive(docs):
    """Production default. Tries paragraph → sentence → word boundaries."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_documents(docs)


def chunk_markdown(raw_markdown: str):
    """Structure-aware split for Markdown — preserves section boundaries."""
    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[
            ("#", "h1"),
            ("##", "h2"),
            ("###", "h3"),
        ]
    )
    size_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
    )
    sections = header_splitter.split_text(raw_markdown)
    return size_splitter.split_documents(sections)


# 5. FULL PIPELINE
def build_chunks(file_paths: list[str]) -> list:
    """Load → clean → chunk. Returns chunks ready for embedding."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )
    all_chunks = []

    for path in file_paths:
        try:
            docs = load_document(path)
            docs = clean_documents(docs)
            chunks = splitter.split_documents(docs)

            for chunk in chunks:
                chunk.metadata["file"] = Path(path).name

            all_chunks.extend(chunks)
            print(f"  {Path(path).name}: {len(chunks)} chunks")

        except Exception as e:
            print(f"  {Path(path).name}: FAILED — {e}")
            continue

    print(f"\nTotal: {len(all_chunks)} chunks ready for embedding")
    return all_chunks


# 6. INSPECT HELPER
def inspect_chunks(chunks, n: int = 5):
    """Print a sample of chunks. Always do this before embedding."""
    import random

    sample = random.sample(chunks, min(n, len(chunks)))
    print(f"\n{'='*60}")
    print(f"CHUNK INSPECTION — {n} random samples")
    print(f"{'='*60}")
    for i, chunk in enumerate(sample, 1):
        print(
            f"\n[{i}] source: {chunk.metadata.get('source', 'unknown')} "
            f"| page: {chunk.metadata.get('page', '-')} "
            f"| chars: {len(chunk.page_content)}"
        )
        print(chunk.page_content[:300])
        print("...")


if __name__ == "__main__":
    files = [
        "docs/sample.pdf",
        "docs/sample.docx",
        "docs/sample.html",
        "docs/readme.md",
    ]

    existing = [f for f in files if Path(f).exists()]

    if not existing:
        print("No files found. Add some docs to a 'docs/' folder and re-run.")
    else:
        chunks = build_chunks(existing)
        inspect_chunks(chunks)

        # Metadata tagging example
        for chunk in chunks:
            chunk.metadata["ingested_at"] = "2025-10-20"
