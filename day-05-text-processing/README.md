# Document Loading in RAG

Real documents are messy. PDFs with scanned pages. DOCX files with tracked changes. HTML crawled from sites with nav menus, footers, and cookie banners baked into the text. If you feed garbage into your pipeline, you get garbage answers and no amount of prompt engineering fixes that.

Your RAG system won't just load `.txt` files. You'll hit:

- **PDFs** — the most common format, and the most painful. Scanned PDFs have no text layer at all.
- **DOCX** — Word documents with formatting, headers, footers, revision history.
- **HTML** — web pages where 40% of the content is navigation, ads, or boilerplate.
- **Markdown** — the cleanest of the bunch, still needs structure-aware handling.

Each format needs a different loader. Each loader produces different quality output.


## Part 1: The Right Tool for Each Format

Don't use one loader for everything. Match loader to format.

```python
from langchain_community.document_loaders import (
    PyPDFLoader,          # PDFs with text layer
    Docx2txtLoader,       # DOCX files
    BSHTMLLoader,         # HTML pages
    UnstructuredMarkdownLoader,  # Markdown
)
from pathlib import Path

def load_document(file_path: str):
    """
    Load document based on file extension.
    """
    path = Path(file_path)
    ext = path.suffix.lower()

    loaders = {
        ".pdf":  PyPDFLoader,
        ".docx": Docx2txtLoader,
        ".html": BSHTMLLoader,
        ".htm":  BSHTMLLoader,
        ".md":   UnstructuredMarkdownLoader,
    }

    if ext not in loaders:
        raise ValueError(f"Unsupported format: {ext}")

    loader = loaders[ext](file_path)
    docs = loader.load()

    print(f"Loaded {len(docs)} document(s) from {path.name}")
    return docs
```

## Part 2: Cleaning Is Non-Negotiable

Raw extracted text is never clean. Here's what you'll find:

- Multiple consecutive newlines from PDF layout
- Page numbers and headers repeated on every page: `"Page 3 of 47"`
- HTML tags that slipped through the parser
- Weird Unicode characters from copy-pasting

None of this helps your embeddings. All of it hurts retrieval accuracy.

```python
import re

def clean_text(text: str) -> str:
    """
    Production text cleaner.
    Order matters — run these in sequence.
    """
    # Remove HTML tags that loaders sometimes miss
    text = re.sub(r'<[^>]+>', ' ', text)

    # Collapse multiple whitespace/newlines into single space
    text = re.sub(r'\s+', ' ', text)

    # Remove page number patterns: "Page 1 of 10", "- 3 -", etc.
    text = re.sub(r'[-–]\s*\d+\s*[-–]', '', text)
    text = re.sub(r'page\s+\d+\s+of\s+\d+', '', text, flags=re.IGNORECASE)

    return text.strip()


def clean_documents(docs):
    """Apply cleaner to all loaded documents. Drop empty ones."""
    cleaned = []
    for doc in docs:
        doc.page_content = clean_text(doc.page_content)
        if len(doc.page_content) >= 50:  # drop near-empty docs
            cleaned.append(doc)
    return cleaned
```

---

## Part 3: Don't Throw Metadata Away

Every document object comes with metadata. Keep it.

```python
docs = load_document("company_policy.pdf")

# PyPDFLoader automatically adds:
# doc.metadata["source"] — file path
# doc.metadata["page"]   — page number (0-indexed)

# Add your own metadata for downstream filtering
for doc in docs:
    doc.metadata["document_type"] = "policy"
    doc.metadata["ingested_at"] = "2025-10-20"

# Later in retrieval, you can filter:
# "only search documents where document_type == policy"
```

This becomes critical when you're running metadata filters on a vector database with 100K+ documents. Build the habit now.


## Part 4: Handling PDFs That Have No Text Layer

Scanned PDFs are images(OCR), not text. A regular PDF loader returns empty strings.

```python
def is_scanned_pdf(file_path: str) -> bool:
    """Detect if a PDF is scanned (image-only) rather than text-based."""
    import pypdf

    reader = pypdf.PdfReader(file_path)
    total_chars = sum(
        len(page.extract_text() or "")
        for page in reader.pages[:3]  # sample first 3 pages
    )
    # Heuristic: fewer than 100 chars per sampled page = probably scanned
    return total_chars < 100 * min(3, len(reader.pages))


def load_pdf_smart(file_path: str):
    """Use text extraction for normal PDFs. Fall back to OCR for scanned."""
    if is_scanned_pdf(file_path):
        # Requires: pip install pytesseract pdf2image
        # Requires: tesseract-ocr system package installed
        from langchain_community.document_loaders import UnstructuredPDFLoader
        loader = UnstructuredPDFLoader(file_path, strategy="ocr_only")
        print(f"[OCR] Scanned PDF detected: {file_path}")
    else:
        loader = PyPDFLoader(file_path)
        print(f"[TEXT] Normal PDF detected: {file_path}")

    return loader.load()
```

OCR is slow (5–30 seconds per page) and imperfect. Flag scanned documents early rather than letting them silently produce empty chunks downstream.


## Part 5: Chunking

The goal of chunking is simple: split documents into pieces small enough for the embedding model to encode accurately, large enough to preserve meaningful context.

## Three Strategies That Actually Matter

### 1. Fixed-Size Chunking

Split by character count with overlap. Fast, simple, predictable.

```python
from langchain_text_splitters import CharacterTextSplitter

splitter = CharacterTextSplitter(
    chunk_size=1000,      # characters per chunk
    chunk_overlap=200,    # overlap between consecutive chunks
    separator="\n\n",     # prefer splitting at paragraph breaks
)

chunks = splitter.split_documents(docs)
```

The overlap is not optional. Without it, a sentence that straddles a chunk boundary gets cut in half - and the retrieval query that matches it finds nothing.

**When to use it:** quick prototyping, uniform documents like product descriptions or support tickets.

**When not to use it:** technical docs, legal contracts, anything with hierarchical structure.


### 2. Recursive Character Splitting

This is production default. Tries a list of separators in order - paragraphs, then sentences, then words - so it splits at the most natural boundary available.

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    # Tries these separators in order until chunks are small enough
    separators=["\n\n", "\n", ". ", " ", ""],
)

chunks = splitter.split_documents(docs)

# Each chunk preserves the original metadata
# chunks[0].metadata["source"] still points to the source file
# chunks[0].metadata["page"] still tells you which page it came from
```

This is what you should be using by default. It rarely produces mid-sentence splits and handles most document types well.


### 3. Markdown-Aware Splitting

If your documents have structure - headers, sections, code blocks - use a splitter that understands it. Recursive splitting will cut across section boundaries. Markdown splitting won't.

```python
from langchain_text_splitters import MarkdownHeaderTextSplitter

# Split on headers, add them as metadata
header_splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=[
        ("#", "h1"),
        ("##", "h2"),
        ("###", "h3"),
    ]
)

# Then size-limit the resulting sections
size_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100,
)

# Two-pass: split by structure, then by size
md_docs = header_splitter.split_text(raw_markdown)
final_chunks = size_splitter.split_documents(md_docs)

# Each chunk now carries header metadata:
# chunk.metadata["h1"] = "Phase 3: RAG"
# chunk.metadata["h2"] = "Tools to Use"
```

The header metadata is searchable. You can filter to only chunks from the "Tools to Use" sections across all your docs.

---

## Chunk Size: The Tradeoff

There's no universal right answer. Here's how to think about it:

| Chunk Size | Retrieval | Generation |
|---|---|---|
| Small (200–400 tokens) | More precise — easier to find the exact passage | Less context per chunk — LLM may lack surrounding info |
| Medium (500–1000 tokens) | Good balance for most use cases | Enough context for coherent answers |
| Large (1000–2000 tokens) | Coarser matches — may retrieve irrelevant content | More context, but noisy |

Start at 1000 characters / 200 token overlap. Measure retrieval accuracy. Adjust from there — don't guess.


## Full Pipeline: Load → Clean → Chunk

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

def build_chunks(file_paths: list[str]) -> list:
    """
    Full pipeline: load, clean, chunk.
    Returns chunks ready for embedding.
    """
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

            # Tag chunks with document type for later filtering
            for chunk in chunks:
                chunk.metadata["file"] = Path(path).name

            all_chunks.extend(chunks)
            print(f"{Path(path).name}: {len(chunks)} chunks")

        except Exception as e:
            print(f"{Path(path).name}: {e}")
            continue

    print(f"\nTotal: {len(all_chunks)} chunks ready for embedding")
    return all_chunks


# Usage
files = ["docs/annual_report.pdf", "docs/faq.html", "docs/readme.md"]
chunks = build_chunks(files)
```


## What You Should Know Before Moving On

**Chunk quality determines retrieval quality.** You can't fix bad chunks with a better embedding model or a fancier vector database. Get this right first.

**Recursive splitting is the right default.** Fixed-size is for prototypes. Semantic chunking (splitting on meaning rather than size) exists but adds latency and complexity — reach for it only when recursive splitting demonstrably fails.

**Overlap has a cost.** 20% overlap on a 1M document corpus means storing and embedding 20% more chunks. That's real money at scale. Don't set overlap higher than you need.

**Always inspect your chunks.** Print 10 random chunks after splitting. If you see cut-off sentences, misaligned content, or chunks that are just whitespace and page numbers, your loader or cleaner needs work - not your chunker.

**PyPDFLoader vs pdfplumber vs Unstructured** - PyPDF is fast and good enough for text PDFs. pdfplumber handles tables better. Unstructured handles complex layouts (multi-column, headers, figures) but is slower and heavier. Match the tool to the document complexity.
