"""Build a FAISS index from sample_docs/*.txt.

Run from the project root with: python ingest.py
"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
DOCS_DIR = Path(os.getenv("SAMPLE_DOCS_DIR", str(BASE_DIR / "sample_docs")))
INDEX_DIR = Path(os.getenv("FAISS_INDEX_DIR", str(BASE_DIR / "faiss_index")))
EMBEDDING_MODEL = os.getenv(
    "GOOGLE_EMBEDDING_MODEL", "models/gemini-embedding-001"
)
CHUNK_SIZE = max(100, int(os.getenv("CHUNK_SIZE", "700")))
CHUNK_OVERLAP = max(0, int(os.getenv("CHUNK_OVERLAP", "100")))


def load_and_split() -> list[Document]:
    """Load text files and split them with a real, configurable overlap."""
    if not DOCS_DIR.exists():
        raise FileNotFoundError(f"Documents directory not found: {DOCS_DIR}")

    files = sorted(DOCS_DIR.glob("*.txt"))
    if not files:
        raise FileNotFoundError(f"No .txt documents found in {DOCS_DIR}")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=min(CHUNK_OVERLAP, CHUNK_SIZE - 1),
        separators=["\n\n", "\n", ". ", " ", ""],
        add_start_index=True,
    )

    chunks: list[Document] = []
    for path in files:
        text = path.read_text(encoding="utf-8").strip()
        if not text:
            print(f"Skipping empty document: {path.name}")
            continue

        parent_doc = Document(
            page_content=text,
            metadata={"source": path.name},
        )
        split_docs = splitter.split_documents([parent_doc])
        for chunk_number, chunk in enumerate(split_docs, start=1):
            chunk.metadata["chunk_id"] = f"{path.stem}-{chunk_number:04d}"
            chunks.append(chunk)

    if not chunks:
        raise ValueError(f"No non-empty text could be indexed from {DOCS_DIR}")
    return chunks


def main() -> None:
    if not os.getenv("GOOGLE_API_KEY"):
        raise RuntimeError(
            "GOOGLE_API_KEY is missing. Set it in .env or your environment."
        )

    print(f"Loading documents from: {DOCS_DIR}")
    chunks = load_and_split()
    print(f"Prepared {len(chunks)} chunks.")

    print("Creating embeddings and building FAISS index...")
    embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)
    vectorstore = FAISS.from_documents(chunks, embeddings)
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(INDEX_DIR))

    print(f"Index saved to: {INDEX_DIR}")
    print(f"Indexed chunks: {len(chunks)}")
    print("Keep the generated index.faiss and index.pkl together.")


if __name__ == "__main__":
    main()
