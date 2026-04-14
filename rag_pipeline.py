import argparse
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import faiss
import numpy as np
import pytesseract
from pdf2image import convert_from_path
from sentence_transformers import SentenceTransformer


DEFAULT_EMBED_MODEL = "BAAI/bge-m3"


def clean_text(text: str) -> str:
    text = text.replace("\u200d", " ").replace("\ufeff", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def chunk_text(text: str, chunk_size: int = 900, overlap: int = 150) -> List[str]:
    if not text:
        return []
    chunks: List[str] = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == len(text):
            break
        start = max(0, end - overlap)
    return chunks


def ocr_pdf(pdf_path: Path, language: str = "sin+eng", dpi: int = 300) -> str:
    print(f"--- Starting OCR for: {pdf_path.name} ---")
    pages = convert_from_path(str(pdf_path), dpi=dpi)
    num_pages = len(pages)
    print(f"Found {num_pages} pages.")
    
    page_texts: List[str] = []
    for i, img in enumerate(pages, start=1):
        print(f"  [>] Processing page {i}/{num_pages}...", end="\r")
        txt = pytesseract.image_to_string(img, lang=language)
        page_texts.append(f"[PAGE {i}] {clean_text(txt)}")
    
    print(f"\nFinished OCR for {pdf_path.name}")
    return "\n".join(page_texts)


@dataclass
class RAGStore:
    index: faiss.IndexFlatIP
    metadata: List[Dict[str, Any]]
    embed_model_name: str
    dim: int

    def save(self, out_dir: Path) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(out_dir / "history.index"))
        payload = {
            "metadata": self.metadata,
            "embed_model_name": self.embed_model_name,
            "dim": self.dim,
        }
        (out_dir / "history_meta.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    @staticmethod
    def load(out_dir: Path) -> "RAGStore":
        index = faiss.read_index(str(out_dir / "history.index"))
        payload = json.loads((out_dir / "history_meta.json").read_text(encoding="utf-8"))
        return RAGStore(
            index=index,
            metadata=payload["metadata"],
            embed_model_name=payload["embed_model_name"],
            dim=payload["dim"],
        )


def normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-12
    return vectors / norms


def build_index(
    pdf_paths: List[Path],
    out_dir: Path,
    embed_model_name: str = DEFAULT_EMBED_MODEL,
    tesseract_lang: str = "sin+eng",
) -> None:
    all_chunks: List[str] = []
    metadata: List[Dict[str, Any]] = []

    # Track OCR progress
    for pdf in pdf_paths:
        extracted = ocr_pdf(pdf, language=tesseract_lang)
        clean = clean_text(extracted)
        chunks = chunk_text(clean)
        print(f"Created {len(chunks)} chunks from {pdf.name}")
        for i, ch in enumerate(chunks):
            all_chunks.append(ch)
            metadata.append({"source": str(pdf), "chunk_id": i, "text": ch})

    if not all_chunks:
        raise ValueError("No text chunks extracted from PDFs.")

    print(f"\nLoading embedding model: {embed_model_name}...")
    model = SentenceTransformer(embed_model_name)
    
    print(f"Encoding {len(all_chunks)} chunks into vectors (this may take a while)...")
    # show_progress_bar=True is already here, which is good!
    vectors = model.encode(all_chunks, convert_to_numpy=True, show_progress_bar=True)
    
    print("Normalizing vectors and building FAISS index...")
    vectors = normalize(vectors.astype("float32"))

    dim = vectors.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(vectors)

    print(f"Saving store to {out_dir}...")
    store = RAGStore(index=index, metadata=metadata, embed_model_name=embed_model_name, dim=dim)
    store.save(out_dir)


def retrieve(
    query: str,
    out_dir: Path,
    k: int = 5,
) -> List[Dict[str, Any]]:
    store = RAGStore.load(out_dir)
    model = SentenceTransformer(store.embed_model_name)
    qvec = model.encode([query], convert_to_numpy=True).astype("float32")
    qvec = normalize(qvec)
    scores, indices = store.index.search(qvec, k)
    results: List[Dict[str, Any]] = []
    for rank, (score, idx) in enumerate(zip(scores[0], indices[0]), start=1):
        if idx < 0:
            continue
        m = store.metadata[idx]
        results.append(
            {
                "rank": rank,
                "score": float(score),
                "source": m["source"],
                "chunk_id": m["chunk_id"],
                "text": m["text"],
            }
        )
    return results


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="OCR + FAISS multilingual RAG pipeline (offline).")
    sub = p.add_subparsers(dest="command", required=True)

    build_p = sub.add_parser("build", help="OCR PDFs and build FAISS index")
    build_p.add_argument(
        "--pdfs",
        nargs="+",
        default=["gr8.pdf", "gr9.pdf", "gr10.pdf", "gr11.pdf"],
        help="Input textbook PDF paths",
    )
    build_p.add_argument("--out_dir", default="vector_store", help="Output directory for FAISS store")
    build_p.add_argument("--embed_model", default=DEFAULT_EMBED_MODEL, help="Embedding model")
    build_p.add_argument("--tesseract_lang", default="sin+eng", help="Tesseract language pack")

    q_p = sub.add_parser("query", help="Query built FAISS index")
    q_p.add_argument("--query", required=True, help="Sinhala or English query")
    q_p.add_argument("--out_dir", default="vector_store", help="Vector store directory")
    q_p.add_argument("--k", type=int, default=5, help="Top-k retrieval results")

    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "build":
        pdf_paths = [Path(p) for p in args.pdfs]
        for p in pdf_paths:
            if not p.exists():
                raise FileNotFoundError(f"Missing PDF: {p}")
        build_index(
            pdf_paths=pdf_paths,
            out_dir=Path(args.out_dir),
            embed_model_name=args.embed_model,
            tesseract_lang=args.tesseract_lang,
        )
        print(f"Built FAISS store at: {args.out_dir}")
    elif args.command == "query":
        results = retrieve(query=args.query, out_dir=Path(args.out_dir), k=args.k)
        print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
