import argparse
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import faiss
import numpy as np
import pytesseract
import requests
from pdf2image import convert_from_path
from sentence_transformers import SentenceTransformer


DEFAULT_EMBED_MODEL = "BAAI/bge-m3"
DEFAULT_CLEANUP_MODEL = "Tharusha_Dilhara_Jayadeera/singemma"
#DEFAULT_CLEANUP_MODEL = "gemma3:12b"

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


def ocr_pdf(pdf_path: Path, language: str = "sin+eng", dpi: int = 300) -> List[Dict[str, Any]]:
    print(f"--- Starting OCR for: {pdf_path.name} ---")
    pages = convert_from_path(str(pdf_path), dpi=dpi)
    num_pages = len(pages)
    print(f"Found {num_pages} pages.")
    
    page_texts: List[Dict[str, Any]] = []
    for i, img in enumerate(pages, start=1):
        print(f"  [>] Processing page {i}/{num_pages}...", end="\r")
        txt = pytesseract.image_to_string(img, lang=language)
        page_texts.append({"page": i, "text": txt})
    
    print(f"\nFinished OCR for {pdf_path.name}")
    return page_texts


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


def ollama_cleanup(
    text: str,
    model: str,
    host: str,
    timeout: int,
) -> str:
    prompt = (
        "SYSTEM: You are a professional historical archivist. \n"
        "TASK: Fix the following OCR-extracted Sinhala text. \n"
        "STRICT RULES:\n"
        "1. DO NOT summarize. \n"
        "2. DO NOT omit any historical dates, names, or locations. \n"
        "3. Preserve every sentence found in the raw text. \n"
        "4. Fix spelling and rejoin broken Sinhala words. \n"
        "5. Output ONLY the fixed Sinhala text. No preamble.\n\n"
        f"RAW TEXT:\n{text}"
    )
    url = f"{host}/api/chat"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "options": {"temperature": 0.1},
    }
    response = requests.post(url, json=payload, timeout=timeout)
    response.raise_for_status()
    data = response.json()
    return str(data["message"]["content"]).strip()


def log_raw_vs_cleaned(
    log_path: Path,
    source: str,
    page: int,
    raw_text: str,
    cleaned_text: str,
    max_chars: int = 900,
) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    entry = (
        f"=== {source} | PAGE {page} ===\n"
        f"[RAW]\n{raw_text[:max_chars]}\n\n"
        f"[CLEANED]\n{cleaned_text[:max_chars]}\n\n"
    )
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(entry)


def clean_pages_with_ollama(
    pages: List[Dict[str, Any]],
    model: str,
    host: str,
    timeout: int,
    batch_size: int,
    log_path: Path,
    source: str,
) -> List[Dict[str, Any]]:
    cleaned_pages: List[Dict[str, Any]] = []
    for start in range(0, len(pages), batch_size):
        batch = pages[start : start + batch_size]
        with ThreadPoolExecutor(max_workers=batch_size) as executor:
            futures = [
                executor.submit(ollama_cleanup, item["text"], model, host, timeout) for item in batch
            ]
            for item, future in zip(batch, futures):
                cleaned = clean_text(future.result())
                log_raw_vs_cleaned(log_path, source, item["page"], item["text"], cleaned)
                cleaned_pages.append({"page": item["page"], "text": cleaned})
    return cleaned_pages


def build_index(
    pdf_paths: List[Path],
    out_dir: Path,
    embed_model_name: str = DEFAULT_EMBED_MODEL,
    tesseract_lang: str = "sin+eng",
    cleanup_model: str = DEFAULT_CLEANUP_MODEL,
    cleanup_host: str = "http://localhost:11434",
    cleanup_timeout: int = 1200,
    cleanup_batch: int = 3,
    cleanup_log_path: Path | None = None,
) -> None:
    all_chunks: List[str] = []
    metadata: List[Dict[str, Any]] = []

    # Track OCR progress
    for pdf in pdf_paths:
        extracted_pages = ocr_pdf(pdf, language=tesseract_lang)
        log_path = cleanup_log_path or (out_dir / "ocr_cleanup_samples.txt")
        cleaned_pages = clean_pages_with_ollama(
            pages=extracted_pages,
            model=cleanup_model,
            host=cleanup_host,
            timeout=cleanup_timeout,
            batch_size=cleanup_batch,
            log_path=log_path,
            source=pdf.name,
        )
        page_chunk_total = 0
        for page in cleaned_pages:
            chunks = chunk_text(page["text"])
            for i, ch in enumerate(chunks):
                all_chunks.append(ch)
                metadata.append(
                    {"source": str(pdf), "page": page["page"], "chunk_id": i, "text": ch}
                )
                page_chunk_total += 1
        print(f"Created {page_chunk_total} cleaned chunks from {pdf.name}")

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
        #default=["gr8.pdf", "gr9.pdf", "gr10.pdf", "gr11.pdf"],
        default=["gr-8-pages.pdf", "gr-9-pages.pdf", "gr10-pages.pdf", "gr-11-pages.pdf"],
        help="Input textbook PDF paths",
    )
    build_p.add_argument("--out_dir", default="vector_store", help="Output directory for FAISS store")
    build_p.add_argument("--embed_model", default=DEFAULT_EMBED_MODEL, help="Embedding model")
    build_p.add_argument("--tesseract_lang", default="sin+eng", help="Tesseract language pack")
    build_p.add_argument("--cleanup_model", default=DEFAULT_CLEANUP_MODEL, help="Ollama cleanup model")
    build_p.add_argument("--cleanup_host", default="http://localhost:11434", help="Ollama host URL")
    build_p.add_argument("--cleanup_timeout", type=int, default=1200, help="Ollama request timeout (s)")
    build_p.add_argument("--cleanup_batch", type=int, default=3, help="Pages per async batch")
    build_p.add_argument(
        "--cleanup_log",
        default="vector_store/ocr_cleanup_samples.txt",
        help="Path for raw vs cleaned samples log",
    )

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
            cleanup_model=args.cleanup_model,
            cleanup_host=args.cleanup_host,
            cleanup_timeout=args.cleanup_timeout,
            cleanup_batch=args.cleanup_batch,
            cleanup_log_path=Path(args.cleanup_log),
        )
        print(f"Built FAISS store at: {args.out_dir}")
    elif args.command == "query":
        results = retrieve(query=args.query, out_dir=Path(args.out_dir), k=args.k)
        print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
