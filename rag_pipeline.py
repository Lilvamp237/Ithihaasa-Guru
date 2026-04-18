import argparse
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

# Force Offline mode for HuggingFace
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_HUB_OFFLINE'] = '1'

import faiss
import numpy as np
import pytesseract
import requests
from pdf2image import convert_from_path
from sentence_transformers import SentenceTransformer

# --- CONFIGURATION ---
DEFAULT_EMBED_MODEL = "BAAI/bge-m3"
# Recommended: Use 4b for cleanup/ingestion if 12b is too slow for your hardware
DEFAULT_CLEANUP_MODEL = "Tharusha_Dilhara_Jayadeera/singemma" 

def clean_text(text: str) -> str:
    """Removes non-printable Sinhala artifacts and normalizes whitespace."""
    text = text.replace("\u200d", " ").replace("\ufeff", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def chunk_text(text: str, chunk_size: int = 900, overlap: int = 150) -> List[str]:
    """Splits cleaned text into overlapping segments for RAG."""
    if not text: return []
    chunks: List[str] = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end].strip()
        if chunk: chunks.append(chunk)
        if end == len(text): break
        start = max(0, end - overlap)
    return chunks

def ocr_pdf(pdf_path: Path, language: str = "sin+eng", dpi: int = 300) -> List[Dict[str, Any]]:
    """Extracts raw text from PDF pages using OCR."""
    print(f"--- Starting OCR for: {pdf_path.name} ---")
    pages = convert_from_path(str(pdf_path), dpi=dpi)
    num_pages = len(pages)
    
    page_texts: List[Dict[str, Any]] = []
    for i, img in enumerate(pages, start=1):
        print(f"   [>] OCR Extraction: Page {i}/{num_pages}...", end="\r")
        txt = pytesseract.image_to_string(img, lang=language)
        page_texts.append({"page": i, "text": txt})
    print(f"\nFinished OCR for {pdf_path.name}")
    return page_texts

@dataclass
class RAGStore:
    """Handles the local storage and loading of the FAISS index and metadata[cite: 24]."""
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
    def load(out_dir: Path) -> Optional["RAGStore"]:
        idx_path = out_dir / "history.index"
        meta_path = out_dir / "history_meta.json"
        if not idx_path.exists() or not meta_path.exists():
            return None
        
        index = faiss.read_index(str(idx_path))
        payload = json.loads(meta_path.read_text(encoding="utf-8"))
        return RAGStore(
            index=index,
            metadata=payload["metadata"],
            embed_model_name=payload["embed_model_name"],
            dim=payload["dim"],
        )

def normalize(vectors: np.ndarray) -> np.ndarray:
    """L2 Normalization for Inner Product search."""
    norms = np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-12
    return vectors / norms

def ollama_intelligent_ingest(text: str, model: str, host: str, timeout: int) -> Dict[str, Any]:
    """Agent that cleans text and extracts historical entities for the Ontology[cite: 25, 26]."""
    prompt = (
        "SYSTEM: You are a professional Sinhala Historical Archivist. \n"
        "TASK: Fix the provided OCR text and extract historical entities. \n"
        "STRICT JSON OUTPUT FORMAT:\n"
        "{\n"
        "  \"cleaned_text\": \"The corrected full Sinhala text\",\n"
        "  \"entities\": [\n"
        "    {\"name\": \"Entity Name\", \"type\": \"Governor/Event/Reform/Place\", \"power\": \"Portuguese/Dutch/British\"}\n"
        "  ]\n"
        "}\n\n"
        f"RAW TEXT:\n{text}"
    )
    url = f"{host}/api/chat"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "format": "json",
        "options": {"temperature": 0.1},
    }
    try:
        response = requests.post(url, json=payload, timeout=timeout)
        response.raise_for_status()
        return json.loads(response.json()["message"]["content"])
    except Exception as e:
        print(f"Agent Extraction Error: {e}")
        return {"cleaned_text": clean_text(text), "entities": []}

def log_raw_vs_cleaned(log_path: Path, source: str, page: int, raw_text: str, cleaned_text: str) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    entry = f"=== {source} | PAGE {page} ===\n[RAW]\n{raw_text[:900]}\n\n[CLEANED]\n{cleaned_text[:900]}\n\n"
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(entry)

def print_dashboard(source: str, pages: int, chunks: int, entities: List[Dict], start_time: float):
    """Displays the execution Status Dashboard in the terminal."""
    elapsed = time.time() - start_time
    print("\n" + "="*60)
    print(f" 📜 ITHIHAASA GURU: INGESTION DASHBOARD ")
    print("="*60)
    print(f"SOURCE: {source}")
    print(f"STATUS: Completed ✅ | TIME: {elapsed:.2f}s")
    print("-" * 60)
    print(f"📊 STATS: Pages: {pages} | Total FAISS Chunks: {chunks}")
    print("-" * 60)
    print(f"🔍 DISCOVERED ONTOLOGY ENTITIES:")
    seen = set()
    for ent in entities[:8]: # Show top 8
        if ent['name'] not in seen:
            print(f"   • {ent['name']} ({ent['type']}) -> {ent['power']}")
            seen.add(ent['name'])
    print("="*60 + "\n")

def build_index(
    pdf_paths: List[Path],
    out_dir: Path,
    embed_model_name: str = DEFAULT_EMBED_MODEL,
    tesseract_lang: str = "sin+eng",
    cleanup_model: str = DEFAULT_CLEANUP_MODEL,
    cleanup_host: str = "http://localhost:11434",
    cleanup_timeout: int = 1200,
    cleanup_batch: int = 1,
    cleanup_log_path: Optional[Path] = None,
) -> None:
    """Main pipeline for building or appending to the RAG database."""
    store = RAGStore.load(out_dir)
    if store:
        print(f"Loaded existing index. Appending new books...")
        metadata = store.metadata
        index = store.index
    else:
        print("Creating a new history knowledge base.")
        metadata, index = [], None

    embed_model = SentenceTransformer(embed_model_name)
    log_path = cleanup_log_path or (out_dir / "ocr_cleanup_samples.txt")

    for pdf in pdf_paths:
        if any(m['source'] == str(pdf) for m in metadata):
            print(f"Skipping {pdf.name} (Already in database).")
            continue

        start_time = time.time()
        extracted_pages = ocr_pdf(pdf, language=tesseract_lang)
        
        book_entities = []
        book_chunks = []
        
        # Process sequential for hardware stability
        for page in extracted_pages:
            print(f"   [>] Agent cleaning & extracting: Page {page['page']}...", end="\r")
            result = ollama_intelligent_ingest(page["text"], cleanup_model, cleanup_host, cleanup_timeout)
            
            cleaned = result.get("cleaned_text", clean_text(page["text"]))
            book_entities.extend(result.get("entities", []))
            
            log_raw_vs_cleaned(log_path, pdf.name, page["page"], page["text"], cleaned)
            
            chunks = chunk_text(cleaned)
            for i, ch in enumerate(chunks):
                book_chunks.append(ch)
                metadata.append({"source": str(pdf), "page": page["page"], "chunk_id": i, "text": ch})

        if book_chunks:
            print(f"\nEncoding {len(book_chunks)} new chunks for {pdf.name}...")
            new_vectors = normalize(embed_model.encode(book_chunks, show_progress_bar=True).astype("float32"))
            
            if index is None:
                index = faiss.IndexFlatIP(new_vectors.shape[1])
            index.add(new_vectors)
            
            # Save incrementally
            store = RAGStore(index=index, metadata=metadata, embed_model_name=embed_model_name, dim=index.d)
            store.save(out_dir)
            print_dashboard(pdf.name, len(extracted_pages), len(book_chunks), book_entities, start_time)

def retrieve(query: str, out_dir: Path, k: int = 5) -> List[Dict[str, Any]]:
    """Retrieves relevant local history context and metadata."""
    store = RAGStore.load(out_dir)
    if not store: return []
    model = SentenceTransformer(store.embed_model_name)
    qvec = normalize(model.encode([query], convert_to_numpy=True).astype("float32"))
    scores, indices = store.index.search(qvec, k)
    
    results = []
    for rank, (score, idx) in enumerate(zip(scores[0], indices[0]), start=1):
        if idx < 0: continue
        m = store.metadata[idx]
        results.append({
            "rank": rank, 
            "score": float(score), 
            "source": m["source"], 
            "page": m["page"], 
            "chunk_id": m.get("chunk_id", 0), # Ensure this key exists!
            "text": m["text"]
        })
    return results

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="OCR + FAISS intelligent RAG pipeline.")
    sub = p.add_subparsers(dest="command", required=True)

    build_p = sub.add_parser("build", help="Build or append to FAISS index")
    build_p.add_argument("--pdfs", nargs="+", default=["gr-8-pages.pdf", "gr-9-pages.pdf", "gr10-pages.pdf", "gr-11-pages.pdf"])
    build_p.add_argument("--out_dir", default="vector_store")
    build_p.add_argument("--embed_model", default=DEFAULT_EMBED_MODEL)
    build_p.add_argument("--tesseract_lang", default="sin+eng")
    build_p.add_argument("--cleanup_model", default=DEFAULT_CLEANUP_MODEL)
    build_p.add_argument("--cleanup_host", default="http://localhost:11434")
    build_p.add_argument("--cleanup_timeout", type=int, default=1200)
    build_p.add_argument("--cleanup_batch", type=int, default=1)
    build_p.add_argument("--cleanup_log", default="vector_store/ocr_cleanup_samples.txt")

    q_p = sub.add_parser("query", help="Query local history index")
    q_p.add_argument("--query", required=True)
    q_p.add_argument("--out_dir", default="vector_store")
    q_p.add_argument("--k", type=int, default=5)

    return p.parse_args()

def main() -> None:
    args = parse_args()
    if args.command == "build":
        pdf_paths = [Path(p) for p in args.pdfs]
        for p in pdf_paths:
            if not p.exists(): raise FileNotFoundError(f"Missing PDF: {p}")
        
        # Fixed: cleanup_log_path is now defined in build_index signature above
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
        print(f"Database updated at: {args.out_dir}")
    elif args.command == "query":
        results = retrieve(query=args.query, out_dir=Path(args.out_dir), k=args.k)
        print(json.dumps(results, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
