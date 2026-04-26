import argparse
import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

# Force Offline mode for HuggingFace
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_HUB_OFFLINE'] = '0'

import faiss
import numpy as np
import pytesseract
import requests
from pdf2image import convert_from_path
from sentence_transformers import SentenceTransformer

# --- CONFIGURATION ---
DEFAULT_EMBED_MODEL = "BAAI/bge-m3"
DEFAULT_CLEANUP_MODEL = "Tharusha_Dilhara_Jayadeera/singemma" 

def clean_raw_artifacts(text: str) -> str:
    """Basic pre-cleaning to remove invisible artifacts before chunking."""
    text = text.replace("\u200d", " ").replace("\ufeff", " ").replace('\u200c', '')
    return re.sub(r"\s+", " ", text).strip()

def semantic_chunk_text(text: str, max_chars: int = 1000) -> List[str]:
    """Splits text naturally by paragraphs and sentences, avoiding mid-word cuts."""
    if not text: return []
    chunks = []
    
    # Split by double newlines (paragraphs)
    paragraphs = re.split(r'\n\s*\n', text)
    current_chunk = ""

    for para in paragraphs:
        para = para.strip()
        if not para: continue

        if len(current_chunk) + len(para) < max_chars:
            current_chunk += para + "\n\n"
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            
            # If a single paragraph is massive, split it by sentences
            if len(para) >= max_chars:
                sentences = re.split(r'(?<=[.!?])\s+', para)
                sub_chunk = ""
                for sent in sentences:
                    if len(sub_chunk) + len(sent) < max_chars:
                        sub_chunk += sent + " "
                    else:
                        if sub_chunk: chunks.append(sub_chunk.strip())
                        sub_chunk = sent + " "
                current_chunk = sub_chunk
            else:
                current_chunk = para + "\n\n"

    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return chunks

def ocr_pdf(pdf_path: Path, language: str = "sin+eng", dpi: int = 300) -> List[Dict[str, Any]]:
    print(f"--- Starting OCR for: {pdf_path.name} ---")
    pages = convert_from_path(str(pdf_path), dpi=dpi)
    page_texts = []
    for i, img in enumerate(pages, start=1):
        print(f"   [>] OCR Extraction: Page {i}/{len(pages)}...", end="\r")
        txt = pytesseract.image_to_string(img, lang=language)
        page_texts.append({"page": i, "text": clean_raw_artifacts(txt)})
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
        faiss.write_index(self.index, str(out_dir / "history_advanced.index"))
        payload = {"metadata": self.metadata, "embed_model_name": self.embed_model_name, "dim": self.dim}
        (out_dir / "history_meta_advanced.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    @staticmethod
    def load(out_dir: Path) -> Optional["RAGStore"]:
        idx_path = out_dir / "history_advanced.index"
        meta_path = out_dir / "history_meta_advanced.json"
        if not idx_path.exists() or not meta_path.exists(): return None
        index = faiss.read_index(str(idx_path))
        payload = json.loads(meta_path.read_text(encoding="utf-8"))
        return RAGStore(index=index, metadata=payload["metadata"], embed_model_name=payload["embed_model_name"], dim=payload["dim"])

def normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-12
    return vectors / norms

def ollama_process_chunk(text: str, model: str, host: str, timeout: int) -> Dict[str, Any]:
    prompt = (
        "SYSTEM: You are a professional Sinhala Historical Archivist. \n"
        "TASK: \n"
        "1. Fix the provided OCR Sinhala text.\n"
        "2. Extract historical entities.\n"
        "3. Generate 3 to 5 potential questions in Sinhala that a student might ask, which can be answered by this text.\n"
        "STRICT JSON OUTPUT FORMAT:\n"
        "{\n"
        "  \"cleaned_text\": \"The corrected full Sinhala text\",\n"
        "  \"entities\": [{\"name\": \"Entity Name\", \"type\": \"Person/Place/Event\", \"power\": \"Portuguese/Dutch/British/Sinhala\"}],\n"
        "  \"synthetic_questions\": [\"Question 1?\", \"Question 2?\"]\n"
        "}\n\n"
        f"RAW OCR CHUNK:\n{text}"
    )
    payload = {"model": model, "messages": [{"role": "user", "content": prompt}], "stream": False, "format": "json", "options": {"temperature": 0.1}}
    try:
        response = requests.post(f"{host}/api/chat", json=payload, timeout=timeout)
        response.raise_for_status()
        return json.loads(response.json()["message"]["content"])
    except Exception as e:
        print(f"\nAgent Error: {e}")
        return {"cleaned_text": text, "entities": [], "synthetic_questions": []}

def print_dashboard(source: str, pages: int, chunks: int, entities: List[Dict], start_time: float):
    elapsed = time.time() - start_time
    print("\n" + "="*60)
    print(f" 📜 ITHIHAASA GURU: INGESTION DASHBOARD ")
    print("="*60)
    print(f"SOURCE: {source} | Completed ✅ | TIME: {elapsed:.2f}s")
    print(f"📊 STATS: Pages: {pages} | Total FAISS Chunks: {chunks}")
    print("="*60 + "\n")

def build_index(pdf_paths: List[Path], out_dir: Path, embed_model_name: str, tesseract_lang: str, cleanup_model: str, cleanup_host: str, cleanup_timeout: int):
    store = RAGStore.load(out_dir)
    metadata, index = (store.metadata, store.index) if store else ([], None)
    embed_model = SentenceTransformer(embed_model_name)

    for pdf in pdf_paths:
        if any(m.get('source') == str(pdf) for m in metadata):
            print(f"Skipping {pdf.name} (Already in database).")
            continue

        start_time = time.time()
        extracted_pages = ocr_pdf(pdf, language=tesseract_lang)
        book_embed_texts = []
        book_entities = []
        
        for page in extracted_pages:
            raw_chunks = semantic_chunk_text(page["text"])
            
            for chunk_idx, raw_chunk in enumerate(raw_chunks):
                if len(raw_chunk) < 50: continue 
                
                print(f"   [>] Processing: Page {page['page']} | Chunk {chunk_idx + 1}/{len(raw_chunks)}...", end="\r")
                result = ollama_process_chunk(raw_chunk, cleanup_model, cleanup_host, cleanup_timeout)
                
                cleaned_text = result.get("cleaned_text", raw_chunk)
                questions = result.get("synthetic_questions", [])
                book_entities.extend(result.get("entities", []))
                
                # HYDE: Combine synthetic questions with cleaned text for superior FAISS matching
                augmented_text_for_embed = f"Questions: {' '.join(questions)}\n\nContent: {cleaned_text}"
                book_embed_texts.append(augmented_text_for_embed)
                
                metadata.append({
                    "source": str(pdf), 
                    "page": page["page"], 
                    "chunk_id": chunk_idx, 
                    "text": cleaned_text, 
                    "entities": result.get("entities", [])
                })

        if book_embed_texts:
            print(f"\nEncoding {len(book_embed_texts)} augmented chunks for {pdf.name}...")
            new_vectors = normalize(embed_model.encode(book_embed_texts, show_progress_bar=True).astype("float32"))
            
            if index is None: index = faiss.IndexFlatIP(new_vectors.shape[1])
            index.add(new_vectors)
            
            store = RAGStore(index=index, metadata=metadata, embed_model_name=embed_model_name, dim=index.d)
            store.save(out_dir)
            print_dashboard(pdf.name, len(extracted_pages), len(book_embed_texts), book_entities, start_time)

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
            "chunk_id": m.get("chunk_id", 0), 
            "text": m["text"] # Returns the cleaned text, hiding the synthetic questions from the user
        })
    return results

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Advanced Semantic OCR + FAISS intelligent RAG pipeline.")
    sub = p.add_subparsers(dest="command", required=True)

    build_p = sub.add_parser("build", help="Build or append to FAISS index")
    build_p.add_argument("--pdfs", nargs="+", default=["gr-8-pages.pdf", "gr-9-pages.pdf", "gr10-pages.pdf", "gr-11-pages.pdf"])
    build_p.add_argument("--out_dir", default="vector_store_v2") # Changed default to v2 so you don't corrupt your old DB
    build_p.add_argument("--embed_model", default=DEFAULT_EMBED_MODEL)
    build_p.add_argument("--tesseract_lang", default="sin+eng")
    build_p.add_argument("--cleanup_model", default=DEFAULT_CLEANUP_MODEL)
    build_p.add_argument("--cleanup_host", default="http://localhost:11434")
    build_p.add_argument("--cleanup_timeout", type=int, default=1200)

    q_p = sub.add_parser("query", help="Query local history index")
    q_p.add_argument("--query", required=True)
    q_p.add_argument("--out_dir", default="vector_store_v2")
    q_p.add_argument("--k", type=int, default=5)

    return p.parse_args()

def main() -> None:
    args = parse_args()
    if args.command == "build":
        pdf_paths = [Path(p) for p in args.pdfs]
        for p in pdf_paths:
            if not p.exists(): raise FileNotFoundError(f"Missing PDF: {p}")
        
        build_index(
            pdf_paths=pdf_paths,
            out_dir=Path(args.out_dir),
            embed_model_name=args.embed_model,
            tesseract_lang=args.tesseract_lang,
            cleanup_model=args.cleanup_model,
            cleanup_host=args.cleanup_host,
            cleanup_timeout=args.cleanup_timeout,
        )
        print(f"Database updated at: {args.out_dir}")
    elif args.command == "query":
        results = retrieve(query=args.query, out_dir=Path(args.out_dir), k=args.k)
        print(json.dumps(results, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()