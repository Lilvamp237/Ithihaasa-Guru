import os
import json
import time
import requests
import faiss
import numpy as np
from pathlib import Path
from tqdm import tqdm
from typing import List, Dict, Any

# Force Offline Mode
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_HUB_OFFLINE'] = '0'
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'

from docling.document_converter import DocumentConverter
from docling.datamodel.pipeline_options import PdfPipelineOptions, RapidOcrOptions
from docling.datamodel.accelerator_options import AcceleratorOptions, AcceleratorDevice
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

# --- CONFIGURATION ---
OLLAMA_HOST = "http://localhost:11434"
CLEANUP_MODEL = "Tharusha_Dilhara_Jayadeera/singemma # Or your specific Sinhala model" #"gemma3:12b" 
EMBED_MODEL_NAME = "BAAI/bge-m3"
OUT_DIR = Path("vector_store_refined")
PDF_FILES = ["gr-8-pages.pdf", "gr-9-pages.pdf", "gr10-pages.pdf", "gr-11-pages.pdf"]

def ollama_refine(text: str) -> Dict[str, str]:
    """Uses LLM to fix OCR and generate a potential question for the chunk."""
    prompt = (
        "SYSTEM: You are a professional Sinhala History Editor.\n"
        "TASK: 1. Fix OCR errors and Sinhala diacritics (Pillam) in the text. "
        "2. Generate one short question that this text perfectly answers.\n"
        "STRICT JSON FORMAT:\n"
        "{\n"
        "  \"cleaned_text\": \"...\",\n"
        "  \"potential_question\": \"...\"\n"
        "}\n\n"
        f"TEXT:\n{text}"
    )
    
    try:
        response = requests.post(
            f"{OLLAMA_HOST}/api/chat",
            json={
                "model": CLEANUP_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "format": "json",
                "stream": False,
                "options": {"temperature": 0.1}
            },
            timeout=120
        )
        return json.loads(response.json()["message"]["content"])
    except Exception as e:
        return {"cleaned_text": text, "potential_question": ""}

def rebuild_knowledge_base():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    converter = DocumentConverter()
    embed_model = SentenceTransformer(EMBED_MODEL_NAME)
    
    # Recursive splitter handles Sinhala sentence structure better than fixed-size
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", "।", ".", " "] # Added Sinhala full stop if present
    )

    all_metadata = []
    all_vectors = []

    for pdf_name in PDF_FILES:
        pdf_path = Path(pdf_name)
        if not pdf_path.exists():
            print(f"Skipping {pdf_name}: File not found.")
            continue

        print(f"\n--- Processing {pdf_name} with Docling ---")
        # Step 1: Layout-aware conversion to Markdown
        result = converter.convert(str(pdf_path))
        markdown_text = result.document.export_to_markdown()

        # Step 2: Structural Chunking
        raw_chunks = splitter.split_text(markdown_text)
        print(f"Created {len(raw_chunks)} structural chunks.")

        # Step 3: LLM Refinement Loop
        for i, raw_text in enumerate(tqdm(raw_chunks, desc=f"Refining {pdf_name}")):
            refined = ollama_refine(raw_text)
            cleaned_text = refined.get("cleaned_text", raw_text)
            question = refined.get("potential_question", "")

            # The 'Searchable Text' includes the question to boost retrieval accuracy
            searchable_text = f"Question: {question} \nContext: {cleaned_text}"
            
            # Create Embedding
            vector = embed_model.encode(searchable_text, normalize_embeddings=True)
            
            all_vectors.append(vector)
            all_metadata.append({
                "source": pdf_name,
                "chunk_id": i,
                "text": cleaned_text,
                "question_meta": question
            })

    # Step 4: Save to FAISS
    if all_vectors:
        print("\nSaving Refined Knowledge Base...")
        vectors_np = np.array(all_vectors).astype('float32')
        index = faiss.IndexFlatIP(vectors_np.shape[1])
        index.add(vectors_np)
        
        faiss.write_index(index, str(OUT_DIR / "history.index"))
        
        meta_payload = {
            "metadata": all_metadata,
            "embed_model_name": EMBED_MODEL_NAME,
            "dim": vectors_np.shape[1]
        }
        (OUT_DIR / "history_meta.json").write_text(
            json.dumps(meta_payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"Success! {len(all_metadata)} chunks indexed in {OUT_DIR}")

if __name__ == "__main__":
    rebuild_knowledge_base()