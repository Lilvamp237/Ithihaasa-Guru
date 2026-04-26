import os
import json
import torch
import requests
import faiss
import numpy as np
from pathlib import Path
from tqdm import tqdm
from docling.document_converter import DocumentConverter
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

# --- CONFIGURATION ---
OLLAMA_HOST = "http://localhost:11434"
CLEANUP_MODEL = "Tharusha_Dilhara_Jayadeera/singemma" 
EMBED_MODEL_NAME = "BAAI/bge-m3"
OUT_DIR = Path("vector_store_refined")
REFINED_JSON = OUT_DIR / "refined_data_stage1.json"
PDF_FILES = ["gr-8-pages.pdf", "gr-9-pages.pdf", "gr10-pages.pdf", "gr-11-pages.pdf"]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def ollama_refine(text: str):
    """Phase 1: LLM Cleanup Logic"""
    prompt = (
        "ඔබ ඉතිහාස සංස්කාරකවරයෙකි. මෙම පෙළෙහි ඇත්තේ පැරණි (Legacy/ASCII) අකුරු ක්‍රමයකි (උදා: Y%S ,xldfõ = ශ්‍රී ලංකාවේ).\n"
        "කරුණාකර මෙම සම්පූර්ණ ඡේදය නිවැරදි සිංහල යුනිකෝඩ් (Unicode) වලට පරිවර්තනය කරන්න. "
        "එහි ඇති අක්ෂර වින්‍යාස දෝෂද නිවැරදි කරන්න. "
        "අවසානයේදී මෙම ඡේදයට අදාළ කෙටි ප්‍රශ්නයක් සහ පිරිසිදු කළ පෙළ JSON ආකාරයෙන් ලබා දෙන්න.\n"
        "FORMAT: {\"cleaned_text\": \"...\", \"potential_question\": \"...\"}\n\n"
        f"TEXT: {text}"
    )
    try:
        response = requests.post(f"{OLLAMA_HOST}/api/chat", json={
            "model": CLEANUP_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "format": "json", "stream": False, "options": {"temperature": 0.1}
        }, timeout=120)
        return json.loads(response.json()["message"]["content"])
    except:
        return {"cleaned_text": text, "potential_question": ""}

def run_phase_1_cleanup():
    """Extracts text using Docling and cleans it using Singemma."""
    if REFINED_JSON.exists():
        print(f"✅ Phase 1 data found at {REFINED_JSON}. Skipping cleanup.")
        return

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    converter = DocumentConverter()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, separators=["\n\n", "\n", ".", " "])
    
    all_refined_chunks = []

    for pdf_name in PDF_FILES:
        if not Path(pdf_name).exists(): continue
        print(f"\n📖 Extracting Layout: {pdf_name}")
        result = converter.convert(pdf_name)
        markdown_text = result.document.export_to_markdown()
        raw_chunks = splitter.split_text(markdown_text)

        for i, raw_text in enumerate(tqdm(raw_chunks, desc=f"Cleaning {pdf_name}")):
            refined = ollama_refine(raw_text)
            all_refined_chunks.append({
                "source": pdf_name,
                "text": refined.get("cleaned_text", raw_text),
                "question": refined.get("potential_question", "")
            })
            # Save every 10 chunks as a backup
            if i % 10 == 0:
                REFINED_JSON.write_text(json.dumps(all_refined_chunks, ensure_ascii=False), encoding="utf-8")

    REFINED_JSON.write_text(json.dumps(all_refined_chunks, ensure_ascii=False), encoding="utf-8")
    print(f"🏁 Phase 1 Complete. {len(all_refined_chunks)} chunks cleaned.")

def run_phase_2_indexing():
    """Loads cleaned data and builds FAISS index using GPU."""
    print("\n🚀 Starting Phase 2: High-Speed Indexing...")
    data = json.loads(REFINED_JSON.read_text(encoding="utf-8"))
    
    # Load embedding model ONLY now. It gets 100% of VRAM.
    embed_model = SentenceTransformer(EMBED_MODEL_NAME, device=DEVICE, model_kwargs={"use_safetensors": True})
    
    all_vectors = []
    for item in tqdm(data, desc="Creating Vectors"):
        search_text = f"ප්‍රශ්නය: {item['question']} \nකරුණු: {item['text']}"
        vector = embed_model.encode(search_text, normalize_embeddings=True)
        all_vectors.append(vector)

    vectors_np = np.array(all_vectors).astype('float32')
    index = faiss.IndexFlatIP(vectors_np.shape[1])
    index.add(vectors_np)
    
    faiss.write_index(index, str(OUT_DIR / "history.index"))
    (OUT_DIR / "history_meta.json").write_text(json.dumps({"metadata": data}, ensure_ascii=False))
    print(f"✨ Success! Vector store built at {OUT_DIR}")

if __name__ == "__main__":
    run_phase_1_cleanup()
    # OPTIONAL: You can manually clear VRAM here if needed by closing Ollama between phases
    run_phase_2_indexing()