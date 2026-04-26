import os
import json
import torch
import requests
from pathlib import Path
from tqdm import tqdm
from docling.document_converter import DocumentConverter
from langchain_text_splitters import RecursiveCharacterTextSplitter

# --- CONFIGURATION ---
OLLAMA_HOST = "http://localhost:11434"
CLEANUP_MODEL = "Tharusha_Dilhara_Jayadeera/singemma"
OUT_DIR = Path("vector_store_refined")
LOG_FILE = OUT_DIR / "docling_refinement_log.txt"
CHECKPOINT_FILE = OUT_DIR / "docling_checkpoint.json"
PDF_FILES = ["gr-8-pages.pdf", "gr-9-pages.pdf", "gr10-pages.pdf", "gr-11-pages.pdf"]

def legacy_to_unicode_prepass(text: str) -> str:
    """Pre-converts obvious ASCII patterns to help the LLM."""
    mappings = {
        "Y%S": "ශ්‍රී", ",xld": "ලංකා", "fõ": "වේ", "ishji": "ශතවර්ෂය",
        "mD;=.SiS": "පෘතුගීසි", ",kafoaiS": "ලන්දේසි", "bx.%Sis": "ඉංග්‍රීසි",
        "hqfrdam": "යුරෝපා", "foaYmd,k": "දේශපාලන", "wd¾Ól": "ආර්ථික",
        "iudc": "සමාජ", "hg;a": "යටත්", "b;sydih": "ඉතිහාසය", "rch": "රජු"
    }
    for legacy, uni in mappings.items():
        text = text.replace(legacy, uni)
    return text

def refine_with_singemma(raw_text: str):
    """Sends the ASCII text to Singemma to get clean Sinhala Unicode."""
    pre_cleaned = legacy_to_unicode_prepass(raw_text)
    
    prompt = (
        "ඔබ ඉතිහාස සංස්කාරකවරයෙකි. පහත ඇත්තේ ASCII (Legacy) අකුරු සහිත ඡේදයකි. "
        "එය සම්පූර්ණයෙන්ම නිවැරදි සිංහල යුනිකෝඩ් (Unicode) වලට පරිවර්තනය කර, "
        "අක්ෂර වින්‍යාසය සහ පිල්ලම් දෝෂ නිවැරදි කරන්න. "
        "අවසානයේදී මෙම ඡේදයට අදාළ කෙටි ප්‍රශ්නයක් සහ පිරිසිදු කළ පෙළ JSON ආකාරයෙන් පමණක් ලබා දෙන්න.\n"
        "STRICT JSON FORMAT: {\"cleaned_text\": \"...\", \"potential_question\": \"...\"}\n\n"
        f"DATA: {pre_cleaned}"
    )

    try:
        response = requests.post(f"{OLLAMA_HOST}/api/chat", json={
            "model": CLEANUP_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "format": "json", "stream": False, "options": {"temperature": 0.1}
        }, timeout=150)
        
        return json.loads(response.json()["message"]["content"])
    except:
        return {"cleaned_text": "[ERROR: REFINE FAILED]", "potential_question": ""}

def run_docling_refinement():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    converter = DocumentConverter()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    
    all_data = []

    with open(LOG_FILE, "a", encoding="utf-8") as log:
        for pdf in PDF_FILES:
            if not Path(pdf).exists(): continue
            print(f"\n📑 Docling is parsing: {pdf}")
            
            # Step 1: Layout-aware extraction
            result = converter.convert(pdf)
            markdown = result.document.export_to_markdown()
            chunks = splitter.split_text(markdown)

            for i, chunk in enumerate(tqdm(chunks, desc=f"Refining {pdf}")):
                # Step 2: LLM Refinement
                refined = refine_with_singemma(chunk)
                
                entry = {
                    "source": pdf,
                    "raw": chunk[:100] + "...", 
                    "text": refined.get("cleaned_text", ""),
                    "question": refined.get("potential_question", "")
                }
                all_data.append(entry)

                # Step 3: Log the progress (Like your previous ocr_cleanup_samples.txt)
                log.write(f"=== PAGE CHUNK {i} ({pdf}) ===\n")
                log.write(f"[RAW]: {chunk[:100]}\n")
                log.write(f"[CLEANED]: {entry['text']}\n")
                log.write(f"[QUESTION]: {entry['question']}\n\n")
                log.flush()

                # Step 4: Save JSON Checkpoint
                CHECKPOINT_FILE.write_text(json.dumps(all_data, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n✅ Done! Check {LOG_FILE} to see the converted Sinhala text.")

if __name__ == "__main__":
    run_docling_refinement()