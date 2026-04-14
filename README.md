# Sinhala Open-Ended Answer Scorer (Offline)

Offline scorer for Sri Lankan Colonial History answers in Sinhala.

## Stack
- LLM: Ollama `gemma3:12b`
- Workflow: LangGraph (stateful + cyclic grading validation)
- RAG: FAISS + `BAAI/bge-m3` multilingual embeddings
- Ontology: `rdflib` + Turtle ontology
- UI: Streamlit

## Files
- `history_config.json` - 5 Sinhala questions + 20-mark structured marking guides + Sinhala keywords
- `rag_pipeline.py` - OCR 4 PDFs and build/query FAISS vector store
- `ontology_colonial.ttl` - English schema, Sinhala labels, domain/range constraints
- `scorer_workflow.py` - LangGraph orchestrator/retrieval/grading/justification/output agents
- `app.py` - Streamlit frontend

## Prerequisites (offline runtime)
1. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Install Tesseract OCR and Sinhala language data (`sin`).
3. Ensure Poppler is installed (required by `pdf2image`).
4. Start Ollama and pull model:
   ```bash
   ollama pull gemma3:12b
   ollama run gemma3:12b
   ```

## Build OCR + FAISS index (from 4 textbooks)
```bash
python rag_pipeline.py build --pdfs gr8.pdf gr9.pdf gr10.pdf gr11.pdf --out_dir vector_store --embed_model BAAI/bge-m3 --tesseract_lang sin+eng
```

## Optional retrieval test
```bash
python rag_pipeline.py query --out_dir vector_store --query "ලන්දේසි පාලනයේ වෙළඳාම" --k 5
```

## Run Streamlit UI
```bash
streamlit run app.py
```

## Output
The UI returns:
- total score out of 20
- marks by each criterion
- Sinhala feedback
- evidence-based justification from RAG + ontology facts
