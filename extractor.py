import fitz  # PyMuPDF
import ollama
import json
from neo4j import GraphDatabase

# --- CONFIGURATION ---
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "Sumu2237"
MODEL_NAME = "gemma3:12b"  # Or "singemma" if it handles JSON well

class HistoryOntologyBuilder:
    def __init__(self):
        self.driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

    def close(self):
        self.driver.close()

    def extract_text_from_pdf(self, pdf_path):
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc:
            text += page.get_text()
        
        if not text.strip():
            print(f"⚠️ WARNING: No text extracted from {pdf_path}. Is it a scanned image?")
        else:
            print(f"✅ Extracted {len(text)} characters from {pdf_path}")
        return text

    def get_triples_from_llm(self, text_chunk):
        prompt = f"""
        Extract historical facts from this Sinhala text as a JSON list.
        Format: [{{"subject": "...", "predicate": "...", "object": "..."}}]
        Text: {text_chunk}
        """
        try:
            response = ollama.generate(model='gemma3:12b', prompt=prompt)
            raw_content = response['response'].strip()
            
            # Debug: See what the LLM is actually saying
            print(f"--- LLM Raw Response ---\n{raw_content[:200]}...") 

            # Extract JSON if LLM added conversational filler
            if "[" in raw_content and "]" in raw_content:
                start = raw_content.find("[")
                end = raw_content.rfind("]") + 1
                return json.loads(raw_content[start:end])
            return []
        except Exception as e:
            print(f"❌ LLM or JSON Error: {e}")
            return []

    def sync_to_neo4j(self, triples):
        with self.driver.session() as session:
            for t in triples:
                # Basic Merge without APOC just to be safe for now
                query = """
                MERGE (s:Entity {name: $sub})
                MERGE (o:Entity {name: $obj})
                """
                session.run(query, sub=t['subject'], obj=t['object'])
                print(f"🔗 Linked: {t['subject']} -> {t['object']}")

    def process_books(self, pdf_list):
        for pdf in pdf_list:
            print(f"Processing {pdf}...")
            full_text = self.extract_text_from_pdf(pdf)
            # Chunking text into ~2000 chars to fit context window
            chunks = [full_text[i:i+2000] for i in range(0, len(full_text), 2000)]
            
            for chunk in chunks:
                triples = self.get_triples_from_llm(chunk)
                if triples:
                    self.sync_to_neo4j(triples)
            print(f"Finished {pdf}")

# --- EXECUTION ---
if __name__ == "__main__":
    builder = HistoryOntologyBuilder()
    my_books = ["gr10-pages.pdf"]
    #my_books = ["gr8his.pdf", "gr9his.pdf", "gr10his.pdf", "gr11his.pdf"]
    builder.process_books(my_books)
    builder.close()