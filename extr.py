import os
import json
import ollama
import pytesseract
from pdf2image import convert_from_path
from neo4j import GraphDatabase

# --- CONFIGURATION ---
# If Tesseract is not in your PATH, uncomment and set the line below:
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "Sumu2237"
MODEL_NAME = "gemma3:12b"

class SinhalaHistoryOntologyBuilder:
    def __init__(self):
        self.driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

    def close(self):
        self.driver.close()

    def pdf_to_unicode_text(self, pdf_path):
        """Converts PDF to images and runs Sinhala OCR."""
        print(f"📄 Processing PDF: {pdf_path}")
        # DPI 300 is ideal for accurate Sinhala character recognition
        pages = convert_from_path(pdf_path, 300)
        full_text = ""
        
        for i, page in enumerate(pages):
            print(f"  🔍 OCR-ing Page {i+1}...")
            # 'sin' specifies the Sinhala language pack
            text = pytesseract.image_to_string(page, lang='sin')
            full_text += text + "\n"
        
        return full_text

    def extract_triples(self, text_chunk):
        """Uses Gemma 3:12b to extract Subject-Predicate-Object triples."""
        prompt = f"""
        ඔබ ඉතිහාසඥයෙක්. ලබා දී ඇති සිංහල පෙළ ඇසුරෙන් ඓතිහාසික කරුණු (triples) නිවැරදිව උකහා ගන්න.
        ප්‍රතිදානය JSON ආකෘතියෙන් පමණක් ලබා දෙන්න.

        අනුගමනය කළ යුතු නීති:
        1. විෂය (subject), ක්‍රියාව (predicate), සහ වස්තුව (object) යන අංශ තුනම සිංහල යුනිකෝඩ් (Unicode) වලින් තිබිය යුතුය.
        2. සම්බන්ධතා (predicate) කෙටි විය යුතුය (උදා: "පාලනය_කළේ", "අත්සන්_කළේ").
        
        පෙළ: 
        {text_chunk}

        ප්‍රතිදාන ආකෘතිය:
        [
          {{"subject": "පෘතුගීසීන්", "predicate": "පැමිණියේ", "object": "1498"}},
          {{"subject": "පළමුවන විමලධර්මසූරිය රජු", "predicate": "පරාජය_කළේ", "object": "අසවේදු"}}
        ]
        """
        try:
            response = ollama.generate(model=MODEL_NAME, prompt=prompt)
            raw_response = response['response'].strip()
            
            # Extract JSON from the markdown block if present
            if "```json" in raw_response:
                raw_response = raw_response.split("```json")[1].split("```")[0].strip()
            elif "[" in raw_response:
                start = raw_response.find("[")
                end = raw_response.rfind("]") + 1
                raw_response = raw_response[start:end]
            
            return json.loads(raw_response)
        except Exception as e:
            print(f"⚠️ Error parsing LLM response: {e}")
            return []

    def sync_to_neo4j(self, triples):
        """Pushes triples to the Neo4j Knowledge Graph."""
        with self.driver.session() as session:
            for t in triples:
                # Using APOC for dynamic relationship types based on the predicate
                query = """
                MERGE (s:Entity {name: $sub})
                MERGE (o:Entity {name: $obj})
                WITH s, o
                CALL apoc.merge.relationship(s, $pred, {}, {}, o) YIELD rel
                RETURN rel
                """
                try:
                    session.run(query, sub=t['subject'], obj=t['object'], pred=t['predicate'].replace(" ", "_"))
                    print(f"  🔗 Saved: {t['subject']} -[{t['predicate']}]-> {t['object']}")
                except Exception as e:
                    print(f"  ❌ Neo4j Sync Error: {e}")

    def process_all_books(self, pdf_paths):
        for path in pdf_paths:
            content = self.pdf_to_unicode_text(path)
            # Split text into chunks to manage LLM context window
            chunks = [content[i:i+2500] for i in range(0, len(content), 2500)]
            
            for chunk in chunks:
                triples = self.extract_triples(chunk)
                if triples:
                    self.sync_to_neo4j(triples)
        print("✅ Finished populating the Knowledge Graph.")

# --- EXECUTION ---
if __name__ == "__main__":
    # Add your 4 PDF filenames here
    target_pdfs = ["gr10his.pdf"] 
    
    builder = SinhalaHistoryOntologyBuilder()
    builder.process_all_books(target_pdfs)
    builder.close()