import json
import re
from pathlib import Path

from rdflib import Graph, Literal, Namespace, URIRef
from rdflib.namespace import RDF, RDFS

def clean_uri(name: str) -> str:
    """Creates a valid URI component keeping ALL Sinhala characters and vowel modifiers."""
    # 1. Replace spaces with underscores
    clean = re.sub(r'\s+', '_', name.strip())
    # 2. Remove ONLY specific English punctuation/symbols that break URIs
    clean = re.sub(r'[!@#$%^&*()+=\[\]{};:\'",.<>/?\\|~`]', '', clean)
    return clean

def update_ontology(
    json_path: str = "vector_store_v2/history_meta_advanced.json", 
    ttl_path: str = "ontology_colonial.ttl"
):
    print(f"Loading existing ontology from {ttl_path}...")
    g = Graph()
    
    if Path(ttl_path).exists():
        g.parse(ttl_path, format="ttl")
    
    EX = Namespace("http://example.org/colonial#")
    g.bind("ex", EX)

    print(f"Loading extracted entities from {json_path}...")
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    new_entities_count = 0

    for item in data.get("metadata", []):
        entities = item.get("entities", [])
        for ent in entities:
            name = ent.get("name")
            ent_type = ent.get("type", "")
            power = ent.get("power", "")

            if not name:
                continue

            entity_uri = EX[clean_uri(name)]

            if (entity_uri, None, None) not in g:
                new_entities_count += 1

            g.add((entity_uri, RDFS.label, Literal(name, lang="si")))

            ent_type_lower = ent_type.lower()
            if "person" in ent_type_lower or "governor" in ent_type_lower or "ruler" in ent_type_lower:
                g.add((entity_uri, RDF.type, EX.Ruler))
            elif "event" in ent_type_lower:
                g.add((entity_uri, RDF.type, EX.HistoricalEvent))
            elif "reform" in ent_type_lower:
                g.add((entity_uri, RDF.type, EX.Reform))
            else:
                g.add((entity_uri, RDF.type, EX.Entity)) 

            if "portuguese" in power.lower():
                g.add((entity_uri, EX.associated_with, EX.Portuguese))
            elif "dutch" in power.lower():
                g.add((entity_uri, EX.associated_with, EX.Dutch))
            elif "british" in power.lower():
                g.add((entity_uri, EX.associated_with, EX.British))

    print(f"Discovered and injected {new_entities_count} new unique entities/relationships.")
    g.serialize(destination=ttl_path, format="ttl")
    print(f"✅ Ontology successfully updated and saved to {ttl_path}!")

if __name__ == "__main__":
    update_ontology(
        json_path="vector_store_v2/history_meta_advanced.json", 
        ttl_path="ontology_colonial.ttl"
    )