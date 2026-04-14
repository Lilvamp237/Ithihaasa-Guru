from pathlib import Path

from rdflib import Graph


def validate_ontology(ttl_path: str = "ontology_colonial.ttl") -> None:
    g = Graph()
    g.parse(ttl_path, format="ttl")
    print(f"Ontology parsed successfully. Triple count: {len(g)}")


if __name__ == "__main__":
    path = Path("ontology_colonial.ttl")
    if not path.exists():
        raise FileNotFoundError(path)
    validate_ontology(str(path))
