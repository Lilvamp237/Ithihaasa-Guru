import json
from pathlib import Path

meta_path = Path("vector_store/history_meta.json")
data = json.loads(meta_path.read_text(encoding="utf-8"))

# Overwrite IDs with a global counter
for i, entry in enumerate(data["metadata"]):
    entry["chunk_id"] = i

meta_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"Patched {len(data['metadata'])} entries with unique IDs.")