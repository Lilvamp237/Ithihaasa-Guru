import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List

import faiss
import numpy as np
import requests
from sentence_transformers import SentenceTransformer


DEFAULT_EMBED_MODEL = "BAAI/bge-m3"


def normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-12
    return vectors / norms


def load_payload(path: Path) -> Dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if "metadata" not in payload:
        raise ValueError("Input JSON must contain a 'metadata' field.")
    return payload


def load_checkpoint(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {"cleaned_texts": {}, "last_index": -1}
    data = json.loads(path.read_text(encoding="utf-8"))
    cleaned = {int(k): v for k, v in data.get("cleaned_texts", {}).items()}
    data["cleaned_texts"] = cleaned
    return data


def save_checkpoint(path: Path, data: Dict[str, Any]) -> None:
    serializable = {**data, "cleaned_texts": {str(k): v for k, v in data["cleaned_texts"].items()}}
    path.write_text(json.dumps(serializable, ensure_ascii=False, indent=2), encoding="utf-8")


def ollama_clean_text(
    text: str,
    model: str,
    host: str,
    timeout: int,
    max_retries: int,
) -> str:
    prompt = (
        "You are a Sinhala language expert. Fix the following OCR-corrupted text by correcting "
        "spelling errors, joining broken words, and ensuring historical accuracy. Return only the "
        "cleaned Sinhala text.\n\n"
        f"{text}"
    )
    url = f"{host}/api/chat"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "options": {"temperature": 0.1},
    }
    last_error: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            response = requests.post(url, json=payload, timeout=timeout)
            response.raise_for_status()
            data = response.json()
            content = data["message"]["content"]
            return content.strip()
        except Exception as exc:  # noqa: BLE001 - surface failure after retries
            last_error = exc
            sleep_for = 2**attempt
            print(f"[WARN] Ollama request failed (attempt {attempt}/{max_retries}): {exc}")
            time.sleep(sleep_for)
    raise RuntimeError("Ollama request failed after retries.") from last_error


def build_index(texts: List[str], embed_model_name: str, out_dir: Path) -> None:
    print(f"[INFO] Loading embedding model: {embed_model_name}")
    model = SentenceTransformer(embed_model_name)
    print(f"[INFO] Encoding {len(texts)} chunks...")
    vectors = model.encode(texts, convert_to_numpy=True, show_progress_bar=True).astype("float32")
    vectors = normalize(vectors)
    dim = vectors.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(vectors)

    out_dir.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(out_dir / "history.index"))
    print(f"[INFO] Saved FAISS index to {out_dir / 'history.index'}")
    return dim


def main() -> None:
    parser = argparse.ArgumentParser(description="Refine OCR text with Ollama and rebuild FAISS index.")
    parser.add_argument(
        "--input_json",
        default="vector_store/history_meta.json",
        help="Path to OCR-extracted JSON metadata (default: vector_store/history_meta.json)",
    )
    parser.add_argument(
        "--out_dir",
        default="vector_store",
        help="Output directory for FAISS index and history_meta.json (default: vector_store)",
    )
    parser.add_argument(
        "--checkpoint",
        default="vector_store/refine_ocr_checkpoint.json",
        help="Checkpoint file for resumable processing (default: vector_store/refine_ocr_checkpoint.json)",
    )
    parser.add_argument("--model", default="gemma3:12b", help="Ollama model name")
    parser.add_argument("--host", default="http://localhost:11434", help="Ollama host URL")
    parser.add_argument("--timeout", type=int, default=180, help="Ollama request timeout in seconds")
    parser.add_argument("--max_retries", type=int, default=3, help="Max retries per chunk")
    parser.add_argument("--checkpoint_every", type=int, default=1, help="Save checkpoint every N chunks")
    args = parser.parse_args()

    input_path = Path(args.input_json)
    out_dir = Path(args.out_dir)
    checkpoint_path = Path(args.checkpoint)

    payload = load_payload(input_path)
    metadata = payload["metadata"]
    embed_model_name = payload.get("embed_model_name") or DEFAULT_EMBED_MODEL

    checkpoint = load_checkpoint(checkpoint_path)
    cleaned_texts: Dict[int, str] = checkpoint.get("cleaned_texts", {})

    total = len(metadata)
    print(f"[INFO] Total chunks: {total}")
    print(f"[INFO] Resuming with {len(cleaned_texts)} chunks already cleaned.")

    for idx, item in enumerate(metadata):
        if idx in cleaned_texts:
            continue
        text = item.get("text", "")
        if not text:
            cleaned_texts[idx] = ""
            continue
        print(f"[INFO] Cleaning chunk {idx + 1}/{total}...")
        cleaned = ollama_clean_text(
            text=text,
            model=args.model,
            host=args.host,
            timeout=args.timeout,
            max_retries=args.max_retries,
        )
        cleaned_texts[idx] = cleaned

        if (idx + 1) % max(1, args.checkpoint_every) == 0:
            checkpoint.update(
                {
                    "input_path": str(input_path),
                    "out_dir": str(out_dir),
                    "last_index": idx,
                    "total": total,
                    "cleaned_texts": cleaned_texts,
                    "completed": False,
                }
            )
            save_checkpoint(checkpoint_path, checkpoint)

    updated_metadata = []
    for idx, item in enumerate(metadata):
        updated_metadata.append({**item, "text": cleaned_texts.get(idx, item.get("text", ""))})

    dim = build_index([m["text"] for m in updated_metadata], embed_model_name, out_dir)
    out_payload = {
        "metadata": updated_metadata,
        "embed_model_name": embed_model_name,
        "dim": dim,
    }
    (out_dir / "history_meta.json").write_text(
        json.dumps(out_payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[INFO] Updated metadata written to {out_dir / 'history_meta.json'}")

    checkpoint.update(
        {
            "input_path": str(input_path),
            "out_dir": str(out_dir),
            "last_index": total - 1,
            "total": total,
            "cleaned_texts": cleaned_texts,
            "completed": True,
        }
    )
    save_checkpoint(checkpoint_path, checkpoint)
    print(f"[INFO] Checkpoint saved to {checkpoint_path}")


if __name__ == "__main__":
    main()
