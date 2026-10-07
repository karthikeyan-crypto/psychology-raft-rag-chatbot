from __future__ import annotations

import argparse
import json
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

EMBED_MODEL = "BAAI/bge-small-en-v1.5"
DIMENSION = 384


def load_documents(source_dir: Path) -> list[dict]:
    documents = []

    for path in sorted(source_dir.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in {".txt", ".md"}:
            continue

        text = path.read_text(encoding="utf-8", errors="ignore").strip()
        if not text:
            continue

        documents.append({"source": str(path), "text": text})

    if not documents:
        raise RuntimeError(f"No .txt or .md source files found in {source_dir}")

    return documents


def build_index(source_dir: Path, index_dir: Path) -> None:
    documents = load_documents(source_dir)

    embedder = SentenceTransformer(EMBED_MODEL)
    vectors = embedder.encode(
        [item["text"] for item in documents],
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype("float32")

    dimension = vectors.shape[1]
    if dimension != DIMENSION:
        raise RuntimeError(
            f"Expected {DIMENSION}-dimensional embeddings, got {dimension}."
        )

    # Cosine similarity becomes inner product when vectors are normalized.
    index = faiss.IndexFlatIP(dimension)
    index.add(vectors)

    index_dir.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(index_dir / "student_wellbeing.faiss"))

    with (index_dir / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(documents, handle, ensure_ascii=False, indent=2)

    print(f"Indexed {len(documents)} documents.")
    print(f"Index directory: {index_dir}")


class LocalRetriever:
    def __init__(
        self,
        index_dir: str | Path,
        model_name: str = EMBED_MODEL,
    ) -> None:
        index_path = Path(index_dir)

        self.index = faiss.read_index(
            str(index_path / "student_wellbeing.faiss")
        )

        with (index_path / "metadata.json").open(
            "r",
            encoding="utf-8",
        ) as handle:
            self.documents = json.load(handle)

        self.embedder = SentenceTransformer(model_name)

    def search(self, question: str, top_k: int = 4) -> list[dict]:
        vector = self.embedder.encode(
            [question],
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).astype("float32")

        scores, indices = self.index.search(vector, top_k)

        matches = []

        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self.documents):
                continue

            item = dict(self.documents[idx])
            item["score"] = float(score)
            matches.append(item)

        return matches


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the local student-wellbeing FAISS index."
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=Path("data/sources"),
    )
    parser.add_argument(
        "--index-dir",
        type=Path,
        default=Path("data/vector_index"),
    )

    args = parser.parse_args()
    build_index(args.source_dir, args.index_dir)


if __name__ == "__main__":
    main()
