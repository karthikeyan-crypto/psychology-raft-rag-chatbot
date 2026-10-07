from pathlib import Path

from inference.bot.local_model import answer_with_context
from inference.rag_local import LocalRetriever


INDEX_DIR = Path("data/vector_index")
TOP_K = 4


def main() -> None:
    if not INDEX_DIR.exists():
        raise RuntimeError(
            "FAISS index not found. Run inference/rag_local.py first."
        )

    retriever = LocalRetriever(INDEX_DIR)

    print("Psychology RAFT-RAG local demo")
    print("Type 'exit' to stop.")
    print()

    while True:
        question = input("Student: ").strip()

        if question.lower() in {"exit", "quit"}:
            break

        if not question:
            continue

        matches = retriever.search(question, top_k=TOP_K)

        answer = answer_with_context(
            question=question,
            contexts=matches,
        )

        print()
        print("Assistant:", answer)
        print()

        if matches:
            print("Retrieved sources:")
            for i, match in enumerate(matches, start=1):
                print(
                    f"{i}. score={match['score']:.3f} | {match['source']}"
                )
        else:
            print("Retrieved sources: none above the confidence threshold.")
        print()


if __name__ == "__main__":
    main()
