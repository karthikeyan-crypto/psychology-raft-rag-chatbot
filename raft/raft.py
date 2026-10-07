from __future__ import annotations

import argparse
import json
import logging
import os
import random
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import PyPDF2
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import OpenAI
from tqdm import tqdm

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("psychology-raft")

SYSTEM_PROMPT = """You are a psychology psychoeducation dataset generator.
Use only the supplied source context.
Do not diagnose, prescribe treatment, or claim to replace a licensed professional.
Prefer cautious, educational wording and explicitly acknowledge when the context is insufficient.
"""


@dataclass(frozen=True)
class Chunk:
    text: str
    source: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a psychology RAFT dataset from trusted source documents."
    )
    parser.add_argument("--datapath", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("./data/psychology_raft.jsonl"),
    )
    parser.add_argument("--questions", type=int, default=3)
    parser.add_argument("--chunk-size", type=int, default=2500)
    parser.add_argument("--chunk-overlap", type=int, default=250)
    parser.add_argument("--distractors", type=int, default=3)
    parser.add_argument(
        "--oracle-probability",
        type=float,
        default=1.0,
        help="Probability that the relevant chunk is included in the training context.",
    )
    parser.add_argument(
        "--model",
        default="gpt-4.1-mini",
        help="OpenAI model used only to generate synthetic RAFT questions and answers.",
    )
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument(
        "--max-chunks",
        type=int,
        default=None,
        help="Useful for a small end-to-end test before generating the full dataset.",
    )
    return parser.parse_args()


def iter_source_files(path: Path) -> Iterable[Path]:
    if path.is_file():
        yield path
        return

    extensions = {".pdf", ".txt", ".json"}
    for file_path in sorted(path.rglob("*")):
        if file_path.is_file() and file_path.suffix.lower() in extensions:
            yield file_path


def read_document(path: Path) -> str:
    suffix = path.suffix.lower()

    if suffix == ".pdf":
        parts: list[str] = []
        with path.open("rb") as handle:
            reader = PyPDF2.PdfReader(handle)
            for page in reader.pages:
                parts.append(page.extract_text() or "")
        return "\n".join(parts)

    if suffix == ".json":
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        if isinstance(data, dict) and isinstance(data.get("text"), str):
            return data["text"]
        return json.dumps(data, ensure_ascii=False)

    if suffix == ".txt":
        return path.read_text(encoding="utf-8", errors="ignore")

    raise ValueError(f"Unsupported document type: {path.suffix}")


def build_chunks(
    datapath: Path,
    chunk_size: int,
    chunk_overlap: int,
) -> list[Chunk]:
    if chunk_overlap >= chunk_size:
        raise ValueError("--chunk-overlap must be smaller than --chunk-size")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks: list[Chunk] = []
    for file_path in iter_source_files(datapath):
        text = read_document(file_path).strip()
        if not text:
            logger.warning("Skipping empty document: %s", file_path)
            continue

        for piece in splitter.split_text(text):
            piece = piece.strip()
            if piece:
                chunks.append(Chunk(text=piece, source=str(file_path)))

    logger.info("Created %d chunks from %s", len(chunks), datapath)
    return chunks


def call_openai(
    client: OpenAI,
    model: str,
    instructions: str,
    user_input: str,
    *,
    max_output_tokens: int = 300,
    temperature: float = 0.2,
    retries: int = 3,
) -> str:
    last_error: Exception | None = None

    for attempt in range(retries):
        try:
            response = client.responses.create(
                model=model,
                instructions=instructions,
                input=user_input,
                max_output_tokens=max_output_tokens,
                temperature=temperature,
            )
            text = response.output_text.strip()
            if text:
                return text
            raise RuntimeError("OpenAI returned an empty output.")
        except Exception as exc:
            last_error = exc
            if attempt == retries - 1:
                break
            sleep_seconds = 2 ** attempt
            logger.warning(
                "OpenAI request failed (%s). Retrying in %ss...",
                exc,
                sleep_seconds,
            )
            time.sleep(sleep_seconds)

    raise RuntimeError(f"OpenAI request failed after {retries} attempts.") from last_error


def generate_questions(
    client: OpenAI,
    model: str,
    chunk: Chunk,
    count: int,
) -> list[str]:
    prompt = f"""Generate exactly {count} short user questions that can be answered
using only the psychology source excerpt below.

Rules:
- One question per line.
- Do not number the questions.
- Do not ask for diagnosis or personalized medical treatment.
- Prefer educational questions about definitions, concepts, common signs,
  evidence-based general information, coping/management principles, or prevention.
- Do not use knowledge that is not present in the excerpt.

SOURCE ({chunk.source}):
{chunk.text}
"""

    raw = call_openai(
        client,
        model,
        "You create grounded psychology psychoeducation questions from source material.",
        prompt,
        max_output_tokens=max(128, count * 80),
        temperature=0.4,
    )

    questions: list[str] = []
    for line in raw.splitlines():
        question = line.strip()
        question = question.removeprefix("- ").removeprefix("* ")
        if question and question[-1] != "?":
            question += "?"
        if question and question not in questions:
            questions.append(question)

    return questions[:count]


def build_training_example(
    client: OpenAI,
    model: str,
    chunks: list[Chunk],
    oracle_index: int,
    question: str,
    distractor_count: int,
    oracle_probability: float,
) -> dict:
    oracle = chunks[oracle_index]
    candidate_indices = [i for i in range(len(chunks)) if i != oracle_index]
    selected_distractors = random.sample(
        candidate_indices,
        min(distractor_count, len(candidate_indices)),
    )

    context_chunks = [chunks[i] for i in selected_distractors]
    include_oracle = random.random() < oracle_probability
    if include_oracle:
        context_chunks.append(oracle)

    random.shuffle(context_chunks)

    if include_oracle:
        answer_prompt = f"""Answer the question using only the source excerpt.
Give a concise, educational answer in 2-5 sentences.
Do not diagnose the user or give individualized medical treatment.
Do not mention the dataset or the generation process.

SOURCE:
{oracle.text}

QUESTION:
{question}
"""
        answer = call_openai(
            client,
            model,
            SYSTEM_PROMPT,
            answer_prompt,
            max_output_tokens=220,
            temperature=0.2,
        )
    else:
        answer = (
            "The provided context does not contain enough information to answer "
            "this question reliably."
        )

    context_text = "\n\n".join(
        f"<DOCUMENT source=\"{item.source}\">\n{item.text}\n</DOCUMENT>"
        for item in context_chunks
    )

    user_prompt = (
        "Use the retrieved context below to answer the psychology "
        "psychoeducation question.\n\n"
        f"{context_text}\n\n"
        f"QUESTION: {question}"
    )

    return {
        "id": str(uuid.uuid4()),
        "question": question,
        "answer": answer,
        "oracle_included": include_oracle,
        "oracle_source": oracle.source,
        "context": [
            {"source": item.source, "text": item.text}
            for item in context_chunks
        ],
        "oracle_context": {"source": oracle.source, "text": oracle.text},
        "instruction": user_prompt,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": answer},
        ],
    }


def process_chunk(
    client: OpenAI,
    model: str,
    chunks: list[Chunk],
    chunk_index: int,
    questions_per_chunk: int,
    distractors: int,
    oracle_probability: float,
) -> list[dict]:
    chunk = chunks[chunk_index]
    questions = generate_questions(client, model, chunk, questions_per_chunk)

    examples: list[dict] = []
    for question in questions:
        examples.append(
            build_training_example(
                client=client,
                model=model,
                chunks=chunks,
                oracle_index=chunk_index,
                question=question,
                distractor_count=distractors,
                oracle_probability=oracle_probability,
            )
        )
    return examples


def main() -> None:
    load_dotenv()
    args = parse_args()

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Store it in Colab Secrets or the environment; "
            "never put the key directly in this file."
        )

    if not 0.0 <= args.oracle_probability <= 1.0:
        raise ValueError("--oracle-probability must be between 0 and 1.")

    if args.questions < 1:
        raise ValueError("--questions must be at least 1.")

    if args.distractors < 0:
        raise ValueError("--distractors cannot be negative.")

    chunks = build_chunks(
        args.datapath,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )
    if args.max_chunks is not None:
        chunks = chunks[: args.max_chunks]

    if not chunks:
        raise RuntimeError("No usable document chunks were found.")

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    args.output.parent.mkdir(parents=True, exist_ok=True)

    total_examples = 0
    with args.output.open("w", encoding="utf-8") as handle:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = [
                executor.submit(
                    process_chunk,
                    client,
                    args.model,
                    chunks,
                    index,
                    args.questions,
                    args.distractors,
                    args.oracle_probability,
                )
                for index in range(len(chunks))
            ]

            for future in tqdm(
                as_completed(futures),
                total=len(futures),
                desc="Generating RAFT examples",
                unit="chunk",
            ):
                examples = future.result()
                for example in examples:
                    handle.write(json.dumps(example, ensure_ascii=False) + "\n")
                    total_examples += 1
                handle.flush()

    logger.info(
        "Finished. Wrote %d examples to %s",
        total_examples,
        args.output,
    )


if __name__ == "__main__":
    main()
