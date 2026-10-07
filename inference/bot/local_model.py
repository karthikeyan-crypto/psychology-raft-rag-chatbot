import os
from typing import Iterable

import requests


OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434").rstrip("/")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:1.5b-instruct")
OLLAMA_TIMEOUT = int(os.getenv("OLLAMA_TIMEOUT", "120"))

SYSTEM_PROMPT = """You are a student psychology psychoeducation assistant.

Your job is to explain information using ONLY the retrieved knowledge context.

Rules:
1. Do not diagnose the student.
2. Do not claim to be a psychologist, psychiatrist, therapist, or doctor.
3. Do not prescribe medication or a personalised treatment plan.
4. Do not invent facts, studies, statistics, or sources.
5. If the retrieved context does not support the answer, clearly say:
   "I don't have enough information in the current knowledge base to answer that reliably."
6. Keep normal answers clear, calm, practical, and student-friendly.
7. For immediate self-harm or suicide risk, prioritise immediate human help and
   crisis support instead of normal educational generation.
"""


def _ollama_chat(messages: list[dict]) -> str:
    response = requests.post(
        f"{OLLAMA_HOST}/api/chat",
        json={
            "model": OLLAMA_MODEL,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": 0.2,
                "top_p": 0.9,
            },
        },
        timeout=OLLAMA_TIMEOUT,
    )
    response.raise_for_status()
    data = response.json()
    answer = data.get("message", {}).get("content", "").strip()

    if not answer:
        raise RuntimeError("Ollama returned an empty response.")

    return answer


def _contains_high_risk_signal(text: str) -> bool:
    lowered = text.lower()
    signals = (
        "suicide",
        "kill myself",
        "end my life",
        "want to die",
        "self harm",
        "self-harm",
    )
    return any(signal in lowered for signal in signals)


def crisis_response() -> str:
    return (
        "I’m sorry you’re dealing with something this serious. I can’t provide "
        "emergency or clinical care. If you may act on thoughts of suicide or "
        "self-harm, contact local emergency services, go to the nearest emergency "
        "department, or stay with a trusted person who can help you get immediate "
        "human support. In India, Tele-MANAS provides 24x7 tele-mental-health "
        "support at 14416 or 1800-89-14416."
    )


def answer_with_context(
    question: str,
    contexts: Iterable[dict],
    conversation_history: Iterable[dict] | None = None,
) -> str:
    if _contains_high_risk_signal(question):
        return crisis_response()

    context_blocks = []
    for item in contexts:
        source = str(item.get("source", "unknown"))
        text = str(item.get("text", "")).strip()
        if text:
            context_blocks.append(
                f'<SOURCE name="{source}">\n{text}\n</SOURCE>'
            )

    if not context_blocks:
        return (
            "I don't have enough information in the current knowledge base "
            "to answer that reliably."
        )

    context = "\n\n".join(context_blocks)

    history = []
    for message in list(conversation_history or [])[-6:]:
        role = message.get("role")
        content = str(message.get("content", "")).strip()
        if role in {"user", "assistant"} and content:
            history.append({"role": role, "content": content})

    user_prompt = f"""Retrieved psychology knowledge:

{context}

Student question:
{question}

Answer only from the retrieved knowledge above. If it is insufficient, say so.
Do not add outside facts. Do not diagnose. Keep the answer student-friendly.
"""

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(history)
    messages.append({"role": "user", "content": user_prompt})

    return _ollama_chat(messages)
