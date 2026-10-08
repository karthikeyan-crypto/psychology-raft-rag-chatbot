from __future__ import annotations

import os
from pathlib import Path

import streamlit as st
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from inference.rag_local import (
    DEFAULT_MIN_RETRIEVAL_SCORE,
    LocalRetriever,
    build_index,
)


APP_DIR = Path(__file__).resolve().parent
MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
ADAPTER_PATH = Path(
    os.getenv(
        "LAX_ADAPTER_PATH",
        "/content/drive/MyDrive/psychology-qwen-qlora-v2",
    )
)
INDEX_DIR = APP_DIR / "data" / "vector_index"
SOURCE_DIR = APP_DIR / "data" / "sources"

TOP_K = 3
MIN_RETRIEVAL_SCORE = DEFAULT_MIN_RETRIEVAL_SCORE

# Lightweight normalization for common short-chat inputs/typos.
COMMON_CORRECTIONS = {
    "lonly": "lonely",
    "lonley": "lonely",
    "stresed": "stressed",
    "stressd": "stressed",
}

SYSTEM_PROMPT = """You are LAX, a student wellbeing psychoeducation assistant.

STRICT GROUNDING RULES:
1. Use ONLY the retrieved knowledge as factual authority.
2. Every factual claim must be supported by the retrieved knowledge.
3. Do not add facts from pretrained knowledge.
4. Do not invent numbers, durations, examples, causes, mechanisms, or recommendations.
5. Do not diagnose mental-health conditions.
6. Do not claim to be a psychologist, therapist, psychiatrist, or doctor.
7. Do not recommend medication or personalised treatment.
8. If the retrieved knowledge is insufficient, say:
   "I don't have enough information in the current knowledge base to answer that reliably."

STYLE:
- Be concise and student-friendly.
- Give the most important supported points first.
- Prefer short bullet points when several actions are relevant.
- Avoid repetition.
- Usually stay under 120 words.
"""

HIGH_RISK_TERMS = (
    "suicide",
    "kill myself",
    "end my life",
    "want to die",
    "self harm",
    "self-harm",
)


def normalize_question(question: str) -> str:
    words = question.split()
    return " ".join(COMMON_CORRECTIONS.get(word.lower(), word) for word in words)


def conversation_response(question: str) -> str | None:
    normalized = " ".join(question.lower().split())

    greetings = {
        "hi",
        "hello",
        "hey",
        "hey lax",
        "hi lax",
        "hello lax",
    }

    if normalized in greetings:
        return (
            "Hi! I'm LAX. I can help with student wellbeing topics "
            "such as academic stress, sleep, routines, coping, and "
            "when to seek support. What would you like to talk about?"
        )

    casual = {
        "how are you",
        "how are you?",
        "what's up",
        "whats up",
    }

    if normalized in casual:
        return (
            "I'm here and ready to help. You can tell me what you're "
            "dealing with, such as stress, sleep, study pressure, or "
            "feeling lonely."
        )

    return None


def safety_response(question: str) -> str | None:
    lowered = question.lower()

    if any(term in lowered for term in HIGH_RISK_TERMS):
        return (
            "I'm sorry you're dealing with something this serious. "
            "I can't provide emergency or clinical care. If you may act "
            "on thoughts of suicide or self-harm, contact local emergency "
            "services, go to the nearest emergency department, or stay "
            "with a trusted person who can help you get immediate human "
            "support. In India, Tele-MANAS is available at 14416 or "
            "1800-89-14416."
        )

    return None


@st.cache_resource(show_spinner="Loading LAX AI model and knowledge base...")
def load_lax():
    if not ADAPTER_PATH.exists():
        raise FileNotFoundError(
            f"LoRA adapter not found at: {ADAPTER_PATH}"
        )

    adapter_file = ADAPTER_PATH / "adapter_model.safetensors"
    if not adapter_file.exists():
        raise FileNotFoundError(
            f"LoRA weights not found at: {adapter_file}"
        )

    if not INDEX_DIR.exists():
        build_index(SOURCE_DIR, INDEX_DIR)

    retriever = LocalRetriever(INDEX_DIR)

    compute_dtype = (
        torch.bfloat16
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        else torch.float16
    )

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
    )
    base_model.config.use_cache = False

    model = PeftModel.from_pretrained(
        base_model,
        str(ADAPTER_PATH),
        is_trainable=False,
    )
    model.eval()

    return tokenizer, model, retriever


def generate_lax_answer(question: str, tokenizer, model, retriever):
    question = normalize_question(question)

    casual = conversation_response(question)
    if casual:
        return casual, []

    emergency = safety_response(question)
    if emergency:
        return emergency, []

    matches = retriever.search(question, top_k=TOP_K)

    strong_matches = [
        item
        for item in matches
        if float(item["score"]) >= MIN_RETRIEVAL_SCORE
    ]

    if not strong_matches:
        return (
            "I don't have enough information in the current knowledge "
            "base to answer that reliably.",
            matches,
        )

    context = "\n\n".join(
        f'<SOURCE name="{item["source"]}">\n'
        f'{item["text"]}\n'
        f'</SOURCE>'
        for item in strong_matches
    )

    user_prompt = f"""Retrieved knowledge:

{context}

Student question:
{question}

Answer ONLY from the retrieved knowledge.
Do not add outside facts.
Do not invent unsupported details.
"""

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    model_inputs = tokenizer(
        prompt,
        return_tensors="pt",
        padding=False,
    )

    device = next(model.parameters()).device
    model_inputs = {
        key: value.to(device)
        for key, value in model_inputs.items()
    }

    with torch.no_grad():
        outputs = model.generate(
            input_ids=model_inputs["input_ids"],
            attention_mask=model_inputs["attention_mask"],
            max_new_tokens=180,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    generated = outputs[
        0,
        model_inputs["input_ids"].shape[-1]:
    ]

    answer = tokenizer.decode(
        generated,
        skip_special_tokens=True,
    ).strip()

    if not answer:
        answer = (
            "I don't have enough information in the current knowledge "
            "base to answer that reliably."
        )

    return answer, strong_matches


st.set_page_config(
    page_title="LAX — Student Wellbeing Companion",
    page_icon="🧠",
    layout="centered",
)

if "messages" not in st.session_state:
    st.session_state.messages = []

st.title("LAX")
st.caption("Your student wellbeing companion")

st.markdown(
    "Ask LAX about academic stress, sleep, routines, coping, "
    "social connection, or when to seek help."
)

with st.expander("About LAX"):
    st.write(
        "LAX provides student wellbeing psychoeducation from the "
        "project's curated knowledge base. It is not a psychologist, "
        "therapist, doctor, or diagnostic tool."
    )

try:
    tokenizer, model, retriever = load_lax()
except Exception as exc:
    st.error(f"LAX could not start: {exc}")
    st.stop()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

st.markdown("### Talk to LAX")

question = st.text_input(
    "Your question",
    placeholder="Example: How can I manage exam stress?",
    key="question_input",
)

send = st.button("Send", type="primary")

if send and question.strip():
    question = question.strip()

    st.session_state.messages.append(
        {"role": "user", "content": question}
    )

    with st.chat_message("user"):
        st.markdown(question)

    with st.spinner("LAX is thinking..."):
        answer, matches = generate_lax_answer(
            question,
            tokenizer,
            model,
            retriever,
        )

    st.session_state.messages.append(
        {"role": "assistant", "content": answer}
    )

    with st.chat_message("assistant"):
        st.markdown(answer)

        if matches:
            with st.expander("Sources used"):
                for item in matches:
                    source_name = Path(item["source"]).name
                    st.write(
                        f"{source_name} — similarity {item['score']:.3f}"
                    )

st.divider()
st.caption(
    "LAX is a student wellbeing and psychoeducation assistant. "
    "For urgent safety concerns, seek immediate human help."
)

if st.button("Clear conversation"):
    st.session_state.messages = []
    st.rerun()
