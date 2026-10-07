import os

import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

API_URL = os.getenv("PSYCHOLOGY_API_URL", "http://127.0.0.1:8000").rstrip("/")
REQUEST_TIMEOUT = 90

st.set_page_config(
    page_title="Psychology RAFT-RAG Chatbot",
    page_icon="🧠",
    layout="centered",
)

with st.sidebar:
    st.title("🧠 Psychology RAFT-RAG")
    st.caption("Psychology psychoeducation and emotional-support research demo.")
    st.warning(
        "This chatbot is not a therapist, psychologist, doctor, "
        "diagnostic system, or emergency service."
    )

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": (
                "Hi! I can help with psychology and mental-health "
                "education using the project's knowledge base. "
                "How can I help?"
            ),
        }
    ]

def get_bot_response(question: str) -> str:
    try:
        response = requests.post(
            f"{API_URL}/generate",
            json={
                "question": question,
                "conversation_history": st.session_state.messages,
            },
            timeout=REQUEST_TIMEOUT,
        )
        response.raise_for_status()
        data = response.json()

        if "response" not in data:
            return "The backend returned an unexpected response format."

        return str(data["response"])

    except requests.exceptions.Timeout:
        return "The backend took too long to respond. Please try again."
    except requests.exceptions.ConnectionError:
        return (
            "I could not connect to the chatbot backend. "
            "Please start the FastAPI server and check PSYCHOLOGY_API_URL."
        )
    except requests.exceptions.RequestException as exc:
        return f"The chatbot backend returned an error: {exc}"

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

if prompt := st.chat_input("Ask a psychology education question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving knowledge and generating a response..."):
            response = get_bot_response(prompt)
            st.write(response)

    st.session_state.messages.append(
        {"role": "assistant", "content": response}
    )
