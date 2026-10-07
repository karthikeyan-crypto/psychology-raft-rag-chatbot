# Psychology RAFT-RAG Chatbot

A research and educational chatbot that combines **Retrieval-Augmented Generation (RAG)** with **Retrieval-Augmented Fine-Tuning (RAFT)** to provide grounded psychology psychoeducation and emotional-support responses.

> **Important:** This project is for education, research, and demonstration. It is not a therapist, psychologist, doctor, diagnostic system, or emergency service. It must not be used to diagnose or treat a mental-health condition.

## Project goal

The project adapts the original RAFT mental-health chatbot implementation into a student-friendly psychology RAG pipeline:

```
User question
     ↓
Safety check
     ↓
RAG retrieval
     ↓
Relevant psychology knowledge
     ↓
RAFT/QLoRA fine-tuned language model
     ↓
Grounded response + source context
```

The final system is intended to:

- answer psychology and mental-health education questions using retrieved sources;
- prefer evidence from the configured knowledge base instead of unsupported claims;
- learn to use relevant context and ignore distracting context through RAFT;
- provide a simple chat interface;
- expose retrieved context for evaluation and demonstration;
- include a safety layer for high-risk conversations.

## Planned technology stack

| Component | Planned choice |
|---|---|
| GPU environment | Google Colab T4 |
| Base language model | Qwen/Qwen2.5-1.5B-Instruct |
| Fine-tuning | QLoRA + PEFT |
| RAFT question/answer generation | OpenAI API |
| Embeddings | BAAI/bge-small-en-v1.5 |
| Vector database | Pinecone Serverless |
| Backend | FastAPI |
| Public development endpoint | ngrok |
| Frontend | Streamlit |

The model and dependency versions will be pinned as the implementation is migrated so that the Colab workflow remains reproducible.

## Repository structure

```
finetune/        QLoRA/RAFT fine-tuning notebooks
inference/       RAG and chatbot notebooks/application
raft/            RAFT dataset-generation and formatting code
eval/            Evaluation resources
docs/            Project documentation
proposal/        Original project proposal/materials
```

## Development roadmap

1. Prepare the Google Colab T4 environment.
2. Prepare trusted psychology source documents.
3. Generate psychology RAFT training data.
4. Fine-tune Qwen2.5-1.5B-Instruct with QLoRA.
5. Build the BGE-small + Pinecone RAG pipeline.
6. Add the safety layer.
7. Connect FastAPI and Streamlit.
8. Evaluate retrieval, groundedness, relevance, and safety.
9. Compare the base model, fine-tuned model, and RAFT + RAG system.

## Safety and scope

The chatbot should:

- clearly distinguish psychoeducation from professional medical advice;
- avoid diagnosing users;
- avoid claiming to be a human professional;
- avoid inventing citations or clinical facts;
- encourage appropriate professional support when a situation requires it;
- provide an appropriate crisis response when a user indicates immediate danger or self-harm risk.

## API keys

**Never commit API keys, Pinecone credentials, Hugging Face tokens, or other secrets to this repository.**

Use environment variables or Colab secrets instead. A local configuration example can be kept in `.env.example`, while the real `.env` file remains ignored by Git.

## Attribution

This repository is a fork and planned adaptation of:

- `Irine-Juliet/RAFT-mental-health-chatbot`

The RAFT implementation and related project structure are being adapted rather than presented as entirely original code.

Before redistributing substantial portions of upstream code or data, check the upstream repository's licensing and attribution requirements.

## Status

**Work in progress.** The repository is being migrated from the original implementation toward the psychology-focused Qwen + QLoRA + RAFT + RAG architecture described above.
