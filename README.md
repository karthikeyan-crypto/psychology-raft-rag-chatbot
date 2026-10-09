# Psychology RAFT-RAG Chatbot

A research and educational chatbot that combines Retrieval-Augmented Generation (RAG) with Retrieval-Augmented Fine-Tuning (RAFT) for student psychology psychoeducation and emotional support.

> Important: This project is for education, research, and demonstration. It is not a therapist, psychologist, doctor, diagnostic system, or emergency service. It must not be used to diagnose or treat a mental-health condition.

## Project goal

The system is designed around common student well-being needs:
- academic and exam stress;
- anxiety and excessive worry;
- sleep and healthy routines;
- coping and emotional regulation;
- concentration and workload overload;
- social connection and loneliness;
- knowing when to seek professional support.

The core rule is simple:

> The chatbot should answer from retrieved knowledge. When the knowledge base does not support the answer, it should say so instead of guessing.

## Local-first architecture

Student question -> safety check -> local BGE-small embeddings -> local FAISS search -> expanded retry if needed -> topic-matched trusted source fallback if available -> local Qwen2.5-1.5B-Instruct + LoRA -> grounded answer or explicit abstention.

If retrieval misses a topic that is already documented, LAX retries with related search terms and can pass the matching existing source file directly to the model. If no relevant source exists, LAX must say that the knowledge base does not contain enough information; it must not use the model's pretrained knowledge to invent a mental-health answer. This improves retrieval resilience but does not guarantee that every generated claim is correct, so the regression checks in `eval/RAG_GROUNDING_TESTS.md` should be run after changes.

## Technology stack

| Component | Choice |
|---|---|
| Fine-tuning environment | Google Colab T4 |
| Base language model | Qwen/Qwen2.5-1.5B-Instruct |
| Streamlit app model | Qwen2.5-1.5B-Instruct + saved LoRA adapter through Transformers/PEFT |
| Optional CLI demo model | qwen2.5:1.5b-instruct through Ollama |
| Fine-tuning | QLoRA + PEFT |
| Embeddings | BAAI/bge-small-en-v1.5 |
| Local vector database | FAISS |
| Backend | FastAPI |
| Frontend | Streamlit |

Qwen2.5-1.5B-Instruct is published under the Apache-2.0 license. The model has a finite context window, so the accurate claim is no external API quota, not literally unlimited tokens.

FAISS is a local similarity-search library for dense vectors and supports cosine-style search through normalized vectors and inner products.

## Knowledge sources

The project uses authoritative sources such as WHO, NIMH, and Government of India mental-health resources. See data/source_manifest.json.

Do not add random internet advice, Reddit posts, or unverified mental-health claims to the corpus.

## Local demo

1. Install Ollama.
2. Download the model with: ollama pull qwen2.5:1.5b-instruct
3. Put trusted source files in data/sources/.
4. Build the FAISS index:
   python inference/rag_local.py --source-dir data/sources --index-dir data/vector_index
5. Start the local demo:
   python inference/local_demo.py

The Streamlit app loads the saved QLoRA adapter with Transformers/PEFT and runs generation in the selected local/Colab runtime. The optional command-line demo uses Ollama. The live chat does not send the student's question to a paid cloud LLM API.

## Safety and scope

The chatbot should distinguish psychoeducation from professional care, avoid diagnosis, avoid medication prescribing, avoid invented facts, and encourage human support when appropriate.

For India, Government of India Tele-MANAS provides 24x7 tele-mental-health support through 14416 or 1800-89-14416.

## Attribution

This repository is a fork and planned adaptation of Irine-Juliet/RAFT-mental-health-chatbot.
Before redistributing substantial portions of upstream code or data, check the upstream repository's licensing and attribution requirements.

## Status

Work in progress. The project is being migrated toward a local Qwen + QLoRA + RAFT + BGE + FAISS architecture focused on student well-being.

## Run LAX in CPU-only Google Colab

If your Colab GPU quota is exhausted, open the CPU-only launcher notebook:

**[Open LAX CPU-only Colab notebook](https://colab.research.google.com/github/karthikeyan-crypto/psychology-raft-rag-chatbot/blob/main/cloud/01_run_lax_app_colab_cpu.ipynb)**

It clones this repository, installs `requirements-colab-cpu.txt` without replacing Colab's PyTorch build, checks/mounts the saved `psychology-qwen-qlora-v2` adapter from Drive (with ZIP-upload fallback), builds the local FAISS index if needed, starts Streamlit, displays logs, and opens a temporary Cloudflare Tunnel URL.

CPU mode loads the fine-tuned Qwen2.5 model in full precision and disables the separate Qwen3 general-chat model by default to reduce RAM use. This can be slow and requires several GB of RAM. It does not retrain or overwrite the adapter. Run the notebook cells from top to bottom and keep the tunnel cell running while using the temporary URL.

