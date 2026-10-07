# Local model for the live demo

## Recommended model

Use:

    qwen2.5:1.5b-instruct

through Ollama.

The upstream Hugging Face model is Qwen2.5-1.5B-Instruct and is licensed under Apache-2.0. It is small enough for a practical local demonstration.

## Why local inference

The live chatbot should not depend on a paid external LLM API.

With Ollama:

- inference runs on the demo machine;
- there is no OpenAI/Anthropic token quota for the live chat;
- the demo keeps working without Internet access after the model is downloaded;
- the application can make repeated judge queries without consuming a cloud API allowance.

This is **not literally unlimited tokens**. The model still has a finite context window and your machine still has finite CPU/RAM/GPU resources. "No API quota" is the accurate claim.

## Start the model

    ollama run qwen2.5:1.5b-instruct

Ollama exposes a local HTTP API at:

    http://localhost:11434

The project should call this local endpoint only for live inference.

## Important architecture rule

Do not send the student's private chat message to a cloud LLM during the live demo.

Use:

    User
      ↓
    Safety check
      ↓
    Local Qwen
      ↑
    Retrieved psychology context
      ↑
    Local/remote vector database
