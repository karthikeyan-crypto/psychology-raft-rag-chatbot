# Psychology RAFT dataset generator

This folder contains the dataset-generation code used to prepare psychology source material for the QLoRA fine-tuning stage.

## What this generator does

1. Reads trusted psychology documents (.pdf, .txt, or .json).
2. Splits them into overlapping chunks.
3. Uses an OpenAI model to generate grounded questions from each chunk.
4. Creates RAFT-style contexts containing the relevant source plus distractor chunks.
5. Generates a concise answer from the relevant source only.
6. Exports chat-style JSONL records that can be consumed by the QLoRA notebook.

The generator does not store API keys in the repository. Set OPENAI_API_KEY through Colab Secrets or an environment variable.

## Small end-to-end test

    python raft.py --datapath ./data/psychology_sources --output ./data/test.jsonl --max-chunks 2 --questions 2

## Full dataset generation

    python raft.py --datapath ./data/psychology_sources --output ./data/psychology_raft.jsonl --questions 3 --distractors 3

## Output fields

Each JSONL record contains:

- question
- answer
- context
- oracle_context
- oracle_included
- instruction
- messages

The messages field is already in chat format for the QLoRA notebook.

## Important

Use only trusted, legally reusable psychology/mental-health educational sources. The generated dataset is for psychoeducation research and should not be treated as a clinical knowledge base without human review.
