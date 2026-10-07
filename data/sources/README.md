# Source documents

Put the downloaded and legally reusable source documents for the student-wellbeing knowledge base in this directory.

Preferred formats for the local retrieval index:

- .txt
- .md

The project intentionally does not commit copies of external source PDFs here unless their redistribution terms permit it.

After preparing the source files, build the local FAISS index from the repository root:

    python inference/rag_local.py --source-dir data/sources --index-dir data/vector_index
