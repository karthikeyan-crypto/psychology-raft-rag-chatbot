# Student Well-Being Knowledge Base

This directory defines the source material for the psychology RAG system.

## Goal

The chatbot is designed for **student psychoeducation and emotional support**, not diagnosis or treatment.

The knowledge base focuses on:

1. Academic and exam stress
2. Anxiety and excessive worry
3. Sleep and sleep routines
4. Healthy daily habits and self-care
5. Relaxation, breathing, and coping strategies
6. Concentration, overload, and prioritisation
7. Social connection and support
8. Recognising when symptoms interfere with daily life
9. Seeking professional support
10. Crisis and immediate-safety responses

## Source policy

Use authoritative, current, and legally reusable sources whenever possible.

Preferred sources for this project are:

- World Health Organization (WHO)
- Government of India / Ministry of Health and Family Welfare
- National Institute of Mental Health (NIMH)

Do not add Reddit posts, anonymous blogs, scraped social-media advice, or unverified "mental-health tips" to the training corpus.

## Grounding rule

At runtime, the assistant should answer from the retrieved knowledge context.

When the retrieved context does not support an answer, the assistant should clearly say that the available knowledge base does not contain enough information instead of guessing.

## Local files

Place downloaded source PDFs/text files under:

    data/sources/

Do not commit source documents unless their licence/redistribution terms allow it.

The repository keeps a source manifest so the project remains reproducible without copying copyrighted text into Git.
