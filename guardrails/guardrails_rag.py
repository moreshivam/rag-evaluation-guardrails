"""
RAG FUSION WITH GUARDRAILS
───────────────────────────
Wraps the RAG Fusion pipeline with input and output validation.

Input guardrails (before any retrieval):
  1. Length check   — block empty or >1000 char queries
  2. Topic check    — LLM classifier blocks off-topic questions

Output guardrails (after generation):
  3. Faithfulness   — LLM-as-judge checks answer against the EXACT
                      RRF-ranked context that was passed to the LLM.
                      Flags hallucinations without blocking.

Why RAG Fusion + Guardrails together:
  - RAG Fusion gives better context (RRF-ranked, multi-query)
  - Faithfulness check uses that same context → more accurate judgement
  - No double-retrieval: context is retrieved once, used for both answer + check

Flow:
  question
    → [INPUT]  length check
    → [INPUT]  topic classifier (LLM call)
    → query_generator → 3 queries
    → reciprocal_rank_fusion → ranked docs
    → answer_chain → answer          ← same docs used here
    → [OUTPUT] faithfulness check    ← and here
    → result dict
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from rag_fusion_chain import (
    llm,
    query_generator,
    reciprocal_rank_fusion,
    format_docs,
    answer_chain,
)

load_dotenv(override=True)

MAX_QUESTION_LENGTH = 1000

TOPIC_SCOPE = (
    "AI, Machine Learning, Deep Learning, LLMs, Transformers, RAG, "
    "Prompt Engineering, LangChain, LangGraph, MCP, Embeddings, "
    "Vector databases, and related GenAI/MLOps topics"
)

# ── Guardrail 1: Topic check ──────────────────────────────────────────────────
_topic_prompt = ChatPromptTemplate.from_template(
    f"""You are a topic classifier. The knowledge base covers: {TOPIC_SCOPE}.
Is the following question related to these topics?
Respond with ONLY "YES" or "NO".

Question: {{question}}"""
)
_topic_checker = _topic_prompt | llm | StrOutputParser()

def check_input(question: str) -> tuple[bool, str]:
    if not question.strip():
        return False, "Please enter a non-empty question."
    if len(question) > MAX_QUESTION_LENGTH:
        return False, f"Question exceeds {MAX_QUESTION_LENGTH} character limit."
    verdict = _topic_checker.invoke({"question": question}).strip().upper()
    if verdict.startswith("NO"):
        return False, (
            "This question is outside the knowledge base scope. "
            f"Supported topics: {TOPIC_SCOPE}."
        )
    return True, ""

# ── Guardrail 2: Faithfulness check ──────────────────────────────────────────
_faithfulness_prompt = ChatPromptTemplate.from_template("""You are a fact-checker.
Determine if the answer below is fully supported by the provided context.
An "I don't know" answer is always considered SUPPORTED.
Respond with ONLY "SUPPORTED" or "UNSUPPORTED".

Context:
{context}

Answer:
{answer}""")

_faithfulness_checker = _faithfulness_prompt | llm | StrOutputParser()

def check_output(answer: str, context: str) -> tuple[bool, str]:
    verdict = _faithfulness_checker.invoke({"context": context, "answer": answer}).strip().upper()
    if verdict.startswith("UNSUPPORTED"):
        return False, "[WARNING] Answer may contain information not found in the source documents."
    return True, ""

# ── Guarded RAG Fusion ────────────────────────────────────────────────────────
def guarded_rag(question: str) -> dict:
    """
    Returns:
      status  : "ok" | "warning" | "blocked"
      answer  : generated answer, or None if blocked
      message : rejection/warning text, or None
      sources : list of {source, page} dicts
      queries : the 3 alternative queries generated (None if blocked)
    """
    # ── Input guardrails ──────────────────────────────────────────────────────
    valid, reason = check_input(question)
    if not valid:
        return {
            "status": "blocked",
            "answer": None,
            "message": reason,
            "sources": [],
            "queries": None,
        }

    # ── RAG Fusion: generate queries → RRF rank → build context ──────────────
    queries = [q for q in query_generator.invoke(question) if q.strip()]
    ranked_docs = reciprocal_rank_fusion(queries)
    top_docs = ranked_docs[:4]
    context = format_docs(top_docs)

    # ── Generate answer from the exact same context ───────────────────────────
    answer = answer_chain.invoke({"context": context, "question": question})

    # ── Output guardrail: faithfulness against the context LLM actually saw ───
    faithful, warning = check_output(answer, context)

    return {
        "status": "warning" if not faithful else "ok",
        "answer": answer,
        "message": warning if not faithful else None,
        "sources": [
            {
                "source": d.metadata.get("source", "unknown"),
                "page": d.metadata.get("page", "?"),
            }
            for d in top_docs
        ],
        "queries": queries,
    }

if __name__ == "__main__":
    print("RAG Fusion + Guardrails ready! Type 'quit' to exit.\n")
    while True:
        question = input("Your question: ").strip()
        if not question:
            continue
        if question.lower() == "quit":
            break

        print("\nChecking input guardrails...")
        result = guarded_rag(question)

        if result["status"] == "blocked":
            print(f"\nBLOCKED: {result['message']}")
        else:
            if result["queries"]:
                print("\nAlternative queries used:")
                for i, q in enumerate(result["queries"], 1):
                    print(f"  {i}. {q}")

            if result["message"]:
                print(f"\n{result['message']}")

            print("\n--- Answer ---")
            print(result["answer"])
            print("\nSources used:")
            for s in result["sources"]:
                print(f"  {s['source']} | Page {s['page']}")
        print("-" * 60 + "\n")
