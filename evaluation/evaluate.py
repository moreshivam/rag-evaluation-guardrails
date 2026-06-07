"""
RAG EVALUATION with RAGAS — backed by RAG Fusion
─────────────────────────────────────────────────
Evaluates the RAG Fusion pipeline using three RAGAS metrics:

  faithfulness      — Is the answer grounded in the RRF-ranked context? (no hallucination)
  answer_relevancy  — Does the answer actually address the question?
  context_precision — Are the top-ranked chunks relevant to the question?

How RAG Fusion improves evaluation accuracy vs basic RAG:
  - 3 alternative queries → broader chunk coverage
  - RRF ranking → the best chunks rise to the top
  - Evaluation sees the same context the LLM actually used (no re-retrieval)

Usage:
  cd evaluation
  python evaluate.py

Results printed to stdout + saved to evaluation_results.json.
"""

import json
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag_fusion_chain import (
    query_generator,
    reciprocal_rank_fusion,
    format_docs,
    answer_chain,
)

# ── Test cases (grounded in the PDFs in data/) ────────────────────────────────
TEST_CASES = [
    {
        "question": "What is RAG and why is it used?",
        "ground_truth": (
            "RAG (Retrieval-Augmented Generation) combines document retrieval with "
            "LLM generation to produce accurate, grounded answers and reduce hallucinations."
        ),
    },
    {
        "question": "What is prompt engineering?",
        "ground_truth": (
            "Prompt engineering is designing and optimizing input prompts to guide LLMs "
            "toward desired outputs, including techniques like few-shot prompting, "
            "chain-of-thought, and system prompts."
        ),
    },
    {
        "question": "What is LangChain used for?",
        "ground_truth": (
            "LangChain is a framework for building LLM-powered applications, providing "
            "abstractions for chains, agents, memory, and integrations with various models."
        ),
    },
    {
        "question": "What is the difference between precision and recall in machine learning?",
        "ground_truth": (
            "Precision measures how many predicted positives are actually positive. "
            "Recall measures how many actual positives were correctly identified."
        ),
    },
    {
        "question": "What are vector embeddings in NLP?",
        "ground_truth": (
            "Embeddings are dense vector representations of text that capture semantic "
            "meaning, enabling similarity search and downstream ML tasks."
        ),
    },
    {
        "question": "What is LangGraph?",
        "ground_truth": (
            "LangGraph is a library for building stateful, multi-actor LLM applications "
            "using a graph-based control flow, enabling cycles and conditional branching."
        ),
    },
]

def collect_rag_fusion_outputs(test_cases: list[dict]) -> list[dict]:
    """
    For each test question:
      1. Generate 3 alternative queries
      2. Retrieve + RRF-rank chunks
      3. Generate answer from those exact chunks
      4. Return (question, answer, contexts, ground_truth) — contexts = what LLM actually saw
    """
    results = []
    for i, tc in enumerate(test_cases, 1):
        q = tc["question"]
        print(f"  [{i}/{len(test_cases)}] {q[:70]}...")

        # Step 1: generate alternative queries
        queries = query_generator.invoke(q)
        print(f"         Generated queries: {[x.strip() for x in queries if x.strip()]}")

        # Step 2: RRF-ranked docs
        ranked_docs = reciprocal_rank_fusion(queries)
        top_docs = ranked_docs[:4]  # use top 4 for context

        # Step 3: build context and generate answer from the SAME chunks
        context = format_docs(top_docs)
        answer = answer_chain.invoke({"context": context, "question": q})

        results.append(
            {
                "question": q,
                "answer": answer,
                "contexts": [d.page_content for d in top_docs],
                "ground_truth": tc.get("ground_truth", ""),
                "num_unique_chunks": len(ranked_docs),
            }
        )
    return results

def run_ragas(results: list[dict]):
    try:
        from ragas import evaluate
        from ragas.metrics import faithfulness, answer_relevancy, context_precision
        from datasets import Dataset
    except ImportError:
        print("\nRAGAS not installed. Run: pip install ragas>=0.1.0,<0.2.0 datasets")
        return None

    dataset = Dataset.from_dict(
        {
            "question":     [r["question"] for r in results],
            "answer":       [r["answer"] for r in results],
            "contexts":     [r["contexts"] for r in results],
            "ground_truth": [r["ground_truth"] for r in results],
        }
    )
    return evaluate(dataset, metrics=[faithfulness, answer_relevancy, context_precision])

if __name__ == "__main__":
    print("=" * 60)
    print("RAG Fusion Evaluation — collecting outputs...")
    print("=" * 60 + "\n")

    results = collect_rag_fusion_outputs(TEST_CASES)

    print("\nRunning RAGAS metrics...")
    scores = run_ragas(results)

    if scores is not None:
        print("\n── RAGAS Summary ─────────────────────────────────────")
        print(scores)

        df = scores.to_pandas()
        output = {
            "backend": "rag_fusion",
            "summary": {
                "faithfulness":      float(df["faithfulness"].mean()),
                "answer_relevancy":  float(df["answer_relevancy"].mean()),
                "context_precision": float(df["context_precision"].mean()),
            },
            "per_question": df.to_dict(orient="records"),
        }
        out_path = os.path.join(os.path.dirname(__file__), "evaluation_results.json")
        with open(out_path, "w") as f:
            json.dump(output, f, indent=2, default=str)
        print(f"\nResults saved to {out_path}")

    print("\n── Raw Q&A Output ────────────────────────────────────")
    for r in results:
        print(f"\nQ: {r['question']}")
        print(f"A: {r['answer'][:300]}...")
        print(f"   RRF ranked {r['num_unique_chunks']} unique chunks, used top 4")
    print("=" * 60)
