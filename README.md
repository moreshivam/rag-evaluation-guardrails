# rag-evaluation-guardrails

A progressive RAG pipeline built on LangChain + Groq, showing three retrieval strategies, automated evaluation with RAGAS, input/output guardrails, and LangSmith tracing.

## Project structure

```
.
├── data/                           # Source PDFs (knowledge base)
│   ├── GenAI_Part1_LLM_Foundations_Prompt_Engineering.pdf
│   ├── GenAI_Part2_RAG_Engineering.pdf
│   ├── GenAI_Part3_LangChain_LangGraph_MCP.pdf
│   └── ML_DL_Essential_Questions.pdf
│
├── 01_basic_rag/
│   └── rag.py                      # Single query → top-k chunks → answer
│
├── 02_multi_query/
│   └── rag.py                      # 3 query variations → merge → answer
│
├── 03_rag_fusion/
│   └── rag.py                      # Multi-query + Reciprocal Rank Fusion
│
├── evaluation/
│   └── evaluate.py                 # RAGAS evaluation backed by RAG Fusion
│
├── guardrails/
│   └── guardrails_rag.py           # Input + output guardrails over RAG Fusion
│
├── rag_fusion_chain.py             # Shared RAG Fusion module (imported by evaluation + guardrails)
├── ingest.py                       # Load PDFs → chunk → embed → store in ChromaDB
├── requirements.txt
├── .env.example                    # Copy to .env and fill in your keys
└── .gitignore
```

---

## Quick start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure API keys
```bash
cp .env.example .env
```

Edit `.env` and fill in your keys:
```
GROQ_API_KEY=your_groq_api_key_here          # free at console.groq.com
LANGCHAIN_API_KEY=your_langsmith_key_here    # free at smith.langchain.com
LANGCHAIN_TRACING_V2=true                    # set false to disable tracing
LANGCHAIN_PROJECT=rag-evaluation-guardrails
LANGCHAIN_ENDPOINT=https://api.smith.langchain.com
```

### 3. Ingest documents (run once)
```bash
python ingest.py
```
Loads all PDFs from `data/`, splits into 500-char chunks, embeds with `all-MiniLM-L6-v2` (local, ~80 MB first run), stores in `./vectorstore/`.

### 4. Run a RAG strategy

**Basic RAG** — simplest, single query:
```bash
cd 01_basic_rag && python rag.py
```

**Multi-Query RAG** — 3 query variations, merged results:
```bash
cd 02_multi_query && python rag.py
```

**RAG Fusion** — best quality via RRF ranking:
```bash
cd 03_rag_fusion && python rag.py
```

**Guarded RAG Fusion** — RAG Fusion with input/output validation:
```bash
cd guardrails && python guardrails_rag.py
```

### 5. Evaluate
```bash
cd evaluation && python evaluate.py
```
Runs 6 test questions through RAG Fusion, scores with RAGAS, saves results to `evaluation/evaluation_results.json`.

---

## RAG strategies compared

| Strategy | How it works | Weakness |
|---|---|---|
| **01 Basic RAG** | Single query → top-4 chunks → answer | Misses chunks if phrasing differs |
| **02 Multi-Query** | LLM generates 3 queries → deduplicate → answer | No ranking — first-seen chunk wins |
| **03 RAG Fusion** | Multi-query + Reciprocal Rank Fusion scoring | More LLM calls, slightly slower |

---

## Shared RAG Fusion module

`rag_fusion_chain.py` at the project root is imported by both `evaluation/` and `guardrails/`.
It exports:

| Export | Description |
|---|---|
| `chain` | Full end-to-end chain: question → answer |
| `answer_chain` | Answer step only: {context, question} → answer |
| `query_generator` | question → [q1, q2, q3] |
| `reciprocal_rank_fusion` | queries → RRF-ranked docs |
| `format_docs` | docs → single context string |
| `retriever` | Raw ChromaDB retriever |
| `llm` | Groq LLM instance |

---

## Guardrails

`guardrails/guardrails_rag.py` wraps RAG Fusion with two validation layers:

| Layer | Check | Action |
|---|---|---|
| Input | Length — max 1000 chars | Block |
| Input | Topic scope — AI/ML/GenAI (LLM classifier) | Block |
| Output | Faithfulness — LLM-as-judge vs exact RRF context | Warn |

The faithfulness check uses the **same RRF-ranked context** passed to the LLM — no second retrieval.

Returns a result dict:
```python
{
  "status":  "ok" | "warning" | "blocked",
  "answer":  "...",
  "message": "...",   # warning/rejection, None if ok
  "sources": [{"source": "file.pdf", "page": 0}],
  "queries": ["alt q1", "alt q2", "alt q3"]
}
```

---

## Evaluation metrics (RAGAS)

Evaluation runs against RAG Fusion — contexts captured from the same RRF run used to generate the answer.

| Metric | What it measures |
|---|---|
| `faithfulness` | Are answer claims grounded in retrieved context? (0 = hallucinated, 1 = fully supported) |
| `answer_relevancy` | Does the answer address the question? |
| `context_precision` | Are the top-ranked chunks relevant to the question? |

---

## LangSmith Tracing

With `LANGCHAIN_TRACING_V2=true` set in `.env`, every chain invocation is automatically traced. No code changes needed. Each trace in the LangSmith dashboard shows:

- Full prompt sent to LLM at each step
- Alternative queries generated
- Retrieved chunks with scores
- Latency per step
- Token usage and cost

Disable tracing: set `LANGCHAIN_TRACING_V2=false` in `.env`.

---

## Stack

| Component | Choice |
|---|---|
| LLM | Groq `llama-3.1-8b-instant` |
| Embeddings | `all-MiniLM-L6-v2` via HuggingFace (local) |
| Vector DB | ChromaDB (local file) |
| Framework | LangChain |
| Evaluation | RAGAS |
| Tracing | LangSmith |
