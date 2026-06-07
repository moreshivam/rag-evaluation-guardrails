"""
Shared RAG Fusion module — imported by evaluation/ and guardrails/.

Exports:
  chain                  full end-to-end chain  (question → answer)
  answer_chain           answer step only        ({context, question} → answer)
  query_generator        question → [q1, q2, q3]
  reciprocal_rank_fusion queries → RRF-ranked docs
  format_docs            docs → single context string
  retriever              raw ChromaDB retriever
  llm                    Groq LLM instance
"""

import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(dotenv_path=os.path.join(_BASE_DIR, ".env"), override=True)

# ── Vectorstore ───────────────────────────────────────────────────────────────
_embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = Chroma(
    persist_directory=os.path.join(_BASE_DIR, "vectorstore"),
    embedding_function=_embeddings,
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# ── LLM ──────────────────────────────────────────────────────────────────────
llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)

# ── Step 1: Query generator ───────────────────────────────────────────────────
_query_prompt = ChatPromptTemplate.from_template("""You are an AI assistant.
Generate 3 different versions of the given question to retrieve relevant documents.
Write one question per line. Do not number them. Do not add explanations.

Original question: {question}

3 alternative versions:""")

query_generator = (
    _query_prompt
    | llm
    | StrOutputParser()
    | (lambda x: x.strip().split("\n"))
)

# ── Step 2: Reciprocal Rank Fusion ────────────────────────────────────────────
def reciprocal_rank_fusion(queries: list[str], k: int = 60) -> list:
    scores: dict[str, float] = {}
    docs_map: dict[str, object] = {}
    for query in queries:
        if not query.strip():
            continue
        for rank, doc in enumerate(retriever.invoke(query.strip()), start=1):
            content = doc.page_content
            if content not in docs_map:
                docs_map[content] = doc
                scores[content] = 0.0
            scores[content] += 1.0 / (k + rank)
    return sorted(docs_map.values(), key=lambda d: scores[d.page_content], reverse=True)

# ── Helpers ───────────────────────────────────────────────────────────────────
def format_docs(docs) -> str:
    return "\n\n".join(doc.page_content for doc in docs)

# ── Answer step (used directly when we need the context separately) ───────────
_answer_prompt = ChatPromptTemplate.from_template("""You are a helpful assistant.
Answer the question using ONLY the context below.
If the answer is not in the context, say "I don't know based on the provided documents."

Context:
{context}

Question: {question}

Answer:""")

answer_chain = _answer_prompt | llm | StrOutputParser()

# ── Full end-to-end chain (question → answer) ─────────────────────────────────
chain = (
    {
        "context": query_generator | reciprocal_rank_fusion | format_docs,
        "question": RunnablePassthrough(),
    }
    | _answer_prompt
    | llm
    | StrOutputParser()
)
