"""
MULTI-QUERY RAG
───────────────
Improves on basic RAG by generating multiple query variations.

Basic RAG problem: a single query may miss relevant chunks if phrased differently.

Multi-Query solution: LLM generates 3 alternative queries → retrieve for each
→ deduplicate → merge all chunks → answer.

Limitation: no intelligence in merging — a chunk that appears in all 3 result
sets is treated the same as one that appears in only 1. See 03_rag_fusion
for smarter ranking via Reciprocal Rank Fusion.

Flow:
  question
    → LLM generates 3 alternative queries
    → retrieve chunks for each query
    → deduplicate and merge
    → LLM → answer
"""

import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", ".env"), override=True)

# ── Vectorstore ───────────────────────────────────────────────────────────────
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = Chroma(
    persist_directory="../vectorstore",
    embedding_function=embeddings,
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# ── LLM ──────────────────────────────────────────────────────────────────────
llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)

# ── Step 1: Query generator ───────────────────────────────────────────────────
query_prompt = ChatPromptTemplate.from_template("""You are an AI assistant.
Generate 3 different versions of the given question to retrieve relevant documents.
Write one question per line. Do not number them. Do not add explanations.

Original question: {question}

3 alternative versions:""")

query_generator = (
    query_prompt
    | llm
    | StrOutputParser()
    | (lambda x: x.strip().split("\n"))
)

# ── Step 2: Retrieve and deduplicate ──────────────────────────────────────────
def retrieve_and_merge(queries: list[str]) -> list:
    seen: set[str] = set()
    merged = []
    for query in queries:
        if not query.strip():
            continue
        for doc in retriever.invoke(query.strip()):
            if doc.page_content not in seen:
                seen.add(doc.page_content)
                merged.append(doc)
    return merged

def format_docs(docs) -> str:
    return "\n\n".join(doc.page_content for doc in docs)

# ── Step 3: Answer prompt ─────────────────────────────────────────────────────
answer_prompt = ChatPromptTemplate.from_template("""You are a helpful assistant.
Answer the question using ONLY the context below.
If the answer is not in the context, say "I don't know based on the provided documents."

Context:
{context}

Question: {question}

Answer:""")

# ── Step 4: Full chain ────────────────────────────────────────────────────────
chain = (
    {
        "context": query_generator | retrieve_and_merge | format_docs,
        "question": RunnablePassthrough(),
    }
    | answer_prompt
    | llm
    | StrOutputParser()
)

if __name__ == "__main__":
    print("Multi-Query RAG ready! Type 'quit' to exit.\n")
    while True:
        question = input("Your question: ").strip()
        if not question:
            continue
        if question.lower() == "quit":
            break

        print("\nGenerating alternative queries...")
        queries = query_generator.invoke(question)
        for i, q in enumerate(queries, 1):
            if q.strip():
                print(f"  {i}. {q.strip()}")

        merged = retrieve_and_merge(queries)
        print(f"\nMerged {len(merged)} unique chunks")

        print("\n--- Answer ---")
        print(chain.invoke(question))
        print("-" * 60 + "\n")
