"""
BASIC RAG (Retrieval-Augmented Generation)
──────────────────────────────────────────
Simplest pipeline: embed question → retrieve top-k chunks → LLM answers.

Limitation: if your question is phrased differently from the stored text,
you may miss relevant chunks. See 02_multi_query for an improvement.

Flow:
  question → retrieve top-k chunks → LLM → answer
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
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

# ── LLM ──────────────────────────────────────────────────────────────────────
llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)

# ── Prompt ────────────────────────────────────────────────────────────────────
prompt = ChatPromptTemplate.from_template("""You are a helpful assistant.
Answer the question using ONLY the context below.
If the answer is not in the context, say "I don't know based on the provided documents."

Context:
{context}

Question: {question}

Answer:""")

def format_docs(docs) -> str:
    return "\n\n".join(doc.page_content for doc in docs)

# ── Chain ─────────────────────────────────────────────────────────────────────
chain = (
    {
        "context": retriever | format_docs,
        "question": RunnablePassthrough(),
    }
    | prompt
    | llm
    | StrOutputParser()
)

if __name__ == "__main__":
    print("Basic RAG ready! Type 'quit' to exit.\n")
    while True:
        question = input("Your question: ").strip()
        if not question:
            continue
        if question.lower() == "quit":
            break

        docs = retriever.invoke(question)
        print("\nRetrieved chunks:")
        for i, doc in enumerate(docs, 1):
            source = doc.metadata.get("source", "unknown")
            page = doc.metadata.get("page", "?")
            print(f"  [{i}] {source} | Page {page}")
            print(f"  {doc.page_content[:150]}...")

        print("\n--- Answer ---")
        print(chain.invoke(question))
        print("-" * 60 + "\n")
