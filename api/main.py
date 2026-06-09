"""
RAG Fusion — FastAPI Server
────────────────────────────
Exposes the RAG Fusion pipeline as REST endpoints.

Endpoints:
  POST /chat          → main Q&A with conversation history
  GET  /health        → service health check
  GET  /sessions      → list active sessions
  DELETE /session/{id}→ clear a session's history

Run:
  pip install fastapi uvicorn
  cd rag-evaluation-guardrails
  uvicorn api.main:app --reload --port 8000
"""

import os
import sys
import uuid
import time
from collections import defaultdict

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag_fusion_chain import (
    query_generator,
    reciprocal_rank_fusion,
    format_docs,
    answer_chain,
    llm,
)
from guardrails.guardrails_rag import check_input, check_output
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# ── App setup ─────────────────────────────────────────────────────────────────
app = FastAPI(
    title="RAG Fusion API",
    version="1.0.0",
    description="RAG Fusion pipeline with conversation history and guardrails",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],       # restrict to your frontend domain in production
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Session store (in-memory) ─────────────────────────────────────────────────
# key: session_id → list of {role, content} dicts
# in production replace with Redis
sessions: dict[str, list[dict]] = defaultdict(list)

MAX_HISTORY_TURNS = 5   # keep last 5 turns per session

# ── Prompt with conversation history ─────────────────────────────────────────
chat_prompt = ChatPromptTemplate.from_template("""You are a helpful assistant.
Answer the question using ONLY the context below.
If the answer is not in the context, say "I don't know based on the provided documents."

Previous conversation:
{history}

Context:
{context}

Question: {question}

Answer:""")

chat_chain = chat_prompt | llm | StrOutputParser()

# ── Request / Response models ─────────────────────────────────────────────────
class ChatRequest(BaseModel):
    question:   str
    session_id: str | None = None    # if None, new session created

class SourceInfo(BaseModel):
    source: str
    page:   int | str

class ChatResponse(BaseModel):
    session_id:  str
    status:      str                 # ok / warning / blocked
    answer:      str | None
    message:     str | None          # warning or rejection text
    sources:     list[SourceInfo]
    queries:     list[str] | None    # 3 alternative queries used
    latency_ms:  float

# ── Helper: format history for prompt ─────────────────────────────────────────
def format_history(history: list[dict]) -> str:
    if not history:
        return "No previous conversation."
    lines = []
    for turn in history:
        role = "User" if turn["role"] == "user" else "Assistant"
        lines.append(f"{role}: {turn['content']}")
    return "\n".join(lines)

# ── Main chat endpoint ────────────────────────────────────────────────────────
@app.post("/chat", response_model=ChatResponse)
def chat(body: ChatRequest):
    t_start = time.time()

    # create new session if none provided
    session_id = body.session_id or str(uuid.uuid4())

    # ── Input guardrails ──────────────────────────────────────────────────────
    valid, reason = check_input(body.question)
    if not valid:
        return ChatResponse(
            session_id  = session_id,
            status      = "blocked",
            answer      = None,
            message     = reason,
            sources     = [],
            queries     = None,
            latency_ms  = round((time.time() - t_start) * 1000, 2),
        )

    # ── RAG Fusion retrieval ──────────────────────────────────────────────────
    queries     = [q for q in query_generator.invoke(body.question) if q.strip()]
    ranked_docs = reciprocal_rank_fusion(queries)
    top_docs    = ranked_docs[:4]
    context     = format_docs(top_docs)

    # ── Build history string for prompt ──────────────────────────────────────
    history = format_history(sessions[session_id])

    # ── Generate answer with history ─────────────────────────────────────────
    answer = chat_chain.invoke({
        "history":  history,
        "context":  context,
        "question": body.question,
    })

    # ── Output guardrail ──────────────────────────────────────────────────────
    faithful, warning = check_output(answer, context)

    # ── Save turn to session history ──────────────────────────────────────────
    sessions[session_id].append({"role": "user",      "content": body.question})
    sessions[session_id].append({"role": "assistant", "content": answer})

    # keep only last MAX_HISTORY_TURNS turns (each turn = 2 entries)
    max_entries = MAX_HISTORY_TURNS * 2
    if len(sessions[session_id]) > max_entries:
        sessions[session_id] = sessions[session_id][-max_entries:]

    return ChatResponse(
        session_id  = session_id,
        status      = "warning" if not faithful else "ok",
        answer      = answer,
        message     = warning if not faithful else None,
        sources     = [
            SourceInfo(
                source = d.metadata.get("source", "unknown"),
                page   = d.metadata.get("page", "?"),
            )
            for d in top_docs
        ],
        queries     = queries,
        latency_ms  = round((time.time() - t_start) * 1000, 2),
    )

# ── Health check ──────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {
        "status":           "ok",
        "active_sessions":  len(sessions),
    }

# ── Session management ────────────────────────────────────────────────────────
@app.get("/sessions")
def list_sessions():
    return {
        "active_sessions": len(sessions),
        "session_ids":     list(sessions.keys()),
    }

@app.delete("/session/{session_id}")
def clear_session(session_id: str):
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    sessions.pop(session_id)
    return {"status": "session cleared", "session_id": session_id}

# ── Run ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
