"""
NEXUS — AI Research Intelligence Platform
MVP Backend · FastAPI + Groq (free tier) · Two-Agent Architecture

Agent 1: Paper Analyzer Agent
  → Ingests raw paper text, extracts structured knowledge:
     title, summary, key concepts, methodology, findings, keywords, domain.

Agent 2: Synthesis Agent
  → Takes all analyzed papers, finds thematic connections and research gaps,
     then proposes novel hypotheses backed by evidence from the library.

Run:
    pip install -r requirements.txt
    uvicorn main:app --reload --port 8000

Then open: http://localhost:8000
"""

import os, json, uuid, asyncio
from datetime import datetime
from typing import Optional

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

# ─────────────────────────────────────────
#  App Setup
# ─────────────────────────────────────────
app = FastAPI(
    title="Nexus Research Intelligence API",
    version="1.0.0",
    description="Multi-Agent AI for academic research — powered by Groq + LLaMA 3",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────
#  Groq Client (free tier — no credit card)
#  Get your free key at: https://console.groq.com
# ─────────────────────────────────────────
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
client: Optional[Groq] = None

def get_client() -> Groq:
    global client
    if not GROQ_API_KEY:
        raise HTTPException(
            status_code=503,
            detail="GROQ_API_KEY not set. Add it to your .env file. "
                   "Get a free key at https://console.groq.com"
        )
    if client is None:
        client = Groq(api_key=GROQ_API_KEY)
    return client

# Model: fastest free model on Groq — ~700 tokens/sec
MODEL = "llama-3.1-8b-instant"

# ─────────────────────────────────────────
#  In-memory store (MVP; replace with DB in prod)
# ─────────────────────────────────────────
paper_store: dict[str, dict] = {}
synthesis_cache: Optional[dict] = None

# ─────────────────────────────────────────
#  Request / Response Models
# ─────────────────────────────────────────
class PaperInput(BaseModel):
    text: str
    title: str = "Untitled Paper"

class SynthesisRequest(BaseModel):
    focus: str = ""        # optional focus query for the synthesis

class ChatRequest(BaseModel):
    question: str

class PaperAnalysis(BaseModel):
    paper_id: str
    title: str
    summary: str
    key_concepts: list[str]
    methodology: str
    main_findings: list[str]
    limitations: list[str]
    keywords: list[str]
    research_domain: str
    analyzed_at: str

# ─────────────────────────────────────────
#  ███████╗ AGENT 1 — PAPER ANALYZER
#
#  Responsibility:
#    · Parse raw academic text
#    · Extract structured knowledge schema
#    · Return typed JSON consumed by Agent 2
# ─────────────────────────────────────────
async def paper_analyzer_agent(text: str, title: str) -> dict:
    """
    AGENT 1 — Paper Analyzer
    Input:  raw paper text (truncated to 3000 chars for MVP)
    Output: structured knowledge JSON
    """
    groq = get_client()
    truncated = text[:3500]

    system_prompt = """You are the Paper Analyzer Agent in the Nexus research intelligence system.
Your sole job is to parse academic papers and extract structured knowledge.

ALWAYS return valid JSON with exactly these fields:
{
  "title": "full paper title",
  "summary": "2-3 sentence executive summary",
  "key_concepts": ["concept1", "concept2", "concept3"],
  "methodology": "brief description of experimental or theoretical methods",
  "main_findings": ["finding 1", "finding 2", "finding 3"],
  "limitations": ["limitation 1", "limitation 2"],
  "keywords": ["kw1", "kw2", "kw3", "kw4"],
  "research_domain": "e.g. Machine Learning, Bioinformatics, etc."
}

Be precise and academic. Extract only what is stated in the text."""

    user_prompt = f"""Analyze this research paper and return the structured JSON.

Title hint: {title}

Paper text:
{truncated}"""

    response = groq.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system",  "content": system_prompt},
            {"role": "user",    "content": user_prompt},
        ],
        response_format={"type": "json_object"},
        temperature=0.15,
        max_tokens=1200,
    )

    return json.loads(response.choices[0].message.content)


# ─────────────────────────────────────────
#  ████████╗ AGENT 2 — SYNTHESIS AGENT
#
#  Responsibility:
#    · Receive the structured outputs from Agent 1 (for all papers)
#    · Discover thematic links, conflicts, and white-space in research
#    · Generate novel, evidence-backed hypotheses
#    · Produce a literature synthesis narrative
# ─────────────────────────────────────────
async def synthesis_agent(papers: list[dict], focus: str = "") -> dict:
    """
    AGENT 2 — Synthesis Agent
    Input:  list of PaperAnalysis dicts (output of Agent 1, across all papers)
    Output: synthesis JSON with connections, gaps, and hypotheses
    """
    groq = get_client()

    # Build compact representation for the prompt
    summaries = [
        {
            "title":           p.get("title", "Untitled"),
            "domain":          p.get("research_domain", "Unknown"),
            "key_concepts":    p.get("key_concepts", [])[:5],
            "main_findings":   p.get("main_findings", [])[:3],
            "methodology":     p.get("methodology", "N/A"),
        }
        for p in papers
    ]

    system_prompt = """You are the Synthesis Agent in the Nexus research intelligence system.
You receive structured analyses of multiple academic papers (produced by the Paper Analyzer Agent).
Your job is to:
  1. Identify common themes and cross-paper connections
  2. Surface research gaps — things no paper in the set has addressed
  3. Generate 3 novel, specific, evidence-backed research hypotheses
  4. Write a 2-paragraph synthesis narrative

Return ONLY valid JSON with this schema:
{
  "common_themes": ["theme 1", "theme 2", "theme 3"],
  "key_connections": [
    {"papers": ["Paper A", "Paper B"], "connection": "both use X to achieve Y"},
    ...
  ],
  "research_gaps": ["gap 1", "gap 2", "gap 3"],
  "novel_hypotheses": [
    {
      "hypothesis": "specific testable hypothesis statement",
      "supporting_evidence": "which paper concepts support this",
      "potential_impact": "why this matters"
    },
    ...
  ],
  "synthesis_narrative": "two-paragraph academic synthesis"
}

Be specific, rigorous, and cite paper titles in your responses."""

    focus_note = f"\n\nFocus the synthesis on: {focus}" if focus.strip() else ""

    user_prompt = f"""Synthesise these {len(papers)} research papers.{focus_note}

Paper analyses:
{json.dumps(summaries, indent=2)}"""

    response = groq.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
        response_format={"type": "json_object"},
        temperature=0.62,
        max_tokens=1800,
    )

    return json.loads(response.choices[0].message.content)


# ─────────────────────────────────────────
#  ██████╗  AGENT CHAT (bonus)
#
#  A lightweight conversational layer on top of the library.
#  Answers questions about the stored papers.
# ─────────────────────────────────────────
async def chat_agent(question: str, library: list[dict]) -> str:
    """Bonus: conversational Q&A over the paper library."""
    groq = get_client()

    context = "\n".join(
        f"- {p.get('title')}: {p.get('summary', '')} Concepts: {', '.join(p.get('key_concepts', []))}"
        for p in library
    )

    response = groq.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a knowledgeable research assistant. "
                    "Answer questions using ONLY the provided research library context. "
                    "Always cite specific paper titles. Be concise and precise."
                ),
            },
            {
                "role": "user",
                "content": f"Library context:\n{context}\n\nQuestion: {question}",
            },
        ],
        temperature=0.3,
        max_tokens=600,
    )

    return response.choices[0].message.content


# ─────────────────────────────────────────
#  API Endpoints
# ─────────────────────────────────────────
@app.get("/", response_class=FileResponse)
async def root():
    return FileResponse("static/app.html")


@app.get("/api/health")
async def health():
    return {
        "status": "ok",
        "model": MODEL,
        "papers_in_library": len(paper_store),
        "api_key_set": bool(GROQ_API_KEY),
        "timestamp": datetime.utcnow().isoformat(),
    }


@app.post("/api/analyze")
async def analyze_paper(payload: PaperInput):
    """
    Endpoint consumed by Agent 1 — Paper Analyzer.
    Accepts raw paper text, returns structured analysis.
    """
    if len(payload.text.strip()) < 50:
        raise HTTPException(400, "Paper text is too short (minimum 50 characters).")

    analysis = await paper_analyzer_agent(payload.text, payload.title)

    paper_id = str(uuid.uuid4())[:8]
    analysis["paper_id"] = paper_id
    analysis["analyzed_at"] = datetime.utcnow().isoformat()
    analysis.setdefault("title", payload.title)

    paper_store[paper_id] = analysis
    global synthesis_cache
    synthesis_cache = None  # invalidate synthesis cache

    return {
        "success":   True,
        "paper_id":  paper_id,
        "agent":     "Paper Analyzer Agent",
        "analysis":  analysis,
    }


@app.post("/api/synthesize")
async def synthesize(payload: SynthesisRequest):
    """
    Endpoint consumed by Agent 2 — Synthesis Agent.
    Synthesises all papers currently in the library.
    """
    papers = list(paper_store.values())
    if not papers:
        raise HTTPException(
            400,
            "No papers in the library yet. Analyze at least one paper first."
        )

    result = await synthesis_agent(papers, payload.focus)
    result["papers_analyzed"] = len(papers)
    result["agent"] = "Synthesis Agent"
    result["synthesized_at"] = datetime.utcnow().isoformat()

    global synthesis_cache
    synthesis_cache = result
    return {"success": True, "synthesis": result}


@app.get("/api/papers")
async def list_papers():
    """Return all analyzed papers."""
    return {"papers": list(paper_store.values()), "count": len(paper_store)}


@app.delete("/api/papers/{paper_id}")
async def delete_paper(paper_id: str):
    if paper_id not in paper_store:
        raise HTTPException(404, "Paper not found.")
    del paper_store[paper_id]
    global synthesis_cache
    synthesis_cache = None
    return {"success": True}


@app.post("/api/chat")
async def chat(payload: ChatRequest):
    """Chat with your library (bonus agent)."""
    papers = list(paper_store.values())
    if not papers:
        raise HTTPException(400, "Add papers to your library first.")
    answer = await chat_agent(payload.question, papers)
    return {"answer": answer}


@app.get("/api/synthesis/cached")
async def cached_synthesis():
    if not synthesis_cache:
        return {"available": False}
    return {"available": True, "synthesis": synthesis_cache}


# ─────────────────────────────────────────
#  Static files
# ─────────────────────────────────────────
app.mount("/static", StaticFiles(directory="static"), name="static")


if __name__ == "__main__":
    import uvicorn
    print("\n🔬 Nexus Research Intelligence — MVP")
    print("=" * 42)
    print("  Agent 1 : Paper Analyzer")
    print("  Agent 2 : Synthesis Agent")
    print("  AI      : Groq + LLaMA 3.1 (free)")
    print("  Docs    : http://localhost:8000/docs")
    print("=" * 42 + "\n")
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
