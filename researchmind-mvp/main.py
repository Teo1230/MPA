"""
ResearchMind — AI Research Intelligence Platform
MVP Backend · FastAPI + Groq (free tier) · Three-Agent Architecture

Agent 1: Paper Analyzer Agent
  → Ingests raw paper text, extracts structured knowledge:
     title, summary, key concepts, methodology, findings, keywords, domain.

Agent 2: Synthesis Agent
  → Takes all analyzed papers, finds thematic connections and research gaps,
     then proposes novel hypotheses backed by evidence from the library.

Agent 3: Paper Critic Agent
  → Evaluates methodology rigour, statistical soundness, reproducibility,
     novelty, and potential biases of individual papers. Scores each paper.

Run:
    pip install -r requirements.txt
    uvicorn main:app --reload --port 8000

Then open: http://localhost:8000
"""

from __future__ import annotations

import os, json, uuid, io, re
from datetime import datetime
from typing import Optional, List, Dict

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from groq import AsyncGroq          # AsyncGroq avoids the httpx 'proxies' issue
from dotenv import load_dotenv

load_dotenv()

# ─────────────────────────────────────────
#  App Setup
# ─────────────────────────────────────────
app = FastAPI(
    title="ResearchMind Research Intelligence API",
    version="1.1.0",
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
client: Optional[AsyncGroq] = None

def get_client() -> AsyncGroq:
    """Return a singleton AsyncGroq client. AsyncGroq uses httpx.AsyncClient
    which does NOT pass the removed 'proxies' kwarg — fixing the TypeError."""
    global client
    if not GROQ_API_KEY:
        raise HTTPException(
            status_code=503,
            detail="GROQ_API_KEY not set. Add it to your .env file. "
                   "Get a free key at https://console.groq.com"
        )
    if client is None:
        client = AsyncGroq(api_key=GROQ_API_KEY)
    return client

# Model: fastest free model on Groq — ~700 tokens/sec
MODEL = "llama-3.1-8b-instant"

# ─────────────────────────────────────────
#  In-memory store (MVP; replace with DB in prod)
# ─────────────────────────────────────────
paper_store: Dict[str, dict] = {}
synthesis_cache: Optional[dict] = None

# ─────────────────────────────────────────
#  Request / Response Models
# ─────────────────────────────────────────
class PaperInput(BaseModel):
    text: str
    title: str = "Untitled Paper"

class URLInput(BaseModel):
    url: str

class SynthesisRequest(BaseModel):
    focus: str = ""        # optional focus query for the synthesis

class ChatRequest(BaseModel):
    question: str

class PaperAnalysis(BaseModel):
    paper_id: str
    title: str
    summary: str
    key_concepts: List[str]
    methodology: str
    main_findings: List[str]
    limitations: List[str]
    keywords: List[str]
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
async def paper_analyzer_agent(text: str, title: str) -> dict:  # AGENT 1
    """
    AGENT 1 — Paper Analyzer
    Input:  raw paper text (truncated to 3000 chars for MVP)
    Output: structured knowledge JSON
    """
    groq = get_client()
    truncated = text[:9000]   # more context → richer extraction

    system_prompt = """You are the Paper Analyzer Agent in the ResearchMind research intelligence system.
Your sole job is to parse academic papers and extract structured knowledge.

ALWAYS return valid JSON with exactly these fields:
{
  "title": "full paper title",
  "summary": "2-3 sentence executive summary",
  "key_concepts": ["concept1", "concept2", "concept3"],
  "methodology": "brief description of experimental or theoretical methods used",
  "datasets_used": ["dataset or corpus name if mentioned, else empty list"],
  "evaluation_metrics": ["metrics used, e.g. F1, accuracy, AUROC, etc."],
  "main_findings": ["finding 1", "finding 2", "finding 3"],
  "limitations": ["limitation 1", "limitation 2"],
  "keywords": ["kw1", "kw2", "kw3", "kw4"],
  "research_domain": "e.g. Clinical NLP, Bioinformatics, etc."
}

Be precise and academic. Extract only what is stated in the text. If a field is not mentioned, use an empty list or 'Not specified'."""

    user_prompt = f"""Analyze this research paper and return the structured JSON.

Title hint: {title}

Paper text:
{truncated}"""

    response = await groq.chat.completions.create(   # await — AsyncGroq
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
async def synthesis_agent(papers: List[dict], focus: str = "") -> dict:  # AGENT 2
    """
    AGENT 2 — Synthesis Agent
    Input:  list of PaperAnalysis dicts (output of Agent 1, across all papers)
    Output: synthesis JSON with connections, gaps, and hypotheses
    """
    groq = get_client()

    # Build rich representation — give Agent 2 everything Agent 1 extracted
    summaries = [
        {
            "title":              p.get("title", "Untitled"),
            "domain":             p.get("research_domain", "Unknown"),
            "summary":            p.get("summary", ""),
            "key_concepts":       p.get("key_concepts", [])[:8],
            "methodology":        p.get("methodology", "N/A"),
            "datasets_used":      p.get("datasets_used", []),
            "evaluation_metrics": p.get("evaluation_metrics", []),
            "main_findings":      p.get("main_findings", [])[:5],
            "limitations":        p.get("limitations", [])[:3],
            "keywords":           p.get("keywords", [])[:6],
        }
        for p in papers
    ]

    system_prompt = """You are the Synthesis Agent in the ResearchMind research intelligence system.
You receive structured analyses of multiple academic papers (produced by the Paper Analyzer Agent).

Your job:
  1. Find SPECIFIC, CONCRETE connections between papers — shared methods, datasets, findings, contradictions, or complementary approaches. If papers are in the same domain, find exactly how they relate.
  2. Surface research gaps — important questions none of the papers address
  3. Generate 3 novel, TESTABLE and SPECIFIC research hypotheses by combining insights across papers. Hypotheses must reference actual paper content.
  4. Write a 2-paragraph synthesis narrative that reads like a proper academic literature review introduction — cite paper titles by name.

Return ONLY valid JSON with this schema:
{
  "common_themes": ["theme 1", "theme 2", "theme 3"],
  "key_connections": [
    {"papers": ["Paper A title", "Paper B title"], "connection": "specific description of how they relate — shared method, dataset, finding, or complementary approach"},
    ...
  ],
  "research_gaps": ["specific gap 1", "specific gap 2", "specific gap 3"],
  "novel_hypotheses": [
    {
      "hypothesis": "specific, testable hypothesis combining insights from named papers",
      "supporting_evidence": "cite specific findings or methods from the papers",
      "potential_impact": "concrete impact on the field"
    },
    ...
  ],
  "synthesis_narrative": "two-paragraph academic synthesis that reads like a literature review — name papers, compare their approaches, and identify the open research frontier"
}

Rules:
- Name papers explicitly by title in connections and hypotheses
- If papers share a domain (e.g. clinical NLP, mental health), deeply analyse methodological similarities/differences
- Never give generic connections like 'both use machine learning' — be specific about WHICH methods and HOW they differ or complement each other
- Hypotheses must be falsifiable and grounded in the actual paper content"""

    focus_note = f"\n\nFocus the synthesis on: {focus}" if focus.strip() else ""

    user_prompt = f"""Synthesise these {len(papers)} research papers.{focus_note}

Paper analyses:
{json.dumps(summaries, indent=2)}"""

    response = await groq.chat.completions.create(   # await — AsyncGroq
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
        response_format={"type": "json_object"},
        temperature=0.62,
        max_tokens=2800,
    )

    return json.loads(response.choices[0].message.content)


# ─────────────────────────────────────────
#  ███████╗ AGENT 3 — PAPER CRITIC
#
#  Responsibility:
#    · Evaluate methodology rigour, statistical soundness, reproducibility
#    · Flag weaknesses, biases, and potential issues
#    · Assign a credibility score (0–10) with breakdown
#    · Suggest improvements and follow-up experiments
# ─────────────────────────────────────────
async def paper_critic_agent(paper: dict) -> dict:  # AGENT 3
    """
    AGENT 3 — Paper Critic
    Input:  structured paper analysis from Agent 1
    Output: critique JSON with scores, weaknesses, and improvement suggestions
    """
    groq = get_client()

    system_prompt = """You are the Paper Critic Agent in the ResearchMind research intelligence system.
You receive a structured analysis of an academic paper (produced by the Paper Analyzer Agent).
Your job is to critically evaluate the paper's quality, rigour, and scientific soundness.

ALWAYS return valid JSON with exactly these fields:
{
  "credibility_score": 8.2,
  "score_breakdown": {
    "methodology_rigour": 8,
    "statistical_soundness": 7,
    "reproducibility": 9,
    "novelty": 8,
    "clarity": 9
  },
  "strengths": ["strength 1", "strength 2"],
  "weaknesses": ["weakness 1", "weakness 2"],
  "potential_biases": ["bias 1"],
  "reproducibility_notes": "brief assessment of whether results can be reproduced",
  "suggested_improvements": ["improvement 1", "improvement 2"],
  "follow_up_experiments": ["experiment idea 1", "experiment idea 2"],
  "overall_verdict": "one sentence summary of the paper quality"
}

Be rigorous, constructive, and specific. Reference paper concepts in your critique."""

    user_prompt = f"""Critically evaluate this research paper analysis:

{json.dumps(paper, indent=2)}"""

    response = await groq.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
        response_format={"type": "json_object"},
        temperature=0.35,
        max_tokens=1200,
    )

    return json.loads(response.choices[0].message.content)


# ─────────────────────────────────────────
#  ██████╗  AGENT CHAT (bonus)
#
#  A lightweight conversational layer on top of the library.
#  Answers questions about the stored papers.
# ─────────────────────────────────────────
async def chat_agent(question: str, library: List[dict]) -> str:  # BONUS AGENT
    """Bonus: conversational Q&A over the paper library."""
    groq = get_client()

    context = "\n".join(
        f"- {p.get('title')}: {p.get('summary', '')} Concepts: {', '.join(p.get('key_concepts', []))}"
        for p in library
    )

    response = await groq.chat.completions.create(   # await — AsyncGroq
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


@app.post("/api/fetch-paper")
async def fetch_paper(payload: URLInput):
    """
    Fetch a paper from a public URL (PDF or HTML) and extract its text.
    Supports direct PDF URLs (e.g. ACL Anthology, arXiv) and HTML pages.
    """
    url = payload.url.strip()
    if not url.startswith("http"):
        raise HTTPException(400, "URL must start with http:// or https://")

    try:
        async with httpx.AsyncClient(
            follow_redirects=True,
            timeout=30,
            headers={"User-Agent": "ResearchMind/1.0 (academic paper fetcher)"},
        ) as client:
            resp = await client.get(url)
            resp.raise_for_status()
    except httpx.HTTPError as e:
        raise HTTPException(502, f"Could not fetch URL: {e}")

    content_type = resp.headers.get("content-type", "")
    is_pdf = "pdf" in content_type or url.lower().endswith(".pdf")

    extracted_text = ""
    detected_title = url.split("/")[-1].replace(".pdf", "").replace("-", " ").replace("_", " ").title()

    if is_pdf:
        try:
            from pypdf import PdfReader
            reader = PdfReader(io.BytesIO(resp.content))
            pages = []
            for page in reader.pages[:25]:          # cap at 25 pages
                t = page.extract_text() or ""
                pages.append(t)
            extracted_text = "\n".join(pages)

            # Try to grab a better title from first-page text
            first_page = pages[0] if pages else ""
            lines = [l.strip() for l in first_page.splitlines() if l.strip()]
            if lines:
                detected_title = lines[0][:120]   # first non-empty line is usually the title

        except Exception as e:
            raise HTTPException(500, f"PDF extraction failed: {e}. Install pypdf: pip install pypdf")
    else:
        # HTML — strip tags, grab visible text
        try:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(resp.text, "html.parser")
            for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
                tag.decompose()
            extracted_text = soup.get_text(separator="\n", strip=True)
            title_tag = soup.find("title")
            if title_tag:
                detected_title = title_tag.get_text(strip=True)[:120]
        except Exception as e:
            raise HTTPException(500, f"HTML extraction failed: {e}")

    extracted_text = re.sub(r"\n{3,}", "\n\n", extracted_text).strip()

    if len(extracted_text) < 100:
        raise HTTPException(400, "Could not extract meaningful text from this URL. Try pasting the text directly.")

    return {
        "success": True,
        "url": url,
        "detected_title": detected_title,
        "text": extracted_text[:12000],   # cap; Agent 1 uses first 9000
        "char_count": len(extracted_text),
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


@app.post("/api/critique/{paper_id}")
async def critique_paper(paper_id: str):
    """
    Endpoint consumed by Agent 3 — Paper Critic.
    Accepts a paper_id already in the library and returns a quality critique.
    """
    if paper_id not in paper_store:
        raise HTTPException(404, "Paper not found. Analyze it first via /api/analyze.")
    paper = paper_store[paper_id]
    critique = await paper_critic_agent(paper)
    critique["paper_id"]  = paper_id
    critique["agent"]     = "Paper Critic Agent"
    critique["critiqued_at"] = datetime.utcnow().isoformat()
    # Store critique alongside the paper
    paper["critique"] = critique
    return {"success": True, "paper_id": paper_id, "critique": critique}


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

# Serve the logo from the parent folder
@app.get("/logo.png", include_in_schema=False)
async def logo():
    logo_path = os.path.join(os.path.dirname(__file__), "..", "logo.png")
    if os.path.exists(logo_path):
        return FileResponse(logo_path, media_type="image/png")
    raise HTTPException(404, "logo.png not found in parent directory")

# Favicon
@app.get("/favicon.svg", include_in_schema=False)
async def favicon_svg():
    p = os.path.join(os.path.dirname(__file__), "..", "favicon.svg")
    if os.path.exists(p):
        return FileResponse(p, media_type="image/svg+xml")
    raise HTTPException(404, "favicon.svg not found")

@app.get("/favicon.ico", include_in_schema=False)
async def favicon_ico():
    # Redirect .ico requests to the SVG (modern browsers handle SVG favicons)
    p = os.path.join(os.path.dirname(__file__), "..", "favicon.svg")
    if os.path.exists(p):
        return FileResponse(p, media_type="image/svg+xml")
    raise HTTPException(404, "favicon not found")

# Serve the landing page from the parent folder
@app.get("/researchmind-landing.html", include_in_schema=False)
async def landing():
    landing_path = os.path.join(os.path.dirname(__file__), "..", "researchmind-landing.html")
    if os.path.exists(landing_path):
        return FileResponse(landing_path, media_type="text/html")
    raise HTTPException(404, "researchmind-landing.html not found in parent directory")

# Alias — serve the app at the full relative path too
@app.get("/researchmind-business.docx", include_in_schema=False)
async def business_doc():
    p = os.path.join(os.path.dirname(__file__), "..", "researchmind-business.docx")
    if os.path.exists(p):
        return FileResponse(p, media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                            headers={"Content-Disposition": "attachment; filename=researchmind-business.docx"})

@app.get("/researchmind-pitch.pptx", include_in_schema=False)
async def pitch_deck():
    p = os.path.join(os.path.dirname(__file__), "..", "researchmind-pitch.pptx")
    if os.path.exists(p):
        return FileResponse(p, media_type="application/vnd.openxmlformats-officedocument.presentationml.presentation",
                            headers={"Content-Disposition": "attachment; filename=researchmind-pitch.pptx"})

@app.get("/researchmind-mvp/static/app.html", include_in_schema=False)
async def app_alias():
    return FileResponse("static/app.html")


if __name__ == "__main__":
    import uvicorn
    print("\n🔬 ResearchMind — AI Research Intelligence")
    print("=" * 46)
    print("  Agent 1 : Paper Analyzer")
    print("  Agent 2 : Synthesis Agent")
    print("  Agent 3 : Paper Critic")
    print("  AI      : Groq + LLaMA 3.1 (free)")
    print("  Docs    : http://localhost:8000/docs")
    print("=" * 46 + "\n")
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
