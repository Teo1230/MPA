# 🔬 ResearchMind — AI Research Intelligence Platform

> Multi-agent AI that turns academic papers into living knowledge graphs, surfacing hidden connections and generating novel hypotheses — powered by **Groq + LLaMA 3.1** (free, no credit card).

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    ResearchMind MVP                          │
│                                                             │
│   User Input (paper text)                                   │
│         │                                                   │
│         ▼                                                   │
│   ┌─────────────┐  structured JSON   ┌─────────────┐        │
│   │  Agent 1    │ ─────────────────► │   Agent 2   │        │
│   │  Paper      │ ◄── all papers ─── │  Synthesis  │        │
│   │  Analyzer   │                    │   Agent     │        │
│   └─────────────┘                    └─────────────┘        │
│         │                                   │               │
│         ▼                                   ▼               │
│   ┌─────────────┐   quality critique        hypotheses,     │
│   │  Agent 3    │   score, weaknesses,      themes, gaps    │
│   │  Paper      │   improvements                            │
│   │  Critic     │                                           │
│   └─────────────┘                                           │
└─────────────────────────────────────────────────────────────┘
```

**Agent 1 — Paper Analyzer** parses raw paper text and extracts structured knowledge (title, summary, key concepts, methodology, findings, limitations, keywords, domain).

**Agent 2 — Synthesis Agent** ingests all Agent 1 outputs, identifies cross-paper themes, surfaces research gaps, and generates novel evidence-backed hypotheses.

**Agent 3 — Paper Critic** evaluates each paper's methodology rigour, statistical soundness, reproducibility, and novelty. Returns a credibility score (0–10) with breakdown, weaknesses, and improvement suggestions.

**Bonus: Chat Agent** — conversational Q&A over your entire paper library.

---

## Quick Start

### 1. Get a free Groq API key

Go to [console.groq.com](https://console.groq.com) → sign up → create an API key (free, no credit card required).

### 2. Set your API key

Create a `.env` file inside the `researchmind-mvp/` folder:

```bash
echo "GROQ_API_KEY=your_key_here" > .env
```

### 3. Install dependencies

```bash
cd researchmind-mvp
pip install -r requirements.txt
```

### 4. Run the server

```bash
python3 -m uvicorn main:app --reload --port 8000
```

> **Note:** Use `python3 -m uvicorn` (not just `uvicorn`) if the command isn't found — this is because pip installs scripts to a user bin folder that may not be on your PATH.

### 5. Open the app

- **App UI:** [http://localhost:8000](http://localhost:8000)
- **Landing page:** [http://localhost:8000/researchmind-landing.html](http://localhost:8000/researchmind-landing.html)
- **API docs:** [http://localhost:8000/docs](http://localhost:8000/docs)
- **File access:** Open `researchmind-mvp/static/app.html` directly in your browser (demo mode)

---

## Demo Mode

The frontend works **without the backend** — open `researchmind-mvp/static/app.html` directly in your browser. It loads three pre-analyzed papers (Attention Is All You Need, AlphaFold, GNN Survey) so you can explore the UI immediately.

---

## API Reference

| Method | Endpoint | Agent | Description |
|--------|----------|-------|-------------|
| `GET` | `/api/health` | — | Server status + paper count |
| `POST` | `/api/analyze` | Agent 1 | Analyze a paper |
| `POST` | `/api/synthesize` | Agent 2 | Synthesize all papers |
| `POST` | `/api/critique/{id}` | Agent 3 | Critique paper quality |
| `GET` | `/api/papers` | — | List all analyzed papers |
| `DELETE` | `/api/papers/{id}` | — | Remove a paper |
| `POST` | `/api/chat` | Chat | Chat with your library |
| `GET` | `/api/synthesis/cached` | — | Retrieve last synthesis |

### Example: Analyze a paper

```bash
curl -X POST http://localhost:8000/api/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "title": "Attention Is All You Need",
    "text": "We propose a new simple network architecture, the Transformer..."
  }'
```

### Example: Critique a paper

```bash
curl -X POST http://localhost:8000/api/critique/abc12345
```

### Example: Synthesize

```bash
curl -X POST http://localhost:8000/api/synthesize \
  -H "Content-Type: application/json" \
  -d '{"focus": "attention mechanisms in protein folding"}'
```

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Backend | FastAPI 0.111, Python 3.9+ |
| AI Inference | Groq Cloud — LLaMA 3.1 8B Instant (~700 tok/s) |
| HTTP Client | AsyncGroq + httpx < 0.28 |
| Frontend | Vanilla JS, Three.js r128, Canvas 2D, GSAP 3.12 |
| Fonts | Space Grotesk, Inter, JetBrains Mono |

---

## Project Structure

```
MPA/
├── researchmind-landing.html    # Marketing landing page
├── researchmind-business.docx  # Business foundation document
├── researchmind-pitch.pptx     # Pitch deck (10 slides)
├── logo.png                    # Brand logo
└── researchmind-mvp/
    ├── main.py                 # FastAPI app + 3 AI agents
    ├── requirements.txt        # Pinned dependencies
    ├── .env                    # Your GROQ_API_KEY (create this)
    ├── README.md               # This file
    └── static/
        └── app.html            # Full frontend SPA
```

---

## Team

| Member | Role |
|--------|------|
| Teodor Mihailescu | Full-Stack Developer & AI Engineer |
| Anastasia Sandu | Frontend Developer & UX Designer |
| Selena Hurloi | Product Manager & Business Analyst |
| Adela Danescu | Backend Developer & QA Engineer |

MPA 2026 · University of Bucharest · Faculty of Mathematics and Computer Science

---

## Troubleshooting

**`zsh: command not found: uvicorn`**
Use `python3 -m uvicorn main:app --reload` instead.

**`503 GROQ_API_KEY not set`**
Create the `.env` file with your key: `echo "GROQ_API_KEY=gsk_..." > .env`

**`TypeError: __init__() got an unexpected keyword argument 'proxies'`**
Run `pip install "httpx<0.28.0"` — the requirements.txt already pins this, so re-run `pip install -r requirements.txt`.

**Papers not persisting between restarts**
The MVP uses an in-memory store. Restart = fresh library. A production version would use SQLite or PostgreSQL.

---

## License

MIT — built for MPA 2026. Free to use, modify, and deploy.
