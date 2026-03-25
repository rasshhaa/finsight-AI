"""
FinSight AI – FastAPI Backend
Personal Finance Intelligence Assistant powered by Groq (LLaMA 3.3)
Run: python -m uvicorn main:app --reload --port 8000
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional
import httpx
import json
import os

# ─── Config ───────────────────────────────────────────────────────────────────

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL   = "llama-3.3-70b-versatile"

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY environment variable is not set. Please set it to your Groq API key.")

# ─── App ──────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="FinSight AI API",
    description="LLM-powered personal finance intelligence assistant",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

if os.path.exists("index.html"):
    @app.get("/", response_class=FileResponse)
    async def serve_frontend():
        return FileResponse("index.html")

# ─── Models ───────────────────────────────────────────────────────────────────

class AnalyseRequest(BaseModel):
    statement_text: str
    currency: str = "AED"
    goal: Optional[str] = "general"      # general | save | invest | debt
    monthly_income: Optional[float] = None

class SpendingCategory(BaseModel):
    name: str
    amount: float
    percentage: float
    trend: str          # up | down | stable
    flag: bool          # true if overspending

class SavingsForecast(BaseModel):
    current_monthly_savings: float
    projected_3m: float
    projected_6m: float
    projected_12m: float
    savings_rate_pct: float

class Recommendation(BaseModel):
    priority: str       # high | medium | low
    category: str
    title: str
    detail: str
    potential_saving: Optional[float] = None

class FinSightResponse(BaseModel):
    summary: str
    total_income: float
    total_expenses: float
    net_cashflow: float
    categories: list[SpendingCategory]
    forecast: SavingsForecast
    recommendations: list[Recommendation]
    health_score: int       # 0-100
    health_label: str       # Poor | Fair | Good | Excellent

# ─── Groq Helper ──────────────────────────────────────────────────────────────

async def call_groq(system_prompt: str, user_message: str) -> dict:
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {GROQ_API_KEY}",
    }
    payload = {
        "model": GROQ_MODEL,
        "max_tokens": 3000,
        "temperature": 0.15,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_message},
        ],
    }

    async with httpx.AsyncClient(timeout=60.0) as client:
        resp = await client.post(GROQ_API_URL, headers=headers, json=payload)

    if resp.status_code != 200:
        detail = resp.json().get("error", {}).get("message", "Groq API error")
        raise HTTPException(status_code=502, detail=detail)

    raw = resp.json()["choices"][0]["message"]["content"].strip()

    # Strip markdown fences
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        start = raw.find("{")
        end   = raw.rfind("}") + 1
        if start != -1 and end > start:
            return json.loads(raw[start:end])
        raise HTTPException(status_code=500, detail="AI returned unparseable response")

# ─── System Prompt ────────────────────────────────────────────────────────────

def build_prompt(currency: str, goal: str, income: Optional[float]) -> str:
    income_line = f"The user's declared monthly income is {currency} {income}." if income else ""
    goal_map = {
        "general": "Provide a balanced general financial health overview.",
        "save":    "Focus on maximising savings and identifying unnecessary expenses.",
        "invest":  "Focus on freeing up capital for investment and wealth building.",
        "debt":    "Focus on debt reduction strategies and cash flow improvement.",
    }
    return f"""You are FinSight AI, an expert personal finance assistant. You analyse bank statements and transaction histories to deliver precise, data-grounded financial insights — never generic advice.

Currency: {currency}. {income_line}
Goal mode: {goal_map.get(goal, goal_map['general'])}

IMPORTANT: Respond ONLY with a valid JSON object. No preamble, no markdown, no extra text.

Required JSON structure:
{{
  "summary": "2-3 sentence personalised financial health summary based on the actual data",
  "total_income": <number>,
  "total_expenses": <number>,
  "net_cashflow": <number>,
  "categories": [
    {{
      "name": "Category name (e.g. Food & Dining, Rent, Transport)",
      "amount": <number>,
      "percentage": <number 0-100>,
      "trend": "up|down|stable",
      "flag": <true if this category is unusually high>
    }}
  ],
  "forecast": {{
    "current_monthly_savings": <number>,
    "projected_3m": <number>,
    "projected_6m": <number>,
    "projected_12m": <number>,
    "savings_rate_pct": <number 0-100>
  }},
  "recommendations": [
    {{
      "priority": "high|medium|low",
      "category": "category name",
      "title": "Short actionable title",
      "detail": "Specific advice grounded in the user's actual spending data",
      "potential_saving": <number or null>
    }}
  ],
  "health_score": <integer 0-100>,
  "health_label": "Poor|Fair|Good|Excellent"
}}

Extract all transactions from the input. Categorise them intelligently. Identify income vs expenses. Calculate realistic savings forecasts. Give 4-6 recommendations grounded in the actual data — reference specific amounts and merchants where possible."""

# ─── Routes ───────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok", "model": GROQ_MODEL}


@app.post("/analyse", response_model=FinSightResponse)
async def analyse_statement(req: AnalyseRequest):
    if len(req.statement_text.strip()) < 30:
        raise HTTPException(status_code=400, detail="Statement text too short.")

    system_prompt = build_prompt(req.currency, req.goal, req.monthly_income)
    user_msg = f"Analyse this bank statement / transaction history:\n\n{req.statement_text[:6000]}"

    result = await call_groq(system_prompt, user_msg)

    cats = [
        SpendingCategory(
            name=c.get("name",""),
            amount=float(c.get("amount",0)),
            percentage=float(c.get("percentage",0)),
            trend=c.get("trend","stable"),
            flag=bool(c.get("flag", False))
        ) for c in result.get("categories", [])
    ]

    fc = result.get("forecast", {})
    forecast = SavingsForecast(
        current_monthly_savings=float(fc.get("current_monthly_savings", 0)),
        projected_3m=float(fc.get("projected_3m", 0)),
        projected_6m=float(fc.get("projected_6m", 0)),
        projected_12m=float(fc.get("projected_12m", 0)),
        savings_rate_pct=float(fc.get("savings_rate_pct", 0)),
    )

    recs = [
        Recommendation(
            priority=r.get("priority","medium"),
            category=r.get("category",""),
            title=r.get("title",""),
            detail=r.get("detail",""),
            potential_saving=r.get("potential_saving"),
        ) for r in result.get("recommendations", [])
    ]

    return FinSightResponse(
        summary=result.get("summary",""),
        total_income=float(result.get("total_income",0)),
        total_expenses=float(result.get("total_expenses",0)),
        net_cashflow=float(result.get("net_cashflow",0)),
        categories=cats,
        forecast=forecast,
        recommendations=recs,
        health_score=int(result.get("health_score",50)),
        health_label=result.get("health_label","Fair"),
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)