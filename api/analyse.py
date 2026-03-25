import json
import os
import httpx
from typing import Optional, List

# Configuration
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = "llama-3.3-70b-versatile"

# Models (simplified for serverless)
class AnalyseRequest:
    def __init__(self, statement_text: str, currency: str = "AED", goal: str = "general", monthly_income: Optional[float] = None):
        self.statement_text = statement_text
        self.currency = currency
        self.goal = goal
        self.monthly_income = monthly_income

class SpendingCategory:
    def __init__(self, name: str, amount: float, percentage: float, trend: str, flag: bool):
        self.name = name
        self.amount = amount
        self.percentage = percentage
        self.trend = trend
        self.flag = flag

    def to_dict(self):
        return {
            "name": self.name,
            "amount": self.amount,
            "percentage": self.percentage,
            "trend": self.trend,
            "flag": self.flag
        }

class SavingsForecast:
    def __init__(self, current_monthly_savings: float, projected_3m: float, projected_6m: float, projected_12m: float, savings_rate_pct: float):
        self.current_monthly_savings = current_monthly_savings
        self.projected_3m = projected_3m
        self.projected_6m = projected_6m
        self.projected_12m = projected_12m
        self.savings_rate_pct = savings_rate_pct

    def to_dict(self):
        return {
            "current_monthly_savings": self.current_monthly_savings,
            "projected_3m": self.projected_3m,
            "projected_6m": self.projected_6m,
            "projected_12m": self.projected_12m,
            "savings_rate_pct": self.savings_rate_pct
        }

class Recommendation:
    def __init__(self, priority: str, category: str, title: str, detail: str, potential_saving: Optional[float] = None):
        self.priority = priority
        self.category = category
        self.title = title
        self.detail = detail
        self.potential_saving = potential_saving

    def to_dict(self):
        return {
            "priority": self.priority,
            "category": self.category,
            "title": self.title,
            "detail": self.detail,
            "potential_saving": self.potential_saving
        }

class FinSightResponse:
    def __init__(self, summary: str, total_income: float, total_expenses: float, net_cashflow: float,
                 categories: List[SpendingCategory], forecast: SavingsForecast, recommendations: List[Recommendation],
                 health_score: int, health_label: str):
        self.summary = summary
        self.total_income = total_income
        self.total_expenses = total_expenses
        self.net_cashflow = net_cashflow
        self.categories = categories
        self.forecast = forecast
        self.recommendations = recommendations
        self.health_score = health_score
        self.health_label = health_label

    def to_dict(self):
        return {
            "summary": self.summary,
            "total_income": self.total_income,
            "total_expenses": self.total_expenses,
            "net_cashflow": self.net_cashflow,
            "categories": [c.to_dict() for c in self.categories],
            "forecast": self.forecast.to_dict(),
            "recommendations": [r.to_dict() for r in self.recommendations],
            "health_score": self.health_score,
            "health_label": self.health_label
        }

# Helper functions
def build_prompt(currency: str, goal: str, income: Optional[float]) -> str:
    income_line = f"The user's declared monthly income is {currency} {income}." if income else ""
    goal_map = {
        "general": "Provide a balanced general financial health overview.",
        "save": "Focus on maximising savings and identifying unnecessary expenses.",
        "invest": "Focus on freeing up capital for investment and wealth building.",
        "debt": "Focus on debt reduction strategies and cash flow improvement.",
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
            {"role": "user", "content": user_message},
        ],
    }

    async with httpx.AsyncClient(timeout=60.0) as client:
        resp = await client.post(GROQ_API_URL, headers=headers, json=payload)

    if resp.status_code != 200:
        error_msg = resp.json().get("error", {}).get("message", "Groq API error")
        return {"statusCode": 502, "body": {"error": error_msg}}

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
        end = raw.rfind("}") + 1
        if start != -1 and end > start:
            try:
                return json.loads(raw[start:end])
            except:
                pass
        return {"statusCode": 500, "body": {"error": "Failed to parse AI response"}}

# Main handler
def handler(request):
    if not GROQ_API_KEY:
        return {
            "statusCode": 500,
            "headers": {"Content-Type": "application/json"},
            "body": {"error": "GROQ_API_KEY not configured"}
        }

    try:
        # Parse request body
        body = json.loads(request.get('body', '{}'))
        req = AnalyseRequest(
            statement_text=body.get('statement_text', ''),
            currency=body.get('currency', 'AED'),
            goal=body.get('goal', 'general'),
            monthly_income=body.get('monthly_income')
        )

        if len(req.statement_text.strip()) < 30:
            return {
                "statusCode": 400,
                "headers": {"Content-Type": "application/json"},
                "body": {"error": "Statement text too short."}
            }

        # This would need to be async, but Vercel Python runtime may not support async handlers well
        # For now, we'll make it synchronous
        import asyncio
        result = asyncio.run(process_analysis(req))

        if isinstance(result, dict) and 'statusCode' in result:
            return result

        response = FinSightResponse(
            summary=result.get("summary", ""),
            total_income=float(result.get("total_income", 0)),
            total_expenses=float(result.get("total_expenses", 0)),
            net_cashflow=float(result.get("net_cashflow", 0)),
            categories=[
                SpendingCategory(
                    name=c.get("name", ""),
                    amount=float(c.get("amount", 0)),
                    percentage=float(c.get("percentage", 0)),
                    trend=c.get("trend", "stable"),
                    flag=bool(c.get("flag", False))
                ) for c in result.get("categories", [])
            ],
            forecast=SavingsForecast(
                current_monthly_savings=float(result.get("forecast", {}).get("current_monthly_savings", 0)),
                projected_3m=float(result.get("forecast", {}).get("projected_3m", 0)),
                projected_6m=float(result.get("forecast", {}).get("projected_6m", 0)),
                projected_12m=float(result.get("forecast", {}).get("projected_12m", 0)),
                savings_rate_pct=float(result.get("forecast", {}).get("savings_rate_pct", 0)),
            ),
            recommendations=[
                Recommendation(
                    priority=r.get("priority", "medium"),
                    category=r.get("category", ""),
                    title=r.get("title", ""),
                    detail=r.get("detail", ""),
                    potential_saving=r.get("potential_saving")
                ) for r in result.get("recommendations", [])
            ],
            health_score=int(result.get("health_score", 50)),
            health_label=result.get("health_label", "Fair")
        )

        return {
            "statusCode": 200,
            "headers": {"Content-Type": "application/json"},
            "body": response.to_dict()
        }

    except Exception as e:
        return {
            "statusCode": 500,
            "headers": {"Content-Type": "application/json"},
            "body": {"error": str(e)}
        }

async def process_analysis(req: AnalyseRequest) -> dict:
    system_prompt = build_prompt(req.currency, req.goal, req.monthly_income)
    user_msg = f"Analyse this bank statement / transaction history:\n\n{req.statement_text[:6000]}"

    return await call_groq(system_prompt, user_msg)