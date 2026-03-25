import json
import os
import urllib.request
import urllib.error
from http.server import BaseHTTPRequestHandler

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL   = "llama-3.3-70b-versatile"


def build_prompt(currency: str, goal: str, income) -> str:
    income_line = f"The user's declared monthly income is {currency} {income}." if income else ""
    goal_map = {
        "general": "Provide a balanced general financial health overview.",
        "save":    "Focus on maximising savings and identifying unnecessary expenses.",
        "invest":  "Focus on freeing up capital for investment and wealth building.",
        "debt":    "Focus on debt reduction strategies and cash flow improvement.",
    }
    return f"""You are FinSight AI, an expert personal finance assistant. Analyse bank statements and deliver precise, data-grounded financial insights.

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
      "name": "Category name",
      "amount": <number>,
      "percentage": <number 0-100>,
      "trend": "up|down|stable",
      "flag": <true if unusually high>
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
      "detail": "Specific advice referencing actual merchants and amounts",
      "potential_saving": <number or null>
    }}
  ],
  "health_score": <integer 0-100>,
  "health_label": "Poor|Fair|Good|Excellent"
}}

Give 4-6 recommendations referencing specific merchants and amounts from the data."""


def call_groq(statement_text: str, currency: str, goal: str, income) -> dict:
    payload = json.dumps({
        "model": GROQ_MODEL,
        "max_tokens": 3000,
        "temperature": 0.15,
        "messages": [
            {"role": "system", "content": build_prompt(currency, goal, income)},
            {"role": "user",   "content": f"Analyse this bank statement:\n\n{statement_text[:6000]}"},
        ],
    }).encode("utf-8")

    req = urllib.request.Request(
        GROQ_API_URL,
        data=payload,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {GROQ_API_KEY}",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        error_body = e.read().decode("utf-8")
        raise ValueError(f"Groq API error {e.code}: {error_body}")

    raw = data["choices"][0]["message"]["content"].strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        start, end = raw.find("{"), raw.rfind("}") + 1
        if start != -1 and end > start:
            return json.loads(raw[start:end])
        raise ValueError("AI returned unparseable response")


CORS_HEADERS = [
    ("Access-Control-Allow-Origin",  "*"),
    ("Access-Control-Allow-Methods", "POST, OPTIONS"),
    ("Access-Control-Allow-Headers", "Content-Type"),
    ("Content-Type", "application/json"),
]


class handler(BaseHTTPRequestHandler):

    def _send(self, status: int, body: str):
        encoded = body.encode("utf-8")
        self.send_response(status)
        for k, v in CORS_HEADERS:
            self.send_header(k, v)
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def do_OPTIONS(self):
        self._send(200, "")

    def do_POST(self):
        try:
            # ── Guard: API key must exist ──────────────────────────────
            if not GROQ_API_KEY:
                self._send(500, json.dumps({
                    "detail": "GROQ_API_KEY is not set in environment variables."
                }))
                return

            # ── Parse body ─────────────────────────────────────────────
            length = int(self.headers.get("Content-Length", 0))
            raw_body = self.rfile.read(length)

            try:
                body = json.loads(raw_body)
            except json.JSONDecodeError:
                self._send(400, json.dumps({"detail": "Invalid JSON body."}))
                return

            text     = body.get("statement_text", "").strip()
            currency = body.get("currency", "AED")
            goal     = body.get("goal", "general")
            income   = body.get("monthly_income")

            if len(text) < 30:
                self._send(400, json.dumps({"detail": "Statement text too short."}))
                return

            # ── Call Groq ──────────────────────────────────────────────
            result = call_groq(text, currency, goal, income)
            self._send(200, json.dumps(result))

        except ValueError as e:
            # Groq API errors with full details
            self._send(502, json.dumps({"detail": str(e)}))

        except Exception as e:
            self._send(500, json.dumps({"detail": f"{type(e).__name__}: {str(e)}"}))