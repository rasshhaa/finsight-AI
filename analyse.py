from http.server import BaseHTTPRequestHandler
import json
import urllib.request

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


def call_groq(statement_text: str, currency: str, goal: str, income) -> dict:
    system_prompt = build_prompt(currency, goal, income)
    user_msg = f"Analyse this bank statement / transaction history:\n\n{statement_text[:6000]}"

    payload = json.dumps({
        "model": GROQ_MODEL,
        "max_tokens": 3000,
        "temperature": 0.15,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_msg},
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

    with urllib.request.urlopen(req, timeout=60) as resp:
        data = json.loads(resp.read().decode("utf-8"))

    raw = data["choices"][0]["message"]["content"].strip()

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
        raise ValueError("AI returned unparseable response")


class handler(BaseHTTPRequestHandler):

    def do_OPTIONS(self):
        self.send_response(200)
        self._cors()
        self.end_headers()

    def do_POST(self):
        try:
            length  = int(self.headers.get("Content-Length", 0))
            body    = json.loads(self.rfile.read(length))

            text    = body.get("statement_text", "").strip()
            currency= body.get("currency", "AED")
            goal    = body.get("goal", "general")
            income  = body.get("monthly_income")

            if len(text) < 30:
                self._respond(400, {"detail": "Statement text too short."})
                return

            result = call_groq(text, currency, goal, income)
            self._respond(200, result)

        except Exception as e:
            self._respond(500, {"detail": str(e)})

    def _cors(self):
        self.send_header("Access-Control-Allow-Origin",  "*")
        self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def _respond(self, status: int, data: dict):
        body = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self._cors()
        self.send_header("Content-Type",   "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass  # silence default logging