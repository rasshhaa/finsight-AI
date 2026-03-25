import os

GROQ_MODEL = "llama-3.3-70b-versatile"

def handler(request):
    return {
        "statusCode": 200,
        "headers": {"Content-Type": "application/json"},
        "body": {
            "status": "ok",
            "model": GROQ_MODEL
        }
    }