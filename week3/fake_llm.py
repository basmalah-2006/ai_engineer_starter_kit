import json
import re


def _clean_text(text):
    return " ".join(re.split(r"\s+", text.strip()))


def fake_complete(messages, model=None):
    """
    Offline fake LLM.

    It returns valid JSON in the same format expected by the Pydantic schema.
    Useful when no OPENAI_API_KEY is available.
    """

    user_text = ""

    for message in reversed(messages):
        if message.get("role") == "user":
            user_text = message.get("content", "")
            break

    text = user_text.lower()

    intent = "general"
    urgency = "low"

    sales_keywords = [
        "buy",
        "purchase",
        "pricing",
        "plans",
        "demo",
        "sales",
        "upgrade",
    ]

    billing_keywords = [
        "payment",
        "invoice",
        "billing",
        "charge",
        "refund",
        "subscription",
        "paid",
        "payment failed",
    ]

    support_keywords = [
        "cannot login",
        "can't login",
        "error",
        "crash",
        "bug",
        "not working",
        "broken",
        "issue",
        "problem",
        "help",
    ]

    spam_keywords = [
        "spam",
        "lottery",
        "you won",
        "free money",
        "click here",
        "urgent winner",
    ]

    high_urgency_keywords = [
        "urgent",
        "asap",
        "emergency",
        "cannot access",
        "can't access",
        "down",
        "blocked",
        "immediately",
    ]

    medium_urgency_keywords = [
        "soon",
        "today",
        "important",
        "please help",
    ]

    if any(keyword in text for keyword in sales_keywords):
        intent = "sales"

    if any(keyword in text for keyword in billing_keywords):
        intent = "billing"

    if any(keyword in text for keyword in support_keywords):
        intent = "support"

    if any(keyword in text for keyword in spam_keywords):
        intent = "spam"

    if any(keyword in text for keyword in high_urgency_keywords):
        urgency = "high"
    elif any(keyword in text for keyword in medium_urgency_keywords):
        urgency = "medium"
    else:
        urgency = "low"

    summary = _clean_text(user_text)[:180]

    if len(summary) < 3:
        summary = "No clear summary provided."

    return json.dumps(
        {
            "intent": intent,
            "urgency": urgency,
            "summary": summary,
        },
        ensure_ascii=False,
    )