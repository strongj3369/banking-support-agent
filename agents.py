"""
The multi-agent layer.

  Classifier Agent   one message  ->  Positive Feedback | Negative Feedback | Query
  Feedback Handler   positive -> thank-you;  negative -> new ticket + apology
  Query Handler      pull the ticket number, look it up, report the status

"Multi-agent" here means one router and three specialists, each with its own
input contract and its own failure path. There is no framework: the routing,
the fallback and the escalation are ordinary Python, which is what makes them
testable.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time

from anthropic import Anthropic

import db

# Load .env if present. Without this the API key is invisible to the process and
# every classification silently lands on the fallback — which looks like the
# model getting everything wrong.
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))
except Exception:
    pass

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("agents")

MODEL = "claude-haiku-4-5"

CATEGORIES = ("Positive Feedback", "Negative Feedback", "Query")
SENTIMENTS = ("positive", "neutral", "negative")

# A bare 6-digit run of digits is how the spec writes ticket numbers, with or
# without a leading '#'.
TICKET_RE = re.compile(r"#?\b(\d{6})\b")

_client: Anthropic | None = None


class MissingAPIKey(RuntimeError):
    pass


def api_key() -> str | None:
    """The key from the environment, a .env file, or Streamlit secrets."""
    key = os.environ.get("ANTHROPIC_API_KEY")
    if key:
        return key
    try:  # Streamlit Community Cloud stores secrets outside the environment.
        import streamlit as st
        return st.secrets.get("ANTHROPIC_API_KEY")
    except Exception:
        return None


def client() -> Anthropic:
    global _client
    if _client is None:
        key = api_key()
        if not key:
            raise MissingAPIKey(
                "ANTHROPIC_API_KEY is not set. Put it in a .env file next to "
                "agents.py, or in Streamlit secrets when deployed."
            )
        _client = Anthropic(api_key=key)
    return _client


# ---------------------------------------------------------------------------
# 1. Classifier Agent
# ---------------------------------------------------------------------------

CLASSIFIER_SYSTEM = (
    "You classify inbound messages to a retail bank's customer support desk.\n"
    "Return exactly one category:\n"
    '  "Positive Feedback"  the customer is thanking, praising, or reporting '
    "that something was resolved well. No action is being requested.\n"
    '  "Negative Feedback"  the customer is reporting a problem, complaining, '
    "or expressing dissatisfaction. Something is wrong and needs fixing.\n"
    '  "Query"  the customer is asking for information, most often the status '
    "of an existing ticket. A message containing a 6-digit ticket number and "
    "asking about it is always a Query, even if the tone is annoyed.\n\n"
    "Also return `sentiment` (positive, neutral, or negative), which is a "
    "SEPARATE axis from the category: an angry status request is Query with "
    "negative sentiment, and it changes the WARMTH of the reply, not the route.\n"
    "Also return `customer_name` if the message states one, else null.\n\n"
    "Return STRICT JSON only, with exactly these keys: "
    '{"category": string, "sentiment": string, "customer_name": string|null}'
)


def _parse_json(text: str) -> dict | None:
    """Model output -> dict. Tolerates code fences. None on any failure."""
    candidate = (text or "").strip()
    if candidate.startswith("```"):
        candidate = re.sub(r"^```[a-zA-Z]*\s*", "", candidate)
        candidate = re.sub(r"\s*```$", "", candidate).strip()
    try:
        parsed = json.loads(candidate)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass
    start, end = candidate.find("{"), candidate.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        parsed = json.loads(candidate[start : end + 1])
        return parsed if isinstance(parsed, dict) else None
    except Exception:
        return None


def classify(message: str) -> dict:
    """
    Classify one message. NEVER raises and never returns an invalid category.

    Three failure modes are absorbed here: the API call raising, the model
    returning unparseable text, and the model returning a category outside the
    permitted set. All three land on the same code-level fallback, and every
    fallback is logged with what actually came back — a silent default makes a
    miss and a success look identical.
    """
    fallback_used = False

    # Deterministic pre-check: a message that is only a ticket number is a
    # Query, and costs zero tokens to route.
    stripped = message.strip()
    if TICKET_RE.fullmatch(stripped) or re.fullmatch(r"#?\d{6}\??", stripped):
        return {
            "category": "Query", "sentiment": "neutral",
            "customer_name": None, "fallback_used": False, "short_circuit": True,
        }

    raw = None
    try:
        response = client().messages.create(
            model=MODEL,
            max_tokens=200,
            system=CLASSIFIER_SYSTEM,
            messages=[{"role": "user", "content": message}],
        )
        raw = "".join(b.text for b in response.content if getattr(b, "type", "") == "text")
    except MissingAPIKey:
        raise
    except Exception as exc:
        logger.warning(f"classifier API call failed: {exc}")

    parsed = _parse_json(raw) if raw else None
    category = (parsed or {}).get("category")

    if category not in CATEGORIES:
        # Fallback in CODE, not in the prompt. "Negative Feedback" is the safe
        # default: it opens a ticket and puts a human in the loop. Defaulting
        # to Positive would thank someone for a complaint and drop it.
        logger.warning(f"classifier fallback: {message[:60]!r} -> {category!r}")
        category, fallback_used = "Negative Feedback", True

    sentiment = (parsed or {}).get("sentiment")
    if sentiment not in SENTIMENTS:
        sentiment = "neutral"

    name = (parsed or {}).get("customer_name")
    if not isinstance(name, str) or not name.strip():
        name = None

    return {
        "category": category, "sentiment": sentiment, "customer_name": name,
        "fallback_used": fallback_used, "short_circuit": False,
    }


# ---------------------------------------------------------------------------
# 2. Feedback Handler Agent
# ---------------------------------------------------------------------------

THANKS_SYSTEM = (
    "You write one short thank-you reply from a retail bank's support team. "
    "Two sentences maximum. Warm, specific to what the customer mentioned, "
    "never effusive, no emoji, no marketing. If a name is supplied, use it. "
    "Return the message text only."
)


def handle_positive(message: str, name: str | None) -> str:
    who = name or "there"
    try:
        response = client().messages.create(
            model=MODEL,
            max_tokens=150,
            system=THANKS_SYSTEM,
            messages=[{
                "role": "user",
                "content": f"Customer name: {name or 'unknown'}\nMessage: {message}",
            }],
        )
        text = "".join(b.text for b in response.content if getattr(b, "type", "") == "text").strip()
        if text:
            return text
    except Exception as exc:
        logger.warning(f"thank-you generation failed: {exc}")

    # Template fallback in the spec's own format.
    return f"Thank you for your kind words, {who}! We're delighted to assist you."


def handle_negative(message: str, name: str | None) -> tuple[str, str]:
    """Open a ticket and apologise. Returns (response, ticket_number)."""
    ticket_number = db.create_ticket(name or "Unknown Customer", message)
    response = (
        f"We apologize for the inconvenience. A new ticket #{ticket_number} has "
        "been generated, and our team will follow up shortly."
    )
    return response, ticket_number


# ---------------------------------------------------------------------------
# 3. Query Handler Agent
# ---------------------------------------------------------------------------

def handle_query(message: str) -> tuple[str, str | None, str]:
    """
    Look up a ticket. Returns (response, ticket_number, action).

    The spec shows only the happy path. Three others exist and each gets a
    defined answer rather than a crash or a guess.
    """
    match = TICKET_RE.search(message)

    if not match:
        # No number in the message at all.
        return (
            "I can look that up right away — could you share your 6-digit "
            "ticket number? If you don't have one, tell me what happened and "
            "I'll open a new ticket.",
            None,
            "no_ticket_number_in_message",
        )

    ticket_number = match.group(1)
    ticket = db.get_ticket(ticket_number)

    if ticket is None:
        # Well-formed number, no such ticket. Do not invent a status.
        return (
            f"I couldn't find a ticket #{ticket_number} on your account. Please "
            "double-check the number, or tell me what happened and I'll open a "
            "new ticket for you.",
            ticket_number,
            "ticket_not_found",
        )

    return (
        f"Your ticket #{ticket_number} is currently marked as: {ticket['status']}.",
        ticket_number,
        "status_returned",
    )


# ---------------------------------------------------------------------------
# Coordination
# ---------------------------------------------------------------------------

def route(message: str, log: bool = True) -> dict:
    """Classify one message, dispatch it, log the turn, return the trace."""
    started = time.time()

    verdict = classify(message)
    category = verdict["category"]
    name = verdict["customer_name"]

    ticket_number = None

    if category == "Positive Feedback":
        agent = "Feedback Handler (positive)"
        response = handle_positive(message, name)
        action = "thank_you_sent"

    elif category == "Negative Feedback":
        agent = "Feedback Handler (negative)"
        response, ticket_number = handle_negative(message, name)
        action = "ticket_created"

    else:
        agent = "Query Handler"
        response, ticket_number, action = handle_query(message)

    latency_ms = int((time.time() - started) * 1000)

    result = {
        "message": message,
        "classification": category,
        "sentiment": verdict["sentiment"],
        "customer_name": name,
        "agent": agent,
        "action": action,
        "ticket_number": ticket_number,
        "response": response,
        "latency_ms": latency_ms,
        "fallback_used": verdict["fallback_used"],
        "short_circuit": verdict["short_circuit"],
    }

    if log:
        db.log_event(
            message=message, classification=category, sentiment=verdict["sentiment"],
            agent=agent, action=action, ticket_number=ticket_number,
            response=response, latency_ms=latency_ms,
            fallback_used=int(verdict["fallback_used"]),
        )
    return result


if __name__ == "__main__":
    db.init_db()
    for m in [
        "Thanks for sorting out my net banking login issue.",
        "My debit card replacement still hasn't arrived.",
        "Could you check the status of ticket 650932?",
    ]:
        r = route(m)
        print(f"\n{m}\n  -> {r['classification']} / {r['sentiment']} -> {r['agent']}")
        print(f"  {r['response']}")
