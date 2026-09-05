"""
Support database for the Banking Customer Support AI Agent.

Two tables:
  support_tickets  the tickets the Negative Feedback Handler creates and the
                   Query Handler reads
  agent_logs       one row per handled message, for the Logs & Debugging view

SQLite, because the capstone needs a support database, not a database project.
"""

from __future__ import annotations

import random
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent / "support.db"

VALID_STATUSES = ("Open", "In Progress", "Resolved", "Escalated")


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

def init_db() -> None:
    with connect() as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS support_tickets (
                ticket_number TEXT PRIMARY KEY,
                customer_name TEXT NOT NULL,
                message       TEXT NOT NULL,
                status        TEXT NOT NULL DEFAULT 'Open',
                created_at    TEXT NOT NULL,
                updated_at    TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS api_usage (
                day   TEXT PRIMARY KEY,
                calls INTEGER NOT NULL DEFAULT 0
            );

            CREATE TABLE IF NOT EXISTS agent_logs (
                id             INTEGER PRIMARY KEY AUTOINCREMENT,
                ts             TEXT NOT NULL,
                message        TEXT NOT NULL,
                classification TEXT,
                sentiment      TEXT,
                agent          TEXT,
                action         TEXT,
                ticket_number  TEXT,
                response       TEXT,
                latency_ms     INTEGER,
                fallback_used  INTEGER DEFAULT 0
            );
            """
        )


# ---------------------------------------------------------------------------
# Ticket numbers
# ---------------------------------------------------------------------------

def generate_ticket_number() -> str:
    """
    A NEW, UNIQUE 6-digit ticket number.

    The spec says "unique", so the number is checked against the table rather
    than trusted to chance. With 900,000 possible values a naive random pick
    collides eventually; this loop makes that impossible instead of unlikely.
    """
    with connect() as conn:
        for _ in range(50):
            candidate = str(random.randint(100000, 999999))
            hit = conn.execute(
                "SELECT 1 FROM support_tickets WHERE ticket_number = ?",
                (candidate,),
            ).fetchone()
            if hit is None:
                return candidate
    raise RuntimeError("Could not allocate a unique ticket number after 50 tries")


# ---------------------------------------------------------------------------
# Ticket operations
# ---------------------------------------------------------------------------

def create_ticket(customer_name: str, message: str) -> str:
    ticket_number = generate_ticket_number()
    stamp = now_iso()
    with connect() as conn:
        conn.execute(
            "INSERT INTO support_tickets "
            "(ticket_number, customer_name, message, status, created_at, updated_at) "
            "VALUES (?, ?, ?, 'Open', ?, ?)",
            (ticket_number, customer_name, message, stamp, stamp),
        )
    return ticket_number


def get_ticket(ticket_number: str) -> dict | None:
    with connect() as conn:
        row = conn.execute(
            "SELECT * FROM support_tickets WHERE ticket_number = ?",
            (ticket_number,),
        ).fetchone()
    return dict(row) if row else None


def update_status(ticket_number: str, status: str) -> bool:
    if status not in VALID_STATUSES:
        return False
    with connect() as conn:
        cur = conn.execute(
            "UPDATE support_tickets SET status = ?, updated_at = ? "
            "WHERE ticket_number = ?",
            (status, now_iso(), ticket_number),
        )
    return cur.rowcount > 0


def list_tickets(limit: int = 200) -> list[dict]:
    with connect() as conn:
        rows = conn.execute(
            "SELECT * FROM support_tickets ORDER BY created_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
    return [dict(r) for r in rows]


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def log_event(**kwargs) -> None:
    fields = (
        "message", "classification", "sentiment", "agent", "action",
        "ticket_number", "response", "latency_ms", "fallback_used",
    )
    values = [kwargs.get(f) for f in fields]
    with connect() as conn:
        conn.execute(
            f"INSERT INTO agent_logs (ts, {', '.join(fields)}) "
            f"VALUES (?, {', '.join('?' * len(fields))})",
            [now_iso(), *values],
        )


def list_logs(limit: int = 200) -> list[dict]:
    with connect() as conn:
        rows = conn.execute(
            "SELECT * FROM agent_logs ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
    return [dict(r) for r in rows]


# ---------------------------------------------------------------------------
# API usage cap
# ---------------------------------------------------------------------------

# This app is deployed publicly to demonstrate the project. The demo runs on a
# personal API key, so the live routing path is capped. Everything that does
# not call the model — the ticket table, the logs, the evaluation results —
# stays available whether or not the cap has been reached.
MAX_CALLS_PER_DAY = 200


def today() -> str:
    return datetime.now(timezone.utc).date().isoformat()


def calls_today() -> int:
    with connect() as conn:
        row = conn.execute(
            "SELECT calls FROM api_usage WHERE day = ?", (today(),)
        ).fetchone()
    return row["calls"] if row else 0


def record_call() -> int:
    """Increment today's counter and return the new total."""
    with connect() as conn:
        conn.execute(
            "INSERT INTO api_usage (day, calls) VALUES (?, 1) "
            "ON CONFLICT(day) DO UPDATE SET calls = calls + 1",
            (today(),),
        )
        row = conn.execute(
            "SELECT calls FROM api_usage WHERE day = ?", (today(),)
        ).fetchone()
    return row["calls"]


def budget_remaining() -> int:
    return max(0, MAX_CALLS_PER_DAY - calls_today())


# ---------------------------------------------------------------------------
# Seed data
# ---------------------------------------------------------------------------

# Pre-existing tickets in mixed states. Without these the Query Handler has
# nothing to find and the dashboard has no history to show.
SEED_TICKETS = [
    ("650932", "Priya Raman",     "Debit card replacement not received after 3 weeks.",      "Resolved"),
    ("784521", "Marcus Bell",     "Duplicate charge on my credit card statement.",           "In Progress"),
    ("310447", "Aisha Karim",     "Net banking login fails with error NB-402.",              "Open"),
    ("902188", "Tom Ferreira",    "Wire transfer to my landlord never arrived.",             "Escalated"),
    ("445019", "Dana Whitfield",  "Mobile app crashes when I open the statements tab.",      "Resolved"),
    ("128736", "Ravi Chandran",   "Overdraft fee charged although balance was positive.",    "In Progress"),
    ("573902", "Elena Sokolova",  "Cannot add a payee, verification code never arrives.",    "Open"),
    ("861254", "Jerome Baptiste", "Interest rate on my savings account changed without notice.", "Resolved"),
]


def seed_db(reset: bool = False) -> int:
    init_db()
    with connect() as conn:
        if reset:
            conn.execute("DELETE FROM support_tickets")
            conn.execute("DELETE FROM agent_logs")
        stamp = now_iso()
        inserted = 0
        for number, name, message, status in SEED_TICKETS:
            cur = conn.execute(
                "INSERT OR IGNORE INTO support_tickets "
                "(ticket_number, customer_name, message, status, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (number, name, message, status, stamp, stamp),
            )
            inserted += cur.rowcount
    return inserted


if __name__ == "__main__":
    n = seed_db(reset=True)
    print(f"Database ready at {DB_PATH}")
    print(f"Seeded {n} tickets across {len(set(s[3] for s in SEED_TICKETS))} statuses.")
