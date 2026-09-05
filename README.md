# Banking Customer Support AI Agent — Multi-Agent Architecture

MS-AGS Capstone, Course-End Project 2. Jennifer Strong, September 2026.

A Classifier Agent labels each inbound customer message and routes it to the
matching specialist:

| Classification | Handler | Database action |
|---|---|---|
| Positive Feedback | Feedback Handler | none — generates a thank-you |
| Negative Feedback | Feedback Handler | INSERT a new unique ticket |
| Query | Query Handler | SELECT the ticket status |

**Measured routing accuracy: 97.7% (42/43)** on a labelled test set, zero
classifier fallbacks triggered.

## Files

| File | Purpose |
|---|---|
| `db.py` | SQLite schema, unique ticket-number allocation, ticket CRUD, per-turn logging, seed data |
| `agents.py` | Classifier Agent, Feedback Handler, Query Handler, and the router |
| `evaluate.py` | 43-case labelled test set, routing accuracy, per-class metrics, confusion matrix |
| `app.py` | Streamlit dashboard — routing demo, tickets, logs, evaluation |

No agent framework. The routing, fallback and escalation are ordinary Python,
which is what makes them testable.

## Run it

```bash
pip install -r requirements.txt

cp .env.example .env          # add your ANTHROPIC_API_KEY
python db.py                  # create and seed the database
python evaluate.py            # run the test set, write eval_results.json
streamlit run app.py          # open the dashboard
```

The API key is read from the environment or a gitignored `.env`. No key appears
in the source.

## Design notes

- **Category and sentiment are separate axes.** An angry status request is a
  Query with negative sentiment — the sentiment changes the warmth of the reply,
  not the route.
- **Ticket numbers are checked for uniqueness**, not trusted to chance. The
  column is a PRIMARY KEY; an unchecked random draw eventually collides and the
  insert fails at exactly the moment a customer is complaining.
- **The classifier's fallback is in code, and it is logged.** On an API error,
  unparseable output, or an out-of-set label, it returns `Negative Feedback` —
  which opens a ticket and puts a human in the loop. Defaulting to Positive
  would thank someone for a complaint and drop it.
- **A deterministic short-circuit runs first.** A message that is only a ticket
  number is routed with zero tokens.
- **The Query Handler answers its three unspecified cases** — ticket found,
  ticket not found, no number in the message — rather than crashing or inventing
  a status.

## Known weakness

Politely phrased indirect complaints score 2/3; every other case type scores
100%. The single miss is *"Hope you can help — my new card hasn't turned up
yet."* It falls into the Query Handler's no-number path, which asks for a ticket
number and offers to open one, so the customer is still invited to raise the
issue. The failure costs a turn, not a complaint.
