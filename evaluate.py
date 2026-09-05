"""
Evaluation harness.

Measures the thing that actually decides whether this system works: does the
Classifier Agent send each message to the right specialist? Everything
downstream is deterministic, so routing accuracy is the system's accuracy.

The test set deliberately includes hard cases — praise that mentions a past
problem, complaints phrased politely, status requests with an angry tone —
because a test set of unambiguous messages measures nothing.

    python evaluate.py            # full run, writes eval_results.json
"""

from __future__ import annotations

import json
import time
from collections import Counter, defaultdict
from pathlib import Path

import agents
import db

RESULTS_PATH = Path(__file__).resolve().parent / "eval_results.json"

# (message, expected_category, note)
# note marks WHY a case is here, so a regression is diagnosable rather than
# just a number going down.
TEST_SET = [
    # ---- clear positive ----
    ("Thanks for sorting out my net banking login issue.", "Positive Feedback", "clear"),
    ("Thank you for resolving my credit card issue so quickly.", "Positive Feedback", "clear"),
    ("Your branch staff in Croydon were excellent today.", "Positive Feedback", "clear"),
    ("Really impressed with how fast the new app is.", "Positive Feedback", "clear"),
    ("Appreciate the callback yesterday, it sorted everything out.", "Positive Feedback", "clear"),
    ("Just wanted to say the fraud team handled this brilliantly.", "Positive Feedback", "clear"),
    ("Five stars for whoever fixed my standing order.", "Positive Feedback", "clear"),

    # ---- positive that MENTIONS a past problem (the trap) ----
    ("My card finally arrived, thanks for chasing it up.", "Positive Feedback", "praise mentions past problem"),
    ("The duplicate charge is gone now — appreciate you sorting it.", "Positive Feedback", "praise mentions past problem"),
    ("After all the trouble with the transfer, your team came through. Thank you.", "Positive Feedback", "praise mentions past problem"),
    ("Took a while but the overdraft fee was refunded. Cheers.", "Positive Feedback", "praise mentions past problem"),

    # ---- clear negative ----
    ("My debit card replacement still hasn't arrived.", "Negative Feedback", "clear"),
    ("I've been charged twice for the same transaction.", "Negative Feedback", "clear"),
    ("The mobile app crashes every time I open statements.", "Negative Feedback", "clear"),
    ("Nobody has called me back in four days.", "Negative Feedback", "clear"),
    ("My wire transfer never reached the recipient and it's been a week.", "Negative Feedback", "clear"),
    ("This is the third time I've had to explain this to your staff.", "Negative Feedback", "clear"),
    ("I was charged an overdraft fee despite being in credit.", "Negative Feedback", "clear"),
    ("The verification code never arrives so I can't add a payee.", "Negative Feedback", "clear"),

    # ---- negative phrased politely (the other trap) ----
    ("I'd be grateful if someone could look into a missing payment.", "Negative Feedback", "complaint phrased politely"),
    ("Hope you can help — my new card hasn't turned up yet.", "Negative Feedback", "complaint phrased politely"),
    ("Sorry to bother you, but the interest rate changed without any notice.", "Negative Feedback", "complaint phrased politely"),
    ("Thanks in advance for looking at this: my balance is showing wrong.", "Negative Feedback", "complaint opens with thanks"),

    # ---- clear query ----
    ("Could you check the status of ticket 650932?", "Query", "clear"),
    ("What's happening with ticket 784521?", "Query", "clear"),
    ("Any update on 310447?", "Query", "clear"),
    ("Status of my ticket #902188 please.", "Query", "clear"),
    ("Is ticket 445019 closed yet?", "Query", "clear"),
    ("Checking in on ticket 128736.", "Query", "clear"),

    # ---- query with negative tone (category vs sentiment) ----
    ("I'm still waiting. What is the status of ticket 573902?", "Query", "angry query"),
    ("This is ridiculous. Ticket 861254 — where are we?", "Query", "angry query"),
    ("Third time asking about ticket 650932. Status?", "Query", "angry query"),

    # ---- query, no ticket number ----
    ("Can you tell me where my complaint has got to?", "Query", "query without number"),
    ("What's the status of the case I opened last week?", "Query", "query without number"),

    # ---- general information queries ----
    ("What are your branch opening hours on Saturday?", "Query", "general info"),
    ("How long does an international transfer usually take?", "Query", "general info"),
    ("What documents do I need to open a joint account?", "Query", "general info"),

    # ---- short / awkward inputs ----
    ("650932", "Query", "bare ticket number"),
    ("#784521?", "Query", "bare ticket number"),
    ("thanks!", "Positive Feedback", "very short"),
    ("terrible service", "Negative Feedback", "very short"),
    ("Card declined again.", "Negative Feedback", "very short"),
    ("Cheers.", "Positive Feedback", "very short"),
]


def run(verbose: bool = True) -> dict:
    db.init_db()

    rows = []
    correct = 0
    fallbacks = 0
    latencies = []
    confusion = defaultdict(Counter)
    by_note = defaultdict(lambda: {"n": 0, "correct": 0})

    started = time.time()
    for i, (message, expected, note) in enumerate(TEST_SET, 1):
        # log=False: evaluation must not pollute the demo logs or open tickets.
        verdict = agents.classify(message)
        actual = verdict["category"]
        ok = actual == expected

        correct += ok
        fallbacks += verdict["fallback_used"]
        confusion[expected][actual] += 1
        by_note[note]["n"] += 1
        by_note[note]["correct"] += ok

        rows.append({
            "message": message, "expected": expected, "actual": actual,
            "sentiment": verdict["sentiment"], "note": note, "correct": ok,
            "fallback_used": verdict["fallback_used"],
        })
        if verbose:
            mark = "OK  " if ok else "MISS"
            print(f"  [{i:2}/{len(TEST_SET)}] {mark} {expected:18} -> {actual:18} | {message[:52]}")

    elapsed = time.time() - started
    n = len(TEST_SET)

    # Per-class precision / recall from the confusion matrix.
    per_class = {}
    for category in agents.CATEGORIES:
        tp = confusion[category][category]
        fn = sum(v for k, v in confusion[category].items() if k != category)
        fp = sum(confusion[e][category] for e in agents.CATEGORIES if e != category)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        per_class[category] = {
            "support": tp + fn, "precision": round(precision, 3),
            "recall": round(recall, 3), "f1": round(f1, 3),
        }

    results = {
        "n": n,
        "correct": correct,
        "routing_accuracy": round(correct / n, 3),
        "fallbacks_triggered": fallbacks,
        "elapsed_seconds": round(elapsed, 1),
        "per_class": per_class,
        "confusion_matrix": {e: dict(c) for e, c in confusion.items()},
        "by_case_type": {
            k: {**v, "accuracy": round(v["correct"] / v["n"], 3)}
            for k, v in sorted(by_note.items())
        },
        "misses": [r for r in rows if not r["correct"]],
        "rows": rows,
    }

    RESULTS_PATH.write_text(json.dumps(results, indent=2))

    if verbose:
        print(f"\n{'='*68}")
        print(f"Routing accuracy : {correct}/{n}  ({results['routing_accuracy']:.1%})")
        print(f"Fallbacks fired  : {fallbacks}")
        print(f"Elapsed          : {elapsed:.1f}s")
        print(f"\nPer class:")
        for cat, m in per_class.items():
            print(f"  {cat:18} P {m['precision']:.2f}  R {m['recall']:.2f}  "
                  f"F1 {m['f1']:.2f}  (n={m['support']})")
        print(f"\nBy case type:")
        for note, m in results["by_case_type"].items():
            print(f"  {note:32} {m['correct']}/{m['n']}  ({m['accuracy']:.0%})")
        if results["misses"]:
            print(f"\nMisses:")
            for m in results["misses"]:
                print(f"  expected {m['expected']:18} got {m['actual']:18} | {m['message'][:50]}")
        print(f"\nWritten to {RESULTS_PATH.name}")

    return results


if __name__ == "__main__":
    run()
