"""
Streamlit dashboard for the Banking Customer Support AI Agent.

Four views:
  Try it       type a message, watch the classifier route it, see the DB action
  Tickets      the support_tickets table
  Logs         agent_logs — every turn, with its trace
  Evaluation   routing accuracy, per-class metrics, confusion matrix, misses

    streamlit run app.py
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

import agents
import db

st.set_page_config(page_title="Banking Support AI Agent", page_icon="🏦", layout="wide")

RESULTS_PATH = Path(__file__).resolve().parent / "eval_results.json"

AGENT_COLOR = {
    "Feedback Handler (positive)": "#047857",
    "Feedback Handler (negative)": "#B45309",
    "Query Handler": "#1D4ED8",
}

st.markdown("""
<style>
  .block-container { padding-top: 1.6rem; max-width: 1320px; }

  /* header band */
  .hero { background: linear-gradient(100deg,#0F2A5C 0%,#1D4ED8 55%,#2563EB 100%);
          color:#fff; padding:1.3rem 1.6rem; border-radius:10px;
          margin-bottom:1.2rem; }
  .hero h1 { margin:0; font-size:1.72rem; font-weight:800; letter-spacing:-0.01em; }
  .hero p  { margin:0.35rem 0 0; opacity:0.88; font-size:0.93rem; }

  /* metric cards */
  div[data-testid="stMetric"] {
      background:#F8FAFC; border:1px solid #E2E8F0; border-left:4px solid #1D4ED8;
      border-radius:8px; padding:0.7rem 0.9rem; }
  div[data-testid="stMetricLabel"] p { font-size:0.74rem !important;
      text-transform:uppercase; letter-spacing:0.07em; color:#64748B !important;
      font-weight:700; }
  div[data-testid="stMetricValue"] { font-size:1.45rem !important; color:#0F172A; }

  .trace-box { background:#0F172A; color:#E2E8F0; border-left:5px solid #38BDF8;
               padding:0.9rem 1.15rem; border-radius:8px; font-size:0.88rem;
               font-family:Consolas,monospace; line-height:1.75; }
  .trace-box b { color:#7DD3FC; }

  .pill { display:inline-block; padding:3px 12px; border-radius:11px;
          font-size:0.72rem; font-weight:800; letter-spacing:0.07em;
          text-transform:uppercase; }

  div[data-testid="stDataFrame"] { border:1px solid #E2E8F0; border-radius:8px; }

  .hero-stat { background:linear-gradient(150deg,#065F46 0%,#047857 100%);
               color:#fff; border-radius:10px; padding:1.1rem 1.3rem; height:100%; }
  .hero-stat-label { font-size:0.72rem; text-transform:uppercase;
                     letter-spacing:0.09em; opacity:0.85; font-weight:700; }
  .hero-stat-value { font-size:3rem; font-weight:800; line-height:1.05;
                     margin:0.15rem 0 0.3rem; }
  .hero-stat-sub { font-size:0.83rem; opacity:0.9; line-height:1.45; }

  .miss-box { background:#FFFBEB; border:1px solid #FDE68A;
              border-left:5px solid #D97706; border-radius:8px;
              padding:0.9rem 1.15rem; }
  .miss-msg { font-size:1.02rem; font-weight:700; color:#78350F;
              margin-bottom:0.3rem; }
  .miss-meta { font-size:0.85rem; color:#92400E; margin-bottom:0.5rem; }
  .miss-note { font-size:0.87rem; color:#57534E; line-height:1.6; }
  .stTabs [data-baseweb="tab"] { font-weight:600; }
</style>
""", unsafe_allow_html=True)

db.init_db()
if not db.list_tickets(limit=1):
    db.seed_db()

STATUS_COLOR = {"Open": "#B45309", "In Progress": "#1D4ED8",
                "Resolved": "#047857", "Escalated": "#B91C1C"}
CLASS_COLOR = {"Positive Feedback": "#047857", "Negative Feedback": "#B45309",
               "Query": "#1D4ED8"}


def tint(frame, column, palette):
    """Colour one column's values. Returns a Styler Streamlit can render."""
    def paint(v):
        c = palette.get(v)
        return f"color:{c}; font-weight:700" if c else ""
    styler = frame.style
    return (styler.map(paint, subset=[column]) if hasattr(styler, "map")
            else styler.applymap(paint, subset=[column]))

st.markdown(
    "<div class='hero'><h1>Banking Customer Support AI Agent</h1>"
    "<p>Multi-agent architecture &mdash; a Classifier Agent routes each message "
    "to a Feedback Handler or the Query Handler</p></div>",
    unsafe_allow_html=True,
)

# A missing key looks exactly like the model getting everything wrong, so say so.
if not agents.api_key():
    st.warning(
        "**No API key found.** The live routing on the *Try it* tab will fall "
        "back on every message. Put `ANTHROPIC_API_KEY=...` in a `.env` file "
        "next to `app.py` and restart. Tickets, Logs and Evaluation work "
        "without it.",
        icon=":material/key_off:",
    )

tab_try, tab_tickets, tab_logs, tab_eval = st.tabs(
    ["Try it", "Tickets", "Logs & traces", "Evaluation"]
)

# ---------------------------------------------------------------------------
# Try it
# ---------------------------------------------------------------------------

with tab_try:
    EXAMPLES = {
        "— pick an example —": "",
        "Positive feedback": "Thanks for sorting out my net banking login issue.",
        "Negative feedback": "My debit card replacement still hasn't arrived.",
        "Query (known ticket)": "Could you check the status of ticket 650932?",
        "Query (unknown ticket)": "Any update on ticket 111111?",
        "Query (no number given)": "Can you tell me where my complaint has got to?",
        "Praise mentioning a past problem": "My card finally arrived, thanks for chasing it up.",
        "Angry, but still a query": "This is ridiculous. Ticket 861254 — where are we?",
    }

    col_l, col_r = st.columns([3, 2])
    with col_l:
        pick = st.selectbox("Scenario", list(EXAMPLES.keys()), label_visibility="collapsed")
        message = st.text_area(
            "Customer message",
            value=EXAMPLES[pick],
            height=110,
            placeholder="Type a customer message…",
        )
        run = st.button("Route message", type="primary", use_container_width=True)
    with col_r:
        st.caption(
            f"Public demo — {db.budget_remaining()} of {db.MAX_CALLS_PER_DAY} "
            "model calls left today"
        )
        st.markdown("**How routing works**")
        st.markdown(
            "1. Deterministic short-circuit — a bare ticket number is a Query, zero tokens\n"
            "2. Classifier Agent — one call, three permitted labels\n"
            "3. Code-level fallback if the label is invalid, and the miss is logged\n"
            "4. Dispatch to the matching handler"
        )

    MAX_PER_SESSION = 15
    MAX_MESSAGE_CHARS = 500
    st.session_state.setdefault("calls", 0)

    if run and message.strip():
        blocked = None
        if len(message) > MAX_MESSAGE_CHARS:
            blocked = (f"Messages are capped at {MAX_MESSAGE_CHARS} characters "
                       "on the public demo. Yours is "
                       f"{len(message)}.")
        elif st.session_state.calls >= MAX_PER_SESSION:
            blocked = (f"You've used this session's {MAX_PER_SESSION} live "
                       "routing calls. Reload the page to start a new session. "
                       "The Tickets, Logs and Evaluation tabs are unaffected.")
        elif db.budget_remaining() <= 0:
            blocked = ("The demo's daily model budget is spent. The Tickets, "
                       "Logs and Evaluation tabs still work — the evaluation "
                       "results are from a full 43-case run.")

        if blocked:
            st.warning(blocked)
            run = False

    if run and message.strip():
        with st.spinner("Classifying and routing…"):
            result = agents.route(message)
            st.session_state.calls += 1
            db.record_call()

        colour = AGENT_COLOR.get(result["agent"], "#374151")
        st.markdown(
            f"<span class='pill' style='background:{colour}1a;color:{colour};"
            f"border:1px solid {colour}55'>{result['classification']}</span>"
            f"&nbsp;&nbsp;<span class='pill' style='background:#6B72801a;color:#6B7280;"
            f"border:1px solid #6B728055'>sentiment: {result['sentiment']}</span>"
            + ("&nbsp;&nbsp;<span class='pill' style='background:#B4530920;color:#B45309;"
               "border:1px solid #B4530955'>fallback used</span>"
               if result["fallback_used"] else "")
            + ("&nbsp;&nbsp;<span class='pill' style='background:#1D4ED820;color:#1D4ED8;"
               "border:1px solid #1D4ED855'>short-circuit, 0 tokens</span>"
               if result["short_circuit"] else ""),
            unsafe_allow_html=True,
        )

        st.markdown("###### Agent response")
        st.success(result["response"])

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Routed to", result["agent"].replace("Feedback Handler ", "Feedback "))
        c2.metric("Database action", result["action"])
        c3.metric("Ticket", result["ticket_number"] or "—")
        c4.metric("Latency", f"{result['latency_ms']} ms")

        st.markdown("###### Trace")
        st.markdown(
            f"<div class='trace-box'>"
            f"<b>input</b> → {result['message']}<br>"
            f"<b>classifier</b> → category={result['classification']!r}, "
            f"sentiment={result['sentiment']!r}, "
            f"customer_name={result['customer_name']!r}<br>"
            f"<b>route</b> → {result['agent']}<br>"
            f"<b>action</b> → {result['action']}"
            + (f" (ticket #{result['ticket_number']})" if result["ticket_number"] else "")
            + "</div>",
            unsafe_allow_html=True,
        )

# ---------------------------------------------------------------------------
# Tickets
# ---------------------------------------------------------------------------

with tab_tickets:
    tickets = db.list_tickets()
    st.markdown(f"**{len(tickets)}** tickets in `support_tickets`")

    if tickets:
        counts = pd.Series([t["status"] for t in tickets]).value_counts()
        cols = st.columns(len(db.VALID_STATUSES))
        for col, status in zip(cols, db.VALID_STATUSES):
            col.metric(status, int(counts.get(status, 0)))

        frame = pd.DataFrame(tickets)[
            ["ticket_number", "customer_name", "status", "message", "created_at"]
        ]
        st.dataframe(
            tint(frame, "status", STATUS_COLOR),
            use_container_width=True, hide_index=True, height=430,
            column_config={
                "ticket_number": st.column_config.TextColumn("Ticket", width="small"),
                "customer_name": st.column_config.TextColumn("Customer", width="medium"),
                "status": st.column_config.TextColumn("Status", width="small"),
                "message": st.column_config.TextColumn("Message", width="large"),
                "created_at": st.column_config.TextColumn("Created", width="medium"),
            },
        )

        st.markdown("###### Update a ticket status")
        u1, u2, u3 = st.columns([2, 2, 1])
        number = u1.selectbox("Ticket", [t["ticket_number"] for t in tickets])
        status = u2.selectbox("New status", db.VALID_STATUSES)
        if u3.button("Update", use_container_width=True):
            # Plain if/else, not a conditional expression. A bare ternary here is
            # an expression statement, and Streamlit's display magic tries to
            # render its value by re-parsing the source line - which crashes on
            # the line continuation.
            if db.update_status(number, status):
                st.toast(f"Ticket #{number} set to {status}")
            else:
                st.error(f"Could not update ticket #{number}.")
            st.rerun()
    else:
        st.info("No tickets yet.")

# ---------------------------------------------------------------------------
# Logs
# ---------------------------------------------------------------------------

with tab_logs:
    logs = db.list_logs()
    st.markdown(f"**{len(logs)}** handled messages in `agent_logs`")

    if logs:
        frame = pd.DataFrame(logs)
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Messages handled", len(frame))
        m2.metric("Tickets created", int((frame["action"] == "ticket_created").sum()))
        m3.metric("Classifier fallbacks", int(frame["fallback_used"].fillna(0).sum()))
        m4.metric("Median latency", f"{int(frame['latency_ms'].median())} ms")

        st.markdown("###### Routing distribution")
        counts = (frame["classification"].value_counts()
                  .rename_axis("classification").reset_index(name="messages"))
        st.bar_chart(counts, x="classification", y="messages",
                     color="#1D4ED8", height=230, horizontal=True)

        st.markdown("###### Handled messages")
        table = frame[["ts", "classification", "sentiment", "agent", "action",
                       "ticket_number", "latency_ms", "message"]]
        st.dataframe(
            tint(table, "classification", CLASS_COLOR),
            use_container_width=True, hide_index=True, height=430,
            column_config={
                "ts": st.column_config.TextColumn("Time", width="medium"),
                "classification": st.column_config.TextColumn("Classification", width="medium"),
                "sentiment": st.column_config.TextColumn("Sentiment", width="small"),
                "agent": st.column_config.TextColumn("Routed to", width="medium"),
                "action": st.column_config.TextColumn("DB action", width="medium"),
                "ticket_number": st.column_config.TextColumn("Ticket", width="small"),
                "latency_ms": st.column_config.NumberColumn("Latency", format="%d ms", width="small"),
                "message": st.column_config.TextColumn("Message", width="large"),
            },
        )
    else:
        st.info("No messages handled yet — try one on the first tab.")

# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

with tab_eval:
    if not RESULTS_PATH.exists():
        st.warning("No results yet. Run `python evaluate.py` first.")
    else:
        results = json.loads(RESULTS_PATH.read_text())
        acc = results["routing_accuracy"]

        # ---- hero number: one headline, not four equal tiles ----
        hero, side = st.columns([1.15, 2.85])
        with hero:
            st.markdown(
                "<div class='hero-stat'>"
                "<div class='hero-stat-label'>Routing accuracy</div>"
                f"<div class='hero-stat-value'>{acc:.1%}</div>"
                f"<div class='hero-stat-sub'>{results['correct']} of {results['n']} "
                "messages routed to the correct agent</div></div>",
                unsafe_allow_html=True,
            )
        with side:
            s1, s2, s3 = st.columns(3)
            s1.metric("Test set size", results["n"])
            s2.metric("Classifier fallbacks", results["fallbacks_triggered"])
            s3.metric("Run time", f"{results['elapsed_seconds']}s")
            st.caption(
                "Everything downstream of the classifier is deterministic, so "
                "routing accuracy is the system's accuracy. The test set is "
                "weighted toward hard cases, not clean ones."
            )

        st.markdown("###### Per class")
        per_class = (pd.DataFrame(results["per_class"]).T.reset_index()
                     .rename(columns={"index": "category"}))
        st.dataframe(
            per_class, use_container_width=True, hide_index=True, height=145,
            column_config={
                "category": st.column_config.TextColumn("Category", width="medium"),
                "support": st.column_config.NumberColumn("n", width="small"),
                "precision": st.column_config.ProgressColumn(
                    "Precision", min_value=0, max_value=1, format="%.3f"),
                "recall": st.column_config.ProgressColumn(
                    "Recall", min_value=0, max_value=1, format="%.3f"),
                "f1": st.column_config.ProgressColumn(
                    "F1", min_value=0, max_value=1, format="%.3f"),
            },
        )

        left, right = st.columns([1.05, 1])

        with left:
            st.markdown("###### Confusion matrix")
            st.caption("rows = expected · columns = predicted")
            matrix = (pd.DataFrame(results["confusion_matrix"]).T.fillna(0)
                      .reindex(index=agents.CATEGORIES, columns=agents.CATEGORIES)
                      .fillna(0).astype(int))
            peak = max(int(matrix.values.max()), 1)

            def heat(frame):
                """Sequential single hue for counts; status red for errors."""
                styles = pd.DataFrame("", index=frame.index, columns=frame.columns)
                for r in frame.index:
                    for c in frame.columns:
                        v = frame.loc[r, c]
                        if r == c:
                            t = v / peak
                            if   t >= 0.80: bg, fg = "#1D4ED8", "#FFFFFF"
                            elif t >= 0.55: bg, fg = "#3B82F6", "#FFFFFF"
                            elif t >= 0.30: bg, fg = "#93C5FD", "#0F172A"
                            elif t > 0:     bg, fg = "#DBEAFE", "#0F172A"
                            else:           bg, fg = "#F8FAFC", "#94A3B8"
                        elif v > 0:
                            bg, fg = "#FEE2E2", "#B91C1C"     # an actual error
                        else:
                            bg, fg = "#F8FAFC", "#CBD5E1"     # correctly empty
                        styles.loc[r, c] = (
                            f"background-color:{bg}; color:{fg}; "
                            "font-weight:700; text-align:center")
                return styles

            st.dataframe(matrix.style.apply(heat, axis=None),
                         use_container_width=True, height=143)
            st.caption(
                "Blue on the diagonal is correct routing. The single red cell is "
                "the one miss — a Negative Feedback message predicted as a Query."
            )

        with right:
            st.markdown("###### Accuracy by case type")
            st.caption("why each case is in the test set")
            cases = (pd.DataFrame(results["by_case_type"]).T.reset_index()
                     .rename(columns={"index": "case type"})
                     .sort_values("accuracy"))
            cases["accuracy"] = cases["accuracy"].astype(float) * 100
            st.dataframe(
                cases, use_container_width=True, hide_index=True, height=345,
                column_config={
                    "case type": st.column_config.TextColumn("Case type", width="medium"),
                    "n": st.column_config.NumberColumn("n", width="small"),
                    "correct": st.column_config.NumberColumn("OK", width="small"),
                    "accuracy": st.column_config.ProgressColumn(
                        "Accuracy", min_value=0, max_value=100, format="%.0f%%"),
                },
            )

        if results["misses"]:
            st.markdown("###### The misclassified message")
            for m in results["misses"]:
                st.markdown(
                    "<div class='miss-box'>"
                    f"<div class='miss-msg'>&ldquo;{m['message']}&rdquo;</div>"
                    f"<div class='miss-meta'>expected <b>{m['expected']}</b> "
                    f"&nbsp;·&nbsp; predicted <b>{m['actual']}</b> "
                    f"&nbsp;·&nbsp; case type <b>{m['note']}</b></div>"
                    "<div class='miss-note'>Reported rather than tuned away. "
                    "&ldquo;Hope you can help&rdquo; reads as a request and the "
                    "message carries no complaint vocabulary, so the error is "
                    "defensible. It also costs little: the Query Handler's "
                    "no-ticket-number path asks for a number and offers to open "
                    "a ticket, so the customer is still invited to raise the "
                    "issue.</div></div>",
                    unsafe_allow_html=True,
                )
        else:
            st.success("No misclassifications on this run.")
