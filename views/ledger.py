import pandas as pd
import streamlit as st

from core import auth, ui
from core import graph as G
from core.state import get_pipeline, scope_label

emp = auth.require("ledger")
pipe = get_pipeline()

ui.page_header("Transfer graph", "Optional directed path & cycle analysis on a sender → receiver transfer ledger",
               scope_label(emp))
ui.note("The FINSIGHT dataset has no sender/receiver columns, so directed fund tracing needs a transfer ledger. "
        "Upload one with columns sender_account, receiver_account, amount (timestamp optional). "
        "It is validated before analysis.")
st.write("")

template = pd.DataFrame({"sender_account": ["A3001", "A3002"], "receiver_account": ["A3002", "A3003"],
                         "amount": [5000.0, 4800.0], "timestamp": ["2025-02-01 10:00", "2025-02-01 11:30"]})
c1, c2, c3 = st.columns([2, 1, 1])
up = c1.file_uploader("Transfer ledger (CSV)", type=["csv"])
c2.download_button("⬇️ Ledger template", template.to_csv(index=False).encode(), "ledger_template.csv", "text/csv")
if c3.button("Load synthetic demo ledger"):
    st.session_state["ledger_raw"] = G.synthetic_demo_ledger(pipe.scored.account_id.unique().tolist())
    st.session_state["ledger_src"] = "synthetic"
if up is not None:
    try:
        st.session_state["ledger_raw"] = pd.read_csv(up)
        st.session_state["ledger_src"] = up.name
        auth.audit("LEDGER_UPLOAD", emp["employee_id"], up.name)
    except Exception as e:  # noqa: BLE001
        st.error(f"Could not read CSV: {e}")

raw = st.session_state.get("ledger_raw")
if raw is None:
    st.info("Upload a ledger or load the synthetic demo ledger to start.")
    st.stop()

if st.session_state.get("ledger_src") == "synthetic":
    st.warning("⚠️ SYNTHETIC DEMO DATA – generated to demonstrate the analysis (a planted 4-account loop and a "
               "5-account layering chain). It is not observed customer data.")

ledger, errors, warnings = G.validate_ledger(raw)
for w in warnings:
    st.caption(f"⚠️ {w}")
if errors:
    for e in errors:
        st.error(e)
    st.stop()

DG = G.ledger_graph(ledger)
cycles = G.find_cycles(DG)
flows = G.flow_stats(DG)
ui.kpi_row([
    ("Valid transfers", f"{len(ledger):,}", f"{ui.money(ledger.amount.sum())} total", "info"),
    ("Accounts", f"{DG.number_of_nodes():,}", f"{DG.number_of_edges():,} directed links", "info"),
    ("Cycles found", f"{len(cycles):,}", "funds returning to origin (≤ 6 hops)", "danger" if len(cycles) else "success"),
    ("Pass-through accounts", f"{int(flows.pass_through_flag.sum()):,}", "money in ≈ money out (layering)",
     "warning" if flows.pass_through_flag.sum() else "success"),
])

t1, t2, t3 = st.tabs(["🔁 Cycles", "🧭 Path tracer", "↔️ Pass-through accounts"])
with t1:
    if cycles.empty:
        st.success("No circular flows detected.")
        st.plotly_chart(G.ledger_figure(DG), width="stretch", config={"scrollZoom": True})
    else:
        st.dataframe(cycles.drop(columns="accounts"), hide_index=True, width="stretch", height=220)
        i = st.selectbox("Highlight cycle", range(len(cycles)), format_func=lambda k: cycles.cycle[k])
        st.plotly_chart(G.ledger_figure(DG, cycles.accounts[i], closed=True), width="stretch",
                        config={"scrollZoom": True})
with t2:
    nodes = sorted(DG.nodes)
    a, b, c = st.columns([1, 1, 0.6])
    src = a.selectbox("From account", nodes)
    dst = b.selectbox("To account", nodes, index=min(1, len(nodes) - 1))
    cutoff = c.slider("Max hops", 2, 8, 6)
    paths = G.find_paths(DG, src, dst, cutoff)
    if paths.empty:
        st.info("No directed path between these accounts within the hop limit.")
        st.plotly_chart(G.ledger_figure(DG, [src, dst]), width="stretch", config={"scrollZoom": True})
    else:
        st.dataframe(paths.drop(columns="accounts"), hide_index=True, width="stretch", height=200)
        st.plotly_chart(G.ledger_figure(DG, paths.accounts[0]), width="stretch", config={"scrollZoom": True})
with t3:
    st.dataframe(flows, hide_index=True, width="stretch",
                 column_config={"pass_through_ratio": st.column_config.ProgressColumn("Pass-through", min_value=0,
                                                                                      max_value=1, format="%.2f")})
