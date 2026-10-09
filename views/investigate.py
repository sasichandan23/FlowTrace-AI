import plotly.express as px
import streamlit as st

from core import auth, ui
from core import graph as G
from core.config import RISK_COLORS
from core.model import RULES, VERDICTS, explain_rows
from core.report import build_report
from core.state import get_pipeline, scope_banks, scope_label, scoped

emp = auth.require("investigate")
pipe = get_pipeline()
cust = scoped(pipe.customers, emp)

ui.page_header("Customer investigation", "Full history, ML explanation and AML verdict for one customer",
               scope_label(emp))

options = list(cust.customer_id)
pre = st.session_state.get("investigate_customer")
idx = options.index(pre) if pre in options else 0
lookup = cust.set_index("customer_id")
cid = st.selectbox("Customer (sorted by risk)", options, index=idx,
                   format_func=lambda c: f"{c} · {lookup.loc[c, 'risk_band']} risk · score {lookup.loc[c, 'customer_risk']:.1f}")
st.session_state["investigate_customer"] = cid

grant = auth.access_grant(cid)
if not grant:
    ui.confidential("Customer transaction history is confidential. Record a case reference and the purpose of "
                    "access before viewing. This access is written to the audit log.")
    with st.form("grant"):
        c1, c2 = st.columns([1, 2])
        case_ref = c1.text_input("Case / alert reference", placeholder="e.g. AML-2026-0142")
        reason = c2.selectbox("Purpose", ["Alert triage", "Suspicious activity investigation",
                                          "SAR preparation", "Regulatory request", "Quality assurance review"])
        ok = st.form_submit_button("Confirm and view history", type="primary")
    if ok:
        if len(case_ref.strip()) < 4:
            st.error("Enter a valid case or alert reference (at least 4 characters).")
        else:
            auth.grant_access(cid, case_ref.strip(), reason)
            st.rerun()
    st.stop()

c = lookup.loc[cid].copy()
c["customer_id"] = cid
txns = pipe.scored[pipe.scored.customer_id == cid].copy()
reasons = explain_rows(pipe, txns.index.tolist())

# linked entities
accts, devs = sorted(txns.account_id.unique()), sorted(txns.device_id.unique())
linked_rows = pipe.scored[(pipe.scored.account_id.isin(accts) | pipe.scored.device_id.isin(devs))
                          & (pipe.scored.customer_id != cid)]
linked_c = linked_rows.groupby("customer_id")["bank_id"].first()
visible = scope_banks(emp)
high_ids = set(pipe.customers.loc[pipe.customers.risk_band == "High", "customer_id"])
linked = {
    "accounts": accts, "devices": devs, "shared_customers": int(len(linked_c)),
    "external_customers": int((~linked_c.isin(visible)).sum()),
    "high_risk_neighbours": int(len(set(linked_c.index) & high_ids)),
}

ui.confidential(f"Access granted under case {grant['case_ref']} ({grant['reason']}) at {grant['at']} – logged.")
st.write("")
band = c["risk_band"]
ui.verdict_box(band, VERDICTS[band],
               f"Customer risk score {c['customer_risk']:.1f}/100 · {int(c['high_risk_txns'])} high-risk of "
               f"{int(c['transactions'])} transaction(s) · signals: {c['rules_triggered'] or 'none'}")
fraud = c["ml_fraud_max"]
fraud_band = "High" if fraud >= 60 else "Medium" if fraud >= 40 else "Low"
ui.kpi_row([
    ("ML AML score (max)", f"{c['ml_aml_max']:.1f}", "Random Forest, out-of-fold", "info"),
    ("Fraud model (max)", f"{fraud:.1f}", f"{fraud_band} fraud risk", {"High": "danger", "Medium": "warning", "Low": "success"}[fraud_band]),
    ("Total amount", f"{c['total_amount']:,.2f}", f"{int(c['transactions'])} transactions", "info"),
    ("Network exposure", f"{linked['shared_customers']}", f"linked customers · {linked['high_risk_neighbours']} high-risk", "warning" if linked["high_risk_neighbours"] else "info"),
])

t1, t2, t3, t4 = st.tabs(["🧠 Why flagged", "📜 Transaction history", "🕸️ Network", "📄 Report"])
with t1:
    a, b = st.columns(2)
    with a:
        st.markdown("##### AML typologies detected")
        hits = [(r, int(txns[f"rule_{r.id}"].sum())) for r in RULES]
        hits = [(r, n) for r, n in hits if n]
        if not hits:
            st.success("No AML typology rules triggered.")
        for r, n in hits:
            st.markdown(f"{ui.badge(r.typology, 'danger' if r.weight >= 20 else 'warning')} **{r.name}** – "
                        f"{n} txn(s)  \n<span style='color:#64748B'>{r.description}</span>", unsafe_allow_html=True)
    with b:
        st.markdown("##### What pushed the ML model towards *High*")
        if reasons.empty:
            st.success("No factor pushed the model towards high risk.")
        else:
            fig = px.bar(reasons[::-1], x="impact", y="factor", orientation="h", hover_data=["value"],
                         color_discrete_sequence=[RISK_COLORS["High"]])
            fig.update_layout(xaxis_title=f"contribution to P(High) · {reasons['method'].iloc[0]}", yaxis_title=None)
            st.plotly_chart(ui.style_fig(fig, 300, legend=False), width="stretch")
            st.dataframe(reasons[["factor", "value"]], hide_index=True, width="stretch")
with t2:
    hist = txns.sort_values("transaction_time")
    fig = px.scatter(hist, x="transaction_time", y="amount", color="risk_band", size="risk_score",
                     color_discrete_map=RISK_COLORS, hover_data=["transaction_id", "account_id", "rules_triggered"],
                     title="Transactions over time (size = risk score)")
    st.plotly_chart(ui.style_fig(fig, 320), width="stretch")
    st.dataframe(hist[["transaction_id", "transaction_time", "amount", "merchant_category", "payment_method",
                       "country", "account_id", "device_id", "merchant_id", "risk_score", "risk_band", "aml_risk",
                       "rules_triggered"]],
                 hide_index=True, width="stretch",
                 column_config={"risk_score": st.column_config.ProgressColumn("Risk", min_value=0, max_value=100, format="%.1f"),
                                "transaction_time": st.column_config.DatetimeColumn("Time", format="YYYY-MM-DD HH:mm")})
with t3:
    @st.cache_resource(show_spinner=False)
    def entity_graph():
        return G.build_entity_graph(get_pipeline().scored)
    EG = entity_graph()
    center = G.node_id("customer", cid)
    H = G.local_graph(EG, center, radius=2, max_nodes=70)
    st.plotly_chart(G.entity_figure(H, center, visible, height=520), width="stretch",
                    config={"scrollZoom": True, "displaylogo": False})
    st.caption(f"Accounts: {', '.join(accts)} · Devices: {', '.join(devs)} · "
               f"{linked['external_customers']} linked customer(s) at other banks are masked.")
with t4:
    html = build_report(c, txns, reasons, linked, emp, grant)
    st.markdown("Download a self-contained, printable report (open in a browser → *Print → Save as PDF*).")
    st.download_button("⬇️ Download investigation report (HTML)", html.encode("utf-8"),
                       f"FlowTrace_report_{cid}.html", "text/html", type="primary",
                       on_click=lambda: auth.audit("REPORT_DOWNLOAD", emp["employee_id"], cid, grant["case_ref"]))
    with st.expander("Preview report"):
        st.components.v1.html(html, height=700, scrolling=True)
