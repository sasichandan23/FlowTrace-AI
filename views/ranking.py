import pandas as pd
import plotly.express as px
import streamlit as st

from core import auth, ui
from core.config import BAND_THRESHOLDS, COLORS, RISK_ORDER, SCORE_WEIGHTS
from core.model import RULES, account_summary, feature_label
from core.state import get_pipeline, scope_label, scoped

emp = auth.require("ranking")
pipe = get_pipeline()
df = scoped(pipe.scored, emp)
cust = scoped(pipe.customers, emp)
acc = account_summary(df)

ui.page_header("Risk ranking", "Baseline Random Forest on aml_risk labels + AML typology rules + anomaly score",
               scope_label(emp))

w = SCORE_WEIGHTS
ui.note(f"Risk score = {int(w['ml'] * 100)}% Random Forest (aml_risk) + {int(w['rules'] * 100)}% rule signals + "
        f"{int(w['anomaly'] * 100)}% Isolation-Forest anomaly · High ≥ {BAND_THRESHOLDS['High']:.0f}, "
        f"Medium ≥ {BAND_THRESHOLDS['Medium']:.0f}. Every transaction is scored out-of-fold (by a model that never saw it).")
st.write("")

f1, f2, f3 = st.columns([1, 1.6, 1])
bands = f1.multiselect("Risk band", RISK_ORDER, default=["High", "Medium"])
rule_names = [r.name for r in RULES]
rules_sel = f2.multiselect("Must include signal", rule_names)
search = f3.text_input("Search ID", placeholder="C1005, A3205…").strip().upper()


def filt(d: pd.DataFrame, id_cols: list[str]) -> pd.DataFrame:
    if bands:
        d = d[d.risk_band.isin(bands)]
    for r in rules_sel:
        d = d[d.rules_triggered.str.contains(r, regex=False)]
    if search:
        mask = False
        for c in id_cols:
            mask = mask | d[c].astype(str).str.upper().str.contains(search, regex=False)
        d = d[mask]
    return d


score_cfg = lambda label: st.column_config.ProgressColumn(label, min_value=0, max_value=100, format="%.1f")  # noqa: E731
tab_c, tab_a, tab_t, tab_m = st.tabs(["👤 Customers", "🏦 Accounts", "💸 Transactions", "📈 Model performance"])

with tab_c:
    view = filt(cust, ["customer_id"])
    st.caption(f"{len(view)} of {len(cust)} customers")
    show = view[["customer_id", "customer_risk", "risk_band", "verdict", "transactions", "high_risk_txns",
                 "total_amount", "accounts", "devices", "rules_triggered"]]
    event = st.dataframe(show, hide_index=True, width="stretch", height=460,
                         on_select="rerun", selection_mode="single-row",
                         column_config={"customer_risk": score_cfg("Risk score"),
                                        "total_amount": st.column_config.NumberColumn("Total amount", format="%.2f"),
                                        "rules_triggered": st.column_config.TextColumn("Signals", width="large")})
    sel = event.selection.rows if event else []
    if sel:
        cid = show.iloc[sel[0]]["customer_id"]
        if auth.has_perm(emp, "investigate"):
            if st.button(f"🔎 Open investigation for {cid}", type="primary"):
                st.session_state["investigate_customer"] = cid
                st.switch_page("views/investigate.py")
        else:
            st.info("Customer history and investigation reports require the AML Investigator role.")

with tab_a:
    view = filt(acc, ["account_id"])
    st.caption(f"{len(view)} of {len(acc)} accounts · accounts shared by many customers are a mule-account signal")
    st.dataframe(view[["account_id", "account_risk", "risk_band", "customers", "devices", "merchants",
                       "transactions", "total_amount", "circular_transfers", "rules_triggered"]],
                 hide_index=True, width="stretch", height=460,
                 column_config={"account_risk": score_cfg("Risk score"),
                                "total_amount": st.column_config.NumberColumn("Total amount", format="%.2f"),
                                "rules_triggered": st.column_config.TextColumn("Signals", width="large")})

with tab_t:
    view = filt(df, ["transaction_id", "customer_id", "account_id"])
    st.caption(f"{len(view)} of {len(df)} transactions")
    st.dataframe(view[["transaction_id", "customer_id", "account_id", "transaction_time", "amount", "risk_score",
                       "risk_band", "ml_aml_score", "rule_score", "anomaly_score", "aml_risk", "rules_triggered"]],
                 hide_index=True, width="stretch", height=460,
                 column_config={"risk_score": score_cfg("Risk score"),
                                "ml_aml_score": st.column_config.NumberColumn("ML score", format="%.1f"),
                                "rule_score": st.column_config.NumberColumn("Rule score", format="%.0f"),
                                "anomaly_score": st.column_config.NumberColumn("Anomaly", format="%.1f"),
                                "aml_risk": st.column_config.TextColumn("Label (aml_risk)"),
                                "transaction_time": st.column_config.DatetimeColumn("Time", format="YYYY-MM-DD HH:mm"),
                                "rules_triggered": st.column_config.TextColumn("Signals", width="large")})

with tab_m:
    m, hm = pipe.aml.metrics, pipe.extras["hybrid_metrics"]
    st.caption("Evaluated with 5-fold stratified cross-validation on all banks' cleaned data "
               f"({m['n_samples']} transactions). Labels: dataset column aml_risk.")
    ui.kpi_row([
        ("RF accuracy", f"{m['accuracy']:.1%}", "baseline Random Forest", "info"),
        ("RF macro-F1", f"{m['macro_f1']:.3f}", "balanced across Low/Medium/High", "info"),
        ("Hybrid macro-F1", f"{hm['macro_f1']:.3f}", "RF + rules + anomaly", "success"),
        ("Fraud RF macro-F1", f"{pipe.fraud.metrics['macro_f1']:.3f}", "secondary model (fraud_risk)", "warning"),
    ])
    c1, c2 = st.columns(2)
    with c1:
        fig = px.imshow(m["confusion"], text_auto=True, color_continuous_scale=["#EFF4FF", COLORS["info"]],
                        labels=dict(x="Predicted", y="Actual (aml_risk)", color="Count"),
                        title="Random Forest confusion matrix (out-of-fold)")
        st.plotly_chart(ui.style_fig(fig, 360, legend=False), width="stretch")
    with c2:
        imp = pipe.aml.importances.head(12)[::-1]
        fig = px.bar(x=imp.values, y=[feature_label(i) for i in imp.index], orientation="h",
                     title="Top features (Random Forest importance)", color_discrete_sequence=[COLORS["info"]])
        fig.update_layout(xaxis_title="importance", yaxis_title=None)
        st.plotly_chart(ui.style_fig(fig, 360, legend=False), width="stretch")
    rep = m["report"].loc[["Low", "Medium", "High"], ["precision", "recall", "f1-score", "support"]].round(3)
    st.markdown("**Per-class results (Random Forest)**")
    st.dataframe(rep, width="stretch")
    st.markdown("**AML typology rules** – hit rates on labelled data")
    st.dataframe(pipe.rule_stats.rename(columns={"share_of_labelled_high": "% of High-labelled hit",
                                                 "share_of_labelled_low": "% of Low-labelled hit"}),
                 hide_index=True, width="stretch")

st.download_button("⬇️ Download ranked results (CSV)",
                   cust.to_csv(index=False).encode(), "flowtrace_ranked_customers.csv", "text/csv",
                   on_click=lambda: auth.audit("EXPORT", emp["employee_id"], "ranked_customers.csv", scope_label(emp)))
