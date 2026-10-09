import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from core import auth, ui
from core.config import COLORS, RISK_COLORS, RISK_ORDER
from core.state import get_pipeline, scope_label, scoped

emp = auth.require("overview")
pipe = get_pipeline()
df = scoped(pipe.scored, emp)
cust = scoped(pipe.customers, emp)

ui.page_header("Overview", "Transaction activity and AML risk across your bank's customers", scope_label(emp))

if df.empty:
    st.info("No transactions in scope.")
    st.stop()

ui.kpi_row([
    ("Transactions", f"{len(df):,}", f"{df.transaction_time.min():%d %b %Y} – {df.transaction_time.max():%d %b %Y}", "info"),
    ("Customers", f"{df.customer_id.nunique():,}", "in your access scope", "info"),
    ("Accounts", f"{df.account_id.nunique():,}", f"{df.device_id.nunique():,} devices · {df.merchant_id.nunique():,} merchants", "info"),
    ("Transaction volume", ui.money(df.amount.sum()), f"avg {df.amount.mean():,.0f} per txn", "info"),
])
st.write("")
hi_t = int((df.risk_band == "High").sum())
hi_c = int((cust.risk_band == "High").sum())
ui.kpi_row([
    ("High-risk transactions", f"{hi_t:,}", f"{100 * hi_t / len(df):.1f}% of transactions", "danger"),
    ("Customers to escalate", f"{hi_c:,}", "verdict: likely money laundering", "danger"),
    ("Under monitoring", f"{int((cust.risk_band == 'Medium').sum()):,}", "elevated-risk customers", "warning"),
    ("Circular transfers", f"{int(df.rule_circular.sum()):,}", "funds returned to origin", "warning"),
])

st.subheader("Risk-label distribution")
c1, c2 = st.columns([1.4, 1])
with c1:
    rows = []
    for col, name in [("aml_risk", "AML risk"), ("fraud_risk", "Fraud risk"), ("credit_risk", "Credit risk")]:
        vc = df[col].value_counts()
        for lab in RISK_ORDER:
            rows.append({"label": name, "risk": lab, "transactions": int(vc.get(lab, 0))})
    fig = px.bar(pd.DataFrame(rows), x="label", y="transactions", color="risk", barmode="group",
                 color_discrete_map=RISK_COLORS, category_orders={"risk": RISK_ORDER}, text="transactions",
                 title="Dataset labels (as provided)")
    fig.update_traces(textposition="outside", cliponaxis=False)
    st.plotly_chart(ui.style_fig(fig, 360), width="stretch")
with c2:
    vc = df.risk_band.value_counts().reindex(RISK_ORDER).fillna(0)
    fig = go.Figure(go.Pie(labels=vc.index, values=vc.values, hole=0.6,
                           marker=dict(colors=[RISK_COLORS[r] for r in vc.index]), sort=False,
                           textinfo="label+value"))
    fig.update_layout(title="FlowTrace model risk bands",
                      annotations=[dict(text=f"<b>{len(df)}</b><br>txns", showarrow=False, font_size=16)])
    st.plotly_chart(ui.style_fig(fig, 360), width="stretch")

st.subheader("Transaction activity")
main = df[df.time_outlier == 0]
daily = (main.set_index("transaction_time").resample("W-MON")
         .agg(transactions=("transaction_id", "count"), amount=("amount", "sum"),
              high=("risk_band", lambda s: (s == "High").sum())).reset_index())
fig = go.Figure()
fig.add_bar(x=daily.transaction_time, y=daily.amount, name="Amount", marker_color="#BFDBFE", yaxis="y2")
fig.add_scatter(x=daily.transaction_time, y=daily.transactions, name="Transactions", mode="lines",
                line=dict(color=COLORS["info"], width=2.5))
fig.add_scatter(x=daily.transaction_time, y=daily.high, name="High-risk", mode="lines+markers",
                line=dict(color=COLORS["danger"], width=2), marker=dict(size=5))
fig.update_layout(title="Weekly transactions, high-risk count and amount",
                  yaxis=dict(title="Transactions"), yaxis2=dict(title="Amount", overlaying="y", side="right",
                                                                 showgrid=False))
st.plotly_chart(ui.style_fig(fig, 380), width="stretch")
if df.time_outlier.sum():
    st.caption(f"{int(df.time_outlier.sum())} transaction(s) dated far outside the main window are excluded from this "
               "timeline and flagged in Data quality.")

c1, c2 = st.columns(2)
with c1:
    hourly = df.groupby("hour").agg(transactions=("transaction_id", "count"), avg_risk=("risk_score", "mean")).reset_index()
    fig = px.bar(hourly, x="hour", y="transactions", color="avg_risk",
                 color_continuous_scale=[COLORS["success"], COLORS["warning"], COLORS["danger"]],
                 title="Transactions by hour of day (colour = average risk score)")
    fig.update_layout(coloraxis_colorbar=dict(title="Avg risk"))
    st.plotly_chart(ui.style_fig(fig, 340), width="stretch")
with c2:
    cat = df.groupby(["merchant_category", "risk_band"]).size().reset_index(name="transactions")
    fig = px.bar(cat, y="merchant_category", x="transactions", color="risk_band", orientation="h",
                 color_discrete_map=RISK_COLORS, category_orders={"risk_band": RISK_ORDER},
                 title="Merchant category by model risk band")
    fig.update_layout(yaxis_title=None)
    st.plotly_chart(ui.style_fig(fig, 340), width="stretch")

c1, c2, c3 = st.columns(3)
with c1:
    pm = df.groupby("payment_method").agg(transactions=("transaction_id", "count"),
                                          high=("risk_band", lambda s: 100 * (s == "High").mean())).reset_index()
    fig = px.bar(pm, x="payment_method", y="transactions", title="Payment methods",
                 color_discrete_sequence=[COLORS["info"]], hover_data={"high": ":.1f"})
    fig.update_layout(xaxis_title=None)
    st.plotly_chart(ui.style_fig(fig, 320, legend=False), width="stretch")
with c2:
    ct = df.groupby(["country", "risk_band"]).size().reset_index(name="transactions")
    fig = px.bar(ct, x="country", y="transactions", color="risk_band", color_discrete_map=RISK_COLORS,
                 category_orders={"risk_band": RISK_ORDER}, title="Countries")
    fig.update_layout(xaxis_title=None)
    st.plotly_chart(ui.style_fig(fig, 320), width="stretch")
with c3:
    fig = px.histogram(df, x="amount", color="aml_risk", nbins=40, log_y=True, color_discrete_map=RISK_COLORS,
                       category_orders={"aml_risk": RISK_ORDER}, title="Amount distribution by AML label")
    fig.update_layout(barmode="stack", yaxis_title="count (log)")
    st.plotly_chart(ui.style_fig(fig, 320), width="stretch")
