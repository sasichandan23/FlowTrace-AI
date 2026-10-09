import pandas as pd
import streamlit as st

from core import auth, ui
from core import data as D
from core.model import account_summary
from core.state import get_pipeline, scope_label, scoped

emp = auth.require("exports")
pipe = get_pipeline()
df = scoped(pipe.scored, emp)
cust = scoped(pipe.customers, emp)
raw = scoped(pipe.raw, emp)

ui.page_header("Exports", "Download ranked results and data-quality reports for your scope", scope_label(emp))
ui.confidential("Exports contain confidential customer data. Handle under your bank's data-protection policy. "
                "Every download is recorded in the audit log.")
st.write("")


def log(name):
    return lambda: auth.audit("EXPORT", emp["employee_id"], name, scope_label(emp))


TXN_COLS = ["transaction_id", "customer_id", "account_id", "device_id", "merchant_id", "bank_id", "transaction_time",
            "amount", "merchant_category", "payment_method", "country", "risk_score", "risk_band", "ml_aml_score",
            "ml_aml_pred", "ml_fraud_score", "rule_score", "anomaly_score", "aml_risk", "fraud_risk", "rules_triggered"]

dq = D.quality_report(raw.drop(columns="bank_id"))
dup = D.duplicate_report(raw.drop(columns="bank_id"))
dq_csv = pd.concat([dq.assign(section="column_profile_raw"),
                    pd.DataFrame([{"section": "duplicates", "column": "transaction_id", "metric": k, "value": v}
                                  for k, v in dup.items() if k != "conflicting_rows"]),
                    pipe.cleaning_log.assign(section="cleaning_log")], ignore_index=True)

items = [
    ("Ranked customers", "One row per customer: risk score, band, verdict, signals.", cust,
     "flowtrace_ranked_customers.csv"),
    ("Ranked transactions", "Every transaction with ML, rule and anomaly scores.", df[TXN_COLS],
     "flowtrace_ranked_transactions.csv"),
    ("Ranked accounts", "Account-level risk and association counts.", account_summary(df),
     "flowtrace_ranked_accounts.csv"),
    ("Data-quality report", "Column profile, duplicate-ID checks and cleaning log.", dq_csv,
     "flowtrace_data_quality.csv"),
    ("Cleaned dataset", "The cleaned FINSIGHT data for your scope.", scoped(pipe.clean, emp),
     "flowtrace_clean_data.csv"),
]
for title, desc, data, fname in items:
    a, b, c = st.columns([2, 1, 1])
    a.markdown(f"**{title}**  \n<span style='color:#64748B'>{desc}</span>", unsafe_allow_html=True)
    b.markdown(f"{len(data):,} rows")
    c.download_button("⬇️ Download CSV", data.to_csv(index=False).encode(), fname, "text/csv",
                      key=fname, on_click=log(fname), width="stretch")
    st.divider()
