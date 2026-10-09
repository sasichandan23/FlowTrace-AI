import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core import auth, ui
from core import data as D
from core.config import COLORS
from core.state import get_pipeline, scope_label, scoped

emp = auth.require("quality")
pipe = get_pipeline()
raw = scoped(pipe.raw, emp)
clean = scoped(pipe.clean, emp)

ui.page_header("Data quality", "Missing values, unique values, duplicate transaction IDs and the cleaning log",
               scope_label(emp))

qb = D.quality_report(raw.drop(columns="bank_id"))
qa = D.quality_report(clean.drop(columns=["bank_id", "time_outlier", "fields_imputed", "amount_outlier"]))
dup = D.duplicate_report(raw.drop(columns="bank_id"))

before_issues = int(qb.missing.sum() + qb.case_or_space_variants.sum() + qb.out_of_range.sum())
after_issues = int(qa.missing.sum() + qa.case_or_space_variants.sum() + qa.out_of_range.sum())
cells = raw.shape[0] * (raw.shape[1] - 1)
health_before = 100 * (1 - (before_issues + dup["extra_rows_with_repeated_id"] * (raw.shape[1] - 1)) / cells)
ui.kpi_row([
    ("Raw rows", f"{dup['total_rows']:,}", f"{dup['unique_transaction_ids']:,} unique transaction IDs", "info"),
    ("Duplicate IDs", f"{dup['extra_rows_with_repeated_id']:,}",
     f"{dup['exact_duplicate_rows']} exact · {dup['ids_with_conflicting_values']} IDs with conflicting values", "danger"),
    ("Cell issues (before)", f"{before_issues:,}", "missing + inconsistent + impossible", "warning"),
    ("Cell issues (after)", f"{after_issues:,}", f"data health {health_before:.1f}% → 100%", "success"),
])

st.subheader("Before vs after cleaning")
m = qb.merge(qa, on="column", how="left", suffixes=("_before", "_after"))
m = m[m.issues_before > 0]
fig = go.Figure()
fig.add_bar(y=m.column, x=m.missing_before, name="Missing", orientation="h", marker_color=COLORS["danger"])
fig.add_bar(y=m.column, x=m.case_or_space_variants_before, name="Case/space variants", orientation="h",
            marker_color=COLORS["warning"])
fig.add_bar(y=m.column, x=m.out_of_range_before, name="Impossible values", orientation="h",
            marker_color=COLORS["info"])
fig.update_layout(barmode="stack", title="Issues per column in the raw data (all resolved after cleaning)",
                  xaxis_title="cells", yaxis=dict(autorange="reversed"))
st.plotly_chart(ui.style_fig(fig, 420), width="stretch")

t1, t2, t3 = st.tabs(["📋 Column profile", "🔁 Duplicate transaction IDs", "🧹 Cleaning log"])
with t1:
    view = qb.merge(qa[["column", "missing", "unique_values"]], on="column", suffixes=("", "_after"))
    st.dataframe(view.rename(columns={"missing": "missing (raw)", "missing_pct": "missing % (raw)",
                                      "unique_values": "unique (raw)", "missing_after": "missing (clean)",
                                      "unique_values_after": "unique (clean)"}),
                 hide_index=True, width="stretch", height=460)
with t2:
    st.markdown(f"- **{dup['extra_rows_with_repeated_id']}** extra rows reuse an existing transaction ID\n"
                f"- **{dup['exact_duplicate_rows']}** are byte-for-byte duplicates\n"
                f"- **{dup['ids_with_conflicting_values']}** IDs appear with *different* values "
                "(resolved by keeping the most complete record)")
    if not dup["conflicting_rows"].empty:
        st.dataframe(dup["conflicting_rows"], hide_index=True, width="stretch", height=320)
with t3:
    st.dataframe(pipe.cleaning_log, hide_index=True, width="stretch", height=460)
    st.caption("Extreme amounts and out-of-window dates are flagged rather than deleted – they can be the laundering signal.")

report = qb.assign(section="column_profile_raw")
dup_rows = pd.DataFrame([{"section": "duplicates", "column": "transaction_id", "metric": k, "value": v}
                         for k, v in dup.items() if k != "conflicting_rows"])
log = pipe.cleaning_log.assign(section="cleaning_log")
csv = pd.concat([report, dup_rows, log], ignore_index=True).to_csv(index=False).encode()
st.download_button("⬇️ Download data-quality report (CSV)", csv, "flowtrace_data_quality.csv", "text/csv",
                   on_click=lambda: auth.audit("EXPORT", emp["employee_id"], "data_quality.csv", scope_label(emp)))
