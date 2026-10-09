import pandas as pd
import plotly.express as px
import streamlit as st

from core import auth, ui
from core.config import COLORS

emp = auth.require("audit")
auth.audit("AUDIT_LOG_VIEW", emp["employee_id"])

ui.page_header("Audit log", "Every sign-in, access to customer history, report and export", "Group compliance")

log = pd.DataFrame(auth.read_audit())
if log.empty:
    st.info("No audit events yet.")
    st.stop()
log["timestamp"] = pd.to_datetime(log["timestamp"])
log = log.sort_values("timestamp", ascending=False)

sensitive = {"CUSTOMER_HISTORY_ACCESS", "REPORT_DOWNLOAD", "EXPORT"}
ui.kpi_row([
    ("Events", f"{len(log):,}", "total recorded", "info"),
    ("Failed sign-ins", f"{int((log.action == 'LOGIN_FAILED').sum()):,}",
     f"{int((log.action == 'LOGIN_BLOCKED').sum())} blocked by lockout", "danger"),
    ("History accesses", f"{int((log.action == 'CUSTOMER_HISTORY_ACCESS').sum()):,}", "with case reference", "warning"),
    ("Exports & reports", f"{int(log.action.isin({'EXPORT', 'REPORT_DOWNLOAD'}).sum()):,}", "files downloaded", "info"),
])

c1, c2 = st.columns([1, 2])
actions = c1.multiselect("Action", sorted(log.action.unique()))
who = c2.multiselect("Employee", sorted(log.employee_id.unique()))
view = log
if actions:
    view = view[view.action.isin(actions)]
if who:
    view = view[view.employee_id.isin(who)]

counts = view.groupby("action").size().reset_index(name="events").sort_values("events")
color = {a: COLORS["danger"] if "FAIL" in a or "BLOCK" in a or "DENIED" in a else
         COLORS["warning"] if a in sensitive else COLORS["info"] for a in counts.action}
fig = px.bar(counts, x="events", y="action", orientation="h", color="action", color_discrete_map=color)
st.plotly_chart(ui.style_fig(fig, 280, legend=False), width="stretch")
st.dataframe(view, hide_index=True, width="stretch", height=420,
             column_config={"timestamp": st.column_config.DatetimeColumn("Time", format="YYYY-MM-DD HH:mm:ss")})
st.download_button("⬇️ Download audit log (CSV)", view.to_csv(index=False).encode(), "flowtrace_audit_log.csv",
                   "text/csv")
