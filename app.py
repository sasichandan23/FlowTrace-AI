"""FlowTrace AI – entry point.  Run:  streamlit run app.py"""
import streamlit as st

st.set_page_config(page_title="FlowTrace AI", page_icon="🛡️", layout="wide", initial_sidebar_state="expanded")

from core import auth, ui  # noqa: E402
from core.config import APP_NAME, APP_TAGLINE, BANKS, SESSION_TIMEOUT_MIN  # noqa: E402

ui.inject_css()


def login_page() -> None:
    st.markdown("<div style='height:4vh'></div>", unsafe_allow_html=True)
    left, right = st.columns([1.1, 1], gap="large")
    with left:
        st.markdown(f"""
<div class="ft-login-brand">
  <div class="eyebrow">Data Glitch 2026 · FINSIGHT · PS3</div>
  <h1>🛡️ {APP_NAME}</h1>
  <p style="font-size:1.1rem;margin-top:6px">{APP_TAGLINE}</p>
  <hr style="border-color:#334155">
  <p>✔ Ranks suspicious customers and accounts with ML + AML typology rules</p>
  <p>✔ Traces money through customer–account–device–merchant networks</p>
  <p>✔ Explains every alert and generates investigation reports</p>
  <p>✔ Bank-scoped, role-based access with a full audit trail</p>
</div>""", unsafe_allow_html=True)
    with right:
        st.markdown("### Employee sign-in")
        st.caption("Use the employee ID issued by your bank. Access is limited to your bank's customers.")
        if st.session_state.get("flash"):
            st.warning(st.session_state.pop("flash"))
        with st.form("login", clear_on_submit=False):
            emp_id = st.text_input("Employee ID", placeholder="e.g. NWB-AN-101")
            pw = st.text_input("Password", type="password")
            ok = st.form_submit_button("Sign in", type="primary", width="stretch")
        if ok:
            emp, err = auth.authenticate(emp_id, pw)
            if emp:
                auth.login(emp)
                st.rerun()
            else:
                st.error(err)
        ui.confidential("Authorised bank personnel only. All sign-ins and data access are logged and monitored.")
        st.caption(f"Sessions expire after {SESSION_TIMEOUT_MIN} minutes of inactivity.")


emp = auth.current_employee()
if emp and not auth.enforce_timeout():
    st.rerun()
if not emp:
    login_page()
    st.stop()

from core.state import get_pipeline  # noqa: E402  (heavy ML imports only after sign-in)

pipe = get_pipeline()  # train / load once, shared by all pages

# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.markdown(f"## 🛡️ {APP_NAME}")
    st.markdown(f"**{emp['name']}**  \n{auth.ROLE_LABELS[emp['role']]}  \n`{emp['employee_id']}`")
    if emp["role"] == "admin":
        opts = ["ALL"] + list(BANKS)
        st.selectbox("Bank scope", opts, key="bank_scope",
                     format_func=lambda b: "All banks" if b == "ALL" else BANKS[b])
    else:
        st.markdown(f"🏦 **{BANKS[emp['bank_id']]}**")
    st.divider()

pages = {"Monitoring": [], "Investigation": [], "Data & governance": []}
P = lambda path, title, icon, **kw: st.Page(path, title=title, icon=icon, **kw)  # noqa: E731
if auth.has_perm(emp, "overview"):
    pages["Monitoring"].append(P("views/overview.py", "Overview", ":material/dashboard:", default=True))
if auth.has_perm(emp, "ranking"):
    pages["Monitoring"].append(P("views/ranking.py", "Risk ranking", ":material/leaderboard:"))
if auth.has_perm(emp, "network"):
    pages["Monitoring"].append(P("views/network.py", "Network analysis", ":material/hub:"))
if auth.has_perm(emp, "investigate"):
    pages["Investigation"].append(P("views/investigate.py", "Customer investigation", ":material/person_search:"))
if auth.has_perm(emp, "ledger"):
    pages["Investigation"].append(P("views/ledger.py", "Transfer graph", ":material/account_tree:"))
if auth.has_perm(emp, "quality"):
    pages["Data & governance"].append(P("views/quality.py", "Data quality", ":material/fact_check:"))
if auth.has_perm(emp, "exports"):
    pages["Data & governance"].append(P("views/exports.py", "Exports", ":material/download:"))
if auth.has_perm(emp, "audit"):
    pages["Data & governance"].append(P("views/audit.py", "Audit log", ":material/policy:"))

nav = st.navigation({k: v for k, v in pages.items() if v})

with st.sidebar:
    st.divider()
    if st.button("Sign out", width="stretch"):
        auth.logout()
        st.rerun()
    st.caption("🔒 Confidential – access is logged")

nav.run()
