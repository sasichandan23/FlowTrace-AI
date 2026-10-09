"""Employee authentication, role-based access control and audit logging.

* Passwords are stored only as salted PBKDF2-HMAC-SHA256 hashes (config/employees.json).
* Each employee belongs to one bank and only sees that bank's customers.
* Roles gate pages: analysts see aggregates & rankings, investigators can open a
  customer's full history (with a recorded case justification), the group
  compliance officer (admin) sees all banks and the audit log.
* Failed logins lock the ID for a few minutes; idle sessions expire.
* Every login, history access, report and export is written to logs/audit.log.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import secrets
import threading
import time
from datetime import datetime

import streamlit as st

from .config import (AUDIT_LOG, EMPLOYEES_FILE, LOCKOUT_MIN, LOG_DIR, MAX_FAILED_LOGINS,
                     PBKDF2_ITERATIONS, SESSION_TIMEOUT_MIN)

ROLE_LABELS = {
    "analyst": "AML Analyst",
    "investigator": "AML Investigator",
    "admin": "Group Compliance Officer",
}
BASE = {"overview", "ranking", "network", "quality", "exports"}
ROLE_PERMS = {
    "analyst": BASE,
    "investigator": BASE | {"investigate", "ledger"},
    "admin": BASE | {"investigate", "ledger", "audit", "all_banks"},
}

_lock = threading.Lock()


# --------------------------------------------------------------------------- hashing
def hash_password(password: str, salt_hex: str | None = None) -> tuple[str, str]:
    salt_hex = salt_hex or secrets.token_hex(16)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode(), bytes.fromhex(salt_hex), PBKDF2_ITERATIONS)
    return salt_hex, dk.hex()


def load_employees() -> dict[str, dict]:
    if not EMPLOYEES_FILE.exists():
        return {}
    data = json.loads(EMPLOYEES_FILE.read_text(encoding="utf-8"))
    return {e["employee_id"].upper(): e for e in data}


# --------------------------------------------------------------------------- audit
def audit(action: str, employee_id: str = "-", target: str = "", detail: str = "") -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    entry = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "employee_id": employee_id,
        "action": action,
        "target": target,
        "detail": detail,
    }
    with _lock, open(AUDIT_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


def read_audit() -> list[dict]:
    if not AUDIT_LOG.exists():
        return []
    out = []
    for line in AUDIT_LOG.read_text(encoding="utf-8").splitlines():
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


# --------------------------------------------------------------------------- lockout
@st.cache_resource
def _login_tracker() -> dict:
    return {}


def _locked_until(emp_id: str) -> float:
    rec = _login_tracker().get(emp_id)
    return rec["locked_until"] if rec else 0.0


def _register_failure(emp_id: str) -> int:
    tracker = _login_tracker()
    with _lock:
        rec = tracker.setdefault(emp_id, {"fails": 0, "locked_until": 0.0})
        rec["fails"] += 1
        if rec["fails"] >= MAX_FAILED_LOGINS:
            rec["locked_until"] = time.time() + LOCKOUT_MIN * 60
            rec["fails"] = 0
            return 0
        return MAX_FAILED_LOGINS - rec["fails"]


def _reset_failures(emp_id: str) -> None:
    _login_tracker().pop(emp_id, None)


# --------------------------------------------------------------------------- session
def authenticate(employee_id: str, password: str) -> tuple[dict | None, str]:
    emp_id = employee_id.strip().upper()
    if not emp_id or not password:
        return None, "Enter your employee ID and password."
    if _locked_until(emp_id) > time.time():
        mins = int((_locked_until(emp_id) - time.time()) // 60) + 1
        audit("LOGIN_BLOCKED", emp_id, detail="account locked")
        return None, f"Too many failed attempts. This ID is locked for {mins} more minute(s)."
    emp = load_employees().get(emp_id)
    # Always hash, even for unknown IDs, so timing does not reveal valid IDs.
    salt = emp["salt"] if emp else "00" * 16
    _, digest = hash_password(password, salt)
    if emp and emp.get("active", True) and hmac.compare_digest(digest, emp["password_hash"]):
        _reset_failures(emp_id)
        audit("LOGIN_SUCCESS", emp_id, detail=emp["bank_id"])
        return {k: v for k, v in emp.items() if k not in ("salt", "password_hash")}, ""
    left = _register_failure(emp_id)
    audit("LOGIN_FAILED", emp_id)
    if left == 0:
        return None, f"Too many failed attempts. This ID is locked for {LOCKOUT_MIN} minutes."
    return None, f"Invalid employee ID or password. {left} attempt(s) left before lockout."


def current_employee() -> dict | None:
    return st.session_state.get("employee")


def login(emp: dict) -> None:
    st.session_state["employee"] = emp
    st.session_state["last_active"] = time.time()
    st.session_state["access_grants"] = {}


def logout(reason: str = "LOGOUT") -> None:
    emp = current_employee()
    if emp:
        audit(reason, emp["employee_id"])
    for k in list(st.session_state.keys()):
        del st.session_state[k]


def enforce_timeout() -> bool:
    """Return False (and log out) if the session has been idle too long."""
    last = st.session_state.get("last_active", 0)
    if time.time() - last > SESSION_TIMEOUT_MIN * 60:
        logout("SESSION_TIMEOUT")
        st.session_state["flash"] = "Your session expired after inactivity. Please sign in again."
        return False
    st.session_state["last_active"] = time.time()
    return True


def has_perm(emp: dict | None, perm: str) -> bool:
    return bool(emp) and perm in ROLE_PERMS.get(emp.get("role", ""), set())


def require(perm: str) -> dict:
    """Page guard: stop rendering unless the signed-in employee holds `perm`."""
    emp = current_employee()
    if not emp:
        st.error("Please sign in.")
        st.stop()
    if not has_perm(emp, perm):
        audit("ACCESS_DENIED", emp["employee_id"], target=perm)
        st.error("You are not authorised to view this page. The attempt has been logged.")
        st.stop()
    return emp


# Per-customer access grants (investigator must state a case reference first)
def grant_access(customer_id: str, case_ref: str, reason: str) -> None:
    emp = current_employee()
    st.session_state.setdefault("access_grants", {})[customer_id] = {
        "case_ref": case_ref, "reason": reason, "at": datetime.now().isoformat(timespec="seconds"),
    }
    audit("CUSTOMER_HISTORY_ACCESS", emp["employee_id"], target=customer_id,
          detail=f"case={case_ref}; reason={reason}")


def access_grant(customer_id: str) -> dict | None:
    return st.session_state.get("access_grants", {}).get(customer_id)
