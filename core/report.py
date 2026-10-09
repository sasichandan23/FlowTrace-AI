"""Customer investigation report (self-contained, printable HTML)."""
from __future__ import annotations

from datetime import datetime
from html import escape

import pandas as pd

from .config import APP_NAME, BANKS, COLORS, RISK_COLORS
from .model import RULES, VERDICTS

RECOMMENDATIONS = {
    "High": [
        "Escalate to the Money Laundering Reporting Officer for a Suspicious Activity Report (SAR) decision.",
        "Place enhanced due diligence (EDD) on the customer and all linked accounts.",
        "Review linked devices and shared accounts listed below for connected customers.",
        "Consider temporary restrictions on outbound transfers pending review.",
    ],
    "Medium": [
        "Add the customer to enhanced monitoring for the next 90 days.",
        "Request source-of-funds documentation for the flagged transactions.",
        "Re-score automatically on new activity; escalate if any typology repeats.",
    ],
    "Low": [
        "No action required beyond standard monitoring.",
        "Re-assess automatically if new transactions trigger any AML typology.",
    ],
}


def _table(df: pd.DataFrame) -> str:
    head = "".join(f"<th>{escape(str(c))}</th>" for c in df.columns)
    body = "".join("<tr>" + "".join(f"<td>{escape(str(v))}</td>" for v in row) + "</tr>"
                   for row in df.itertuples(index=False))
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def build_report(customer: pd.Series, txns: pd.DataFrame, reasons: pd.DataFrame,
                 linked: dict, employee: dict, grant: dict) -> str:
    band = customer["risk_band"]
    col = RISK_COLORS[band]
    now = datetime.now().strftime("%d %b %Y, %H:%M")
    rule_rows = []
    for r in RULES:
        n = int(txns[f"rule_{r.id}"].sum())
        if n:
            rule_rows.append({"Typology": r.typology, "Signal": r.name, "Transactions": n,
                              "What it means": r.description})
    typ_html = _table(pd.DataFrame(rule_rows)) if rule_rows else "<p>No AML typology rules were triggered.</p>"
    reasons_html = (_table(reasons[["factor", "value"]].rename(columns={"factor": "Factor", "value": "Customer value"}))
                    if not reasons.empty else "<p>No factor pushed the model towards high risk.</p>")
    hist = txns.sort_values("transaction_time")[[
        "transaction_id", "transaction_time", "amount", "merchant_category", "payment_method", "country",
        "account_id", "device_id", "risk_score", "risk_band", "rules_triggered"]].copy()
    hist["transaction_time"] = hist["transaction_time"].dt.strftime("%Y-%m-%d %H:%M")
    hist.columns = ["Txn ID", "Time", "Amount", "Category", "Method", "Country", "Account", "Device",
                    "Score", "Band", "Signals"]
    recs = "".join(f"<li>{escape(x)}</li>" for x in RECOMMENDATIONS[band])
    fraud_band = "High" if customer["ml_fraud_max"] >= 60 else "Medium" if customer["ml_fraud_max"] >= 40 else "Low"

    return f"""<!doctype html><html><head><meta charset="utf-8">
<title>{APP_NAME} – Investigation report {escape(customer['customer_id'])}</title>
<style>
body{{font-family:Segoe UI,Arial,sans-serif;color:{COLORS['navy']};margin:32px auto;max-width:1000px;padding:0 20px;font-size:14px}}
.conf{{background:#FEF2F2;border:1px solid #FECACA;color:{COLORS['danger']};padding:8px 12px;border-radius:8px;font-weight:700;font-size:12px;letter-spacing:.05em}}
h1{{margin:18px 0 4px}} h2{{margin-top:28px;border-bottom:2px solid #E2E8F0;padding-bottom:6px;font-size:18px}}
.meta{{color:#64748B}} .verdict{{background:{col};color:#fff;border-radius:10px;padding:16px 20px;margin-top:16px}}
.verdict .v{{font-size:22px;font-weight:700}} .grid{{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin-top:14px}}
.k{{border:1px solid #E2E8F0;border-left:4px solid {COLORS['info']};border-radius:8px;padding:10px}}
.k b{{display:block;font-size:20px}} .k span{{color:#64748B;font-size:12px;text-transform:uppercase}}
table{{border-collapse:collapse;width:100%;font-size:12px;margin-top:8px}} th{{background:#F1F5F9;text-align:left}}
th,td{{border:1px solid #E2E8F0;padding:6px 8px;vertical-align:top}} .foot{{color:#64748B;font-size:12px;margin-top:28px}}
</style></head><body>
<div class="conf">CONFIDENTIAL – FOR AUTHORISED AML/COMPLIANCE USE ONLY. DO NOT DISCLOSE TO THE CUSTOMER (tipping-off).</div>
<h1>AML Investigation Report – Customer {escape(customer['customer_id'])}</h1>
<div class="meta">{escape(BANKS.get(customer['bank_id'], customer['bank_id']))} · Generated {now} by {escape(employee['name'])} ({escape(employee['employee_id'])})
· Case ref {escape(grant.get('case_ref', '-'))} · Reason: {escape(grant.get('reason', '-'))}</div>
<div class="verdict"><div>AML VERDICT · {escape(band.upper())} RISK</div><div class="v">{escape(VERDICTS[band])}</div>
<div>Customer risk score {customer['customer_risk']:.1f}/100 · {int(customer['high_risk_txns'])} high-risk transaction(s) out of {int(customer['transactions'])}</div></div>
<div class="grid">
<div class="k"><span>Customer risk</span><b>{customer['customer_risk']:.1f}</b></div>
<div class="k"><span>ML AML score (max)</span><b>{customer['ml_aml_max']:.1f}</b></div>
<div class="k"><span>Fraud model (max)</span><b>{customer['ml_fraud_max']:.1f} · {fraud_band}</b></div>
<div class="k"><span>Total amount</span><b>{customer['total_amount']:,.2f}</b></div>
<div class="k"><span>Transactions</span><b>{int(customer['transactions'])}</b></div>
<div class="k"><span>Accounts used</span><b>{int(customer['accounts'])}</b></div>
<div class="k"><span>Devices used</span><b>{int(customer['devices'])}</b></div>
<div class="k"><span>Countries</span><b>{int(customer['countries'])}</b></div>
</div>
<h2>1. Money-laundering typologies detected</h2>{typ_html}
<h2>2. Why the ML model flagged this customer</h2>
<p>Top factors pushing the Random Forest towards <b>High</b> AML risk (method: {escape(reasons['method'].iloc[0] if not reasons.empty else '-')}).</p>{reasons_html}
<h2>3. Network exposure</h2>
<p>Accounts used: {escape(', '.join(linked['accounts']))}<br>Devices used: {escape(', '.join(linked['devices']))}<br>
Other customers sharing these accounts/devices: <b>{linked['shared_customers']}</b> (of which {linked['external_customers']} at other banks)<br>
Linked customers already rated high risk: <b>{linked['high_risk_neighbours']}</b></p>
<h2>4. Transaction history</h2>{_table(hist)}
<h2>5. Recommended actions</h2><ul>{recs}</ul>
<h2>6. Method</h2>
<p>Risk score = 70% Random Forest probability (trained on labelled <i>aml_risk</i>, scored out-of-fold) + 25% AML typology rules
+ 5% Isolation-Forest anomaly score. Transaction bands: High ≥ 45, Medium ≥ 27. Customer score = 0.7 × max + 0.3 × mean transaction score;
customer bands: High ≥ 55 (or two or more high-risk transactions), Medium ≥ 40.</p>
<div class="foot">This report is decision support produced by {APP_NAME}. The final determination on suspicious activity rests with an
authorised compliance officer. Access to this report is logged.</div>
</body></html>"""
