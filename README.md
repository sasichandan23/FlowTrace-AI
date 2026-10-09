# FlowTrace AI – Intelligent AML Network Analysis

Data Glitch 2026 · Track 01 FINSIGHT · Problem Statement 3.
A confidential web application for bank AML staff: it cleans the FINSIGHT
dataset, trains risk models, ranks suspicious customers/accounts, maps their
networks and produces explainable investigation reports.

## Run

```bash
pip install -r requirements.txt
python scripts/seed_employees.py      # creates demo employees (first time only)
streamlit run app.py
```

Open http://localhost:8501 and sign in with an ID from `DEMO_CREDENTIALS.md`.
The first sign-in trains the models (~30 s); later runs load them from `data/.cache/`.

## What's inside

| Page | Who | What |
|---|---|---|
| Overview | all | Transaction / customer / account counts, risk-label distribution, weekly activity, hour/category/method/country charts |
| Risk ranking | all | Ranked customers, accounts, transactions; filters; model performance (confusion matrix, per-class F1, feature importance); rule hit rates |
| Network analysis | all | Interactive local graph of account ↔ customer / device / merchant links; most-shared accounts and devices |
| Customer investigation | investigator, admin | Case-reference gate → AML verdict, typologies, SHAP reasons, history, network, downloadable report |
| Transfer graph | investigator, admin | Optional: upload a sender → receiver ledger → validation, cycle detection, path tracing, pass-through (layering) accounts |
| Data quality | all | Missing / unique values, duplicate transaction-ID checks, before-vs-after cleaning, cleaning log, CSV report |
| Exports | all | Ranked customers / transactions / accounts, data-quality report, cleaned data (CSV) |
| Audit log | admin | Every sign-in, failed sign-in, history access, report and export |

## Models

* **Baseline Random Forest** (300 trees, class-balanced) trained on the dataset's `aml_risk` labels; 5-fold
  stratified CV. Every transaction is scored *out-of-fold* (by a model that never saw it).
* **AML typology rules**: circular transfer, shell-account link, rapid movement, beneficiary fan-out, mule
  account, shared-device ring, new-account high value, abnormal amount, repeated failed attempts.
* **Isolation Forest** anomaly score for unseen patterns.
* **Risk score** = 70 % RF + 25 % rules + 5 % anomaly. Transaction bands High ≥ 45, Medium ≥ 27;
  customer score = 0.7 × max + 0.3 × mean, High ≥ 55 (or ≥ 2 high-risk transactions), Medium ≥ 40.
* **SHAP** explains which factors pushed each customer towards High risk.
* A second Random Forest on `fraud_risk` is shown as a supporting fraud signal.

## Security & confidentiality

* Passwords stored only as salted PBKDF2-SHA256 hashes (200k iterations); constant-time comparison.
* Each employee sees only their own bank's customers; other banks' customers are masked in graphs.
* Role-based pages – unauthorised pages are not even registered for that role.
* Customer history requires a case reference + purpose, recorded in the audit log.
* 5 failed sign-ins lock the ID for 5 minutes; sessions expire after 15 minutes idle.
* Reports carry a confidentiality / anti-tipping-off banner.

**Demo note:** the dataset has no bank column, so customers are assigned to three fictional banks by
customer number (mod 3). The synthetic demo ledger on the Transfer graph page is generated, not observed data.
