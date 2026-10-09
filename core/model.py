"""Risk models: baseline Random Forest on aml_risk labels, AML typology rules,
Isolation-Forest anomaly score and SHAP explanations."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from . import data as D
from .config import BAND_THRESHOLDS, ENTITY_THRESHOLDS, SCORE_WEIGHTS

LABEL_ORDER = ["Low", "Medium", "High"]
NUM_FEATURES = [
    "amount", "distance_from_home_km", "failed_attempts_24h", "transactions_1h",
    "device_age_days", "account_age_days", "income_monthly", "loan_amount",
    "credit_score", "existing_debt", "previous_defaults", "transfer_count_24h",
    "unique_beneficiaries_7d", "circular_transfer_flag", "shell_account_score",
    "hour", "is_night", "weekday", "debt_to_income", "loan_to_income", "amount_to_income",
    "customers_per_account", "customers_per_device", "accounts_per_customer",
    "devices_per_customer", "merchants_per_account", "txns_per_customer",
    "amount_vs_customer_avg", "amount_outlier", "fields_imputed",
]
CAT_FEATURES = ["merchant_category", "payment_method", "country"]

FEATURE_LABELS = {
    "amount": "Transaction amount",
    "distance_from_home_km": "Distance from home (km)",
    "failed_attempts_24h": "Failed attempts (24h)",
    "transactions_1h": "Transactions in last hour",
    "device_age_days": "Device age (days)",
    "account_age_days": "Account age (days)",
    "income_monthly": "Monthly income",
    "loan_amount": "Loan amount",
    "credit_score": "Credit score",
    "existing_debt": "Existing debt",
    "previous_defaults": "Previous defaults",
    "transfer_count_24h": "Transfers in 24h",
    "unique_beneficiaries_7d": "Unique beneficiaries (7d)",
    "circular_transfer_flag": "Circular transfer flag",
    "shell_account_score": "Shell-account score",
    "hour": "Hour of day",
    "is_night": "Night-time transaction",
    "weekday": "Day of week",
    "debt_to_income": "Debt-to-income ratio",
    "loan_to_income": "Loan-to-income ratio",
    "amount_to_income": "Amount vs monthly income",
    "customers_per_account": "Customers sharing the account",
    "customers_per_device": "Customers sharing the device",
    "accounts_per_customer": "Accounts used by customer",
    "devices_per_customer": "Devices used by customer",
    "merchants_per_account": "Merchants per account",
    "txns_per_customer": "Transactions by customer",
    "amount_vs_customer_avg": "Amount vs customer's average",
    "amount_outlier": "Extreme amount flag",
    "fields_imputed": "Missing fields (imputed)",
}


def feature_label(name: str) -> str:
    if name in FEATURE_LABELS:
        return FEATURE_LABELS[name]
    for c in CAT_FEATURES:
        if name.startswith(c + "_"):
            return f"{c.replace('_', ' ').title()} = {name[len(c) + 1:]}"
    return name


# --------------------------------------------------------------------------- rules
@dataclass
class Rule:
    id: str
    name: str
    typology: str
    weight: int
    description: str


RULES = [
    Rule("circular", "Circular transfer", "Circular transactions", 30,
         "Funds routed back to their origin (circular_transfer_flag = 1)"),
    Rule("shell", "Shell-account link", "Shell accounts", 20,
         "Shell-account score ≥ 0.40 (top ~8% of transactions)"),
    Rule("rapid", "Rapid fund movement", "Rapid fund movement", 15,
         "≥ 5 transfers in 24h or ≥ 6 transactions in 1 hour"),
    Rule("fanout", "Beneficiary fan-out", "Layered transfers", 15,
         "≥ 7 unique beneficiaries in 7 days – money split across many receivers"),
    Rule("mule", "Mule-account pattern", "Layered transfers", 10,
         "Account used by ≥ 7 different customers"),
    Rule("device", "Shared-device ring", "Coordinated network", 10,
         "Device used by ≥ 9 different customers"),
    Rule("newacct", "New account, high value", "Rapid fund movement", 10,
         "Account younger than 1 year moving a top-10% amount"),
    Rule("outlier", "Abnormal amount", "Structuring / outlier", 10,
         "Amount far above normal (> Q3 + 3×IQR)"),
    Rule("failed", "Repeated failed attempts", "Account takeover", 5,
         "≥ 3 failed attempts in 24h"),
]
RULES_BY_ID = {r.id: r for r in RULES}


def apply_rules(df: pd.DataFrame, amount_p90: float) -> pd.DataFrame:
    hits = pd.DataFrame(index=df.index)
    hits["circular"] = df["circular_transfer_flag"] == 1
    hits["shell"] = df["shell_account_score"] >= 0.40
    hits["rapid"] = (df["transfer_count_24h"] >= 5) | (df["transactions_1h"] >= 6)
    hits["fanout"] = df["unique_beneficiaries_7d"] >= 7
    hits["mule"] = df["customers_per_account"] >= 7
    hits["device"] = df["customers_per_device"] >= 9
    hits["newacct"] = (df["account_age_days"] < 365) & (df["amount"] >= amount_p90)
    hits["outlier"] = df["amount_outlier"] == 1
    hits["failed"] = df["failed_attempts_24h"] >= 3
    return hits


# --------------------------------------------------------------------------- matrix
def build_matrix(df: pd.DataFrame, columns: list[str] | None = None) -> pd.DataFrame:
    X = pd.concat([df[NUM_FEATURES].astype(float),
                   pd.get_dummies(df[CAT_FEATURES], prefix=CAT_FEATURES, dtype=float)], axis=1)
    if columns is not None:
        X = X.reindex(columns=columns, fill_value=0.0)
    return X


@dataclass
class TargetModel:
    target: str
    model: RandomForestClassifier
    classes: list[str]
    metrics: dict
    importances: pd.Series
    oof_proba: np.ndarray
    explainer: object = None


def train_target(X: pd.DataFrame, y: pd.Series, target: str) -> TargetModel:
    rf = RandomForestClassifier(n_estimators=300, min_samples_leaf=2, class_weight="balanced",
                                random_state=42, n_jobs=-1)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof = cross_val_predict(rf, X, y, cv=cv, method="predict_proba")
    rf.fit(X, y)
    classes = list(rf.classes_)
    pred = np.array(classes)[oof.argmax(axis=1)]
    labels = [c for c in LABEL_ORDER if c in classes]
    metrics = {
        "accuracy": accuracy_score(y, pred),
        "macro_f1": f1_score(y, pred, average="macro"),
        "weighted_f1": f1_score(y, pred, average="weighted"),
        "report": pd.DataFrame(classification_report(y, pred, labels=labels, output_dict=True,
                                                     zero_division=0)).T,
        "confusion": pd.DataFrame(confusion_matrix(y, pred, labels=labels), index=labels, columns=labels),
        "n_samples": int(len(y)),
        "class_counts": y.value_counts().reindex(labels).fillna(0).astype(int).to_dict(),
    }
    imp = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
    return TargetModel(target, rf, classes, metrics, imp, oof)


def _risk_score(proba: np.ndarray, classes: list[str]) -> np.ndarray:
    idx = {c: i for i, c in enumerate(classes)}
    high = proba[:, idx["High"]] if "High" in idx else 0
    med = proba[:, idx["Medium"]] if "Medium" in idx else 0
    return 100 * (high + 0.5 * med)


def band(score: float, thresholds: dict = BAND_THRESHOLDS) -> str:
    if score >= thresholds["High"]:
        return "High"
    if score >= thresholds["Medium"]:
        return "Medium"
    return "Low"


def entity_band(score: float) -> str:
    return band(score, ENTITY_THRESHOLDS)


# --------------------------------------------------------------------------- pipeline
@dataclass
class Pipeline:
    raw: pd.DataFrame
    clean: pd.DataFrame
    cleaning_log: pd.DataFrame
    quality_before: pd.DataFrame
    quality_after: pd.DataFrame
    duplicates: dict
    scored: pd.DataFrame
    X: pd.DataFrame
    aml: TargetModel
    fraud: TargetModel
    iso: IsolationForest
    rule_stats: pd.DataFrame
    customers: pd.DataFrame
    extras: dict = field(default_factory=dict)


def run_pipeline(path=None) -> Pipeline:
    raw = D.load_raw(path) if path else D.load_raw()
    raw_banked = raw.copy()
    raw_banked["bank_id"] = D.assign_bank(raw_banked["customer_id"])
    clean, log = D.clean(raw)
    feat = D.add_features(clean)
    X = build_matrix(feat)

    aml = train_target(X, feat["aml_risk"], "aml_risk")
    fraud = train_target(X, feat["fraud_risk"], "fraud_risk")

    iso = IsolationForest(n_estimators=300, random_state=42).fit(X)
    raw_anom = -iso.score_samples(X)
    anomaly = 100 * (raw_anom - raw_anom.min()) / (raw_anom.max() - raw_anom.min() + 1e-9)

    p90 = float(feat["amount"].quantile(0.90))
    hits = apply_rules(feat, p90)
    weights = pd.Series({r.id: r.weight for r in RULES})
    rule_score = (hits.astype(int) * weights).sum(axis=1).clip(upper=100)

    scored = feat.copy()
    # Out-of-fold probabilities: every transaction is scored by a model that never saw it.
    scored["ml_aml_score"] = _risk_score(aml.oof_proba, aml.classes).round(1)
    scored["ml_aml_pred"] = np.array(aml.classes)[aml.oof_proba.argmax(axis=1)]
    scored["ml_fraud_score"] = _risk_score(fraud.oof_proba, fraud.classes).round(1)
    scored["ml_fraud_pred"] = np.array(fraud.classes)[fraud.oof_proba.argmax(axis=1)]
    scored["rule_score"] = rule_score.astype(float)
    scored["anomaly_score"] = anomaly.round(1)
    scored["risk_score"] = (SCORE_WEIGHTS["ml"] * scored["ml_aml_score"]
                            + SCORE_WEIGHTS["rules"] * scored["rule_score"]
                            + SCORE_WEIGHTS["anomaly"] * scored["anomaly_score"]).round(1)
    scored["risk_band"] = scored["risk_score"].map(band)
    for r in RULES:
        scored[f"rule_{r.id}"] = hits[r.id].astype(int)
    scored["rules_triggered"] = hits.apply(lambda row: ", ".join(RULES_BY_ID[k].name for k, v in row.items() if v), axis=1)
    scored["rule_count"] = hits.sum(axis=1)
    scored = scored.sort_values("risk_score", ascending=False).reset_index(drop=True)
    X = build_matrix(scored, columns=list(X.columns))

    rule_stats = pd.DataFrame([{
        "rule": r.name, "typology": r.typology, "weight": r.weight, "description": r.description,
        "hits": int(hits[r.id].sum()),
        "share_of_labelled_high": round(100 * float(hits.loc[feat["aml_risk"] == "High", r.id].mean()), 1),
        "share_of_labelled_low": round(100 * float(hits.loc[feat["aml_risk"] == "Low", r.id].mean()), 1),
    } for r in RULES])

    hybrid_pred = scored["risk_band"]
    hybrid = {
        "accuracy": accuracy_score(scored["aml_risk"], hybrid_pred),
        "macro_f1": f1_score(scored["aml_risk"], hybrid_pred, average="macro"),
        "confusion": pd.DataFrame(confusion_matrix(scored["aml_risk"], hybrid_pred, labels=LABEL_ORDER),
                                  index=LABEL_ORDER, columns=LABEL_ORDER),
    }

    return Pipeline(
        raw=raw_banked, clean=clean, cleaning_log=log,
        quality_before=D.quality_report(raw), quality_after=D.quality_report(clean),
        duplicates=D.duplicate_report(raw), scored=scored, X=X, aml=aml, fraud=fraud, iso=iso,
        rule_stats=rule_stats, customers=customer_summary(scored),
        extras={"amount_p90": p90, "hybrid_metrics": hybrid},
    )


# --------------------------------------------------------------------------- aggregation
def _union_rules(series: pd.Series) -> str:
    names = set()
    for s in series:
        if s:
            names.update(x.strip() for x in s.split(","))
    return ", ".join(sorted(names))


def customer_summary(scored: pd.DataFrame) -> pd.DataFrame:
    g = scored.groupby("customer_id")
    out = g.agg(
        bank_id=("bank_id", "first"),
        transactions=("transaction_id", "count"),
        total_amount=("amount", "sum"),
        max_risk=("risk_score", "max"),
        mean_risk=("risk_score", "mean"),
        high_risk_txns=("risk_band", lambda s: int((s == "High").sum())),
        ml_aml_max=("ml_aml_score", "max"),
        ml_fraud_max=("ml_fraud_score", "max"),
        accounts=("account_id", "nunique"),
        devices=("device_id", "nunique"),
        merchants=("merchant_id", "nunique"),
        countries=("country", "nunique"),
        circular_transfers=("rule_circular", "sum"),
        labelled_aml_high=("aml_risk", lambda s: int((s == "High").sum())),
        rules_triggered=("rules_triggered", _union_rules),
    ).reset_index()
    out["customer_risk"] = (0.7 * out["max_risk"] + 0.3 * out["mean_risk"]).round(1)
    out["risk_band"] = out["customer_risk"].map(entity_band)
    out.loc[out["high_risk_txns"] >= 2, "risk_band"] = "High"
    out["verdict"] = out["risk_band"].map(VERDICTS)
    out["total_amount"] = out["total_amount"].round(2)
    out["mean_risk"] = out["mean_risk"].round(1)
    return out.sort_values(["customer_risk"], ascending=False).reset_index(drop=True)


def account_summary(scored: pd.DataFrame) -> pd.DataFrame:
    g = scored.groupby("account_id")
    out = g.agg(
        customers=("customer_id", "nunique"),
        devices=("device_id", "nunique"),
        merchants=("merchant_id", "nunique"),
        transactions=("transaction_id", "count"),
        total_amount=("amount", "sum"),
        max_risk=("risk_score", "max"),
        mean_risk=("risk_score", "mean"),
        circular_transfers=("rule_circular", "sum"),
        rules_triggered=("rules_triggered", _union_rules),
    ).reset_index()
    out["account_risk"] = (0.7 * out["max_risk"] + 0.3 * out["mean_risk"]).round(1)
    out["risk_band"] = out["account_risk"].map(entity_band)
    out["total_amount"] = out["total_amount"].round(2)
    out["mean_risk"] = out["mean_risk"].round(1)
    return out.sort_values("account_risk", ascending=False).reset_index(drop=True)


VERDICTS = {
    "High": "Likely money laundering – escalate",
    "Medium": "Elevated risk – enhanced monitoring",
    "Low": "No significant risk",
}


# --------------------------------------------------------------------------- explanations
def explain_rows(pipe: Pipeline, row_index: list[int], top_n: int = 6) -> pd.DataFrame:
    """Top features pushing the AML model towards 'High' for the given scored rows."""
    X = pipe.X.loc[row_index]
    try:
        import shap
        if pipe.aml.explainer is None:
            pipe.aml.explainer = shap.TreeExplainer(pipe.aml.model)
        sv = pipe.aml.explainer.shap_values(X)
        hi = pipe.aml.classes.index("High")
        if isinstance(sv, list):
            vals = sv[hi]
        else:
            vals = sv[:, :, hi]
        contrib = pd.DataFrame(vals, index=X.index, columns=X.columns).mean(axis=0)
        method = "SHAP"
    except Exception:  # fallback: importance × standardised deviation
        z = (X - pipe.X.mean()) / (pipe.X.std() + 1e-9)
        contrib = (z.mean(axis=0) * pipe.aml.importances.reindex(X.columns)).fillna(0)
        method = "importance-weighted deviation"
    top = contrib.sort_values(ascending=False).head(top_n)
    top = top[top > 0]
    values = X[top.index].mean(axis=0)
    return pd.DataFrame({
        "factor": [feature_label(f) for f in top.index],
        "value": [round(float(values[f]), 3) for f in top.index],
        "impact": top.values.round(4),
        "method": method,
    })
