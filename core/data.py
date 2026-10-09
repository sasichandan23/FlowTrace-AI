"""Loading, profiling and cleaning of the FINSIGHT transaction dataset."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import BANK_ORDER, RAW_CSV

ID_COLS = ["transaction_id", "customer_id", "device_id", "merchant_id", "account_id"]
TEXT_COLS = ["merchant_category", "payment_method", "country"]
LABEL_COLS = ["fraud_risk", "credit_risk", "aml_risk"]
COUNT_COLS = [
    "failed_attempts_24h", "transactions_1h", "previous_defaults",
    "transfer_count_24h", "unique_beneficiaries_7d", "circular_transfer_flag",
]
NUMERIC_COLS = [
    "amount", "distance_from_home_km", "failed_attempts_24h", "transactions_1h",
    "device_age_days", "account_age_days", "income_monthly", "loan_amount",
    "credit_score", "existing_debt", "previous_defaults", "transfer_count_24h",
    "unique_beneficiaries_7d", "circular_transfer_flag", "shell_account_score",
]

PAYMENT_MAP = {"upi": "UPI", "netbanking": "NetBanking", "card": "Card", "wallet": "Wallet"}
COUNTRY_MAP = {"usa": "USA", "uk": "UK", "uae": "UAE", "india": "India", "singapore": "Singapore"}
VALID_RANGES = {
    "credit_score": (300, 900),
    "device_age_days": (0, 3650),
    "shell_account_score": (0, 1),
    "amount": (0, None),
}


def load_raw(path=RAW_CSV) -> pd.DataFrame:
    return pd.read_csv(path)


def assign_bank(customer_ids: pd.Series) -> pd.Series:
    """Stable demo mapping of customer -> home bank (numeric part mod 3)."""
    nums = customer_ids.astype(str).str.extract(r"(\d+)")[0].fillna("0").astype(int)
    return nums.mod(len(BANK_ORDER)).map(dict(enumerate(BANK_ORDER)))


# --------------------------------------------------------------------------- profiling
def _is_blank(s: pd.Series) -> pd.Series:
    return s.isna() | (s.astype(str).str.strip() == "")


def quality_report(df: pd.DataFrame) -> pd.DataFrame:
    """Column-level data-quality profile."""
    rows = []
    for c in df.columns:
        if c == "bank_id":
            continue
        s = df[c]
        blank = _is_blank(s)
        row = {
            "column": c,
            "dtype": str(s.dtype),
            "missing": int(blank.sum()),
            "missing_pct": round(100 * float(blank.mean()), 2),
            "unique_values": int(s[~blank].nunique()),
            "case_or_space_variants": 0,
            "out_of_range": 0,
        }
        if s.dtype == object or str(s.dtype) == "string":
            vals = s[~blank].astype(str)
            row["case_or_space_variants"] = int(vals.nunique() - vals.str.strip().str.lower().nunique())
        if c in VALID_RANGES and pd.api.types.is_numeric_dtype(s):
            lo, hi = VALID_RANGES[c]
            bad = pd.Series(False, index=s.index)
            if lo is not None:
                bad |= s < lo
            if hi is not None:
                bad |= s > hi
            row["out_of_range"] = int(bad.sum())
        row["issues"] = row["missing"] + row["case_or_space_variants"] + row["out_of_range"]
        rows.append(row)
    return pd.DataFrame(rows)


def duplicate_report(df: pd.DataFrame) -> dict:
    """Duplicate transaction-ID checks."""
    dup_mask = df["transaction_id"].duplicated(keep=False)
    dup_rows = df[dup_mask]
    distinct = dup_rows.drop_duplicates()
    conflicting_ids = distinct.loc[distinct["transaction_id"].duplicated(), "transaction_id"].unique()
    return {
        "total_rows": int(len(df)),
        "unique_transaction_ids": int(df["transaction_id"].nunique()),
        "exact_duplicate_rows": int(df.duplicated().sum()),
        "extra_rows_with_repeated_id": int(df["transaction_id"].duplicated().sum()),
        "ids_with_conflicting_values": int(len(conflicting_ids)),
        "conflicting_rows": distinct[distinct["transaction_id"].isin(conflicting_ids)].sort_values("transaction_id"),
    }


# --------------------------------------------------------------------------- cleaning
def _canon(s: pd.Series, mapping: dict | None = None) -> pd.Series:
    s = s.astype("string").str.strip().replace("", pd.NA)
    if mapping:
        return s.str.lower().map(mapping).fillna(s.str.title()).astype("string")
    return s.str.title()


def clean(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (clean dataframe, cleaning log)."""
    df = raw.copy()
    log: list[dict] = []

    def add(step, column, n, detail):
        log.append({"step": step, "column": column, "rows_affected": int(n), "detail": detail})

    # 1. Normalise text: case, whitespace, canonical spellings
    for c in TEXT_COLS + LABEL_COLS:
        before = df[c].astype("string")
        mapping = PAYMENT_MAP if c == "payment_method" else COUNTRY_MAP if c == "country" else None
        df[c] = _canon(df[c], mapping)
        changed = (before.fillna("<NA>") != df[c].fillna("<NA>")) & before.notna() & df[c].notna()
        if changed.sum():
            add("Standardise text values", c, changed.sum(),
                f"Unified case/whitespace variants into {df[c].nunique()} canonical values")

    # 2. Duplicates
    n = len(df)
    df = df.drop_duplicates()
    add("Remove exact duplicate rows", "all", n - len(df), "Identical records after text normalisation")
    n = len(df)
    df["_completeness"] = df.notna().sum(axis=1)
    df = (df.sort_values(["transaction_id", "_completeness"], ascending=[True, False])
            .drop_duplicates("transaction_id", keep="first")
            .drop(columns="_completeness"))
    add("Resolve conflicting duplicate IDs", "transaction_id", n - len(df),
        "Same transaction ID with different values – kept the most complete record")

    # 3. Types
    df["transaction_time"] = pd.to_datetime(df["transaction_time"], errors="coerce")
    bad_t = df["transaction_time"].isna().sum()
    if bad_t:
        add("Parse timestamps", "transaction_time", bad_t, "Unparseable timestamps set to missing")

    for c in COUNT_COLS:
        s = pd.to_numeric(df[c], errors="coerce")
        frac = s.notna() & (s % 1 != 0)
        if frac.sum():
            add("Invalid count values", c, frac.sum(), f"Non-integer counts ({s[frac].tolist()[:3]}) set to missing")
        df[c] = s.where(~frac)

    # 4. Impossible values
    for c, (lo, hi) in VALID_RANGES.items():
        s = pd.to_numeric(df[c], errors="coerce")
        bad = pd.Series(False, index=s.index)
        if lo is not None:
            bad |= s < lo
        if hi is not None:
            bad |= s > hi
        if bad.sum():
            add("Impossible values", c, bad.sum(),
                f"Outside valid range {lo}–{hi}: {sorted(s[bad].round(2).tolist())[:6]} set to missing")
        df[c] = s.where(~bad)

    # 5. Out-of-window timestamps (flagged, kept)
    month = df["transaction_time"].dt.to_period("M")
    share = month.map(month.value_counts(normalize=True))
    df["time_outlier"] = (share < 0.01).astype(int)
    if df["time_outlier"].sum():
        add("Flag out-of-window dates", "transaction_time", df["time_outlier"].sum(),
            "Transactions far outside the main reporting window – flagged, kept for review")

    # 6. Imputation (track how many fields were imputed per row)
    impute_cols = NUMERIC_COLS + TEXT_COLS
    df["fields_imputed"] = df[impute_cols].isna().sum(axis=1)
    for c in NUMERIC_COLS:
        miss = df[c].isna().sum()
        if miss:
            med = df[c].median()
            if c in COUNT_COLS:
                med = round(med)
            df[c] = df[c].fillna(med)
            add("Impute missing numeric", c, miss, f"Filled with median ({med:,.2f})")
    for c in TEXT_COLS:
        miss = df[c].isna().sum()
        if miss:
            df[c] = df[c].fillna("Unknown")
            add("Impute missing category", c, miss, "Filled with 'Unknown'")
    for c in COUNT_COLS:
        df[c] = df[c].astype(int)

    # 7. Extreme amounts (flagged, kept – they may be the signal)
    q1, q3 = df["amount"].quantile([0.25, 0.75])
    cutoff = q3 + 3 * (q3 - q1)
    df["amount_outlier"] = (df["amount"] > cutoff).astype(int)
    add("Flag extreme amounts", "amount", df["amount_outlier"].sum(),
        f"Amount above Q3 + 3×IQR ({cutoff:,.0f}) – flagged, kept as a risk signal")

    # 8. Home bank (demo mapping)
    df["bank_id"] = assign_bank(df["customer_id"])
    add("Assign home bank", "bank_id", len(df), "Demo mapping of customers to 3 banks for access control")

    for c in TEXT_COLS + LABEL_COLS:
        df[c] = df[c].astype(str)
    return df.reset_index(drop=True), pd.DataFrame(log)


# --------------------------------------------------------------------------- features
def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    t = df["transaction_time"]
    df["hour"] = t.dt.hour.fillna(12).astype(int)
    df["is_night"] = df["hour"].between(0, 5).astype(int)
    df["weekday"] = t.dt.weekday.fillna(0).astype(int)
    annual = (df["income_monthly"] * 12).replace(0, np.nan)
    df["debt_to_income"] = (df["existing_debt"] / annual).fillna(0)
    df["loan_to_income"] = (df["loan_amount"] / annual).fillna(0)
    df["amount_to_income"] = (df["amount"] / df["income_monthly"].replace(0, np.nan)).fillna(0)

    # Network / association features
    df["customers_per_account"] = df.groupby("account_id")["customer_id"].transform("nunique")
    df["customers_per_device"] = df.groupby("device_id")["customer_id"].transform("nunique")
    df["accounts_per_customer"] = df.groupby("customer_id")["account_id"].transform("nunique")
    df["devices_per_customer"] = df.groupby("customer_id")["device_id"].transform("nunique")
    df["merchants_per_account"] = df.groupby("account_id")["merchant_id"].transform("nunique")
    df["txns_per_customer"] = df.groupby("customer_id")["transaction_id"].transform("count")
    cust_mean = df.groupby("customer_id")["amount"].transform("mean")
    df["amount_vs_customer_avg"] = (df["amount"] / cust_mean).fillna(1)
    return df
