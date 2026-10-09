"""Central configuration: paths, colours, banks and security settings."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RAW_CSV = DATA_DIR / "01_FINSIGHT_UNPROCESSED.csv"
CONFIG_DIR = ROOT / "config"
EMPLOYEES_FILE = CONFIG_DIR / "employees.json"
LOG_DIR = ROOT / "logs"
AUDIT_LOG = LOG_DIR / "audit.log"

APP_NAME = "FlowTrace AI"
APP_TAGLINE = "Intelligent Anti-Money-Laundering Network Analysis"

# Status palette (requested) + supporting neutrals
COLORS = {
    "success": "#1E7E34",   # forest green
    "warning": "#D97706",   # amber / gold
    "danger": "#B91C1C",    # crimson
    "info": "#2563EB",      # royal blue
    "navy": "#0F1B2D",
    "slate": "#334155",
    "muted": "#64748B",
    "border": "#E2E8F0",
    "bg": "#F7F8FA",
    "card": "#FFFFFF",
    "purple": "#7C3AED",
    "teal": "#0F766E",
}
RISK_COLORS = {"High": COLORS["danger"], "Medium": COLORS["warning"], "Low": COLORS["success"]}
RISK_ORDER = ["High", "Medium", "Low"]

# Demo banks. The dataset has no bank column, so every customer is assigned a
# stable home bank (customer number mod 3) – see core/data.py::assign_bank.
BANKS = {
    "NWB": "Northwind Bank",
    "MTB": "Meridian Trust Bank",
    "CFB": "Crescent Federal Bank",
}
BANK_ORDER = ["NWB", "MTB", "CFB"]

# Security
SESSION_TIMEOUT_MIN = 15
MAX_FAILED_LOGINS = 5
LOCKOUT_MIN = 5
PBKDF2_ITERATIONS = 200_000

# Risk scoring
SCORE_WEIGHTS = {"ml": 0.70, "rules": 0.25, "anomaly": 0.05}
BAND_THRESHOLDS = {"High": 45.0, "Medium": 27.0}  # transaction level, tuned on out-of-fold scores vs aml_risk
ENTITY_THRESHOLDS = {"High": 55.0, "Medium": 40.0}  # customer / account level (0.7 x max + 0.3 x mean)
