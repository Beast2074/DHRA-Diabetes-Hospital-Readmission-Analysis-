import os
from pathlib import Path
from dotenv import load_dotenv

# project folders
BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

DATA_RAW = BASE_DIR / "data" / "raw"
DATA_PROCESSED = BASE_DIR / "data" / "processed"
SQL_DIR = BASE_DIR / "sql"
OUTPUT_DIR = BASE_DIR / "outputs"
CHART_DIR = OUTPUT_DIR / "charts"
SQL_RESULTS_DIR = OUTPUT_DIR / "sql_results"

DATA_URL = "https://archive.ics.uci.edu/static/public/296/diabetes+130-us+hospitals+for+years+1999-2008.zip"

# mysql login comes from .env (see .env.example)
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = int(os.getenv("DB_PORT", 3306))
DB_USER = os.getenv("DB_USER", "root")
DB_PASSWORD = os.getenv("DB_PASSWORD", "")
DB_NAME = os.getenv("DB_NAME", "readmission_db")

# cost model assumptions (sources in README)
READMISSION_COST = 15200   # avg cost of one 30-day readmission, AHRQ HCUP 2018
INTERVENTION_COST = 500    # care transition program cost per patient
RISK_REDUCTION = 0.20      # program cuts readmission risk by 20% (CTI trial saw ~30%)

# discharge codes = expired or hospice, these patients can't be readmitted
EXCLUDED_DISCHARGE_IDS = [11, 13, 14, 19, 20, 21]

RANDOM_STATE = 42
TOP_PCT = 0.20  # share of discharges the care team can realistically follow up
