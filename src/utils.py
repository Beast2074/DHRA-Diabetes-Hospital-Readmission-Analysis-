import json

from config import OUTPUT_DIR

METRICS_FILE = OUTPUT_DIR / "key_metrics.json"


def save_metrics(section, values):
    # each step saves its main numbers here so build_report.py can use them later
    data = load_metrics()
    data[section] = values
    with open(METRICS_FILE, "w") as f:
        json.dump(data, f, indent=2, default=float)


def load_metrics():
    if METRICS_FILE.exists():
        with open(METRICS_FILE) as f:
            return json.load(f)
    return {}
