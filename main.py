import argparse
import time

from src import (download_data, clean_data, load_to_mysql, run_sql_analysis,
                 eda_stats, risk_model, cost_benefit, build_report)

# the whole pipeline in order, each step can also be run alone with python -m src.<file>
STEPS = [
    ("download raw data", download_data.main),
    ("clean data + build tables", clean_data.main),
    ("load into mysql", load_to_mysql.main),
    ("run sql analysis", run_sql_analysis.main),
    ("eda + statistical tests", eda_stats.main),
    ("train + evaluate risk model", risk_model.main),
    ("cost-benefit analysis", cost_benefit.main),
    ("build report + dashboard", build_report.main),
]


def main():
    parser = argparse.ArgumentParser(description="diabetes readmission analytics pipeline")
    parser.add_argument("--start", type=int, default=1, help="step number to start from (1-8)")
    args = parser.parse_args()

    total = time.time()
    for i, (name, func) in enumerate(STEPS, start=1):
        if i < args.start:
            continue
        print(f"\n===== step {i}/{len(STEPS)}: {name} =====")
        t = time.time()
        func()
        print(f"-- done in {time.time() - t:.1f}s")

    print(f"\npipeline finished in {(time.time() - total) / 60:.1f} min, results are in outputs/")


if __name__ == "__main__":
    main()
