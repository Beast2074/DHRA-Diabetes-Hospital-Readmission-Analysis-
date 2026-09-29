import re

import pandas as pd
from sqlalchemy import text

from config import SQL_DIR, SQL_RESULTS_DIR, READMISSION_COST
from src.db import get_engine, get_connection


def parse_queries(path):
    # grab every "-- name:" block with its question and sql text
    content = open(path, encoding="utf-8").read()
    blocks = re.split(r"^-- name:\s*", content, flags=re.MULTILINE)[1:]
    queries = []
    for block in blocks:
        lines = block.strip().splitlines()
        name = lines[0].strip()
        question = ""
        sql_lines = []
        for line in lines[1:]:
            if line.startswith("-- question:"):
                question = line.replace("-- question:", "").strip()
            elif not line.strip().startswith("--"):
                sql_lines.append(line)
        sql = "\n".join(sql_lines).strip().rstrip(";")
        queries.append({"name": name, "question": question, "sql": sql})
    return queries


def run_procedure(call_sql):
    # CALL returns an extra status result, so run it on the raw connector and clear the rest
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(call_sql)
    df = pd.DataFrame(cur.fetchall(), columns=cur.column_names)
    while cur.nextset():
        if cur.with_rows:
            cur.fetchall()
    cur.close()
    conn.close()
    # decimals from mysql -> normal numbers (text columns stay as they are)
    for col in df.columns:
        try:
            df[col] = pd.to_numeric(df[col])
        except (ValueError, TypeError):
            pass
    return df


def main():
    SQL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    queries = parse_queries(SQL_DIR / "04_analysis_queries.sql")

    engine = get_engine()
    index_rows = []
    with engine.connect() as conn:
        # session variable used inside the queries
        conn.execute(text(f"SET @readmit_cost = {READMISSION_COST}"))
        for q in queries:
            if q["sql"].upper().startswith("CALL"):
                df = run_procedure(q["sql"])
            else:
                df = pd.read_sql(text(q["sql"]), conn)
            # mysql SUM() comes back as decimal/float, turn whole numbers back into ints
            for col in df.columns:
                if df[col].dtype == "float64" and (df[col].dropna() % 1 == 0).all():
                    df[col] = df[col].astype("Int64")
            df.to_csv(SQL_RESULTS_DIR / f"{q['name']}.csv", index=False)
            index_rows.append({"query": q["name"], "question": q["question"], "rows": len(df)})
            print(f"  {q['name']:<36} {len(df):>3} rows")
    engine.dispose()

    pd.DataFrame(index_rows).to_csv(SQL_RESULTS_DIR / "_query_index.csv", index=False)
    print(f"ran {len(queries)} sql queries -> {SQL_RESULTS_DIR}")


if __name__ == "__main__":
    main()
