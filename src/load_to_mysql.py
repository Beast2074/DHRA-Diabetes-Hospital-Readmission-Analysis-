import time

import pandas as pd

from config import DATA_PROCESSED, SQL_DIR, DB_NAME
from src.db import get_connection, run_sql_file

# parents first so foreign keys don't fail
LOAD_ORDER = [
    "admission_type",
    "admission_source",
    "discharge_disposition",
    "medications",
    "patients",
    "encounters",
    "encounter_diagnoses",
    "encounter_medications",
]

BATCH_SIZE = 5000


def create_database():
    conn = get_connection(use_db=False)
    cur = conn.cursor()
    cur.execute(f"CREATE DATABASE IF NOT EXISTS `{DB_NAME}` CHARACTER SET utf8mb4")
    conn.commit()
    cur.close()
    conn.close()


def insert_table(conn, table, df):
    cols = ", ".join(f"`{c}`" for c in df.columns)
    placeholders = ", ".join(["%s"] * len(df.columns))
    sql = f"INSERT INTO `{table}` ({cols}) VALUES ({placeholders})"

    # convert numpy types to plain python so the connector is happy
    df = df.astype(object).where(df.notna(), None)
    rows = df.values.tolist()

    cur = conn.cursor()
    for i in range(0, len(rows), BATCH_SIZE):
        cur.executemany(sql, rows[i:i + BATCH_SIZE])
    conn.commit()
    cur.close()
    return len(rows)


def main():
    start = time.time()
    create_database()

    # build tables, then views and stored procedures
    run_sql_file(SQL_DIR / "01_schema.sql")
    print("schema created")

    conn = get_connection()
    for table in LOAD_ORDER:
        df = pd.read_csv(DATA_PROCESSED / f"{table}.csv", keep_default_na=False, na_values=[""])
        n = insert_table(conn, table, df)
        print(f"  loaded {table:<24} {n:>9,} rows")

    # quick integrity check: row counts in mysql should match the csv files
    cur = conn.cursor()
    for table in LOAD_ORDER:
        cur.execute(f"SELECT COUNT(*) FROM `{table}`")
        db_count = cur.fetchone()[0]
        csv_count = sum(1 for _ in open(DATA_PROCESSED / f"{table}.csv")) - 1
        assert db_count == csv_count, f"row count mismatch in {table}"
    cur.close()
    conn.close()

    run_sql_file(SQL_DIR / "02_views.sql")
    run_sql_file(SQL_DIR / "03_procedures.sql")
    print("views + stored procedures created")
    print(f"mysql load done in {time.time() - start:.1f}s")


if __name__ == "__main__":
    main()
