import mysql.connector
import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.engine import URL

from config import DB_HOST, DB_PORT, DB_USER, DB_PASSWORD, DB_NAME


def get_connection(use_db=True):
    # raw connector, used for DDL and bulk inserts
    return mysql.connector.connect(
        host=DB_HOST,
        port=DB_PORT,
        user=DB_USER,
        password=DB_PASSWORD,
        database=DB_NAME if use_db else None,
    )


def get_engine():
    # sqlalchemy engine so pandas can read_sql straight into a dataframe
    url = URL.create(
        "mysql+mysqlconnector",
        username=DB_USER,
        password=DB_PASSWORD,
        host=DB_HOST,
        port=DB_PORT,
        database=DB_NAME,
    )
    return create_engine(url)


def query_df(sql, params=None):
    engine = get_engine()
    with engine.connect() as conn:
        df = pd.read_sql(sql, conn, params=params)
    engine.dispose()
    return df


def split_sql_script(text):
    # split a .sql file into statements, handles DELIMITER $$ blocks for procedures
    statements = []
    delimiter = ";"
    buffer = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.upper().startswith("DELIMITER"):
            delimiter = stripped.split()[1]
            continue
        if not buffer and (stripped == "" or stripped.startswith("--")):
            continue
        buffer.append(line)
        if stripped.endswith(delimiter):
            stmt = "\n".join(buffer).strip()
            stmt = stmt[: -len(delimiter)].strip()
            if stmt:
                statements.append(stmt)
            buffer = []
    if buffer and "\n".join(buffer).strip():
        statements.append("\n".join(buffer).strip())
    return statements


def run_sql_file(path, use_db=True):
    text = open(path, encoding="utf-8").read()
    conn = get_connection(use_db=use_db)
    cur = conn.cursor()
    for stmt in split_sql_script(text):
        cur.execute(stmt)
        # some statements (like CALL) return rows, clear them before the next one
        if cur.with_rows:
            cur.fetchall()
    conn.commit()
    cur.close()
    conn.close()
