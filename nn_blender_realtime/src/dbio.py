from __future__ import annotations

import sqlite3

import pandas as pd


def read_sqlite_df(db_path: str, sql_query: str) -> pd.DataFrame:
    conn = sqlite3.connect(db_path)  # open sqlite connection
    try:
        df = pd.read_sql_query(sql_query, conn)  # load query results into df
    finally:
        conn.close()  # close connection
    return df


def stream_sqlite_rows(db_path: str, stream_query: str, last_id: int, limit: int) -> pd.DataFrame:
    conn = sqlite3.connect(db_path)  # open sqlite connection
    try:
        cur = conn.cursor()  # init cursor
        cur.execute(stream_query, (last_id, limit))  # grab new rows since last id
        cols = [d[0] for d in cur.description]  # column names
        rows = cur.fetchall()  # the data
        out = pd.DataFrame(rows, columns=cols)  # build df
    finally:
        conn.close()  # close connection
    return out
