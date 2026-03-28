import sqlite3
import json
from contextlib import contextmanager
from config import DB_PATH


@contextmanager
def _get_conn():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def _init():
    with _get_conn() as conn:
        conn.execute(
            "CREATE TABLE IF NOT EXISTS encounters ("
            "id TEXT PRIMARY KEY, "
            "patient_name TEXT DEFAULT '', "
            "timestamp TEXT DEFAULT '', "
            "chief_complaint TEXT DEFAULT '', "
            "data TEXT NOT NULL)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_enc_ts ON encounters (timestamp DESC)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_enc_pt ON encounters (patient_name)"
        )

_init()


def save_encounter(data):
    eid  = data.get("id", "")
    name = data.get("patient_name", "")
    ts   = data.get("timestamp", "")
    cc   = (data.get("entities") or {}).get("chief_complaint", "")
    with _get_conn() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO encounters "
            "(id, patient_name, timestamp, chief_complaint, data) VALUES (?, ?, ?, ?, ?)",
            (eid, name, ts, cc, json.dumps(data))
        )


def get_encounters(page=1, limit=20):
    page   = max(1, page)
    limit  = max(1, min(100, limit))
    offset = (page - 1) * limit
    with _get_conn() as conn:
        rows = conn.execute(
            "SELECT data FROM encounters ORDER BY timestamp DESC LIMIT ? OFFSET ?",
            (limit, offset)
        ).fetchall()
    return [json.loads(r["data"]) for r in rows]


def search_encounters(q="", patient="", date_from="", date_to=""):
    sql    = "SELECT data FROM encounters WHERE 1=1"
    params = []
    if patient:
        sql += " AND patient_name LIKE ?"
        params.append("%" + patient + "%")
    if q:
        sql += " AND (chief_complaint LIKE ? OR data LIKE ?)"
        params.extend(["%" + q + "%", "%" + q + "%"])
    if date_from:
        sql += " AND timestamp >= ?"
        params.append(date_from)
    if date_to:
        sql += " AND timestamp <= ?"
        params.append(date_to)
    sql += " ORDER BY timestamp DESC LIMIT 100"
    with _get_conn() as conn:
        rows = conn.execute(sql, params).fetchall()
    return [json.loads(r["data"]) for r in rows]


def delete_encounter(eid):
    with _get_conn() as conn:
        conn.execute("DELETE FROM encounters WHERE id = ?", (eid,))
