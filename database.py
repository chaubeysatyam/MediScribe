import sqlite3
import json
import os

DB = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mediscribe.db")

def _init():
    conn = sqlite3.connect(DB)
    conn.execute("CREATE TABLE IF NOT EXISTS encounters (id TEXT PRIMARY KEY, data TEXT)")
    conn.commit()
    conn.close()
_init()

def save_encounter(data):
    conn = sqlite3.connect(DB)
    conn.execute("INSERT OR REPLACE INTO encounters (id, data) VALUES (?, ?)",
                 (data.get("id", ""), json.dumps(data)))
    conn.commit()
    conn.close()

def get_encounters():
    conn = sqlite3.connect(DB)
    rows = conn.execute("SELECT data FROM encounters ORDER BY rowid DESC LIMIT 50").fetchall()
    conn.close()
    return [json.loads(r[0]) for r in rows]

def delete_encounter(eid):
    conn = sqlite3.connect(DB)
    conn.execute("DELETE FROM encounters WHERE id = ?", (eid,))
    conn.commit()
    conn.close()
