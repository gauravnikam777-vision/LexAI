"""One-time DB migration: add chat_sessions table."""
import sqlite3, os

db_path = os.path.join(os.path.dirname(__file__), 'database.db')
conn = sqlite3.connect(db_path)
conn.execute("""
    CREATE TABLE IF NOT EXISTS chat_sessions (
        id           INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id      INTEGER NOT NULL,
        history_json TEXT    NOT NULL DEFAULT '[]',
        updated_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        UNIQUE(user_id)
    )
""")
conn.commit()
tables = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
print("OK - All tables:", [t[0] for t in tables])
conn.close()
