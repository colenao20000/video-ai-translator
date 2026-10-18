"""Database Schema Migrations."""

from .database import Database


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS schema_version (
    version INTEGER PRIMARY KEY
);

CREATE TABLE IF NOT EXISTS projects (
    project_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    video_path TEXT NOT NULL,
    project_data TEXT NOT NULL,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS analysis_cache (
    video_hash TEXT NOT NULL,
    analysis_type TEXT NOT NULL, -- 'diarization', 'stt', 'translation', 'tts'
    data TEXT NOT NULL,
    created_at REAL NOT NULL,
    PRIMARY KEY (video_hash, analysis_type)
);

CREATE TABLE IF NOT EXISTS app_settings (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""


def run_migrations(db: Database) -> None:
    conn = db.get_connection()
    with conn:
        conn.executescript(SCHEMA_SQL)
        cur = conn.cursor()
        cur.execute("SELECT version FROM schema_version WHERE version = 1;")
        if not cur.fetchone():
            cur.execute("INSERT OR REPLACE INTO schema_version (version) VALUES (1);")
