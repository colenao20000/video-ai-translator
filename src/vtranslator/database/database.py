"""SQLite Database Engine with WAL mode and Thread Safety."""

import os
import sqlite3
import threading
from typing import Optional


class Database:
    _instance: Optional["Database"] = None
    _lock = threading.Lock()

    def __init__(self, db_path: Optional[str] = None):
        if db_path is None:
            home = os.path.expanduser("~/.vtranslator")
            os.makedirs(home, exist_ok=True)
            db_path = os.path.join(home, "vtranslator.db")
        self.db_path = db_path
        self._local = threading.local()
        self._init_connection()

    @classmethod
    def get_instance(cls, db_path: Optional[str] = None) -> "Database":
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls(db_path)
            return cls._instance

    def get_connection(self) -> sqlite3.Connection:
        if not hasattr(self._local, "conn") or self._local.conn is None:
            conn = sqlite3.connect(self.db_path, timeout=15.0)
            conn.row_factory = sqlite3.Row
            # Bật chế độ WAL và Foreign Keys để đảm bảo hiệu năng và tính toàn vẹn
            conn.execute("PRAGMA journal_mode = WAL;")
            conn.execute("PRAGMA foreign_keys = ON;")
            conn.execute("PRAGMA synchronous = NORMAL;")
            self._local.conn = conn
        return self._local.conn

    def _init_connection(self) -> None:
        conn = self.get_connection()
        conn.commit()

    def close(self) -> None:
        if hasattr(self._local, "conn") and self._local.conn:
            self._local.conn.close()
            self._local.conn = None
