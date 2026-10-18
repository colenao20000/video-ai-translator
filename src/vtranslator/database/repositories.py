"""Database Repositories for Projects, Cache, and Settings."""

import json
import time
from typing import List, Optional, Dict, Any
from .database import Database
from ..domain.project import Project


class ProjectRepository:
    def __init__(self, db: Database):
        self.db = db

    def save(self, project: Project) -> None:
        conn = self.db.get_connection()
        project.mark_updated()
        data_json = project.model_dump_json()
        with conn:
            conn.execute(
                """
                INSERT INTO projects (project_id, name, video_path, project_data, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(project_id) DO UPDATE SET
                    name = excluded.name,
                    video_path = excluded.video_path,
                    project_data = excluded.project_data,
                    updated_at = excluded.updated_at;
                """,
                (
                    project.project_id,
                    project.name,
                    project.video_path,
                    data_json,
                    project.created_at,
                    project.updated_at
                )
            )

    def get(self, project_id: str) -> Optional[Project]:
        conn = self.db.get_connection()
        cur = conn.cursor()
        cur.execute("SELECT project_data FROM projects WHERE project_id = ?;", (project_id,))
        row = cur.fetchone()
        if not row:
            return None
        return Project.model_validate_json(row["project_data"])

    def list_all(self) -> List[Dict[str, Any]]:
        conn = self.db.get_connection()
        cur = conn.cursor()
        cur.execute(
            "SELECT project_id, name, video_path, created_at, updated_at FROM projects ORDER BY updated_at DESC;"
        )
        return [dict(row) for row in cur.fetchall()]

    def delete(self, project_id: str) -> bool:
        conn = self.db.get_connection()
        with conn:
            cur = conn.execute("DELETE FROM projects WHERE project_id = ?;", (project_id,))
            return cur.rowcount > 0


class AnalysisCacheRepository:
    def __init__(self, db: Database):
        self.db = db

    def get(self, video_hash: str, analysis_type: str) -> Optional[Any]:
        conn = self.db.get_connection()
        cur = conn.cursor()
        cur.execute(
            "SELECT data FROM analysis_cache WHERE video_hash = ? AND analysis_type = ?;",
            (video_hash, analysis_type)
        )
        row = cur.fetchone()
        if not row:
            return None
        return json.loads(row["data"])

    def set(self, video_hash: str, analysis_type: str, data: Any) -> None:
        conn = self.db.get_connection()
        data_json = json.dumps(data, ensure_ascii=False)
        with conn:
            conn.execute(
                """
                INSERT INTO analysis_cache (video_hash, analysis_type, data, created_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(video_hash, analysis_type) DO UPDATE SET
                    data = excluded.data,
                    created_at = excluded.created_at;
                """,
                (video_hash, analysis_type, data_json, time.time())
            )

    def delete(self, video_hash: str) -> None:
        conn = self.db.get_connection()
        with conn:
            conn.execute("DELETE FROM analysis_cache WHERE video_hash = ?;", (video_hash,))


class SettingsRepository:
    def __init__(self, db: Database):
        self.db = db

    def get(self, key: str, default: str = "") -> str:
        conn = self.db.get_connection()
        cur = conn.cursor()
        cur.execute("SELECT value FROM app_settings WHERE key = ?;", (key,))
        row = cur.fetchone()
        return row["value"] if row else default

    def set(self, key: str, value: str) -> None:
        conn = self.db.get_connection()
        with conn:
            conn.execute(
                "INSERT INTO app_settings (key, value) VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value = excluded.value;",
                (key, value)
            )
