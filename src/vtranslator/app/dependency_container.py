"""Dependency Injection Container."""

import os
from ..database.database import Database
from ..database.migrations import run_migrations
from ..database.repositories import ProjectRepository, AnalysisCacheRepository, SettingsRepository
from ..services.cache_service import CacheService
from ..services.keychain_service import KeychainService
from ..audio.separator import AudioSeparatorService
from .settings import AppSettings
from .logging_config import setup_logging


class DependencyContainer:
    def __init__(self, db_path: str = None):
        self.logger = setup_logging()
        self.db = Database.get_instance(db_path)
        run_migrations(self.db)

        self.project_repo = ProjectRepository(self.db)
        self.cache_repo = AnalysisCacheRepository(self.db)
        self.settings_repo = SettingsRepository(self.db)

        self.cache_service = CacheService(self.cache_repo)
        self.keychain_service = KeychainService()
        saved_settings = self.settings_repo.get("app_settings_json")
        try:
            self.app_settings = AppSettings.model_validate_json(saved_settings) if saved_settings else AppSettings()
        except Exception as exc:
            self.logger.warning("Saved settings could not be loaded; using defaults: %s", exc)
            self.app_settings = AppSettings()
        self.app_settings.ensure_directories()
        self.separator_service = AudioSeparatorService(
            self.app_settings.temp_dir, self.app_settings.ffmpeg_path
        )
