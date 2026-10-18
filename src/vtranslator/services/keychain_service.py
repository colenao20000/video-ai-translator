"""Keychain Service for securely storing API Secrets in macOS Keychain."""

import logging
from typing import Optional

logger = logging.getLogger("vtranslator.keychain")
SERVICE_NAME = "vtranslator_desktop"


class KeychainService:
    def __init__(self, service_name: str = SERVICE_NAME):
        self.service_name = service_name
        self._memory_cache = {}

    def get_secret(self, key_name: str) -> str:
        """Lấy secret từ Keychain hệ thống hoặc in-memory cache fallback."""
        if key_name in self._memory_cache:
            return self._memory_cache[key_name]
        try:
            import keyring
            secret = keyring.get_password(self.service_name, key_name)
            return secret or ""
        except Exception as e:
            logger.warning(f"Keyring không truy cập được ({e}), sử dụng memory fallback.")
            return self._memory_cache.get(key_name, "")

    def set_secret(self, key_name: str, value: str) -> bool:
        """Lưu secret vào Keychain an toàn."""
        self._memory_cache[key_name] = value
        try:
            import keyring
            if value:
                keyring.set_password(self.service_name, key_name, value)
            else:
                try:
                    keyring.delete_password(self.service_name, key_name)
                except Exception:
                    pass
            return True
        except Exception as e:
            logger.warning(f"Keyring không lưu được ({e}), đã lưu tạm trong bộ nhớ.")
            return True
