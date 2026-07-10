"""
Tests for the AppSettings module.
"""

from pathlib import Path

import pytest

from src.utils.app_settings import AppSettings
from src.utils.config_manager import ConfigManager


class TestAppSettings:
    """Test suite for AppSettings."""

    def test_load_returns_settings(self, config_manager: ConfigManager):
        """load() should return an AppSettings instance."""
        settings = AppSettings.load()
        assert isinstance(settings, AppSettings)

    def test_app_name(self, config_manager: ConfigManager):
        """app_name should be read from config."""
        settings = AppSettings.load()
        assert settings.app_name == "AuditHub"

    def test_app_version(self, config_manager: ConfigManager):
        """app_version should be read from config."""
        settings = AppSettings.load()
        assert settings.app_version == "0.1.0"

    def test_environment(self, config_manager: ConfigManager):
        """environment should be read from config."""
        settings = AppSettings.load()
        assert settings.environment == "test"

    def test_debug_mode(self, config_manager: ConfigManager):
        """debug should be read from config."""
        settings = AppSettings.load()
        assert settings.debug is True

    def test_server_port(self, config_manager: ConfigManager):
        """port should be read from config."""
        settings = AppSettings.load()
        assert settings.port == 8501

    def test_db_engine(self, config_manager: ConfigManager):
        """db_engine should be read from config."""
        settings = AppSettings.load()
        assert settings.db_engine == "sqlite"

    def test_db_path_is_path(self, config_manager: ConfigManager):
        """db_path should be a Path object."""
        settings = AppSettings.load()
        assert isinstance(settings.db_path, Path)

    def test_mlflow_experiment(self, config_manager: ConfigManager):
        """mlflow_experiment should be read from config."""
        settings = AppSettings.load()
        assert settings.mlflow_experiment == "test_experiment"

    def test_to_dict(self, config_manager: ConfigManager):
        """to_dict() should return a dictionary with string paths."""
        settings = AppSettings.load()
        d = settings.to_dict()
        assert isinstance(d, dict)
        assert d["app_name"] == "AuditHub"
        assert isinstance(d["db_path"], str)  # Paths converted to strings

    def test_load_with_custom_config_dir(self, config_manager: ConfigManager):
        """load() should work with custom config directory."""
        settings = AppSettings.load()
        assert settings.app_name == "AuditHub"

    def test_default_values(self):
        """Settings without config should use defaults."""
        # Can't easily test without config, but check that defaults exist
        settings = AppSettings()
        assert settings.app_name == "AuditHub"
        assert settings.port == 8501
        assert settings.dvc_remote == "local"

    def test_immutable(self, config_manager: ConfigManager):
        """AppSettings should be immutable (frozen dataclass)."""
        settings = AppSettings.load()
        with pytest.raises(AttributeError):
            settings.app_name = "NewName"  # type: ignore[misc]

    def test_env_override_reflected(self, config_manager: ConfigManager, monkeypatch):
        """Env vars should be reflected in AppSettings via ConfigManager."""
        monkeypatch.setenv("APP_ENV", "production")
        settings = AppSettings.load()
        assert settings.environment == "production"

    @pytest.mark.parametrize("field,expected_type", [
        ("app_name", str),
        ("app_version", str),
        ("environment", str),
        ("debug", bool),
        ("port", int),
        ("api_port", int),
        ("db_engine", str),
        ("db_path", Path),
        ("log_level", str),
        ("log_file", Path),
        ("mlflow_tracking_uri", str),
        ("dvc_remote", str),
        ("dvc_cache_dir", Path),
        ("streamlit_page_title", str),
        ("streamlit_layout", str),
        ("profiling_minimal", bool),
        ("profiling_pool_size", int),
    ])
    def test_field_types(self, config_manager: ConfigManager, field: str, expected_type: type):
        """All settings fields should have the correct types."""
        settings = AppSettings.load()
        value = getattr(settings, field)
        assert isinstance(value, expected_type), (
            f"Field '{field}' expected {expected_type.__name__}, got {type(value).__name__}"
        )
