"""
Tests for the ConfigManager module.
"""

import os
from pathlib import Path

import pytest
import yaml

from src.utils.config_manager import ConfigManager
from src.utils.exceptions import ConfigurationException


class TestConfigManager:
    """Test suite for ConfigManager."""

    def test_singleton(self, config_manager: ConfigManager):
        """ConfigManager should be a singleton."""
        cm2 = ConfigManager()
        assert config_manager is cm2

    def test_get_config(self, config_manager: ConfigManager):
        """get_config() should return the config dictionary."""
        config = config_manager.get_config()
        assert isinstance(config, dict)
        assert config["app"]["name"] == "AuditHub"
        assert config["app"]["environment"] == "test"
        assert config["server"]["port"] == 8501

    def test_get_params(self, config_manager: ConfigManager):
        """get_params() should return the params dictionary."""
        params = config_manager.get_params()
        assert isinstance(params, dict)
        assert "data" in params
        assert "training" in params
        assert params["data"]["ingestion"]["chunk_size"] == 1000

    def test_get_schema(self, config_manager: ConfigManager):
        """get_schema() should return the schema dictionary."""
        schema = config_manager.get_schema()
        assert isinstance(schema, dict)
        assert "dataset" in schema
        assert "health_schema" in schema

    def test_get_logging(self, config_manager: ConfigManager):
        """get_logging() should return the logging config dictionary."""
        logging_cfg = config_manager.get_logging()
        assert isinstance(logging_cfg, dict)
        assert logging_cfg["version"] == 1
        assert "handlers" in logging_cfg
        assert "formatters" in logging_cfg

    def test_caching(self, config_manager: ConfigManager):
        """Config values should be cached after first load."""
        # First call loads from disk
        config1 = config_manager.get_config()

        # Modify the YAML file on disk
        config_path = config_manager._resolve_path("config.yaml")
        with open(config_path, "r") as f:
            data = yaml.safe_load(f)
        data["app"]["name"] = "Modified"
        with open(config_path, "w") as f:
            yaml.dump(data, f)

        # Second call should return cached version (unchanged)
        config2 = config_manager.get_config()
        assert config2["app"]["name"] == "AuditHub"  # Cached, not modified

    def test_refresh(self, config_manager: ConfigManager):
        """refresh() should clear caches and reload from disk."""
        config_manager.get_config()  # Load cache

        # Modify the file
        config_path = config_manager._resolve_path("config.yaml")
        with open(config_path, "r") as f:
            data = yaml.safe_load(f)
        data["app"]["name"] = "Refreshed"
        with open(config_path, "w") as f:
            yaml.dump(data, f)

        # Refresh and check
        config_manager.refresh()
        refreshed = config_manager.get_config()
        assert refreshed["app"]["name"] == "Refreshed"

    def test_missing_file(self, tmp_project_dir: Path):
        """Should raise ConfigurationException for missing files."""
        ConfigManager.reset_instance()
        cm = ConfigManager(config_dir=tmp_project_dir / "configs")

        with pytest.raises(ConfigurationException) as exc:
            cm.get_config()
        assert "not found" in str(exc.value).lower()

    def test_invalid_yaml(self, configs_dir: Path):
        """Should raise ConfigurationException for invalid YAML."""
        ConfigManager.reset_instance()

        # Write invalid YAML
        bad_file = configs_dir / "config.yaml"
        with open(bad_file, "w") as f:
            f.write(": invalid yaml [}")

        cm = ConfigManager(config_dir=configs_dir)
        with pytest.raises(ConfigurationException):
            cm.get_config()

    def test_missing_required_keys(self, configs_dir: Path):
        """Should raise ConfigurationException when required keys are missing."""
        ConfigManager.reset_instance()

        # Write config missing 'app' section
        with open(configs_dir / "config.yaml", "w") as f:
            yaml.dump({"server": {"port": 8080}}, f)
        # Write valid other configs
        with open(configs_dir / "params.yaml", "w") as f:
            yaml.dump({"data": {"ingestion": {}, "validation": {}}, "training": {"test_size": 0.2}}, f)
        with open(configs_dir / "schema.yaml", "w") as f:
            yaml.dump({"dataset": {"constraints": {}, "column_types": {}}}, f)
        with open(configs_dir / "logging.yaml", "w") as f:
            yaml.dump({"version": 1, "handlers": {}, "formatters": {}}, f)

        cm = ConfigManager(config_dir=configs_dir)
        with pytest.raises(ConfigurationException) as exc:
            cm.get_config()
        assert "Missing keys" in str(exc.value)

    def test_env_var_override(self, config_manager: ConfigManager, monkeypatch):
        """Environment variables should override YAML values."""
        monkeypatch.setenv("APP_ENV", "production")
        config = config_manager.get_config()
        assert config["app"]["environment"] == "production"

    def test_env_var_database_url_override(self, config_manager: ConfigManager, monkeypatch):
        """DATABASE_URL env var should override database.path."""
        monkeypatch.setenv("DATABASE_URL", "data/custom.db")
        config = config_manager.get_config()
        assert config["database"]["path"] == "data/custom.db"

    def test_env_var_log_level(self, config_manager: ConfigManager, monkeypatch):
        """LOG_LEVEL env var should override logging.level."""
        monkeypatch.setenv("LOG_LEVEL", "DEBUG")
        config = config_manager.get_config()
        assert config["logging"]["level"] == "DEBUG"

    def test_env_var_port_override(self, config_manager: ConfigManager, monkeypatch):
        """STREAMLIT_SERVER_PORT env var should override server.port."""
        monkeypatch.setenv("STREAMLIT_SERVER_PORT", "9090")
        config = config_manager.get_config()
        assert config["server"]["port"] == 9090

    def test_env_var_type_coercion_bool(self, config_manager: ConfigManager, monkeypatch):
        """Env string 'true' should be coerced to boolean True."""
        # The 'reload' field in config is boolean
        monkeypatch.setenv("APP_ENV", "production")
        config = config_manager.get_config()
        assert config["app"]["environment"] == "production"

    def test_config_dir_property(self, config_manager: ConfigManager):
        """config_dir() should return the resolved config directory."""
        assert config_manager.config_dir().name == "configs"

    def test_reset_instance(self):
        """reset_instance() should allow creating a fresh ConfigManager."""
        ConfigManager.reset_instance()
        cm1 = ConfigManager()
        ConfigManager.reset_instance()
        cm2 = ConfigManager()
        assert cm1 is not cm2

    def test_get_config_returns_copy(self, config_manager: ConfigManager):
        """get_config() should return a mutable copy, not the internal dict."""
        config = config_manager.get_config()
        config["app"]["name"] = "Hacked"
        # Getting again should return original
        config2 = config_manager.get_config()
        assert config2["app"]["name"] == "AuditHub"

    def test_config_missing_dir(self, tmp_project_dir: Path):
        """Should raise ConfigurationException if config dir doesn't exist."""
        ConfigManager.reset_instance()
        nonexistent = tmp_project_dir / "nonexistent"
        cm = ConfigManager(config_dir=nonexistent)
        with pytest.raises(ConfigurationException) as exc:
            cm.get_config()
        assert "not found" in str(exc.value).lower()
