# app/core/config_manager.py
import json
import os
from pathlib import Path
from typing import Any, Optional, Dict, List # Import List

from PyQt6.QtCore import QStandardPaths

from app import config
# Assuming logger setup is correct now
from app.core.utils.logging_setup import logger

class ConfigManager:
    """Handles loading and saving application settings."""

    def __init__(self, app_name: str = config.APP_NAME, org_name: str = config.ORGANIZATION_NAME):
        self.app_name = app_name
        self.org_name = org_name
        self._config_path: Path = self._get_config_path()
        self._settings: Dict[str, Any] = self._load_settings()
        self._ensure_defaults()

    def _get_config_path(self) -> Path:
        """Determines the path for the configuration file."""
        try:
            # Use AppConfigLocation for user-specific settings
            app_data_dir = QStandardPaths.writableLocation(QStandardPaths.StandardLocation.AppConfigLocation)
            config_dir = Path(app_data_dir) # Already includes Org/App on some systems

            # Ensure Org/App structure robustly
            if self.org_name and self.org_name != config_dir.parent.name and self.org_name != config_dir.name:
                config_dir = config_dir / self.org_name
            if self.app_name and self.app_name != config_dir.name:
                config_dir = config_dir / self.app_name

            config_dir.mkdir(parents=True, exist_ok=True)
            return config_dir / "settings.json"
        except Exception as e:
            logger.error(f"Could not determine config path: {e}. Using fallback './settings.json'", exc_info=True)
            # Fallback to current directory (less ideal)
            fallback_path = Path(".") / "settings.json"
            return fallback_path


    def _load_settings(self) -> Dict[str, Any]:
        """Loads settings from the JSON file."""
        if self._config_path.exists():
            try:
                with open(self._config_path, 'r', encoding='utf-8') as f:
                    loaded_settings = json.load(f)
                    if isinstance(loaded_settings, dict):
                        logger.info(f"Settings loaded successfully from {self._config_path}")
                        return loaded_settings
                    else:
                         logger.warning(f"Settings file {self._config_path} does not contain a valid JSON object. Using defaults.")
                         return {}
            except (json.JSONDecodeError, IOError, OSError) as e:
                logger.warning(f"Could not load settings file '{self._config_path}': {e}. Using defaults.")
        else:
             logger.info(f"Settings file not found at {self._config_path}. Using defaults.")
        return {}

    def _ensure_defaults(self) -> None:
        """Ensures default settings exist if not loaded or invalid."""
        changed = False
        if config.SETTINGS_NOTES_FOLDER not in self._settings or not isinstance(self.get_setting(config.SETTINGS_NOTES_FOLDER), (str, type(None))):
            logger.debug(f"Setting default for {config.SETTINGS_NOTES_FOLDER}")
            self._settings[config.SETTINGS_NOTES_FOLDER] = None
            changed = True
        if config.SETTINGS_EMBEDDING_MODEL not in self._settings or self.get_setting(config.SETTINGS_EMBEDDING_MODEL) not in config.AVAILABLE_EMBEDDING_MODELS:
            logger.debug(f"Setting default for {config.SETTINGS_EMBEDDING_MODEL}")
            self._settings[config.SETTINGS_EMBEDDING_MODEL] = config.DEFAULT_EMBEDDING_MODEL
            changed = True
        # --- Ensure default for indexed extensions ---
        current_extensions = self.get_setting(config.SETTINGS_INDEXED_EXTENSIONS)
        if not isinstance(current_extensions, list) or not all(isinstance(ext, str) for ext in current_extensions):
            logger.debug(f"Setting default for {config.SETTINGS_INDEXED_EXTENSIONS}")
            self._settings[config.SETTINGS_INDEXED_EXTENSIONS] = config.DEFAULT_INDEXED_EXTENSIONS
            changed = True

        if changed:
             self.save_settings() # Save if defaults were applied

    def save_settings(self) -> None:
        """Saves the current settings to the JSON file."""
        logger.debug(f"Attempting to save settings to {self._config_path}")
        try:
            # Ensure parent directory exists right before saving
            self._config_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self._config_path, 'w', encoding='utf-8') as f:
                json.dump(self._settings, f, indent=4, ensure_ascii=False)
            logger.info(f"Settings saved successfully to {self._config_path}")
        except (IOError, OSError) as e:
            logger.error(f"Could not save settings file '{self._config_path}': {e}")
            # TODO: Implement error dialog in UI if saving fails

    def get_setting(self, key: str, default: Optional[Any] = None) -> Optional[Any]:
        """Gets a specific setting value."""
        return self._settings.get(key, default)

    def set_setting(self, key: str, value: Any) -> None:
        """Sets a specific setting value and saves all settings."""
        logger.debug(f"Setting '{key}' to '{value}'")
        self._settings[key] = value
        self.save_settings() # Auto-save on set

    def get_notes_folder(self) -> Optional[str]:
        """Gets the configured notes folder path."""
        return self.get_setting(config.SETTINGS_NOTES_FOLDER)

    def set_notes_folder(self, path: Optional[str]) -> None:
        """Sets the notes folder path."""
        # Optional: Add validation for path existence/type here if desired
        self.set_setting(config.SETTINGS_NOTES_FOLDER, path)

    def get_embedding_model_name(self) -> str:
        """Gets the name of the configured embedding model."""
        model_name = self.get_setting(config.SETTINGS_EMBEDDING_MODEL)
        # Ensure the saved model is still valid, otherwise return default
        if model_name in config.AVAILABLE_EMBEDDING_MODELS:
            return model_name
        else:
             logger.warning(f"Saved embedding model '{model_name}' is no longer available. Returning default '{config.DEFAULT_EMBEDDING_MODEL}'.")
             # Optionally update the setting to the default
             # self.set_embedding_model_name(config.DEFAULT_EMBEDDING_MODEL)
             return config.DEFAULT_EMBEDDING_MODEL


    def set_embedding_model_name(self, model_name: str) -> None:
        """Sets the name of the embedding model."""
        if model_name in config.AVAILABLE_EMBEDDING_MODELS:
            self.set_setting(config.SETTINGS_EMBEDDING_MODEL, model_name)
        else:
            logger.warning(f"Attempted to set unknown embedding model: {model_name}")

    # --- Methods for Indexed Extensions ---
    def get_indexed_extensions(self) -> List[str]:
        """Gets the list of file extensions to be indexed."""
        extensions = self.get_setting(config.SETTINGS_INDEXED_EXTENSIONS)
        # Validate the loaded setting
        if isinstance(extensions, list) and all(isinstance(ext, str) for ext in extensions):
            # Basic validation: ensure extensions start with a dot (optional but good practice)
            valid_extensions = [ext for ext in extensions if ext.startswith('.')]
            if len(valid_extensions) != len(extensions):
                 logger.warning("Some configured extensions do not start with '.'. They will be ignored.")
            return valid_extensions
        else:
            logger.warning(f"Invalid format for '{config.SETTINGS_INDEXED_EXTENSIONS}' in settings. Returning default.")
            # Optionally fix the setting file here
            # self.set_indexed_extensions(config.DEFAULT_INDEXED_EXTENSIONS)
            return config.DEFAULT_INDEXED_EXTENSIONS

    def set_indexed_extensions(self, extensions: List[str]) -> None:
        """Sets the list of file extensions to be indexed."""
        if isinstance(extensions, list) and all(isinstance(ext, str) for ext in extensions):
            # Clean up: ensure dots, lowercase, remove duplicates
            cleaned_extensions = sorted(list(set([
                ext.lower() if ext.startswith('.') else f".{ext.lower()}"
                for ext in extensions if ext # Ignore empty strings
            ])))
            self.set_setting(config.SETTINGS_INDEXED_EXTENSIONS, cleaned_extensions)
        else:
            logger.error(f"Attempted to set invalid value for indexed extensions: {extensions}. Must be a list of strings.")

    def get_db_path(self) -> Optional[Path]:
        """Gets the full path to the database file within the notes folder."""
        notes_folder = self.get_notes_folder()
        if notes_folder:
            try:
                 # Ensure notes_folder is a valid directory path
                 notes_path = Path(notes_folder)
                 if notes_path.is_dir():
                      return notes_path / config.SETTINGS_DB_FILENAME
                 else:
                      logger.error(f"Configured notes folder is not a valid directory: {notes_folder}")
                      return None
            except Exception as e:
                 logger.error(f"Error constructing DB path from notes folder '{notes_folder}': {e}")
                 return None
        return None

# Create a singleton instance for easy access
config_manager = ConfigManager()