# app/core/state_manager.py
import json
from pathlib import Path
from typing import Optional, Dict, Any

from app.utils.logging_setup import logger

class StateManager:
    """Manages saving and loading application state, like last indexed commit."""

    def __init__(self, state_dir: Path):
        """
        Initializes the StateManager.

        Args:
            state_dir: The directory where state files should be stored
                       (e.g., '.ogre_index' within the notes folder).
        """
        self.state_dir = state_dir
        self.state_file = self.state_dir / "index_state.json"
        self._state: Dict[str, Any] = self._load_state()

    def _ensure_state_dir(self):
        """Creates the state directory if it doesn't exist."""
        try:
            self.state_dir.mkdir(parents=True, exist_ok=True)
            logger.debug(f"Ensured state directory exists: {self.state_dir}")
        except OSError as e:
            logger.error(f"Could not create state directory '{self.state_dir}': {e}")
            raise # Re-raise as state cannot be saved/loaded

    def _load_state(self) -> Dict[str, Any]:
        """Loads state from the JSON file."""
        self._ensure_state_dir() # Ensure dir exists before trying to load
        if self.state_file.exists():
            try:
                with open(self.state_file, 'r', encoding='utf-8') as f:
                    state = json.load(f)
                    if isinstance(state, dict):
                        logger.info(f"Loaded index state from {self.state_file}")
                        return state
                    else:
                         logger.warning(f"State file {self.state_file} has invalid format. Resetting state.")
                         return {}
            except (json.JSONDecodeError, IOError, OSError) as e:
                logger.warning(f"Could not load state file '{self.state_file}': {e}. Resetting state.")
                return {}
        else:
             logger.info(f"State file {self.state_file} not found. Initializing empty state.")
             return {}

    def save_state(self) -> None:
        """Saves the current state to the JSON file."""
        self._ensure_state_dir() # Ensure dir exists before saving
        try:
            with open(self.state_file, 'w', encoding='utf-8') as f:
                json.dump(self._state, f, indent=4)
            logger.info(f"Saved index state to {self.state_file}")
        except (IOError, OSError) as e:
            logger.error(f"Could not save state file '{self.state_file}': {e}")
            # Should we raise here? Depends if saving state is critical path.

    def get_last_indexed_commit(self) -> Optional[str]:
        """Gets the commit hash of the last successful index."""
        commit = self._state.get("last_indexed_commit")
        return str(commit) if isinstance(commit, str) else None

    def set_last_indexed_commit(self, commit_hash: Optional[str]):
        """Sets the commit hash and saves the state."""
        if commit_hash is None:
             if "last_indexed_commit" in self._state:
                 del self._state["last_indexed_commit"]
                 logger.info("Cleared last indexed commit hash.")
                 self.save_state()
        elif isinstance(commit_hash, str):
            if self.get_last_indexed_commit() != commit_hash:
                 self._state["last_indexed_commit"] = commit_hash
                 logger.info(f"Set last indexed commit hash to: {commit_hash[:10]}")
                 self.save_state()
        else:
             logger.warning(f"Attempted to set invalid commit hash type: {type(commit_hash)}")

    def clear_state(self):
         """Clears all stored state."""
         self._state = {}
         self.save_state()
         logger.info("Cleared all index state.")