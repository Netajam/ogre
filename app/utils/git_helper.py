# app/utils/git_helper.py
import os
from pathlib import Path
# --- Import TYPE_CHECKING and Any ---
from typing import Optional, List, Tuple, Dict, Any, TYPE_CHECKING
# -------------------------------------
from .logging_setup import logger

# --- TYPE_CHECKING block for static analysis ---
# This block is evaluated by type checkers (like Pylance) but not at runtime.
if TYPE_CHECKING:
    import git
    from git.exc import InvalidGitRepositoryError, NoSuchPathError
    # Define an alias for the Repo type for cleaner hints
    GitRepoType = git.Repo
else:
    # Define placeholders for runtime if GitPython is not installed
    # This prevents NameErrors if code accidentally tries to use these types directly
    GitRepoType = Any # Or object, if Any feels too broad at runtime
    InvalidGitRepositoryError = Exception # Fallback base exception
    NoSuchPathError = Exception         # Fallback base exception
# -----------------------------------------------

# --- Runtime import check ---
try:
    import git
    # We still need the actual exceptions for runtime checks if git *is* imported
    from git.exc import InvalidGitRepositoryError as ActualInvalidGitRepoError
    from git.exc import NoSuchPathError as ActualNoSuchPathError
    GITPYTHON_AVAILABLE = True
except ImportError:
    git = None # Set git to None for runtime checks
    # Keep runtime exception placeholders consistent
    ActualInvalidGitRepoError = InvalidGitRepositoryError
    ActualNoSuchPathError = NoSuchPathError
    GITPYTHON_AVAILABLE = False
    logger.warning("GitPython library not found. Git-based indexing disabled. pip install GitPython")
# --- End Runtime import check ---


class GitHelper:
    """Provides helper functions for interacting with a Git repository."""

    def __init__(self, repo_path_str: str):
        self.repo_path = Path(repo_path_str).resolve()
        # --- Use the defined type alias in the hint ---
        self.repo: Optional[GitRepoType] = self._get_repo()
        # -------------------------------------------

    # --- Use the defined type alias in the hint ---
    def _get_repo(self) -> Optional[GitRepoType]:
    # -------------------------------------------
        """Initializes the GitPython Repo object."""
        if not GITPYTHON_AVAILABLE:
            # This check ensures 'git' module is available before using it
            logger.info("GitPython not available, cannot initialize Git repo object.")
            return None
        try:
            # Now 'git' is guaranteed to be the imported module here
            repo = git.Repo(self.repo_path, search_parent_directories=True)
            repo_root_resolved = Path(repo.working_dir).resolve()
            if not str(self.repo_path).startswith(str(repo_root_resolved)):
                 logger.warning(f"Notes folder '{self.repo_path}' is not inside found Git repo '{repo_root_resolved}'.")
                 return None
            logger.info(f"Initialized Git repo for path: {repo_root_resolved}")
            return repo
        # Use the runtime exception types for catching errors
        except ActualInvalidGitRepoError as e:
            logger.info(f"Path '{self.repo_path}' is not a valid Git repository.")
            return None
        except ActualNoSuchPathError as e:
             logger.error(f"Path '{self.repo_path}' does not exist.")
             return None
        except Exception as e: # Catch other potential GitPython or general errors
            logger.error(f"Error initializing Git repo at '{self.repo_path}': {e}", exc_info=True)
            return None

    def is_valid_repo(self) -> bool:
        """Checks if a valid Git repository was found for the path."""
        return self.repo is not None

    def get_current_commit_hash(self) -> Optional[str]:
        """Gets the SHA hash of the current HEAD commit."""
        if not self.repo: return None
        try:
            # Access attributes directly now
            return self.repo.head.commit.hexsha
        except Exception as e:
            logger.error(f"Error getting current commit hash: {e}", exc_info=True)
            return None

    def has_uncommitted_changes(self) -> bool:
        """Checks if there are uncommitted changes."""
        if not self.repo: return True
        try:
            is_dirty = self.repo.is_dirty(untracked_files=True)
            if is_dirty: logger.warning("Git repo has uncommitted changes or untracked files.")
            return is_dirty
        except Exception as e:
            logger.error(f"Error checking for uncommitted changes: {e}", exc_info=True)
            return True

    def get_changed_files(self, commit_hash_a: str, commit_hash_b: str = 'HEAD') -> Optional[Dict[str, List[Any]]]:
        """Gets files changed between two commits."""
        if not self.repo or not GITPYTHON_AVAILABLE: return None
        try:
            commit_a = self.repo.commit(commit_hash_a)
            commit_b = self.repo.commit(commit_hash_b)
            diff_index = commit_a.diff(commit_b)
            changes: Dict[str, List[Any]] = {'A': [], 'D': [], 'M': [], 'R': []}
            for diff_item in diff_index:
                change_type = diff_item.change_type
                if change_type == 'A': changes['A'].append(diff_item.b_path)
                elif change_type == 'D': changes['D'].append(diff_item.a_path)
                elif change_type == 'M': changes['M'].append(diff_item.b_path)
                elif change_type == 'R': changes['R'].append((diff_item.a_path, diff_item.b_path))
                elif change_type == 'T': logger.debug(f"Type change treated as modified: {diff_item.b_path}"); changes['M'].append(diff_item.b_path)
            return changes
        # Use specific runtime exception type if available
        except git.BadName if git else Exception as e:
            if isinstance(e, Exception) and not git: logger.error(f"Error during diff (GitPython likely not installed): {e}"); return None
            else: logger.error(f"Invalid commit hash provided for diff: {e}"); return None
        except Exception as e:
            logger.error(f"Error getting changed files between {commit_hash_a} and {commit_hash_b}: {e}", exc_info=True)
            return None

    def get_untracked_files(self) -> List[str]:
        """Gets a list of untracked files (relative paths)."""
        if not self.repo: return []
        try:
            return self.repo.untracked_files
        except Exception as e:
            logger.error(f"Error getting untracked files: {e}", exc_info=True)
            return []

    def get_repo_root(self) -> Optional[Path]:
         """Gets the resolved absolute path to the repository's root directory."""
         if not self.repo: return None
         try:
             return Path(self.repo.working_dir).resolve()
         except Exception as e:
              logger.error(f"Error getting repo working directory: {e}", exc_info=True)
              return None