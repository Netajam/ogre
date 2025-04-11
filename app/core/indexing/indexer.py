# app/core/indexing/indexer.py
import time
import traceback
import sqlite3
from pathlib import Path
from typing import Optional, Callable, Tuple, List, Set, Any, Dict
import numpy as np

# Core component imports
from app.core.database.db_manager import DbManager
from app.core.embedding.embedding_manager import EmbeddingManager
from app.core.indexing.file_handler import FileHandler
from app.core.indexing.chunking import split_text
# Utils and State
from app.utils.logging_setup import logger
# --- Import GitHelper and StateManager ---
from app.utils.git_helper import GitHelper, GITPYTHON_AVAILABLE
from app.core.state_manager import StateManager
# ---

EMBEDDING_BATCH_SIZE = 64

class Indexer:
    """Orchestrates file indexing using Git diffs or modification times."""

    def __init__(self, notes_folder: str, db_path: Path, embedding_model_name: str,
                 allowed_extensions: List[str], progress_callback: Optional[Callable[[str], None]] = None,
                 finished_callback: Optional[Callable[[bool, str], None]] = None):
        """Initializes the Indexer."""
        self.notes_folder_path = Path(notes_folder).resolve()
        self.db_path = db_path
        self.embedding_model_name = embedding_model_name
        self.allowed_extensions = [ext.lower() for ext in allowed_extensions] # Ensure lowercase
        self.progress_callback = progress_callback
        self.finished_callback = finished_callback
        self._is_cancelled = False
        self.embedder: Optional[Any] = None
        self.embedding_dim: Optional[int] = None

        self.git_helper = GitHelper(str(self.notes_folder_path))
        self.state_manager = StateManager(self.notes_folder_path / ".ogre_index")

        logger.debug(f"Indexer initialized. Folder: '{notes_folder}', DB: '{db_path}', Model: '{embedding_model_name}', Extensions: {self.allowed_extensions}")

    # --- Callback/Cancel Methods ---
    def _emit_progress(self, message: str):
        if self.progress_callback:
            try: self.progress_callback(message)
            except Exception as e: logger.error(f"Error in progress callback: {e}")
        logger.info(f"Progress: {message}")

    def _emit_finished(self, success: bool, message: str):
        if self.finished_callback:
            try: self.finished_callback(success, message)
            except Exception as e: logger.error(f"Error in finished callback: {e}")
        logger.info(f"Finished: Success={success}, Message='{message}'")

    def cancel(self):
        if not self._is_cancelled:
            self._emit_progress("Cancellation requested...")
            self._is_cancelled = True
            logger.info("Cancellation flag set for Indexer.")

    # --- Main Execution ---
    def run_indexing(self):
        """Runs the full indexing process, attempting Git-based diff first."""
        start_time = time.time()
        self._is_cancelled = False
        db_manager: Optional[DbManager] = None
        proc_count, add_count, del_count, ren_count = 0, 0, 0, 0
        total_files_analyzed = 0 # For reporting
        strategy = "Mod-time scan" # Default strategy

        try:
            self._emit_progress("Initializing indexing...")
            if not self.allowed_extensions: raise ValueError("No allowed extensions.")

            self.embedder, self.embedding_dim = self._initialize_embedder()
            if self._is_cancelled: raise InterruptedError("Cancelled after model load.")

            # Determine Strategy
            use_git = False
            if GITPYTHON_AVAILABLE and self.git_helper.is_valid_repo():
                if self.git_helper.has_uncommitted_changes():
                    self._emit_progress("WARN: Git repo has uncommitted changes. Falling back to mod-time scan.")
                else:
                    use_git = True
                    strategy = "Git diff"
                    logger.info("Using Git diff for incremental indexing.")
            else:
                logger.info("Not a Git repo or GitPython unavailable. Using modification time scan.")

            # Database Context
            with DbManager(self.db_path) as db_manager:
                db_manager.initialize_db(self.embedding_dim)
                if self._is_cancelled: raise InterruptedError("Cancelled after DB init.")

                if use_git:
                    proc_count, add_count, del_count, ren_count, total_files_analyzed = self._run_git_based_indexing(db_manager)
                else:
                    proc_count, add_count, del_count, total_files_analyzed = self._run_mod_time_based_indexing(db_manager)
                    ren_count = 0 # No rename detection in mod-time mode

            db_manager = None
            if self._is_cancelled: raise InterruptedError("Cancelled by user.")

            # Final Report
            self._generate_final_report(start_time, total_files_analyzed, proc_count,
                                        add_count, del_count, ren_count, strategy)

        except InterruptedError:
             end_time = time.time(); duration = end_time - start_time
             message = f"Indexing cancelled after {duration:.2f}s."
             self._emit_finished(False, message); logger.info(message)
        except (ValueError, ImportError) as e:
             message = f"Indexing failed due to setup error: {e}"
             logger.error(message, exc_info=True); self._emit_finished(False, message)
        except ConnectionError as e:
            message = f"Indexing failed due to connection error: {e}"
            logger.error(message, exc_info=True); self._emit_finished(False, message)
        except sqlite3.Error as e:
            db_traceback = traceback.format_exc()
            message = f"Indexing failed due to database error: {e}\n{db_traceback}"
            logger.error(message); self._emit_finished(False, f"Indexing failed due to database error: {e}")
        except Exception as e:
            message = f"An unexpected error occurred during indexing: {e}"
            logger.error(message, exc_info=True); self._emit_finished(False, message)
        finally:
             if db_manager and db_manager.conn: db_manager.close()
             self.embedder = None; logger.debug("Indexer run finished.")

    # --- Git-Based Indexing Logic ---

    def _run_git_based_indexing(self, db_manager: DbManager) -> Tuple[int, int, int, int, int]:
        """Performs indexing based on git diff. Returns counts: proc, add, del, ren, analyzed."""
        last_commit = self.state_manager.get_last_indexed_commit()
        current_commit = self.git_helper.get_current_commit_hash()
        repo_root = self.git_helper.get_repo_root()

        if not current_commit or not repo_root: raise RuntimeError("Could not get current commit/repo root.")

        proc_count, add_count, del_count, ren_count = 0, 0, 0, 0
        files_to_reindex_abs: List[Path] = []
        total_analyzed = 0 # Git doesn't easily give total files scanned in diff

        if last_commit == current_commit:
            self._emit_progress("No new commits. Checking untracked files...")
            untracked_abs = self._get_relevant_untracked_files(repo_root)
            total_analyzed = len(untracked_abs) # Rough estimate
            if untracked_abs: files_to_reindex_abs = untracked_abs
            else: self._emit_progress("No relevant untracked files found.")
        elif not last_commit:
            self._emit_progress("First index run (Git). Indexing all relevant files...")
            files_to_reindex_abs = self._get_all_tracked_files_in_commit(repo_root)
            total_analyzed = len(files_to_reindex_abs)
        else:
            self._emit_progress(f"Finding changes between {last_commit[:7]} and {current_commit[:7]}...")
            changes = self.git_helper.get_changed_files(last_commit, current_commit)
            untracked_rel = self.git_helper.get_untracked_files() # Get relative untracked paths

            if changes is None: raise RuntimeError("Failed to get changed files.")

            relevant_changes = self._filter_git_changes(changes, repo_root)
            relevant_untracked_abs = self._filter_untracked_files(untracked_rel, repo_root) # Filter and make absolute

            # Deletions
            if relevant_changes['D']:
                self._emit_progress(f"Processing {len(relevant_changes['D'])} deleted files...")
                del_count += self._delete_files_from_index(db_manager, relevant_changes['D'], repo_root)

            # Renames
            if relevant_changes['R']:
                self._emit_progress(f"Processing {len(relevant_changes['R'])} renamed files...")
                ren_count += self._rename_files_in_index(db_manager, relevant_changes['R'], repo_root)

            # Combine A, M, Untracked relative paths for re-indexing
            rel_paths_to_reindex = relevant_changes['A'] + relevant_changes['M'] + [str(p.relative_to(repo_root)) for p in relevant_untracked_abs]
            files_to_reindex_abs = [repo_root / Path(p) for p in set(rel_paths_to_reindex)] # Unique absolute paths
            total_analyzed = len(changes.get('A',[]))+len(changes.get('D',[]))+len(changes.get('M',[]))+len(changes.get('R',[]))+len(untracked_rel)

        # Process combined list for Add/Modify/Untracked
        if files_to_reindex_abs:
            self._emit_progress(f"Processing {len(files_to_reindex_abs)} added/modified/untracked files...")
            indexed_paths_set = set(db_manager.get_indexed_files().keys()) # Get current index state
            proc_count, add_count = self._process_files_list(db_manager, files_to_reindex_abs, indexed_paths_set)

        if not self._is_cancelled: self.state_manager.set_last_indexed_commit(current_commit)
        return proc_count, add_count, del_count, ren_count, total_analyzed

    # --- Mod-Time Based Indexing Logic (Fallback) ---

    def _run_mod_time_based_indexing(self, db_manager: DbManager) -> Tuple[int, int, int, int]:
        """Performs indexing based on file modification times. Returns counts: proc, add, del, analyzed"""
        logger.info("Running indexing based on modification times...")
        analysis_result = self._scan_and_analyze_files_modtime(db_manager)
        total_analyzed = analysis_result["total_files_on_disk"]

        if self._is_cancelled: raise InterruptedError("Cancelled after mod-time analysis.")

        deleted_count = self._delete_files_from_index_modtime(db_manager, analysis_result["files_to_delete"])
        if self._is_cancelled: raise InterruptedError("Cancelled during mod-time deletion.")

        processed_count, added_chunks = self._process_files_list(
            db_manager, analysis_result["files_to_process"], analysis_result["indexed_paths_set"]
        )
        return processed_count, added_chunks, deleted_count, total_analyzed

    # --- Shared Helper Methods ---

    def _initialize_embedder(self) -> Tuple[Any, int]:
        """Loads the embedding model and determines its dimension."""
        self._emit_progress(f"Loading embedding model: {self.embedding_model_name}...")
        embedder = EmbeddingManager.get_embedder(self.embedding_model_name)
        if not embedder: raise ValueError(f"Failed to load embedder '{self.embedding_model_name}'.")
        embedding_dim = embedder.dimension
        if not isinstance(embedding_dim, int) or embedding_dim <= 0: raise ValueError(f"Invalid dimension ({embedding_dim}).")
        self._emit_progress(f"Embedding model loaded (Dimension: {embedding_dim}).")
        return embedder, embedding_dim

    def _scan_and_analyze_files_modtime(self, db_manager: DbManager) -> Dict[str, Any]:
        """Scans based on mod times (Fallback logic)."""
        self._emit_progress(f"Scanning (mod-time) for files ({', '.join(self.allowed_extensions)}) in: {self.notes_folder_path}")
        all_target_files_paths = FileHandler.find_files_by_extension(str(self.notes_folder_path), self.allowed_extensions)
        total_files_on_disk = len(all_target_files_paths)

        self._emit_progress("Checking existing index status...")
        indexed_files_status = db_manager.get_indexed_files()
        indexed_paths_set = set(indexed_files_status.keys())

        if not all_target_files_paths:
            logger.info("No target files found.")
            return {"files_to_process": [], "files_to_delete": indexed_paths_set, "total_files_on_disk": 0, "indexed_paths_set": indexed_paths_set}

        current_paths_on_disk_set = {str(p.resolve()) for p in all_target_files_paths}
        logger.debug(f"Found {len(current_paths_on_disk_set)} files on disk, {len(indexed_paths_set)} files in index.")

        files_to_process, files_to_delete = self._analyze_file_changes( # Reuses analysis logic
            all_target_files_paths, indexed_files_status, current_paths_on_disk_set
        )

        logger.info(f"Mod-time analysis complete: {len(files_to_process)} process, {len(files_to_delete)} delete.")
        return {"files_to_process": files_to_process, "files_to_delete": files_to_delete, "total_files_on_disk": total_files_on_disk, "indexed_paths_set": indexed_paths_set}

    def _analyze_file_changes(self, disk_files: List[Path], index_status: Dict[str, float], disk_paths_set: Set[str]) -> Tuple[List[Path], Set[str]]:
        """Compares files on disk with index status to determine changes (Used by both strategies)."""
        index_paths_set = set(index_status.keys())
        files_to_process: List[Path] = []
        files_to_delete: Set[str] = index_paths_set - disk_paths_set

        self._emit_progress(f"Analyzing {len(disk_files)} files for changes...")
        for i, file_path_obj in enumerate(disk_files):
            if self._is_cancelled: raise InterruptedError("Cancelled during analysis.")
            if (i + 1) % 50 == 0: self._emit_progress(f"Analyzing file {i+1}/{len(disk_files)}...")

            file_path_str = str(file_path_obj.resolve())
            current_mod_time = FileHandler.get_last_modified(file_path_obj)
            if current_mod_time is None:
                logger.warning(f"Skipping {file_path_obj.name}: Can't get mod time."); files_to_delete.discard(file_path_str); continue

            indexed_mod_time = index_status.get(file_path_str, 0)
            is_modified = current_mod_time > (indexed_mod_time + 1)

            if file_path_str not in index_paths_set: files_to_process.append(file_path_obj)
            elif is_modified: files_to_process.append(file_path_obj)
        return files_to_process, files_to_delete

    def _delete_files_from_index(self, db_manager: DbManager, deleted_rel_paths: List[str], repo_root: Path) -> int:
        """Deletes files (given as relative paths) from the index."""
        count = len(deleted_rel_paths); deleted_count = 0
        if not count: return 0
        self._emit_progress(f"Removing {count} deleted file(s) from index...")
        for rel_path in deleted_rel_paths:
            if self._is_cancelled: raise InterruptedError("Cancelled during file deletion.")
            abs_path_str = str(repo_root / Path(rel_path))
            try: db_manager.delete_chunks_for_file(abs_path_str); deleted_count += 1
            except Exception as e: logger.error(f"Failed to delete {rel_path}: {e}", exc_info=True)
            if deleted_count % 10 == 0: self._emit_progress(f"Removed {deleted_count}/{count} files...")
        self._emit_progress(f"Finished removing {deleted_count} deleted file(s).")
        return deleted_count

    def _delete_files_from_index_modtime(self, db_manager: DbManager, deleted_abs_paths: Set[str]) -> int:
        """Deletes files (given as absolute paths) from the index."""
        count = len(deleted_abs_paths); deleted_count = 0
        if not count: return 0
        self._emit_progress(f"Removing {count} missing file(s) from index...")
        for abs_path_str in deleted_abs_paths:
             if self._is_cancelled: raise InterruptedError("Cancelled during file deletion.")
             try: db_manager.delete_chunks_for_file(abs_path_str); deleted_count += 1
             except Exception as e: logger.error(f"Failed to delete {abs_path_str}: {e}", exc_info=True)
             if deleted_count % 10 == 0: self._emit_progress(f"Removed {deleted_count}/{count} files...")
        self._emit_progress(f"Finished removing {deleted_count} missing file(s).")
        return deleted_count

    def _rename_files_in_index(self, db_manager: DbManager, renamed_paths: List[Tuple[str, str]], repo_root: Path) -> int:
        """Updates file paths in the index for renamed files."""
        renamed_count = 0
        if not renamed_paths: return 0
        self._emit_progress(f"Processing {len(renamed_paths)} renamed files...")
        for old_rel_path, new_rel_path in renamed_paths:
             if self._is_cancelled: raise InterruptedError("Cancelled during file rename.")
             old_abs_path_str = str(repo_root / Path(old_rel_path))
             new_abs_path_str = str(repo_root / Path(new_rel_path))
             if db_manager.rename_file_path(old_abs_path_str, new_abs_path_str):
                 renamed_count += 1
        return renamed_count

    def _process_files_list(self, db_manager: DbManager, files_to_process: List[Path], indexed_paths_set: Set[str]) -> Tuple[int, int]:
        """Helper to process a list of files (absolute paths)."""
        total_processed = 0; total_added_chunks = 0
        total_files = len(files_to_process)
        if not total_files: return 0, 0
        self._emit_progress(f"Processing {total_files} files...")

        for i, file_path_obj in enumerate(files_to_process):
             if self._is_cancelled: raise InterruptedError("Cancelled during file processing.")
             self._emit_progress(f"Processing file {i+1}/{total_files}: {file_path_obj.name}")
             mod_time = FileHandler.get_last_modified(file_path_obj)
             if mod_time is None: logger.warning(f"Skipping {file_path_obj.name}: Cannot get mod time."); continue
             try:
                 # Pass the absolute path string for checking in indexed_paths_set
                 abs_path_str = str(file_path_obj.resolve())
                 # Check if path exists in set (using abs path string)
                 file_exists_in_index = abs_path_str in indexed_paths_set
                 # Pass the file path object and existence flag to single file processor
                 chunks_added = self._process_single_file(file_path_obj, mod_time, db_manager, file_exists_in_index)
                 total_added_chunks += chunks_added; total_processed += 1
             except InterruptedError: raise
             except Exception as e: logger.error(f"Failed processing {file_path_obj.name}: {e}", exc_info=True)

        logger.info(f"Finished file processing. Processed: {total_processed}, Chunks Added: {total_added_chunks}")
        return total_processed, total_added_chunks

    # Modified to accept simple boolean for existence check
    def _process_single_file(self, file_path_obj: Path, current_mod_time: float, db_manager: DbManager, file_existed_in_index: bool) -> int:
        """Orchestrates processing for one file."""
        if self._is_cancelled: raise InterruptedError("Cancelled.")
        file_path_str = str(file_path_obj.resolve()) # Use resolved path for consistency

        try:
            if file_existed_in_index:
                logger.debug(f"Deleting existing chunks for modified file: {file_path_obj.name}")
                db_manager.delete_chunks_for_file(file_path_str) # Delete using resolved path
                if self._is_cancelled: raise InterruptedError("Cancelled.")

            content = FileHandler.read_file_content(file_path_obj)
            if content is None or not content.strip(): logger.warning(f"Skipping {file_path_obj.name}: No content."); return 0
            chunks = split_text(content)
            if not chunks: logger.warning(f"No chunks for {file_path_obj.name}."); return 0

            all_embeddings = self._embed_file_chunks(file_path_obj, chunks)
            if all_embeddings is None: return 0 # Embedding failed
            if self._is_cancelled: raise InterruptedError("Cancelled.")

            chunks_data = self._prepare_chunk_data(file_path_str, current_mod_time, chunks, all_embeddings)
            if chunks_data:
                self._store_chunk_data(db_manager, file_path_obj.name, chunks_data)
                return len(chunks_data)
            else: logger.warning(f"No valid chunks to store for {file_path_obj.name}."); return 0

        except InterruptedError: raise
        except sqlite3.Error as db_error: logger.error(f"DB error processing {file_path_obj.name}: {db_error}", exc_info=True); return 0
        except Exception as e: logger.error(f"Unexpected error processing {file_path_obj.name}: {e}", exc_info=True); return 0


    def _embed_file_chunks(self, file_path_obj: Path, chunks: List[str]) -> Optional[np.ndarray]:
        """Generates embeddings for file chunks."""
        num_chunks = len(chunks); self._emit_progress(f"Embedding {num_chunks} chunk(s) for {file_path_obj.name}...")
        try:
            all_embeddings = self.embedder.embed_batch(chunks, batch_size=EMBEDDING_BATCH_SIZE)
            if not isinstance(all_embeddings, np.ndarray) or all_embeddings.shape != (num_chunks, self.embedding_dim):
                 logger.error(f"Embedding {file_path_obj.name}: Unexpected shape {all_embeddings.shape if isinstance(all_embeddings, np.ndarray) else type(all_embeddings)}. Expected ({num_chunks}, {self.embedding_dim})"); return None
            return all_embeddings
        except (RuntimeError, ConnectionError, ConnectionAbortedError) as e: logger.error(f"Embedding failed for {file_path_obj.name}: {e}", exc_info=True); self._emit_progress(f"ERROR embedding {file_path_obj.name}."); return None
        except Exception as e: logger.error(f"Unexpected error embedding {file_path_obj.name}: {e}", exc_info=True); self._emit_progress(f"ERROR embedding {file_path_obj.name}."); return None

    def _prepare_chunk_data(self, file_path_str: str, mod_time: float, chunks: List[str], embeddings: np.ndarray) -> List[Tuple[str, int, str, np.ndarray, float]]:
        """Prepares data tuples for DB insertion."""
        chunks_data = []
        for i, (text, emb) in enumerate(zip(chunks, embeddings)):
            if not isinstance(emb, np.ndarray): logger.error(f"INTERNAL: Non-numpy embedding index {i} for {file_path_str}."); continue
            chunks_data.append((file_path_str, i, text, emb.astype(np.float32), mod_time))
        return chunks_data

    def _store_chunk_data(self, db_manager: DbManager, file_name: str, chunks_data: List[Tuple[str, int, str, np.ndarray, float]]):
        """Stores prepared chunk data in the database."""
        if not chunks_data: return
        self._emit_progress(f"Storing {len(chunks_data)} chunk(s) for {file_name}...")
        db_manager.add_chunks_batch(chunks_data)

    def _generate_final_report(self, start_time: float, total_files_analyzed: int,
                               processed_files: int, # removed total_intended_to_process
                               added_chunks: int, deleted_files: int, renamed_files: int, strategy: str):
        """Creates and emits the final summary message."""
        end_time = time.time(); duration = end_time - start_time
        summary = (f"Indexing ({strategy}) finished in {duration:.2f}s. "
                   f"Analyzed: {total_files_analyzed} files. "
                   f"Processed: {processed_files} files. "
                   f"Added/Updated: {added_chunks} chunks. "
                   f"Deleted: {deleted_files} files. "
                   f"Renamed: {renamed_files} files.")
        self._emit_finished(True, summary)
    def _get_all_tracked_files_in_commit(self, repo_root: Path) -> List[Path]:
        """Gets all tracked files in HEAD matching extensions/folder."""
        if not self.git_helper or not self.git_helper.repo:
            logger.warning("Git helper not available, cannot get tracked files.")
            return []

        relevant_files = []
        self._emit_progress("Listing all tracked files in repository...")
        try:
            # Iterate through all files tracked in the current commit (HEAD)
            # Use ls_files() for potentially better performance than traversing tree
            # tracked_files_rel = self.git_helper.repo.git.ls_files().splitlines()
            # for rel_path_str in tracked_files_rel:
            #      abs_path = (repo_root / Path(rel_path_str)).resolve()
            #      if abs_path.is_relative_to(self.notes_folder_path) and abs_path.suffix.lower() in self.allowed_extensions:
            #          relevant_files.append(abs_path)

            # Alternative: Traverse tree (might be easier to ensure correct commit)
            for item in self.git_helper.repo.head.commit.tree.traverse():
                if item.type == 'blob': # It's a file
                    abs_path = (repo_root / Path(item.path)).resolve()
                    # Check if within notes folder scope and has allowed extension
                    # Use is_relative_to for robust path checking
                    try: # Add try-except for is_relative_to in case paths are on different drives etc.
                        is_relevant = (abs_path.is_relative_to(self.notes_folder_path) and
                                        abs_path.suffix.lower() in self.allowed_extensions)
                    except ValueError:
                        is_relevant = False # Not relative if error occurs

                    if is_relevant:
                        relevant_files.append(abs_path)
            logger.info(f"Found {len(relevant_files)} relevant tracked files for initial indexing.")
            return relevant_files

        except Exception as e:
            logger.error(f"Error getting tracked files from Git: {e}", exc_info=True)
            return [] # Return empty list on error
# Inside class Indexer in app/core/indexing/indexer.py:

    # --- Add this missing method ---
    def _filter_git_changes(self, changes: Dict[str, List[Any]], repo_root: Path) -> Dict[str, List[Any]]:
        """Filters git changes dictionary to include only files relevant to indexing."""
        filtered_changes: Dict[str, List[Any]] = {'A': [], 'D': [], 'M': [], 'R': []}
        notes_folder_abs = self.notes_folder_path # Absolute path of notes folder

        # Determine notes folder path relative to repo root, if applicable
        notes_folder_rel = None
        try:
            # Only calculate relative path if notes folder is truly inside repo root
            if notes_folder_abs.is_relative_to(repo_root):
                notes_folder_rel = notes_folder_abs.relative_to(repo_root)
        except ValueError:
            # Happens if notes_folder_abs and repo_root are on different drives (Windows)
            # or one is not a subpath of the other. Treat as not relative.
            logger.warning(f"Notes folder '{notes_folder_abs}' is not inside Git repo '{repo_root}'. Filtering might be incomplete.")
            # In this case, filtering based on relative path won't work reliably.
            # We might need to rely only on absolute path checks or extension checks.
            # For now, we'll proceed, but be aware of this edge case.
            pass

        # Iterate through Added, Deleted, Modified, Renamed files from Git diff
        for change_type, paths_or_tuples in changes.items():
            for item in paths_or_tuples:
                # Determine the relevant relative path(s) reported by Git to check
                paths_to_check_rel_str: List[str] = []
                if change_type in ['A', 'M']: paths_to_check_rel_str.append(item)
                elif change_type == 'D': paths_to_check_rel_str.append(item)
                elif change_type == 'R': paths_to_check_rel_str.extend(item) # Check both old and new path

                is_relevant = False
                for rel_path_str in paths_to_check_rel_str:
                    rel_path = Path(rel_path_str)
                    abs_path = (repo_root / rel_path).resolve() # Construct absolute path

                    # Check 1: Is it within the target notes folder?
                    # Use startswith on the string representation for cross-drive safety
                    path_in_notes_folder = str(abs_path).startswith(str(notes_folder_abs))

                    # Check 2: Does it have an allowed extension?
                    extension_allowed = abs_path.suffix.lower() in self.allowed_extensions

                    if path_in_notes_folder and extension_allowed:
                        is_relevant = True
                        break # Found a relevant path for this change item, no need to check more

                if is_relevant:
                    # Add the original item (path string or tuple) to the filtered dict
                    filtered_changes[change_type].append(item)
                    logger.debug(f"Relevant change ({change_type}): {item}")

        logger.debug(f"Filtered Git changes: A={len(filtered_changes['A'])}, D={len(filtered_changes['D'])}, M={len(filtered_changes['M'])}, R={len(filtered_changes['R'])}")
        return filtered_changes
    # -----------------------------

    def _filter_untracked_files(self, untracked_rel_paths: List[str], repo_root: Path) -> List[Path]:
        """
        Filters a list of untracked files (relative paths) to include only those
        relevant to the notes folder and allowed extensions.

        Args:
            untracked_rel_paths: List of file paths relative to the repo root.
            repo_root: Absolute path to the repository root.

        Returns:
            List of absolute Paths for relevant untracked files.
        """
        relevant_untracked_abs = []
        notes_folder_abs = self.notes_folder_path # Absolute path

        for rel_path_str in untracked_rel_paths:
            try:
                abs_path = (repo_root / Path(rel_path_str)).resolve()

                # Check 1: Is it within the target notes folder?
                # Use startswith on the string representation for cross-drive safety
                path_in_notes_folder = str(abs_path).startswith(str(notes_folder_abs))

                # Check 2: Does it have an allowed extension?
                extension_allowed = abs_path.suffix.lower() in self.allowed_extensions

                if path_in_notes_folder and extension_allowed:
                    relevant_untracked_abs.append(abs_path)
                    logger.debug(f"Relevant untracked file found: {abs_path}")

            except Exception as e:
                # Catch potential errors resolving paths etc.
                logger.warning(f"Error processing untracked file path '{rel_path_str}': {e}")
                continue # Skip problematic path

        logger.debug(f"Found {len(relevant_untracked_abs)} relevant untracked files.")
        return relevant_untracked_abs
