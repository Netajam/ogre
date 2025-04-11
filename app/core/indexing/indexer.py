import time
import traceback
import sqlite3
from pathlib import Path
from typing import Optional, Callable, Tuple, List, Set, Any, Dict
import numpy as np

from app.core.database.db_manager import DbManager
from app.core.embedding.embedding_manager import EmbeddingManager
from app.core.indexing.file_handler import FileHandler
from app.core.indexing.chunking import split_text

try:
    from app.utils.logging_setup import logger
except ImportError:
     import logging
     logger = logging.getLogger(__name__)
     logger.warning("Using fallback logger for Indexer.")


EMBEDDING_BATCH_SIZE = 64 # Process N chunks at a time for embedding API/model

class Indexer:
    """Orchestrates the file discovery, chunking, embedding, and indexing process."""

    def __init__(self,
                 notes_folder: str,
                 db_path: Path,
                 embedding_model_name: str,
                 allowed_extensions: List[str],
                 progress_callback: Optional[Callable[[str], None]] = None,
                 finished_callback: Optional[Callable[[bool, str], None]] = None):
        """Initializes the Indexer."""
        self.notes_folder = notes_folder
        self.db_path = db_path
        self.embedding_model_name = embedding_model_name
        self.allowed_extensions = allowed_extensions
        self.progress_callback = progress_callback
        self.finished_callback = finished_callback
        self._is_cancelled = False
        self.embedder: Optional[Any] = None
        self.embedding_dim: Optional[int] = None
        logger.debug(f"Indexer initialized. Folder: '{notes_folder}', DB: '{db_path}', Model: '{embedding_model_name}', Extensions: {allowed_extensions}")

    def cancel(self):
        """Signals the indexing process to stop gracefully at the next check."""
        if not self._is_cancelled:
            self._emit_progress("Cancellation requested...")
            self._is_cancelled = True
            logger.info("Cancellation flag set for Indexer.")

    def run_indexing(self):
        """Runs the full indexing process using helper methods."""
        start_time = time.time()
        self._is_cancelled = False
        processed_files_count = 0
        added_chunks_count = 0
        deleted_files_count = 0
        total_files_on_disk = 0
        db_manager: Optional[DbManager] = None

        try:
            self._emit_progress("Initializing indexing...")
            if not self.allowed_extensions: raise ValueError("No allowed file extensions configured.")

            self.embedder, self.embedding_dim = self._initialize_embedder()
            if self._is_cancelled: raise InterruptedError("Cancelled after model load.")

            with DbManager(self.db_path) as db_manager:
                db_manager.initialize_db(self.embedding_dim)
                if self._is_cancelled: raise InterruptedError("Cancelled after DB initialization.")

                analysis_result = self._scan_and_analyze_files(db_manager)
                if self._is_cancelled: raise InterruptedError("Cancelled after file analysis.")

                if analysis_result["total_files_on_disk"] == 0 and not analysis_result["files_to_delete"]:
                    self._emit_finished(True, "No target files found. Indexing complete.")
                    return

                deleted_files_count = self._remove_deleted_files_from_index(db_manager, analysis_result["files_to_delete"])
                if self._is_cancelled: raise InterruptedError("Cancelled during file deletion.")

                processed_files_count, added_chunks_count = self._process_changed_files(
                    db_manager,
                    analysis_result["files_to_process"],
                    analysis_result["indexed_paths_set"]
                )

            db_manager = None

            if self._is_cancelled: raise InterruptedError("Cancelled by user.")

            self._generate_final_report(start_time, total_files_on_disk, processed_files_count,
                                        len(analysis_result["files_to_process"]), added_chunks_count, deleted_files_count)

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
            if db_manager is not None and db_manager.conn is not None:
                logger.warning("Closing DB connection in finally block.")
                db_manager.close()
            self.embedder = None # Clear embedder reference
            logger.debug("Indexer run_indexing method finished.")

    def _emit_progress(self, message: str):
        """Safely emits a progress message via callback and logs it."""
        if self.progress_callback:
            try: self.progress_callback(message)
            except Exception as e: logger.error(f"Error in progress callback: {e}")
        logger.info(f"Progress: {message}")


    def _emit_finished(self, success: bool, message: str):
        """Safely emits the finished signal via callback and logs it."""
        if self.finished_callback:
            try: self.finished_callback(success, message)
            except Exception as e: logger.error(f"Error in finished callback: {e}")
        logger.info(f"Finished: Success={success}, Message='{message}'")


    def _initialize_embedder(self) -> Tuple[Any, int]:
        """Loads the embedding model and determines its dimension."""
        self._emit_progress(f"Loading embedding model: {self.embedding_model_name}...")
        embedder = EmbeddingManager.get_embedder(self.embedding_model_name)
        if not embedder: raise ValueError(f"Failed to load embedder '{self.embedding_model_name}'.")
        embedding_dim = embedder.dimension
        if not isinstance(embedding_dim, int) or embedding_dim <= 0: raise ValueError(f"Invalid dimension ({embedding_dim}).")
        self._emit_progress(f"Embedding model loaded (Dimension: {embedding_dim}).")
        return embedder, embedding_dim

    def _scan_and_analyze_files(self, db_manager: DbManager) -> Dict[str, Any]:
        """
        Scans for files, gets index status, analyzes changes.

        Returns:
            Dict containing: 'files_to_process', 'files_to_delete',
                             'total_files_on_disk', 'indexed_paths_set'.
        """
        self._emit_progress(f"Scanning for files ({', '.join(self.allowed_extensions)}) in: {self.notes_folder}")
        all_target_files_paths = FileHandler.find_files_by_extension(self.notes_folder, self.allowed_extensions)
        total_files_on_disk = len(all_target_files_paths)

        self._emit_progress("Checking existing index status...")
        indexed_files_status = db_manager.get_indexed_files()
        indexed_paths_set = set(indexed_files_status.keys())

        if not all_target_files_paths:
            logger.info("No target files found.")
            return {
                "files_to_process": [],
                "files_to_delete": indexed_paths_set,
                "total_files_on_disk": 0,
                "indexed_paths_set": indexed_paths_set
            }

        current_paths_on_disk_set = {str(p.resolve()) for p in all_target_files_paths}
        logger.debug(f"Found {len(current_paths_on_disk_set)} files on disk, {len(indexed_paths_set)} files in index.")

        files_to_process, files_to_delete = self._analyze_file_changes(
            all_target_files_paths, indexed_files_status, current_paths_on_disk_set
        )

        logger.info(f"Analysis complete: {len(files_to_process)} files to process, {len(files_to_delete)} files to delete.")
        return {
            "files_to_process": files_to_process,
            "files_to_delete": files_to_delete,
            "total_files_on_disk": total_files_on_disk,
            "indexed_paths_set": indexed_paths_set
        }

    def _analyze_file_changes(self,
                              disk_files: List[Path],
                              index_status: Dict[str, float],
                              disk_paths_set: Set[str]) -> Tuple[List[Path], Set[str]]:
        """
        Compares files on disk with index status to determine changes.

        Args:
            disk_files: List of Path objects found on disk.
            index_status: Dictionary {path_str: mod_time} from the database.
            disk_paths_set: Set of resolved paths found on disk.

        Returns:
            Tuple (files_to_process: List[Path], files_to_delete: Set[str])
        """
        index_paths_set = set(index_status.keys())
        files_to_process: List[Path] = []
        files_to_delete: Set[str] = index_paths_set - disk_paths_set

        self._emit_progress(f"Analyzing {len(disk_files)} found files for changes...")
        processed_count = 0
        for file_path_obj in disk_files:
            processed_count += 1
            if self._is_cancelled: raise InterruptedError("Cancelled during file analysis loop.")

            file_path_str = str(file_path_obj.resolve())
            if processed_count % 50 == 0: self._emit_progress(f"Analyzing file {processed_count}/{len(disk_files)}...")

            current_mod_time = FileHandler.get_last_modified(file_path_obj)
            if current_mod_time is None:
                logger.warning(f"Skipping analysis for {file_path_obj.name}: Can't get mod time.")
                files_to_delete.discard(file_path_str) # Don't delete if we can't check it
                continue

            indexed_mod_time = index_status.get(file_path_str, 0)
            is_modified = current_mod_time > (indexed_mod_time + 1) # Mod time tolerance

            if file_path_str not in index_paths_set:
                logger.debug(f"File to add: {file_path_obj.name}")
                files_to_process.append(file_path_obj)
            elif is_modified:
                logger.debug(f"File to update: {file_path_obj.name} (Disk:{current_mod_time:.0f}, Index:{indexed_mod_time:.0f})")
                files_to_process.append(file_path_obj)

        return files_to_process, files_to_delete


    def _remove_deleted_files_from_index(self, db_manager: DbManager, files_to_delete_paths: Set[str]) -> int:
        """Removes records for given file paths from the index."""
        count = len(files_to_delete_paths)
        if not count: return 0
        self._emit_progress(f"Removing {count} deleted/missing file(s) from index...")
        processed = 0
        for path in files_to_delete_paths:
             if self._is_cancelled: raise InterruptedError("Cancelled during file deletion.")
             try:
                 logger.debug(f"Deleting chunks for: {path}")
                 db_manager.delete_chunks_for_file(path)
                 processed += 1
                 if processed % 10 == 0: self._emit_progress(f"Removed {processed}/{count} files...")
             except Exception as e: logger.error(f"Failed to delete {path}: {e}", exc_info=True); self._emit_progress(f"ERROR deleting {Path(path).name}.")
        self._emit_progress(f"Finished removing {processed} deleted file(s).")
        return processed


    def _process_changed_files(self, db_manager: DbManager, files_to_process: List[Path], indexed_paths_set: Set[str]) -> Tuple[int, int]:
        """Processes new and modified files by calling helpers for each file."""
        total_to_process = len(files_to_process); total_processed = 0; total_added_chunks = 0
        if not total_to_process: return 0, 0
        self._emit_progress(f"Processing {total_to_process} new/modified file(s)...")
        for i, file_path_obj in enumerate(files_to_process):
            if self._is_cancelled: raise InterruptedError("Cancelled during file processing loop.")
            mod_time = FileHandler.get_last_modified(file_path_obj)
            if mod_time is None: logger.warning(f"Skipping {file_path_obj.name}: Can't get mod time."); continue
            self._emit_progress(f"Processing file {i+1}/{total_to_process}: {file_path_obj.name}")
            try:
                chunks_added = self._process_single_file(file_path_obj, mod_time, db_manager, indexed_paths_set)
                total_added_chunks += chunks_added; total_processed += 1
            except InterruptedError: raise
            except Exception as e: logger.error(f"Failed processing {file_path_obj.name}: {e}", exc_info=True); self._emit_progress(f"CRITICAL ERROR processing {file_path_obj.name}.")
            if self._is_cancelled: raise InterruptedError("Cancelled after processing a file.")
        logger.info(f"Finished processing changed files. Processed: {total_processed}, Chunks Added: {total_added_chunks}")
        return total_processed, total_added_chunks


    def _process_single_file(self, file_path_obj: Path, current_mod_time: float, db_manager: DbManager, indexed_paths_set: Set[str]) -> int:
        """Orchestrates reading, chunking, embedding, and storing for one file."""
        file_path_str = str(file_path_obj.resolve())
        if self._is_cancelled: raise InterruptedError("Cancelled before processing file.")

        try:
            if file_path_str in indexed_paths_set:
                logger.debug(f"Deleting existing chunks for modified file: {file_path_obj.name}")
                db_manager.delete_chunks_for_file(file_path_str)
                if self._is_cancelled: raise InterruptedError("Cancelled after deleting old chunks.")

            content = FileHandler.read_file_content(file_path_obj)
            if content is None or not content.strip():
                logger.warning(f"Skipping file {file_path_obj.name}: No readable content.")
                return 0
            chunks = split_text(content)
            if not chunks:
                logger.warning(f"No chunks generated for file: {file_path_obj.name}.")
                return 0

            all_embeddings = self._embed_file_chunks(file_path_obj, chunks)
            if all_embeddings is None:
                return 0
            if self._is_cancelled: raise InterruptedError("Cancelled after embedding.")

            chunks_data = self._prepare_chunk_data(file_path_str, current_mod_time, chunks, all_embeddings)
            if chunks_data:
                self._store_chunk_data(db_manager, file_path_obj.name, chunks_data)
                return len(chunks_data)
            else:
                logger.warning(f"No valid chunks/embeddings prepared for {file_path_obj.name}.")
                return 0

        except InterruptedError: raise
        except sqlite3.Error as db_error:
             logger.error(f"Database error processing {file_path_obj.name}: {db_error}", exc_info=True)
             self._emit_progress(f"ERROR storing chunks for {file_path_obj.name}.")
             return 0
        except Exception as e:
             logger.error(f"Unexpected error processing {file_path_obj.name}: {e}", exc_info=True)
             self._emit_progress(f"ERROR processing {file_path_obj.name}.")
             return 0

    def _embed_file_chunks(self, file_path_obj: Path, chunks: List[str]) -> Optional[np.ndarray]:
        """Generates embeddings for the chunks of a single file, handling errors."""
        num_chunks = len(chunks)
        self._emit_progress(f"Embedding {num_chunks} chunk(s) for {file_path_obj.name}...")
        try:
            all_embeddings = self.embedder.embed_batch(chunks, batch_size=EMBEDDING_BATCH_SIZE)
            if not isinstance(all_embeddings, np.ndarray) or all_embeddings.shape != (num_chunks, self.embedding_dim):
                 logger.error(f"Embedding result {file_path_obj.name}: Unexpected shape {all_embeddings.shape if isinstance(all_embeddings, np.ndarray) else type(all_embeddings)}. Expected ({num_chunks}, {self.embedding_dim})")
                 return None
            return all_embeddings
        except (RuntimeError, ConnectionError, ConnectionAbortedError) as embed_error:
             logger.error(f"Embedding failed for {file_path_obj.name}: {embed_error}", exc_info=True)
             self._emit_progress(f"ERROR embedding {file_path_obj.name}. Skipping file.")
             return None
        except Exception as embed_error:
             logger.error(f"Unexpected error embedding {file_path_obj.name}: {embed_error}", exc_info=True)
             self._emit_progress(f"ERROR embedding {file_path_obj.name}. Skipping file.")
             return None

    def _prepare_chunk_data(self, file_path_str: str, mod_time: float, chunks: List[str], embeddings: np.ndarray) -> List[Tuple[str, int, str, np.ndarray, float]]:
        """Prepares the list of tuples for database insertion."""
        chunks_data: List[Tuple[str, int, str, np.ndarray, float]] = []
        for chunk_index, (chunk_text, embedding) in enumerate(zip(chunks, embeddings)):
            if not isinstance(embedding, np.ndarray):
                logger.error(f"INTERNAL: Non-numpy embedding at index {chunk_index} for {file_path_str}. Skipping.")
                continue
            chunks_data.append(
                (file_path_str, chunk_index, chunk_text, embedding.astype(np.float32), mod_time)
            )
        return chunks_data

    def _store_chunk_data(self, db_manager: DbManager, file_name: str, chunks_data: List[Tuple[str, int, str, np.ndarray, float]]):
        """Stores the prepared chunk data in the database."""
        if not chunks_data: return # Should not happen if called correctly, but safe check
        self._emit_progress(f"Storing {len(chunks_data)} chunk(s) for {file_name}...")
        db_manager.add_chunks_batch(chunks_data) # Let DbManager handle exceptions

    def _generate_final_report(self, start_time: float, total_files_analyzed: int,
                               processed_files: int, total_intended_to_process: int,
                               added_chunks: int, deleted_files: int):
        """Creates and emits the final summary message."""
        end_time = time.time()
        duration = end_time - start_time
        summary_message = (f"Indexing finished in {duration:.2f}s. "
                           f"Analyzed {total_files_analyzed} files. "
                           f"Processed {processed_files}/{total_intended_to_process} new/modified files. "
                           f"Added/Updated {added_chunks} chunks. "
                           f"Removed {deleted_files} files from index.")
        self._emit_finished(True, summary_message)