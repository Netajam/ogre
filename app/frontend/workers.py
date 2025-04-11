# app/frontend/workers.py
from pathlib import Path
from typing import Optional, List
from PyQt6.QtCore import QObject, pyqtSignal

# Need logger, Indexer, QueryService, ResultItem etc.
from app.utils.logging_setup import logger
from app.core.indexing.indexer import Indexer
from app.core.services.query_service import QueryService
from app.core.services.result_item import ResultItem


# --- Background Worker for Indexing ---
class IndexerWorker(QObject):
    finished = pyqtSignal(bool, str)
    progress = pyqtSignal(str)

    def __init__(self, notes_folder: str, db_path: Path, embedding_model_name: str, allowed_extensions: List[str]):
        super().__init__()
        self.notes_folder = notes_folder
        self.db_path = db_path
        self.embedding_model_name = embedding_model_name
        self.allowed_extensions = allowed_extensions
        self.indexer: Optional[Indexer] = None
        self._is_running = False

    def run(self):
        if self._is_running: logger.warning("IndexerWorker already running."); return
        self._is_running = True; logger.info("IndexerWorker started.")
        try:
            self.indexer = Indexer(
                notes_folder=self.notes_folder, db_path=self.db_path,
                embedding_model_name=self.embedding_model_name, allowed_extensions=self.allowed_extensions,
                progress_callback=self.progress.emit, finished_callback=self.finished.emit
            )
            self.indexer.run_indexing()
        except Exception as e:
             error_msg = f"Critical error in IndexerWorker: {e}"; logger.error(error_msg, exc_info=True)
             try: self.finished.emit(False, error_msg)
             except Exception as sig_e: logger.error(f"Failed to emit finished signal: {sig_e}")
        finally:
             logger.info("IndexerWorker finished."); self.indexer = None; self._is_running = False

    def stop(self):
        if self.indexer: logger.info("Requesting indexer cancellation..."); self.indexer.cancel()
        else: logger.warning("Stop requested but no active Indexer.")


# --- Background Worker for Querying ---
class QueryWorker(QObject):
    results_ready = pyqtSignal(list)
    llm_answer_ready = pyqtSignal(str)
    error_occurred = pyqtSignal(str)

    def __init__(self, query_text: str, db_path: Path, embedding_model_name: str, llm_model_name: str, notes_folder: str, mode: str):
        super().__init__()
        self.query_text = query_text
        self.db_path = db_path
        self.embedding_model_name = embedding_model_name
        self.llm_model_name = llm_model_name
        self.notes_folder = notes_folder
        self.mode = mode
        self._is_cancelled = False # Cancellation flag

    def run(self):
        logger.info(f"QueryWorker started for query: '{self.query_text[:50]}...', mode: {self.mode}")
        query_service = None # Define outside try for potential use in finally? No, created inside.
        try:
            query_service = QueryService( # Create service instance here
                db_path=self.db_path, embedding_model_name=self.embedding_model_name,
                llm_model_name=self.llm_model_name, notes_folder=self.notes_folder
            )

            # Step 1: Always search chunks
            if self._is_cancelled: return
            results: List[ResultItem] = query_service.search_chunks(self.query_text)
            if self._is_cancelled: return
            self.results_ready.emit(results)
            logger.info(f"QueryWorker found {len(results)} results, emitted results_ready.")

            # Step 2: Generate LLM Answer only if mode is auto_answer
            if self.mode == "auto_answer":
                if results:
                    if self._is_cancelled: return
                    logger.info("QueryWorker generating LLM answer (auto mode).")
                    llm_answer = query_service.generate_llm_answer(self.query_text, results)
                    if self._is_cancelled: return
                    self.llm_answer_ready.emit(llm_answer)
                    logger.info("QueryWorker finished LLM answer (auto mode).")
                else:
                    if self._is_cancelled: return
                    logger.info("QueryWorker skipping LLM answer (auto mode, no results).")
                    self.llm_answer_ready.emit("No relevant context found in notes to generate an answer.")
            else:
                # For chunks_only or edit_prompt modes, worker is done after search.
                # Emit informative signal on llm_answer_ready channel for consistency in thread cleanup.
                logger.info(f"QueryWorker finished after chunk search (mode: {self.mode}).")
                self.llm_answer_ready.emit(f"[Processing stopped after chunk retrieval - Mode: {self.mode}]")

        except Exception as e:
            if not self._is_cancelled: # Only report error if not cancelled
                 error_msg = f"Error during query processing: {e}"; logger.error(error_msg, exc_info=True)
                 self.error_occurred.emit(error_msg)
            else:
                 logger.info("Query processing aborted due to cancellation.")
        finally:
            logger.info("QueryWorker run finished or cancelled.")

    def stop(self):
        """Signals the worker to stop processing *between* steps."""
        logger.info("QueryWorker cancellation requested.")
        self._is_cancelled = True
        # Note: Doesn't stop active network calls or DB queries instantly.


# --- Optional: Worker for LLM Generation from Edited Prompt ---
class LLMGenWorker(QObject):
    llm_answer_ready = pyqtSignal(str)
    error_occurred = pyqtSignal(str)

    def __init__(self, final_prompt: str, db_path: Path, emb_model: str, llm_model: str, notes_folder: str):
        super().__init__()
        self.final_prompt = final_prompt
        # Need args to create QueryService to access generate_llm_answer_from_prompt
        self.db_path = db_path
        self.emb_model = emb_model
        self.llm_model = llm_model
        self.notes_folder = notes_folder
        self._is_cancelled = False # Add basic cancellation

    def run(self):
        logger.info("LLMGenWorker run method started.") # <-- ADD
        try:
            logger.debug("LLMGenWorker: Creating QueryService...")
            q_service = QueryService(self.db_path, self.emb_model, self.llm_model, self.notes_folder)
            logger.debug("LLMGenWorker: Calling generate_llm_answer_from_prompt...")
            answer = q_service.generate_llm_answer_from_prompt(self.final_prompt)
            logger.info(f"LLMGenWorker: Received answer (len: {len(answer)}): '{answer[:100]}...'") # <-- ADD
            if self._is_cancelled: logger.info("LLMGenWorker: Cancelled after generation."); return # <-- Check cancellation
            self.llm_answer_ready.emit(answer)
        except Exception as e:
            error_msg = f"LLM Generation Error: {e}" # <-- Capture specific error
            logger.error(f"LLMGenWorker: {error_msg}", exc_info=True) # <-- ADD Log
            if not self._is_cancelled:
                self.error_occurred.emit(error_msg) # Emit specific error
            else:
                logger.info("LLMGenWorker: Error occurred but task was cancelled.")
        finally:
            logger.info("LLMGenWorker run method finished.") # <-- ADD
    def stop(self):
        logger.info("LLMGenWorker cancellation requested.")
        self._is_cancelled = True