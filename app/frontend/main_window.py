# app/frontend/main_window.py
import sys
import os
from pathlib import Path
from typing import Optional, List, Dict, Type, Tuple, Callable, Any

# --- PyQt6 Imports ---
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QObject, QTimer # Added QTimer
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QSplitter, QStatusBar,
    QMessageBox, QProgressBar, QApplication, QFileDialog, QListWidgetItem, QDialog # Added QDialog
)
from PyQt6.QtGui import QAction, QIcon, QTextCursor

# --- Application Imports ---
from app import config
from app.core.config_manager import config_manager
from app.utils.logging_setup import logger
# Core components needed for type hints or args
from app.core.indexing.indexer import Indexer
from app.core.services.query_service import QueryService
from app.core.services.result_item import ResultItem
from app.core.indexing.file_handler import FileHandler
# --- Import Widgets ---
from .widgets.top_panel import TopPanelWidget
from .widgets.query_panel import QueryPanelWidget
from .widgets.results_panel import ResultsPanelWidget
from .widgets.display_panel import DisplayPanelWidget
# --- Import Dialogs ---
from .prompt_dialog import PromptEditDialog
# --- Import Workers ---
from .workers import IndexerWorker, QueryWorker, LLMGenWorker


# --- Main Application Window (Refactored) ---
class MainWindow(QMainWindow):
    """
    Main application window orchestrating UI panels and background tasks.
    Connects signals from panels to appropriate application logic slots.
    Manages application state related to background tasks and results.
    """

    def __init__(self):
        super().__init__()
        logger.info("Initializing MainWindow...")
        self.setWindowTitle(f"{config.APP_NAME} v{config.APP_VERSION}")
        self.setGeometry(200, 200, 950, 750) # Adjusted size

        # Core components & State
        self._config_mgr = config_manager
        self._notes_folder_path: Optional[str] = self._config_mgr.get_notes_folder()
        self._indexing_thread: Optional[QThread] = None
        self._indexing_worker: Optional[IndexerWorker] = None
        self._query_thread: Optional[QThread] = None
        self._query_worker: Optional[Any] = None # Can be QueryWorker or LLMGenWorker
        self._current_query_mode: str = "auto_answer"
        self._current_results: Optional[List[ResultItem]] = None
        self._current_llm_answer: Optional[str] = None
        self._current_note_display_path: Optional[Path] = None
        # --- Add Explicit Busy Flag ---
        self._is_processing_query: bool = False # Flag for query workflow

        # UI Panels - Instantiated and laid out
        self._setup_ui_panels()
        self._create_menus()
        self._connect_panel_signals()
        self._update_ui_state()

        logger.info("MainWindow initialization complete.")

    # --- UI Setup ---

# Inside class MainWindow:
    def _setup_ui_panels(self) -> None:
        """Creates and arranges the main UI panel widgets."""
        logger.info("Setting up UI panels...")
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # Instantiate Panels (UI setup happens within their constructors)
        self.top_panel = TopPanelWidget(parent=self)
        self.query_panel = QueryPanelWidget(parent=self) # Instantiates QueryPanelWidget
        self.results_panel = ResultsPanelWidget(parent=self)
        self.display_panel = DisplayPanelWidget(parent=self)

        # Arrange Panels using Splitter
        self.bottom_splitter = QSplitter(Qt.Orientation.Horizontal, parent=self)
        self.bottom_splitter.addWidget(self.results_panel)
        self.bottom_splitter.addWidget(self.display_panel)
        self.bottom_splitter.setStretchFactor(0, 1)
        self.bottom_splitter.setStretchFactor(1, 2)

        # Add panels to main layout
        main_layout.addWidget(self.top_panel)
        main_layout.addWidget(self.query_panel) # Adds the QueryPanelWidget instance
        main_layout.addWidget(self.bottom_splitter, stretch=1)

        initial_label_text = f"Notes: {self._notes_folder_path}" if self._notes_folder_path else "No folder selected."
        self.top_panel.set_folder_label(initial_label_text)   
        self._setup_status_bar()
        logger.info("UI panels setup finished.")
    def _setup_status_bar(self) -> None:
        """Sets up the status bar."""
        self.status_bar = QStatusBar(self)
        self.setStatusBar(self.status_bar)
        self.progress_bar = QProgressBar(self)
        self.progress_bar.setRange(0, 100); self.progress_bar.setValue(0); self.progress_bar.setVisible(False)
        self.progress_bar.setAlignment(Qt.AlignmentFlag.AlignCenter); self.progress_bar.setMaximumWidth(200)
        self.status_bar.addPermanentWidget(self.progress_bar)
        self.status_bar.showMessage("Ready.")

    def _create_menus(self) -> None:
        """Creates the main menu bar."""
        logger.debug("Creating menus...")
        menu_bar = self.menuBar()
        file_menu = menu_bar.addMenu("&File")
        settings_action = QAction("&Settings", self); settings_action.setStatusTip("Configure application settings"); settings_action.triggered.connect(self._open_settings)
        exit_action = QAction("&Exit", self); exit_action.setShortcut("Ctrl+Q"); exit_action.setStatusTip("Exit the application"); exit_action.triggered.connect(self.close)
        file_menu.addAction(settings_action); file_menu.addSeparator(); file_menu.addAction(exit_action)
        logger.debug("Menus created.")

    def _connect_panel_signals(self) -> None:
        """Connects signals from custom panel widgets to MainWindow slots."""
        logger.debug("Connecting panel signals...")
        self.top_panel.selectFolderClicked.connect(self._select_notes_folder)
        self.top_panel.indexClicked.connect(self._start_indexing)
        self.query_panel.querySubmitted.connect(self._handle_query)
        self.query_panel.stopQueryClicked.connect(self._stop_query)
        self.results_panel.results_list.itemClicked.connect(self._on_result_item_clicked)
        logger.debug("Panel signals connected.")

    # --- State Management ---

    def _update_ui_state(self) -> None:
        """Updates the enabled/disabled state of UI panels based on current state."""
        logger.debug("Updating UI state...")
        folder_selected = bool(self._notes_folder_path)
        is_indexing = self._indexing_thread is not None and self._indexing_thread.isRunning()
        # --- Use custom flag for querying state ---
        is_querying = self._is_processing_query
        # ----------------------------------------
        is_busy = is_indexing or is_querying

        self.top_panel.set_folder_label(f"Notes: {self._notes_folder_path}" if folder_selected else "No folder selected.")
        self.top_panel.set_busy(is_busy)

        db_path = self._config_mgr.get_db_path(); db_exists = db_path is not None and db_path.exists()
        can_query = folder_selected and db_exists and not is_busy
        self.query_panel.set_query_enabled(can_query)
        self.query_panel.set_stop_enabled(is_querying) # Enable stop based on flag

        self.display_panel.set_busy(is_busy)

        self.progress_bar.setVisible(is_busy)
        if not is_busy: self.progress_bar.setValue(0)

        if not is_busy:
            status = "Ready." if folder_selected else "Please select a notes folder."
            # Preserve potentially important messages briefly after task ends
            current_message = self.status_bar.currentMessage()
            if "complete" not in current_message.lower() and "fail" not in current_message.lower() and "cancel" not in current_message.lower():
                self.status_bar.showMessage(status)
        # Else: Message handled by task signals/start

        logger.debug(f"UI state updated: folder={folder_selected}, busy={is_busy}, can_query={can_query}")

    # --- Background Worker Management (Helper) ---

    def _setup_and_start_worker(self, worker_class: Type[QObject], thread_attr: str, worker_attr: str, on_finish_slot: Callable, **worker_args) -> Tuple[Optional[QThread], Optional[QObject]]:
        """Helper to create, configure, and start a background worker thread."""
        # Check using thread isRunning() as it's reliable for thread lifetime
        existing_thread: Optional[QThread] = getattr(self, thread_attr, None)
        if existing_thread and existing_thread.isRunning():
            object_name = worker_class.__name__.replace("Worker", "")
            logger.warning(f"{object_name} worker already running (thread check).")
            QMessageBox.information(self, "Busy", f"{object_name} process is already running.")
            return None, None

        setattr(self, thread_attr, None); setattr(self, worker_attr, None)
        thread = QThread(self); worker = worker_class(**worker_args); worker.moveToThread(thread)
        thread.started.connect(worker.run); thread.finished.connect(worker.deleteLater); thread.finished.connect(on_finish_slot)
        setattr(self, thread_attr, thread); setattr(self, worker_attr, worker)
        thread.start(); logger.info(f"{worker_class.__name__} thread started.")
        return thread, worker


    # --- Action Slots (Triggered by Panel Signals) ---

    def _select_notes_folder(self) -> None:
        """Handles signal from TopPanelWidget's selectFolderClicked."""
        logger.info("Select Notes Folder triggered.")
        if self._is_processing_query or (self._indexing_thread and self._indexing_thread.isRunning()):
            QMessageBox.warning(self, "Busy", "Cannot change folder while processing.")
            return
        current_folder = self._notes_folder_path or str(Path.home())
        folder = QFileDialog.getExistingDirectory(self, "Select Notes Folder", current_folder)
        if folder:
            folder_path = str(Path(folder).resolve())
            logger.info(f"Notes folder selected: {folder_path}")
            self._notes_folder_path = folder_path
            self._config_mgr.set_notes_folder(folder_path)
            self._clear_previous_query_state()
            self._update_ui_state()
        else: logger.info("Folder selection cancelled.")

    def _start_indexing(self) -> None:
        """Handles signal from TopPanelWidget's indexClicked."""
        logger.info("Start Indexing triggered.")
        if not self._can_start_task(): return

        db_path = self._config_mgr.get_db_path(); embedding_model_name = self._config_mgr.get_embedding_model_name()
        allowed_extensions = self._config_mgr.get_indexed_extensions()
        if not db_path: QMessageBox.critical(self, "Error", "No DB path."); return
        if not allowed_extensions: QMessageBox.critical(self, "Config Error", "No indexable extensions."); return
        try: db_path.parent.mkdir(parents=True, exist_ok=True)
        except OSError as e: QMessageBox.critical(self, "Error", f"Could not create DB dir: {e}"); return

        logger.info("Attempting to start indexing...")
        self.status_bar.showMessage("Starting indexing..."); self.progress_bar.setRange(0, 0); self.progress_bar.setValue(-1)
        self._update_ui_state() # Update state before starting

        thread, worker = self._setup_and_start_worker(
            worker_class=IndexerWorker, thread_attr="_indexing_thread", worker_attr="_indexing_worker",
            on_finish_slot=self._clear_indexing_references,
            notes_folder=self._notes_folder_path, db_path=db_path,
            embedding_model_name=embedding_model_name, allowed_extensions=allowed_extensions
        )
        if worker:
            worker.finished.connect(self._on_indexing_finished)
            worker.progress.connect(self._update_progress)
        else:
            # If worker start failed, reset UI state
            self._update_ui_state()

    def _handle_query(self, query_text: str, mode: str):
        """Handles querySubmitted signal from QueryPanelWidget."""
        logger.debug(f"Handle query signal received. Mode: {mode}, Query: '{query_text[:50]}...'")
        if not self._can_start_task(): return # Checks internal flags now

        self._current_query_mode = mode
        db_path = self._config_mgr.get_db_path(); embedding_model_name = self._config_mgr.get_embedding_model_name()
        llm_model_name = self._config_mgr.get_llm_model_name()
        if not db_path or not db_path.exists(): QMessageBox.warning(self, "DB Not Found", "Index notes first."); return

        logger.info(f"Starting query task, mode: {mode}")
        self._clear_previous_query_state()
        # --- Set Busy Flag ---
        self._is_processing_query = True
        # --------------------
        self.status_bar.showMessage("Processing query..."); self.progress_bar.setRange(0, 0); self.progress_bar.setValue(-1)
        self._update_ui_state() # Update UI to reflect busy state

        thread, worker = self._setup_and_start_worker(
            worker_class=QueryWorker, thread_attr="_query_thread", worker_attr="_query_worker",
            on_finish_slot=self._clear_query_references, # Cleanup slot when thread *truly* finishes
            # Pass args needed by QueryWorker.__init__
            query_text=query_text, db_path=db_path,
            embedding_model_name=embedding_model_name, llm_model_name=llm_model_name,
            notes_folder=self._notes_folder_path, mode=mode
        )
        if worker:
            # Connect QueryWorker specific signals
            worker.results_ready.connect(self._on_search_results)
            worker.llm_answer_ready.connect(self._on_llm_answer_ready)
            worker.error_occurred.connect(self._on_query_error)
            # Remove direct thread quit connections here - let worker finish fully
            # worker.llm_answer_ready.connect(thread.quit) # REMOVED
            # worker.error_occurred.connect(thread.quit) # REMOVED
        else:
            logger.warning("Failed to start QueryWorker.")
            # Reset busy state if worker didn't start
            self._is_processing_query = False
            self._update_ui_state()

    def _stop_query(self):
        """Handles signal from QueryPanelWidget's stopQueryClicked."""
        logger.info("Stop query triggered.")
        # Check our flag first, then worker existence
        if self._is_processing_query and self._query_worker:
            if hasattr(self._query_worker, 'stop'):
                logger.info("Calling stop() on query worker.")
                self._query_worker.stop()
                self.status_bar.showMessage("Attempting to stop query...", 3000)
                # Don't force quit thread, let worker finish its current step & check flag
            else:
                 logger.warning(f"Worker {type(self._query_worker).__name__} has no stop() method.")
        else: logger.warning("Stop query requested, but no query process active.")

    # --- Helper Slots ---
    def _can_start_task(self) -> bool:
        """Checks if a background task can be started."""
        if not self._notes_folder_path: QMessageBox.warning(self, "Folder Not Set", "Select notes folder."); return False
        # Check indexing thread directly
        if self._indexing_thread and self._indexing_thread.isRunning(): QMessageBox.information(self, "Busy", "Wait for indexing."); return False
        # --- Check our explicit query flag ---
        if self._is_processing_query:
            # Optionally double-check thread state for robustness
            # if self._query_thread and self._query_thread.isRunning():
            #    logger.warning("_can_start_task: Flag is True and thread is running.")
            # else:
            #    logger.warning("_can_start_task: Flag is True, but thread is NOT running (state inconsistency?).")
            QMessageBox.information(self, "Busy", "Wait for the current query operation to complete.")
            return False
        # -----------------------------------
        return True

    def _clear_previous_query_state(self):
        """Clears UI and state related to the previous query."""
        self.results_panel.clear_results()
        self.display_panel.clear_displays()
        self._current_results = None; self._current_llm_answer = None; self._current_note_display_path = None
        self.display_panel.show_view(0) # Default to note view

    # --- Worker Signal Slots ---

    def _on_indexing_finished(self, success: bool, message: str) -> None:
        """Handles IndexerWorker's finished signal."""
        logger.info(f"Indexing finished signal. Success: {success}, Msg: {message}")
        self.status_bar.showMessage(message, 10000)
        if success: QMessageBox.information(self, "Indexing Complete", message)
        elif "cancelled" in message.lower(): QMessageBox.warning(self, "Indexing Cancelled", message)
        else: QMessageBox.critical(self, "Indexing Error", message)
        # Cleanup happens via _clear_indexing_references connected to thread finished

    def _on_search_results(self, results: List[ResultItem]):
        """Handles QueryWorker's results_ready signal."""
        logger.info(f"Received {len(results)} search results for mode '{self._current_query_mode}'.")
        self._current_results = results
        self.results_panel.update_results(results)

        # Update status based on next step
        if self._current_query_mode == "auto_answer":
            if results: self.status_bar.showMessage(f"Found {len(results)} chunks. Generating answer...", 20000)
            else: self.status_bar.showMessage("No chunks found. Cannot generate answer.", 5000)
        elif self._current_query_mode == "edit_prompt":
            if results: self.status_bar.showMessage(f"Found {len(results)} chunks. Ready to edit prompt.", 10000)
            else: self.status_bar.showMessage("No chunks found to generate prompt.", 5000)
            # Trigger prompt editor display *after* this slot finishes
            if results: QTimer.singleShot(0, self._show_prompt_editor)
            else: self.query_panel.clear_prompt() # Clear if no results for edit mode
        elif self._current_query_mode == "chunks_only":
            self.status_bar.showMessage(f"Displayed {len(results)} relevant chunks.", 5000)
            self.query_panel.clear_prompt()
        # Thread cleanup / UI update happens later

    def _on_llm_answer_ready(self, answer: str):
        """Handles QueryWorker's/LLMGenWorker's llm_answer_ready signal."""
        # This signal now also indicates the end of the worker's task in non-auto modes
        is_dummy_signal = "[Processing stopped after chunk retrieval" in answer

        if not is_dummy_signal:
            logger.info("LLM answer received.")
            self._current_llm_answer = answer
            self.display_panel.set_llm_answer(answer)
            self.display_panel.show_view(1) # Switch to LLM view
            self.query_panel.clear_prompt()
            self.status_bar.showMessage("Query complete.", 5000)
        else:
            logger.debug("Dummy llm_answer_ready signal received, query workflow step complete.")
            # For chunks_only/edit_prompt mode where user cancelled dialog,
            # this signal marks the end of the worker thread's lifecycle.
            # No UI action needed here, cleanup will happen via _clear_query_references

        # --- Trigger thread quit here, after processing the final signal ---
        if self._query_thread and not self._query_thread.isFinished():
            logger.debug("Requesting query thread quit after final signal processing.")
            self._query_thread.quit()
        # --------------------------------------------------------------------


    def _on_query_error(self, error_message: str):
        """Handles QueryWorker's/LLMGenWorker's error_occurred signal."""
        logger.error(f"Query error signal received: {error_message}")
        self.status_bar.showMessage("Query failed.", 5000)
        QMessageBox.critical(self, "Query Error", f"An error occurred:\n{error_message}")
        self._clear_previous_query_state()
        self.query_panel.clear_prompt()
        # --- Clear busy flag on error ---
        self._is_processing_query = False
        # --- Trigger thread quit here too ---
        if self._query_thread and not self._query_thread.isFinished():
            logger.debug("Requesting query thread quit after error signal processing.")
            self._query_thread.quit()
        # ----------------------------------
        self._update_ui_state() # Update UI immediately on error


    def _clear_indexing_references(self):
        """Slot to clean up after indexing thread finishes."""
        logger.debug("Clearing indexing thread references.")
        self._indexing_thread = None; self._indexing_worker = None
        self._update_ui_state() # Update UI now

    def _clear_query_references(self):
        """Slot to clean up after query thread finishes."""
        logger.debug("Clearing query thread references.")
        # --- Clear busy flag when thread is confirmed finished ---
        self._is_processing_query = False
        # ------------------------------------------------------
        self._query_thread = None; self._query_worker = None
        self._update_ui_state() # Update UI now

    def _update_progress(self, message: str) -> None:
        """Handles IndexerWorker's progress signal."""
        logger.debug(f"Indexing progress: {message}")
        self.status_bar.showMessage(message)
        # ... (Progress bar update logic) ...

    # --- UI Interaction Slots ---

    def _on_result_item_clicked(self, item: QListWidgetItem):
        """Handles clicking on an item in the results list panel.
           Loads the full note content into the Note View and highlights the selected chunk.
           (Based on previous implementation logic)
        """
        result_data: Optional[ResultItem] = item.data(Qt.ItemDataRole.UserRole)

        # Use isinstance for a more robust check
        if not isinstance(result_data, ResultItem):
            logger.warning("Clicked list item does not contain valid ResultItem data.")
            # Show error in status bar, maybe clear display or show error there too
            self.status_bar.showMessage("Error: Could not load data for this item.", 4000)
            # Optionally clear the note view or display an error message within it
            # self.display_panel.note_output_view.setText("Error: Could not load data for this item.")
            # self.display_panel.show_view(0)
            return

        # Get data from ResultItem (adjust attribute names if needed)
        file_path: Path = result_data.file_path
        # *** IMPORTANT: Verify this attribute name in your ResultItem class ***
        chunk_text: str = result_data.content # Or result_data.chunk_text ?

        # Log using info level, include chunk info if available (like index or start)
        # log_chunk_info = f" - Chunk starting: '{chunk_text[:30]}...'" if chunk_text else "" # Example
        # logger.info(f"Result item clicked: {file_path.name}{log_chunk_info}")
        logger.info(f"Result item clicked: Preparing to display {file_path.name}") # Simpler log

        self.status_bar.showMessage(f"Loading content for {file_path.name}...")
        # QApplication.processEvents() # Avoid unless necessary, can cause issues

        try:
            # Read the full content using FileHandler
            full_content = FileHandler.read_file_content(file_path)

            if full_content is None:
                logger.error(f"FileHandler failed to read file: {file_path}")
                # Show a more user-friendly message box
                QMessageBox.warning(self, "Error Reading File",
                                    f"Could not read the content of the file:\n{file_path}")
                self.status_bar.showMessage(f"Error reading file: {file_path.name}", 5000)
                # Clear the display panel's note view
                self.display_panel.note_output_view.clear()
                self.display_panel.show_view(0) # Ensure note view is active
                return

            # --- Update the correct display panel widget ---
            target_view = self.display_panel.note_output_view
            target_view.setPlainText(full_content)
            logger.debug(f"Set full content ({len(full_content)} chars) for {file_path.name} in note_output_view.")

            # --- Highlight the chunk ---
            cursor = target_view.textCursor()
            start_pos = full_content.find(chunk_text)

            if start_pos != -1:
                cursor.setPosition(start_pos)
                # Use len(chunk_text) for highlighting length
                cursor.movePosition(QTextCursor.MoveOperation.Right, QTextCursor.MoveMode.KeepAnchor, len(chunk_text))
                target_view.setTextCursor(cursor)
                # Use QTimer.singleShot for reliable scrolling after potential layout updates
                QTimer.singleShot(0, target_view.ensureCursorVisible)
                # target_view.ensureCursorVisible() # Immediate call might sometimes not work as expected
                logger.debug(f"Highlighted chunk and ensured visibility in {file_path.name}.")
            else:
                # Log if the exact chunk text wasn't found (might happen with slight variations)
                logger.warning(f"Could not find exact chunk text within {file_path.name}. Scrolling to top.")
                cursor.movePosition(QTextCursor.MoveOperation.Start)
                target_view.setTextCursor(cursor)
                target_view.ensureCursorVisible()

            # --- Ensure the Note View is visible ---
            self.display_panel.show_view(0) # Switch to Note view (index 0)

            # --- Update status bar ---
            self.status_bar.showMessage(f"Displayed content for {file_path.name}", 5000)

            # --- Store the path (optional) ---
            self._current_note_display_path = file_path

        except FileNotFoundError: # More specific error handling
             logger.error(f"File not found: {file_path}")
             QMessageBox.warning(self, "File Not Found", f"The note file could not be found:\n{file_path}")
             self.status_bar.showMessage(f"Error: File not found - {file_path.name}", 5000)
             self.display_panel.note_output_view.clear()
             self.display_panel.show_view(0)
        except PermissionError:
              logger.error(f"Permission denied reading: {file_path}")
              QMessageBox.warning(self, "Permission Error", f"Could not read the note file due to permissions:\n{file_path}")
              self.status_bar.showMessage(f"Error: Permission denied - {file_path.name}", 5000)
              self.display_panel.note_output_view.clear()
              self.display_panel.show_view(0)
        except Exception as e:
            # General error handling
            logger.error(f"Error displaying file content for {file_path}: {e}", exc_info=True)
            QMessageBox.critical(self, "Error", f"An unexpected error occurred while displaying the note:\n{file_path}\n\n{e}")
            self.display_panel.note_output_view.setText(f"Error loading content:\n{e}") # Show error in view
            self.display_panel.show_view(0) # Ensure note view is active
            self.status_bar.showMessage("Error displaying content.", 5000)

    def _open_settings(self) -> None:
        """Opens the settings dialog (placeholder)."""
        logger.info("Settings action triggered.")
        QMessageBox.information(self, "Settings", "Settings dialog not implemented yet.")

# Inside class MainWindow:

    def _show_prompt_editor(self):
        """Creates the initial prompt and shows the editing dialog."""
        if not self._current_results:
            QMessageBox.warning(self, "No Context", "No results to generate prompt."); return

        # --- Get the original query text from the UI panel ---
        query_text = self.query_panel.get_query_text() # Assumes QueryPanelWidget has this method
        if not query_text:
            # This case might happen if the user cleared the input after submitting
            # We might need to store the last submitted query text in MainWindow state
            # For now, show an error or maybe use a placeholder
            QMessageBox.warning(self, "No Query", "Could not retrieve original query text."); return
        # ------------------------------------------------------

        logger.debug("Showing prompt editor dialog.")
        try:
            db_path = self._config_mgr.get_db_path()
            llm_model_name = self._config_mgr.get_llm_model_name()
            emb_model_name = self._config_mgr.get_embedding_model_name()
            notes_folder = self._notes_folder_path
            if not all([db_path, llm_model_name, emb_model_name, notes_folder]):
                 logger.error("Missing config for prompt editor service.")
                 QMessageBox.critical(self, "Config Error", "Missing configuration for prompt generation.")
                 return

            temp_q_service = QueryService(db_path, emb_model_name, llm_model_name, notes_folder)
            initial_prompt = temp_q_service.create_llm_prompt(query_text, self._current_results)
            del temp_q_service
        except Exception as e:
            logger.error(f"Failed to create initial prompt: {e}", exc_info=True)
            QMessageBox.critical(self, "Error", f"Failed to generate prompt for editing:\n{e}")
            self._clear_query_references() # Clean up state if prompt gen failed
            return

        if not initial_prompt:
            QMessageBox.information(self, "Empty Prompt", "Could not generate prompt (context empty/invalid).")
            self._clear_query_references() # Clean up state
            return

        # --- Pass the retrieved query_text to the dialog ---
        dialog = PromptEditDialog(initial_prompt, query_text, self)
        # ---------------------------------------------------
        dialog_result = dialog.exec()

        if dialog_result == QDialog.DialogCode.Accepted:
            final_prompt = dialog.get_edited_prompt()
            logger.info("User accepted edited prompt. Initiating LLM task...")
            self._initiate_llm_task_from_prompt(final_prompt)
        else:
            logger.info("User cancelled prompt editing.")
            self.status_bar.showMessage("Prompt editing cancelled.", 3000)
            # Ensure cleanup runs if the original query thread might still be finishing
            self._clear_query_references() # Explicitly call cleanup
    def _initiate_llm_task_from_prompt(self, prompt: str):
        """Initiates the LLM generation task directly."""
        logger.debug("Initiating LLM generation task...")

        # --- Use custom flag check ---
        if not self._can_start_task(): # Checks _is_processing_query now
            logger.error("Cannot start LLM generation task - App state is busy.")
            # Show message box as the button click seemed to succeed initially
            QMessageBox.warning(self, "Busy", "Could not start LLM generation, another process is still active.")
            self._update_ui_state()
            return
        # ---------------------------

        logger.info("Starting LLM generation task...")
        # --- Set Busy Flag ---
        self._is_processing_query = True
        # --------------------
        self.display_panel.set_llm_answer("Generating answer from edited prompt...")
        self._current_llm_answer = None; self.display_panel.show_view(1)
        self.status_bar.showMessage("Generating LLM answer..."); self.progress_bar.setRange(0, 0); self.progress_bar.setValue(-1)
        self._update_ui_state()

        thread, worker = self._setup_and_start_worker(
            worker_class=LLMGenWorker, thread_attr="_query_thread", worker_attr="_query_worker",
            on_finish_slot=self._clear_query_references,
            final_prompt=prompt, db_path=self._config_mgr.get_db_path(),
            emb_model=self._config_mgr.get_embedding_model_name(), llm_model=self._config_mgr.get_llm_model_name(),
            notes_folder=self._notes_folder_path
        )
        if worker:
            worker.llm_answer_ready.connect(self._on_llm_answer_ready)
            worker.error_occurred.connect(self._on_query_error)
            # --- Connect signals to quit thread ---
            worker.llm_answer_ready.connect(self._query_thread.quit)
            worker.error_occurred.connect(self._query_thread.quit)
            # --------------------------------------
        else:
            logger.warning("Failed to start LLMGenWorker.")
            # Reset busy state if worker didn't start
            self._is_processing_query = False
            self._update_ui_state()


    # --- Window Closing ---
    def closeEvent(self, event) -> None:
        # ... (Implementation remains the same, checks _is_processing_query via _can_start_task indirectly) ...
        logger.debug("Close event triggered.");
        is_indexing = self._indexing_thread and self._indexing_thread.isRunning();
        # Check our flag for query state
        is_querying = self._is_processing_query
        if is_indexing or is_querying:
            op = "Indexing" if is_indexing else "Querying"; logger.warning(f"Close requested during {op}.")
            reply = QMessageBox.question(self, 'Confirm Exit', f"{op} is in progress. Exit anyway?", QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No, QMessageBox.StandardButton.No)
            if reply == QMessageBox.StandardButton.Yes:
                 logger.info(f"User confirmed exit during {op}.")
                 if is_indexing and self._indexing_worker: self._indexing_worker.stop()
                 if is_querying and self._query_worker and hasattr(self._query_worker, 'stop'): self._query_worker.stop()
                 event.accept()
            else: logger.info(f"User cancelled exit during {op}."); event.ignore()
        else: logger.info("Closing application."); event.accept()