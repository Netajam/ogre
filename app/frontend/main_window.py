# app/frontend/main_window.py
import sys
import os
from pathlib import Path
from typing import Optional, List

# --- PyQt6 Imports ---
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QObject
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QFileDialog, QLineEdit, QTextEdit, QStatusBar, QMessageBox, QProgressBar,
    QSizePolicy, QApplication
)
from PyQt6.QtGui import QAction, QIcon # Assuming you might add icons later

# --- Application Imports ---
from app import config
from app.core.config_manager import config_manager # Use the singleton instance
# Import the application logger
from app.utils.logging_setup import logger
# Import the real Indexer (used by IndexerWorker)
from app.core.indexing.indexer import Indexer

# Placeholder for Settings Dialog (Phase 2/3)
# from .settings_dialog import SettingsDialog

# --- Indexer Worker Thread ---
class IndexerWorker(QObject):
    """
    Worker object that runs the indexing process in a separate thread.
    Communicates progress and completion via signals.
    """
    # Signals:
    # finished: Emitted when indexing completes or fails. Args: success (bool), message (str)
    # progress: Emitted periodically with status updates. Args: message (str)
    finished = pyqtSignal(bool, str)
    progress = pyqtSignal(str)

    def __init__(self,
                 notes_folder: str,
                 db_path: Path,
                 embedding_model_name: str,
                 allowed_extensions: List[str]): # <-- Added parameter
        super().__init__()
        self.notes_folder = notes_folder
        self.db_path = db_path
        self.embedding_model_name = embedding_model_name
        self.allowed_extensions = allowed_extensions # <-- Store it
        self.indexer: Optional[Indexer] = None
        self._is_running = False

    def run(self):
        """The main method executed when the thread starts."""
        if self._is_running:
            logger.warning("IndexerWorker.run() called while already running.")
            return

        self._is_running = True
        logger.info("IndexerWorker started.")
        try:
            # Create the actual Indexer instance
            self.indexer = Indexer(
                notes_folder=self.notes_folder,
                db_path=self.db_path,
                embedding_model_name=self.embedding_model_name,
                                allowed_extensions=self.allowed_extensions,
                # Connect signals directly to the Indexer's callbacks
                progress_callback=self.progress.emit,
                finished_callback=self.finished.emit
            )
            # Run the main indexing logic
            # The Indexer itself will emit the finished signal via the callback
            self.indexer.run_indexing()

        except Exception as e:
             # This is a fallback catch-all. Errors should ideally be caught
             # and reported gracefully by the Indexer itself.
             error_msg = f"Critical error caught in IndexerWorker run loop: {e}"
             logger.error(error_msg, exc_info=True)
             # Emit finished signal indicating failure if not already emitted by Indexer
             try:
                 self.finished.emit(False, error_msg)
             except Exception as sig_e:
                 logger.error(f"Failed to emit finished signal from worker fallback: {sig_e}")
        finally:
             # Ensure references are cleared and state is reset
             logger.info("IndexerWorker run finished.")
             self.indexer = None # Release indexer instance
             self._is_running = False

    def stop(self):
        """Requests the running Indexer instance to cancel its operation."""
        if self.indexer:
            logger.info("Requesting indexer cancellation via IndexerWorker...")
            self.indexer.cancel() # Call the Indexer's cancel method
        else:
            # This might happen if stop is called after run finished or before it started
            logger.warning("Stop requested for IndexerWorker, but no active Indexer instance found.")


# --- Main Application Window ---
class MainWindow(QMainWindow):
    """Main application window."""

    def __init__(self):
        super().__init__()
        logger.info("Initializing MainWindow...")
        self.setWindowTitle(f"{config.APP_NAME} v{config.APP_VERSION}")
        # Set initial size and position (optional)
        self.setGeometry(200, 200, 800, 600) # x, y, width, height

        # --- Member Variables ---
        self._config_mgr = config_manager # Use the shared config manager instance
        self._notes_folder_path: Optional[str] = self._config_mgr.get_notes_folder()
        self._indexing_thread: Optional[QThread] = None
        self._indexing_worker: Optional[IndexerWorker] = None

        # --- UI Setup ---
        self._setup_ui() # Create widgets and layouts
        self._create_menus() # Create File, Help menus etc.
        self._update_ui_state() # Set initial enabled/disabled states

        logger.info("MainWindow initialization complete.")

    def _setup_ui(self) -> None:
        """Sets up the main UI elements, widgets, and layouts."""
        logger.info("Starting UI setup...") # <-- Log Start

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        logger.info("Central widget set.") # <-- Log After Central Widget

        # Main vertical layout for the central widget
        main_layout = QVBoxLayout() # Create layout first
        central_widget.setLayout(main_layout) # Assign layout to central widget
        logger.info("Main layout created and assigned.") # <-- Log After Main Layout

        # --- Folder Selection Area (Horizontal Layout) ---
        folder_layout = QHBoxLayout()
        self.select_folder_button = QPushButton("Select Notes Folder")
        self.select_folder_button.clicked.connect(self._select_notes_folder)

        self.folder_label = QLabel("No folder selected." if not self._notes_folder_path else f"Notes Folder: {self._notes_folder_path}")
        self.folder_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred) # Allow label to expand horizontally
        self.folder_label.setWordWrap(True) # Wrap text if path is long

        folder_layout.addWidget(self.select_folder_button)
        folder_layout.addWidget(self.folder_label, stretch=1) # Label takes available horizontal space
        main_layout.addLayout(folder_layout) # Add the horizontal layout to the main vertical layout
        logger.info("Folder selection UI added.") # <-- Log After Folder UI

        # --- Indexing Button ---
        self.index_button = QPushButton("Index Notes")
        self.index_button.clicked.connect(self._start_indexing)
        main_layout.addWidget(self.index_button)
        logger.info("Index button added.") # <-- Log After Index Button

        # --- Spacer (Optional) ---
        # main_layout.addSpacerItem(QSpacerItem(20, 10, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding))

        # --- Prompt Area ---
        main_layout.addWidget(QLabel("Enter your query:"))
        self.prompt_input = QLineEdit()
        self.prompt_input.setPlaceholderText("Ask something about your notes...")
        self.prompt_input.returnPressed.connect(self._handle_query) # Trigger query on Enter key
        main_layout.addWidget(self.prompt_input)

        self.ask_button = QPushButton("Ask")
        self.ask_button.clicked.connect(self._handle_query)
        # Align ask button? Example:
        # ask_button_layout = QHBoxLayout()
        # ask_button_layout.addStretch()
        # ask_button_layout.addWidget(self.ask_button)
        # ask_button_layout.addStretch()
        # main_layout.addLayout(ask_button_layout)
        main_layout.addWidget(self.ask_button) # Simpler layout for now
        logger.info("Prompt UI added.") # <-- Log After Prompt UI

        # --- Answer Area ---
        main_layout.addWidget(QLabel("Answer:"))
        self.answer_output = QTextEdit()
        self.answer_output.setReadOnly(True)
        self.answer_output.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding) # Allow text area to grow
        main_layout.addWidget(self.answer_output)
        logger.info("Answer area added.") # <-- Log After Answer Area

        # --- Status Bar ---
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready.") # Initial status message

        # --- Progress Bar (in Status Bar) ---
        self.progress_bar = QProgressBar()
        # Initial setup for progress bar (range/value will be set when shown/updated)
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(False) # Initially hidden
        self.progress_bar.setAlignment(Qt.AlignmentFlag.AlignCenter) # Center percentage text
        self.progress_bar.setMaximumWidth(200) # Optional: Limit width
        # Add progress bar as a permanent widget on the right side of the status bar
        self.status_bar.addPermanentWidget(self.progress_bar)
        logger.info("Status bar and progress bar added.") # <-- Log After Status Bar

        logger.info("Finished UI setup.") # <-- Log End


    def _create_menus(self) -> None:
        """Creates the main menu bar and its actions."""
        logger.debug("Creating menus...")
        menu_bar = self.menuBar()

        # --- File Menu ---
        file_menu = menu_bar.addMenu("&File")

        # Settings Action
        settings_action = QAction("&Settings", self)
        settings_action.setStatusTip("Configure application settings") # Tooltip for status bar
        settings_action.triggered.connect(self._open_settings)
        file_menu.addAction(settings_action)

        file_menu.addSeparator()

        # Exit Action
        exit_action = QAction("&Exit", self)
        exit_action.setStatusTip("Exit the application")
        exit_action.setShortcut("Ctrl+Q") # Example shortcut
        exit_action.triggered.connect(self.close) # Connect to the main window's close method
        file_menu.addAction(exit_action)

        # --- Help Menu (Optional) ---
        # help_menu = menu_bar.addMenu("&Help")
        # about_action = QAction("&About", self)
        # about_action.setStatusTip(f"About {config.APP_NAME}")
        # # about_action.triggered.connect(self._show_about_dialog) # Need to implement _show_about_dialog
        # help_menu.addAction(about_action)

        logger.debug("Menus created.")


    def _update_ui_state(self) -> None:
        """
        Updates the enabled/disabled state and text of UI elements based on
        the application's current state (folder selected, indexing running, etc.).
        """
        logger.debug("Updating UI state...")
        folder_selected = bool(self._notes_folder_path)
        is_indexing = self._indexing_thread is not None and self._indexing_thread.isRunning()

        # Update folder label
        if folder_selected:
            self.folder_label.setText(f"Notes Folder: {self._notes_folder_path}")
        else:
            self.folder_label.setText("No folder selected. Please select your notes folder.")

        # Enable/disable based on folder selection and indexing status
        self.select_folder_button.setEnabled(not is_indexing)
        self.index_button.setEnabled(folder_selected and not is_indexing)

        # Query controls depend on folder selection AND whether indexing is running
        # We might also check if a database exists later for more refinement
        db_path = self._config_mgr.get_db_path()
        db_exists = db_path is not None and db_path.exists() # Basic check
        can_query = folder_selected and not is_indexing # and db_exists # Add db_exists check later if desired

        self.prompt_input.setEnabled(can_query)
        self.ask_button.setEnabled(can_query)

        # Show/hide progress bar
        self.progress_bar.setVisible(is_indexing)
        if not is_indexing:
            self.progress_bar.setValue(0) # Reset progress bar when not running

        # Update status bar message based on state?
        if is_indexing:
            # Progress message is handled by _update_progress signal
            pass
        elif folder_selected:
            self.status_bar.showMessage("Ready.")
        else:
            self.status_bar.showMessage("Please select a notes folder.")

        logger.debug(f"UI state updated: folder_selected={folder_selected}, is_indexing={is_indexing}, can_query={can_query}")


    def _select_notes_folder(self) -> None:
        """Opens a dialog to select the notes folder and updates configuration."""
        logger.info("Select Notes Folder button clicked.")
        current_folder = self._notes_folder_path or str(Path.home()) # Start in current folder or home
        folder = QFileDialog.getExistingDirectory(
            self,
            "Select Notes Folder",
            current_folder,
            # Options can be added, e.g., QFileDialog.Option.ShowDirsOnly
        )
        if folder: # User selected a folder (didn't cancel)
            folder_path = str(Path(folder).resolve()) # Use absolute path
            logger.info(f"Notes folder selected: {folder_path}")
            self._notes_folder_path = folder_path
            self._config_mgr.set_notes_folder(folder_path) # Save to config
            self._update_ui_state() # Update UI based on new folder path
            # No automatic indexing, user must click "Index Notes"
        else:
             logger.info("Folder selection cancelled by user.")


    def _start_indexing(self) -> None:
        """Starts the indexing process in a separate thread."""
        logger.info("Index Notes button clicked.")
        if not self._notes_folder_path:
            logger.warning("Indexing attempt failed: Notes folder not set.")
            QMessageBox.warning(self, "Folder Not Set", "Please select a notes folder first.")
            return
        if self._indexing_thread and self._indexing_thread.isRunning():
            logger.warning("Indexing attempt failed: Indexing already in progress.")
            QMessageBox.information(self, "Indexing Running", "Indexing is already in progress.")
            return

        # Get necessary parameters from config manager
        db_path = self._config_mgr.get_db_path()
        if not db_path:
            # This should not happen if notes_folder_path is set, but check anyway
            logger.error("Indexing failed: Could not determine database path.")
            QMessageBox.critical(self, "Error", "Could not determine database path. Check notes folder configuration.")
            return

        embedding_model_name = self._config_mgr.get_embedding_model_name()
        allowed_extensions = self._config_mgr.get_indexed_extensions()
        if not allowed_extensions:
             logger.error("Indexing failed: No file extensions are configured for indexing. Check settings.")
             QMessageBox.critical(self, "Configuration Error", "No file extensions configured for indexing.\nPlease check the application settings (currently requires manual edit of settings.json).")
             return
        logger.info(f"Starting indexing process for folder: {self._notes_folder_path}")
        logger.info(f"Using database: {db_path}")
        logger.info(f"Using embedding model: {embedding_model_name}")

        # Ensure the directory for the DB exists (DbManager also does this, but good practice)
        try:
            db_path.parent.mkdir(parents=True, exist_ok=True)
        except OSError as e:
             logger.error(f"Could not create directory for database {db_path.parent}: {e}")
             QMessageBox.critical(self, "Error", f"Could not create directory for database: {e}")
             return

        # --- Setup Thread and Worker ---
        self.status_bar.showMessage("Starting indexing...")
        self.progress_bar.setRange(0, 0) # Set to indeterminate initially
        self.progress_bar.setValue(-1) # Style hint for indeterminate in some themes
        self._update_ui_state() # Disable buttons etc.

        self._indexing_thread = QThread(self) # Pass parent for lifetime management?
        self._indexing_worker = IndexerWorker(
            notes_folder=self._notes_folder_path,
            db_path=db_path,
            embedding_model_name=embedding_model_name,
            allowed_extensions=allowed_extensions 

        )
        self._indexing_worker.moveToThread(self._indexing_thread)

        # --- Connect Signals ---
        # Worker -> UI Thread
        self._indexing_worker.finished.connect(self._on_indexing_finished)
        self._indexing_worker.progress.connect(self._update_progress)
        # Thread Control / Cleanup
        self._indexing_thread.started.connect(self._indexing_worker.run)
        self._indexing_worker.finished.connect(self._indexing_thread.quit) # Stop thread event loop when worker done
        # Ensure worker is deleted after thread finishes its event loop
        self._indexing_thread.finished.connect(self._indexing_worker.deleteLater)
        # Ensure thread object itself is deleted after it finishes
        self._indexing_thread.finished.connect(self._clear_indexing_references) # Also clears our references

        # --- Start Thread ---
        self._indexing_thread.start()
        logger.info("Indexing thread started.")


    def _on_indexing_finished(self, success: bool, message: str) -> None:
        """Slot called when the IndexerWorker emits the 'finished' signal."""
        # This slot is executed in the UI thread
        logger.info(f"Indexing finished signal received. Success: {success}, Message: {message}")

        # Update status bar immediately
        self.status_bar.showMessage(message, 10000) # Show message for 10 seconds

        # Reset progress bar (state updated in _clear_indexing_references/ _update_ui_state)
        self.progress_bar.setVisible(False)
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)

        # Show message box to user
        if success:
            QMessageBox.information(self, "Indexing Complete", message)
        else:
            # Check if it was a user cancellation
            if "cancelled" in message.lower():
                 QMessageBox.warning(self, "Indexing Cancelled", message)
            else:
                 # Show critical error for other failures
                 QMessageBox.critical(self, "Indexing Error", message)

        # Note: UI state update happens in _clear_indexing_references after thread truly finishes


    def _clear_indexing_references(self):
        """Slot called when the indexing QThread finishes its execution."""
        # This slot is executed in the UI thread
        logger.debug("Indexing thread finished signal received, clearing references.")
        self._indexing_thread = None
        self._indexing_worker = None
        # Update UI state *after* references are cleared and thread is known to be done
        self._update_ui_state()


    def _update_progress(self, message: str) -> None:
        """Slot called when the IndexerWorker emits the 'progress' signal."""
        # This slot is executed in the UI thread
        logger.debug(f"Indexing progress: {message}")
        self.status_bar.showMessage(message)

        # --- Attempt to parse progress for determinate progress bar ---
        # This parsing is brittle; a better approach is dedicated progress signals (e.g., percent)
        if "Processing file" in message:
             try:
                 parts = message.split(" ")
                 if "/" in parts[2]: # Expected format: "Processing file 5/10: ..."
                     current, total = map(int, parts[2].split('/')[0:2])
                     if total > 0:
                         percent = int((current / total) * 95) # Leave last 5% for embedding/storing phase
                         self.progress_bar.setRange(0, 100)
                         self.progress_bar.setValue(percent)
                     else: # Avoid division by zero if total is 0
                         self.progress_bar.setRange(0, 0) # Indeterminate
                         self.progress_bar.setValue(-1)
                 else: # Unexpected format
                     self.progress_bar.setRange(0, 0) # Indeterminate
                     self.progress_bar.setValue(-1)
             except (IndexError, ValueError):
                  logger.warning(f"Could not parse progress percentage from message: '{message}'")
                  self.progress_bar.setRange(0, 0) # Fallback to indeterminate
                  self.progress_bar.setValue(-1)
        elif "Embedding batch" in message or "Storing chunks" in message:
             # Use indeterminate or show a fixed high percentage during final stages
             self.progress_bar.setRange(0, 100)
             self.progress_bar.setValue(98) # Example: show near completion
        elif "Scanning" in message or "Checking index" in message or "Initializing" in message:
             self.progress_bar.setRange(0, 0) # Indeterminate during setup
             self.progress_bar.setValue(-1)


    def _handle_query(self) -> None:
        """Handles the 'Ask' button click or Enter press in the prompt input."""
        logger.debug("Handle query triggered.")
        query_text = self.prompt_input.text().strip()
        if not query_text:
            logger.debug("Query input is empty, ignoring.")
            return # Ignore empty queries

        # Check prerequisites
        if not self._notes_folder_path:
             logger.warning("Query attempt failed: Notes folder not set.")
             QMessageBox.warning(self, "Folder Not Set", "Please select a notes folder first.")
             return
        if self._indexing_thread and self._indexing_thread.isRunning():
             logger.warning("Query attempt failed: Indexing is currently running.")
             QMessageBox.information(self, "Indexing Running", "Please wait for indexing to finish before asking questions.")
             return

        # --- Placeholder for actual query logic (Phase 3) ---
        # This should run in a separate thread similar to indexing!
        logger.info(f"Received query: '{query_text}'")
        self.answer_output.setText("Thinking...") # Provide immediate feedback
        self.status_bar.showMessage(f"Processing query: {query_text[:50]}...")
        QApplication.processEvents() # Ensure UI updates are shown

        try:
            # --- TODO: Replace with actual query service call in Phase 3 ---
            # Example structure:
            # query_thread = QThread()
            # query_worker = QueryWorker(query_text, self._config_mgr.get_db_path(), self._config_mgr.get_embedding_model_name())
            # query_worker.moveToThread(query_thread)
            # query_worker.result_ready.connect(self._on_query_result)
            # query_worker.error_occurred.connect(self._on_query_error)
            # query_thread.started.connect(query_worker.run)
            # query_worker.finished.connect(query_thread.quit)
            # query_worker.finished.connect(query_worker.deleteLater)
            # query_thread.finished.connect(query_thread.deleteLater)
            # query_thread.start()
            # -----------------------------------------------------------

            # --- Simulate work for now ---
            import time
            time.sleep(1.5)
            # result = query_service.answer_query(query_text) # Future implementation
            result = f"Placeholder answer for query: '{query_text}'.\nActual LLM integration and vector search needed (Phase 3)."
            logger.info("Simulated query processing finished.")
            self.answer_output.setText(result)
            self.status_bar.showMessage("Query processed.", 5000) # Show for 5 seconds
            self.prompt_input.clear() # Clear input after processing

        except Exception as e:
             # Catch errors during the (simulated) query process
             logger.error(f"Error during query processing: {e}", exc_info=True)
             QMessageBox.critical(self, "Query Error", f"An error occurred while processing the query:\n{e}")
             self.answer_output.setText("Error processing query.")
             self.status_bar.showMessage("Query failed.", 5000)


    def _open_settings(self) -> None:
        """Opens the settings dialog (placeholder)."""
        logger.info("Settings action triggered.")
        # In a later phase, this will instantiate and show the SettingsDialog:
        # settings_dialog = SettingsDialog(self) # Pass self as parent
        # settings_dialog.settings_changed.connect(self._on_settings_changed) # Optional signal
        # settings_dialog.exec() # Show as modal dialog
        QMessageBox.information(self, "Settings", "Settings dialog not implemented yet (Phase 3).")


    # def _on_settings_changed(self):
    #     """ Slot called if settings dialog emits a signal indicating changes. """
    #     logger.info("Settings potentially changed, updating UI state.")
    #     # Re-read relevant settings if necessary
    #     self._notes_folder_path = self._config_mgr.get_notes_folder()
    #     # Update UI elements that depend on settings
    #     self._update_ui_state()


    def closeEvent(self, event) -> None:
        """Handles the window closing event (e.g., clicking the 'X' button)."""
        logger.debug("Close event triggered.")
        # Check if indexing is running and ask for confirmation
        if self._indexing_thread and self._indexing_thread.isRunning():
             logger.warning("Close requested while indexing is in progress.")
             reply = QMessageBox.question(self, 'Confirm Exit',
                                        "Indexing is in progress. Exiting now might leave the index incomplete.\nAre you sure you want to exit?",
                                        QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                                        QMessageBox.StandardButton.No) # Default to No

             if reply == QMessageBox.StandardButton.Yes:
                 logger.info("User confirmed exit during indexing. Attempting cancellation...")
                 if self._indexing_worker:
                     # Politely ask the worker (and its Indexer) to stop
                     self._indexing_worker.stop()
                     # Give it a brief moment to potentially react? Not strictly necessary
                     # QApplication.processEvents()
                     # time.sleep(0.1)

                 # Note: We don't forcefully terminate the thread here as it can
                 # cause data corruption. We allow the app to close, and the thread
                 # will exit when the process terminates or when it finishes/cancels.
                 logger.info("Proceeding with exit.")
                 event.accept() # Allow the window to close
             else:
                 logger.info("User cancelled exit during indexing.")
                 event.ignore() # Prevent the window from closing
        else:
             logger.info("Closing application.")
             event.accept() # Allow closing if indexing is not running