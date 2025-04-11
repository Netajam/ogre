# app/frontend/main_window.py
import sys
import os
from pathlib import Path
from typing import Optional, List

# --- PyQt6 Imports ---
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QObject, QUrl
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QFileDialog, QLineEdit, QTextEdit, QStatusBar, QMessageBox, QProgressBar,
    QSizePolicy, QApplication, QListWidget, QListWidgetItem, QSplitter
)
from PyQt6.QtGui import QAction, QIcon, QTextCursor, QDesktopServices

# --- Application Imports ---
from app import config
from app.core.config_manager import config_manager
from app.utils.logging_setup import logger
from app.core.indexing.indexer import Indexer
from app.core.services.query_service import QueryService
from app.core.services.result_item import ResultItem
from app.core.indexing.file_handler import FileHandler


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
        if self._is_running:
            logger.warning("IndexerWorker.run() called while already running.")
            return
        self._is_running = True
        logger.info("IndexerWorker started.")
        try:
            self.indexer = Indexer(
                notes_folder=self.notes_folder,
                db_path=self.db_path,
                embedding_model_name=self.embedding_model_name,
                allowed_extensions=self.allowed_extensions,
                progress_callback=self.progress.emit,
                finished_callback=self.finished.emit
            )
            self.indexer.run_indexing()
        except Exception as e:
             error_msg = f"Critical error caught in IndexerWorker run loop: {e}"
             logger.error(error_msg, exc_info=True)
             try:
                 self.finished.emit(False, error_msg)
             except Exception as sig_e:
                 logger.error(f"Failed to emit finished signal from worker fallback: {sig_e}")
        finally:
             logger.info("IndexerWorker run finished.")
             self.indexer = None
             self._is_running = False

    def stop(self):
        if self.indexer:
            logger.info("Requesting indexer cancellation via IndexerWorker...")
            self.indexer.cancel()
        else:
            logger.warning("Stop requested for IndexerWorker, but no active Indexer instance found.")


# --- Background Worker for Querying ---
class QueryWorker(QObject):
    results_ready = pyqtSignal(list) # list will contain ResultItem objects
    error_occurred = pyqtSignal(str)

    def __init__(self, query_text: str, db_path: Path, embedding_model_name: str, notes_folder: str):
        super().__init__()
        self.query_text = query_text
        self.db_path = db_path
        self.embedding_model_name = embedding_model_name
        self.notes_folder = notes_folder

    def run(self):
        logger.info(f"QueryWorker started for query: '{self.query_text[:50]}...'")
        try:
            query_service = QueryService(
                db_path=self.db_path,
                embedding_model_name=self.embedding_model_name,
                notes_folder=self.notes_folder
            )
            results: List[ResultItem] = query_service.search_chunks(self.query_text)
            self.results_ready.emit(results)
            logger.info(f"QueryWorker finished successfully, found {len(results)} results.")
        except Exception as e:
            error_msg = f"Error during query execution: {e}"
            logger.error(error_msg, exc_info=True)
            self.error_occurred.emit(error_msg)


# --- Main Application Window ---
class MainWindow(QMainWindow):
    """Main application window."""

    def __init__(self):
        super().__init__()
        logger.info("Initializing MainWindow...")
        self.setWindowTitle(f"{config.APP_NAME} v{config.APP_VERSION}")
        self.setGeometry(200, 200, 900, 700)

        self._config_mgr = config_manager
        self._notes_folder_path: Optional[str] = self._config_mgr.get_notes_folder()
        self._indexing_thread: Optional[QThread] = None
        self._indexing_worker: Optional[IndexerWorker] = None
        self._query_thread: Optional[QThread] = None
        self._query_worker: Optional[QueryWorker] = None

        self._setup_ui()
        self._create_menus()
        self._update_ui_state()

        logger.info("MainWindow initialization complete.")

    def _setup_ui(self) -> None:
        """Sets up the main UI elements, widgets, and layouts."""
        logger.info("Starting UI setup...")
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # Top Area: Folder Selection and Indexing
        top_layout = QHBoxLayout()
        self.select_folder_button = QPushButton("Select Notes Folder")
        self.select_folder_button.clicked.connect(self._select_notes_folder)
        self.folder_label = QLabel("No folder selected.")
        self.folder_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        self.folder_label.setWordWrap(True)
        self.index_button = QPushButton("Index Notes")
        self.index_button.clicked.connect(self._start_indexing)
        top_layout.addWidget(self.select_folder_button)
        top_layout.addWidget(self.folder_label, 1)
        top_layout.addWidget(self.index_button)
        main_layout.addLayout(top_layout)

        # Middle Area: Query Input
        query_layout = QHBoxLayout()
        query_layout.addWidget(QLabel("Query:"))
        self.prompt_input = QLineEdit()
        self.prompt_input.setPlaceholderText("Ask something about your notes...")
        self.prompt_input.returnPressed.connect(self._handle_query)
        query_layout.addWidget(self.prompt_input, 1)
        self.ask_button = QPushButton("Ask")
        self.ask_button.clicked.connect(self._handle_query)
        query_layout.addWidget(self.ask_button)
        main_layout.addLayout(query_layout)

        # Bottom Area: Results List and Content View (using QSplitter)
        bottom_splitter = QSplitter(Qt.Orientation.Horizontal)

        # Left side: Results List
        results_widget = QWidget()
        results_layout = QVBoxLayout(results_widget)
        results_layout.setContentsMargins(0,0,0,0)
        results_layout.addWidget(QLabel("Relevant Chunks:"))
        self.results_list = QListWidget()
        self.results_list.setWordWrap(False)
        self.results_list.itemClicked.connect(self._on_result_item_clicked)
        results_layout.addWidget(self.results_list)
        bottom_splitter.addWidget(results_widget)

        # Right side: Content/Answer View
        content_widget = QWidget()
        content_layout = QVBoxLayout(content_widget)
        content_layout.setContentsMargins(0,0,0,0)
        content_layout.addWidget(QLabel("Note Content:"))
        self.answer_output = QTextEdit()
        self.answer_output.setReadOnly(True)
        self.answer_output.setLineWrapMode(QTextEdit.LineWrapMode.WidgetWidth)
        font = self.answer_output.font(); font.setPointSize(11); self.answer_output.setFont(font)
        content_layout.addWidget(self.answer_output)
        bottom_splitter.addWidget(content_widget)

        bottom_splitter.setStretchFactor(0, 1) # Results list takes ~1/3
        bottom_splitter.setStretchFactor(1, 2) # Content view takes ~2/3
        main_layout.addWidget(bottom_splitter, stretch=1)

        # Status Bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100); self.progress_bar.setValue(0)
        self.progress_bar.setVisible(False); self.progress_bar.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.progress_bar.setMaximumWidth(200)
        self.status_bar.addPermanentWidget(self.progress_bar)
        self.status_bar.showMessage("Ready.")
        logger.info("Finished UI setup.")

    def _create_menus(self) -> None:
        """Creates the main menu bar and its actions."""
        logger.debug("Creating menus...")
        menu_bar = self.menuBar()
        file_menu = menu_bar.addMenu("&File")
        settings_action = QAction("&Settings", self)
        settings_action.setStatusTip("Configure application settings")
        settings_action.triggered.connect(self._open_settings)
        file_menu.addAction(settings_action)
        file_menu.addSeparator()
        exit_action = QAction("&Exit", self)
        exit_action.setStatusTip("Exit the application")
        exit_action.setShortcut("Ctrl+Q")
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        logger.debug("Menus created.")

    def _update_ui_state(self) -> None:
        """Updates the enabled/disabled state of UI elements."""
        logger.debug("Updating UI state...")
        folder_selected = bool(self._notes_folder_path)
        is_indexing = self._indexing_thread is not None and self._indexing_thread.isRunning()
        is_querying = self._query_thread is not None and self._query_thread.isRunning()

        if folder_selected: self.folder_label.setText(f"Notes: {self._notes_folder_path}")
        else: self.folder_label.setText("No folder selected.")

        busy = is_indexing or is_querying
        self.select_folder_button.setEnabled(not busy)
        self.index_button.setEnabled(folder_selected and not busy)

        db_path = self._config_mgr.get_db_path()
        db_exists = db_path is not None and db_path.exists()
        can_query = folder_selected and db_exists and not busy
        self.prompt_input.setEnabled(can_query)
        self.ask_button.setEnabled(can_query)

        self.progress_bar.setVisible(busy)
        if not busy: self.progress_bar.setValue(0)

        if is_indexing: self.status_bar.showMessage("Indexing in progress...")
        elif is_querying: self.status_bar.showMessage("Searching notes...")
        elif folder_selected: self.status_bar.showMessage("Ready.")
        else: self.status_bar.showMessage("Please select a notes folder.")

        logger.debug(f"UI state updated: folder={folder_selected}, indexing={is_indexing}, querying={is_querying}, can_query={can_query}")

    def _select_notes_folder(self) -> None:
        """Opens a dialog to select the notes folder and updates configuration."""
        logger.info("Select Notes Folder button clicked.")
        current_folder = self._notes_folder_path or str(Path.home())
        folder = QFileDialog.getExistingDirectory(self, "Select Notes Folder", current_folder)
        if folder:
            folder_path = str(Path(folder).resolve())
            logger.info(f"Notes folder selected: {folder_path}")
            self._notes_folder_path = folder_path
            self._config_mgr.set_notes_folder(folder_path)
            self.results_list.clear() # Clear results when folder changes
            self.answer_output.clear()
            self._update_ui_state()
        else:
             logger.info("Folder selection cancelled by user.")

    def _start_indexing(self) -> None:
        """Starts the indexing process in a separate thread."""
        logger.info("Index Notes button clicked.")
        if not self._notes_folder_path:
            QMessageBox.warning(self, "Folder Not Set", "Please select a notes folder first."); return
        if self._indexing_thread and self._indexing_thread.isRunning():
            QMessageBox.information(self, "Indexing Running", "Indexing is already in progress."); return
        if self._query_thread and self._query_thread.isRunning():
             QMessageBox.information(self, "Busy", "Please wait for the current query to finish."); return

        db_path = self._config_mgr.get_db_path()
        embedding_model_name = self._config_mgr.get_embedding_model_name()
        allowed_extensions = self._config_mgr.get_indexed_extensions()

        if not db_path: QMessageBox.critical(self, "Error", "Could not determine database path."); return
        if not allowed_extensions: QMessageBox.critical(self, "Config Error", "No file extensions configured for indexing."); return
        try: db_path.parent.mkdir(parents=True, exist_ok=True)
        except OSError as e: QMessageBox.critical(self, "Error", f"Could not create directory for database: {e}"); return

        logger.info(f"Starting indexing: Folder={self._notes_folder_path}, Ext={allowed_extensions}, DB={db_path}, Model={embedding_model_name}")
        self.status_bar.showMessage("Starting indexing..."); self.progress_bar.setRange(0, 0); self.progress_bar.setValue(-1)
        self._update_ui_state()

        self._indexing_thread = QThread(self)
        self._indexing_worker = IndexerWorker(
            notes_folder=self._notes_folder_path,
            db_path=db_path,
            embedding_model_name=embedding_model_name,
            allowed_extensions=allowed_extensions
        )
        self._indexing_worker.moveToThread(self._indexing_thread)
        self._indexing_worker.finished.connect(self._on_indexing_finished)
        self._indexing_worker.progress.connect(self._update_progress)
        self._indexing_thread.started.connect(self._indexing_worker.run)
        self._indexing_worker.finished.connect(self._indexing_thread.quit)
        self._indexing_thread.finished.connect(self._indexing_worker.deleteLater)
        self._indexing_thread.finished.connect(self._clear_indexing_references)
        self._indexing_thread.start()
        logger.info("Indexing thread started.")

    def _on_indexing_finished(self, success: bool, message: str) -> None:
        """Slot called when the IndexerWorker emits the 'finished' signal."""
        logger.info(f"Indexing finished signal. Success: {success}, Msg: {message}")
        self.status_bar.showMessage(message, 10000)
        self.progress_bar.setVisible(False)
        self.progress_bar.setRange(0, 100); self.progress_bar.setValue(0)

        if success: QMessageBox.information(self, "Indexing Complete", message)
        elif "cancelled" in message.lower(): QMessageBox.warning(self, "Indexing Cancelled", message)
        else: QMessageBox.critical(self, "Indexing Error", message)
        # State update happens in _clear_indexing_references

    def _clear_indexing_references(self):
        """Slot called when the indexing QThread finishes."""
        logger.debug("Clearing indexing thread references.")
        self._indexing_thread = None
        self._indexing_worker = None
        self._update_ui_state()

    def _update_progress(self, message: str) -> None:
        """Slot called when the IndexerWorker emits the 'progress' signal."""
        logger.debug(f"Indexing progress: {message}")
        self.status_bar.showMessage(message)
        # Progress bar parsing logic (can be improved)
        if "Processing file" in message:
             try:
                 parts = message.split(" "); current, total = map(int, parts[2].split('/')[0:2])
                 if total > 0: percent = int((current / total) * 95); self.progress_bar.setRange(0, 100); self.progress_bar.setValue(percent)
                 else: self.progress_bar.setRange(0, 0); self.progress_bar.setValue(-1)
             except: self.progress_bar.setRange(0, 0); self.progress_bar.setValue(-1) # Fallback
        elif "Embedding" in message or "Storing" in message: self.progress_bar.setRange(0, 100); self.progress_bar.setValue(98)
        elif "Scanning" in message or "Checking" in message or "Initializing" in message: self.progress_bar.setRange(0, 0); self.progress_bar.setValue(-1)

    def _handle_query(self) -> None:
        """Handles the 'Ask' button click. Starts the QueryWorker."""
        logger.debug("Handle query triggered.")
        query_text = self.prompt_input.text().strip()
        if not query_text: return

        if not self._notes_folder_path: QMessageBox.warning(self, "Folder Not Set", "Please select a notes folder first."); return
        if self._indexing_thread and self._indexing_thread.isRunning(): QMessageBox.information(self, "Busy", "Please wait for indexing to finish."); return
        if self._query_thread and self._query_thread.isRunning(): QMessageBox.information(self, "Busy", "Please wait for the current query to finish."); return

        db_path = self._config_mgr.get_db_path()
        embedding_model_name = self._config_mgr.get_embedding_model_name()

        if not db_path or not db_path.exists(): QMessageBox.warning(self, "Database Not Found", "Notes database not found. Please index notes first."); return

        logger.info(f"Starting query: '{query_text[:50]}...'")
        self.results_list.clear()
        self.answer_output.setText("Searching...") # Show immediate feedback
        self.status_bar.showMessage("Searching notes..."); self.progress_bar.setRange(0, 0); self.progress_bar.setValue(-1)
        self._update_ui_state()

        self._query_thread = QThread(self)
        self._query_worker = QueryWorker(
            query_text=query_text, db_path=db_path,
            embedding_model_name=embedding_model_name, notes_folder=self._notes_folder_path
        )
        self._query_worker.moveToThread(self._query_thread)
        self._query_worker.results_ready.connect(self._on_search_results)
        self._query_worker.error_occurred.connect(self._on_query_error)
        self._query_thread.started.connect(self._query_worker.run)
        self._query_worker.results_ready.connect(self._query_thread.quit)
        self._query_worker.error_occurred.connect(self._query_thread.quit)
        self._query_thread.finished.connect(self._query_worker.deleteLater)
        self._query_thread.finished.connect(self._clear_query_references)
        self._query_thread.start()
        logger.info("Query thread started.")

    def _on_search_results(self, results: List[ResultItem]):
        """Slot called when QueryWorker emits 'results_ready'."""
        logger.info(f"Received {len(results)} search results.")
        self.status_bar.showMessage(f"Found {len(results)} relevant chunks.", 5000)
        self.answer_output.clear() # Clear "Searching..." message

        self.results_list.clear()
        if not results:
            item = QListWidgetItem("No relevant chunks found.")
            item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsSelectable)
            self.results_list.addItem(item)
        else:
            for result_item in results:
                list_item = QListWidgetItem(result_item.display_text())
                list_item.setData(Qt.ItemDataRole.UserRole, result_item)
                list_item.setToolTip(f"Path: {result_item.file_path}\nScore: {result_item.score:.4f}")
                self.results_list.addItem(list_item)

        # TODO (Phase 3c): Optionally generate LLM summary here and display in self.answer_output
        # For now, just clear the prompt
        self.prompt_input.clear()
        # Cleanup handled by _clear_query_references

    def _on_query_error(self, error_message: str):
        """Slot called when QueryWorker emits 'error_occurred'."""
        logger.error(f"Query error signal received: {error_message}")
        self.status_bar.showMessage("Query failed.", 5000)
        QMessageBox.critical(self, "Query Error", f"An error occurred during the search:\n{error_message}")
        self.results_list.clear()
        self.answer_output.setText("Error during search.") # Show error in main view too
        # Cleanup handled by _clear_query_references

    def _clear_query_references(self):
        """Slot called when the query QThread finishes."""
        logger.debug("Query thread finished, clearing references.")
        self._query_thread = None
        self._query_worker = None
        self._update_ui_state()

    def _on_result_item_clicked(self, item: QListWidgetItem):
        """Slot called when an item in the results list is clicked."""
        result_data: Optional[ResultItem] = item.data(Qt.ItemDataRole.UserRole)
        if not result_data:
            logger.warning("Clicked list item has no ResultItem data.")
            self.answer_output.setText("Error: Could not load data for this item.")
            return

        logger.info(f"Result item clicked: {result_data.file_path.name} - Chunk {result_data.chunk_index}")
        self.status_bar.showMessage(f"Loading content for {result_data.file_path.name}...")
        QApplication.processEvents()

        try:
            full_content = FileHandler.read_file_content(result_data.file_path)
            if full_content is None:
                self.answer_output.setText(f"Error: Could not read file:\n{result_data.file_path}")
                self.status_bar.showMessage("Error reading file.", 5000)
                return

            self.answer_output.setPlainText(full_content)
            chunk_text = result_data.content
            cursor = self.answer_output.textCursor()
            start_pos = full_content.find(chunk_text)

            if start_pos != -1:
                cursor.setPosition(start_pos)
                cursor.movePosition(QTextCursor.MoveOperation.Right, QTextCursor.MoveMode.KeepAnchor, len(chunk_text))
                self.answer_output.setTextCursor(cursor)
                self.answer_output.ensureCursorVisible()
                logger.debug(f"Scrolled and highlighted chunk {result_data.chunk_index} in {result_data.file_path.name}.")
            else:
                logger.warning(f"Could not find exact chunk text within {result_data.file_path.name}. Scrolling to top.")
                cursor.movePosition(QTextCursor.MoveOperation.Start)
                self.answer_output.setTextCursor(cursor); self.answer_output.ensureCursorVisible()

            self.status_bar.showMessage(f"Displayed content for {result_data.file_path.name}", 5000)

        except Exception as e:
            logger.error(f"Error displaying file content for {result_data.file_path}: {e}", exc_info=True)
            self.answer_output.setText(f"Error loading content:\n{e}")
            self.status_bar.showMessage("Error displaying content.", 5000)

    def _open_settings(self) -> None:
        """Opens the settings dialog (placeholder)."""
        logger.info("Settings action triggered.")
        QMessageBox.information(self, "Settings", "Settings dialog not implemented yet.")

    def closeEvent(self, event) -> None:
        """Handles the window closing event."""
        logger.debug("Close event triggered.")
        is_indexing = self._indexing_thread and self._indexing_thread.isRunning()
        is_querying = self._query_thread and self._query_thread.isRunning()

        if is_indexing or is_querying:
            operation = "Indexing" if is_indexing else "Querying"
            logger.warning(f"Close requested while {operation} is in progress.")
            reply = QMessageBox.question(self, 'Confirm Exit',
                                        f"{operation} is in progress. Are you sure you want to exit?",
                                        QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                                        QMessageBox.StandardButton.No)
            if reply == QMessageBox.StandardButton.Yes:
                 logger.info(f"User confirmed exit during {operation}.")
                 if is_indexing and self._indexing_worker: self._indexing_worker.stop()
                 # Add query cancellation here if QueryWorker implements stop() later
                 event.accept()
            else:
                 logger.info(f"User cancelled exit during {operation}.")
                 event.ignore()
        else:
             logger.info("Closing application.")
             event.accept()