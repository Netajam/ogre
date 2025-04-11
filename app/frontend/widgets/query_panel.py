# app/frontend/widgets/query_panel.py
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QLineEdit)
from PyQt6.QtCore import pyqtSignal

# Assume logger is available via from app.utils.logging_setup import logger if needed

class QueryPanelWidget(QWidget):
    """Widget for query input and action buttons."""
    querySubmitted = pyqtSignal(str, str) # query_text, mode
    stopQueryClicked = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self):
        outer_layout = QVBoxLayout(self)
        outer_layout.setContentsMargins(0, 5, 0, 5) # Add some vertical margins

        query_layout = QHBoxLayout()
        query_layout.addWidget(QLabel("Query:"))
        self.prompt_input = QLineEdit()
        self.prompt_input.setPlaceholderText("Ask something about your notes...")
        # --- Connect returnPressed to internal slot ---
        self.prompt_input.returnPressed.connect(self._on_prompt_return_pressed)
        query_layout.addWidget(self.prompt_input, 1)
        outer_layout.addLayout(query_layout)

        button_layout = QHBoxLayout()
        self.search_chunks_button = QPushButton("1. Search Chunks")
        self.search_and_answer_button = QPushButton("2. Search & Answer")
        self.search_edit_prompt_button = QPushButton("3. Search & Edit Prompt")
        self.stop_query_button = QPushButton("Stop")
        self.stop_query_button.setEnabled(False) # Initially disabled

        button_layout.addWidget(self.search_chunks_button)
        button_layout.addWidget(self.search_and_answer_button)
        button_layout.addWidget(self.search_edit_prompt_button)
        button_layout.addStretch()
        button_layout.addWidget(self.stop_query_button)
        outer_layout.addLayout(button_layout)

        # --- Connect button clicks to _emit_query helper ---
        self.search_chunks_button.clicked.connect(lambda: self._emit_query("chunks_only"))
        self.search_and_answer_button.clicked.connect(lambda: self._emit_query("auto_answer"))
        self.search_edit_prompt_button.clicked.connect(lambda: self._emit_query("edit_prompt"))
        # --- Connect stop button directly to its signal ---
        self.stop_query_button.clicked.connect(self.stopQueryClicked)

    def _on_prompt_return_pressed(self):
        """Handle Enter key press in the prompt input."""
        # Emit the signal with the current text and default mode ("auto_answer")
        self._emit_query("auto_answer")

    def _emit_query(self, mode: str):
        """Reads text from input and emits the querySubmitted signal."""
        query_text = self.prompt_input.text().strip()
        if query_text:
            # Emit the signal with both text and mode
            self.querySubmitted.emit(query_text, mode)
        # Do not clear the prompt here; MainWindow will clear it via public method

    # --- Public methods for MainWindow to control this panel ---
    def clear_prompt(self):
        self.prompt_input.clear()

    def set_query_enabled(self, enabled: bool):
        """Enable/disable query input and action buttons."""
        self.prompt_input.setEnabled(enabled)
        self.search_chunks_button.setEnabled(enabled)
        self.search_and_answer_button.setEnabled(enabled)
        self.search_edit_prompt_button.setEnabled(enabled)
        # Stop button state is handled separately by set_stop_enabled

    def set_stop_enabled(self, enabled: bool):
        """Enable/disable the stop button."""
        self.stop_query_button.setEnabled(enabled)

    def get_query_text(self) -> str:
        """Returns the current text in the input field."""
        # Added this in case _show_prompt_editor needs it reliably
        return self.prompt_input.text().strip()