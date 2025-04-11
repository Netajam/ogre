# app/frontend/widgets/display_panel.py
from typing import Optional
from pathlib import Path
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
                             QTextEdit, QStackedWidget, QApplication, QLabel) # Added QLabel
from PyQt6.QtCore import pyqtSignal, Qt, QTimer
from PyQt6.QtGui import QTextCursor

# Assuming logger is imported if needed: from app.utils.logging_setup import logger

class DisplayPanelWidget(QWidget):
    """Widget with toggle buttons and stacked view for Note/LLM content."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._has_llm_content = False
        self._has_note_content = False
        self._setup_ui()
        self._connect_signals()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # View Toggle Buttons
        self.view_toggle_layout = QHBoxLayout()
        self.view_llm_button = QPushButton("View Generated Answer")
        self.view_note_button = QPushButton("View Note Content")
        self.view_llm_button.setCheckable(True)
        self.view_note_button.setCheckable(True)
        self.view_toggle_layout.addWidget(self.view_llm_button)
        self.view_toggle_layout.addWidget(self.view_note_button)
        self.view_toggle_layout.addStretch()
        layout.addLayout(self.view_toggle_layout)

        # Stacked Widget
        self.display_stack = QStackedWidget()
        layout.addWidget(self.display_stack)

        # Page 0: Note Content View
        note_content_container = QWidget()
        note_content_layout = QVBoxLayout(note_content_container)
        note_content_layout.setContentsMargins(0, 5, 0, 0)
        note_content_layout.addWidget(QLabel("<b>Note Content:</b>"))
        self.note_output_view = self._create_text_display()
        note_content_layout.addWidget(self.note_output_view)
        self.display_stack.addWidget(note_content_container) # Index 0

        # Page 1: LLM Answer View
        llm_answer_container = QWidget()
        llm_answer_layout = QVBoxLayout(llm_answer_container)
        llm_answer_layout.setContentsMargins(0, 5, 0, 0)
        llm_answer_layout.addWidget(QLabel("<b>Generated Answer:</b>"))
        self.llm_output_view = self._create_text_display()
        llm_answer_layout.addWidget(self.llm_output_view)
        self.display_stack.addWidget(llm_answer_container) # Index 1

        # Initial state
        self.display_stack.setCurrentIndex(0) # Start on note view
        self.view_note_button.setChecked(True)
        # self.view_llm_button.setEnabled(False)  # <-- REMOVE THIS LINE
        # self.view_note_button.setEnabled(False) # <-- REMOVE THIS LINE

    def _create_text_display(self) -> QTextEdit:
        """Creates a standard QTextEdit widget."""
        text_edit = QTextEdit()
        text_edit.setReadOnly(True)
        text_edit.setLineWrapMode(QTextEdit.LineWrapMode.WidgetWidth)
        font = text_edit.font(); font.setPointSize(11); text_edit.setFont(font)
        return text_edit

    def _connect_signals(self):
        """Connect internal signals."""
        self.view_llm_button.toggled.connect(self._on_llm_button_toggled)
        self.view_note_button.toggled.connect(self._on_note_button_toggled)

    # --- Internal Slots for Toggle Buttons ---
    def _on_llm_button_toggled(self, checked):
        """Handles the LLM view button toggle."""
        if checked:
            current_index = self.display_stack.currentIndex()
            if current_index != 1:
                self.display_stack.setCurrentIndex(1)
            if self.view_note_button.isChecked():
                self.view_note_button.blockSignals(True)
                self.view_note_button.setChecked(False)
                self.view_note_button.blockSignals(False)

    def _on_note_button_toggled(self, checked):
        """Handles the Note view button toggle."""
        if checked:
            current_index = self.display_stack.currentIndex()
            if current_index != 0:
                self.display_stack.setCurrentIndex(0)
            if self.view_llm_button.isChecked():
                self.view_llm_button.blockSignals(True)
                self.view_llm_button.setChecked(False)
                self.view_llm_button.blockSignals(False)

    # --- Public Methods for MainWindow ---
    def set_note_content(self, file_path: Path, full_content: str, highlight_chunk: Optional[str] = None):
        """Displays note content and optionally highlights a chunk."""
        self.note_output_view.setPlainText(full_content)
        self._has_note_content = True # Mark that we have note content
        if highlight_chunk:
            cursor = self.note_output_view.textCursor()
            start_pos = full_content.find(highlight_chunk)
            if start_pos != -1:
                cursor.setPosition(start_pos)
                cursor.movePosition(QTextCursor.MoveOperation.Right, QTextCursor.MoveMode.KeepAnchor, len(highlight_chunk))
            else:
                cursor.movePosition(QTextCursor.MoveOperation.Start)
            self.note_output_view.setTextCursor(cursor)
            QTimer.singleShot(0, self.note_output_view.ensureCursorVisible)

    def set_llm_answer(self, answer: str):
        """Displays the LLM generated answer."""
        self.llm_output_view.setPlainText(answer)
        self._has_llm_content = bool(answer and "[Processing stopped" not in answer) # Update flag based on real answer

    def show_view(self, index: int):
        """Programmatically switches the view and updates button check states."""
        if index == 0: # Show Note View
            if not self.view_note_button.isChecked():
                 self.view_note_button.setChecked(True)
        elif index == 1: # Show LLM View
            if not self.view_llm_button.isChecked():
                 self.view_llm_button.setChecked(True)

    def clear_displays(self):
        """Clears both text views and resets content flags."""
        self.note_output_view.clear()
        self.llm_output_view.clear()
        self._has_note_content = False
        self._has_llm_content = False
        # Default back to note view checked, let set_busy handle enable state
        self.view_note_button.setChecked(True)
        self.view_llm_button.setChecked(False)
        # No need to explicitly call set_busy here, MainWindow._update_ui_state will

    def set_busy(self, busy: bool):
        """Enables/disables view toggle buttons based ONLY on busy state."""
        # --- MODIFIED LOGIC ---
        # Buttons should be enabled unless the application is busy
        is_enabled = not busy
        self.view_llm_button.setEnabled(is_enabled)
        self.view_note_button.setEnabled(is_enabled)
        # ----------------------

        # Sync checked state only if buttons are enabled (otherwise they can't be checked)
        if is_enabled:
            current_index = self.display_stack.currentIndex()
            # Only change check state if it's different from the current view index
            if self.view_note_button.isChecked() != (current_index == 0):
                self.view_note_button.setChecked(current_index == 0)
            if self.view_llm_button.isChecked() != (current_index == 1):
                self.view_llm_button.setChecked(current_index == 1)