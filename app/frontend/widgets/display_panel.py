# app/frontend/widgets/display_panel.py
import sys # Keep sys if needed elsewhere, not strictly necessary here
from typing import Optional
from pathlib import Path

# --- Add Markdown library import ---
try:
    import markdown
    MARKDOWN_AVAILABLE = True
except ImportError:
    MARKDOWN_AVAILABLE = False
# --- Add a print statement for immediate feedback during startup ---
# (This will show in the console where you run the app)
print(f"--- MARKDOWN LIBRARY AVAILABLE: {MARKDOWN_AVAILABLE} ---")
# -------------------------------------------------------------

from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
                             QTextEdit, QStackedWidget, QApplication, QLabel)
from PyQt6.QtCore import pyqtSignal, Qt, QTimer
from PyQt6.QtGui import QTextCursor, QFont

# --- Assuming logger is imported if needed ---
# from app.utils.logging_setup import logger
# Replace with your actual logger import if you use it extensively here
import logging # Using standard logging if no app logger passed
logger = logging.getLogger(__name__) # Get a logger for this module
# ----------------------------------------------

# --- Define Markdown Extensions ---
MARKDOWN_EXTENSIONS = ['fenced_code', 'tables', 'attr_list', 'md_in_html', 'footnotes']
MARKDOWN_FILE_EXTENSIONS = ['.md', '.markdown']
# ----------------------------------

class DisplayPanelWidget(QWidget):
    """Widget with toggle buttons and stacked view for Note/LLM content."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._has_llm_content = False
        self._has_note_content = False
        self._current_note_path: Optional[Path] = None
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
        self.note_output_view = self._create_text_display(markdown_support=True)
        note_content_layout.addWidget(self.note_output_view)
        self.display_stack.addWidget(note_content_container) # Index 0

        # Page 1: LLM Answer View
        llm_answer_container = QWidget()
        llm_answer_layout = QVBoxLayout(llm_answer_container)
        llm_answer_layout.setContentsMargins(0, 5, 0, 0)
        llm_answer_layout.addWidget(QLabel("<b>Generated Answer:</b>"))
        self.llm_output_view = self._create_text_display() # No markdown needed for LLM
        llm_answer_layout.addWidget(self.llm_output_view)
        self.display_stack.addWidget(llm_answer_container) # Index 1

        # Initial state
        self.display_stack.setCurrentIndex(0)
        self.view_note_button.setChecked(True)
        # Buttons enabled by default, managed by set_busy

    def _create_text_display(self, markdown_support=False) -> QTextEdit:
        """Creates a standard QTextEdit widget."""
        text_edit = QTextEdit()
        text_edit.setReadOnly(True)
        default_font_size = 11
        # Set base font size, specific styling is better handled by HTML/CSS for Markdown
        font = text_edit.font()
        font.setPointSize(default_font_size)
        text_edit.setFont(font)
        text_edit.setLineWrapMode(QTextEdit.LineWrapMode.WidgetWidth)
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
        """
        Displays note content. Renders as Markdown if applicable, otherwise plain text.
        Optionally highlights a chunk (best effort for Markdown - currently disabled).
        """
        logger.debug(f"set_note_content called for: {file_path}")
        self._current_note_path = file_path # Store path (Removed duplicate)
        is_markdown = file_path.suffix.lower() in MARKDOWN_FILE_EXTENSIONS
        logger.debug(f"File suffix: {file_path.suffix.lower()}, is_markdown: {is_markdown}")

        processed_for_highlighting = False # Flag for highlighting status

        if is_markdown and MARKDOWN_AVAILABLE:
            logger.debug("Attempting Markdown rendering...")
            try:
                content_to_convert = full_content

                # --- Highlighting Strategy for Markdown (Currently Disabled) ---
                # Injecting HTML before conversion is fragile and often breaks Markdown parsing.
                # Commented out for now to ensure rendering works first.
                # A robust solution would require parsing the Markdown AST or complex regex.
                '''
                if highlight_chunk:
                    logger.debug("Attempting to inject HTML for highlighting (experimental)...")
                    start_pos = full_content.find(highlight_chunk)
                    if start_pos != -1:
                        end_pos = start_pos + len(highlight_chunk)
                        if '<' not in highlight_chunk and '>' not in highlight_chunk: # Basic safety check
                            highlight_start_tag = "<span id='highlight' style='background-color: #f0e68c;'>"
                            highlight_end_tag = "</span>"
                            content_to_convert = (
                                full_content[:start_pos] + highlight_start_tag +
                                full_content[start_pos:end_pos] + highlight_end_tag +
                                full_content[end_pos:]
                            )
                            processed_for_highlighting = True
                            logger.debug("Highlight span injected.")
                        else:
                            logger.warning("Highlight chunk contains '<' or '>', skipping HTML injection.")
                    else:
                        logger.warning("Could not find highlight chunk in Markdown source for HTML injection.")
                '''
                # ---------------------------------------------------------------

                logger.debug("Converting content to HTML using markdown library...")
                html_content = markdown.markdown(content_to_convert, extensions=MARKDOWN_EXTENSIONS)
                logger.debug("Markdown conversion successful.")

                # --- Add basic CSS for better rendering ---
                basic_css = """
                <style>
                    body { font-family: sans-serif; line-height: 1.6; font-size: 11pt; } /* Match default font size */
                    h1, h2, h3, h4, h5, h6 { margin-top: 1em; margin-bottom: 0.5em; line-height: 1.2; font-weight: bold; }
                    h1 { font-size: 1.8em; } h2 { font-size: 1.5em; } h3 { font-size: 1.25em; }
                    p { margin-bottom: 1em; }
                    code {
                        font-family: Consolas, Monaco, 'Andale Mono', 'Ubuntu Mono', monospace;
                        background-color: #f0f0f0; padding: 0.2em 0.4em;
                        border-radius: 3px; font-size: 90%;
                    }
                    pre { /* Wrapper for code blocks */
                        background-color: #f0f0f0; padding: 1em;
                        border-radius: 4px; overflow-x: auto;
                        margin-bottom: 1em; /* Add margin to pre tag */
                    }
                    pre > code { /* Code inside pre should not have extra padding/background */
                        background-color: transparent; padding: 0;
                        border-radius: 0; font-size: 100%; /* Reset font size for code block */
                        display: block; /* Ensure code fills pre */
                    }
                    blockquote {
                        border-left: 4px solid #ccc; padding-left: 1em;
                        margin-left: 0; margin-bottom: 1em; color: #666;
                    }
                    table { border-collapse: collapse; margin-bottom: 1em; width: auto; }
                    th, td { border: 1px solid #ccc; padding: 0.5em; text-align: left; }
                    th { background-color: #f0f0f0; font-weight: bold; }
                    ul, ol { padding-left: 2em; margin-bottom: 1em; }
                    img { max-width: 100%; height: auto; }
                    /* Style for injected highlight span (if used) */
                    #highlight { background-color: #f0e68c; }
                </style>
                """
                final_html = basic_css + html_content
                # ------------------------------------------

                self.note_output_view.setHtml(final_html)
                logger.debug(f"Rendered Markdown file {file_path.name} as HTML.")

                # --- Scrolling for Markdown/HTML ---
                # Since highlighting injection is disabled, scroll to top.
                # If highlighting injection is re-enabled and works, the scrollToAnchor logic can be used.
                # if processed_for_highlighting:
                #    logger.debug("Scrolling to highlight anchor...")
                #    QTimer.singleShot(50, lambda: self.note_output_view.scrollToAnchor('highlight'))
                # else:
                logger.debug("Scrolling Markdown view to top.")
                cursor = self.note_output_view.textCursor()
                cursor.movePosition(QTextCursor.MoveOperation.Start)
                self.note_output_view.setTextCursor(cursor)

            except Exception as e:
                # --- Ensure full error with traceback is logged ---
                logger.error(f"Error during Markdown processing for {file_path}: {e}", exc_info=True)
                # -------------------------------------------------
                # Fallback to plain text if conversion fails
                logger.warning("Falling back to plain text display due to Markdown processing error.")
                self.note_output_view.setPlainText(full_content)
                self._highlight_plain_text(full_content, highlight_chunk)

        else:
            # --- Handle Plain Text ---
            if is_markdown and not MARKDOWN_AVAILABLE:
                logger.warning("Markdown library not found. Displaying .md file as plain text.")
            else:
                logger.debug(f"Displaying file {file_path.name} as plain text.")

            self.note_output_view.setPlainText(full_content) # Use original content
            # Apply plain text highlighting if needed
            self._highlight_plain_text(full_content, highlight_chunk)

        self._has_note_content = True # Mark that content is now displayed

    def _highlight_plain_text(self, full_content: str, highlight_chunk: Optional[str]):
        """Helper method to highlight a chunk in plain text view."""
        if not highlight_chunk:
            # Just scroll to top if no chunk specified
            cursor = self.note_output_view.textCursor()
            cursor.movePosition(QTextCursor.MoveOperation.Start)
            self.note_output_view.setTextCursor(cursor)
            # logger.debug("No chunk to highlight in plain text, moving cursor to start.")
            return

        cursor = self.note_output_view.textCursor()
        start_pos = full_content.find(highlight_chunk)
        if start_pos != -1:
            cursor.setPosition(start_pos)
            cursor.movePosition(QTextCursor.MoveOperation.Right, QTextCursor.MoveMode.KeepAnchor, len(highlight_chunk))
            logger.debug("Highlighting plain text chunk.")
        else:
            logger.warning("Could not find highlight chunk in plain text. Moving cursor to start.")
            cursor.movePosition(QTextCursor.MoveOperation.Start)

        self.note_output_view.setTextCursor(cursor)
        # Use singleShot for reliable scrolling after potential layout updates
        QTimer.singleShot(0, self.note_output_view.ensureCursorVisible)


    def set_llm_answer(self, answer: str):
        """Displays the LLM generated answer."""
        self.llm_output_view.setPlainText(answer)
        self._has_llm_content = bool(answer and "[Processing stopped" not in answer)

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
        self._current_note_path = None # Clear stored path
        self.view_note_button.setChecked(True)
        self.view_llm_button.setChecked(False)
        # Let MainWindow._update_ui_state handle button enable state via set_busy

    def set_busy(self, busy: bool):
        """Enables/disables view toggle buttons based ONLY on busy state."""
        is_enabled = not busy
        self.view_llm_button.setEnabled(is_enabled)
        self.view_note_button.setEnabled(is_enabled)

        if is_enabled:
            current_index = self.display_stack.currentIndex()
            if self.view_note_button.isChecked() != (current_index == 0):
                self.view_note_button.setChecked(current_index == 0)
            if self.view_llm_button.isChecked() != (current_index == 1):
                self.view_llm_button.setChecked(current_index == 1)