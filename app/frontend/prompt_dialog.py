# app/frontend/prompt_dialog.py
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QTextEdit, QDialogButtonBox, QLabel, QScrollArea
)
from PyQt6.QtCore import Qt

class PromptEditDialog(QDialog):
    """A dialog to display and edit the LLM prompt."""

    def __init__(self, initial_prompt: str, query: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Edit LLM Prompt")
        self.setMinimumSize(600, 500) # Adjust size as needed

        layout = QVBoxLayout(self)

        # Display original query
        layout.addWidget(QLabel(f"<b>Original Query:</b> {query}"))

        # Add label for prompt editor
        layout.addWidget(QLabel("Generated Prompt for LLM:"))

        # Prompt editor
        self.prompt_edit = QTextEdit()
        self.prompt_edit.setPlainText(initial_prompt)
        self.prompt_edit.setLineWrapMode(QTextEdit.LineWrapMode.WidgetWidth)
        font = self.prompt_edit.font(); font.setPointSize(10); self.prompt_edit.setFont(font)

        # Make text edit scrollable if needed
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setWidget(self.prompt_edit)
        layout.addWidget(scroll_area, stretch=1) # Allow scroll area to expand

        # Standard buttons (Send, Cancel)
        self.button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        self.button_box.button(QDialogButtonBox.StandardButton.Ok).setText("Send to LLM")
        self.button_box.accepted.connect(self.accept) # Connect Ok signal
        self.button_box.rejected.connect(self.reject) # Connect Cancel signal
        layout.addWidget(self.button_box)

        self.setLayout(layout)

    def get_edited_prompt(self) -> str:
        """Returns the potentially modified prompt text."""
        return self.prompt_edit.toPlainText()