# app/frontend/widgets/top_panel.py
from PyQt6.QtWidgets import (QWidget, QHBoxLayout, QPushButton, QLabel, QSizePolicy)
from PyQt6.QtCore import pyqtSignal

class TopPanelWidget(QWidget):
    """Widget for folder selection and indexing controls."""
    # Define signals for button clicks
    selectFolderClicked = pyqtSignal()
    indexClicked = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0) # Remove margins if desired

        self.select_folder_button = QPushButton("Select Notes Folder")
        self.select_folder_button.clicked.connect(self.selectFolderClicked) # Emit signal

        self.folder_label = QLabel("No folder selected.")
        self.folder_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        self.folder_label.setWordWrap(True)

        self.index_button = QPushButton("Index Notes")
        self.index_button.clicked.connect(self.indexClicked) # Emit signal

        layout.addWidget(self.select_folder_button)
        layout.addWidget(self.folder_label, 1) # Stretch label
        layout.addWidget(self.index_button)

    def set_folder_label(self, text: str):
        self.folder_label.setText(text)

    def set_busy(self, busy: bool):
        """Enable/disable buttons based on busy state."""
        self.select_folder_button.setEnabled(not busy)
        self.index_button.setEnabled(not busy and self.folder_label.text() != "No folder selected.") # Also check if folder is selected