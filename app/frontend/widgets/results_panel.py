# app/frontend/widgets/results_panel.py
from typing import List, Optional
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QLabel, QListWidget, QListWidgetItem)
from PyQt6.QtCore import pyqtSignal, Qt

# Assuming ResultItem is accessible, adjust import if needed
from app.core.services.result_item import ResultItem

class ResultsPanelWidget(QWidget):
    """Widget to display the list of relevant chunks."""
    itemClicked = pyqtSignal(QListWidgetItem) # Re-emit itemClicked

    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(QLabel("Relevant Chunks:"))
        self.results_list = QListWidget()
        self.results_list.setWordWrap(False)
        self.results_list.itemClicked.connect(self.itemClicked) # Connect internal signal
        layout.addWidget(self.results_list)

    def update_results(self, results: List[ResultItem]):
        """Clears and populates the list with new results."""
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

    def clear_results(self):
        self.results_list.clear()