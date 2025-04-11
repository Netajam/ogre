# app/core/services/result_item.py
from dataclasses import dataclass
from pathlib import Path

@dataclass
class ResultItem:
    """Holds information about a single retrieved chunk."""
    file_path: Path
    chunk_index: int
    content: str
    score: float # Similarity score (distance)

    def __str__(self) -> str:
        # Basic string representation for display or logging
        return f"File: {self.file_path.name} | Chunk: {self.chunk_index} | Score: {self.score:.4f}"

    def display_text(self) -> str:
        """Text suitable for display in a list widget."""
        # Truncate long content for list display
        preview_len = 80
        content_preview = self.content.replace('\n', ' ').strip()
        if len(content_preview) > preview_len:
            content_preview = content_preview[:preview_len-3] + "..."
        return f"'{content_preview}' [{self.file_path.name} - Chunk {self.chunk_index}] (Score: {self.score:.3f})"