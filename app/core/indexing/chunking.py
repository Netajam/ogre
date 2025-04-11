# app/core/indexing/chunking.py
from typing import List
from langchain_text_splitters import RecursiveCharacterTextSplitter
from app.core.utils.logging_setup import logger

CHUNK_SIZE = 500  # Target size of each chunk in characters
CHUNK_OVERLAP = 50 # Number of characters to overlap between chunks

# Initialize the splitter once
_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    length_function=len,
    # Try common markdown separators first
    separators=["\n\n", "\n", ". ", ", ", " ", ""],
    is_separator_regex=False,
)

def split_text(document_content: str) -> List[str]:
    """Splits a document's content into smaller chunks."""
    if not document_content:
        return []
    try:
        chunks = _splitter.split_text(document_content)
        logger.debug(f"Split text into {len(chunks)} chunks (size~{CHUNK_SIZE}, overlap~{CHUNK_OVERLAP})")
        # Filter out potential empty strings just in case
        return [chunk for chunk in chunks if chunk.strip()]
    except Exception as e:
        logger.error(f"Error during text splitting: {e}")
        return []