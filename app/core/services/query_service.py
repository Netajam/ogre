# app/core/services/query_service.py
from pathlib import Path
from typing import List, Optional
import numpy as np
import sqlite3
from app.core.database.db_manager import DbManager
from app.core.embedding.embedding_manager import EmbeddingManager
from app.core.services.result_item import ResultItem
from app.utils.logging_setup import logger
from app import config # For task types if using API embedders

class QueryService:
    """Handles the process of querying the indexed notes."""

    def __init__(self, db_path: Path, embedding_model_name: str, notes_folder: str):
        self.db_path = db_path
        self.embedding_model_name = embedding_model_name
        self.notes_folder = notes_folder # Store the current notes folder path
        self.embedder = None
        self.db_manager = None

    def _initialize(self):
        """Initializes embedder and db_manager if not already done."""
        if self.embedder is None:
            logger.info("Initializing embedder for QueryService...")
            self.embedder = EmbeddingManager.get_embedder(self.embedding_model_name)
            if not self.embedder:
                raise ValueError(f"Failed to load embedder: {self.embedding_model_name}")
            logger.info("Embedder initialized.")

        # Note: DbManager connection is handled via context manager ('with') below

    def _get_query_embedding(self, query_text: str) -> np.ndarray:
        """Generates embedding for the user query."""
        if not self.embedder:
            self._initialize() # Ensure embedder is loaded

        logger.info("Generating embedding for query...")
        try:
            # --- Handle Task Type for API Embedders (like Gemini) ---
            # If using an API model, we might want a different task type for queries
            task_type = None
            model_config = config.AVAILABLE_EMBEDDING_MODELS.get(self.embedding_model_name)
            if model_config and model_config.get("type") == "api":
                # Example: Use RETRIEVAL_QUERY for Gemini if available
                if "Gemini" in self.embedding_model_name:
                    task_type = config.GEMINI_TASK_RETRIEVAL_QUERY
                    logger.debug(f"Using task type '{task_type}' for API query embedding.")

                # Check if embedder has a specific method or parameter for task type
                # This part depends on the embedder's implementation (e.g., GeminiEmbedder)
                # For simplicity now, assume embed_batch/embed can handle it or defaults okay
                # You might need:
                # if hasattr(self.embedder, 'embed_query'):
                #      return self.embedder.embed_query(query_text)
                # elif hasattr(self.embedder, 'set_task_type'): # Less ideal to change state
                #      self.embedder.set_task_type(task_type)

            # Use the standard batch method (most embedders handle single string input)
            # Assuming embed_batch returns a 2D array [1, dim] for single input
            query_embedding_2d = self.embedder.embed_batch([query_text])

            if query_embedding_2d is None or query_embedding_2d.shape[0] != 1:
                raise RuntimeError("Failed to generate a valid query embedding.")

            logger.info("Query embedding generated successfully.")
            return query_embedding_2d[0] # Return the 1D vector

        except Exception as e:
            logger.error(f"Error generating query embedding: {e}", exc_info=True)
            raise RuntimeError("Failed to generate query embedding") from e


    def search_chunks(self, query_text: str, top_k: int = 10) -> List[ResultItem]:
        """
        Embeds the query and searches the database for relevant chunks
        within the specified notes folder.

        Args:
            query_text: The user's natural language query.
            top_k: The maximum number of relevant chunks to retrieve.

        Returns:
            A list of ResultItem objects, sorted by relevance (similarity score).
        """
        if not query_text:
            return []

        try:
            query_embedding = self._get_query_embedding(query_text)

            logger.info(f"Searching database for top {top_k} chunks related to query...")
            results = []
            # Use DbManager within a context manager
            with DbManager(self.db_path) as db_manager:
                # Pass the current notes_folder to filter results
                raw_results = db_manager.find_similar_chunks(
                    query_embedding=query_embedding,
                    k=top_k,
                    notes_folder_path=self.notes_folder # Pass the folder path here!
                )

            # Convert raw results to ResultItem objects
            for file_path_str, chunk_index, content, score in raw_results:
                 results.append(ResultItem(
                     file_path=Path(file_path_str), # Convert string back to Path
                     chunk_index=chunk_index,
                     content=content,
                     score=score
                 ))

            logger.info(f"Found {len(results)} relevant chunks.")
            return results

        except (ValueError, RuntimeError, ConnectionError, sqlite3.Error) as e:
            logger.error(f"Search failed: {e}", exc_info=True)
            # Re-raise or return empty list? Returning empty might be better for UI
            return []
        except Exception as e:
             logger.error(f"Unexpected error during search: {e}", exc_info=True)
             return []

    # --- Placeholder for future LLM interaction ---
    # def generate_answer(self, query_text: str, retrieved_chunks: List[ResultItem]) -> str:
    #     logger.info("Generating structured answer using LLM (Not Implemented)...")
    #     # 1. Format context from retrieved_chunks
    #     # 2. Get LLMManager instance
    #     # 3. Call LLMManager.generate(...)
    #     # 4. Return LLM response
    #     context = "\n---\n".join([f"Source: {item.file_path.name}, Chunk {item.chunk_index}:\n{item.content}" for item in retrieved_chunks])
    #     return f"Placeholder LLM Answer for '{query_text}'.\n\nRelevant Context:\n{context}"