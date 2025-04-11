# app/core/services/query_service.py
import os # Import os
from pathlib import Path
from typing import List, Optional, Dict, Any
import numpy as np
import sqlite3

try:
    import google.generativeai as genai
    # Import specific types needed for generation if any (or handle dynamically)
    # from google.generativeai import types as genai_types
    from google.api_core import exceptions as google_exceptions
    GENERATIVE_AI_AVAILABLE = True
except ImportError:
    genai = None
    google_exceptions = None
    GENERATIVE_AI_AVAILABLE = False
    print("WARNING: google-generativeai library not found. LLM/Embedding features disabled.")

from app.core.database.db_manager import DbManager
from app.core.embedding.embedding_manager import EmbeddingManager
from app.core.services.result_item import ResultItem
from app.utils.logging_setup import logger
from app import config # Import config for models, task types, API keys

# Maximum tokens/length for context sent to LLM (adjust based on model limits)
MAX_CONTEXT_LENGTH = 30000 # Approximate character limit, tune this!

class QueryService:
    """Handles embedding queries, searching chunks, and generating LLM answers."""

    def __init__(self, db_path: Path, embedding_model_name: str, llm_model_name: str, notes_folder: str):
        self.db_path = db_path
        self.embedding_model_name = embedding_model_name
        self.llm_model_name = llm_model_name # Store LLM model name
        self.notes_folder = str(Path(notes_folder).resolve())
        self.embedder = None
        self.llm_model = None # Add attribute for the LLM model instance
        self._llm_initialized = False

    def _initialize_embedder(self):
        """Initializes the embedding model."""
        if self.embedder is None:
            logger.info(f"Initializing embedder: {self.embedding_model_name}")
            self.embedder = EmbeddingManager.get_embedder(self.embedding_model_name)
            if not self.embedder: raise ValueError(f"Failed to load embedder: {self.embedding_model_name}")
            logger.info("Embedder initialized.")

    def _initialize_llm(self):
        """Initializes the generative LLM model."""
        if not GENERATIVE_AI_AVAILABLE:
             raise ImportError("google-generativeai library needed for LLM features is not installed.")
        if not self._llm_initialized:
             logger.info(f"Initializing LLM: {self.llm_model_name}")
             llm_config = config.AVAILABLE_LLM_MODELS.get(self.llm_model_name)
             if not llm_config: raise ValueError(f"LLM model '{self.llm_model_name}' not found in configuration.")

             api_key_env = llm_config.get("api_key_env")
             api_key = os.getenv(api_key_env) if api_key_env else None
             if not api_key: raise ValueError(f"API key environment variable '{api_key_env}' not set for LLM.")

             try:
                 # Configure API key (might be redundant if embedder already did, but safe)
                 genai.configure(api_key=api_key)
                 # Get the specific model path from config
                 model_path = llm_config.get("path")
                 if not model_path: raise ValueError(f"Model path not defined for LLM '{self.llm_model_name}' in config.")
                 # --- Instantiate the GenerativeModel ---
                 self.llm_model = genai.GenerativeModel(model_path)
                 self._llm_initialized = True
                 logger.info(f"LLM '{model_path}' initialized successfully.")
             except Exception as e:
                  logger.error(f"Failed to initialize LLM '{self.llm_model_name}': {e}", exc_info=True)
                  raise ConnectionError(f"Failed to initialize LLM: {e}") from e


    def _get_query_embedding(self, query_text: str) -> np.ndarray:
        """Generates embedding for the user query."""
        self._initialize_embedder() # Ensure embedder is ready
        logger.info("Generating query embedding...")
        # ... (embedding logic remains the same as before) ...
        try:
            # Use embed_batch, assuming it handles single strings and returns [[embedding]]
            query_embedding_2d = self.embedder.embed_batch([query_text])
            if query_embedding_2d is None or not isinstance(query_embedding_2d, np.ndarray) or query_embedding_2d.shape[0] != 1:
                 raise RuntimeError("Failed to generate valid query embedding.")
            logger.info("Query embedding generated.")
            return query_embedding_2d[0] # Return 1D vector
        except Exception as e: logger.error(f"Error generating query embedding: {e}", exc_info=True); raise RuntimeError("Failed to generate query embedding") from e

    def search_chunks(self, query_text: str, top_k: int = 10) -> List[ResultItem]:
        """Embeds query and searches DB for relevant chunks within the notes folder."""
        if not query_text: return []
        try:
            query_embedding = self._get_query_embedding(query_text)
            logger.info(f"Searching DB for top {top_k} chunks (folder constrained)...")
            results: List[ResultItem] = []
            with DbManager(self.db_path) as db_manager:
                raw_results = db_manager.find_similar_chunks(
                    query_embedding=query_embedding, k=top_k, notes_folder_path=self.notes_folder
                )
            for file_path_str, chunk_index, content, score in raw_results:
                 try: results.append(ResultItem(Path(file_path_str), chunk_index, content, score))
                 except Exception as path_e: logger.error(f"Failed to process result item path '{file_path_str}': {path_e}")
            logger.info(f"Found {len(results)} relevant chunks.")
            return results
        except (ValueError, RuntimeError, ConnectionError, sqlite3.Error) as e: logger.error(f"Chunk search failed: {e}", exc_info=True); return []
        except Exception as e: logger.error(f"Unexpected error during chunk search: {e}", exc_info=True); return []


    def _format_context_for_llm(self, chunks: List[ResultItem]) -> str:
        """Formats the retrieved chunks into a string for the LLM prompt."""
        context_str = "Context from Notes:\n\n"
        current_length = len(context_str)
        separator = "\n---\n"

        for item in chunks:
            # Format: FILE: [Relative Path] (Chunk [Index])\n[Content]
            try:
                # Attempt to make path relative to the notes folder for conciseness
                relative_path = item.file_path.relative_to(self.notes_folder)
            except ValueError:
                relative_path = item.file_path.name # Fallback to just filename

            chunk_header = f"FILE: {relative_path} (Chunk {item.chunk_index})"
            chunk_entry = f"{chunk_header}\n{item.content.strip()}"

            entry_length = len(chunk_entry) + len(separator)

            # Check if adding this chunk exceeds the max length
            if current_length + entry_length > MAX_CONTEXT_LENGTH:
                logger.warning(f"Context length limit ({MAX_CONTEXT_LENGTH} chars) reached. Truncating context for LLM.")
                break # Stop adding chunks

            context_str += chunk_entry + separator
            current_length += entry_length

        return context_str.strip() # Remove trailing separator
    def create_llm_prompt(self, query_text: str, retrieved_chunks: List[ResultItem]) -> str:
        """Creates the LLM prompt string without sending it."""
        if not retrieved_chunks:
            return "" # Or return an informative message

        formatted_context = self._format_context_for_llm(retrieved_chunks)
        prompt = f"""You are a helpful assistant trying to answer the best user query, with your own knwoledge and the provided knowledge from the user own notes.
        User Query:
        {query_text}
        Context from Notes:
        {formatted_context}
        You can use this provided knowledge from personal notes to answer the best the question. Also when using this specific knowledge in your answer please cite the source it comes from""" # Use same template as generate_llm_answer
        return prompt
    
    def generate_llm_answer(self, query_text: str, retrieved_chunks: List[ResultItem]) -> str:
        """
        Generates an LLM answer by first creating a prompt from the query and chunks,
        and then sending that prompt to the LLM.
        """
        logger.info(f"Generating LLM answer directly for query: '{query_text[:50]}...' with {len(retrieved_chunks)} chunks.")

        # Step 1: Create the prompt using the existing method
        prompt = self.create_llm_prompt(query_text, retrieved_chunks)

        # Step 2: Check if a prompt was successfully created
        if not prompt:
            # This case primarily happens if retrieved_chunks is empty.
            # The QueryWorker already checks for empty results before calling this,
            # but this adds robustness.
            logger.warning("generate_llm_answer: Prompt creation failed or resulted in empty prompt (no context?). Returning standard message.")
            return "No relevant context found in notes to generate an answer based on."

        # Step 3: Call the existing method that handles the actual LLM generation
        logger.debug("generate_llm_answer: Calling generate_llm_answer_from_prompt with the automatically created prompt.")
        try:
            # This reuses the LLM initialization, API call, safety settings, and error handling logic
            llm_answer = self.generate_llm_answer_from_prompt(prompt)
            logger.info("generate_llm_answer: Successfully generated answer using created prompt.")
            return llm_answer
        except Exception as e:
            # This catch block is a safeguard; generate_llm_answer_from_prompt should ideally handle
            # specific LLM errors and return error strings already.
            logger.error(f"generate_llm_answer: Unexpected error during call to generate_llm_answer_from_prompt: {e}", exc_info=True)
            return f"Error: An unexpected issue occurred during LLM answer generation: {e}"
    def generate_llm_answer_from_prompt(self, final_prompt: str) -> str:
         logger.info("Entering generate_llm_answer_from_prompt...") # <-- ADD
         try:
             self._initialize_llm() # Ensure LLM is ready
             if not self.llm_model: return "Error: LLM not initialized."
             logger.debug(f"Final Prompt (first 500 chars): {final_prompt[:500]}") # <-- ADD

             safety_settings = [ # Define or fetch safety settings
                 {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                 # ... other settings ...
             ]
             logger.debug("Calling llm_model.generate_content...") # <-- ADD
             response = self.llm_model.generate_content(final_prompt, safety_settings=safety_settings)
             logger.debug(f"LLM Response received. Block reason: {response.prompt_feedback.block_reason if response.prompt_feedback else 'None'}") # <-- ADD

             if not response.parts:
                 # ... (Existing block reason handling) ...
                  block_reason = response.prompt_feedback.block_reason if response.prompt_feedback else "Unknown"
                  error_msg = f"LLM response was blocked. Reason: {block_reason}"
                  logger.warning(error_msg); return f"Error: LLM response blocked ({block_reason})."

             llm_answer = response.text
             logger.info("LLM answer processed successfully from edited prompt.")
             return llm_answer.strip()

         # ... (Existing exception handling remains the same) ...
         except google_exceptions.ResourceExhausted as e: logger.error(...); return "Error: Rate limit..."
         except Exception as e: logger.error(...); return f"Error generating answer: {e}"