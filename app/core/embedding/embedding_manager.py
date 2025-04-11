# app/core/embedding/embedding_manager.py
import os # <-- Add import for os.getenv
from typing import Any, Optional, Dict
from app.config import AVAILABLE_EMBEDDING_MODELS, GEMINI_TASK_RETRIEVAL_DOCUMENT
from app.utils.logging_setup import logger
from .models.sentence_transformer_embedder import SentenceTransformerEmbedder
from .models.gemini_embedder import GeminiEmbedder

# --- Cache for loaded models ---
_embedder_cache: Dict[str, Any] = {}

class EmbeddingManager:
    """Manages the selection and loading of embedding models."""

    @staticmethod
    def get_embedder(model_display_name: str) -> Optional[Any]:
        """
        Gets an instance of the specified embedding model.
        Uses a cache to avoid reloading models. Handles API key retrieval.
        """
        if model_display_name in _embedder_cache:
            logger.debug(f"Returning cached embedder for: {model_display_name}")
            return _embedder_cache[model_display_name]

        if model_display_name not in AVAILABLE_EMBEDDING_MODELS:
            logger.error(f"Unknown embedding model selected: {model_display_name}")
            return None

        model_config = AVAILABLE_EMBEDDING_MODELS[model_display_name]
        model_type = model_config.get("type")
        model_path = model_config.get("path")

        embedder = None
        try:
            if model_type == "local":
                logger.info(f"Loading local SentenceTransformer model: {model_path}")
                embedder = SentenceTransformerEmbedder(model_path)

            elif model_type == "api":
                # --- Handle API Embedders (Currently only Gemini) ---
                api_key_env_var = model_config.get("api_key_env")
                if not api_key_env_var:
                     logger.error(f"API key environment variable name not configured for model: {model_display_name}")
                     return None

                api_key = os.getenv(api_key_env_var)
                if not api_key:
                    logger.error(f"API key environment variable '{api_key_env_var}' not found.")
                    # Optional: Prompt user or show error in UI later
                    return None # Fail initialization if key is missing

                # --- Instantiate Gemini Embedder ---
                logger.info(f"Initializing Gemini API model: {model_path}")
                try:
                    # Pass the task type appropriate for indexing here
                    embedder = GeminiEmbedder(
                        model_name=model_path,
                        api_key=api_key,
                        task_type=GEMINI_TASK_RETRIEVAL_DOCUMENT 
                    )
                except ImportError as e:
                     logger.error(f"Failed to initialize Gemini Embedder due to missing library: {e}")
                    
                     return None
                except (ConnectionError, ValueError) as e: 
                     logger.error(f"Failed to initialize Gemini Embedder: {e}")
                     return None

            else:
                logger.error(f"Unsupported model type '{model_type}' for model: {model_display_name}")

            if embedder:
                config_dim = model_config.get("dimension")
                if config_dim and embedder.dimension != config_dim:
                   logger.warning(f"Model config dimension ({config_dim}) mismatch for {model_display_name}. Using loaded/runtime dimension: {embedder.dimension}")

                _embedder_cache[model_display_name] = embedder
            return embedder

        except Exception as e:
            logger.error(f"Unexpected error getting embedder for '{model_display_name}': {e}", exc_info=True)
            return None

    @staticmethod
    def get_embedding_dimension(model_display_name: str) -> Optional[int]:
         """Gets the embedding dimension for a given model name from config."""
         if model_display_name in AVAILABLE_EMBEDDING_MODELS:
             # Always rely on config for dimension, as API models might not expose it easily
             dimension = AVAILABLE_EMBEDDING_MODELS[model_display_name].get("dimension")
             if dimension:
                 return dimension
             else:
                 logger.error(f"Dimension not specified in config for {model_display_name}.")
                 return None # Indicate failure

         logger.error(f"Cannot determine dimension for unknown model: {model_display_name}")
         return None

    @staticmethod
    def clear_cache():
        """Clears the embedder cache."""
        global _embedder_cache
        _embedder_cache = {}
        logger.info("Embedder cache cleared.")