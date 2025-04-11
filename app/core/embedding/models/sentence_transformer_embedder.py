# app/core/embedding/models/sentence_transformer_embedder.py
from typing import List, Union
import numpy as np
from sentence_transformers import SentenceTransformer

try:
    from app.core.utils.logging_setup import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO)


class SentenceTransformerEmbedder:
    """Handles embedding using Sentence Transformer models."""

    def __init__(self, model_name_or_path: str):
        self.model_name = model_name_or_path
        self.model: SentenceTransformer = self._load_model() 
        self._dimension: int = self._get_model_dimension()
        if self.model:
            logger.info(f"Initialized SentenceTransformer model: {self.model_name} (Dim: {self._dimension})")

    def _load_model(self) -> SentenceTransformer:
        """Loads the Sentence Transformer model."""
        try:
            #TD:  Consider adding device='cpu' if GPU issues arise or if you want to force CPU
            # model = SentenceTransformer(self.model_name, device='cpu')
            model = SentenceTransformer(self.model_name)
            logger.info(f"Successfully loaded SentenceTransformer model: {self.model_name}")
            return model
        except Exception as e:
            logger.error(f"Failed to load SentenceTransformer model '{self.model_name}': {e}", exc_info=True)
            raise ValueError(f"Could not load embedding model: {self.model_name}") from e

    def _get_model_dimension(self) -> int:
        """Gets the embedding dimension of the loaded model."""
        if not self.model:
             raise ValueError("Model not loaded, cannot determine dimension.")
        try:
            dim = self.model.get_sentence_embedding_dimension()
            if dim is None or dim <= 0:
                 logger.error(f"Could not determine embedding dimension for {self.model_name}. get_sentence_embedding_dimension() returned {dim}")
                 raise ValueError(f"Invalid or indeterminable dimension ({dim}) for model {self.model_name}")
            return dim
        except Exception as e:
            logger.error(f"Error getting embedding dimension for {self.model_name}: {e}", exc_info=True)
            raise ValueError(f"Could not get dimension for model {self.model_name}") from e


    @property
    def dimension(self) -> int:
        """Returns the embedding dimension."""
        if self._dimension is None:
            logger.error("Dimension accessed before initialization.")
            raise ValueError("Dimension not initialized.")
        return self._dimension

    def embed(self, texts: Union[str, List[str]]) -> Union[np.ndarray, List[np.ndarray]]:
        """
        Generates embeddings for the given text(s).
        Returns a single numpy array if input is str, or a list of numpy arrays if input is list.
        DEPRECATED in favor of embed_batch for consistency? Or keep for single text convenience?
        Let's align with embed_batch and always return a np.ndarray (potentially 2D).
        """
        if not self.model:
            raise ValueError("Embedding model is not loaded.")
        try:
            embeddings = self.model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
            # Ensure float32 for consistency, especially for storage/comparison
            return embeddings.astype(np.float32)
        except Exception as e:
            logger.error(f"Error generating embeddings with {self.model_name}: {e}", exc_info=True)
            raise RuntimeError(f"Failed to generate embeddings for input: {str(texts)[:100]}...") from e

    def embed_batch(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
         """
         Generates embeddings for a list of texts in batches.
         Returns a 2D numpy array where each row is an embedding.
         """
         if not isinstance(texts, list):
             logger.warning("Input to embed_batch was not a list, converting.")
             texts = [str(texts)]

         if not texts:
             logger.warning("embed_batch called with empty list.")
             return np.empty((0, self.dimension), dtype=np.float32)

         if not self.model:
             raise ValueError("Embedding model is not loaded.")

         try:
             embeddings = self.model.encode(
                 texts,
                 batch_size=batch_size,
                 show_progress_bar=False,
                 convert_to_numpy=True,
                 normalize_embeddings=True 
             )
             # Ensure float32 type
             return embeddings.astype(np.float32)
         except Exception as e:
             logger.error(f"Error generating batch embeddings with {self.model_name}: {e}", exc_info=True)
             raise RuntimeError("Failed to generate batch embeddings") from e