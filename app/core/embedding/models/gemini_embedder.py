# app/core/embedding/models/gemini_embedder.py
import os
import time
from typing import List, Optional
import numpy as np

try:
    # Attempt to import necessary Google libraries
    import google.generativeai as genai
    from google.generativeai import types as genai_types
    from google.api_core import exceptions as google_exceptions
except ImportError:
    # If imports fail, set variables to None to allow conditional checks later
    genai = None
    genai_types = None
    google_exceptions = None
    # Inform the user immediately if the library is missing
    print("--------------------------------------------------------------------------")
    print("WARNING: google-generativeai library not found. Gemini Embedder disabled.")
    print("         Install it using: pip install google-generativeai")
    print("--------------------------------------------------------------------------")

# Import the application's logger and configuration
# Adjust the import path if your logger setup is located differently
try:
    from app.utils.logging_setup import logger
except ImportError:
    # Fallback basic logger if the app's logger isn't available
    import logging
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO)
    logger.warning("Using fallback logger for GeminiEmbedder.")

from app import config # Import config for task type constants and model details

# --- Constants ---
# How many texts to send to the API in one call (adjust based on API limits/performance)
GEMINI_API_BATCH_SIZE = 100
# Delay between retries on rate limit errors (in seconds)
GEMINI_RETRY_DELAY = 5
# Maximum number of retries for rate limit errors before failing
GEMINI_MAX_RETRIES = 3

class GeminiEmbedder:
    """
    Handles embedding generation using the Google Gemini API (via google-generativeai).
    Requires the 'google-generativeai' library to be installed and a valid API key.
    """

    def __init__(self, model_name: str, api_key: str, task_type: str = config.GEMINI_TASK_RETRIEVAL_DOCUMENT):
        """
        Initializes the Gemini Embedder.

        Args:
            model_name: The specific Gemini embedding model name (e.g., "models/embedding-001").
            api_key: The Google AI API key.
            task_type: The task type for optimization (e.g., RETRIEVAL_DOCUMENT, RETRIEVAL_QUERY).
                       Defaults to RETRIEVAL_DOCUMENT, suitable for indexing documents.

        Raises:
            ImportError: If the 'google-generativeai' library is not installed.
            ValueError: If the API key is missing or the embedding dimension cannot be determined.
            ConnectionError: If the Gemini client fails to initialize/configure.
        """
        # --- Pre-check: Ensure the library was imported successfully ---
        if not genai:
             logger.error("Attempted to initialize GeminiEmbedder, but 'google-generativeai' library is not installed.")
             raise ImportError("google-generativeai library is required for GeminiEmbedder but not installed.")

        # --- Store configuration ---
        self.model_name = model_name
        self.api_key = api_key

        # --- Validate and store task type ---
        valid_task_types = [
            config.GEMINI_TASK_RETRIEVAL_DOCUMENT, config.GEMINI_TASK_RETRIEVAL_QUERY,
            config.GEMINI_TASK_SEMANTIC_SIMILARITY, config.GEMINI_TASK_CLASSIFICATION,
            config.GEMINI_TASK_CLUSTERING
            # Add other valid task types from genai.types if needed
        ]
        if task_type not in valid_task_types:
            logger.warning(f"Unknown task_type '{task_type}' provided. Using default '{config.GEMINI_TASK_RETRIEVAL_DOCUMENT}'.")
            self.task_type = config.GEMINI_TASK_RETRIEVAL_DOCUMENT
        else:
            self.task_type = task_type
            logger.debug(f"GeminiEmbedder using task_type: {self.task_type}")

        # --- Determine embedding dimension (must be done before initialization) ---
        try:
            self._dimension = self._get_model_dimension_from_config()
            logger.debug(f"Gemini model {self.model_name} dimension set to: {self._dimension}")
        except ValueError as e:
             logger.error(f"Failed to determine embedding dimension for {model_name}: {e}")
             raise # Re-raise the ValueError

        # --- Initialize the API client/configuration ---
        # Store the success/failure status of initialization
        self.client_initialized: bool = self._initialize_client()

        # --- Final check after initialization attempt ---
        if self.client_initialized:
            logger.info(f"Successfully initialized Gemini Embedder. Model: {self.model_name}, Task: {self.task_type}, Dim: {self._dimension}")
        else:
             # Error should have been logged within _initialize_client
             # Raise a ConnectionError to signal failure to the caller (e.g., EmbeddingManager)
             raise ConnectionError(f"Failed to initialize/configure Gemini client for model {self.model_name}. Check API key and network connectivity.")


    def _get_model_dimension_from_config(self) -> int:
        """
        Retrieves the embedding dimension from the central application configuration
        based on the provided model name.

        Returns:
            The embedding dimension as an integer.

        Raises:
            ValueError: If the model name is not found in the configuration or
                        if the dimension is not specified for the model.
        """
        for _display_name, model_details in config.AVAILABLE_EMBEDDING_MODELS.items():
            # Match based on the API model path and type 'api'
            if model_details.get("path") == self.model_name and model_details.get("type") == "api":
                dimension = model_details.get("dimension")
                if isinstance(dimension, int) and dimension > 0:
                    return dimension
                else:
                    logger.error(f"Invalid or missing dimension ({dimension}) specified in config for Gemini model: {self.model_name}")
                    raise ValueError(f"Invalid or missing embedding dimension in config for {self.model_name}")

        # If the loop completes without finding the model
        logger.error(f"Model name '{self.model_name}' not found in AVAILABLE_EMBEDDING_MODELS configuration.")
        raise ValueError(f"Embedding model '{self.model_name}' not found in application configuration.")


    def _initialize_client(self) -> bool:
        """
        Configures the Google Generative AI client using the provided API key.
        This function primarily uses `genai.configure` as the library often relies
        on module-level configuration for functions like `embed_content`.

        Returns:
            bool: True if configuration was successful, False otherwise.
        """
        if not self.api_key:
            logger.error("Gemini API key is missing. Cannot initialize client.")
            return False # Indicate failure: API key missing

        logger.debug(f"Configuring Gemini API for model: {self.model_name}")
        try:
            # Configure the API key globally for the genai library
            genai.configure(api_key=self.api_key)
            # Optional: Make a small test call to verify connectivity/key validity?
            # Could add cost/latency, maybe only do it if explicitly enabled.
            # e.g., try: genai.get_model(self.model_name) except Exception: ...
            logger.info("Gemini API configured successfully.")
            return True # Indicate success: Configuration successful

        except Exception as e:
            # Catch any exception during configuration
            logger.error(f"Failed to configure Gemini API: {e}", exc_info=True)
            return False # Indicate failure: Configuration failed

    @property
    def dimension(self) -> int:
        """Returns the embedding dimension configured for this model."""
        return self._dimension

    def embed_batch(self, texts: List[str], batch_size: int = GEMINI_API_BATCH_SIZE) -> np.ndarray:
        """
        Generates embeddings for a list of texts using the configured Gemini API model.
        Handles batching, rate limiting retries, and error reporting.

        Args:
            texts: A list of non-empty strings to embed.
            batch_size: The maximum number of texts to send in a single API request.
                        Defaults to GEMINI_API_BATCH_SIZE.

        Returns:
            np.ndarray: A 2D numpy array where each row is an embedding vector
                        (shape: [num_texts, embedding_dim]), dtype=np.float32.

        Raises:
            ConnectionError: If the client was not initialized successfully.
            ConnectionAbortedError: If API rate limits persist after retries.
            RuntimeError: For other API errors or mismatches in response.
        """
        # --- Pre-check: Ensure client initialization was successful ---
        if not self.client_initialized:
            # This should ideally not be reached if __init__ raises ConnectionError, but serves as a safeguard
            logger.error("Attempted to embed content, but Gemini client was not initialized.")
            raise ConnectionError("Gemini client not initialized. Cannot embed content.")

        # --- Input Validation ---
        if not isinstance(texts, list):
             logger.error(f"Input to embed_batch must be a list of strings, got {type(texts)}.")
             # Or convert? Converting might hide issues. Raising is clearer.
             raise TypeError("Input 'texts' must be a list of strings.")
        if not texts:
            logger.warning("embed_batch called with an empty list. Returning empty array.")
            # Return an empty numpy array with the correct shape (0 rows, N columns)
            return np.empty((0, self.dimension), dtype=np.float32)

        # Filter out any empty strings, as the API might reject them
        valid_texts = [text for text in texts if text and text.strip()]
        if len(valid_texts) != len(texts):
            logger.warning(f"Removed {len(texts) - len(valid_texts)} empty or whitespace-only strings from the input batch.")
            if not valid_texts:
                 logger.warning("Batch contains only empty strings after filtering. Returning empty array.")
                 return np.empty((0, self.dimension), dtype=np.float32)
            texts = valid_texts # Use the filtered list

        # --- Batch Processing Loop ---
        all_embeddings = []
        total_texts = len(texts)
        logger.info(f"Starting Gemini embedding for {total_texts} text(s) in batches of {batch_size}.")

        for i in range(0, total_texts, batch_size):
            batch = texts[i : min(i + batch_size, total_texts)] # Ensure the last batch doesn't go over
            if not batch: continue # Skip if somehow an empty batch is created

            current_retries = 0
            while current_retries <= GEMINI_MAX_RETRIES:
                try:
                    logger.debug(f"Sending batch {i//batch_size + 1}/{(total_texts + batch_size - 1)//batch_size} "
                                 f"({len(batch)} texts) to Gemini API (Model: {self.model_name}, Task: {self.task_type})")

                    # Call the Gemini API using the module-level function
                    result = genai.embed_content(
                        model=self.model_name,
                        content=batch,
                        task_type=self.task_type
                    )

                    # Extract embeddings from the result dictionary
                    batch_embeddings = result.get('embedding') # Use .get for safety

                    # --- Validate Response ---
                    if batch_embeddings is None:
                         logger.error(f"Gemini API response for batch did not contain 'embedding' key. Response: {result}")
                         raise RuntimeError("Invalid response structure from Gemini API (missing 'embedding').")
                    if not isinstance(batch_embeddings, list):
                         logger.error(f"Gemini API 'embedding' field is not a list. Type: {type(batch_embeddings)}")
                         raise RuntimeError("Invalid response structure from Gemini API ('embedding' not a list).")
                    if len(batch_embeddings) != len(batch):
                        logger.error(f"API returned {len(batch_embeddings)} embeddings for {len(batch)} texts in batch.")
                        # Decide how to handle: raise error, skip batch, return partial? Raising is safest.
                        raise RuntimeError("Gemini API returned mismatched number of embeddings for the batch.")

                    # Extend the main list with the successfully retrieved batch embeddings
                    all_embeddings.extend(batch_embeddings)
                    logger.debug(f"Received and processed {len(batch_embeddings)} embeddings for current batch.")
                    break # Exit retry loop on success, move to next batch

                except google_exceptions.ResourceExhausted as rate_limit_error:
                    # Handle specific rate limiting error
                    current_retries += 1
                    if current_retries > GEMINI_MAX_RETRIES:
                        logger.error(f"Gemini API rate limit exceeded after {GEMINI_MAX_RETRIES} retries. Last error: {rate_limit_error}")
                        # Raise a specific error indicating persistent rate limiting
                        raise ConnectionAbortedError(f"Gemini API rate limit hit too many times ({GEMINI_MAX_RETRIES} retries).") from rate_limit_error
                    wait_time = GEMINI_RETRY_DELAY * (2 ** (current_retries - 1)) # Exponential backoff
                    logger.warning(f"Gemini API rate limit hit (Attempt {current_retries}/{GEMINI_MAX_RETRIES}). Retrying in {wait_time:.2f}s...")
                    time.sleep(wait_time) # Wait before retrying

                except Exception as api_error:
                    # Handle other potential API errors (e.g., invalid argument, network issues)
                    logger.error(f"Error embedding batch {i//batch_size + 1} with Gemini API: {api_error}", exc_info=True)
                    # Include details about the batch if possible (careful with logging sensitive data)
                    # logger.debug(f"Failed batch content (first item): {batch[0][:100]}...")
                    # Raise a generic runtime error for other API failures
                    raise RuntimeError(f"Failed to generate Gemini embeddings for a batch due to API error: {api_error}") from api_error

            # If the inner while loop finished due to max retries, the error was raised, so no extra check needed here.

        # --- Final Conversion and Validation ---
        logger.info(f"Finished embedding {len(all_embeddings)} texts successfully.")
        try:
            # Convert the list of lists/vectors into a single 2D numpy array
            embedding_array = np.array(all_embeddings, dtype=np.float32)
        except ValueError as e:
             logger.error(f"Failed to convert list of embeddings to numpy array: {e}. Check embedding structure.", exc_info=True)
             # This might happen if embeddings have inconsistent lengths, which shouldn't occur with a single model
             raise RuntimeError("Inconsistent embedding structure received from API.") from e

        # Final sanity check on the output shape
        if embedding_array.ndim != 2 or embedding_array.shape[0] != len(all_embeddings) or embedding_array.shape[1] != self.dimension:
             logger.error(f"Final embedding array has unexpected shape: {embedding_array.shape}. Expected: ({len(all_embeddings)}, {self.dimension})")
             # This indicates a serious internal issue or API inconsistency
             raise RuntimeError("Final embedding array validation failed.")

        return embedding_array

    # Optional: Implement embed_query separately if needed for specific query task types
    # def embed_query(self, query: str) -> np.ndarray:
    #     """Embeds a single query string using the appropriate task type."""
    #     if not self.client_initialized:
    #         raise ConnectionError("Gemini client not initialized.")
    #     if not query or not query.strip():
    #          logger.warning("embed_query called with empty query.")
    #          # Return empty array or raise error? Depends on expected behavior.
    #          return np.empty((0, self.dimension), dtype=np.float32)
    #     try:
    #         logger.debug(f"Embedding query using task type: {config.GEMINI_TASK_RETRIEVAL_QUERY}")
    #         result = genai.embed_content(
    #             model=self.model_name,
    #             content=query,
    #             task_type=config.GEMINI_TASK_RETRIEVAL_QUERY # Use query-specific task type
    #         )
    #         embedding = result.get('embedding')
    #         if embedding is None or not isinstance(embedding, list):
    #             raise RuntimeError("Invalid response structure for query embedding.")
    #         # Gemini returns a list containing one embedding for a single input string
    #         return np.array(embedding, dtype=np.float32).reshape(1, -1) # Ensure 2D array [1, dim]

    #     except Exception as e:
    #         logger.error(f"Error embedding query with Gemini API: {e}", exc_info=True)
    #         raise RuntimeError("Failed to generate Gemini query embedding") from e