# app/config.py
import os
from typing import Dict, Any, List

APP_NAME: str = "Ogre"
APP_VERSION: str = "0.1.0"
ORGANIZATION_NAME: str = "NetajamCorp"
# --- Settings Keys ---
SETTINGS_NOTES_FOLDER: str = "notes_folder"
SETTINGS_EMBEDDING_MODEL: str = "embedding_model"
SETTINGS_DB_FILENAME: str = ".note_query_index.sqlite"
SETTINGS_INDEXED_EXTENSIONS: str = "indexed_extensions" 

# --- Environment Variable for API Key ---
GEMINI_API_KEY_ENV_VAR: str = "GEMINI_API_KEY"
# --- Defaults ---
# Define default extensions list
DEFAULT_INDEXED_EXTENSIONS: List[str] = [".md", ".markdown"]
# --- Embedding Models Configuration ---
AVAILABLE_EMBEDDING_MODELS: Dict[str, Dict[str, Any]] = {
    "all-MiniLM-L6-v2 (Local)": {
        "type": "local",
        "path": "all-MiniLM-L6-v2",
        "dimension": 384
    },
    # --- Gemini Model ---
    "Gemini (API)": {
        "type": "api",
        "path": "models/embedding-001", 
        "dimension": 768, 
        "api_key_env": GEMINI_API_KEY_ENV_VAR 
    },

}

DEFAULT_EMBEDDING_MODEL: str = list(AVAILABLE_EMBEDDING_MODELS.keys())[0]

# --- Other Constants ---
MARKDOWN_EXTENSIONS: List[str] = [".md", ".markdown"]

# --- Task Types for Gemini Embeddings ---
GEMINI_TASK_RETRIEVAL_QUERY = "RETRIEVAL_QUERY" # For embedding user queries
GEMINI_TASK_RETRIEVAL_DOCUMENT = "RETRIEVAL_DOCUMENT" # For embedding notes/documents during indexing
GEMINI_TASK_SEMANTIC_SIMILARITY = "SEMANTIC_SIMILARITY" # General similarity
GEMINI_TASK_CLASSIFICATION = "CLASSIFICATION"
GEMINI_TASK_CLUSTERING = "CLUSTERING"