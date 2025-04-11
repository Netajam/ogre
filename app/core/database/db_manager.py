# app/core/database/db_manager.py
import sqlite3
import time
from pathlib import Path
import os
import json # Keep for potential future use, though not strictly needed for MATCH query params
import sys
import importlib.util
from typing import Optional, List, Tuple, Dict, Any
import numpy as np

try:
    import sqlite_vec
    SQLITE_VEC_AVAILABLE = True
except ImportError:
    sqlite_vec = None
    SQLITE_VEC_AVAILABLE = False
    print("--------------------------------------------------------------------")
    print("WARNING: sqlite-vec library not found. Database operations disabled.")
    print("         Install it using: pip install sqlite-vec")
    print("--------------------------------------------------------------------")

# Import schema definitions and the application logger
from . import schema
# --- Import the serialize_vector function ---
# Ensure utils.py exists in the same directory or adjust path
try:
    from .utils import serialize_vector
except ImportError:
    # Fallback if utils.py doesn't exist or causes issues
    logger.error("Could not import serialize_vector from .utils. Vector serialization will fail.")
    # Define a dummy function to avoid NameError, but log error
    def serialize_vector(vector: np.ndarray) -> bytes:
        logger.error("serialize_vector function not available!")
        raise NotImplementedError("serialize_vector is missing")
# ---
try:
    from app.utils.logging_setup import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO)
    logger.warning("Using fallback logger for DbManager.")


class DbManager:
    """
    Manages interactions with the SQLite database, using the vec0 virtual table
    approach provided by sqlite-vec version 0.1.x.
    """

    def __init__(self, db_path: Path):
        """Initializes the DbManager and establishes a connection."""
        if not SQLITE_VEC_AVAILABLE:
            raise ImportError("sqlite-vec library is required but not installed.")

        self.db_path = db_path
        self.conn: Optional[sqlite3.Connection] = None
        self.cursor: Optional[sqlite3.Cursor] = None

        try:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            logger.debug(f"Ensured database directory exists: {self.db_path.parent}")
        except OSError as e:
            logger.error(f"Failed to create database directory {self.db_path.parent}: {e}")
            raise ConnectionError(f"Could not create database directory: {e}") from e

        self._connect()

    def _connect(self) -> None:
        """Establishes a connection and loads the sqlite-vec extension."""
        try:
            logger.debug(f"Attempting to connect to database using sqlite3: {self.db_path}")
            self.conn = sqlite3.connect(str(self.db_path))

            # Load sqlite-vec extension using its helper
            try:
                self.conn.enable_load_extension(True)
                logger.debug("Enabled loading of SQLite extensions.")
                sqlite_vec.load(self.conn) # Use the library's load function
                logger.info("sqlite-vec extension loaded successfully via sqlite_vec.load().")
            except AttributeError:
                 logger.warning("sqlite3.Connection has no 'enable_load_extension'. Extension loading might fail.")
                 # Continue, maybe load() works without it on some platforms
            except Exception as e:
                 logger.error(f"Failed to load sqlite-vec extension using sqlite_vec.load(): {e}", exc_info=True)
                 if self.conn: self.conn.close()
                 self.conn = None
                 raise ConnectionError("Failed to load required sqlite-vec extension.") from e

            self.cursor = self.conn.cursor()
            logger.info(f"Successfully connected to database: {self.db_path}")

        except sqlite3.Error as e:
            logger.error(f"Error connecting to database {self.db_path}: {e}", exc_info=True)
            self.conn, self.cursor = None, None
            raise ConnectionError(f"Failed to connect to database {self.db_path}: {e}") from e
        except Exception as e:
            logger.error(f"Unexpected error connecting to database {self.db_path}: {e}", exc_info=True)
            self.conn, self.cursor = None, None
            raise ConnectionError(f"Unexpected error connecting to database: {e}") from e

    def initialize_db(self, embedding_dim: int) -> None:
        """Creates the main documents table and the vec0 virtual table."""
        if not self.cursor or not self.conn: raise ConnectionError("Database not connected")
        if not isinstance(embedding_dim, int) or embedding_dim <= 0: raise ValueError("Embedding dimension must be positive.")

        try:
            logger.info(f"Initializing database schema (vec0) with embedding dimension: {embedding_dim}")
            self.cursor.execute("BEGIN TRANSACTION;")

            create_docs_sql = schema.get_create_documents_table_sql()
            logger.debug("Executing SQL: Create documents table")
            self.cursor.execute(create_docs_sql)

            create_vec_sql = schema.get_create_virtual_table_sql(embedding_dim)
            logger.debug("Executing SQL: Create vec_documents virtual table")
            self.cursor.execute(create_vec_sql)

            create_idx_sql = schema.get_create_file_path_index_sql()
            logger.debug("Executing SQL: Create file path index")
            self.cursor.execute(create_idx_sql)

            self.conn.commit()
            logger.info("Database schema (vec0) initialization successful.")

        except sqlite3.Error as e:
            logger.error(f"Error initializing database schema (vec0): {e}", exc_info=True)
            try: self.conn.rollback()
            except Exception as rb_err: logger.error(f"Rollback failed after schema init error: {rb_err}")
            raise

    def add_chunks_batch(self, chunks_data: List[Tuple[str, int, str, np.ndarray, float]]) -> None:
        """Adds chunks and their serialized vectors to the database via iterated inserts."""
        if not self.cursor or not self.conn: raise ConnectionError("Database not connected")
        if not chunks_data: logger.info("No chunks provided to add_chunks_batch."); return

        insert_doc_sql = schema.get_insert_document_sql()
        insert_vec_sql = schema.get_insert_vector_sql()
        added_doc_count = 0
        added_vec_count = 0
        start_time = time.time()

        try:
            logger.debug(f"Starting transaction to insert {len(chunks_data)} chunks/vectors.")
            self.cursor.execute("BEGIN TRANSACTION;")

            for file_path, chunk_index, content, embedding, last_modified in chunks_data:
                # Combine checks for efficiency
                if not file_path or not isinstance(chunk_index, int) or not content or not isinstance(embedding, np.ndarray):
                     logger.warning(f"Skipping invalid chunk data for file {file_path}, index {chunk_index}.")
                     continue
                try:
                    doc_params = (file_path, chunk_index, content, last_modified)
                    self.cursor.execute(insert_doc_sql, doc_params)
                    doc_id = self.cursor.lastrowid
                    if doc_id is None: # Check if insert failed silently (unlikely with PK)
                        logger.error(f"Failed to get lastrowid after inserting doc chunk {chunk_index} for {file_path}.")
                        continue # Skip vector insert if doc insert failed

                    added_doc_count += 1
                    serialized_embedding = serialize_vector(embedding) # Can raise ValueError
                    vec_params = (doc_id, serialized_embedding)
                    self.cursor.execute(insert_vec_sql, vec_params)
                    added_vec_count += 1

                except sqlite3.Error as insert_err:
                     logger.error(f"DB error inserting chunk {chunk_index} for {file_path}: {insert_err}", exc_info=True)
                     # Consider rolling back or just logging and continuing
                     # For now, log and continue to allow partial success
                except ValueError as ser_err:
                     logger.error(f"Serialization error for chunk {chunk_index} file {file_path}: {ser_err}", exc_info=True)
                     # Skip this chunk if serialization fails

            self.conn.commit() # Commit successful inserts
            duration = time.time() - start_time
            logger.info(f"Insert transaction finished. Added {added_doc_count} docs, {added_vec_count} vectors in {duration:.3f}s.")
            if added_doc_count != added_vec_count:
                 logger.warning(f"Mismatch added docs ({added_doc_count}) vs vectors ({added_vec_count}). Check logs.")

        except sqlite3.Error as e: # Error starting transaction or committing
            logger.error(f"Transaction error during batch insert: {e}", exc_info=True)
            try: self.conn.rollback()
            except Exception as rb_err: logger.error(f"Rollback failed after transaction error: {rb_err}")
            raise # Re-raise original error

    def delete_chunks_for_file(self, file_path: str) -> None:
        """Deletes document records and associated vectors for a specific file path."""
        if not self.cursor or not self.conn: raise ConnectionError("Database not connected")
        if not file_path: logger.warning("Attempted delete with empty file path."); return

        select_ids_sql = schema.get_select_ids_for_file_path_sql()
        delete_docs_sql = schema.get_delete_documents_sql()
        deleted_doc_count = 0
        deleted_vec_count = 0

        try:
            logger.debug(f"Starting transaction to delete records for file path: {file_path}")
            self.cursor.execute("BEGIN TRANSACTION;")

            self.cursor.execute(select_ids_sql, (file_path,))
            ids_to_delete = [row[0] for row in self.cursor.fetchall()]

            if ids_to_delete:
                logger.debug(f"Found {len(ids_to_delete)} document IDs to delete.")
                ids_placeholder = "(" + ",".join("?" * len(ids_to_delete)) + ")"
                delete_vec_sql = schema.get_delete_vectors_sql(ids_placeholder)
                logger.debug(f"Deleting {len(ids_to_delete)} vectors...")
                self.cursor.execute(delete_vec_sql, ids_to_delete)
                deleted_vec_count = self.cursor.rowcount

                logger.debug(f"Deleting documents from main table for file: {file_path}")
                self.cursor.execute(delete_docs_sql, (file_path,))
                deleted_doc_count = self.cursor.rowcount

                self.conn.commit()
                logger.info(f"Deleted data for file: {file_path} ({deleted_doc_count} docs, {deleted_vec_count} vectors).")
            else:
                logger.info(f"No documents found for file path: {file_path}. Nothing to delete.")
                self.conn.rollback() # Rollback empty transaction

        except sqlite3.Error as e:
            logger.error(f"Error deleting chunks for file {file_path}: {e}", exc_info=True)
            try: self.conn.rollback()
            except Exception as rb_err: logger.error(f"Rollback failed after delete error: {rb_err}")
            raise

    def get_indexed_files(self) -> Dict[str, float]:
        """Retrieves indexed file paths and their last modified times."""
        if not self.cursor or not self.conn: logger.error("DB not connected."); return {}
        query_sql = schema.get_indexed_files_sql()
        try:
            logger.debug("Querying for indexed files...")
            self.cursor.execute(query_sql)
            results = self.cursor.fetchall()
            indexed_files = {row[0]: row[1] for row in results} if results else {}
            logger.info(f"Retrieved status for {len(indexed_files)} indexed files.")
            return indexed_files
        except sqlite3.Error as e:
            if "no such table" in str(e).lower(): logger.warning(f"Table '{schema.DOCUMENTS_TABLE}' not found."); return {}
            else: logger.error(f"Error getting indexed files: {e}", exc_info=True); return {}

    def find_similar_chunks(
        self,
        query_embedding: np.ndarray,
        k: int = 5,
        notes_folder_path: Optional[str] = None
    ) -> List[Tuple[str, int, str, float]]:
        """Finds the k most similar chunks using the MATCH operator."""
        if not self.cursor or not self.conn: raise ConnectionError("Database not connected")
        if query_embedding is None or not isinstance(query_embedding, np.ndarray) or query_embedding.ndim != 1: raise ValueError("Query embedding must be a 1D numpy array.")
        if not isinstance(k, int) or k <= 0: raise ValueError("k must be a positive integer.")

        try:
            serialized_query_embedding = serialize_vector(query_embedding)
        except ValueError as e: logger.error(f"Failed to serialize query: {e}"); raise ValueError("Invalid query embedding") from e

        filter_by_folder = bool(notes_folder_path)
        query_sql = schema.get_vector_match_sql(k, filter_by_folder)
        params: List[Any] = [serialized_query_embedding]

        if filter_by_folder:
            folder_prefix = os.path.join(notes_folder_path, '')
            params.append(folder_prefix + '%')
            logger.debug(f"Executing MATCH query (k={k}, filter='{notes_folder_path}')")
        else:
            logger.debug(f"Executing MATCH query (k={k}, no filter)")

        try:
            start_time = time.time()
            self.cursor.execute(query_sql, tuple(params))
            results = self.cursor.fetchall()
            duration = time.time() - start_time

            if results: logger.info(f"Found {len(results)} similar chunk(s) using MATCH in {duration:.3f}s.")
            else: logger.info("No similar chunks found matching criteria.")
            return results

        except sqlite3.Error as e:
            if "malformed MATCH operand" in str(e).lower(): logger.error("Malformed MATCH operand.", exc_info=True); raise ValueError("Invalid query vector") from e
            elif "no such table" in str(e).lower() and schema.VEC_DOCUMENTS_TABLE in str(e).lower(): logger.warning(f"Virtual table '{schema.VEC_DOCUMENTS_TABLE}' missing.", exc_info=True); return []
            else: logger.error(f"Error finding similar chunks using MATCH: {e}", exc_info=True); raise

    def close(self) -> None:
        """Closes the database connection safely."""
        if self.conn:
            try:
                logger.debug("Attempting commit before closing DB.")
                self.conn.commit()
                logger.debug("Closing database connection.")
                self.conn.close()
                logger.info(f"Database connection closed: {self.db_path}")
            except sqlite3.Error as e: logger.error(f"Error during commit/close: {e}", exc_info=True)
            finally: self.conn, self.cursor = None, None
        else: logger.debug("Close called but DB not connected.")

    def __enter__(self):
        """Enter runtime context."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit runtime context, ensuring connection closure."""
        logger.debug("Exiting DbManager context, ensuring connection is closed.")
        self.close()