# app/core/database/db_manager.py
import sqlite3
import time
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Any
import numpy as np

try:
    # We still need to import sqlite_vec to make its functions available
    import sqlite_vec
    SQLITE_VEC_AVAILABLE = True
except ImportError:
    sqlite_vec = None # Keep this structure for checks elsewhere if needed
    SQLITE_VEC_AVAILABLE = False
    print("--------------------------------------------------------------------")
    print("WARNING: sqlite-vec library not found. Vector operations disabled.")
    print("         Install it using: pip install sqlite-vec")
    print("--------------------------------------------------------------------")

# Import schema definitions and the application logger
from . import schema
try:
    from app.core.utils.logging_setup import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO)
    logger.warning("Using fallback logger for DbManager.")


class DbManager:
    """
    Manages interactions with the SQLite database, utilizing the sqlite-vec
    extension for vector storage and search.
    """

    def __init__(self, db_path: Path):
        """
        Initializes the DbManager and establishes a connection to the database.

        Args:
            db_path: The Path object pointing to the SQLite database file.

        Raises:
            ImportError: If the sqlite-vec library is not installed.
            ConnectionError: If the database directory cannot be created or
                             if the database connection fails.
        """
        # Check if sqlite_vec was successfully imported (needed for vector functions)
        if not SQLITE_VEC_AVAILABLE:
            raise ImportError("sqlite-vec library is required but not installed.")

        self.db_path = db_path
        # --- Update Type Hints to use sqlite3 ---
        self.conn: Optional[sqlite3.Connection] = None 
        self.cursor: Optional[sqlite3.Cursor] = None  
        # -----------------------------------------

        # Ensure the directory exists before trying to connect
        try:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            logger.debug(f"Ensured database directory exists: {self.db_path.parent}")
        except OSError as e:
            logger.error(f"Failed to create database directory {self.db_path.parent}: {e}")
            raise ConnectionError(f"Could not create database directory: {e}") from e

        # Establish connection during initialization
        self._connect()

    def _connect(self) -> None:
        """Establishes a connection to the SQLite database using the standard sqlite3 module."""
        try:
            logger.debug(f"Attempting to connect to database using sqlite3: {self.db_path}")
            # --- Use sqlite3.connect ---
            self.conn = sqlite3.connect(str(self.db_path))
            # ---------------------------
            try:
                self.conn.enable_load_extension(True)
                logger.debug("Enabled loading of SQLite extensions.")
            except AttributeError:
                 logger.warning("sqlite3.Connection object has no attribute 'enable_load_extension'. This might be an issue on some Python versions/builds.")
            except sqlite3.Error as ext_err:
                 logger.warning(f"Could not enable extension loading: {ext_err}")


            self.cursor = self.conn.cursor()
            logger.info(f"Successfully connected to database: {self.db_path}")

        except sqlite3.Error as e: 
            logger.error(f"Error connecting to database {self.db_path}: {e}", exc_info=True)
            self.conn = None
            self.cursor = None
            raise ConnectionError(f"Failed to connect to database {self.db_path}: {e}") from e
        except Exception as e: # Catch any other unexpected errors during connection
            logger.error(f"Unexpected error connecting to database {self.db_path}: {e}", exc_info=True)
            self.conn = None
            self.cursor = None
            raise ConnectionError(f"Unexpected error connecting to database: {e}") from e


    def initialize_db(self, embedding_dim: int) -> None:
        """
        Creates necessary tables and indexes if they don't already exist.
        Requires the embedding dimension to correctly define the VECTOR column.
        """
        if not self.cursor or not self.conn:
            logger.error("Database connection not established. Cannot initialize.")
            raise ConnectionError("Database not connected")
        if not isinstance(embedding_dim, int) or embedding_dim <= 0:
             logger.error(f"Invalid embedding dimension provided for DB initialization: {embedding_dim}")
             raise ValueError("Embedding dimension must be a positive integer.")

        try:
            logger.info(f"Initializing database schema with embedding dimension: {embedding_dim}")

            create_table_sql = schema.get_create_table_sql(embedding_dim)
            logger.debug(f"Executing SQL: {create_table_sql}")
            self.cursor.execute(create_table_sql)
            logger.debug("Documents table created or already exists.")

            create_fp_index_sql = schema.get_create_file_path_index_sql()
            logger.debug(f"Executing SQL: {create_fp_index_sql}")
            self.cursor.execute(create_fp_index_sql)
            logger.debug("File path index created or already exists.")

            logger.info("Vector index will be created automatically by sqlite-vec upon first insertion if needed.")

            self.conn.commit()
            logger.info("Database schema initialization successful (or tables/indexes already existed).")

        except sqlite3.Error as e:
            logger.error(f"Error initializing database schema: {e}", exc_info=True)
            try:
                self.conn.rollback()
            except Exception as rb_err:
                 logger.error(f"Error during rollback after schema initialization failure: {rb_err}")
            raise


    def add_chunks_batch(self, chunks_data: List[Tuple[str, int, str, np.ndarray, float]]) -> None:
        """
        Adds multiple document chunks (including embeddings) to the database in a single transaction.
        """
        if not self.cursor or not self.conn:
            logger.error("Database connection not established. Cannot add chunks.")
            raise ConnectionError("Database not connected")
        if not chunks_data:
            logger.info("No chunks provided to add_chunks_batch. Nothing to add.")
            return

        insert_sql = schema.get_insert_chunk_sql()
        try:
            start_time = time.time()
            logger.debug(f"Executing batch insert for {len(chunks_data)} chunks...")
            self.cursor.executemany(insert_sql, chunks_data) # Try direct insertion first
            self.conn.commit()
            duration = time.time() - start_time
            logger.info(f"Successfully added batch of {len(chunks_data)} chunks in {duration:.3f}s.")
        except sqlite3.Error as e:
            logger.error(f"Error adding chunks batch: {e}", exc_info=True)
            try:
                self.conn.rollback()
            except Exception as rb_err:
                 logger.error(f"Error during rollback after chunk insertion failure: {rb_err}")
            raise


    def delete_chunks_for_file(self, file_path: str) -> None:
        """
        Deletes all chunks associated with a specific file path from the database.
        """
        if not self.cursor or not self.conn:
            logger.error("Database connection not established. Cannot delete chunks.")
            raise ConnectionError("Database not connected")
        if not file_path:
             logger.warning("Attempted to delete chunks for an empty file path.")
             return

        delete_sql = schema.get_delete_chunks_sql()
        try:
            logger.debug(f"Executing delete for file path: {file_path}")
            self.cursor.execute(delete_sql, (file_path,))
            deleted_count = self.cursor.rowcount
            self.conn.commit()
            if deleted_count > 0:
                logger.info(f"Successfully deleted {deleted_count} chunk(s) for file: {file_path}")
            else:
                logger.info(f"No chunks found in database to delete for file: {file_path}")
        except sqlite3.Error as e:
            logger.error(f"Error deleting chunks for file {file_path}: {e}", exc_info=True)
            try:
                self.conn.rollback()
            except Exception as rb_err:
                 logger.error(f"Error during rollback after chunk deletion failure: {rb_err}")
            raise


    def get_indexed_files(self) -> Dict[str, float]:
        """
        Retrieves a dictionary mapping indexed file paths to their last modified timestamps.
        """
        # This method remains the same
        if not self.cursor or not self.conn:
            logger.error("Database connection not established. Cannot get indexed files.")
            return {}

        query_sql = schema.get_indexed_files_sql()
        indexed_files: Dict[str, float] = {}
        try:
            logger.debug("Querying for indexed files and their last modified times...")
            self.cursor.execute(query_sql)
            results = self.cursor.fetchall()

            if results:
                indexed_files = {row[0]: row[1] for row in results}
                logger.info(f"Retrieved status for {len(indexed_files)} indexed files.")
            else:
                 logger.info("No indexed files found in the database.")

            return indexed_files

        except sqlite3.Error as e:
            if "no such table" in str(e).lower():
                logger.warning(f"Documents table '{schema.DOCUMENTS_TABLE}' not found. Assuming empty index.")
                return {}
            else:
                logger.error(f"Error getting indexed files from database: {e}", exc_info=True)
                return {}


    def find_similar_chunks(self, query_embedding: np.ndarray, k: int = 5) -> List[Tuple[str, int, str, float]]:
        """
        Finds the k most similar document chunks to a given query embedding using vector search.
        """
        if not self.cursor or not self.conn:
            logger.error("Database connection not established. Cannot perform vector search.")
            raise ConnectionError("Database not connected")
        if query_embedding is None or not isinstance(query_embedding, np.ndarray) or query_embedding.ndim != 1:
             logger.error(f"Invalid query embedding provided for search. Shape: {query_embedding.shape if query_embedding is not None else 'None'}")
             raise ValueError("Query embedding must be a 1D numpy array.")
        if not isinstance(k, int) or k <= 0:
             logger.error(f"Invalid value for k (top N results): {k}. Must be a positive integer.")
             raise ValueError("k must be a positive integer.")

        query_embedding = query_embedding.astype(np.float32)

        # --- Check if vector_search function exists (runtime check) ---
        # This helps diagnose if the extension isn't loaded correctly
        try:
             self.cursor.execute("SELECT vector_search(?, ?, ?, ?)", (schema.DOCUMENTS_TABLE, schema.COL_EMBEDDING, query_embedding, 1))
        except sqlite3.OperationalError as e:
             if "no such function: vector_search" in str(e).lower():
                 logger.error("The 'vector_search' function is not available in SQLite. "
                              "Ensure the sqlite-vec library is correctly installed and loadable.", exc_info=True)
                 raise RuntimeError("sqlite-vec extension function 'vector_search' not found.") from e
             else:
                 logger.warning(f"Operational error preparing vector search (may be expected if index is empty): {e}")
        except Exception as e:
            logger.error(f"Unexpected error checking for vector_search function: {e}", exc_info=True)

        # Construct the vector search query using sqlite-vec's function
        query = f"""
            SELECT
                {schema.COL_FILE_PATH},
                {schema.COL_CHUNK_INDEX},
                {schema.COL_CONTENT},
                distance
            FROM vector_search(?, ?, ?, ?)
        """

        params = (
            schema.DOCUMENTS_TABLE,
            schema.COL_EMBEDDING,
            query_embedding,
            k
        )

        try:
            start_time = time.time()
            logger.debug(f"Executing vector search for top {k} similar chunks...")
            self.cursor.execute(query, params)
            results = self.cursor.fetchall()
            duration = time.time() - start_time

            if results:
                logger.info(f"Found {len(results)} similar chunk(s) in {duration:.3f}s.")
                return results
            else:
                logger.info("No similar chunks found matching the query.")
                return []

        except sqlite3.Error as e:
            # Handle specific errors related to vector search again here
            if "no such function: vector_search" in str(e).lower():
                 logger.error("Vector search function not available during query execution.", exc_info=True)
                 raise RuntimeError("sqlite-vec extension function 'vector_search' not found.") from e
            elif "no such virtual table" in str(e).lower() or "no such table: vec_" in str(e).lower(): # Check for internal index table name too
                 logger.warning("Vector index virtual table not found during query. Has indexing run successfully?", exc_info=True)
                 return []
            elif "dimension mismatch" in str(e).lower():
                 logger.error(f"Query vector dimension mismatch with index. Query shape: {query_embedding.shape}", exc_info=True)
                 raise ValueError("Query embedding dimension mismatch.") from e
            else:
                logger.error(f"Error finding similar chunks: {e}", exc_info=True)
                raise


    def close(self) -> None:
        """Closes the database connection safely."""
        if self.conn:
            try:
                logger.debug("Attempting to commit any pending changes before closing DB connection.")
                self.conn.commit()
                logger.debug("Closing database connection.")
                self.conn.close()
                logger.info(f"Database connection closed: {self.db_path}")
            except sqlite3.Error as e:
                 logger.error(f"Error during commit/close of database connection: {e}", exc_info=True)
            finally:
                 self.conn = None
                 self.cursor = None
        else:
             logger.debug("Close called but database connection was already closed or not established.")


    def __enter__(self):
        """Enter the runtime context related to this object (for 'with' statement)."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the runtime context related to this object (for 'with' statement)."""
        logger.debug("Exiting DbManager context, ensuring connection is closed.")
        self.close()