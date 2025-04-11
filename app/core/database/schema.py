# app/core/database/schema.py
from typing import Optional

# --- Main Data Table ---
DOCUMENTS_TABLE = "documents"
COL_ID = "id"                   # Primary Key
COL_FILE_PATH = "file_path"
COL_CHUNK_INDEX = "chunk_index"
COL_CONTENT = "content"
COL_LAST_MODIFIED = "last_modified"

# --- Vector Virtual Table ---
VEC_DOCUMENTS_TABLE = "vec_documents" # Name for the virtual table
VEC_COL_ID = COL_ID             # Links to documents.id
VEC_COL_EMBEDDING = "embedding" # Name of the vector column

# --- Index Names ---
FILE_PATH_INDEX_NAME = "idx_doc_file_path" # Index on file_path in main table


def get_create_documents_table_sql() -> str:
    """Generates SQL to create the main documents table (without vectors)."""
    return f"""
    CREATE TABLE IF NOT EXISTS {DOCUMENTS_TABLE} (
        {COL_ID} INTEGER PRIMARY KEY AUTOINCREMENT,
        {COL_FILE_PATH} TEXT NOT NULL,
        {COL_CHUNK_INDEX} INTEGER NOT NULL,
        {COL_CONTENT} TEXT NOT NULL,
        {COL_LAST_MODIFIED} REAL NOT NULL
    );
    """

def get_create_virtual_table_sql(embedding_dim: int) -> str:
    """Generates SQL to create the vec0 virtual table linked to the main table."""
    if embedding_dim <= 0:
        raise ValueError("Embedding dimension must be positive.")
    # Syntax based on sqlite-vec 0.1.6 example
    return f"""
    CREATE VIRTUAL TABLE IF NOT EXISTS {VEC_DOCUMENTS_TABLE} USING vec0(
        {VEC_COL_ID} INTEGER REFERENCES {DOCUMENTS_TABLE}({COL_ID}),
        {VEC_COL_EMBEDDING} FLOAT[{embedding_dim}]
    );
    """

def get_create_file_path_index_sql() -> str:
    """Generates SQL to create an index on the file path in the main table."""
    return f"""
    CREATE INDEX IF NOT EXISTS {FILE_PATH_INDEX_NAME} ON {DOCUMENTS_TABLE} ({COL_FILE_PATH});
    """

# --- Insertion SQL ---

def get_insert_document_sql() -> str:
    """Generates SQL to insert data into the main documents table."""
    return f"""
    INSERT INTO {DOCUMENTS_TABLE} (
        {COL_FILE_PATH}, {COL_CHUNK_INDEX}, {COL_CONTENT}, {COL_LAST_MODIFIED}
    ) VALUES (?, ?, ?, ?);
    """

def get_insert_vector_sql() -> str:
    """Generates SQL to insert a serialized vector into the virtual table."""
    return f"""
    INSERT INTO {VEC_DOCUMENTS_TABLE} (
        {VEC_COL_ID}, {VEC_COL_EMBEDDING}
    ) VALUES (?, ?);
    """

# --- Deletion SQL ---

def get_select_ids_for_file_path_sql() -> str:
    """Generates SQL to select IDs from the main table for a given file path."""
    return f"SELECT {COL_ID} FROM {DOCUMENTS_TABLE} WHERE {COL_FILE_PATH} = ?;"

def get_delete_documents_sql() -> str:
    """Generates SQL to delete documents by file path."""
    return f"DELETE FROM {DOCUMENTS_TABLE} WHERE {COL_FILE_PATH} = ?;"

def get_delete_vectors_sql(ids_placeholder: str) -> str:
    """Generates SQL to delete vectors based on a list of IDs."""
    # Example ids_placeholder: "(?, ?, ?)"
    return f"DELETE FROM {VEC_DOCUMENTS_TABLE} WHERE {VEC_COL_ID} IN {ids_placeholder};"


# --- Querying SQL ---

def get_vector_match_sql(k_neighbors: int, filter_by_folder: bool) -> str:
    """Generates the SQL query using the MATCH operator for vector search."""
    select_clause = f"""
    SELECT
        t1.{COL_FILE_PATH},
        t1.{COL_CHUNK_INDEX},
        t1.{COL_CONTENT},
        t2.distance
    FROM
        {VEC_DOCUMENTS_TABLE} t2
    LEFT JOIN
        {DOCUMENTS_TABLE} t1 ON t1.{COL_ID} = t2.{VEC_COL_ID}
    """
    where_conditions = [f"t2.{VEC_COL_EMBEDDING} MATCH ?"] # Placeholder for serialized query vector
    if filter_by_folder:
        where_conditions.append(f"t1.{COL_FILE_PATH} LIKE ?")

    where_clause = "WHERE " + " AND ".join(where_conditions)
    k_clause = f"AND k = {k_neighbors}" # k is integer, safe for direct inclusion
    order_clause = "ORDER BY t2.distance"

    return f"{select_clause} {where_clause} {k_clause} {order_clause};"


# --- Other Helper SQL ---

def get_indexed_files_sql() -> str:
    """Generates SQL to retrieve indexed file paths and max mod times."""
    # Query the main documents table
    return f"""
    SELECT {COL_FILE_PATH}, MAX({COL_LAST_MODIFIED})
    FROM {DOCUMENTS_TABLE}
    GROUP BY {COL_FILE_PATH};
    """