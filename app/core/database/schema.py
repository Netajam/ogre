# app/core/database/schema.py
from typing import Optional

# Table name
DOCUMENTS_TABLE = "documents"

# Columns
COL_ID = "id"
COL_FILE_PATH = "file_path"
COL_CHUNK_INDEX = "chunk_index"
COL_CONTENT = "content"
COL_EMBEDDING = "embedding"
COL_LAST_MODIFIED = "last_modified" 

# Vector index name
VECTOR_INDEX_NAME = "idx_doc_embeddings"

def get_create_table_sql(embedding_dim: int) -> str:
    """Generates the SQL statement to create the documents table."""
    if embedding_dim <= 0:
        raise ValueError("Embedding dimension must be positive.")
    # Note: VECTOR type is provided by sqlite-vec
    return f"""
    CREATE TABLE IF NOT EXISTS {DOCUMENTS_TABLE} (
        {COL_ID} INTEGER PRIMARY KEY AUTOINCREMENT,
        {COL_FILE_PATH} TEXT NOT NULL,
        {COL_CHUNK_INDEX} INTEGER NOT NULL,
        {COL_CONTENT} TEXT NOT NULL,
        {COL_EMBEDDING} VECTOR({embedding_dim}),
        {COL_LAST_MODIFIED} REAL NOT NULL
    );
    """

def get_create_file_path_index_sql() -> str:
    """Generates SQL to create an index on the file path."""
    return f"""
    CREATE INDEX IF NOT EXISTS idx_file_path ON {DOCUMENTS_TABLE} ({COL_FILE_PATH});
    """

def get_create_vector_index_sql(metric_type: str = 'cosine') -> str:
    """
    Generates the SQL statement to create a vector index (HNSW by default in sqlite-vec).
    metric_type can be 'cosine' or 'l2'.
    """
    # TODO:Adjust indexing options if needed based on sqlite-vec documentation/performance
    # Example: USING HNSW(metric_type=cosine, ef_construction=100, M=16)
    options = f"metric_type={metric_type}"
    return f"""
    CREATE VIRTUAL TABLE IF NOT EXISTS {VECTOR_INDEX_NAME} USING vector_index(
        {DOCUMENTS_TABLE}({COL_EMBEDDING}), {options}
    );
    """
   

def get_insert_chunk_sql() -> str:
    """Generates the SQL statement for inserting a single chunk."""
    return f"""
    INSERT INTO {DOCUMENTS_TABLE} (
        {COL_FILE_PATH}, {COL_CHUNK_INDEX}, {COL_CONTENT}, {COL_EMBEDDING}, {COL_LAST_MODIFIED}
    ) VALUES (?, ?, ?, ?, ?);
    """

def get_delete_chunks_sql() -> str:
    """Generates SQL to delete chunks for a specific file path."""
    return f"DELETE FROM {DOCUMENTS_TABLE} WHERE {COL_FILE_PATH} = ?;"

def get_indexed_files_sql() -> str:
    """Generates SQL to retrieve all indexed file paths and their last modified times."""
    return f"""
    SELECT {COL_FILE_PATH}, MAX({COL_LAST_MODIFIED})
    FROM {DOCUMENTS_TABLE}
    GROUP BY {COL_FILE_PATH};
    """

