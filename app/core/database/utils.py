# app/core/database/utils.py (Create this new file)
import struct
from typing import List
import numpy as np

def serialize_vector(vector: np.ndarray) -> bytes:
    """
    Serializes a numpy float32 vector into compact bytes using struct.pack.
    """
    if vector is None or not isinstance(vector, np.ndarray) or vector.ndim != 1:
        raise ValueError("Invalid input: Must be a 1D numpy array.")
    # Ensure it's float32, as that's common for embeddings
    vector_float32 = vector.astype(np.float32)
    # '<' means little-endian, 'f' means float (4 bytes)
    # Format string like '<1536f' for a 1536-dim vector
    format_string = '<%sf' % len(vector_float32)
    return struct.pack(format_string, *vector_float32)

# Optional: Add deserialize if needed later, though less common for query results
# def deserialize_vector(data: bytes, dimension: int) -> np.ndarray:
#     """Deserializes bytes back into a numpy float32 vector."""
#     expected_size = dimension * 4 # 4 bytes per float
#     if len(data) != expected_size:
#         raise ValueError(f"Invalid data size for dimension {dimension}. Expected {expected_size}, got {len(data)}")
#     format_string = '<%sf' % dimension
#     vector_tuple = struct.unpack(format_string, data)
#     return np.array(vector_tuple, dtype=np.float32)