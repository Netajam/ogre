# app/core/indexing/file_handler.py
import os
from pathlib import Path
from typing import List, Optional
from app.core.utils.logging_setup import logger

class FileHandler:
    """Handles discovery and reading of files based on specified extensions."""

    @staticmethod
    def find_files_by_extension(folder_path_str: str, allowed_extensions: List[str]) -> List[Path]:
        """
        Finds all files matching the allowed extensions recursively within the given folder.

        Args:
            folder_path_str: The path to the root folder to scan.
            allowed_extensions: A list of file extensions (strings, e.g., [".md", ".txt"])
                                 to include in the search. Extensions should include the leading dot.

        Returns:
            A sorted list of Path objects for the found files. Returns an empty list
            if the folder doesn't exist, is not a directory, or on error.
        """
        if not folder_path_str:
            logger.warning("find_files_by_extension called with empty folder path.")
            return []
        if not allowed_extensions:
             logger.warning("find_files_by_extension called with empty allowed extensions list. No files will be found.")
             return []

        # Validate extensions format (optional, but helpful)
        valid_extensions = [ext for ext in allowed_extensions if isinstance(ext, str) and ext.startswith('.')]
        if len(valid_extensions) != len(allowed_extensions):
            logger.warning(f"Some provided extensions were invalid or missing a leading dot: {allowed_extensions}. Using only valid ones: {valid_extensions}")
        if not valid_extensions:
             logger.error("No valid extensions (starting with '.') provided to find_files_by_extension.")
             return []

        root_path = Path(folder_path_str)
        if not root_path.is_dir():
            logger.error(f"Provided path is not a valid directory: {folder_path_str}")
            return []

        logger.info(f"Scanning for files with extensions {valid_extensions} in: {folder_path_str}")
        found_files: List[Path] = []
        for ext in valid_extensions:
            # rglob pattern: "**/*{.ext}"
            glob_pattern = f"**/*{ext}"
            try:
                # Use rglob for recursive search
                found = list(root_path.rglob(glob_pattern))
                if found:
                     logger.debug(f"Found {len(found)} files with extension {ext}")
                     found_files.extend(found)
            except OSError as e:
                 logger.error(f"Error scanning directory {folder_path_str} for {ext} files: {e}")
            except Exception as e: 
                 logger.error(f"Unexpected error during file search for {ext} in {folder_path_str}: {e}", exc_info=True)


        # Resolve paths to handle potential relative vs absolute issues and normalize case on Windows
        unique_files_resolved = {p.resolve() for p in found_files}
        # Convert back to sorted list
        unique_files_list = sorted(list(unique_files_resolved))

        logger.info(f"Found {len(unique_files_list)} unique target files in total.")
        return unique_files_list

    @staticmethod
    def read_file_content(file_path: Path) -> Optional[str]:
        """Reads the content of a file, trying common encodings."""
        # This method remains the same
        encodings_to_try = ['utf-8', 'latin-1', 'windows-1252']
        for encoding in encodings_to_try:
            try:
                content = file_path.read_text(encoding=encoding)
                logger.debug(f"Successfully read file: {file_path} with encoding {encoding}")
                return content
            except UnicodeDecodeError:
                pass
            except IOError as e:
                logger.error(f"IOError reading file {file_path}: {e}")
                return None # Indicate read error
            except Exception as e:
                logger.error(f"Unexpected error reading file {file_path}: {e}", exc_info=True)
                return None
        logger.error(f"Could not decode file {file_path} with any attempted encoding: {encodings_to_try}.")
        return None

    @staticmethod
    def get_last_modified(file_path: Path) -> Optional[float]:
        """Gets the last modified timestamp of a file."""
        try:
            # Use stat().st_mtime for the modification timestamp
            return file_path.stat().st_mtime
        except FileNotFoundError:
            # This might happen if the file is deleted between finding it and processing it
            logger.warning(f"File not found when getting last modified time: {file_path}")
            return None
        except OSError as e:
            logger.error(f"OSError getting last modified time for {file_path}: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error getting last modified time for {file_path}: {e}", exc_info=True)
            return None