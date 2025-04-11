# app/utils/logging_setup.py
import logging
import sys
import os
from pathlib import Path
from app import config 

def setup_logging():
    log_format = '%(asctime)s - %(name)s [%(levelname)s] - %(message)s (%(filename)s:%(lineno)d)'

    # --- Determine Log File Path ---
    log_dir = None
    try:
        from PyQt6.QtCore import QStandardPaths # Optional import if needed
        app_data_dir = QStandardPaths.writableLocation(QStandardPaths.StandardLocation.AppLocalDataLocation)
        log_base_dir = Path(app_data_dir)
        if config.ORGANIZATION_NAME and config.ORGANIZATION_NAME not in log_base_dir.name:
             log_base_dir = log_base_dir / config.ORGANIZATION_NAME
        if config.APP_NAME not in log_base_dir.name:
             log_base_dir = log_base_dir / config.APP_NAME
        log_dir = log_base_dir / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / "app.log"
    except ImportError:
        # Fallback if PyQt6 isn't available right at startup or in non-GUI context
        project_root = Path(__file__).parent.parent.parent # Assumes utils is app/utils/
        log_dir = project_root / "logs"
        log_dir.mkdir(exist_ok=True) 
        log_file = log_dir / "app.log"
    except Exception as e:
        print(f"Warning: Could not determine standard log directory: {e}. Using ./logs/app.log")
        project_root = Path(__file__).parent.parent.parent
        log_dir = project_root / "logs"
        log_dir.mkdir(exist_ok=True)
        log_file = log_dir / "app.log"


    # --- Configure Logging ---
    # Get the root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG) 

    # Clear existing handlers (useful for reloading)
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
        handler.close()

    # Console Handler (INFO level)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter(log_format)
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)

    # File Handler (DEBUG level)
    try:
        file_handler = logging.FileHandler(log_file, encoding='utf-8', delay=True)
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter(log_format)
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)
        print(f"Logging to file: {log_file}") 
    except IOError as e:
        print(f"Error: Could not set up log file handler at {log_file}: {e}")


    # --- Set Specific Library Levels ---
    logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
    logging.getLogger("sqlite_vec").setLevel(logging.INFO)
    logging.getLogger("urllib3").setLevel(logging.WARNING)



    app_logger = logging.getLogger(config.APP_NAME)


    return app_logger

logger = setup_logging()