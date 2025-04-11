# app/main.py
import sys
import os

# Ensure the app directory is in the Python path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from dotenv import load_dotenv

env_loaded = load_dotenv()
# -------------------------------------------------

from PyQt6.QtWidgets import QApplication
from app.frontend.main_window import MainWindow
from app import config
from app.utils.logging_setup import logger 

# --- Log whether .env was loaded (optional, for debugging) ---
if env_loaded:
    logger.info("Loaded environment variables from .env file.")
else:
    logger.info(".env file not found or not loaded. Using system environment variables.")
# ------------------------------------------------------------

def main() -> None:
    """Main function to run the application."""
    logger.info(f"Starting {config.APP_NAME} v{config.APP_VERSION}")
    QApplication.setApplicationName(config.APP_NAME)
    QApplication.setOrganizationName(config.ORGANIZATION_NAME)
    QApplication.setApplicationVersion(config.APP_VERSION)

    app = QApplication(sys.argv)

    window = MainWindow()
    window.show()

    sys.exit(app.exec())

if __name__ == "__main__":
    main()