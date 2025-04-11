# Ogre: Your Personal Knowledge Retrieval App for Notes

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

Ogre is a desktop application designed to help you unlock the knowledge hidden within your personal notes. It uses a Retrieval-Augmented Generation (RAG) approach, indexing your notes into a local vector database and allowing you to ask questions in natural language to find relevant information.

**(Note: This project is under active development. Core indexing is functional, full RAG query capabilities are planned.)**

## Features

*   **Local Note Indexing:** Select a folder containing your notes, and Ogre will process them.
*   **Configurable File Types:** Index specific file types (defaults to `.md`, `.markdown`, configurable via `settings.json`).
*   **Vector Database:** Uses SQLite with the `sqlite-vec` extension for efficient local vector storage and similarity search. The database is stored within your selected notes folder.
*   **Embedding Model Choice:**
    *   **Local:** Supports Sentence Transformer models (e.g., `all-MiniLM-L6-v2`) running entirely on your machine.
    *   **API:** Supports Google's Gemini embedding models (requires an API key).
*   **Background Processing:** Indexing runs in a background thread to keep the UI responsive.
*   **Desktop GUI:** User-friendly interface built with Python and PyQt6.
*   **Cross-Platform (Goal):** Developed with cross-platform libraries, aiming for compatibility with Windows, macOS, and Linux.

*(Planned Features / Phase 3):*
*   *Full RAG Querying:* Integrate LLM interaction to provide summarized answers based on retrieved notes.
*   *UI Settings Management:* Allow configuration of embedding models and indexed file types directly through the Settings dialog.
*   *Improved Progress Reporting:* More granular feedback during indexing and querying.
*   *Source Linking:* Link answers back to the specific notes they were derived from.

## Technology Stack

*   **Language:** Python 3.10+
*   **UI Framework:** PyQt6
*   **Vector Database:** SQLite + `sqlite-vec` extension
*   **Embeddings (Local):** `sentence-transformers` library
*   **Embeddings (API):** `google-generativeai` library (for Gemini)
*   **Text Splitting:** `langchain-text-splitters`
*   **Environment Variables:** `python-dotenv`

## Installation

1.  **Prerequisites:**
    *   Git
    *   Python 3.10 or later (including `pip`)

2.  **Clone the Repository:**
    ```bash
    git clone https://github.com/Netajam/ogre.git
    cd Ogre
    ```

3.  **Create and Activate Virtual Environment (Recommended):**
    *   **Windows:**
        ```bash
        python -m venv venv
        .\venv\Scripts\activate
        ```
    *   **macOS/Linux:**
        ```bash
        python3 -m venv venv
        source venv/bin/activate
        ```

4.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *(Optional: For development, also install `pip install -r requirements-dev.txt`)*

5.  **Set up API Keys (If using Gemini Embeddings):**
    *   Create a file named `.env` in the root directory of the project (where this README is located).
    *   Add your Gemini API key to the `.env` file:
        ```dotenv
        # .env
        GEMINI_API_KEY="YOUR_ACTUAL_GEMINI_API_KEY"
        ```
    *   **IMPORTANT:** Add the `.env` file to your `.gitignore` file to prevent accidentally committing your secret key!
        ```gitignore
        # .gitignore
        # ... other entries
        .env
        *.sqlite
        *.sqlite-journal
        logs/
        venv/
        __pycache__/
        ```

## Usage

1.  **Activate Virtual Environment** (if not already active):
    *   Windows: `.\venv\Scripts\activate`
    *   macOS/Linux: `source venv/bin/activate`

2.  **Run the Application:**
    ```bash
    python app/main.py
    ```

3.  **Select Notes Folder:**
    *   On first run, or if no folder is set, click the "Select Notes Folder" button.
    *   Choose the main directory containing the notes you want to index.

4.  **Index Your Notes:**
    *   Click the "Index Notes" button.
    *   The application will scan the selected folder for supported file types (see Configuration below), process them, generate embeddings (this may take time, especially the first time a model is downloaded), and store them in the local database (`.note_query_index.sqlite` within your notes folder).
    *   Progress will be shown in the status bar. Wait for the completion message.

5.  **Ask Questions (Phase 3 Placeholder):**
    *   Once indexing is complete, you can type your questions about your notes into the "Enter your query..." box and click "Ask" (or press Enter).
    *   *(Currently, this will show placeholder text. Full RAG functionality is planned for Phase 3).*

6.  **Settings:**
    *   Access `File -> Settings` (Currently shows a placeholder message).
    *   Configuration changes (like embedding model or indexed extensions) currently require manual editing of `settings.json` (see Configuration).

## Configuration

Settings are stored in a `settings.json` file located in the application's standard configuration directory (path logged on startup, e.g., `C:\Users\YourUser\AppData\Local\NetajamCorp\Ogre\settings.json`).

*   **`notes_folder`:** (String | null) The absolute path to the directory containing your notes. Set via the UI.
*   **`embedding_model`:** (String) The display name of the embedding model to use (must be one of the keys in `AVAILABLE_EMBEDDING_MODELS` in `app/config.py`).
    *   Defaults to `"all-MiniLM-L6-v2 (Local)"`.
    *   *Currently changeable only by editing `settings.json`.*
*   **`indexed_extensions`:** (List[String]) A list of file extensions (including the leading dot) that the indexer should process.
    *   Defaults to `[".md", ".markdown"]`.
    *   Example: `[".md", ".txt"]` to include text files.
    *   *Currently changeable only by editing `settings.json`.*
*   **Database File:** A SQLite file named `.note_query_index.sqlite` is created *inside* your selected `notes_folder`. **Important:** If your notes folder is a Git repository, ensure you add `*.sqlite` and `*.sqlite-journal` to its `.gitignore` file to avoid committing the large index database.
*   **API Keys:** The Gemini API key is loaded from the `GEMINI_API_KEY` environment variable (via the `.env` file). It is *not* stored in `settings.json`.

## Development

*   **Dependencies:** Install development requirements: `pip install -r requirements-dev.txt`
*   **Running Tests:** Tests are located in the `tests/` directory. Run them using pytest:
    ```bash
    pytest
    ```

## License

This project uses PyQt6, which is licensed under the GNU General Public License v3 (GPL v3). Therefore, Ogre is also distributed under the **GPL v3 License**. See the [LICENSE](LICENSE) file for details.

Please be mindful of the licenses of all dependencies used in this project.

## Contributing

Contributions are welcome! Please feel free to open an issue or submit a pull request. 