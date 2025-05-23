# Ogre: Your Personal Knowledge Retrieval App for Notes

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
<!-- Optional: Add build status badges etc. here later -->

Ogre is a desktop application designed to help you unlock the knowledge hidden within your personal notes. It uses a Retrieval-Augmented Generation (RAG) approach, indexing your notes into a local vector database and allowing you to ask questions in natural language to find relevant information and view the original context.

**(Note: This project is under active development. LLM integration for summarized answers is planned.)**

<!-- ![Screenshot of Ogre Application](placeholder_screenshot.png) -->
**(Insert a screenshot of the application UI here when available)**

## Features

*   **Local Note Indexing:** Select a folder containing your notes.
*   **Configurable File Types:** Index specific file types (defaults to `.md`, `.markdown`, configurable via `settings.json`).
*   **Vector Database:** Uses SQLite with the `sqlite-vec` extension for efficient local vector storage and similarity search. The database (`.note_query_index.sqlite`) is stored within your selected notes folder.
*   **Embedding Model Choice:**
    *   **Local:** Supports Sentence Transformer models (e.g., `all-MiniLM-L6-v2`) running entirely on your machine.
    *   **API:** Supports Google's Gemini embedding models (requires a Gemini API key via `.env` file).
*   **Background Processing:** Indexing and querying run in background threads to keep the UI responsive.
*   **Semantic Search:** Ask questions in natural language to find relevant chunks of text within your notes.
*   **Contextual Display:**
    *   View a list of the most relevant text chunks found for your query.
    *   Click on a chunk in the list to view the full content of the original note file.
    *   The specific chunk is highlighted within the full note content for easy identification.
*   **Desktop GUI:** User-friendly interface built with Python and PyQt6.
*   **Cross-Platform (Goal):** Developed with cross-platform libraries, aiming for compatibility with Windows, macOS, and Linux.

*(Planned Features / Phase 3b+):*
*   *LLM Integration:* Generate summarized, conversational answers based on the retrieved chunks.
*   *UI Settings Management:* Allow configuration of embedding models and indexed file types directly through the Settings dialog.
*   *Improved Progress Reporting:* More granular feedback during indexing.
*   *Advanced Search Options:* Potential for filtering, date ranges, etc.
*   *Export/Import:* Options for managing the index.

## Technology Stack

*   **Language:** Python 3.10+
*   **UI Framework:** PyQt6
*   **Vector Database:** SQLite + `sqlite-vec` extension (v0.1.x branch - uses `vec0` virtual table)
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
    # Use your actual repository URL here
    git clone https://github.com/Netajam/ogre.git
    cd ogre
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
        # ... other entries ...
        .env
        *.sqlite
        *.sqlite-journal
        logs/
        venv/
        .venv/
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
    *   Click the "Select Notes Folder" button.
    *   Choose the main directory containing the notes you want to index.

4.  **Index Your Notes:**
    *   Click the "Index Notes" button.
    *   Ogre will scan, embed, and index the supported files in the selected folder. This can take time, especially on the first run or with many notes. Wait for the completion message.
    *   **Note:** If you change the embedding model or indexed file types (see Configuration), you should re-index your notes.

5.  **Ask Questions:**
    *   Once indexing is complete, type your question into the "Query:" box.
    *   Click "Ask" or press Enter.

6.  **View Results:**
    *   A list of the most relevant text chunks from your notes will appear in the "Relevant Chunks" panel on the left.
    *   Click on any item in the list.
    *   The full content of the source note file will be displayed in the main "Note Content" panel on the right, automatically scrolled to and highlighting the selected chunk.

7.  **Settings:**
    *   Access `File -> Settings` (Currently shows a placeholder message).
    *   Configuration changes (like embedding model or indexed extensions) currently require manual editing of `settings.json` (see Configuration).

## Configuration

Settings are stored in a `settings.json` file located in the application's standard configuration directory (the path is logged on startup, e.g., `C:\Users\YourUser\AppData\Local\NetajamCorp\Ogre\settings.json`).

*   **`notes_folder`:** (String | null) The absolute path to the directory containing your notes. Set via the UI.
*   **`embedding_model`:** (String) The display name of the embedding model to use (e.g., `"all-MiniLM-L6-v2 (Local)"`, `"Gemini (API)"`). Must match a key in `app/config.py`. Defaults are applied if missing or invalid. *Currently changeable only by editing `settings.json`.*
*   **`indexed_extensions`:** (List[String]) A list of file extensions (e.g., `[".md", ".txt"]`) that the indexer should process. Must include the leading dot. Defaults to `[".md", ".markdown"]`. *Currently changeable only by editing `settings.json`.*
*   **Database File:** A SQLite file named `.note_query_index.sqlite` is created *inside* your selected `notes_folder`. **Important:** If your notes folder is version controlled (e.g., Git), ensure you add `*.sqlite` and `*.sqlite-journal` to its `.gitignore` file.
*   **API Keys:** The Gemini API key is loaded from the `GEMINI_API_KEY` environment variable (set via the `.env` file). It is *not* stored in `settings.json`.

## Development

*   **Dependencies:** Install development requirements: `pip install -r requirements-dev.txt`
*   **Running Tests:** Tests are located in the `tests/` directory. Run them using pytest:
    ```bash
    pytest
    ```
    *(Note: Test coverage may be incomplete, especially for UI and newer features).*

## License

This project uses PyQt6, which is licensed under the GNU General Public License v3 (GPL v3). Therefore, Ogre is also distributed under the **GPL v3 License**. See the [LICENSE](LICENSE) file for details.

Please be mindful of the licenses of all dependencies used in this project.

## Contributing

Contributions are welcome! Please feel free to open an issue to report bugs or suggest features, or submit a pull request with improvements.