# EquationMaster

A tool for extracting, managing, and browsing physics equations from the Open University S385 cosmology course materials. Equations are stored in MongoDB with LaTeX notation, variable definitions, and auto-generated names and descriptions via a local LLM.

## Features

- Browse equations by chapter or variable symbol
- Rendered LaTeX display
- Per-variable descriptions with astropy unit and constant references
- Runnable Python code snippets for each equation
- Filter and search across all equations

## Prerequisites

- Python 3.9+
- [MongoDB](https://www.mongodb.com/try/download/community) running locally on port 27017
- [Ollama](https://ollama.com/) running locally (for LLM-generated names/descriptions and variable extraction)

## Installation

1. Clone the repository:

   ```bash
   git clone <repo-url>
   cd equationMaster
   ```

2. Install Python dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Pull the required Ollama model (default is `gpt-oss:20b`):

   ```bash
   ollama pull gpt-oss:20b
   ```

   To use a different model, set the `OLLAMA_MODEL` environment variable before running any scripts.

## Configuration

All settings can be overridden via environment variables:

| Variable | Default | Description |
|---|---|---|
| `MONGODB_URI` | `mongodb://localhost:27017` | MongoDB connection URI |
| `DATABASE_NAME` | `AstroEquations` | MongoDB database name |
| `COLLECTION_NAME` | `equations` | MongoDB collection name |
| `OLLAMA_MODEL` | `gpt-oss:20b` | Ollama model for LLM generation |

## Populating the Database

The database is populated in three steps.

### Option A: Restore from dump (fastest)

A pre-built database dump is included in `mongodump.gz`. Restore it with:

```bash
mongorestore --gzip --archive=mongodump.gz
```

### Option B: Load from JSON files

The `equationJson/` directory contains extracted equations for chapters 1–11. Load them into MongoDB (this also calls Ollama to generate names and descriptions for any equation that is missing them):

```bash
python load_equations.py
```

This script will:
- Read all `equationJson/chapter*_equations.json` files
- Upsert each equation into MongoDB using a SHA256 hash of its LaTeX as the document ID
- Call Ollama to generate a name and description for any equation that lacks one

### Step 2: Create indexes

After loading equations, create the MongoDB indexes for efficient querying:

```bash
python create_indexes.py
```

### Step 3: Extract variables

Extract variable definitions from each equation and enrich them with unit and astropy references:

```bash
python extract_variables.py          # Process all equations
python extract_variables.py -n 10   # Process only 10 equations (for testing)
```

This script uses sympy to parse the LaTeX and Ollama to describe each variable. LLM interactions are logged to `llm_interactions.log`.

## Running the App

Start the Streamlit web interface:

```bash
streamlit run app.py
```

Then open [http://localhost:8501](http://localhost:8501) in your browser.

## Project Structure

```
equationMaster/
├── app.py                  # Streamlit web UI
├── config.py               # Configuration (MongoDB, Ollama settings)
├── load_equations.py       # Load JSON equations into MongoDB
├── extract_variables.py    # Extract and enrich variable definitions
├── create_indexes.py       # Create MongoDB indexes
├── convert_docx_to_latex.py # Extract equations from Word documents
├── equationJson/           # Equation JSON files by chapter
│   ├── chapter1_equations.json
│   └── ...
├── chapter/                # Source PDF files
├── equations.json          # Consolidated equation reference
├── mongodump.gz            # Pre-built database dump
└── requirements.txt
```

## Equation JSON Schema

Each equation entry follows this structure:

```json
{
  "name": "Equation Name",
  "description": "What the equation describes and its physical significance",
  "latex": "\\LaTeX notation for the equation",
  "reference": "Equation number from source (e.g., 'Equation 1.1' or '(4.34)')"
}
```
