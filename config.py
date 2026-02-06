"""
Configuration for EquationMaster

Override any setting via environment variables.
"""

import os

# MongoDB settings
MONGODB_URI = os.environ.get("MONGODB_URI", "mongodb://localhost:27017")
DATABASE_NAME = os.environ.get("DATABASE_NAME", "AstroEquations")
COLLECTION_NAME = os.environ.get("COLLECTION_NAME", "equations")

# Ollama settings
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "gpt-oss:20b")

# Paths
JSON_DIR = "equationJson"
