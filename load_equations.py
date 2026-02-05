#!/usr/bin/env python3
"""
Equation Loader for MongoDB

Loads equation JSON files into MongoDB with LLM-generated names and descriptions.
Uses a SHA256 hash of the LaTeX as the primary key for deduplication.
"""

import hashlib
import json
import re
import sys
from pathlib import Path

import ollama
from pymongo import MongoClient

from config import MONGODB_URI, DATABASE_NAME, COLLECTION_NAME, OLLAMA_MODEL, JSON_DIR


# Resolve JSON directory path
JSON_DIR = Path(__file__).parent / JSON_DIR


def generate_equation_id(latex: str) -> str:
    """Generate a unique ID from the LaTeX string using SHA256 (truncated to 24 chars)."""
    return hashlib.sha256(latex.encode("utf-8")).hexdigest()[:24]


def get_name_and_description(latex: str, reference: str) -> dict:
    """Use Ollama to generate a name and description for an equation."""
    prompt = f"""You are a cosmology expert. Given the following LaTeX equation from a cosmology textbook, provide:
1. A short descriptive name (2-5 words)
2. A brief description explaining what the equation represents and its physical significance (1-2 sentences)

Equation reference: {reference}
LaTeX: {latex}

Respond in exactly this JSON format, with no additional text:
{{"name": "equation name here", "description": "description here"}}"""

    try:
        response = ollama.chat(
            model=OLLAMA_MODEL,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.3}  # Lower temperature for more consistent output
        )

        # Parse the JSON response
        content = response["message"]["content"].strip()
        # Handle case where model wraps response in markdown code blocks
        if "```" in content:
            match = re.search(r'```(?:json)?\s*(.*?)```', content, re.DOTALL)
            if match:
                content = match.group(1).strip()

        # Try to parse JSON
        try:
            result = json.loads(content)
        except json.JSONDecodeError:
            # Fallback: extract fields with regex
            result = {}
            name_match = re.search(r'"name"\s*:\s*"([^"]*)"', content)
            if name_match:
                result["name"] = name_match.group(1)
            desc_match = re.search(r'"description"\s*:\s*"(.*?)"(?=\s*[,}])', content, re.DOTALL)
            if desc_match:
                result["description"] = desc_match.group(1).replace('\n', ' ').strip()

        return {
            "name": result.get("name", ""),
            "description": result.get("description", "")
        }
    except Exception as e:
        print(f"  Warning: Failed to generate name/description: {e}")
        return {"name": "", "description": ""}


def load_json_file(filepath: Path) -> list:
    """Load equations from a JSON file."""
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def process_equation(equation: dict, chapter: str, collection) -> dict:
    """Process a single equation and upsert to MongoDB."""
    latex = equation.get("latex", "")
    reference = equation.get("reference", "")

    if not latex:
        print(f"  Skipping equation with no LaTeX: {reference}")
        return None

    # Generate the unique ID from LaTeX hash
    equation_id = generate_equation_id(latex)

    # Check if document already exists
    existing = collection.find_one({"_id": equation_id})

    # Determine name and description
    name = equation.get("name", "")
    description = equation.get("description", "")

    # If existing document has name/description, preserve them
    if existing:
        if existing.get("name") and not name:
            name = existing["name"]
        if existing.get("description") and not description:
            description = existing["description"]

    # If still missing, generate with LLM
    if not name or not description:
        print(f"  Generating name/description for {reference}...")
        generated = get_name_and_description(latex, reference)
        if not name:
            name = generated["name"]
        if not description:
            description = generated["description"]

    # Build the document
    doc = {
        "_id": equation_id,
        "latex": latex,
        "reference": reference,
        "chapter": chapter,
        "name": name,
        "description": description
    }

    # Upsert the document
    collection.replace_one({"_id": equation_id}, doc, upsert=True)

    return doc


def main():
    """Main entry point."""
    print(f"Connecting to MongoDB at {MONGODB_URI}...")
    client = MongoClient(MONGODB_URI)
    db = client[DATABASE_NAME]
    collection = db[COLLECTION_NAME]

    # Test Ollama connection with a simple request
    print(f"Testing Ollama connection with model '{OLLAMA_MODEL}'...")
    try:
        ollama.chat(
            model=OLLAMA_MODEL,
            messages=[{"role": "user", "content": "Reply with just the word: OK"}],
            options={"num_predict": 10}
        )
        print("  Ollama connection successful")
    except Exception as e:
        print(f"  Warning: Ollama not available ({e})")
        print("  Equations without name/description will have empty fields")

    # Find all JSON files
    json_files = sorted(JSON_DIR.glob("chapter*_equations.json"))

    if not json_files:
        print(f"No equation JSON files found in {JSON_DIR}")
        sys.exit(1)

    print(f"Found {len(json_files)} equation files")

    total_processed = 0
    total_generated = 0

    for json_file in json_files:
        # Extract chapter number from filename
        chapter = json_file.stem.replace("_equations", "")
        print(f"\nProcessing {json_file.name}...")

        equations = load_json_file(json_file)
        print(f"  Found {len(equations)} equations")

        for equation in equations:
            # Track if we need to generate
            needs_generation = not equation.get("name") or not equation.get("description")

            doc = process_equation(equation, chapter, collection)
            if doc:
                total_processed += 1
                if needs_generation and (doc.get("name") or doc.get("description")):
                    total_generated += 1

    print(f"\n{'='*50}")
    print(f"Complete!")
    print(f"  Total equations processed: {total_processed}")
    print(f"  Names/descriptions generated: {total_generated}")
    print(f"  Database: {DATABASE_NAME}")
    print(f"  Collection: {COLLECTION_NAME}")


if __name__ == "__main__":
    main()
