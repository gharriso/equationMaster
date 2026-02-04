# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

EquationMaster is a Python project for extracting, converting, and managing physics equations from cosmology course materials (Open University S385). \

## Project Structure

- `convert_docx_to_latex.py` - Main script for extracting equations from Word documents containing tables. Converts Greek symbols and math notation to LaTeX.
- `latexExample.py` - Example script demonstrating LaTeX parsing with sympy
- `equationJson/` - JSON files containing extracted equations organized by chapter (chapter1_equations.json through chapter8_equations.json)
- `chapter/` - Source PDF files for each chapter of the cosmology textbook
- `equations.json` - Consolidated equation reference file

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

We create these .json file from the pdf files in the `chapters` directory.  During initial conversion namd and description will be blank.  We'll fill these in later.
## Dependencies

- `python-docx` - For reading Word documents
- `sympy` - For LaTeX parsing and mathematical expression handling


