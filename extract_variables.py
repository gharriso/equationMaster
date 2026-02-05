#!/usr/bin/env python3
"""
Variable Extractor for Equations

Scans equations in MongoDB, extracts variables using sympy,
and uses Ollama to generate descriptions, units, and astropy references.

Usage:
    python extract_variables.py          # Process all equations
    python extract_variables.py -n 10    # Process only 10 equations
    python extract_variables.py -n=5     # Process only 5 equations
"""

import argparse
import json
import re
import sys

import ollama
from pymongo import MongoClient
from sympy import symbols
from sympy.parsing.latex import parse_latex

from config import MONGODB_URI, DATABASE_NAME, COLLECTION_NAME, OLLAMA_MODEL


def get_lhs_of_equation(latex: str) -> str:
    """
    Extract the left-hand side of a LaTeX equation.
    Handles =, \approx, \equiv, \propto, and similar operators.
    """
    # Common relation operators in LaTeX
    relation_ops = [
        r'\\approx',
        r'\\equiv',
        r'\\propto',
        r'\\sim',
        r'\\simeq',
        r'\\rightleftharpoons',
        r'\\longrightarrow',
        r'\\rightarrow',
        r'=',
    ]

    # Try each operator, return LHS from first match
    for op in relation_ops:
        pattern = f'^(.*?){op}'
        match = re.search(pattern, latex)
        if match:
            return match.group(1).strip()

    # No relation operator found, return empty string
    return ""


def extract_symbols_from_latex(latex: str) -> list[str]:
    """
    Extract variable symbols from a LaTeX equation using sympy.
    Falls back to regex extraction if sympy parsing fails.
    """
    symbols_found = set()

    # Try sympy parsing first
    try:
        expr = parse_latex(latex)
        # Get all free symbols
        for sym in expr.free_symbols:
            symbols_found.add(str(sym))
    except Exception as e:
        # Sympy can struggle with some LaTeX notation, fall back to regex
        pass

    # Also use regex to catch symbols sympy might miss
    # This catches single letters, Greek letters, and subscripted variables
    patterns = [
        r'\\([a-zA-Z]+)(?![a-zA-Z{])',  # LaTeX commands like \rho, \sigma
        r'(?<![\\a-zA-Z])([a-zA-Z])(?:_\{?([a-zA-Z0-9,]+)\}?)?',  # Variables like T, R, n_e
        r'\\mathrm\{([a-zA-Z]+)\}',  # \mathrm{text}
    ]

    # Greek letter mapping
    greek_letters = {
        'alpha', 'beta', 'gamma', 'delta', 'epsilon', 'zeta', 'eta', 'theta',
        'iota', 'kappa', 'lambda', 'mu', 'nu', 'xi', 'pi', 'rho', 'sigma',
        'tau', 'upsilon', 'phi', 'chi', 'psi', 'omega',
        'Gamma', 'Delta', 'Theta', 'Lambda', 'Xi', 'Pi', 'Sigma', 'Phi', 'Psi', 'Omega'
    }

    # Extract LaTeX Greek letters
    greek_pattern = r'\\(' + '|'.join(greek_letters) + r')(?![a-zA-Z])'
    for match in re.finditer(greek_pattern, latex):
        symbols_found.add(match.group(1))

    # Extract regular variables with optional subscripts
    var_pattern = r'(?<![\\a-zA-Z])([A-Za-z])(?:_\{?([^}\s]+)\}?)?'
    for match in re.finditer(var_pattern, latex):
        base = match.group(1)
        subscript = match.group(2)
        if subscript:
            # Clean up subscript
            subscript = subscript.replace('\\mathrm{', '').replace('}', '')
            symbols_found.add(f"{base}_{subscript}")
        else:
            symbols_found.add(base)

    # Filter out common LaTeX commands that aren't variables
    noise = {
        'mathrm', 'frac', 'sqrt', 'exp', 'log', 'ln', 'sin', 'cos', 'tan',
        'left', 'right', 'd', 'text', 'cdot', 'times', 'approx', 'equiv',
        'propto', 'sum', 'prod', 'int', 'partial', 'nabla', 'infty',
        'rightarrow', 'leftarrow', 'Rightarrow', 'Leftarrow', 'longrightarrow',
        'rightleftharpoons', 'bar', 'hat', 'vec', 'dot', 'ddot', 'langle', 'rangle'
    }

    symbols_found = {s for s in symbols_found if s.lower() not in noise and len(s) > 0}

    return sorted(list(symbols_found))


def get_variable_info(symbol: str, latex: str, equation_name: str, equation_ref: str) -> dict:
    """Use Ollama to get description, units, and astropy references for a variable."""

    prompt = f"""You are a cosmology and astrophysics expert. Analyze this variable from a cosmology equation.

Equation name: {equation_name}
Equation reference: {equation_ref}
Full equation (LaTeX): {latex}
Variable to analyze: {symbol}

Provide information about this variable in the context of this equation. Respond in exactly this JSON format with no additional text:

{{
    "symbol": "{symbol}",
    "name": "short descriptive name of the variable",
    "description": "1 sentence description of what this variable represents physically",
    "astropy_unit": "the astropy.units representation (e.g., 'u.K' for Kelvin, 'u.m' for meters, 'u.kg' for kilograms, 'u.s' for seconds, 'u.J' for Joules, 'u.W' for Watts, 'u.Hz' for Hertz, 'u.m/u.s' for velocity, 'u.dimensionless_unscaled' if dimensionless) or null if not applicable",
    "astropy_constant": "the astropy.constants name if this is a physical constant (e.g., 'const.k_B' for Boltzmann constant, 'const.G' for gravitational constant, 'const.c' for speed of light, 'const.h' for Planck constant, 'const.m_e' for electron mass, 'const.m_p' for proton mass, 'const.sigma_T' for Thomson cross-section) or null if not a constant"
}}

Common astropy.constants: G (gravitational), c (speed of light), h (Planck), hbar (reduced Planck), k_B (Boltzmann), sigma_sb (Stefan-Boltzmann), m_e (electron mass), m_p (proton mass), m_n (neutron mass), e (elementary charge), sigma_T (Thomson cross-section), a (radiation constant), R (gas constant)

Common astropy.units: K, m, s, kg, J, W, Hz, eV, pc, Mpc, Gyr, yr, solMass, solLum, cm, g"""

    try:
        response = ollama.chat(
            model=OLLAMA_MODEL,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.2}
        )

        content = response["message"]["content"].strip()

        # Handle markdown code blocks
        if "```" in content:
            # Extract content between code blocks
            match = re.search(r'```(?:json)?\s*(.*?)```', content, re.DOTALL)
            if match:
                content = match.group(1).strip()

        # Try to parse JSON
        try:
            result = json.loads(content)
        except json.JSONDecodeError:
            # Try to fix common JSON issues and extract fields with regex
            result = {}

            # Extract name
            name_match = re.search(r'"name"\s*:\s*"([^"]*)"', content)
            if name_match:
                result["name"] = name_match.group(1)

            # Extract description - handle multi-line and escaped quotes
            desc_match = re.search(r'"description"\s*:\s*"(.*?)"(?=\s*[,}]|\s*"astropy)', content, re.DOTALL)
            if desc_match:
                result["description"] = desc_match.group(1).replace('\n', ' ').strip()

            # Extract astropy_unit
            unit_match = re.search(r'"astropy_unit"\s*:\s*(?:"([^"]*)"|null)', content)
            if unit_match:
                result["astropy_unit"] = unit_match.group(1)

            # Extract astropy_constant
            const_match = re.search(r'"astropy_constant"\s*:\s*(?:"([^"]*)"|null)', content)
            if const_match:
                result["astropy_constant"] = const_match.group(1)

        return {
            "symbol": symbol,
            "name": result.get("name", ""),
            "description": result.get("description", ""),
            "astropy_unit": result.get("astropy_unit"),
            "astropy_constant": result.get("astropy_constant")
        }
    except json.JSONDecodeError as e:
        print(f"    Warning: Failed to parse JSON for {symbol}: {e}")
        print(f"    Response was: {content[:200]}...")
        return {
            "symbol": symbol,
            "name": "",
            "description": "",
            "astropy_unit": None,
            "astropy_constant": None
        }
    except Exception as e:
        print(f"    Warning: Failed to get info for {symbol}: {e}")
        return {
            "symbol": symbol,
            "name": "",
            "description": "",
            "astropy_unit": None,
            "astropy_constant": None
        }


def is_symbol_on_lhs(symbol: str, lhs: str) -> bool:
    """Check if a symbol appears on the left-hand side of an equation."""
    if not lhs:
        return False

    # Extract symbols from LHS
    lhs_symbols = extract_symbols_from_latex(lhs)
    return symbol in lhs_symbols


def process_equation(doc: dict, collection) -> int:
    """Process a single equation and extract its variables."""
    latex = doc.get("latex", "")
    equation_id = doc.get("_id")
    equation_name = doc.get("name", "")
    equation_ref = doc.get("reference", "")

    print(f"  Processing {equation_ref}: {equation_name or '(unnamed)'}")

    # Extract the left-hand side of the equation
    lhs = get_lhs_of_equation(latex)
    if lhs:
        print(f"    LHS: {lhs[:50]}{'...' if len(lhs) > 50 else ''}")

    # Extract symbols from full equation
    symbols_list = extract_symbols_from_latex(latex)
    print(f"    Found {len(symbols_list)} symbols: {', '.join(symbols_list)}")

    if not symbols_list:
        return 0

    # Get info for each symbol
    variables = []
    for symbol in symbols_list:
        print(f"    Getting info for '{symbol}'...")
        var_info = get_variable_info(symbol, latex, equation_name, equation_ref)

        # Add LHS flag
        var_info["lhs"] = is_symbol_on_lhs(symbol, lhs)
        if var_info["lhs"]:
            print(f"      (on LHS)")

        variables.append(var_info)

    # Update the document
    collection.update_one(
        {"_id": equation_id},
        {"$set": {"variables": variables}}
    )

    return len(variables)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Extract variables from equations and enrich with LLM-generated metadata."
    )
    parser.add_argument(
        "-n", "--limit",
        type=int,
        default=None,
        help="Limit processing to N equations (default: process all)"
    )
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    print(f"Connecting to MongoDB at {MONGODB_URI}...")
    client = MongoClient(MONGODB_URI)
    db = client[DATABASE_NAME]
    collection = db[COLLECTION_NAME]

    # Test Ollama connection with a simple request
    print(f"Testing Ollama connection with model '{OLLAMA_MODEL}'...")
    try:
        response = ollama.chat(
            model=OLLAMA_MODEL,
            messages=[{"role": "user", "content": "Reply with just the word: OK"}],
            options={"num_predict": 10}
        )
        print(f"  Ollama connection successful")
    except Exception as e:
        print(f"  Error: Ollama not available ({e})")
        print("  Please ensure Ollama is running: ollama serve")
        print(f"  And that the model is pulled: ollama pull {OLLAMA_MODEL}")
        sys.exit(1)

    # Get equations (with optional limit)
    if args.limit:
        equations = list(collection.find({}).limit(args.limit))
        print(f"\nProcessing {len(equations)} equations (limited to {args.limit})")
    else:
        equations = list(collection.find({}))
        print(f"\nFound {len(equations)} equations in database")

    if not equations:
        print("No equations found. Run load_equations.py first.")
        sys.exit(1)

    total_variables = 0

    for i, doc in enumerate(equations, 1):
        if args.limit:
            print(f"\n[{i}/{args.limit}]")
        variables_count = process_equation(doc, collection)
        total_variables += variables_count

    print(f"\n{'='*50}")
    print(f"Complete!")
    print(f"  Equations processed: {len(equations)}")
    print(f"  Total variables extracted: {total_variables}")


if __name__ == "__main__":
    main()
