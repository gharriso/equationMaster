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
import logging
import re
import sys
from datetime import datetime
from pathlib import Path

import ollama
from pymongo import MongoClient
from sympy import symbols
from sympy.parsing.latex import parse_latex

from config import MONGODB_URI, DATABASE_NAME, COLLECTION_NAME, OLLAMA_MODEL


# Set up LLM-specific logging to a separate file
LLM_LOG_FILE = Path(__file__).parent / "llm_interactions.log"
llm_logger = logging.getLogger("llm")
llm_logger.setLevel(logging.DEBUG)
llm_handler = logging.FileHandler(LLM_LOG_FILE, mode='a', encoding='utf-8')
llm_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
llm_logger.addHandler(llm_handler)


def log_llm_request(context: str, prompt: str):
    """Log an LLM request."""
    llm_logger.info(f"{'='*60}")
    llm_logger.info(f"REQUEST: {context}")
    llm_logger.info(f"MODEL: {OLLAMA_MODEL}")
    llm_logger.info(f"PROMPT:\n{prompt}")


def log_llm_response(context: str, response: str, duration_ms: float = None):
    """Log an LLM response."""
    duration_str = f" ({duration_ms:.0f}ms)" if duration_ms else ""
    llm_logger.info(f"RESPONSE{duration_str}:\n{response}")
    llm_logger.info(f"END: {context}")
    llm_logger.info("")


def log_llm_error(context: str, error: str):
    """Log an LLM error."""
    llm_logger.error(f"ERROR in {context}: {error}")
    llm_logger.info("")


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


def get_all_variables_info(symbols_list: list, latex: str, equation_name: str, equation_ref: str) -> list:
    """Use Ollama to get description, units, and astropy references for ALL variables in one call."""

    symbols_str = ", ".join(symbols_list)

    prompt = f"""You are a cosmology and astrophysics expert. Analyze ALL variables from this cosmology equation.

Equation name: {equation_name}
Equation reference: {equation_ref}
Full equation (LaTeX): {latex}
Variables to analyze: {symbols_str}

Provide information about EACH variable in the context of this equation. Respond with a JSON array containing one object per variable. Use exactly this format with no additional text:

[
  {{
    "symbol": "first_symbol",
    "name": "short descriptive name",
    "description": "1 sentence description of what this variable represents physically",
    "astropy_unit": "u.X format (e.g., 'u.K', 'u.m', 'u.kg', 'u.s', 'u.m/u.s', 'u.dimensionless_unscaled') or null",
    "astropy_constant": "const.X format (e.g., 'const.G', 'const.k_B', 'const.c') or null if not a constant"
  }},
  ... one object for each variable ...
]

Common astropy.constants: G (gravitational), c (speed of light), h (Planck), hbar (reduced Planck), k_B (Boltzmann), sigma_sb (Stefan-Boltzmann), m_e (electron mass), m_p (proton mass), m_n (neutron mass), e (elementary charge), sigma_T (Thomson cross-section), a (radiation constant), R (gas constant)

Common astropy.units: K, m, s, kg, J, W, Hz, eV, pc, Mpc, Gyr, yr, solMass, solLum, cm, g

Return a JSON array with exactly {len(symbols_list)} objects, one for each variable: {symbols_str}"""

    context = f"get_all_variables_info({equation_ref}, {len(symbols_list)} vars)"
    log_llm_request(context, prompt)

    try:
        import time
        start_time = time.time()

        response = ollama.chat(
            model=OLLAMA_MODEL,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.2}
        )

        duration_ms = (time.time() - start_time) * 1000
        content = response["message"]["content"].strip()
        log_llm_response(context, content, duration_ms)

        # Handle markdown code blocks
        if "```" in content:
            match = re.search(r'```(?:json)?\s*(.*?)```', content, re.DOTALL)
            if match:
                content = match.group(1).strip()

        # Try to parse JSON array
        try:
            results = json.loads(content)
            if isinstance(results, list):
                # Ensure all symbols are covered
                result_symbols = {r.get("symbol") for r in results}
                variables = []
                for symbol in symbols_list:
                    # Find matching result or create empty one
                    matching = [r for r in results if r.get("symbol") == symbol]
                    if matching:
                        var = matching[0]
                        variables.append({
                            "symbol": symbol,
                            "name": var.get("name", ""),
                            "description": var.get("description", ""),
                            "astropy_unit": var.get("astropy_unit"),
                            "astropy_constant": var.get("astropy_constant")
                        })
                    else:
                        variables.append({
                            "symbol": symbol,
                            "name": "",
                            "description": "",
                            "astropy_unit": None,
                            "astropy_constant": None
                        })
                return variables
        except json.JSONDecodeError:
            pass

        # Fallback: try to extract individual objects from the response
        variables = []
        for symbol in symbols_list:
            # Look for this symbol's entry in the response
            pattern = rf'\{{\s*"symbol"\s*:\s*"{re.escape(symbol)}"[^}}]*\}}'
            match = re.search(pattern, content, re.DOTALL)
            if match:
                try:
                    var = json.loads(match.group(0))
                    variables.append({
                        "symbol": symbol,
                        "name": var.get("name", ""),
                        "description": var.get("description", ""),
                        "astropy_unit": var.get("astropy_unit"),
                        "astropy_constant": var.get("astropy_constant")
                    })
                    continue
                except json.JSONDecodeError:
                    pass
            # Default empty entry
            variables.append({
                "symbol": symbol,
                "name": "",
                "description": "",
                "astropy_unit": None,
                "astropy_constant": None
            })
        return variables

    except Exception as e:
        log_llm_error(context, str(e))
        print(f"    Warning: Failed to get variable info: {e}")
        # Return empty entries for all symbols
        return [{
            "symbol": symbol,
            "name": "",
            "description": "",
            "astropy_unit": None,
            "astropy_constant": None
        } for symbol in symbols_list]


def is_symbol_on_lhs(symbol: str, lhs: str) -> bool:
    """Check if a symbol appears on the left-hand side of an equation."""
    if not lhs:
        return False

    # Extract symbols from LHS
    lhs_symbols = extract_symbols_from_latex(lhs)
    return symbol in lhs_symbols


def generate_sample_script(latex: str, name: str, reference: str, variables: list) -> str:
    """
    Use Ollama to generate a sample Python script for an equation.
    The script will:
    1. Define sympy symbols for each variable
    2. Create a sympy Eq() representation
    3. Assign values with astropy units
    4. Create a lambdified function
    """
    if not variables:
        return ""

    # Build variable info for the prompt
    var_info_lines = []
    for var in variables:
        symbol = var.get("symbol", "")
        var_name = var.get("name", "")
        unit = var.get("astropy_unit", "")
        constant = var.get("astropy_constant", "")
        is_lhs = var.get("lhs", False)

        line = f"  - {symbol}: {var_name}"
        if is_lhs:
            line += " [LEFT HAND SIDE - this is what we solve for]"
        if constant:
            line += f", astropy constant: {constant}"
        elif unit:
            line += f", unit: {unit}"
        var_info_lines.append(line)

    var_info = "\n".join(var_info_lines)

    prompt = f"""You are an expert Python programmer specializing in physics and symbolic mathematics.

Generate a complete, working Python script for this cosmology equation, following the template below exactly.

Equation: {name}
Reference: {reference}
LaTeX: {latex}

Variables:
{var_info}

IMPORTANT RULES:
1. Use Rational() from sympy to enclose ALL numeric fractions and exponents (e.g., Rational(3,2) not 1.5 or 3/2)
2. Define ALL sympy symbols with real=True, positive=True
3. Follow the template structure exactly
4. Pass astropy Quantity values directly to the lambdified function - they work with numpy

TEMPLATE TO FOLLOW:
```python
# Jeans mass for polytropic gas (Equation 10.1)

from sympy import symbols, Eq, solve, lambdify, sqrt, exp, log, pi, init_printing, Rational
from astropy import units as u, constants as const
from IPython.display import display

init_printing()

# Define symbols
G, M_J, T, k_B, m, n = symbols('G M_J T k_B m n', real=True, positive=True)

# Construct the equation
rhs = (Rational(9,4)) * sqrt(Rational(1)/(Rational(2)*pi*n)) * Rational(1)/(m**Rational(2)) * ((k_B*T)/G)**(Rational(3,2))
eq10_1 = Eq(M_J, rhs)

# Display the equation
display(eq10_1)

# Solve for M_J and display the solution
sol = solve(eq10_1, M_J)[0]
display(sol)

# Lambdify the RHS (all RHS variables as arguments)
eq10_1_lambidified = lambdify((G, T, k_B, m, n), rhs, modules='numpy')

# Assign sample values with astropy units
G_val   = const.G          # gravitational constant
T_val   = 10.0 * u.K       # temperature
k_B_val = const.k_B        # Boltzmann constant
m_val   = 1.67e-27 * u.kg  # mean particle mass (proton mass)
n_val   = 1e6 * u.m**-3    # particle number density

# Compute the Jeans mass
M_J_val = eq10_1_lambidified(G_val, T_val, k_B_val, m_val, n_val)

# Print the result
display("Jeans mass")
display(M_J_val.to('kg'))
```

Now generate a similar script for the equation above. Return ONLY the Python code, no explanations."""

    context = f"generate_sample_script({reference})"
    log_llm_request(context, prompt)

    try:
        import time
        start_time = time.time()

        response = ollama.chat(
            model=OLLAMA_MODEL,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.2}
        )

        duration_ms = (time.time() - start_time) * 1000
        content = response["message"]["content"].strip()
        log_llm_response(context, content, duration_ms)

        # Remove markdown code blocks if present
        if "```python" in content:
            match = re.search(r'```python\s*(.*?)```', content, re.DOTALL)
            if match:
                content = match.group(1).strip()
        elif "```" in content:
            match = re.search(r'```\s*(.*?)```', content, re.DOTALL)
            if match:
                content = match.group(1).strip()

        return content

    except Exception as e:
        log_llm_error(context, str(e))
        print(f"    Warning: Failed to generate script: {e}")
        return ""


def process_equation(doc: dict, collection) -> tuple[int, bool]:
    """Process a single equation: extract variables and generate sample script."""
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
        return 0, False

    # Get info for ALL symbols in a single LLM call
    print(f"    Getting info for all {len(symbols_list)} variables...")
    variables = get_all_variables_info(symbols_list, latex, equation_name, equation_ref)

    # Add LHS flag to each variable
    for var_info in variables:
        var_info["lhs"] = is_symbol_on_lhs(var_info["symbol"], lhs)
        if var_info["lhs"]:
            print(f"      {var_info['symbol']} is on LHS")

    # Generate sample Python script
    print(f"    Generating sample script...")
    sample_script = generate_sample_script(latex, equation_name, equation_ref, variables)
    script_generated = bool(sample_script)
    if script_generated:
        print(f"    Script generated ({len(sample_script)} chars)")
    else:
        print(f"    Script generation failed")

    # Update the document with variables and script
    update_data = {"variables": variables}
    if sample_script:
        update_data["sample_script"] = sample_script

    collection.update_one(
        {"_id": equation_id},
        {"$set": update_data}
    )

    return len(variables), script_generated


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

    # Log session start
    llm_logger.info("=" * 60)
    llm_logger.info(f"NEW SESSION STARTED: {datetime.now().isoformat()}")
    llm_logger.info(f"Model: {OLLAMA_MODEL}")
    llm_logger.info("=" * 60)
    print(f"LLM interactions logged to: {LLM_LOG_FILE}")

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
    total_scripts = 0

    for i, doc in enumerate(equations, 1):
        if args.limit:
            print(f"\n[{i}/{args.limit}]")
        else:
            print(f"\n[{i}/{len(equations)}]")
        variables_count, script_generated = process_equation(doc, collection)
        total_variables += variables_count
        if script_generated:
            total_scripts += 1

    print(f"\n{'='*50}")
    print(f"Complete!")
    print(f"  Equations processed: {len(equations)}")
    print(f"  Total variables extracted: {total_variables}")
    print(f"  Sample scripts generated: {total_scripts}")


if __name__ == "__main__":
    main()
