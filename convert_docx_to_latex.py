import docx
import json
import re

def to_latex(formula):
    """
    Basic mapping of text-based symbols to LaTeX commands.
    You can expand this as you find new symbols in your studies.
    """
    replacements = {
        'γ': r'\gamma', 'μ': r'\mu', 'ν': r'\nu', 'ρ': r'\rho',
        'Λ': r'\Lambda', 'π': r'\pi', 'θ': r'\theta', 'ϕ': r'\phi',
        'φ': r'\phi', 'Δ': r'\Delta', 'σ': r'\sigma', 'Ω': r'\Omega',
        'Γ': r'\Gamma', 'ϵ': r'\epsilon', 'τ': r'\tau', 'η': r'\eta',
        '∝': r'\propto', '≈': r'\approx', '≡': r'\equiv', '⇌': r'\rightleftharpoons',
        'ℏ': r'\hbar', 'Σ': r'\Sigma', '∞': r'\infty'
    }
    for char, tex in replacements.items():
        formula = formula.replace(char, tex)
    
    # Simple regex for powers (e.g., a2 -> a^2)
    formula = re.sub(r'([a-zA-Z0-9)])(\d)', r'\1^\2', formula)
    
    # Handle the common 'aa' notation in Friedmann equations
    if "aa2" in formula:
        formula = formula.replace("aa2", r"\left(\frac{\dot{a}}{a}\right)^2")
    elif "aa" in formula:
        formula = formula.replace("aa", r"\frac{\ddot{a}}{a}")
        
    return formula

def extract_equations(file_path):
    doc = docx.Document(file_path)
    equations = []
    seen_refs = set()

    for table in doc.tables:
        # Skip header rows if they exist
        for row in table.rows:
            cells = [cell.text.strip() for cell in row.cells]
            
            # Identify columns based on content
            if "Formula Name" in cells[0] or "Name" in cells[0] or "Equation" in cells[0]:
                continue
                
            if len(cells) >= 3:
                name = cells[0]
                desc = re.sub(r'[\d\s+,.]+$', '', cells[1]).strip() # Clean citations
                
                # Logic to find formula and reference (Reference is usually in the last cell)
                ref = cells[-1]
                
                # Deduplicate by reference code (e.g., (4.34))
                if ref in seen_refs:
                    continue
                
                # Formula is usually the 3rd or 4th cell
                formula_raw = cells[2] if not cells[2].startswith('+') else cells[3]
                
                equations.append({
                    "name": name,
                    "description": desc,
                    "latex": to_latex(formula_raw),
                    "reference": ref
                })
                seen_refs.add(ref)

    return equations

# Run the extraction
eq_list = extract_equations('Summary of Equations.docx')

# Save to JSON
with open('s385_equations.json', 'w', encoding='utf-8') as f:
    json.dump(eq_list, f, indent=4, ensure_ascii=False)

print(f"Successfully extracted {len(eq_list)} unique equations to s385_equations.json")