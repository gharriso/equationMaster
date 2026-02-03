latex= "M_J = \\frac{9}{4}\\sqrt{\\frac{1}{2\\pi n}}\\frac{1}{\\langle m \\rangle^2}\\left(\\frac{k_B T}{G}\\right)^{3/2}" 
from sympy.parsing.latex import parse_latex


expr = parse_latex(latex)
display(expr)
 
variables = expr.free_symbols
print(variables)