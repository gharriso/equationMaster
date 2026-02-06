#!/usr/bin/env python3
"""
EquationMaster - Streamlit UI for browsing cosmology equations.
"""

import logging
import sys

import streamlit as st
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError

from config import MONGODB_URI, DATABASE_NAME, COLLECTION_NAME


# Greek letter names for conversion
GREEK_LETTERS = {
    'alpha', 'beta', 'gamma', 'delta', 'epsilon', 'zeta', 'eta', 'theta',
    'iota', 'kappa', 'lambda', 'mu', 'nu', 'xi', 'pi', 'rho', 'sigma',
    'tau', 'upsilon', 'phi', 'chi', 'psi', 'omega',
    'Gamma', 'Delta', 'Theta', 'Lambda', 'Xi', 'Pi', 'Sigma', 'Phi', 'Psi', 'Omega'
}


def symbol_to_latex(symbol: str) -> str:
    """
    Convert a stored symbol string to proper LaTeX notation.
    Examples:
        'T' -> 'T'
        'n_e' -> 'n_{e}'
        'sigma' -> '\\sigma'
        'k_B' -> 'k_{B}'
        'rho' -> '\\rho'
        'Lambda_cool' -> '\\Lambda_{cool}'
    """
    # Check if it has a subscript
    if '_' in symbol:
        parts = symbol.split('_', 1)
        base = parts[0]
        subscript = parts[1]

        # Convert base if it's a Greek letter
        if base.lower() in {g.lower() for g in GREEK_LETTERS}:
            # Find the correct case version
            for g in GREEK_LETTERS:
                if g.lower() == base.lower():
                    base = f'\\{g}'
                    break

        # Format subscript
        return f'{base}_{{{subscript}}}'
    else:
        # No subscript - check if it's a Greek letter
        if symbol.lower() in {g.lower() for g in GREEK_LETTERS}:
            for g in GREEK_LETTERS:
                if g.lower() == symbol.lower():
                    return f'\\{g}'
        return symbol


# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)
logger = logging.getLogger(__name__)


# Page config
st.set_page_config(
    page_title="EquationMaster",
    page_icon="🔭",
    layout="wide"
)


@st.cache_resource
def get_mongo_client():
    """Get cached MongoDB client."""
    logger.info(f"Creating MongoDB client for {MONGODB_URI}")
    # Add timeouts to prevent hanging
    client = MongoClient(
        MONGODB_URI,
        serverSelectionTimeoutMS=5000,
        connectTimeoutMS=5000,
        socketTimeoutMS=10000
    )
    # Test connection
    try:
        client.admin.command('ping')
        logger.info("MongoDB connection successful")
    except ConnectionFailure as e:
        logger.error(f"MongoDB connection failed: {e}")
        raise
    return client


def get_collection():
    """Get the equations collection."""
    logger.debug("Getting collection")
    client = get_mongo_client()
    return client[DATABASE_NAME][COLLECTION_NAME]


@st.cache_data(ttl=60)
def load_all_data():
    """
    Load all equations from MongoDB in a single query.
    Returns a dict with all data and precomputed indexes.
    """
    logger.info("Loading all data from MongoDB (single query)")
    try:
        collection = get_collection()
        equations = list(collection.find({}))
        logger.info(f"Loaded {len(equations)} equations")

        # Build indexes in memory
        chapters = set()
        symbols = set()
        by_id = {}
        by_symbol = {}  # symbol -> list of equations

        for eq in equations:
            # Index by ID
            by_id[eq['_id']] = eq

            # Collect chapters
            if eq.get('chapter'):
                chapters.add(eq['chapter'])

            # Collect symbols and build symbol index
            for var in eq.get('variables', []):
                symbol = var.get('symbol')
                if symbol:
                    symbols.add(symbol)
                    if symbol not in by_symbol:
                        by_symbol[symbol] = []
                    by_symbol[symbol].append({
                        'equation': eq,
                        'lhs': var.get('lhs', False)
                    })

        # Sort chapters numerically
        sorted_chapters = sorted(chapters, key=lambda x: int(x.replace("chapter", "") or "0"))
        sorted_symbols = sorted(symbols)

        logger.info(f"Indexed: {len(sorted_chapters)} chapters, {len(sorted_symbols)} symbols")

        return {
            'equations': equations,
            'chapters': sorted_chapters,
            'symbols': sorted_symbols,
            'by_id': by_id,
            'by_symbol': by_symbol
        }
    except Exception as e:
        logger.error(f"Error loading data: {e}", exc_info=True)
        return {
            'equations': [],
            'chapters': [],
            'symbols': [],
            'by_id': {},
            'by_symbol': {}
        }


def get_all_chapters():
    """Get list of all chapters (from cached data)."""
    return load_all_data()['chapters']


def get_all_symbols():
    """Get list of all variable symbols (from cached data)."""
    return load_all_data()['symbols']


def get_equations(chapter_filter=None, symbol_filter=None, search_text=None):
    """Get equations with optional filters (from cached data)."""
    data = load_all_data()
    equations = data['equations']

    # Apply filters in memory
    if chapter_filter and chapter_filter != "All":
        equations = [e for e in equations if e.get('chapter') == chapter_filter]

    if symbol_filter and symbol_filter != "All":
        equations = [e for e in equations if any(
            v.get('symbol') == symbol_filter for v in e.get('variables', [])
        )]

    if search_text:
        search_lower = search_text.lower()
        equations = [e for e in equations if (
            search_lower in (e.get('name') or '').lower() or
            search_lower in (e.get('description') or '').lower() or
            search_lower in (e.get('reference') or '').lower()
        )]

    # Sort by chapter and reference
    equations = sorted(equations, key=lambda e: (e.get('chapter', ''), e.get('reference', '')))
    return equations


def get_equation_by_id(equation_id):
    """Get a single equation by ID (from cached data)."""
    return load_all_data()['by_id'].get(equation_id)


def get_equations_by_symbol(symbol, sort_lhs_first=True):
    """
    Get all equations that use a specific variable symbol.
    Returns list of dicts with 'equation' and 'lhs' keys.
    If sort_lhs_first=True, equations where the variable is on LHS are sorted first.
    """
    data = load_all_data()
    results = data['by_symbol'].get(symbol, [])

    if sort_lhs_first:
        # Sort: LHS first, then by chapter and reference
        results = sorted(results, key=lambda r: (
            not r['lhs'],  # False (LHS) sorts before True (not LHS)
            r['equation'].get('chapter', ''),
            r['equation'].get('reference', '')
        ))

    return results


def render_equation_card(eq, show_details_link=True):
    """Render an equation as a card."""
    col1, col2 = st.columns([3, 1])

    with col1:
        # Reference and name
        title = f"**{eq.get('reference', 'Unknown')}**"
        if eq.get('name'):
            title += f" - {eq['name']}"
        st.markdown(title)

        # Rendered LaTeX
        if eq.get('latex'):
            st.latex(eq['latex'])

        # Description
        if eq.get('description'):
            st.caption(eq['description'])

    with col2:
        st.caption(f"Chapter: {eq.get('chapter', 'Unknown').replace('chapter', '')}")
        if eq.get('variables'):
            # Render variable symbols as LaTeX
            symbols_latex = [symbol_to_latex(v['symbol']) for v in eq['variables'][:5]]
            symbols_display = ', '.join([f"${s}$" for s in symbols_latex])
            more = '...' if len(eq['variables']) > 5 else ''
            st.markdown(f"Variables: {symbols_display}{more}", unsafe_allow_html=True)

        if show_details_link:
            if st.button("View Details", key=f"btn_{eq['_id']}"):
                st.session_state.selected_equation = eq['_id']
                st.session_state.page = "detail"
                st.rerun()


def render_equation_detail(eq):
    """Render detailed view of an equation."""
    # Back button
    if st.button("← Back to List"):
        st.session_state.page = "list"
        st.rerun()

    st.divider()

    # Header
    st.header(f"{eq.get('reference', 'Unknown')}")
    if eq.get('name'):
        st.subheader(eq['name'])

    # Main equation display
    st.markdown("### Equation")
    if eq.get('latex'):
        st.latex(eq['latex'])

    # Description
    if eq.get('description'):
        st.markdown("### Description")
        st.write(eq['description'])

    # Metadata
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Metadata")
        st.write(f"**Chapter:** {eq.get('chapter', 'Unknown').replace('chapter', 'Chapter ')}")
        st.write(f"**ID:** `{eq.get('_id')}`")

    with col2:
        st.markdown("### Raw LaTeX")
        st.code(eq.get('latex', ''), language='latex')

    # Variables section
    if eq.get('variables'):
        st.markdown("### Variables")

        for var in eq['variables']:
            # Build expander title with LHS indicator
            lhs_badge = " [LHS]" if var.get('lhs') else ""
            symbol_latex = symbol_to_latex(var['symbol'])
            expander_title = f"${symbol_latex}$ — {var.get('name', 'Unknown')}{lhs_badge}"

            with st.expander(expander_title, expanded=False):
                col1, col2 = st.columns(2)

                with col1:
                    # Render symbol as LaTeX
                    st.markdown("**Symbol:**")
                    st.latex(symbol_latex)
                    st.write(f"**Name:** {var.get('name', 'N/A')}")
                    st.write(f"**Description:** {var.get('description', 'N/A')}")
                    st.write(f"**On LHS:** {'Yes' if var.get('lhs') else 'No'}")

                with col2:
                    st.write(f"**Astropy Unit:** `{var.get('astropy_unit') or 'N/A'}`")
                    st.write(f"**Astropy Constant:** `{var.get('astropy_constant') or 'N/A'}`")

                # Find other equations with this variable (sorted by LHS first)
                related = get_equations_by_symbol(var['symbol'], sort_lhs_first=True)
                # Filter out current equation
                related = [r for r in related if r['equation']['_id'] != eq['_id']]

                if related:
                    st.markdown(f"**Also appears in ({len(related)} equations):**")
                    for item in related[:10]:  # Limit to 10
                        other_eq = item['equation']
                        is_lhs = item['lhs']
                        ref = other_eq.get('reference', 'Unknown')
                        name = other_eq.get('name', '')
                        lhs_indicator = " [LHS]" if is_lhs else ""
                        display = f"{ref}{lhs_indicator}"
                        if name:
                            display += f" - {name}"

                        if st.button(display, key=f"link_{eq['_id']}_{var['symbol']}_{other_eq['_id']}"):
                            st.session_state.selected_equation = other_eq['_id']
                            st.rerun()

                    if len(related) > 10:
                        st.caption(f"... and {len(related) - 10} more")

    # Sample Python script section
    if eq.get('sample_script'):
        st.markdown("### Sample Python Script")
        st.caption("Auto-generated script demonstrating sympy representation with astropy units")

        # Initialize script in session state if not present or if equation changed
        script_key = f"script_{eq['_id']}"
        if script_key not in st.session_state:
            st.session_state[script_key] = eq['sample_script']

        # Toggle between view and edit mode
        col1, col2, col3, col4 = st.columns([1, 1, 1, 3])
        with col1:
            edit_mode = st.toggle("Edit", key=f"edit_toggle_{eq['_id']}")
        with col2:
            if st.button("📋 Copy", key=f"copy_{eq['_id']}"):
                st.session_state[f"copied_{eq['_id']}"] = True
        with col3:
            if st.button("Reset", key=f"reset_{eq['_id']}"):
                st.session_state[script_key] = eq['sample_script']
                st.rerun()

        # Handle copy to clipboard using JavaScript
        if st.session_state.get(f"copied_{eq['_id']}"):
            # Use st.components to inject JavaScript for clipboard
            import streamlit.components.v1 as components
            script_escaped = st.session_state[script_key].replace('\\', '\\\\').replace('`', '\\`').replace('$', '\\$')
            components.html(
                f"""
                <script>
                    navigator.clipboard.writeText(`{script_escaped}`).then(function() {{
                        window.parent.postMessage({{type: 'streamlit:toast', message: 'Copied to clipboard!'}}, '*');
                    }});
                </script>
                <p style="color: green; font-size: 14px;">✓ Copied to clipboard!</p>
                """,
                height=30
            )
            st.session_state[f"copied_{eq['_id']}"] = False

        if edit_mode:
            # Editable text area
            edited_script = st.text_area(
                "Edit script:",
                value=st.session_state[script_key],
                height=400,
                key=f"editor_{eq['_id']}",
                label_visibility="collapsed"
            )
            st.session_state[script_key] = edited_script
        else:
            # Syntax-highlighted view
            st.code(st.session_state[script_key], language='python')

        # Execute button and output
        if st.button("▶ Run Script", key=f"run_{eq['_id']}", type="primary"):
            st.markdown("#### Output")

            import io
            import contextlib

            # Capture stdout and stderr
            stdout_capture = io.StringIO()
            stderr_capture = io.StringIO()

            try:
                with contextlib.redirect_stdout(stdout_capture), contextlib.redirect_stderr(stderr_capture):
                    # Create a safe namespace for execution
                    exec_globals = {"__builtins__": __builtins__}
                    exec(st.session_state[script_key], exec_globals)

                stdout_output = stdout_capture.getvalue()
                stderr_output = stderr_capture.getvalue()

                if stdout_output:
                    st.code(stdout_output, language='text')
                if stderr_output:
                    st.warning(stderr_output)
                if not stdout_output and not stderr_output:
                    st.info("Script executed successfully (no output)")

            except Exception as e:
                st.error(f"Error: {type(e).__name__}: {e}")


def init_session_state():
    """Initialize session state with data loaded once."""
    if 'initialized' not in st.session_state:
        logger.info("Initializing session state (one-time data load)")
        data = load_all_data()
        st.session_state.chapters = ["All"] + data['chapters']
        st.session_state.symbols = ["All"] + data['symbols']
        st.session_state.page = "list"
        st.session_state.selected_equation = None
        st.session_state.initialized = True
        logger.info(f"Session initialized: {len(data['chapters'])} chapters, {len(data['symbols'])} symbols")


def main():
    """Main app entry point."""
    # Initialize session state once
    init_session_state()

    # Title
    st.title("🔭 EquationMaster")
    st.caption("Browse and explore cosmology equations from S385")

    # Route to appropriate page
    if st.session_state.page == "detail" and st.session_state.selected_equation:
        eq = get_equation_by_id(st.session_state.selected_equation)
        if eq:
            render_equation_detail(eq)
        else:
            st.error("Equation not found")
            st.session_state.page = "list"
            st.rerun()
    else:
        # List view with filters
        st.sidebar.header("Filters")

        # Chapter filter (from session state - built once)
        chapter_filter = st.sidebar.selectbox("Chapter", st.session_state.chapters)

        # Variable symbol filter (from session state - built once)
        symbol_filter = st.sidebar.selectbox("Variable Symbol", st.session_state.symbols)

        # Text search
        search_text = st.sidebar.text_input("Search", placeholder="Search name, description...")

        # Get filtered equations
        equations = get_equations(
            chapter_filter=chapter_filter if chapter_filter != "All" else None,
            symbol_filter=symbol_filter if symbol_filter != "All" else None,
            search_text=search_text if search_text else None
        )

        # Sort options
        sort_options = {
            "Chapter & Reference": lambda e: (e.get('chapter', ''), e.get('reference', '')),
            "Name": lambda e: e.get('name', '') or 'zzz',
            "Chapter": lambda e: e.get('chapter', '')
        }
        sort_by = st.sidebar.selectbox("Sort by", list(sort_options.keys()))
        equations = sorted(equations, key=sort_options[sort_by])

        # Display count
        st.sidebar.divider()
        st.sidebar.metric("Equations", len(equations))

        # Refresh button to reload data from MongoDB
        if st.sidebar.button("↻ Refresh Data"):
            load_all_data.clear()  # Clear the cache
            st.session_state.initialized = False  # Force re-initialization
            st.rerun()

        # Display equations
        if not equations:
            st.info("No equations found matching your filters.")
        else:
            for eq in equations:
                with st.container():
                    render_equation_card(eq)
                    st.divider()


if __name__ == "__main__":
    main()
