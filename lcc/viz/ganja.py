"""ganja.js visualization for geometric algebra.

This module provides Jupyter notebook integration with ganja.js, the standard
visualization library for geometric algebra. It handles conversion between
LargeCrimsonCanine's Multivector format and ganja.js's internal representation.

Example usage in Jupyter:
```python
from largecrimsoncanine import Algebra, Multivector
from lcc.viz import show, Graph

# Create a 3D PGA algebra
PGA = Algebra.pga(3)

# Create some geometric elements
p1 = Multivector.pga_point(PGA, 1, 0, 0)
p2 = Multivector.pga_point(PGA, 0, 1, 0)
line = p1 & p2  # Join operation

# Quick display
show([p1, p2, line], algebra="PGA3D")

# Or build incrementally with full control
g = Graph(algebra="PGA3D")
g.add(p1, color="red", label="P1")
g.add(p2, color="blue", label="P2")
g.add(line, color="green")
g.show()
```

Supported algebras:
- R2, R3: Vanilla geometric algebras
- PGA2D, PGA3D: Projective geometric algebra (most common for graphics)
- CGA2D, CGA3D: Conformal geometric algebra

For more on ganja.js: https://enkimute.github.io/ganja.js/
"""

import json
import uuid
from typing import List, Optional, Union, Any, Dict

try:
    from IPython.display import HTML, display
    HAS_IPYTHON = True
except ImportError:
    HAS_IPYTHON = False

GANJA_CDN = "https://enkimute.github.io/ganja.js/ganja.js"

# Algebra signatures for ganja.js
# Format: (p, q, r) where p=positive, q=negative, r=zero signature dimensions
ALGEBRA_SIGNATURES: Dict[str, tuple] = {
    "R2": (2, 0, 0),
    "R3": (3, 0, 0),
    "R4": (4, 0, 0),
    "PGA2D": (2, 0, 1),
    "PGA3D": (3, 0, 1),
    "CGA2D": (3, 1, 0),
    "CGA3D": (4, 1, 0),
    "STA": (1, 3, 0),  # Spacetime algebra
}

# CSS color name to hex mapping for common colors
COLOR_MAP: Dict[str, str] = {
    "red": "#e74c3c",
    "blue": "#3498db",
    "green": "#2ecc71",
    "yellow": "#f1c40f",
    "orange": "#e67e22",
    "purple": "#9b59b6",
    "pink": "#e91e63",
    "cyan": "#00bcd4",
    "white": "#ffffff",
    "black": "#000000",
    "gray": "#95a5a6",
    "grey": "#95a5a6",
}


def _normalize_color(color: str) -> str:
    """Convert color name to hex if needed."""
    if color.startswith("#") or color.startswith("0x"):
        return color
    return COLOR_MAP.get(color.lower(), color)


def _color_to_int(color: str) -> int:
    """Convert hex color string to integer for ganja.js."""
    color = _normalize_color(color)
    if color.startswith("#"):
        return int(color[1:], 16)
    elif color.startswith("0x"):
        return int(color, 16)
    # Try parsing as hex anyway
    try:
        return int(color, 16)
    except ValueError:
        return 0x3498db  # Default blue


def _mv_to_ganja(mv: Any, algebra: str) -> List[float]:
    """Convert Multivector to ganja.js coefficient array.

    ganja.js expects coefficients in a specific basis order that depends
    on the algebra. This function handles the conversion.

    Args:
        mv: A Multivector object from largecrimsoncanine
        algebra: The algebra name (e.g., "PGA3D")

    Returns:
        List of coefficients in ganja.js order
    """
    # Get coefficients from the multivector
    # The Multivector class should have a coefficients() method
    if hasattr(mv, 'coefficients'):
        coeffs = list(mv.coefficients())
    elif hasattr(mv, 'to_list'):
        coeffs = mv.to_list()
    elif hasattr(mv, '__iter__'):
        coeffs = list(mv)
    else:
        raise TypeError(f"Cannot extract coefficients from {type(mv)}")

    return coeffs


def _generate_html(elements: List[dict], algebra: str,
                   width: int = 600, height: int = 400,
                   background: str = "#ffffff",
                   grid: bool = True,
                   labels: bool = True,
                   animate: bool = False) -> str:
    """Generate HTML with embedded ganja.js visualization.

    Args:
        elements: List of element dicts with 'coeffs', 'color', 'label' keys
        algebra: Algebra name for ganja.js
        width: Canvas width in pixels
        height: Canvas height in pixels
        background: Background color
        grid: Whether to show grid lines
        labels: Whether to show labels
        animate: Whether to enable animation

    Returns:
        HTML string with embedded JavaScript
    """
    if algebra not in ALGEBRA_SIGNATURES:
        raise ValueError(f"Unknown algebra '{algebra}'. "
                        f"Supported: {list(ALGEBRA_SIGNATURES.keys())}")

    sig = ALGEBRA_SIGNATURES[algebra]

    # Generate unique ID for this visualization
    viz_id = f"ganja_{uuid.uuid4().hex[:8]}"

    # Build the ganja.js graph function body
    graph_items = []
    for elem in elements:
        coeffs = elem["coeffs"]
        color = _color_to_int(elem.get("color", "blue"))
        label = elem.get("label")

        # Format coefficients as array literal
        coeffs_str = json.dumps(coeffs)

        # Add color as first element (ganja.js convention)
        item = f"0x{color:06x}"
        graph_items.append(item)

        # Add the multivector
        if label and labels:
            graph_items.append(f'"{label}"')
        graph_items.append(f"new Element({coeffs_str})")

    graph_content = ",\n            ".join(graph_items)

    # Grid option
    grid_opt = "true" if grid else "false"

    # Build the HTML
    html = f'''
<div id="{viz_id}" style="width:{width}px;height:{height}px;"></div>
<script src="{GANJA_CDN}"></script>
<script>
(function() {{
    var Algebra = window.Algebra;
    if (!Algebra) {{
        console.error("ganja.js not loaded");
        return;
    }}

    Algebra({sig[0]}, {sig[1]}, {sig[2]}, function() {{
        var Element = this.Element;

        var graph = this.graph(function() {{
            return [
            {graph_content}
            ];
        }}, {{
            grid: {grid_opt},
            width: "{width}px",
            height: "{height}px",
            lineWidth: 3
        }});

        document.getElementById("{viz_id}").appendChild(graph);
    }});
}})();
</script>
'''
    return html


def show(elements: Union[Any, List[Any]],
         algebra: str = "R3",
         width: int = 600,
         height: int = 400,
         colors: Optional[Union[str, List[str]]] = None,
         labels: Optional[Union[str, List[str]]] = None,
         background: str = "#ffffff",
         grid: bool = True,
         **kwargs) -> Optional[str]:
    """Show multivectors in Jupyter notebook using ganja.js.

    Args:
        elements: Single multivector or list of multivectors to display
        algebra: Algebra name ("R2", "R3", "PGA2D", "PGA3D", "CGA2D", "CGA3D")
        width: Canvas width in pixels
        height: Canvas height in pixels
        colors: Color(s) for elements - single string or list
        labels: Label(s) for elements - single string or list
        background: Background color
        grid: Whether to show grid lines
        **kwargs: Additional options passed to ganja.js

    Returns:
        HTML string if not in Jupyter, None if displayed in Jupyter

    Example:
        >>> show([p1, p2, line], algebra="PGA3D", colors=["red", "blue", "green"])
    """
    # Normalize to list
    if not isinstance(elements, list):
        elements = [elements]

    # Normalize colors
    if colors is None:
        colors = ["blue"] * len(elements)
    elif isinstance(colors, str):
        colors = [colors] * len(elements)
    elif len(colors) < len(elements):
        # Extend colors by cycling
        colors = (colors * (len(elements) // len(colors) + 1))[:len(elements)]

    # Normalize labels
    if labels is None:
        labels = [None] * len(elements)
    elif isinstance(labels, str):
        labels = [labels] if len(elements) == 1 else [f"{labels}_{i}" for i in range(len(elements))]
    elif len(labels) < len(elements):
        labels = labels + [None] * (len(elements) - len(labels))

    # Convert to element dicts
    elem_dicts = []
    for mv, color, label in zip(elements, colors, labels):
        elem_dicts.append({
            "coeffs": _mv_to_ganja(mv, algebra),
            "color": color,
            "label": label,
        })

    html = _generate_html(
        elem_dicts,
        algebra,
        width=width,
        height=height,
        background=background,
        grid=grid,
        labels=True,
        **kwargs
    )

    if HAS_IPYTHON:
        display(HTML(html))
        return None
    else:
        return html


class Graph:
    """Incremental graph builder for ganja.js visualizations.

    Allows building up a visualization element by element with full control
    over colors, labels, and styling.

    Example:
        >>> g = Graph(algebra="PGA3D")
        >>> g.add(point1, color="red", label="Origin")
        >>> g.add(point2, color="blue", label="Target")
        >>> g.add(line, color="green")
        >>> g.show()
    """

    def __init__(self, algebra: str = "R3",
                 width: int = 600,
                 height: int = 400,
                 background: str = "#ffffff",
                 grid: bool = True):
        """Initialize a new Graph builder.

        Args:
            algebra: Algebra name for ganja.js
            width: Default canvas width
            height: Default canvas height
            background: Background color
            grid: Whether to show grid lines
        """
        if algebra not in ALGEBRA_SIGNATURES:
            raise ValueError(f"Unknown algebra '{algebra}'. "
                           f"Supported: {list(ALGEBRA_SIGNATURES.keys())}")

        self.algebra = algebra
        self.width = width
        self.height = height
        self.background = background
        self.grid = grid
        self.elements: List[dict] = []

    def add(self, element: Any,
            color: str = "blue",
            label: Optional[str] = None,
            **kwargs) -> "Graph":
        """Add an element to the graph.

        Args:
            element: Multivector to add
            color: Color name or hex string
            label: Optional text label
            **kwargs: Additional styling options

        Returns:
            self for method chaining
        """
        self.elements.append({
            "mv": element,
            "color": color,
            "label": label,
            **kwargs
        })
        return self

    def add_all(self, elements: List[Any],
                colors: Optional[List[str]] = None,
                labels: Optional[List[str]] = None) -> "Graph":
        """Add multiple elements at once.

        Args:
            elements: List of multivectors
            colors: Optional list of colors (cycles if shorter)
            labels: Optional list of labels

        Returns:
            self for method chaining
        """
        if colors is None:
            colors = ["blue"]
        if labels is None:
            labels = [None] * len(elements)

        for i, elem in enumerate(elements):
            color = colors[i % len(colors)]
            label = labels[i] if i < len(labels) else None
            self.add(elem, color=color, label=label)

        return self

    def clear(self) -> "Graph":
        """Remove all elements from the graph.

        Returns:
            self for method chaining
        """
        self.elements = []
        return self

    def show(self, width: Optional[int] = None,
             height: Optional[int] = None) -> Optional[str]:
        """Display the graph in Jupyter notebook.

        Args:
            width: Override default width
            height: Override default height

        Returns:
            HTML string if not in Jupyter, None if displayed
        """
        if not self.elements:
            if HAS_IPYTHON:
                display(HTML("<p>Empty graph - no elements to display</p>"))
                return None
            return "<p>Empty graph - no elements to display</p>"

        # Convert stored elements to coefficient dicts
        elem_dicts = []
        for e in self.elements:
            elem_dicts.append({
                "coeffs": _mv_to_ganja(e["mv"], self.algebra),
                "color": e["color"],
                "label": e.get("label"),
            })

        html = _generate_html(
            elem_dicts,
            self.algebra,
            width=width or self.width,
            height=height or self.height,
            background=self.background,
            grid=self.grid,
            labels=True,
        )

        if HAS_IPYTHON:
            display(HTML(html))
            return None
        else:
            return html

    def _repr_html_(self) -> str:
        """IPython display hook for automatic rendering."""
        return self.show() or ""

    def __len__(self) -> int:
        """Return number of elements in the graph."""
        return len(self.elements)

    def __repr__(self) -> str:
        return f"Graph(algebra='{self.algebra}', elements={len(self.elements)})"
