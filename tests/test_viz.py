"""Tests for lcc.viz ganja.js visualization module.

These tests verify the structure and basic functionality without requiring
a Jupyter environment or browser rendering.
"""

import pytest
from typing import List


class MockMultivector:
    """Mock multivector for testing without the Rust backend."""

    def __init__(self, coeffs: List[float]):
        self._coeffs = coeffs

    def coefficients(self) -> List[float]:
        return self._coeffs

    def __iter__(self):
        return iter(self._coeffs)


class TestImports:
    """Test that module imports work correctly."""

    def test_ganja_import(self):
        """Test importing from lcc.viz."""
        from lcc.viz import show, Graph
        assert callable(show)
        assert Graph is not None

    def test_lcc_import(self):
        """Test importing from top-level lcc."""
        from lcc import show, Graph
        assert callable(show)
        assert Graph is not None

    def test_ganja_module_direct(self):
        """Test importing the ganja module directly."""
        from lcc.viz import ganja
        assert hasattr(ganja, 'show')
        assert hasattr(ganja, 'Graph')
        assert hasattr(ganja, 'ALGEBRA_SIGNATURES')
        assert hasattr(ganja, 'GANJA_CDN')


class TestAlgebraSignatures:
    """Test algebra signature definitions."""

    def test_supported_algebras(self):
        """Test that expected algebras are defined."""
        from lcc.viz.ganja import ALGEBRA_SIGNATURES

        expected = ["R2", "R3", "R4", "PGA2D", "PGA3D", "CGA2D", "CGA3D", "STA"]
        for alg in expected:
            assert alg in ALGEBRA_SIGNATURES, f"Missing algebra: {alg}"

    def test_signature_format(self):
        """Test that signatures are (p, q, r) tuples."""
        from lcc.viz.ganja import ALGEBRA_SIGNATURES

        for name, sig in ALGEBRA_SIGNATURES.items():
            assert isinstance(sig, tuple), f"{name} signature is not a tuple"
            assert len(sig) == 3, f"{name} signature should have 3 elements"
            assert all(isinstance(x, int) for x in sig), f"{name} signature elements must be ints"

    def test_pga3d_signature(self):
        """Test PGA3D has correct signature (3,0,1)."""
        from lcc.viz.ganja import ALGEBRA_SIGNATURES
        assert ALGEBRA_SIGNATURES["PGA3D"] == (3, 0, 1)


class TestColorConversion:
    """Test color conversion utilities."""

    def test_normalize_color_hex(self):
        """Test hex colors pass through."""
        from lcc.viz.ganja import _normalize_color

        assert _normalize_color("#ff0000") == "#ff0000"
        assert _normalize_color("#abc") == "#abc"

    def test_normalize_color_names(self):
        """Test named colors convert to hex."""
        from lcc.viz.ganja import _normalize_color

        assert _normalize_color("red") == "#e74c3c"
        assert _normalize_color("blue") == "#3498db"
        assert _normalize_color("RED") == "#e74c3c"  # Case insensitive

    def test_color_to_int(self):
        """Test hex to integer conversion."""
        from lcc.viz.ganja import _color_to_int

        assert _color_to_int("#ff0000") == 0xff0000
        assert _color_to_int("0xff0000") == 0xff0000
        assert _color_to_int("red") == 0xe74c3c


class TestMultivectorConversion:
    """Test multivector to ganja.js conversion."""

    def test_mv_to_ganja_with_coefficients_method(self):
        """Test conversion using coefficients() method."""
        from lcc.viz.ganja import _mv_to_ganja

        mv = MockMultivector([1.0, 2.0, 3.0, 4.0])
        result = _mv_to_ganja(mv, "R2")

        assert result == [1.0, 2.0, 3.0, 4.0]

    def test_mv_to_ganja_with_iterable(self):
        """Test conversion using __iter__."""
        from lcc.viz.ganja import _mv_to_ganja

        # Use a plain list (which has __iter__ but not coefficients)
        mv = [1.0, 2.0, 3.0]
        result = _mv_to_ganja(mv, "R2")

        assert result == [1.0, 2.0, 3.0]


class TestGraphBuilder:
    """Test the Graph builder class."""

    def test_graph_init(self):
        """Test Graph initialization."""
        from lcc.viz import Graph

        g = Graph(algebra="PGA3D")
        assert g.algebra == "PGA3D"
        assert g.elements == []
        assert len(g) == 0

    def test_graph_init_invalid_algebra(self):
        """Test Graph raises on invalid algebra."""
        from lcc.viz import Graph

        with pytest.raises(ValueError, match="Unknown algebra"):
            Graph(algebra="INVALID")

    def test_graph_add_element(self):
        """Test adding elements to graph."""
        from lcc.viz import Graph

        mv = MockMultivector([1.0, 0.0, 0.0, 0.0])
        g = Graph(algebra="R2")
        g.add(mv, color="red", label="test")

        assert len(g) == 1
        assert g.elements[0]["color"] == "red"
        assert g.elements[0]["label"] == "test"
        assert g.elements[0]["mv"] is mv

    def test_graph_method_chaining(self):
        """Test that add() returns self for chaining."""
        from lcc.viz import Graph

        mv1 = MockMultivector([1.0, 0.0])
        mv2 = MockMultivector([0.0, 1.0])

        g = Graph(algebra="R2")
        result = g.add(mv1, color="red").add(mv2, color="blue")

        assert result is g
        assert len(g) == 2

    def test_graph_add_all(self):
        """Test adding multiple elements at once."""
        from lcc.viz import Graph

        mvs = [MockMultivector([float(i)]) for i in range(3)]

        g = Graph(algebra="R2")
        g.add_all(mvs, colors=["red", "blue"], labels=["a", "b", "c"])

        assert len(g) == 3
        # Colors should cycle
        assert g.elements[0]["color"] == "red"
        assert g.elements[1]["color"] == "blue"
        assert g.elements[2]["color"] == "red"

    def test_graph_clear(self):
        """Test clearing the graph."""
        from lcc.viz import Graph

        g = Graph(algebra="R2")
        g.add(MockMultivector([1.0]), color="red")
        g.add(MockMultivector([2.0]), color="blue")

        assert len(g) == 2
        g.clear()
        assert len(g) == 0

    def test_graph_repr(self):
        """Test string representation."""
        from lcc.viz import Graph

        g = Graph(algebra="PGA3D")
        g.add(MockMultivector([1.0]), color="red")
        g.add(MockMultivector([2.0]), color="blue")

        assert repr(g) == "Graph(algebra='PGA3D', elements=2)"


class TestHtmlGeneration:
    """Test HTML generation for ganja.js."""

    def test_generate_html_invalid_algebra(self):
        """Test that invalid algebra raises error."""
        from lcc.viz.ganja import _generate_html

        with pytest.raises(ValueError, match="Unknown algebra"):
            _generate_html([], "INVALID")

    def test_generate_html_structure(self):
        """Test basic HTML structure."""
        from lcc.viz.ganja import _generate_html

        elements = [{"coeffs": [1.0, 0.0], "color": "red", "label": "P1"}]
        html = _generate_html(elements, "R2", width=400, height=300)

        assert "<script" in html
        assert GANJA_CDN_CHECK in html
        assert "400px" in html or "400" in html
        assert "300px" in html or "300" in html

    def test_generate_html_contains_ganja_algebra(self):
        """Test that HTML contains correct algebra initialization."""
        from lcc.viz.ganja import _generate_html

        html = _generate_html([], "PGA3D")
        # PGA3D is (3, 0, 1)
        assert "Algebra(3, 0, 1" in html


GANJA_CDN_CHECK = "enkimute.github.io/ganja.js"


class TestShowFunction:
    """Test the show() function."""

    def test_show_single_element(self):
        """Test show with single element returns HTML when not in IPython."""
        from lcc.viz.ganja import show, HAS_IPYTHON

        if HAS_IPYTHON:
            pytest.skip("Test only valid when IPython is not available")

        mv = MockMultivector([1.0, 0.0, 0.0, 0.0])
        html = show(mv, algebra="R2")

        assert html is not None
        assert "<script" in html

    def test_show_multiple_elements(self):
        """Test show with multiple elements."""
        from lcc.viz.ganja import show, HAS_IPYTHON

        if HAS_IPYTHON:
            pytest.skip("Test only valid when IPython is not available")

        mvs = [MockMultivector([float(i), 0.0]) for i in range(3)]
        html = show(mvs, algebra="R2", colors=["red", "green", "blue"])

        assert html is not None
        assert "0xe74c3c" in html  # red
        assert "0x2ecc71" in html  # green
        assert "0x3498db" in html  # blue

    def test_show_color_cycling(self):
        """Test that colors cycle when fewer colors than elements."""
        from lcc.viz.ganja import show, HAS_IPYTHON

        if HAS_IPYTHON:
            pytest.skip("Test only valid when IPython is not available")

        mvs = [MockMultivector([float(i)]) for i in range(4)]
        html = show(mvs, algebra="R2", colors=["red", "blue"])

        # Should have 2 reds and 2 blues
        assert html is not None
        assert html.count("0xe74c3c") == 2  # red appears twice
        assert html.count("0x3498db") == 2  # blue appears twice
