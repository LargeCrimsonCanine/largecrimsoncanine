"""Pytest configuration and fixtures."""

import sys
from pathlib import Path

# Add lcc package to Python path for visualization tests
repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))
