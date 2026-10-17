import os
import sys

# Resolve the package from this checkout, not whatever tree an editable
# install points at, so the docs describe (and version) the code beside them.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from samplax import __version__  # noqa: E402

project = "samplax"
author = "Maxwell Bolt"
copyright = "2026, Maxwell Bolt. Apache License 2.0"
release = version = __version__

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "myst_parser",
]

autodoc_member_order = "bysource"
autodoc_typehints = "description"

html_theme = "furo"
html_title = "samplax"

myst_enable_extensions = ["dollarmath"]

exclude_patterns = ["_build"]
