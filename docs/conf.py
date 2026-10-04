project = "samplax"
author = "Maxwell Bolt"
release = "0.1.0"

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

# ift_sde.md documents samplax.integrations, which is internal (see README)
exclude_patterns = ["_build", "ift_sde.md"]
