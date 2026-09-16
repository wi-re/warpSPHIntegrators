# Sphinx + MyST configuration for the warpSPHIntegrators wiki.
#
# The docs are pure prose: they never import the package, so the build works
# in a venv without torch (just `pip install sphinx myst-parser`).

project = 'warpSPHIntegrators'
copyright = '2026, Rene Winchenbach'
author = 'Rene Winchenbach'
release = '0.5.0'

extensions = ['myst_parser']

myst_enable_extensions = [
    'dollarmath',   # $...$ and $$...$$
    'amsmath',      # \begin{array}, \begin{pmatrix}, ...
    'colon_fence',  # ::: callouts
]

html_theme = 'alabaster'
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']
