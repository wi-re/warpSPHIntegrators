"""Compatibility entry point for direct validation of published documentation.

Run: python scripts/validate_readme_examples.py
The validator now executes the source blocks, not hand-reconstructed equivalents.
"""

from validate_doc_examples import main


if __name__ == '__main__':
    raise SystemExit(main())
