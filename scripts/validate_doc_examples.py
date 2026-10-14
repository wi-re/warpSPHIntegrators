"""Execute annotated published Python blocks without injecting missing imports.

Run blocks with the same scope on a page in order. Include blocks must match
their standalone source file verbatim. Skip annotations identify sketches that
require caller definitions, rather than silently validating reconstructed code.
"""

import argparse
from contextlib import redirect_stdout
import io
from pathlib import Path
import re
import sys
from types import ModuleType


ROOT = Path(__file__).resolve().parents[1]
PAGES = ('README.md', 'docs/docs/index.md', 'docs/docs/quickstart.md',
         'docs/docs/api.md', 'docs/docs/solver.md')
FENCE = re.compile(r'^```python[^\n]*\n(.*?)^```[ \t]*$', re.M | re.S)
ANNOTATION = re.compile(r'\[example:(run|skip|include) ([^\n]+?)\]: #\s*$')


def python_blocks(text, label):
    for match in FENCE.finditer(text):
        annotation = ANNOTATION.search(text[:match.start()])
        line = text.count('\n', 0, match.start()) + 2
        if annotation is None:
            raise ValueError(f'{label}:{line}: Python block lacks an example annotation')
        yield annotation.group(1), annotation.group(2), match.group(1), line


def validate_pages(root=ROOT, pages=PAGES):
    ran = skipped = 0
    for page in pages:
        path = root / page
        scopes = {}
        for kind, detail, code, line in python_blocks(path.read_text(), page):
            if kind == 'skip':
                skipped += 1
                continue
            if kind == 'include':
                source = root / detail
                if code != source.read_text():
                    raise ValueError(f'{page}:{line}: included example differs from {detail}')
            scope = '__main__' if kind == 'include' else f'_doc_{path.stem}_{detail}'
            module = scopes.setdefault(scope, ModuleType(scope))
            module.__file__ = str(path)
            previous = sys.modules.get(scope)
            sys.modules[scope] = module
            try:
                # No inherited future flags and no preloaded library names.
                with redirect_stdout(io.StringIO()):
                    exec(compile('\n' * (line-1) + code, str(path), 'exec',
                                 dont_inherit=True), module.__dict__)
            finally:
                if previous is None:
                    sys.modules.pop(scope, None)
                else:
                    sys.modules[scope] = previous
            ran += 1
    return ran, skipped


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    ran, skipped = validate_pages()
    print(f'{ran} published Python blocks executed; {skipped} caller-defined sketches labeled')
    return 0


if __name__ == '__main__':
    sys.exit(main())
