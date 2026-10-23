"""Validate repository links/anchors, notebook syntax and disposition coverage.

Offline: GitHub links into this repository are checked against local files.
Other external URLs and downstream repository paths remain research references.
Site routes are resolved to docs/docs; Docusaurus validates rendering separately.
"""

import ast
import argparse
import hashlib
import importlib
import json
from pathlib import Path
import re
import sys
from urllib.parse import unquote, urlsplit


ROOT = Path(__file__).resolve().parents[1]
IGNORED = {'.git', '.tmp', '.venv', 'node_modules', 'build', '.docusaurus',
           '__pycache__', '.pytest_cache', 'literature', 'dist'}
REPOSITORY_URL = 'https://github.com/wi-re/warpSPHIntegrators/blob/main/'
LINK = re.compile(r'!?\[(?:[^\[\]\n]|\[[^\]\n]*\])*\]\(([^\n)]+)\)')


def prose(text):
    return re.sub(r'^(`{3,}|~{3,})[^\n]*\n.*?^\1[^\n]*$', '', text,
                  flags=re.M | re.S)


def slug(title):
    title = re.sub(r'<[^>]*>', '', title).lower()
    return ''.join(c for c in title if c.isalnum() or c in '_- ').replace(' ', '-')


def anchors(path):
    if path.suffix != '.md':
        return set()
    text = prose(path.read_text())
    found = set(re.findall(r'<a\s+(?:name|id)=["\']([^"\']+)', text))
    counts = {}
    for title in re.findall(r'^#{1,6}\s+(.+?)\s*#*$', text, re.M):
        key = slug(title)
        count = counts.get(key, 0)
        counts[key] = count + 1
        found.add(key if count == 0 else f'{key}-{count}')
    return found


def destination(page, link, root):
    link = unquote(link.strip('<>'))
    if link.startswith(REPOSITORY_URL):
        relative, _, fragment = link[len(REPOSITORY_URL):].partition('#')
        return (root / relative).resolve(), fragment
    elif link.startswith(('http:', 'https:', 'mailto:', 'data:')):
        return None, None
    parsed = urlsplit(link)
    path = parsed.path
    if path == '/docs' or path.startswith('/docs/'):
        relative = path.removeprefix('/docs').strip('/') or 'index'
        target = root / 'docs/docs' / relative
        if not target.suffix:
            target = target.with_suffix('.md')
    elif path.startswith('/'):
        target = root / path.lstrip('/')
    elif not path:
        target = page
    else:
        target = page.parent / path
        if page.is_relative_to(root / 'docs/docs') and not target.suffix:
            target = target.with_suffix('.md')
    target = target.resolve()
    # Cross-repository references are owned by their downstream checkout.
    if not target.is_relative_to(root):
        return None, None
    return target, parsed.fragment


def validate_disposition_paths(entries, root=ROOT):
    """Check current local paths, leaving historical source citations untouched."""
    errors = []
    root = root.resolve()
    for entry in entries:
        for field in ('implementation', 'tests', 'evidence'):
            value = entry.get(field)
            if value is None:
                continue
            references = [value] if isinstance(value, str) else value
            if not isinstance(references, list) or not all(
                    isinstance(ref, str) and ref for ref in references):
                errors.append(f'{entry["name"]}: {field} must be a path, a list of paths or null')
                continue
            for reference in references:
                relative, _, fragment = reference.partition('#')
                target = (root / relative).resolve()
                if not relative or Path(relative).is_absolute() or not target.is_relative_to(root):
                    errors.append(f'{entry["name"]}: {field} is not a repository path: {reference}')
                elif not target.is_file():
                    errors.append(f'{entry["name"]}: missing {field} path {reference}')
                elif fragment and target.suffix == '.md' and fragment not in anchors(target):
                    if not re.fullmatch(r'L\d+(?:-L\d+)?', fragment):
                        errors.append(f'{entry["name"]}: missing {field} anchor {reference}')
    return errors


def validate(root=ROOT):
    errors = []
    pages = notebooks = links = 0
    anchor_cache = {}
    for path in sorted(root.rglob('*')):
        if not path.is_file() or any(part in IGNORED for part in path.relative_to(root).parts):
            continue
        texts = []
        if path.suffix == '.md':
            pages += 1
            texts = [path.read_text()]
        elif path.suffix == '.ipynb':
            notebooks += 1
            notebook = json.loads(path.read_text())
            for i, cell in enumerate(notebook['cells']):
                source = ''.join(cell['source'])
                if cell['cell_type'] == 'markdown':
                    texts.append(source)
                elif cell['cell_type'] == 'code':
                    # IPython magics are not Python source. The ordinary code
                    # around line magics remains syntax-checked.
                    if source.lstrip().startswith('%%'):
                        continue
                    source = '\n'.join('pass' if line.lstrip().startswith(('%', '!'))
                                       else line for line in source.splitlines())
                    try:
                        ast.parse(source, filename=f'{path}:cell {i}')
                    except SyntaxError as exc:
                        errors.append(str(exc))
        for text in texts:
            for link in LINK.findall(prose(text)):
                # Example annotation definitions are not navigation links.
                target, fragment = destination(path, link, root)
                if target is None:
                    continue
                links += 1
                if not target.exists():
                    errors.append(f'{path.relative_to(root)}: missing target {link}')
                elif fragment and target.suffix == '.md':
                    valid = anchor_cache.setdefault(target, anchors(target))
                    if fragment not in valid and not re.fullmatch(r'L\d+(?:-L\d+)?', fragment):
                        errors.append(f'{path.relative_to(root)}: missing anchor {link}')

    from warpSPHIntegrators.schemes import IntegrationSchemes
    ledger = json.loads((root / 'docs/plans/method-dispositions.json').read_text())
    registered = ledger['registered_methods']
    names = [entry['identifier'] for entry in registered]
    expected = {s.identifier.name for s in IntegrationSchemes}
    if len(names) != len(set(names)) or set(names) != expected:
        errors.append('Method disposition ledger does not cover each registered identifier exactly once')
    for entry in registered:
        if entry['status'] != 'implemented':
            errors.append(f'{entry["name"]}: registered method is not recorded as implemented')
        for key in ('version', 'revision', 'file', 'constructor', 'verification'):
            if key not in entry['source']:
                errors.append(f'{entry["name"]}: missing source field {key}')
        for relative in [entry['implementation']['file'], entry['evidence'], *entry['tests']]:
            if not (root / relative).is_file():
                errors.append(f'{entry["name"]}: missing ledger evidence {relative}')

    for entry in registered:
        relative=Path(entry['implementation']['file'])
        if relative.parts[:2] == ('src','warpSPHIntegrators'):
            module='.'.join(relative.with_suffix('').parts[1:])
            if not hasattr(importlib.import_module(module),entry['implementation']['constructor']):
                errors.append(f"{entry['name']}: implementation constructor does not resolve")

    errors.extend(validate_disposition_paths(ledger['other_dispositions'], root))

    migrations = json.loads((root / 'docs/development/migration-map.json').read_text())
    errors.extend(validate_disposition_paths(
        [dict(name=f'Migration {old}', evidence=new) for old, new in migrations.items()], root))

    # Import-only inspection catches stale advanced paths in benchmarks,
    # coefficient tools and conditional test branches without executing them.
    bindings=0
    for directory in ('src','tests','scripts','examples','benchmarks'):
        for path in (root/directory).rglob('*.py'):
            if '__pycache__' in path.parts:continue
            for node in ast.walk(ast.parse(path.read_text())):
                if isinstance(node,ast.ImportFrom) and node.module and node.module.startswith('warpSPHIntegrators'):
                    try:
                        module=importlib.import_module(node.module)
                        for alias in node.names:
                            if alias.name!='*':getattr(module,alias.name);bindings+=1
                    except (ImportError,AttributeError) as exc:
                        errors.append(f'{path.relative_to(root)}: {exc}')
    print(f'{bindings} integrator import bindings checked')

    coefficient_root = root / 'scripts/coefficients'
    manifest = json.loads((coefficient_root / 'promotion-manifest.json').read_text())
    for relative, entry in manifest.items():
        path = coefficient_root / relative
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != entry['maintained_sha256']:
            errors.append(f'Coefficient promotion hash differs: {relative}')

    return errors, (pages, notebooks, links, len(registered))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    errors, counts = validate()
    for error in errors:
        print(error, file=sys.stderr)
    print(f'{counts[0]} Markdown documents, {counts[1]} notebooks, '
          f'{counts[2]} local links, {counts[3]} registered dispositions checked')
    return int(bool(errors))


if __name__ == '__main__':
    sys.exit(main())
