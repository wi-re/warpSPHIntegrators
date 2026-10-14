"""The published blocks must run with only their published imports and context."""

import math
from pathlib import Path
import runpy

import pytest

from warpSPHIntegrators import get_reference_state


ROOT = Path(__file__).resolve().parents[1]
VALIDATOR = runpy.run_path(str(ROOT / 'scripts/validate_doc_examples.py'))


def test_published_examples_execute_verbatim():
    ran, skipped = VALIDATOR['validate_pages']()
    assert ran >= 27
    assert skipped > 0


def test_complete_oscillator_example_matches_the_exact_solution():
    module = runpy.run_path(str(ROOT / 'examples/harmonic_oscillator.py'))
    result = module['run_example']()
    state = get_reference_state(result.state)
    assert result.state.t == pytest.approx(1.0)
    assert state.position.item() == pytest.approx(math.cos(2.0), abs=1e-8)
    assert state.velocity.item() == pytest.approx(-2 * math.sin(2.0), abs=1e-8)
    assert len(result.stages) == 4
    assert result.error is None


def test_missing_example_imports_are_not_injected(tmp_path):
    (tmp_path / 'guide.md').write_text('[example:run demo]: #\n```python\ntorch.tensor([1.0])\n```\n')
    with pytest.raises(NameError, match='torch'):
        VALIDATOR['validate_pages'](tmp_path, ('guide.md',))


def test_unclassified_python_blocks_fail_validation(tmp_path):
    (tmp_path / 'guide.md').write_text('```python\npass\n```\n')
    with pytest.raises(ValueError, match='lacks an example annotation'):
        VALIDATOR['validate_pages'](tmp_path, ('guide.md',))


def test_included_example_drift_fails_validation(tmp_path):
    (tmp_path / 'example.py').write_text('x = 1\n')
    (tmp_path / 'guide.md').write_text('[example:include example.py]: #\n```python\nx = 2\n```\n')
    with pytest.raises(ValueError, match='differs from example.py'):
        VALIDATOR['validate_pages'](tmp_path, ('guide.md',))
