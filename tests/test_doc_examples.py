"""The published blocks must run with only their published imports and context."""

import math
from pathlib import Path
import runpy

import pytest

from warpSPHIntegrators.state import get_reference_state


ROOT = Path(__file__).resolve().parents[1]
VALIDATOR = runpy.run_path(str(ROOT / 'scripts/validate_doc_examples.py'))
REPOSITORY_VALIDATOR = runpy.run_path(str(ROOT / 'scripts/validate_repository.py'))


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


def test_complete_tensor_example_matches_exact_decay():
    module = runpy.run_path(str(ROOT / 'examples/tensor_ode.py'))
    result = module['run_example']()
    assert result.state.t == pytest.approx(1.)
    assert result.state.y.tolist() == pytest.approx([math.exp(-2), 2*math.exp(-2)], rel=1e-8)


def test_nonregistered_evidence_paths_accept_supported_forms_and_ignore_source_history(tmp_path):
    (tmp_path / 'method.py').write_text('')
    (tmp_path / 'evidence.md').write_text('# Source qualification\n')
    entries = [dict(name='prototype', implementation='method.py', tests=['method.py'],
                    evidence='evidence.md#source-qualification', source='removed/old_source.py'),
               dict(name='deferred', implementation=None, tests=None, evidence=['evidence.md'])]
    assert REPOSITORY_VALIDATOR['validate_disposition_paths'](entries, tmp_path) == []


@pytest.mark.parametrize('field', ['implementation', 'tests', 'evidence'])
def test_nonregistered_missing_paths_fail_validation(tmp_path, field):
    entries = [dict(name='candidate', **{field: 'removed-plan.md'})]
    errors = REPOSITORY_VALIDATOR['validate_disposition_paths'](entries, tmp_path)
    assert errors == [f'candidate: missing {field} path removed-plan.md']


def test_nonregistered_missing_evidence_anchor_fails_validation(tmp_path):
    (tmp_path / 'evidence.md').write_text('# Retained design\n')
    entries = [dict(name='candidate', evidence='evidence.md#removed-heading')]
    errors = REPOSITORY_VALIDATOR['validate_disposition_paths'](entries, tmp_path)
    assert errors == ['candidate: missing evidence anchor evidence.md#removed-heading']


@pytest.mark.parametrize('value', [42, {'file': 'evidence.md'}, ['evidence.md', None]])
def test_nonregistered_invalid_path_forms_fail_validation(tmp_path, value):
    errors = REPOSITORY_VALIDATOR['validate_disposition_paths'](
        [dict(name='candidate', evidence=value)], tmp_path)
    assert errors == ['candidate: evidence must be a path, a list of paths or null']


@pytest.mark.parametrize('reference', ['../outside.md', '/tmp/outside.md'])
def test_nonregistered_evidence_must_stay_inside_repository(tmp_path, reference):
    errors = REPOSITORY_VALIDATOR['validate_disposition_paths'](
        [dict(name='candidate', evidence=reference)], tmp_path)
    assert errors == [f'candidate: evidence is not a repository path: {reference}']
