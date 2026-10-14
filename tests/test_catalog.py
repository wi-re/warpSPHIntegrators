"""Public catalog facts must match executable drivers and generated artifacts."""

from dataclasses import FrozenInstanceError
from pathlib import Path
import subprocess
import sys

import pytest

from warpSPHIntegrators import CAPABILITIES, IntegrationSchemes, getIntegrator, testing


@pytest.mark.parametrize('s', IntegrationSchemes, ids=lambda s: s.name)
def test_metadata_preserves_the_scheme_tuple_contract(s):
    assert len(s) == 12
    assert len(s._fields) == 12
    assert 'metadata' not in s._asdict()
    assert s.required_capabilities <= CAPABILITIES
    assert s.metadata.optional_capabilities <= CAPABILITIES
    assert s.has_error_estimate == (s.estimate_order is not None)
    if s.estimate_order is not None:
        assert s.embedded_order is not None and s.estimate_order > 0
    with pytest.raises(FrozenInstanceError):
        s.metadata.estimate_order = 99


@pytest.mark.parametrize('name,propagated,companion,q', [
    ('TR-BDF2', 2, 3, 3), ('Gauss-Legendre 2', 4, 2, 3),
    ('Radau IIA s=2', 3, 2, 3), ('Dormand-Prince 5(4)', 5, 4, 5),
])
def test_estimate_exponent_is_distinct_from_propagated_and_companion_orders(name, propagated, companion, q):
    s = getIntegrator(name)
    assert (s.order, s.embedded_order, s.estimate_order) == (propagated, companion, q)


def test_mandatory_rhs_and_stage_options_match_driver_requirements():
    assert getIntegrator('ETD2RK').required_capabilities == {'linear'}
    for s in IntegrationSchemes:
        if s.family.__name__ == 'IMEX':
            assert not s.required_capabilities
            assert s.metadata.optional_capabilities == {'explicit', 'implicit'}
    for name in ('RKC1', 'RKC2', 'RKL2'):
        assert getIntegrator(name).metadata.required_options == ('s', 'lambda_max')
    assert [s.name for s in IntegrationSchemes if s.metadata.dense_output] == ['Dormand-Prince 5(4)']


def test_estimate_metadata_matches_a_real_step(scheme):
    prob = testing.oscillator_problem()
    result = scheme(prob.initial(), dt=0.1, f=prob.rhs)
    assert (result.error is not None) == scheme.has_error_estimate


def test_generated_catalog_is_current():
    root = Path(__file__).resolve().parents[1]
    run = subprocess.run([sys.executable, 'scripts/generate_catalog.py', '--check'],
                         cwd=root, capture_output=True, text=True)
    assert run.returncode == 0, run.stdout + run.stderr
