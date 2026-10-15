"""Public catalog facts must match executable drivers and generated artifacts."""

from dataclasses import FrozenInstanceError
from pathlib import Path
import subprocess
import sys

import pytest

from warpSPHIntegrators.adapters import CAPABILITIES
from warpSPHIntegrators.schemes import IntegrationSchemes
from warpSPHIntegrators import get_scheme
import warpSPHIntegrators.testing as testing


def test_catalog_filters_compose_and_preserve_registered_records():
    from warpSPHIntegrators.schemes import filter_schemes, ExplicitRK
    pairs = filter_schemes(family=ExplicitRK, min_order=5, has_error_estimate=True, supports_reuse=True)
    assert get_scheme('Dormand-Prince 5(4)') in pairs
    assert all(s.family is ExplicitRK and s.order >= 5 and s.has_error_estimate and s.supports_reuse for s in pairs)
    assert filter_schemes(dense_output=True) == (get_scheme('Dormand-Prince 5(4)'),)
    assert get_scheme('ETD2RK') not in filter_schemes(available_capabilities=set())
    assert get_scheme('ETD2RK') in filter_schemes(available_capabilities={'linear'})
    assert get_scheme('BDF2') in filter_schemes(requires_history=True)
    assert not filter_schemes(family='ExplicitRK', implicit=True)
    assert get_scheme('Velocity Verlet') in filter_schemes(requires_separable=True, supports_drift_fields=True)


@pytest.mark.parametrize('criteria', [{'family':'typo'}, {'min_order':5,'max_order':2},
                                     {'order':False}, {'state_kind':'unknown'},
                                     {'available_capabilities':{'typo'}}, {'implicit':'yes'}])
def test_catalog_invalid_criteria_fail_clearly(criteria):
    from warpSPHIntegrators.schemes import filter_schemes
    with pytest.raises(ValueError):
        filter_schemes(**criteria)


def test_explicit_enum_selection_returns_typed_catalog_records():
    from warpSPHIntegrators.schemes import ExplicitRK
    choices=[ExplicitRK.forwardEuler,ExplicitRK.rungeKutta2,ExplicitRK.sspRK3,
             ExplicitRK.rungeKutta4,ExplicitRK.dormandPrince]
    for order,identifier in enumerate(choices,1):
        scheme=get_scheme(identifier)
        assert scheme in IntegrationSchemes and scheme.order==order and scheme.metadata.state_kind=='tensor'
    assert get_scheme(ExplicitRK.dormandPrince) is get_scheme('Dormand-Prince 5(4)')
    with pytest.raises(ValueError):get_scheme(6)


def test_structural_metadata_matches_existing_driver_contracts():
    assert get_scheme('Velocity Verlet').metadata.requires_separable
    assert get_scheme('Velocity Verlet').metadata.supports_drift_fields
    assert get_scheme('Newmark').metadata.state_kind == 'second_order'
    assert not get_scheme('RK4').metadata.supports_drift_fields
    assert get_scheme('BDF3').metadata.history_length == 2


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
    ('Gauss-Legendre 3', 6, 2, 3), ('Radau IIA s=3', 5, 3, 4),
])
def test_estimate_exponent_is_distinct_from_propagated_and_companion_orders(name, propagated, companion, q):
    s = get_scheme(name)
    assert (s.order, s.embedded_order, s.estimate_order) == (propagated, companion, q)


def test_mandatory_rhs_and_stage_options_match_driver_requirements():
    assert get_scheme('ETD2RK').required_capabilities == {'linear'}
    for s in IntegrationSchemes:
        if s.family.__name__ == 'IMEX':
            assert not s.required_capabilities
            assert s.metadata.optional_capabilities == {'explicit', 'implicit'}
    for name in ('RKC1', 'RKC2', 'RKL2'):
        assert get_scheme(name).metadata.required_options == ('s', 'lambda_max')
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
