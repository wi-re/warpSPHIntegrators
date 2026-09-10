"""Phase 14: the structured RHS interface (NOTES S3.12).

The RHS input surface is two shapes -- a plain callable, or a typed ``RHS`` whose
declared capabilities cover every split a scheme asks for. These tests pin the
four validation gates of the phase plus the split contracts:

* **gate 1** -- every registered scheme reproduces the pre-refactor golden master
  on a bare callable, bit for bit (the refactor must not move an existing
  trajectory).
* **gate 2** -- ARK / IMEX Euler behave identically given ``IMEXRHS`` (now a
  constructor) or a hand-built ``RHS`` with the same halves.
* **gate 3** -- a ``linear``-consuming scheme raises a specific capability error
  on a plain ``f`` and on an ``IMEXRHS``, not a mid-solve shape mismatch.
* **gate 4** -- ``nonlinear`` synthesized from ``f - L.y`` matches an
  independently supplied ``N`` on viscous Burgers to round-off.

plus the ``check_contracts`` split contracts (linearity, additivity, agreement)
and the bare-callable-first-class resolution rules.
"""

import json
import os
import warnings

import pytest
import torch

from warpSPHIntegrators import (
    RHS,
    IMEXRHS,
    SemilinearRHS,
    add_updates,
    check_contracts,
    get_reference_state,
    getIntegrator,
    resolve,
    sub_updates,
    testing,
)
from warpSPHIntegrators.integration import IntegrationSchemes
from warpSPHIntegrators.util import split_return

_GOLDEN_PATH = os.path.join(os.path.dirname(__file__), 'data', 'bitident_golden.json')
_GOLDEN = None


def _golden():
    global _GOLDEN
    if _GOLDEN is None:
        with open(_GOLDEN_PATH) as fh:
            _GOLDEN = json.load(fh)
    return _GOLDEN


def _final_state(scheme, prob):
    system = prob.initial()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        for _ in range(_golden()['n_steps']):
            system = scheme(system, dt=_golden()['dt'], f=prob.rhs).state
    s = get_reference_state(system)
    return {'x': s.x.detach().cpu().tolist(),
            'u': s.u.detach().cpu().tolist(),
            'e': s.e.detach().cpu().tolist()}


# --------------------------------------------------------------------------- #
# Gate 1: bit-identical bare-callable trajectories                             #
# --------------------------------------------------------------------------- #

def test_bitidentical_bare_callable_trajectories():
    """Every registered scheme reproduces the pre-refactor golden on the oscillator.

    The golden (``tests/data/bitident_golden.json``) was captured *before* the
    RHS refactor; the plain-callable dispatch (a bare ``f`` resolves to
    ``explicit=None, implicit=f``) must therefore be bit-identical to the old
    ``isinstance`` branch. The oscillator is one degree of freedom, so the result
    is exact-float deterministic and the comparison is a strict equality.
    """
    golden = _golden()
    prob = testing.PROBLEMS['oscillator']()
    mismatches = []
    for scheme in IntegrationSchemes:
        ref = golden['states'][scheme.name]
        if 'error' in ref:
            continue
        got = _final_state(scheme, prob)
        if got != ref:
            mismatches.append(scheme.name)
    assert not mismatches, (
        f'{len(mismatches)} scheme(s) moved off the pre-refactor bare-callable '
        f'trajectory after the RHS refactor: {mismatches}')


# --------------------------------------------------------------------------- #
# Gate 2: IMEXRHS (constructor) == hand-built RHS with the same halves         #
# --------------------------------------------------------------------------- #

def _oscillator_halves():
    """Split the oscillator x'=u (explicit) / u'=-k x (implicit)."""
    k = 4.0

    def explicit(state, dt, *args, **kwargs):
        s = get_reference_state(state)
        return testing.ParticleUpdate(dxdt=s.u.clone(),
                                      dudt=torch.zeros_like(s.u),
                                      dedt=torch.zeros_like(s.e)), None

    def implicit(state, dt, *args, **kwargs):
        s = get_reference_state(state)
        return testing.ParticleUpdate(dxdt=torch.zeros_like(s.u),
                                      dudt=-k / s.m * s.x,
                                      dedt=torch.zeros_like(s.e)), None

    return explicit, implicit


@pytest.mark.parametrize('scheme_name', ['ARK3(2)4L[2]SA', 'ARK4(3)6L[2]SA', 'IMEX Euler'])
def test_imex_rhs_constructor_matches_handbuilt_rhs(scheme_name):
    """The two ways of building an additive split must give identical trajectories."""
    explicit, implicit = _oscillator_halves()
    via_constructor = IMEXRHS(explicit=explicit, implicit=implicit)
    via_class = RHS(explicit=explicit, implicit=implicit)
    assert sorted(via_constructor.provides) == sorted(via_class.provides) == ['explicit', 'implicit']

    prob = testing.PROBLEMS['oscillator']()
    scheme = getIntegrator(scheme_name)
    sys_a = prob.initial()
    sys_b = prob.initial()
    for _ in range(3):
        sys_a = scheme(sys_a, dt=0.1, f=via_constructor).state
        sys_b = scheme(sys_b, dt=0.1, f=via_class).state
    sa, sb = get_reference_state(sys_a), get_reference_state(sys_b)
    assert torch.equal(sa.x, sb.x) and torch.equal(sa.u, sb.u) and torch.equal(sa.e, sb.e)


# --------------------------------------------------------------------------- #
# Gate 3: a linear-consuming scheme fails with a specific capability error     #
# --------------------------------------------------------------------------- #

def _fake_rosenbrock(state, dt, f, **kwargs):
    """Stand-in for a Phase 7 linear-consuming driver: it resolves ``linear`` up
    front, which is where the capability error must fire (before the solve)."""
    return resolve(f, scheme_name='RosenbrockW (test)', need_linear=True)


def test_linear_capability_error_on_plain_callable():
    prob = testing.PROBLEMS['oscillator']()
    state = prob.initial()
    with pytest.raises(TypeError) as exc:
        _fake_rosenbrock(state, 0.1, prob.rhs)
    msg = str(exc.value)
    assert 'linear' in msg and 'RosenbrockW' in msg
    assert 'plain callable' in msg


def test_linear_capability_error_on_imex_rhs():
    explicit, implicit = _oscillator_halves()
    rhs = IMEXRHS(explicit=explicit, implicit=implicit)
    prob = testing.PROBLEMS['oscillator']()
    state = prob.initial()
    with pytest.raises(TypeError) as exc:
        _fake_rosenbrock(state, 0.1, rhs)
    msg = str(exc.value)
    assert 'linear' in msg and 'RosenbrockW' in msg
    assert 'explicit' in msg and 'implicit' in msg  # names what it *does* provide


def test_linear_capability_resolves_on_semilinear_rhs():
    prob = testing.PROBLEMS['viscousBurgers']()
    state = prob.initial()
    resolved = _fake_rosenbrock(state, 0.1, prob.rhs)  # must not raise
    assert resolved.linear is not None


# --------------------------------------------------------------------------- #
# Gate 4: synthesized nonlinear matches an independently supplied N            #
# --------------------------------------------------------------------------- #

def test_nonlinear_synthesis_matches_independent_n():
    """``SemilinearRHS(linear, combined=f)`` must synthesize ``N = f - L.y`` that
    agrees with the ``N`` supplied directly to ``SemilinearRHS(linear, nonlinear)``."""
    prob = testing.PROBLEMS['viscousBurgers']()
    rhs_split = prob.rhs  # SemilinearRHS(linear=L, nonlinear=N)
    state = prob.initial()

    # Rebuild the same dynamics as (linear, combined) so the nonlinear is derived.
    def combined(state_, dt, *a, **kw):
        k_l, _ = split_return(rhs_split._linear(state_, dt, *a, **kw))
        k_n, _ = split_return(rhs_split._nonlinear(state_, dt, *a, **kw))
        return add_updates(k_l, k_n), None

    rhs_synth = SemilinearRHS(linear=rhs_split._linear, combined=combined)
    assert 'nonlinear' not in rhs_synth.provides  # it is synthesized, not declared

    r_split = resolve(rhs_split, scheme_name='t')
    r_synth = resolve(rhs_synth, scheme_name='t')
    n_direct, _ = split_return(r_split.nonlinear(state, 0.1))
    n_synth, _ = split_return(r_synth.nonlinear(state, 0.1))
    # Round-off: synthesis computes ``f - L.y`` (two extra flops), so a tight
    # allclose is the honest gate, not bitwise equality.
    assert torch.allclose(n_direct.dxdt, n_synth.dxdt, atol=1e-14)


def test_viscous_burgers_split_is_additive():
    """On viscous Burgers the combined f is exactly L.y + N."""
    prob = testing.PROBLEMS['viscousBurgers']()
    rhs = prob.rhs
    state = prob.initial()
    r = resolve(rhs, scheme_name='t')
    f, _ = split_return(rhs(state, 0.1))
    l, _ = split_return(r.linear(state, 0.1))
    n, _ = split_return(r.nonlinear(state, 0.1))
    assert torch.equal(f.dxdt, l.dxdt + n.dxdt)


# --------------------------------------------------------------------------- #
# The split contracts (check_contracts)                                        #
# --------------------------------------------------------------------------- #

def test_check_contracts_semilinear_passes():
    prob = testing.PROBLEMS['viscousBurgers']()
    # Linearity (L(0)=0, L(a+b)=L(a)+L(b)) and semilinear agreement both hold.
    check_contracts(prob.rhs, prob.initial(), 0.1)


def test_check_contracts_additive_passes():
    explicit, implicit = _oscillator_halves()
    rhs = IMEXRHS(explicit=explicit, implicit=implicit)
    prob = testing.PROBLEMS['oscillator']()
    # Additivity: explicit + implicit == f.
    check_contracts(rhs, prob.initial(), 0.1)


def test_check_contracts_catches_a_nonlinear_linear_part():
    """A 'linear' part with a constant offset breaks L(0)=0 and must be caught."""
    prob = testing.PROBLEMS['viscousBurgers']()
    base_linear = prob.rhs._linear
    base_nonlinear = prob.rhs._nonlinear

    def bad_linear(state, dt, *a, **kw):
        k, _ = split_return(base_linear(state, dt, *a, **kw))
        s = get_reference_state(state)
        offset = testing.ParticleUpdate(dxdt=torch.ones_like(s.x) * 1e-3,
                                        dudt=torch.zeros_like(s.x),
                                        dedt=torch.zeros_like(s.x))
        return add_updates(k, offset), None

    rhs = SemilinearRHS(linear=bad_linear, nonlinear=base_nonlinear)
    with pytest.raises(AssertionError) as exc:
        check_contracts(rhs, prob.initial(), 0.1)
    assert 'L(0) == 0' in str(exc.value)


# --------------------------------------------------------------------------- #
# Bare callables stay first-class; resolution rules                            #
# --------------------------------------------------------------------------- #

def test_plain_callable_is_not_an_rhs():
    prob = testing.PROBLEMS['oscillator']()
    f = prob.rhs
    assert not isinstance(f, RHS)
    r = resolve(f, scheme_name='t')
    assert r.explicit is None
    assert r.implicit is f          # fully implicit
    assert r.linear is None
    assert r.nonlinear is f         # with no linear part, the whole f is nonlinear
    assert r.combined is f


def test_resolution_rules_on_declared_parts():
    explicit, implicit = _oscillator_halves()
    rhs = IMEXRHS(explicit=explicit, implicit=implicit)
    r = resolve(rhs, scheme_name='t')
    assert r.explicit is explicit
    assert r.implicit is implicit
    assert r.linear is None
    assert r.nonlinear is None  # no linear part, so nothing to subtract


def test_rhs_requires_at_least_one_part():
    with pytest.raises(ValueError):
        RHS()


def test_provides_reflects_the_declared_parts():
    explicit, implicit = _oscillator_halves()
    assert sorted(IMEXRHS(explicit=implicit, implicit=implicit).provides) == ['explicit', 'implicit']
    assert sorted(SemilinearRHS(linear=implicit, nonlinear=implicit).provides) == ['linear', 'nonlinear']
    assert sorted(SemilinearRHS(linear=implicit, combined=implicit).provides) == ['linear']
    all_four = RHS(explicit=implicit, implicit=implicit, linear=implicit, nonlinear=implicit)
    assert sorted(all_four.provides) == ['explicit', 'implicit', 'linear', 'nonlinear']


def test_overlap_case_additive_defines_combined():
    """When both ``implicit`` and ``linear`` are the same stiff operator, the
    additive halves define f and the semilinear parts are a view of it."""
    def stiff(state, dt, *a, **kw):
        s = get_reference_state(state)
        return testing.ParticleUpdate(dxdt=-3.0 * s.x, dudt=torch.zeros_like(s.x),
                                      dedt=torch.zeros_like(s.x)), None

    rhs = RHS(implicit=stiff, linear=stiff)  # the same operator behind both
    assert 'nonlinear' not in rhs.provides
    r = resolve(rhs, scheme_name='t')
    prob = testing.PROBLEMS['oscillator']()
    state = prob.initial()
    # combined == implicit (the additive view); nonlinear == f - L.y == 0 here
    # (the whole operator is linear).
    f, _ = split_return(rhs(state, 0.1))
    n, _ = split_return(r.nonlinear(state, 0.1))
    x = get_reference_state(state).x
    assert torch.equal(f.dxdt, -3.0 * x)
    assert torch.allclose(n.dxdt, torch.zeros_like(n.dxdt), atol=0.0)


def test_semilinear_rhs_rejects_both_nonlinear_and_combined():
    with pytest.raises(ValueError):
        SemilinearRHS(linear=lambda s, dt: None, nonlinear=lambda s, dt: None,
                      combined=lambda s, dt: None)


# --------------------------------------------------------------------------- #
# End-to-end: the semilinear problem integrates with a plain scheme            #
# --------------------------------------------------------------------------- #

def test_viscous_burgers_conserves_mass():
    """Integrating the semilinear problem via its combined f conserves the mass."""
    prob = testing.PROBLEMS['viscousBurgers'](n=64, nu=0.01)
    scheme = getIntegrator('RK4')
    system = prob.initial()
    m0 = get_reference_state(system).x.sum().item()
    worst = 0.0
    for _ in range(20):
        system = scheme(system, dt=1e-3, f=prob.rhs).state
        worst = max(worst, abs(get_reference_state(system).x.sum().item() - m0))
    assert worst < 1e-12  # round-off level: both halves have zero periodic mean
