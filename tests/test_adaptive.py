"""Adaptive step control and dense output (roadmap Phase 11, NOTES.md S3.17).

The controller is a *helper, not a solver*: ``estimate_error_norm`` +
``propose_dt``, and the caller drives the accept/reject loop (NOTES.md S3.8's
"no solver contract without a downstream" caution). ``run_adaptive`` below is
that loop, the demo the Phase 11 contract leaves to the caller (the same loop
lives in ``scripts/adaptive_benchmark.py``). The multistep refusal is pinned
elsewhere: the schemes emit no estimate (``test_non_embedded_schemes_report_no_error``)
and ``StepHistory`` restarts on any ``dt`` change
(``tests/test_groundwork.py::test_step_history_restarts_on_dt_change``).
"""

import math

import numpy as np
import pytest
import torch

from warpSPHIntegrators import (
    dormand_prince_dense_output,
    estimate_error_norm,
    getIntegrator,
    get_reference_state,
    propose_dt,
    state_norm,
    testing,
)
from warpSPHIntegrators.adaptive import _DP5_B_MAIN
from warpSPHIntegrators.butcher import getButcherTableau

#: The estimate emitters (tests/test_embedded.py): the (p, p-1) embedded pairs
#: plus the built-in estimates. The power q the estimate scales as h^q is the
#: scheme's order, except TR-BDF2, whose published SUNDIALS pair makes the
#: estimate O(h**(order+1)).
EMBEDDED = ['Bogacki-Shampine 3(2)', 'Dormand-Prince 5(4)', 'Cash-Karp 5(4)',
            'TR-BDF2', 'ESDIRK3(2)4L[2]SA', 'ESDIRK4(3)6L[2]SA',
            'ARK3(2)4L[2]SA', 'ARK4(3)6L[2]SA', 'ROS3P', 'EXPRB32']


def estimate_order(scheme) -> int:
    """The power q the estimate scales as h^q (NOTES.md S3.17)."""
    return scheme.order + 1 if scheme.name == 'TR-BDF2' else scheme.order


def run_adaptive(scheme, prob, T, dt0, *, rtol=1e-6, atol=1e-9,
                 safety=0.9, growth_min=0.2, growth_max=5.0):
    """The caller-driven accept/reject loop Phase 11's helper-only contract
    leaves to the caller: step, measure, accept/reject, propose. Returns
    ``(final_state, accepted, rejected, norms)``."""
    q = estimate_order(scheme)
    state = prob.initial()
    t = float(state.t)
    dt = dt0
    accepted = rejected = 0
    norms = []
    attempts = 0
    while t < T - 1e-12:
        step = min(dt, T - t)
        result = scheme(state, dt=step, f=prob.rhs)
        norm = estimate_error_norm(result, rtol=rtol, atol=atol)
        if norm <= 1.0:
            state = result.state
            t = float(state.t)
            accepted += 1
            norms.append(norm)
        else:
            rejected += 1
        proposed = propose_dt(norm, step, q, safety=safety,
                              growth_min=growth_min, growth_max=growth_max)
        if norm > 1.0:
            proposed = min(proposed, step)  # never grow on a rejection
        dt = min(proposed, T - t)
        attempts += 1
        assert attempts < 100000, 'the adaptive run did not converge'
    return state, accepted, rejected, norms


# --------------------------------------------------------------------------- #
# propose_dt: the predictive controller
# --------------------------------------------------------------------------- #

def test_propose_dt_predicts_shrink_and_grow():
    # error 4x the tolerance, order 5: factor = 0.9 * 4**(-1/5)
    assert propose_dt(4.0, 0.1, 5) == pytest.approx(0.1 * 0.9 * 4.0 ** -0.2)
    # error 32x below the tolerance, order 5: factor = 0.9 * 32**(1/5) = 1.8,
    # inside the clamp
    assert propose_dt(1/32, 0.1, 5) == pytest.approx(0.1 * 0.9 * 32.0 ** 0.2)


def test_propose_dt_clamps_to_the_growth_bounds():
    assert propose_dt(1e-12, 0.1, 5) == pytest.approx(0.1 * 5.0)
    assert propose_dt(1e12, 0.1, 5) == pytest.approx(0.1 * 0.2)


def test_propose_dt_zero_error_grows_to_max():
    assert propose_dt(0.0, 0.1, 5) == pytest.approx(0.5)


def test_propose_dt_target_rescales_the_goal():
    # order 1: factor = safety * target / error
    assert propose_dt(2.0, 0.1, 1, target=2.0) == pytest.approx(0.09)
    assert propose_dt(0.5, 0.1, 1, target=0.25) == pytest.approx(0.045)


def test_propose_dt_rejects_bad_arguments():
    for args in [(1.0, 0.0, 5), (1.0, -0.1, 5), (1.0, 0.1, 0), (1.0, 0.1, -5),
                 (-1.0, 0.1, 5)]:
        with pytest.raises(ValueError):
            propose_dt(*args)
    for kwargs in [dict(target=0.0), dict(safety=0.0), dict(growth_min=0.0),
                   dict(growth_min=5.0, growth_max=2.0)]:
        with pytest.raises(ValueError):
            propose_dt(1.0, 0.1, 5, **kwargs)


# --------------------------------------------------------------------------- #
# estimate_error_norm
# --------------------------------------------------------------------------- #

def test_estimate_error_norm_scales_the_difference_against_the_propagated_state():
    s = getIntegrator('Dormand-Prince 5(4)')
    prob = testing.PROBLEMS['oscillator']()
    result = s(prob.initial(), dt=0.1, f=prob.rhs)
    assert estimate_error_norm(result) == pytest.approx(
        state_norm(result.error, reference=result.state), rel=1e-12)


def test_estimate_error_norm_raises_for_schemes_without_estimates():
    # RK4: a one-step scheme with no embedded pair. BDF2: the multistep family
    # (no estimate, and a variable step would restart its history anyway).
    for name in ('RK4', 'BDF2'):
        s = getIntegrator(name)
        prob = testing.PROBLEMS['oscillator']()
        result = s(prob.initial(), dt=0.1, f=prob.rhs)
        assert result.error is None
        with pytest.raises(ValueError, match='estimate-emitting'):
            estimate_error_norm(result)


@pytest.mark.parametrize('name', EMBEDDED)
def test_estimate_error_norm_is_finite_and_positive(name):
    s = getIntegrator(name)
    # ROS3P's and EXPRB32's built-in estimates degenerate to zero on a linear
    # autonomous problem (tests/test_embedded.py); forced is the smallest
    # problem where they are active.
    prob_name = 'forced' if name in ('ROS3P', 'EXPRB32') else 'oscillator'
    prob = testing.PROBLEMS[prob_name]()
    result = s(prob.initial(), dt=0.1, f=prob.rhs)
    n = estimate_error_norm(result)
    assert math.isfinite(n)
    assert n > 0


# --------------------------------------------------------------------------- #
# The caller-driven loop: contract + the roadmap gate
# --------------------------------------------------------------------------- #

def test_adaptive_loop_holds_the_contract_on_the_oscillator():
    s = getIntegrator('Dormand-Prince 5(4)')
    prob = testing.PROBLEMS['oscillator']()
    state, accepted, rejected, norms = run_adaptive(s, prob, T=2.0, dt0=0.1,
                                                    rtol=1e-8, atol=1e-11)
    assert accepted > 0 and rejected > 0
    assert all(n <= 1.0 for n in norms), 'an accepted step must sit at the target'
    ex, eu = prob.exact(2.0)
    sx, su = get_reference_state(state).x, get_reference_state(state).u
    err = abs(float(sx[0]) - ex[0]) + abs(float(su[0]) - eu[0])
    assert err < 1e-5
    assert state.t == pytest.approx(2.0, abs=1e-12)


def test_rejections_trigger_and_gate_holds_on_van_der_pol_mu10():
    """Roadmap gate: step rejection triggers on van der Pol at mu = 10 (the
    Phase 8 hard explicit wall), and the stiff problem completes in materially
    fewer steps under control than at the fixed dt needed for the same final
    error. Measured (scripts/adaptive_benchmark.py): 75 accepted steps at
    error ~ 4e-6, where the fixed ladder needs 312 steps (dt = 0.016) for the
    same error."""
    prob = testing.van_der_pol_problem(mu=10.0)
    T = 5.0
    s = getIntegrator('Dormand-Prince 5(4)')
    # reference: a fine fixed-dt run; its own error floor (~1e-13, measured at
    # two resolutions in the benchmark) is far below every error here
    ref = testing.run(s, prob, dt=5e-4, T=T)
    r = get_reference_state(ref)

    def err(system):
        sst = get_reference_state(system)
        return max(abs(float(sst.x[0]) - float(r.x[0])),
                   abs(float(sst.u[0]) - float(r.u[0])))

    state, accepted, rejected, norms = run_adaptive(s, prob, T, dt0=0.1,
                                                    rtol=1e-5, atol=1e-8)
    assert rejected > 0, 'the gate: rejection must trigger where the wall is'
    assert all(n <= 1.0 for n in norms)
    e_ad = err(state)
    assert e_ad > 1e-9, 'the comparison must sit well above the reference floor'

    # the fixed dt ladder, coarsest first: (step count, error)
    ladder = [(T / dt, err(testing.run(s, prob, dt, T))) for dt in (0.016, 0.008)]
    matching = [n for n, e in ladder if e <= e_ad]
    assert matching, 'the fixed ladder never reached the adaptive error'
    assert accepted < 0.75 * min(matching), (
        f'adaptive used {accepted} steps, the coarsest matching fixed dt '
        f'needs {min(matching)}')


# --------------------------------------------------------------------------- #
# Dormand-Prince dense output (Shampine 1986 quartic)
# --------------------------------------------------------------------------- #

def test_dense_output_weights_match_the_tableau():
    tab = getButcherTableau('DormandPrince')
    assert np.array_equal(_DP5_B_MAIN, tab.b[0]), (
        'the restated propagated weights drifted from butcher.py')


def test_dense_output_endpoints_are_exact():
    """The gate: the interpolant reproduces the propagated solution exactly at
    the step endpoints (theta=0 the step's initial state; theta=1 the step's
    returned state, bit-for-bit)."""
    s = getIntegrator('Dormand-Prince 5(4)')
    prob = testing.PROBLEMS['oscillator']()
    state0 = prob.initial()
    result = s(state0, dt=0.2, f=prob.rhs)
    y0 = dormand_prince_dense_output(state0, result.stages, 0.2, 0.0)
    assert y0 is state0
    y1 = dormand_prince_dense_output(state0, result.stages, 0.2, 1.0)
    r1, s1 = get_reference_state(result.state), get_reference_state(y1)
    for f in ('x', 'u'):
        assert torch.equal(getattr(r1, f), getattr(s1, f)), f
    assert y1.t == pytest.approx(result.state.t)


def test_dense_output_interior_reaches_order_four():
    """The quartic satisfies the continuous order conditions through order 4,
    so the local interpolation error at a fixed theta is O(dt**5) -- measured
    on the linear oscillator, where the stage values are exact polynomials in
    dt and the only error is the polynomial approximation of the exact flow."""
    s = getIntegrator('Dormand-Prince 5(4)')
    prob = testing.PROBLEMS['oscillator']()
    theta = 0.4
    dts = [0.2, 0.1, 0.05, 0.025]
    errs = []
    for dt in dts:
        state0 = prob.initial()
        result = s(state0, dt=dt, f=prob.rhs)
        y = dormand_prince_dense_output(state0, result.stages, dt, theta)
        ex, eu = prob.exact(theta * dt)
        sy, su = get_reference_state(y).x, get_reference_state(y).u
        errs.append(max(abs(float(sy[0]) - ex[0]), abs(float(su[0]) - eu[0])))
    rates = [math.log(a / b) / math.log(2) for a, b in zip(errs, errs[1:])]
    assert all(4.6 < r < 5.4 for r in rates), f'interior rates {rates}'


def test_dense_output_interior_tracks_fine_steps():
    """A loose cross-check: one step of dt queried at theta = 1/2 (t = dt/2)
    vs. one step of dt/2 (t = dt/2) -- the difference is dominated by the
    coarse step's O(dt**5) interpolation error, the fine step's own error is
    one order smaller."""
    s = getIntegrator('Dormand-Prince 5(4)')
    prob = testing.PROBLEMS['oscillator']()
    state0 = prob.initial()
    result = s(state0, dt=0.2, f=prob.rhs)
    y_half = dormand_prince_dense_output(state0, result.stages, 0.2, 0.5)
    fine = s(prob.initial(), dt=0.1, f=prob.rhs).state
    a, b = get_reference_state(y_half), get_reference_state(fine)
    diff = max(abs(float(a.x[0]) - float(b.x[0])),
               abs(float(a.u[0]) - float(b.u[0])))
    assert diff < 1e-3, f'dense output at theta=1/2 deviates {diff} from the ' \
                        'half-step run'


def test_dense_output_rejects_other_schemes_and_bad_theta():
    s = getIntegrator('Bogacki-Shampine 3(2)')
    prob = testing.PROBLEMS['oscillator']()
    state0 = prob.initial()
    result = s(state0, dt=0.1, f=prob.rhs)
    with pytest.raises(ValueError, match='seven stages'):
        dormand_prince_dense_output(state0, result.stages, 0.1, 0.5)
    s5 = getIntegrator('Dormand-Prince 5(4)')
    r5 = s5(state0, dt=0.1, f=prob.rhs)
    for bad in (-0.1, 1.1):
        with pytest.raises(ValueError, match='theta'):
            dormand_prince_dense_output(state0, r5.stages, 0.1, bad)
    with pytest.raises(ValueError, match='nonzero'):
        dormand_prince_dense_output(state0, r5.stages, 0.0, 0.5)
