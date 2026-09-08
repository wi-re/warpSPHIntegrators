"""NOTES.md S1-S5: the implicit/multistep groundwork layer.

Nothing here is consumed by a registered scheme yet (that's Phase 1/2) -- these test
the primitives in isolation, plus one end-to-end check that `history=` threaded
through a real Butcher scheme reproduces what `priorStep=` already does, since the
whole point of S2 is that the two must not drift apart.
"""

from dataclasses import dataclass

import pytest
import torch

from warpSPHIntegrators import (
    BaseState,
    FixedPointSolver,
    RelaxedFixedPointSolver,
    HistoryEntry,
    IntegrationScheme,
    StepHistory,
    constant,
    get_reference_state,
    getIntegrator,
    integrated,
    state_difference,
    state_norm,
    testing,
)
from warpSPHIntegrators.integration import IntegrationSchemes


@dataclass
class MixedState(BaseState):
    x: torch.Tensor = integrated('dxdt', tags=('position',))
    label: str = constant(default='fluid')


def _mixed(x):
    return MixedState(x=torch.tensor(x, dtype=torch.float64))


# --------------------------------------------------------------------------- #
# S1 -- state_difference / state_norm                                         #
# --------------------------------------------------------------------------- #

def test_state_difference_subtracts_integrated_fields_only():
    a = _mixed([3.0, 5.0])
    b = _mixed([1.0, 2.0])
    b.label = 'other'
    diff = state_difference(a, b)
    assert torch.equal(diff.x, torch.tensor([2.0, 3.0], dtype=torch.float64))
    assert diff.label == 'fluid', 'non-integrated fields come from state_a, not b'


def test_state_difference_preserves_a_system_wrapper():
    """A differenced system must stay a system: get_reference_state must still work."""
    prob = testing.PROBLEMS['oscillator']()
    a = prob.initial()
    b = prob.initial()
    b.state.x = b.state.x * 0.5

    diff = state_difference(a, b)
    inner = get_reference_state(diff)
    assert torch.equal(inner.x, get_reference_state(a).x - get_reference_state(b).x)


def test_state_difference_does_not_mutate_inputs():
    a = _mixed([3.0])
    b = _mixed([1.0])
    a_before, b_before = a.x.clone(), b.x.clone()
    state_difference(a, b)
    assert torch.equal(a.x, a_before)
    assert torch.equal(b.x, b_before)


def test_state_norm_matches_hairer_wanner_formula_by_hand():
    state = _mixed([2.0, -2.0])
    reference = _mixed([1.0, 1.0])
    rtol, atol = 0.1, 0.01
    expected_scale = atol + rtol * 1.0
    expected = ((2.0 / expected_scale) ** 2 + (2.0 / expected_scale) ** 2) ** 0.5 / (2 ** 0.5)
    assert state_norm(state, rtol, atol, reference=reference) == pytest.approx(expected)


def test_state_norm_defaults_reference_to_self():
    state = _mixed([1.0])
    assert state_norm(state, rtol=0.0, atol=1.0) == pytest.approx(1.0)


def test_state_norm_zero_for_a_state_with_no_integrated_tensor_fields():
    @dataclass
    class Empty(BaseState):
        label: str = constant(default='x')

    assert state_norm(Empty()) == 0.0


# --------------------------------------------------------------------------- #
# S2 / S2g -- StepHistory                                                     #
# --------------------------------------------------------------------------- #

def test_step_history_tracks_the_latest_entry():
    h = StepHistory(maxlen=3)
    e1 = HistoryEntry(t=0.0, dt=0.1, update='k0', aux=None)
    h = h.pushed(e1)
    assert h.latest is e1
    assert len(h) == 1


def test_step_history_respects_maxlen():
    h = StepHistory(maxlen=2)
    for i in range(5):
        h = h.pushed(HistoryEntry(t=float(i), dt=0.1, update=i))
    assert len(h) == 2
    assert [e.update for e in h] == [3, 4]


def test_step_history_as_prior_step_round_trips_through_unpack():
    from warpSPHIntegrators import unpack_prior_step

    h = StepHistory(maxlen=1).pushed(HistoryEntry(t=0.0, dt=0.1, update='k', aux='r'))
    k, r = unpack_prior_step(h.as_prior_step())
    assert (k, r) == ('k', 'r')


def test_step_history_restarts_on_dt_change():
    h = StepHistory(maxlen=4)
    h = h.pushed(HistoryEntry(t=0.0, dt=0.1, update='a'))
    h = h.pushed(HistoryEntry(t=0.1, dt=0.1, update='b'))
    assert len(h) == 2
    h = h.pushed(HistoryEntry(t=0.2, dt=0.05, update='c'))
    assert len(h) == 1, 'a dt change must drop stale entries rather than mix coefficients'
    assert h.latest.update == 'c'


def test_step_history_restarts_on_uid_identity_change():
    uid_a = torch.arange(4)
    uid_b = torch.arange(4)  # same values, different storage -- must still restart
    h = StepHistory(maxlen=4)
    h = h.pushed(HistoryEntry(t=0.0, dt=0.1, update='a'), uid=uid_a)
    h = h.pushed(HistoryEntry(t=0.1, dt=0.1, update='b'), uid=uid_a)
    assert len(h) == 2
    h = h.pushed(HistoryEntry(t=0.2, dt=0.1, update='c'), uid=uid_b)
    assert len(h) == 1, 'a uid identity change (resort/resize) must invalidate history'


def test_step_history_empty_latest_is_none():
    assert StepHistory(maxlen=2).latest is None
    assert StepHistory(maxlen=2).as_prior_step() is None


# --------------------------------------------------------------------------- #
# S2 end-to-end -- history= must reproduce priorStep= on a real scheme         #
# --------------------------------------------------------------------------- #

def test_history_alone_does_not_enable_reuse():
    """history= must be pure bookkeeping: it must never silently seed priorStep.

    RungeKuttaB used to derive priorStep from history internally, which switched on
    first-stage reuse (and its order cost, per reuse.py) for *every* caller that
    passed history=, including non-FSAL schemes, with none of the warnings
    integration._with_reuse_guard gives an explicit priorStep=. Fixed by keeping the
    two channels independent: history= only ever affects IntegrationResult.history.
    """
    scheme = getIntegrator('RK4')  # not FSAL: reuse would perturb the trajectory
    prob = testing.PROBLEMS['oscillator']()

    plain = prob.initial()
    via_history = prob.initial()
    history = StepHistory(maxlen=1)

    for _ in range(5):
        plain = scheme(plain, dt=0.1, f=prob.rhs).state
        r = scheme(via_history, dt=0.1, f=prob.rhs, history=history)
        via_history, history = r.state, r.history

    s1, s2 = get_reference_state(plain), get_reference_state(via_history)
    assert torch.equal(s1.x, s2.x)
    assert torch.equal(s1.u, s2.u)


def test_history_as_prior_step_reproduces_manual_priorstep_reuse():
    """Opting into reuse via history.as_prior_step() must match the existing manual path.

    This is what a caller who *does* want reuse writes: pass both history= (for
    bookkeeping) and priorStep=history.as_prior_step() (to opt in), explicitly, the
    same way integration._with_reuse_guard already expects to see priorStep.
    """
    scheme = getIntegrator('Bogacki-Shampine 3(2)')  # FSAL: reuse is lossless
    prob = testing.PROBLEMS['oscillator']()

    via_prior = prob.initial()
    prior = None
    via_history = prob.initial()
    history = StepHistory(maxlen=1)

    for _ in range(5):
        r1 = scheme(via_prior, dt=0.1, f=prob.rhs, priorStep=prior)
        via_prior = r1.state
        prior = r1.stages[-1]

        r2 = scheme(via_history, dt=0.1, f=prob.rhs,
                    history=history, priorStep=history.as_prior_step())
        via_history, history = r2.state, r2.history

    s1, s2 = get_reference_state(via_prior), get_reference_state(via_history)
    assert torch.equal(s1.x, s2.x)
    assert torch.equal(s1.u, s2.u)


def test_history_kwarg_is_inert_for_schemes_that_do_not_use_it(scheme):
    """Every scheme must at least tolerate history= without crashing (S5)."""
    prob = testing.PROBLEMS['oscillator']()
    result = scheme(prob.initial(), dt=0.1, f=prob.rhs, history=StepHistory(maxlen=2))
    assert result.state is not None


def test_testing_run_threads_history_without_changing_the_trajectory():
    """`history=True` in the test harness must not perturb a scheme that ignores it."""
    scheme = getIntegrator('RK4')
    prob = testing.PROBLEMS['oscillator']()
    plain = testing.run(scheme, prob, dt=0.1, T=0.5)
    with_history = testing.run(scheme, prob, dt=0.1, T=0.5, history=True)
    assert torch.equal(get_reference_state(plain).x, get_reference_state(with_history).x)


# --------------------------------------------------------------------------- #
# S3 -- IntegrationScheme metadata                                            #
# --------------------------------------------------------------------------- #

def test_every_explicit_one_step_scheme_defaults_to_that_metadata():
    """Written before any implicit or multistep scheme was registered; now scoped to
    the schemes that are actually still one-step and explicit (RK family, Euler,
    Verlet family, ...)."""
    for s in IntegrationSchemes:
        if s.implicit or s.steps > 1:
            continue
        assert s.implicit is False
        assert s.steps == 1
        assert s.stiffly_accurate is False


def test_integration_scheme_accepts_the_new_metadata_fields():
    s = IntegrationScheme(
        function=lambda *a, **k: None, name='probe', identifier=None, order=2,
        implicit=True, steps=1, stiffly_accurate=True, stability='L', startup_order=None,
    )
    assert s.implicit and s.stiffly_accurate and s.stability == 'L'


# --------------------------------------------------------------------------- #
# S4 -- NonlinearSolver / FixedPointSolver                                    #
# --------------------------------------------------------------------------- #

def test_fixed_point_solver_converges_on_a_contraction():
    """y = 0.5*y + 1 has fixed point y=2; Picard converges since |0.5| < 1."""
    solver = FixedPointSolver(iterations=40)
    result = solver.solve(lambda y: 0.5 * y + 1.0, y0=0.0)
    assert result.y == pytest.approx(2.0, abs=1e-9)
    assert result.iterations == 40
    assert result.converged is True, 'no tol given: converged means "ran its fixed schedule"'


def test_fixed_point_solver_respects_the_configured_iteration_count():
    calls = []
    solver = FixedPointSolver(iterations=3)
    solver.solve(lambda y: calls.append(y) or y + 1, y0=0)
    assert len(calls) == 3


def test_fixed_point_solver_iterations_kwarg_overrides_the_default():
    calls = []
    solver = FixedPointSolver(iterations=3)
    solver.solve(lambda y: calls.append(y) or y + 1, y0=0, iterations=5)
    assert len(calls) == 5


def test_fixed_point_solver_early_exits_when_tol_is_met():
    norm = lambda a, b: abs(a - b)
    result = FixedPointSolver(iterations=100).solve(
        lambda y: 0.5 * y + 1.0, y0=0.0, norm=norm, tol=1e-8)
    assert result.converged is True
    assert result.iterations < 100


def test_fixed_point_solver_reports_not_converged_when_tol_is_never_met():
    norm = lambda a, b: abs(a - b)
    result = FixedPointSolver(iterations=2).solve(
        lambda y: 0.5 * y + 1.0, y0=0.0, norm=norm, tol=1e-300)
    assert result.converged is False
    assert result.iterations == 2


def test_relaxed_fixed_point_solver_converges_on_a_state_contraction():
    solver = RelaxedFixedPointSolver(relaxation=0.5, iterations=80)
    initial = _mixed([0.0])

    def step(state):
        out = state.initializeNewState()
        out.x = 0.5 * state.x + 1.0
        return out

    result = solver.solve(step, initial)
    assert result.y.x.tolist() == pytest.approx([2.0], abs=1e-8)
    assert result.diagnostics.termination == 'fixed_iterations'
    assert result.diagnostics.gmres_iterations == 0
