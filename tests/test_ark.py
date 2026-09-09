"""Additive (IMEX) Kennedy-Carpenter ARK pair (NOTES.md S3.9 Phase 5).

Generic behaviour (state cloning, copied/ephemeral fields, stage times, kwargs
passthrough, pure-implicit convergence order, energy behaviour) is already covered
by the parametrized `scheme` fixture in every other test file -- registering two
entries in `IntegrationSchemes` pulled them into all of that for free. This file
covers what is specific to the additive driver: the two-half tableau (per-half order,
be == bi, the implicit half stiffly accurate while the explicit half is not, and the
ARK-context embedded d that differs from the standalone ESDIRK d), the IMEX split
convergence on problems that are only meaningful with a split, the copied-field
ownership rule when both callbacks are active, the two-parameter IMEX stability
function, the priorStep rejection, and the right-hand-side cost of a split step.
"""

import math
from dataclasses import dataclass

import numpy as np
import pytest
import torch

from warpSPHIntegrators import (
    BaseIntegrationSystem,
    BaseState,
    ComponentUpdateSpec,
    IMEXRHS,
    PositionUpdateSpec,
    StepHistory,
    constant,
    copied,
    ephemeral,
    get_reference_state,
    getIntegrator,
    integrated,
    reference_state,
    tagged,
    testing,
)
from warpSPHIntegrators.ark import getARKTableau
from warpSPHIntegrators.fields import update_component, update_position
from warpSPHIntegrators.stability import (
    imex_is_stable,
    imex_stability_function,
    rk_stability_function,
)

ARK_SCHEMES = ['ARK3(2)4L[2]SA', 'ARK4(3)6L[2]SA']
ARK_TABLEAUS = {  # display name -> (tableau key, explicit order, implicit order)
    'ARK3(2)4L[2]SA': ('ARK324L2SA', 2, 3),
    'ARK4(3)6L[2]SA': ('ARK436L2SA', 3, 4),
}


# --------------------------------------------------------------------------- #
# Split test problems (only meaningful with an IMEXRHS; kept local so they do   #
# not enter the generic scheme-fixture tests that pass an ordinary RHS).        #
# --------------------------------------------------------------------------- #

def _imex_transport_relaxation_problem(c: float = 0.5) -> testing.Problem:
    """x' = u (explicit transport), u' = -c u (implicit relaxation); x(0)=1, u(0)=1.

    Linear, closed form: u(t)=e^{-ct}, x(t)=1+(1-e^{-ct})/c. The explicit half reads
    the velocity the implicit half is relaxing, so the halves genuinely couple at the
    stages -- a pure single-RHS method cannot reproduce the split.
    """

    def explicit(system, dt, **kwargs):
        s = get_reference_state(system)
        return testing.ParticleUpdate(dxdt=s.u.clone(),
                                      dudt=torch.zeros_like(s.u),
                                      dedt=torch.zeros_like(s.e)), None

    def implicit(system, dt, **kwargs):
        s = get_reference_state(system)
        return testing.ParticleUpdate(dxdt=torch.zeros_like(s.u),
                                      dudt=-c * s.u,
                                      dedt=torch.zeros_like(s.e)), None

    def initial():
        one = torch.ones(1, dtype=torch.float64)
        return testing.ParticleSystem(
            state=testing.ParticleState(x=one.clone(), u=one.clone(),
                                        e=torch.zeros(1, dtype=torch.float64),
                                        m=one.clone()),
            t=0.0)

    def exact(t):
        u = math.exp(-c * t)
        x = 1.0 + (1.0 - math.exp(-c * t)) / c
        return ([x], [u])

    return testing.Problem(
        'imex_transport_relaxation',
        f"x'=u (explicit), u'=-{c} u (implicit), x(0)=1, u(0)=1",
        IMEXRHS(explicit=explicit, implicit=implicit),
        initial, exact, None, autonomous=True)


def _imex_nonlinear_relaxation_problem(mu: float = 2.0) -> testing.Problem:
    """x' = x^2 (explicit nonlinear forcing) - mu x (implicit relaxation); x(0)=1<mu.

    Bernoulli/logistic form with closed form x(t)=mu/(1+(mu/x0-1)e^{mu t}); u stays 0.
    The explicit half is genuinely nonlinear (x^2), the implicit half linear.
    """
    x0 = 1.0

    def explicit(system, dt, **kwargs):
        s = get_reference_state(system)
        return testing.ParticleUpdate(dxdt=s.x ** 2,
                                      dudt=torch.zeros_like(s.u),
                                      dedt=torch.zeros_like(s.e)), None

    def implicit(system, dt, **kwargs):
        s = get_reference_state(system)
        return testing.ParticleUpdate(dxdt=-mu * s.x,
                                      dudt=torch.zeros_like(s.u),
                                      dedt=torch.zeros_like(s.e)), None

    def initial():
        one = torch.ones(1, dtype=torch.float64)
        return testing.ParticleSystem(
            state=testing.ParticleState(x=one.clone(), u=torch.zeros(1, dtype=torch.float64),
                                        e=torch.zeros(1, dtype=torch.float64), m=one.clone()),
            t=0.0)

    def exact(t):
        x = mu / (1.0 + (mu / x0 - 1.0) * math.exp(mu * t))
        return ([x], [0.0])

    return testing.Problem(
        'imex_nonlinear_relaxation',
        f"x'=x^2 (explicit) - {mu} x (implicit), x(0)=1",
        IMEXRHS(explicit=explicit, implicit=implicit),
        initial, exact, None, autonomous=True)


# --------------------------------------------------------------------------- #
# Two-half tableau                                                            #
# --------------------------------------------------------------------------- #

def _half(c, a, b):
    class _T:
        pass
    t = _T()
    t.c, t.a, t.b = c, a, b
    return t


@pytest.mark.parametrize('name', ARK_SCHEMES)
def test_ark_tableau_row_sums_and_weight_sums(name):
    tab = getARKTableau(ARK_TABLEAUS[name][0])
    for a in (tab.a_explicit, tab.a_implicit):
        for row, c in zip(a, tab.c):
            assert row.sum() == pytest.approx(c, abs=1e-13), f'{name}: row sum != c'
    for w in (tab.b_explicit, tab.b_implicit, tab.d_explicit, tab.d_implicit):
        assert w.sum() == pytest.approx(1.0, abs=1e-13)


@pytest.mark.parametrize('name', ARK_SCHEMES)
def test_ark_tableau_halves_share_their_weights(name):
    """be == bi and de == di elementwise: the additive update is one weight vector on
    (f_exp + f_imp), not two independent ones (this is what the 'b-sums-to-2' paradox
    resolves to)."""
    tab = getARKTableau(ARK_TABLEAUS[name][0])
    assert np.allclose(tab.b_explicit, tab.b_implicit)
    assert np.allclose(tab.d_explicit, tab.d_implicit)


@pytest.mark.parametrize('name', ARK_SCHEMES)
def test_ark_explicit_half_is_strictly_lower_and_not_stiffly_accurate(name):
    tab = getARKTableau(ARK_TABLEAUS[name][0])
    assert np.allclose(np.triu(tab.a_explicit, 1), 0.0), f'{name}: explicit half not explicit'
    assert not np.allclose(tab.b_explicit, tab.a_explicit[-1]), (
        f'{name}: explicit half unexpectedly stiffly accurate')


@pytest.mark.parametrize('name', ARK_SCHEMES)
def test_ark_implicit_half_is_stiffly_accurate(name):
    tab = getARKTableau(ARK_TABLEAUS[name][0])
    assert np.allclose(tab.b_implicit, tab.a_implicit[-1]), (
        f'{name}: the "SA" in the name refers to the implicit half; b_imp != A_i[last]')


@pytest.mark.parametrize('name', ARK_SCHEMES)
def test_ark_combined_method_is_not_fsal(name):
    """The registered flag must track the combined method (not the implicit half)."""
    s = getIntegrator(name)
    assert s.stiffly_accurate is False, f'{name} registered stiffly_accurate'
    assert s.implicit is True
    assert s.reuse_order is None, f'{name} claims a reuse order'
    assert not s.fsal


@pytest.mark.parametrize('name', ARK_SCHEMES)
def test_ark_per_half_order_conditions(name):
    tab = getARKTableau(ARK_TABLEAUS[name][0])
    _, p_exp, p_imp = ARK_TABLEAUS[name]
    for weights, p in ((tab.b_explicit, p_exp), (tab.b_implicit, p_imp)):
        for r in range(1, p + 1):
            assert (weights * tab.c ** (r - 1)).sum() == pytest.approx(1.0 / r, abs=1e-12)


def test_ark324_implicit_half_is_the_registered_esdirk324():
    """a/b/c of the implicit half equal the registered standalone ESDIRK3(2)4L[2]SA."""
    from warpSPHIntegrators.dirk import getDIRKTableau
    ark = getARKTableau('ARK324L2SA')
    esd = getDIRKTableau('ESDIRK324L2SA')
    assert np.allclose(ark.a_implicit, esd.a)
    b_main = esd.b[0] if isinstance(esd.b, tuple) else esd.b
    assert np.allclose(ark.b_implicit, b_main)
    assert np.allclose(ark.c, esd.c)


def test_ark324_ark_pair_d_differs_from_standalone_esdirk_d():
    """The ARK pair shares one d across both halves; it is not the standalone ESDIRK
    d (which is order-2 on its own)."""
    from warpSPHIntegrators.dirk import getDIRKTableau
    ark = getARKTableau('ARK324L2SA')
    esd = getDIRKTableau('ESDIRK324L2SA')
    b_embed = esd.b[1] if isinstance(esd.b, tuple) else None
    assert b_embed is not None
    assert not np.allclose(ark.d_implicit, b_embed), 'ARK-pair d unexpectedly equals standalone d'


# --------------------------------------------------------------------------- #
# Convergence order                                                           #
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize('name', ARK_SCHEMES)
def test_ark_pure_implicit_limit_reaches_claimed_order(name):
    """An ordinary RHS is treated as fully implicit: the step reduces to the implicit
    half (a standalone ESDIRK of the same order)."""
    s = getIntegrator(name)
    for problem_name in ['oscillator', 'forced', 'damped']:
        order, errors = testing.convergence(s, testing.PROBLEMS[problem_name](),
                                            testing.default_step_sizes(0.1, 5), T=2.0)
        assert order == pytest.approx(s.order, abs=0.15), (
            f'{name} pure-implicit on {problem_name}: {order:.3f} vs claimed {s.order}. '
            f'errors={errors}')


@pytest.mark.parametrize('name', ARK_SCHEMES)
@pytest.mark.parametrize('problem_factory', [
    pytest.param(_imex_transport_relaxation_problem, id='transport_relaxation'),
    pytest.param(_imex_nonlinear_relaxation_problem, id='nonlinear_relaxation'),
])
def test_ark_split_reaches_claimed_combined_order(name, problem_factory):
    s = getIntegrator(name)
    problem = problem_factory()
    order, errors = testing.convergence(s, problem, testing.default_step_sizes(0.1, 5), T=2.0)
    assert order == pytest.approx(s.order, abs=0.15), (
        f'{name} split on {problem.name}: {order:.3f} vs claimed {s.order}. errors={errors}')


# --------------------------------------------------------------------------- #
# priorStep / history                                                         #
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize('name', ARK_SCHEMES)
def test_ark_rejects_prior_step_with_a_warning(name):
    s = getIntegrator(name)
    prob = testing.PROBLEMS['oscillator']()
    first = s(prob.initial(), dt=0.1, f=prob.rhs)
    with pytest.warns(RuntimeWarning, match='does not support first-stage reuse'):
        s(prob.initial(), dt=0.1, f=prob.rhs, priorStep=first.stages[-1])


@pytest.mark.parametrize('name', ARK_SCHEMES)
def test_ark_history_is_bookkeeping_only(name):
    s = getIntegrator(name)
    prob = testing.PROBLEMS['oscillator']()
    plain = s(prob.initial(), dt=0.1, f=prob.rhs).state
    with_history = s(prob.initial(), dt=0.1, f=prob.rhs, history=StepHistory(maxlen=2)).state
    assert torch.equal(get_reference_state(plain).x, get_reference_state(with_history).x)


# --------------------------------------------------------------------------- #
# Copied-field ownership when both callbacks are active                        #
# --------------------------------------------------------------------------- #

@dataclass
class SplitDensityState(BaseState):
    x: torch.Tensor = integrated('dxdt', tags=('position',))
    u: torch.Tensor = integrated('dudt', tags=('velocity',))
    e: torch.Tensor = integrated('dedt', tags=('quantity',))
    m: torch.Tensor = constant(tags=('mass',))
    density: torch.Tensor = copied(default=None)
    scratch: torch.Tensor = ephemeral(default=None)


@dataclass
class SplitDensityUpdate:
    dxdt: torch.Tensor = tagged(tags=('position_derivative',))
    dudt: torch.Tensor = tagged(tags=('velocity_derivative',))
    dedt: torch.Tensor = tagged(tags=('quantity_derivative',))


@dataclass
class SplitDensitySystem(BaseIntegrationSystem):
    state: SplitDensityState = reference_state(tags=('particles',))
    t: float = 0.0
    pre_counter: list = None

    def initializeNewState(self, *args, **kwargs):
        return SplitDensitySystem(state=get_reference_state(self).initializeNewState(),
                                  t=self.t, pre_counter=self.pre_counter)

    def preprocess(self, initialState, dt, *args, **kwargs):
        # A shared counter, so the final copied field's value identifies which
        # preprocess (implicit buffer vs. explicit clone) it came from.
        self.pre_counter[0] += 1
        s = get_reference_state(self)
        s.density = torch.full_like(s.x, float(self.pre_counter[0]))
        s.scratch = torch.full_like(s.x, -float(self.pre_counter[0]))
        return self

    def apply_position_update(self, update, spec: PositionUpdateSpec, **kwargs):
        return update_position(self, update, spec, 'position', 'position_derivative',
                               'velocity', 'velocity_derivative')

    def apply_velocity_update(self, update, spec: ComponentUpdateSpec, **kwargs):
        return update_component(self, update, spec, 'velocity', 'velocity_derivative')

    def apply_quantity_update(self, update, spec: ComponentUpdateSpec, **kwargs):
        return update_component(self, update, spec, 'quantity', 'quantity_derivative')

    def apply_state_update(self, update, spec: ComponentUpdateSpec, **kwargs):
        self.apply_position_update(
            update, PositionUpdateSpec(derivative_dt=spec.derivative_dt, blend=spec.blend), **kwargs)
        self.apply_velocity_update(update, spec, **kwargs)
        self.apply_quantity_update(update, spec, **kwargs)
        return self


def _split_density_system():
    one = torch.ones(1, dtype=torch.float64)
    return SplitDensitySystem(
        state=SplitDensityState(x=one.clone(), u=torch.zeros(1, dtype=torch.float64),
                                e=torch.zeros(1, dtype=torch.float64), m=one.clone()),
        t=0.0, pre_counter=[0])


def _split_density_rhs():
    def explicit(system, dt, **kwargs):
        s = get_reference_state(system)
        return SplitDensityUpdate(dxdt=s.u.clone(), dudt=torch.zeros_like(s.u),
                                  dedt=torch.zeros_like(s.e)), None

    def implicit(system, dt, **kwargs):
        s = get_reference_state(system)
        return SplitDensityUpdate(dxdt=torch.zeros_like(s.u), dudt=-4.0 * s.x,
                                  dedt=torch.zeros_like(s.e)), None
    return IMEXRHS(explicit=explicit, implicit=implicit)


@pytest.mark.parametrize('name', ARK_SCHEMES)
def test_ark_split_copied_field_comes_from_the_implicit_buffer(name):
    """With both callbacks active, the final copied field is taken from the implicit
    buffer (the last implicit preprocess), not the throwaway explicit clone (whose
    last preprocess is the very last one of the step and is discarded)."""
    s = getIntegrator(name)
    system = _split_density_system()
    result = s(system, dt=0.05, f=_split_density_rhs())
    final = get_reference_state(result.state)
    assert final.density is not None, f'{name}: copied field lost on a split step'
    # The explicit clone's final preprocess is the last preprocess of the step, so its
    # value is `pre_counter`; the implicit buffer's (the one that survives) is one less.
    assert float(final.density[0]) == pytest.approx(system.pre_counter[0] - 1), (
        f'{name}: copied field {float(final.density[0])} is not the implicit buffer value '
        f'(pre_counter={system.pre_counter[0]})')


@pytest.mark.parametrize('name', ARK_SCHEMES)
def test_ark_split_ephemeral_field_does_not_leak(name):
    s = getIntegrator(name)
    result = s(_split_density_system(), dt=0.05, f=_split_density_rhs())
    assert get_reference_state(result.state).scratch is None, (
        f'{name}: an ephemeral() field leaked out of a split step')


# --------------------------------------------------------------------------- #
# Two-parameter IMEX stability                                                #
# --------------------------------------------------------------------------- #

def _component_half(tableau, which):
    a = tableau.a_explicit if which == 'explicit' else tableau.a_implicit
    b = tableau.b_explicit if which == 'explicit' else tableau.b_implicit
    return _half(tableau.c, a, b)


@pytest.mark.parametrize('name', ARK_SCHEMES)
def test_ark_imex_stability_pure_limits_match_the_halves(name):
    tab = getARKTableau(ARK_TABLEAUS[name][0])
    for z in [-0.5, -2.0, -5.0, -10.0, -2.0 + 1.0j]:
        assert abs(imex_stability_function(tab, 0.0, z)
                   - rk_stability_function(_component_half(tab, 'implicit'), z)) < 1e-12
    for z in [-0.5, -1.0, -1.5, 0.5, -0.5 + 0.5j]:
        assert abs(imex_stability_function(tab, z, 0.0)
                   - rk_stability_function(_component_half(tab, 'explicit'), z)) < 1e-12


@pytest.mark.parametrize('name', ARK_SCHEMES)
def test_ark_imex_is_l_stable_along_the_implicit_axis(name):
    """The implicit half is L-stable, so damping the pure-implicit axis damps the
    additive method: |R(0, z)| -> 0 as z -> -inf."""
    tab = getARKTableau(ARK_TABLEAUS[name][0])
    assert abs(imex_stability_function(tab, 0.0, -100.0)) < 0.1


@pytest.mark.parametrize('name', ARK_SCHEMES)
def test_ark_imex_is_stable_in_the_published_region(name):
    tab = getARKTableau(ARK_TABLEAUS[name][0])
    # Mixed stiff points inside the L[2] region, plus the Dahlquist origin.
    for ze, zi in [(0.0, 0.0), (-1.0, -1.0), (-0.5, -50.0), (-1.0, -20.0)]:
        assert imex_is_stable(tab, ze, zi), f'{name} unstable at (ze,zi)=({ze},{zi})'


# --------------------------------------------------------------------------- #
# Right-hand-side cost of a split step                                        #
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize('name', ARK_SCHEMES)
def test_ark_split_counts_explicit_evaluations_per_stage(name):
    """Each stage evaluates the explicit half exactly once, so a split step makes
    s explicit RHS evaluations (the implicit half is a solved fixed point on top)."""
    s = getIntegrator(name)
    s_evals = [0]

    def explicit(system, dt, **kwargs):
        s_evals[0] += 1
        st = get_reference_state(system)
        return testing.ParticleUpdate(dxdt=st.u.clone(),
                                      dudt=torch.zeros_like(st.u),
                                      dedt=torch.zeros_like(st.e)), None

    def implicit(system, dt, **kwargs):
        st = get_reference_state(system)
        return testing.ParticleUpdate(dxdt=torch.zeros_like(st.u),
                                      dudt=-0.5 * st.u,
                                      dedt=torch.zeros_like(st.e)), None

    prob = _imex_transport_relaxation_problem().initial()
    stages = len(getARKTableau(ARK_TABLEAUS[name][0]).c)
    s(prob, dt=0.1, f=IMEXRHS(explicit=explicit, implicit=implicit))
    assert s_evals[0] == stages, (
        f'{name}: expected {stages} explicit evaluations, got {s_evals[0]}')


@pytest.mark.parametrize('name', ARK_SCHEMES)
def test_ark_split_reports_implicit_solve_diagnostics(name):
    """Every implicit stage with a nonzero diagonal records a solve; stage 0 (explicit
    first stage) records None. The diagnostics carry the residual / Krylov counts."""
    s = getIntegrator(name)
    prob = _imex_transport_relaxation_problem()
    result = s(prob.initial(), dt=0.1, f=prob.rhs)
    diag = result.solver_diagnostics
    tab = getARKTableau(ARK_TABLEAUS[name][0])
    assert diag[0] is None, f'{name}: explicit first stage should record no solve'
    for i in range(1, len(tab.c)):
        assert tab.a_implicit[i, i] != 0
        assert diag[i] is not None, f'{name}: implicit stage {i} recorded no solve'
        assert diag[i].rhs_evaluations >= 1
