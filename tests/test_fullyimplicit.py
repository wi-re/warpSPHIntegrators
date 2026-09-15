"""The coupled fully implicit block driver (NOTES.md S3.18).

Gauss-Legendre 2 (order 4, A-stable, symplectic) and Radau IIA s=2 (order 3,
L-stable, stiffly accurate) solve their coupled stage equations as ONE s-by-s
block system with a single JFNK solve per step (``fullyimplicit.py``). The
state of that solve is a ``BlockState`` (one substate per stage), and this file
pins:

* the ``BlockState`` bridges in ``fields.py`` -- the substate-major ``i:name``
  naming, the pooled norm, and round-trips through flatten/unflatten;
* the two tableaus' analytic properties (collocation, stiff accuracy,
  companion relations) and their stability functions, with exact values
  (``4/11`` at ``z = -1`` for Radau, etc.) as coefficient-drift gates;
* the measured order of the full driver (4 and 3) on nonlinear problems, and
  the embedded estimate (the null-stage order-2 companion) at rate 3 for both;
* the JVP matvec of the block system against a hand-computed dense Jacobian --
  the off-diagonal ``a_ij`` couplings are what make this a block test rather
  than two independent 1x1 solves;
* scale (a 256-DOF stiff advection step), warm starts, diagnostics, and the
  reuse classification ("none: the block is one joint system").
"""

import math

import pytest
import torch

from warpSPHIntegrators import (
    BlockState,
    JFNKSolver,
    StepHistory,
    flatten_integrated,
    getIntegrator,
    get_reference_state,
    integrated_field_names,
    state_difference,
    state_norm,
    testing,
    unflatten_integrated,
)
from warpSPHIntegrators.enums import IntegrationSchemeType
from warpSPHIntegrators.fields import replace_integrated_fields
from warpSPHIntegrators.fullyimplicit import getBlockTableau
from warpSPHIntegrators.jfnk import fd_matvec, jvp_matvec
from warpSPHIntegrators.reuse import (is_fsal, step_reuse_analysis,
                                      step_reuse_order, supports_step_reuse)
from warpSPHIntegrators.stability import rk_stability_function
from warpSPHIntegrators.testing import ParticleState, ParticleSystem
from warpSPHIntegrators.util import updateStateEuler, updateStep

# --------------------------------------------------------------------------- #
# The BlockState bridges                                                      #
# --------------------------------------------------------------------------- #


def _particle_system(t=0.0, x=1.0, u=0.0, e=0.0):
    return ParticleSystem(
        state=ParticleState(
            x=torch.tensor([x], dtype=torch.float64),
            u=torch.tensor([u], dtype=torch.float64),
            e=torch.tensor([e], dtype=torch.float64),
            m=torch.tensor([1.0], dtype=torch.float64)),
        t=t)


def test_flatten_is_substate_major_and_round_trips():
    """'0:x', '0:u', '0:e', '1:x', ... -- the solver's cost model (NOTES.md
    S3.4) is per-substate, and the JVP layout must match flatten exactly."""
    block = BlockState((_particle_system(t=0.0, x=1.0, u=2.0),
                        _particle_system(t=0.5, x=3.0, u=4.0)))
    names = integrated_field_names(block)
    assert names == ['0:x', '0:u', '0:e', '1:x', '1:u', '1:e'], names

    flat = flatten_integrated(block)
    assert flat.tolist() == [1.0, 2.0, 0.0, 3.0, 4.0, 0.0], flat.tolist()

    back = unflatten_integrated(flat, block)
    assert isinstance(back, BlockState) and len(back.states) == 2
    for s0, s1 in zip(block.states, back.states):
        for name in ('x', 'u', 'e', 'm'):
            assert torch.equal(
                getattr(get_reference_state(s0), name),
                getattr(get_reference_state(s1), name)), f'{name} not preserved'
        assert s0.t == s1.t


def test_block_state_difference_is_a_block_and_pools():
    """Per-substate differences, and the pooled norm sums the squared parts
    across every substate BEFORE the root (one shared norm, one JFNK
    residual), with the scale set by a BlockState reference of ones."""
    a = BlockState((_particle_system(x=1.0, u=2.0), _particle_system(x=3.0, u=4.0)))
    b = BlockState((_particle_system(x=1.1, u=2.2), _particle_system(x=3.9, u=4.4)))
    diff = state_difference(a, b)
    assert isinstance(diff, BlockState) and len(diff.states) == 2
    for da, (ex, eu) in zip(diff.states, ((-0.1, -0.2), (-0.9, -0.4))):
        ref = get_reference_state(da)
        assert ref.x.tolist() == pytest.approx([ex]), ref.x.tolist()
        assert ref.u.tolist() == pytest.approx([eu]), ref.u.tolist()
        assert ref.e.tolist() == pytest.approx([0.0])

    # Weighted RMS with scale atol + rtol*|reference| = 1 (rtol=1, atol=0,
    # reference ones): sqrt((0.1^2 + 0.2^2 + 0 + 0.9^2 + 0.4^2 + 0) / 6).
    ref = BlockState((_particle_system(x=1.0, u=1.0, e=1.0),
                      _particle_system(x=1.0, u=1.0, e=1.0)))
    norm = state_norm(diff, 1.0, 0.0, reference=ref)
    assert norm == pytest.approx(math.sqrt(1.02 / 6.0), rel=1e-12)
    # NOT the per-substate sum of two pooled roots.
    assert norm != pytest.approx(math.sqrt(0.05 / 3.0) + math.sqrt(0.97 / 3.0))


def test_block_state_difference_rejects_mismatched_shapes():
    a = BlockState((_particle_system(), _particle_system()))
    with pytest.raises(TypeError):
        state_difference(a, _particle_system())
    with pytest.raises(TypeError):
        state_difference(a, BlockState((_particle_system(),)))
    with pytest.raises(TypeError):
        state_difference(_particle_system(), a)
    with pytest.raises(TypeError):
        state_difference(BlockState((_particle_system(),)), a)


def test_block_state_norm_rejects_mismatched_reference():
    a = BlockState((_particle_system(), _particle_system()))
    with pytest.raises(TypeError):
        state_norm(a, 1.0, 0.0, reference=_particle_system())
    with pytest.raises(TypeError):
        state_norm(a, 1.0, 0.0, reference=BlockState((_particle_system(),)))
    with pytest.raises(TypeError):
        state_norm(_particle_system(), 1.0, 0.0, reference=a)


def test_replace_and_unflatten_reject_bad_block_keys():
    block = BlockState((_particle_system(), _particle_system()))
    with pytest.raises(ValueError):
        replace_integrated_fields(block, {'nope:x': torch.tensor([1.0])})
    with pytest.raises(ValueError):
        replace_integrated_fields(block, {'5:x': torch.tensor([1.0])})
    with pytest.raises(ValueError):
        unflatten_integrated(torch.ones(5), block)


# --------------------------------------------------------------------------- #
# The tableaus                                                                #
# --------------------------------------------------------------------------- #


def test_block_schemes_are_registered_with_their_metadata():
    for name, expected_order, expected_stability, expected_identifier in (
            ('Gauss-Legendre 2', 4, 'A', IntegrationSchemeType.gaussLegendre2),
            ('Radau IIA s=2', 3, 'L', IntegrationSchemeType.radauIia2)):
        scheme = getIntegrator(name)
        assert scheme.identifier == expected_identifier
        assert scheme.order == expected_order
        assert scheme.stability == expected_stability
        assert scheme.implicit
        assert not scheme.fsal
        assert scheme.function.blockTableau is not None


def test_gauss_legendre_tableau_properties():
    """Gauss collocation: b is not the last row (NOT stiffly accurate), the
    companion b-bar is the min-norm order-2 rule over (k0, k1, k2), and the
    collocation identity holds. (The tableau is NOT symmetric -- a_12 != a_21;
    its symplecticity for separable Hamiltonians is pinned by measurement in
    tests/test_hamiltonian.py, not by any symmetry of the coefficients.)"""
    t = getBlockTableau('gaussLegendre2')
    c = torch.tensor([0.5 - math.sqrt(3.0) / 6.0, 0.5 + math.sqrt(3.0) / 6.0],
                     dtype=torch.float64)
    a = torch.tensor([[0.25, 0.25 - math.sqrt(3.0) / 6.0],
                      [0.25 + math.sqrt(3.0) / 6.0, 0.25]], dtype=torch.float64)
    b = torch.tensor([0.5, 0.5], dtype=torch.float64)
    b_bar = torch.tensor([1.0 / 6.0, (5.0 - math.sqrt(3.0)) / 12.0,
                          (5.0 + math.sqrt(3.0)) / 12.0], dtype=torch.float64)
    assert torch.allclose(torch.tensor(t.c), c, atol=1e-15)
    assert torch.allclose(torch.tensor(t.a), a, atol=1e-15)
    assert torch.allclose(torch.tensor(t.b), b, atol=1e-15)
    assert torch.allclose(torch.tensor(t.companion_b), b_bar, atol=1e-15)
    # Row sums and weight sums.
    assert torch.allclose(torch.tensor(t.a) @ torch.ones(2, dtype=torch.float64),
                          torch.tensor(t.c), atol=1e-15)
    assert sum(t.b) == pytest.approx(1.0, abs=1e-15)
    assert sum(t.companion_b) == pytest.approx(1.0, abs=1e-15)
    # Companion first moment: sum(b_bar_i * c_i) with c_0 = 0 must be 1/2.
    c_bar = torch.tensor([0.0, *t.c])
    assert float(torch.tensor(t.companion_b) @ c_bar) == pytest.approx(0.5, abs=1e-15)
    # NOT stiffly accurate: b is not the last row of a.
    assert not torch.allclose(torch.tensor(t.b), torch.tensor(t.a)[1], atol=1e-12)
    # Coupled and genuinely non-symmetric: a_12 != a_21 (a_12 < 0).
    assert t.a[0][1] == pytest.approx((3.0 - 2.0 * math.sqrt(3.0)) / 12.0)
    assert t.a[1][0] == pytest.approx((3.0 + 2.0 * math.sqrt(3.0)) / 12.0)
    assert t.a[0][1] != t.a[1][0]
    # The symmetries the Gauss points actually give: b_1 = b_2 = 1/2,
    # a_11 = a_22 = 1/4, c_1 + c_2 = 1.
    assert t.b[0] == pytest.approx(0.5)
    assert t.b[1] == pytest.approx(0.5)
    assert t.a[0][0] == pytest.approx(0.25)
    assert t.a[1][1] == pytest.approx(0.25)
    assert t.c[0] + t.c[1] == pytest.approx(1.0, abs=1e-15)
    # Collocation: (A c)_i = c_i^2 / 2 at the Gauss points.
    assert torch.allclose(torch.tensor(t.a) @ torch.tensor(t.c),
                          torch.tensor(t.c) ** 2 / 2.0, atol=1e-14)


def test_radau_iia_tableau_properties():
    """Radau IIA: b is the LAST row (stiffly accurate), a is coupled
    (a_12 != 0, so this is NOT a DIRK -- the whole point of the block driver),
    the companion b-bar is the min-norm order-2 rule over (k0, k1, k2), and
    the collocation identity holds at the Radau points."""
    t = getBlockTableau('radauIia2')
    c = torch.tensor([1.0 / 3.0, 1.0], dtype=torch.float64)
    a = torch.tensor([[5.0 / 12.0, -1.0 / 12.0],
                      [0.75, 0.25]], dtype=torch.float64)
    b = torch.tensor([0.75, 0.25], dtype=torch.float64)
    b_bar = torch.tensor([2.0 / 7.0, 9.0 / 28.0, 11.0 / 28.0], dtype=torch.float64)
    assert torch.allclose(torch.tensor(t.c), c, atol=1e-15)
    assert torch.allclose(torch.tensor(t.a), a, atol=1e-15)
    assert torch.allclose(torch.tensor(t.b), b, atol=1e-15)
    assert torch.allclose(torch.tensor(t.companion_b), b_bar, atol=1e-15)
    assert torch.allclose(torch.tensor(t.a) @ torch.ones(2, dtype=torch.float64),
                          torch.tensor(t.c), atol=1e-15)
    assert sum(t.b) == pytest.approx(1.0, abs=1e-15)
    assert sum(t.companion_b) == pytest.approx(1.0, abs=1e-15)
    c_bar = torch.tensor([0.0, *t.c])
    assert float(torch.tensor(t.companion_b) @ c_bar) == pytest.approx(0.5, abs=1e-15)
    # Stiffly accurate: b = last row of a.
    assert torch.allclose(torch.tensor(t.b), torch.tensor(t.a)[1], atol=1e-15)
    # Coupled: the first stage sees the second. A DIRK driver cannot take
    # this tableau at all -- that is what the block driver is for.
    assert t.a[0][1] == pytest.approx(-1.0 / 12.0)
    # Collocation: (A c)_i = c_i^2 / 2 at the Radau points {1/3, 1}.
    assert torch.allclose(torch.tensor(t.a) @ torch.tensor(t.c),
                          torch.tensor(t.c) ** 2 / 2.0, atol=1e-14)


# --------------------------------------------------------------------------- #
# The stability functions (with exact values as drift gates)                  #
# --------------------------------------------------------------------------- #


def _rk(scheme_name, z):
    scheme = getIntegrator(scheme_name)
    return complex(rk_stability_function(scheme.function.blockTableau, z))


def _max_abs_on_left_half_plane(scheme_name):
    worst = 0.0
    for i in range(-100, 101):
        for j in range(0, 101):
            z = -j * 0.1 + 1j * i * 0.1
            if z == 0:
                continue
            worst = max(worst, abs(_rk(scheme_name, z)))
    return worst


def test_radau_iia_stability_function():
    """R(z) = (1 + z/3) / (1 - 2z/3 + z^2/6): A-stable on the whole left
    half-plane, L-stable (R(inf) = 0), with the exact values pinned."""
    for z in (-1.0, -3.0, 1.0j, 2.0 + 0.0j):
        expected = (1.0 + z / 3.0) / (1.0 - 2.0 * z / 3.0 + z * z / 6.0)
        assert abs(_rk('Radau IIA s=2', z) - expected) < 1e-12, f'z = {z}'
    assert abs(_rk('Radau IIA s=2', -1.0) - 4.0 / 11.0) < 1e-12
    # R(-10) is NEGATIVE (-0.095890): L-stable decay with a sign flip, so the
    # gate pins the magnitude.
    assert abs(_rk('Radau IIA s=2', -10.0)) == pytest.approx(0.095890, abs=1e-5)
    # L-stability: strong damping.
    assert abs(_rk('Radau IIA s=2', -100.0)) < 0.1
    assert _max_abs_on_left_half_plane('Radau IIA s=2') <= 1.0 + 1e-12


def test_gauss_legendre_stability_function():
    """R(z) = (1 + z/2 + z^2/12) / (1 - z/2 + z^2/12): A-stable, on the unit
    circle for imaginary z (symplectic => no dissipation), but NOT L-stable
    (R(z) -> +1, not 0, as z -> -inf) -- the reason it is not the stiff-niche
    scheme: the stiffest modes are barely damped at all."""
    for z in (-1.0, -3.0, 1.0j, 2.0 + 0.0j):
        expected = (1.0 + z / 2.0 + z * z / 12.0) / (1.0 - z / 2.0 + z * z / 12.0)
        assert abs(_rk('Gauss-Legendre 2', z) - expected) < 1e-12, f'z = {z}'
    assert abs(_rk('Gauss-Legendre 2', -10.0) - 0.302326) < 1e-5
    # Not L-stable: R(-100) = 2353/2653 ~= 0.88692 (close to 1, not near 0).
    assert abs(_rk('Gauss-Legendre 2', -100.0)) == pytest.approx(2353.0 / 2653.0, abs=1e-5)
    # Unit modulus on the imaginary axis (the symplectic signature).
    for y in (0.1, 1.0, 5.0, 10.0):
        assert abs(abs(_rk('Gauss-Legendre 2', 1j * y)) - 1.0) < 1e-12
    assert _max_abs_on_left_half_plane('Gauss-Legendre 2') <= 1.0 + 1e-12


# --------------------------------------------------------------------------- #
# The driver: order, embedded estimate, stages                                #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize('name,order', [('Gauss-Legendre 2', 4), ('Radau IIA s=2', 3)])
@pytest.mark.parametrize('problem', ['kepler', 'forced'])
def test_driver_order_on_nonlinear_problems(name, order, problem):
    """The order is a property of the full solve (the fixed point of the
    coupled system), not of the tableau alone."""
    scheme = getIntegrator(name)
    prob = testing.PROBLEMS[problem]()
    dts = testing.default_step_sizes(0.1, 5)
    measured, _ = testing.convergence(scheme, prob, dts, 1.0)
    assert measured == pytest.approx(order, abs=0.2), \
        f'{name} on {problem}: measured order {measured}, expected {order}'


def _final_estimate_size(scheme, prob, dt, T):
    """The magnitude of the embedded error estimate at the end of [0, T]."""
    system = prob.initial()
    result = None
    for _ in range(int(round(T / dt))):
        result = scheme(system, dt=dt, f=prob.rhs)
        system = result.state
    err = get_reference_state(result.error)
    return float(err.x.abs().sum() + err.u.abs().sum())


@pytest.mark.parametrize('name', ['Gauss-Legendre 2', 'Radau IIA s=2'])
@pytest.mark.parametrize('problem', ['oscillator', 'forced', 'kepler'])
def test_companion_estimate_converges_at_rate_three(name, problem):
    """The embedded pair (null stage k_0 = f(t^n, y^n)) is order 2 for BOTH
    tableaus (the 2-stage-only order-2 pair is degenerate -- b itself -- so
    the estimate quadratures the null stage too), so main-minus-companion is
    O(h^3): the q = 3 the adaptive controller is told."""
    scheme = getIntegrator(name)
    prob = testing.PROBLEMS[problem]()
    dts = [0.1, 0.05, 0.025, 0.0125]
    T = 0.5
    sizes = [_final_estimate_size(scheme, prob, dt, T) for dt in dts]
    rates = [math.log(a / b) / math.log(2.0)
             for a, b in zip(sizes, sizes[1:])]
    assert all(abs(r - 3.0) < 0.3 for r in rates), \
        f'{name} on {problem}: estimate rates {rates}, expected 3'


def test_stages_carry_per_stage_updates_and_aux():
    """`stages` is the block's StageResult list (len = s), with the stage
    update as `update` and the raw f-evaluation as `aux` -- the shape the
    warmStart contract and the history hooks expect."""
    for name in ('Gauss-Legendre 2', 'Radau IIA s=2'):
        scheme = getIntegrator(name)
        problem = testing.oscillator_problem()
        result = scheme(problem.initial(), dt=0.05, f=problem.rhs)
        assert len(result.stages) == 2
        for stage in result.stages:
            assert stage.update is not None and stage.aux is not None


def test_warm_start_changes_the_guess_not_the_answer():
    """The warm-start path (the previous step's stages, blended through the
    block Euler) must converge to the same fixed point as the cold start --
    a guess, nothing more."""
    for name in ('Gauss-Legendre 2', 'Radau IIA s=2'):
        scheme = getIntegrator(name)
        problem = testing.oscillator_problem()
        r1 = scheme(problem.initial(), dt=0.05, f=problem.rhs)
        warm = scheme(r1.state, dt=0.05, f=problem.rhs, warmStart=r1.stages)
        cold = scheme(r1.state, dt=0.05, f=problem.rhs)
        xw = get_reference_state(warm.state).x
        xc = get_reference_state(cold.state).x
        assert (xw - xc).abs().max() < 1e-7, (
            f'{name}: warm step 2 {xw.item()} != cold step 2 {xc.item()}')
        # And the run is still right: both land on the exact oscillator.
        assert abs(xc.item() - math.cos(2.0 * 0.1)) < 1e-4


def test_warm_start_length_mismatch_is_an_error():
    for name in ('Gauss-Legendre 2', 'Radau IIA s=2'):
        scheme = getIntegrator(name)
        problem = testing.oscillator_problem()
        first = scheme(problem.initial(), dt=0.05, f=problem.rhs)
        with pytest.raises(ValueError, match='warmStart'):
            scheme(first.state, dt=0.05, f=problem.rhs, warmStart=first.stages[:1])


def test_prior_step_is_rejected_like_the_dirk_schemes():
    for name in ('Gauss-Legendre 2', 'Radau IIA s=2'):
        scheme = getIntegrator(name)
        problem = testing.oscillator_problem()
        first = scheme(problem.initial(), dt=0.05, f=problem.rhs)
        with pytest.warns(RuntimeWarning, match='does not support first-stage reuse'):
            scheme(problem.initial(), dt=0.05, f=problem.rhs,
                   priorStep=first.stages[-1])


def test_history_is_bookkeeping_only():
    """Like the DIRK family: history= is threaded through for the hooks but
    never changes the answer (the block driver has no multistep cold start)."""
    for name in ('Gauss-Legendre 2', 'Radau IIA s=2'):
        scheme = getIntegrator(name)
        problem = testing.oscillator_problem()
        plain = scheme(problem.initial(), dt=0.05, f=problem.rhs)
        with_result = scheme(problem.initial(), dt=0.05, f=problem.rhs,
                             history=StepHistory(maxlen=2))
        assert torch.equal(get_reference_state(plain.state).x,
                           get_reference_state(with_result.state).x)
        assert with_result.history is not None


# --------------------------------------------------------------------------- #
# The block JVP matvec: the dense-Jacobian gate                               #
# --------------------------------------------------------------------------- #


def _block_step_fn(problem, dt, tableau, t0=0.0):
    """The step map the driver solves, as a plain BlockState -> BlockState
    function: Y_i = y^0 + dt * sum_j a_ij f(t_i, Y_j) -- the same composition
    (updateStep + updateStateEuler) the driver itself uses, so the Jacobian
    being checked is the driver's, not a re-derivation of it."""
    a = torch.tensor(tableau.a, dtype=torch.float64)
    s = len(tableau.c)
    t_i = [t0 + float(c) * dt for c in tableau.c]
    y0 = problem.initial()

    def step_fn(Y):
        assert isinstance(Y, BlockState) and len(Y.states) == s
        ks = []
        for j in range(s):
            Y_j = Y.states[j]
            Y_j.t = t_i[j]
            k_j, _r_j = updateStep(y0, Y_j, dt, problem.rhs)
            ks.append(k_j)
        outs = []
        for i in range(s):
            out_i = y0.initializeNewState()
            for j in range(s):
                aij = float(a[i, j])
                if aij != 0.0:
                    out_i = updateStateEuler(out_i, ks[j], aij * dt, copyState=False)
            out_i.t = t_i[i]
            outs.append(out_i)
        return BlockState(tuple(outs))

    return step_fn


def _initial_block(problem, dt, tableau):
    """A cold-start BlockState at the stage times (the driver's own guess)."""
    t0 = float(problem.initial().t)
    subs = []
    for c in tableau.c:
        sub = problem.initial().initializeNewState()
        sub.t = t0 + float(c) * dt
        subs.append(sub)
    return BlockState(tuple(subs))


def test_jvp_matvec_matches_the_hand_computed_dense_jacobian():
    """On the linear oscillator the block map is exactly linear, so the JVP
    must reproduce the dense 6x6 block Jacobian to machine precision, on the
    flat (x_0, u_0, e_0, x_1, u_1, e_1) layout. The off-diagonal a_ij terms
    (a_12 = -1/12 for Radau; both off-diagonals for GL2) are what make this
    a BLOCK test: two independent 3x3 Jacobians would not see the
    cross-substate coupling at all.

    The map is Y_i = y^0 + dt * sum_j a_ij f(Y_j) with f = (u, -k x, 0), so
    G(Y) = Y - map(Y) has
        G_i.x = x_i - dt * sum_j a_ij u_j
        G_i.u = u_i + dt k sum_j a_ij x_j
        G_i.e = e_i
    and the JVP matvec returns J_G v.
    """
    k = 4.0
    dt = 0.05
    for name in ('Gauss-Legendre 2', 'Radau IIA s=2'):
        tableau = getIntegrator(name).function.blockTableau
        problem = testing.oscillator_problem(k=k)
        step_fn = _block_step_fn(problem, dt, tableau)
        y0 = _initial_block(problem, dt, tableau)

        a = torch.tensor(tableau.a, dtype=torch.float64)
        J = torch.zeros(6, 6, dtype=torch.float64)
        for i in range(2):
            for j in range(2):
                if i == j:
                    J[3 * i + 0, 3 * i + 0] = 1.0
                    J[3 * i + 1, 3 * i + 1] = 1.0
                    J[3 * i + 2, 3 * i + 2] = 1.0
                J[3 * i + 0, 3 * j + 1] -= dt * a[i, j]
                J[3 * i + 1, 3 * j + 0] += dt * k * a[i, j]
        mv = jvp_matvec(step_fn, y0)
        for seed in (0, 1, 2):
            torch.manual_seed(seed)
            v = torch.randn(6, dtype=torch.float64)
            err = (mv(v) - J @ v).abs().max()
            assert err < 1e-12, f'{name}: JVP matvec vs dense Jacobian, err {err:.3e}'


def test_jvp_and_fd_matvecs_agree_on_the_block_system():
    for name in ('Gauss-Legendre 2', 'Radau IIA s=2'):
        tableau = getIntegrator(name).function.blockTableau
        problem = testing.oscillator_problem()
        dt = 0.05
        step_fn = _block_step_fn(problem, dt, tableau)
        y0 = _initial_block(problem, dt, tableau)
        y_flat = flatten_integrated(y0)
        G_y = y_flat - flatten_integrated(step_fn(y0))
        mv_fd = fd_matvec(step_fn, y0, y_flat, G_y)
        mv_jvp = jvp_matvec(step_fn, y0)
        torch.manual_seed(0)
        v = torch.randn(6, dtype=torch.float64)
        assert torch.allclose(mv_fd(v), mv_jvp(v), rtol=1e-4, atol=1e-8), \
            f'{name}: JVP and FD matvecs disagree'


def test_jvp_solver_and_fd_solver_agree_end_to_end():
    """The block solve is one JFNK system; its answer must not depend on
    which matvec supplied the Jacobian-vector products."""
    for name in ('Gauss-Legendre 2', 'Radau IIA s=2'):
        scheme = getIntegrator(name)
        problem = testing.oscillator_problem()
        r_jvp = scheme(problem.initial(), dt=0.05, f=problem.rhs,
                       solver=JFNKSolver(matvec='jvp'))
        r_fd = scheme(problem.initial(), dt=0.05, f=problem.rhs,
                      solver=JFNKSolver(matvec='fd'))
        assert (get_reference_state(r_jvp.state).x -
                get_reference_state(r_fd.state).x).abs().max() < 1e-8, name


# --------------------------------------------------------------------------- #
# Scale, diagnostics, and the reuse classification                            #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize('name,bound', [
    # Both A-stable, so the upwind modes (z in [-200, 0]) are damped or
    # bounded. Radau IIA's L-damping leaves the step almost intact
    # (max|u| ~= 1.11, the mild overshoot of its negative a_12).
    ('Radau IIA s=2', 1.5),
    # Gauss-Legendre's tableau has a negative off-diagonal (a_12 < 0), so its
    # implicit solution is not monotone: at CFL 100 it overshoots the jump to
    # max|u| ~= 1.69 even though every per-mode amplification |R(z)| <= 1.
    # A-stability bounds it; the gate pins that bound, not monotonicity.
    ('Gauss-Legendre 2', 2.0),
])
def test_scale_gate_at_256_dof(name, bound):
    """The roadmap's Phase 6 gate: a 256-DOF stiff semi-discretised
    advection step (CFL 100 -- deep in the stiff regime, where only the
    implicit block is even admissible) runs, *converges*, and stays bounded.

    The linear solve must actually converge (``termination == 'tolerance'``,
    not 'stagnation'): the upwind operator's 256 modes give the 1536-unknown
    block system 128 complex eigenvalue pairs spread over a two-dimensional
    annulus (|lam| from ~0.93 to ~82.5), and restarted GMRES(30) does not
    converge it within the numel budget on that spectrum -- measured, and
    cross-checked against SciPy's GMRES, which fails the same way. A
    full-length Krylov space (gmres_restart = the flat block dimension)
    converges in ~256 matvecs (residual ~1e-10), so that is what the gate
    uses; a 'stagnation' exit on this system is a failed solve, not a floor.
    """
    scheme = getIntegrator(name)
    prob = testing.PROBLEMS['advection'](n=256, ic='step')
    h = 1.0 / 256
    dt = 100.0 * h  # CFL = 100
    # Flat block dimension: 2 stages x 3 integrated fields (x, u, e) x 256.
    result = scheme(prob.initial(), dt=dt, f=prob.rhs,
                    solver=JFNKSolver(matvec='jvp', max_iterations=50,
                                      gmres_restart=2 * 3 * 256))
    diag = result.solver_diagnostics[0]
    assert diag.termination == 'tolerance', (
        f'{name} at CFL 100 must converge the block solve, got '
        f'termination={diag.termination!r} residual={diag.residual!r}')
    u = get_reference_state(result.state).x
    u0 = get_reference_state(prob.initial()).x.abs().max()
    assert torch.isfinite(u).all()
    assert u.abs().max() <= bound * u0, (
        f'{name} at CFL 100 must stay bounded: max|u| = {u.abs().max():.4g} '
        f'vs u0 = {u0:.4g}')


def test_reuse_analysis_says_the_block_has_no_reusable_stage():
    """A coupled tableau has no independent first stage to splice: the
    reuse machinery must say 'none' (with the warmStart pointer), not guess."""
    for name in ('Gauss-Legendre 2', 'Radau IIA s=2'):
        scheme = getIntegrator(name)
        analysis = step_reuse_analysis(scheme)
        assert analysis.order is None
        assert not analysis.fsal
        assert not supports_step_reuse(scheme)
        assert step_reuse_order(scheme) is None
        assert is_fsal(scheme) is False
        assert 'warmStart' in analysis.reason


def test_diagnostic_reports_one_scaled_block_solve():
    """One solve per step (NOT s), with the x-s cost scaling documented in
    the diagnostics -- the roadmap's Phase 6 cost-model gate: the reported
    RHS count is a multiple of the stage count (2), one map evaluation being
    2 f-evaluations."""
    for name in ('Gauss-Legendre 2', 'Radau IIA s=2'):
        scheme = getIntegrator(name)
        problem = testing.oscillator_problem()
        result = scheme(problem.initial(), dt=0.05, f=problem.rhs,
                        solver=JFNKSolver(matvec='jvp'))
        assert len(result.solver_diagnostics) == 1
        diag = result.solver_diagnostics[0]
        assert diag is not None
        assert diag.rhs_evaluations % 2 == 0
        assert diag.termination in ('tolerance', 'stagnation')
