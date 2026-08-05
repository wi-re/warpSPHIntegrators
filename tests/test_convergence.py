"""Every registered scheme must achieve its registered convergence order.

This is the regression net for NOTES.md 2.2 and 2.3: several schemes used to evaluate
the right-hand side at the correct *state* but the wrong *time*, which is invisible on
an autonomous problem and collapses the order to 1 on a time-dependent one. So every
order assertion runs on both `oscillator` (autonomous) and `forced` (non-autonomous),
and `kepler` adds a nonlinear problem that a linear one cannot stand in for.
"""

import pytest

from conftest import ORDER_TOLERANCE, order_of


#: Splitting integrators, whose registered order holds only for a *separable*
#: Hamiltonian H = T(p) + V(q) -- that is, a force depending on position alone. With a
#: velocity-dependent force (artificial viscosity, drag, any real SPH momentum
#: equation) the splitting is no longer valid: Leap Frog and Velocity Verlet become
#: implicit and what is implemented is the explicit shortcut, and the Forest-Ruth
#: variants lose their composition property outright. All four drop to first order.
#:
#: This is a property of the schemes, not a defect in this library -- but it is a
#: sharp edge for SPH in particular, so `test_splitting_schemes_need_a_separable_force`
#: pins it down and it is documented in the README scheme table.
#:
#: Symplectic Euler is deliberately NOT in this set: its kick-drift-kick form evaluates
#: the force at a state where both position and velocity are half-advanced, so it keeps
#: second order even for a velocity-dependent force.
SEPARABLE_HAMILTONIAN_ONLY = {'Leap Frog', 'Velocity Verlet', 'PEFRL', 'VEFRL'}


@pytest.mark.parametrize('problem_name', ['oscillator', 'forced', 'kepler'])
def test_scheme_achieves_registered_order(scheme, problem_name, step_sizes):
    order, errors = order_of(scheme, problem_name, step_sizes)
    assert order is not None, f'{scheme.name} produced no usable errors on {problem_name}: {errors}'
    assert order >= scheme.order - ORDER_TOLERANCE, (
        f'{scheme.name} on {problem_name}: registered order {scheme.order}, '
        f'measured {order:.2f}. Errors: {[f"{e:.3e}" for e in errors]}'
    )


@pytest.mark.parametrize('problem_name', ['oscillator', 'forced', 'kepler'])
def test_error_decreases_monotonically(scheme, problem_name, step_sizes):
    """A halved step must not increase the error. Catches instability, not just order."""
    _, errors = order_of(scheme, problem_name, step_sizes)
    for coarse, fine, dt_c, dt_f in zip(errors, errors[1:], step_sizes, step_sizes[1:]):
        if fine < 1e-13:
            continue  # at the roundoff floor, ordering is meaningless
        assert fine < coarse, (
            f'{scheme.name} on {problem_name}: error grew from {coarse:.3e} at dt={dt_c} '
            f'to {fine:.3e} at dt={dt_f}'
        )


def test_splitting_schemes_need_a_separable_force(scheme, step_sizes):
    """On a velocity-dependent force, only the four splitting schemes lose order.

    Every other scheme must still reach its registered order, so this doubles as a
    check that nothing else quietly assumes the force sees position only.
    """
    order, errors = order_of(scheme, 'damped', step_sizes)
    assert order is not None
    if scheme.name in SEPARABLE_HAMILTONIAN_ONLY:
        assert order >= 1 - ORDER_TOLERANCE, (
            f'{scheme.name} on damped: measured {order:.2f}, expected at least first order'
        )
        assert order < 2 - ORDER_TOLERANCE, (
            f'{scheme.name} now reaches order {order:.2f} on a velocity-dependent force. '
            f'If that is a deliberate improvement, drop it from SEPARABLE_HAMILTONIAN_ONLY.'
        )
    else:
        assert order >= scheme.order - ORDER_TOLERANCE, (
            f'{scheme.name} on damped: registered order {scheme.order}, measured {order:.2f}'
        )
