import pytest
import torch

from warpSPHIntegrators import getIntegrator, testing
from warpSPHIntegrators.integration import IntegrationSchemes


def test_newmark_reaches_second_order_on_the_oscillator():
    scheme = getIntegrator('Newmark')
    prob = testing.PROBLEMS['oscillator']()
    dts = testing.default_step_sizes(0.1, 5)
    order, errors = testing.convergence(scheme, prob, dts, T=2.0)
    assert order == pytest.approx(2.0, abs=0.15), (
        f'Newmark on oscillator: measured order {order:.3f}, expected 2. errors={errors}'
    )


def test_newmark_average_acceleration_variant_stays_second_order():
    scheme = getIntegrator('Newmark')
    prob = testing.PROBLEMS['oscillator']()
    dts = testing.default_step_sizes(0.1, 5)
    order, _ = testing.convergence(scheme, prob, dts, T=2.0)
    assert order == pytest.approx(2.0, abs=0.15)


IMPLICIT_SCHEMES = [scheme.name for scheme in IntegrationSchemes if scheme.implicit]


@pytest.mark.parametrize('scheme_name', IMPLICIT_SCHEMES)
def test_implicit_scheme_advances_hidden_quantity_and_feeds_it_back_into_acceleration(scheme_name):
    scheme = getIntegrator(scheme_name)
    prob = testing.PROBLEMS['oscillator']()

    def feedback_rhs(system, dt, **kwargs):
        state = testing.get_reference_state(system)
        return testing.ParticleUpdate(
            dxdt=state.u.clone(),
            dudt=-(4.0 + state.e) * state.x,
            dedt=torch.ones_like(state.e),
        ), None

    def uncoupled_rhs(system, dt, **kwargs):
        state = testing.get_reference_state(system)
        return testing.ParticleUpdate(
            dxdt=state.u.clone(),
            dudt=-4.0 * state.x,
            dedt=torch.ones_like(state.e),
        ), None

    first = scheme(prob.initial(), dt=0.1, f=feedback_rhs).state
    second = scheme(first, dt=0.1, f=feedback_rhs).state
    uncoupled_first = scheme(prob.initial(), dt=0.1, f=uncoupled_rhs).state
    uncoupled_second = scheme(uncoupled_first, dt=0.1, f=uncoupled_rhs).state
    first_state = testing.get_reference_state(first)
    second_state = testing.get_reference_state(second)
    uncoupled_second_state = testing.get_reference_state(uncoupled_second)

    assert first_state.e.tolist() == pytest.approx([0.1])
    assert second_state.e.tolist() == pytest.approx([0.2])
    assert abs(second_state.u.item() - uncoupled_second_state.u.item()) > 1e-6


def test_newmark_rejects_invalid_beta_or_gamma():
    prob = testing.PROBLEMS['oscillator']()
    for kwargs in ({'beta': -0.1}, {'gamma': -0.1}, {'beta': 1.1}, {'gamma': 1.1}):
        with pytest.raises(ValueError, match='beta|gamma'):
            getIntegrator('Newmark')(prob.initial(), dt=0.1, f=prob.rhs, **kwargs)


def test_newmark_is_registered_as_a_second_order_implicit_scheme():
    scheme = getIntegrator('Newmark')
    assert scheme.order == 2
    assert scheme.implicit is True
    assert scheme.steps == 1
