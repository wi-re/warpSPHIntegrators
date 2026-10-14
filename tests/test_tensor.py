"""Ordinary tensor ODEs retain protocol parity, state ownership and gradients."""

import math
import pytest
import torch

from warpSPHIntegrators import (
    TensorState, tensor_rhs, filter_schemes, getIntegrator, StepHistory,
    IMEXRHS, SemilinearRHS, estimate_error_norm, dormand_prince_dense_output,
)


@pytest.mark.parametrize('scheme', filter_schemes(state_kind='tensor'), ids=lambda s: s.name)
def test_tensor_adaptation_advances_every_compatible_driver(scheme):
    y = torch.tensor([[1., 2.], [3., 4.]], dtype=torch.float64)
    initial = y.clone()
    state = TensorState(y, t=0.3)
    rhs = tensor_rhs(SemilinearRHS(lambda t, y: -y))
    options = {'s': 3} if scheme.metadata.required_options else {}
    history = StepHistory(maxlen=5)
    for _ in range(6):
        result = scheme(state, .001, rhs, history=history, **options)
        state, history = result.state, result.history or history
    assert state.t == pytest.approx(.306)
    assert torch.allclose(state.y, initial * math.exp(-.006), atol=2e-5, rtol=2e-5)
    assert torch.equal(y, initial)
    assert state.y.shape == y.shape and state.y.dtype == y.dtype and state.y.device == y.device


@pytest.mark.parametrize('name', ['RK4', 'Dormand-Prince 5(4)', 'Backward Euler (implicit)',
                                 'Gauss-Legendre 2'])
def test_tensor_state_and_bound_parameter_gradients_match_finite_differences(name):
    scheme = getIntegrator(name)
    y = torch.tensor([.3, .7], dtype=torch.float64, requires_grad=True)
    rate = torch.tensor(.4, dtype=torch.float64, requires_grad=True)
    def run(y, rate):
        return scheme(TensorState(y), .02, tensor_rhs(lambda t, y, rate: rate*y, rate=rate)).state.y.sum()
    loss = run(y, rate)
    gy, gp = torch.autograd.grad(loss, (y, rate))
    eps = 1e-5
    for i in range(2):
        direction = torch.zeros_like(y); direction[i] = eps
        fd = (run(y.detach()+direction, rate.detach()) - run(y.detach()-direction, rate.detach()))/(2*eps)
        assert gy[i] == pytest.approx(float(fd), rel=1e-5, abs=1e-8)
    fd = (run(y.detach(), rate.detach()+eps) - run(y.detach(), rate.detach()-eps))/(2*eps)
    assert gp.item() == pytest.approx(float(fd), rel=1e-5, abs=1e-8)


def test_tensor_rhs_preserves_split_capabilities_and_binds_parameters():
    y = torch.tensor([1.], dtype=torch.float64)
    f = tensor_rhs(IMEXRHS(lambda t, y, a: a*y, lambda t, y, a: -2*a*y), a=2.)
    assert f.provides == {'explicit', 'implicit'}
    result = getIntegrator('ARK3(2)4L[2]SA')(TensorState(y), .01, f)
    assert result.state.y.item() == pytest.approx(math.exp(-.02), abs=1e-7)
    assert result.error is not None
    assert estimate_error_norm(result) >= 0


def test_tensor_adapter_uses_actual_stage_times_and_dp5_dense_output():
    state = TensorState(torch.tensor(1., dtype=torch.float64), t=.2)
    result = getIntegrator('Dormand-Prince 5(4)')(state, .1, tensor_rhs(lambda t, y: y*0+t))
    assert result.state.y.item() == pytest.approx(1.+(.3**2-.2**2)/2)
    dense = dormand_prince_dense_output(state, result.stages, .1, .5)
    assert dense.y.item() == pytest.approx(1.+(.25**2-.2**2)/2)


@pytest.mark.parametrize('function,exception', [
    (lambda t, y: 1., TypeError),
    (lambda t, y: y.unsqueeze(0), ValueError),
    (lambda t, y: y.float(), ValueError),
])
def test_tensor_rhs_rejects_invalid_derivative_contract(function, exception):
    with pytest.raises(exception):
        getIntegrator('RK4')(TensorState(torch.ones(2, dtype=torch.float64)), .1, tensor_rhs(function))


def test_tensor_adapter_rejects_second_order_drivers_and_integer_state():
    with pytest.raises(TypeError):
        TensorState(torch.ones(2, dtype=torch.int64))
    with pytest.raises(ValueError, match='position/velocity'):
        getIntegrator('Velocity Verlet')(TensorState(torch.ones(2)), .1, tensor_rhs(lambda t,y:-y))


@pytest.mark.parametrize('name,options', [
    ('RKC2', {'s':1.5}), ('RKC2', {'lambda_max':float('nan')}),
    ('ROS3P', {'gmres_restart':0}), ('EXPRB32', {'phi_dim':0}),
    ('ETD2RK', {'phi_tol':-1}),
])
def test_family_numerical_controls_fail_before_rhs_evaluation(name, options):
    def rhs(t, y):
        pytest.fail('invalid controls must fail before RHS evaluation')
    with pytest.raises(ValueError):
        getIntegrator(name)(TensorState(torch.ones(2)), .01, tensor_rhs(rhs), **options)
