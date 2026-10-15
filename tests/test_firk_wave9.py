"""Registered s=3 coefficient/source, companion and consumer gates."""
import json
import math
from pathlib import Path
import numpy as np
import pytest
import torch
from warpSPHIntegrators import TensorState, tensor_rhs, get_scheme, solve

ORACLE=json.loads((Path(__file__).resolve().parents[1]/'scripts/coefficients/wave9/oracle.json').read_text())
NAMES=[('gaussLegendre3','Gauss-Legendre 3',2),('radauIia3','Radau IIA s=3',3)]
OPTS=dict(matvec='jvp',tol=1e-13,newton_tol=1e-9)


@pytest.mark.parametrize('key,name,companion_order',NAMES)
def test_coefficients_match_the_pinned_julia_constructor(key,name,companion_order):
    assert ORACLE['monorepo_tree']=='bae9b8e5f1d00e51c554648e5094e215164a19f8'
    tab=get_scheme(name).function.blockTableau
    for field in ('a','b','c','companion_b'):
        np.testing.assert_allclose(getattr(tab,field),ORACLE['schemes'][key][field],atol=2e-14,rtol=2e-14)
    for q in range(3):np.testing.assert_allclose(tab.a@tab.c**q,tab.c**(q+1)/(q+1),atol=2e-14)
    c=np.r_[0.,tab.c];a=np.zeros((4,4));a[1:,1:]=tab.a;bh=tab.companion_b
    assert bh.sum()==pytest.approx(1,abs=1e-14)
    assert bh@c==pytest.approx(.5,abs=1e-14)
    if companion_order==3:
        assert bh@c**2==pytest.approx(1/3,abs=1e-14)
        assert bh@a@c==pytest.approx(1/6,abs=1e-14)
    assert abs(bh@c**companion_order-1/(companion_order+1))>.01


@pytest.mark.parametrize('key,name,companion_order',NAMES)
@pytest.mark.parametrize('nonautonomous',[False,True])
def test_nonlinear_one_step_state_and_error_match_independent_dense_newton(key,name,companion_order,nonautonomous):
    def f(t,y):
        return torch.stack((-y[0]+y[1]**2+(.1*math.sin(t) if nonautonomous else 0),
                            -.4*y[1]+.2*y[0]**2+(.1*math.cos(t) if nonautonomous else 0)))
    result=get_scheme(name)(TensorState(torch.tensor([.7,-.4],dtype=torch.float64),t=.3),.1,tensor_rhs(f),solver_opts=OPTS)
    assert result.step_status.tolerance_converged
    expected=ORACLE['schemes'][key]['nonautonomous' if nonautonomous else 'autonomous']
    np.testing.assert_allclose(result.state.y.detach().numpy(),expected['state'],atol=2e-13,rtol=0)
    np.testing.assert_allclose(result.error.y.detach().numpy(),expected['error'],atol=2e-13,rtol=0)


@pytest.mark.parametrize('key,name,companion_order',NAMES)
def test_nonlinear_companion_branch_has_its_own_declared_order(key,name,companion_order):
    errors=[];main_errors=[]
    for h in (.1,.05,.025):
        result=get_scheme(name)(TensorState(torch.ones(1,dtype=torch.float64)),h,tensor_rhs(lambda t,y:y*y),solver_opts=OPTS)
        exact=1/(1-h)
        errors.append(abs(float((result.state.y-result.error.y)[0])-exact))
        main_errors.append(abs(float(result.state.y[0])-exact))
    assert math.log(errors[-2]/errors[-1],2)>companion_order+1-.3
    assert errors[-1]>100*main_errors[-1]


@pytest.mark.parametrize('key,name,companion_order',NAMES)
def test_null_rhs_evaluation_is_only_paid_when_its_weight_is_nonzero(key,name,companion_order):
    times=[]
    def f(t,y):times.append(t);return -y
    result=get_scheme(name)(TensorState(torch.ones(1,dtype=torch.float64)),.1,tensor_rhs(f),solver_opts=OPTS)
    expected=0 if key=='gaussLegendre3' else 1
    assert times.count(0.)==expected
    assert get_scheme(name).metadata.estimate_extra_stages==expected
    assert len(result.stages)==3 and result.error is not None


@pytest.mark.parametrize('key,name,companion_order',NAMES)
def test_state_parameter_and_estimate_gradients_match_finite_difference(key,name,companion_order):
    def step(value,track):
        y=torch.tensor([.7],dtype=torch.float64,requires_grad=track)
        rate=torch.tensor(value,dtype=torch.float64,requires_grad=track)
        result=get_scheme(name)(TensorState(y),.1,tensor_rhs(lambda t,y:-rate*y*y),solver_opts=OPTS)
        return result.state.y.sum()+result.error.y.sum(),y,rate
    output,y,rate=step(2.,True);gy,gp=torch.autograd.grad(output,(y,rate))
    eps=1e-4;finite=(step(2.+eps,False)[0]-step(2.-eps,False)[0])/(2*eps)
    assert math.isfinite(gy.item())
    assert gp.item()==pytest.approx(finite.item(),rel=2e-6,abs=1e-10)


@pytest.mark.parametrize('key,name,companion_order',NAMES)
def test_adaptive_registered_consumer_uses_the_measured_estimate_exponent(key,name,companion_order):
    errors=[];steps=[]
    for rtol in (1e-4,1e-6):
        result=solve(name,TensorState(torch.ones(1,dtype=torch.float64)),1.,tensor_rhs(lambda t,y:-3*y),
                     dt=.2,adaptive=True,rtol=rtol,atol=rtol*1e-3,solver_opts=OPTS)
        assert result.successful and result.state.t==1.
        errors.append(abs(result.state.y.item()-math.exp(-3)))
        steps.append(result.accepted_steps)
    assert errors[-1]<errors[0] and errors[-1]<1e-6
    assert steps[-1]>steps[0]
