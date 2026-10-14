"""Stage-parameterized collocation: source, algebra, dynamics and run ownership."""
import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch

from warpSPHIntegrators import (Collocation, TensorState, tensor_rhs, solve, getIntegrator,
    BlockState, stage_schur_preconditioner)

ORACLE=Path(__file__).resolve().parents[1]/'scripts/coefficients/wave9/higher_stages.json'
OPTIONS=dict(matvec='jvp',tol=1e-13,newton_tol=1e-9)


@pytest.mark.parametrize('family',['gauss','radau'])
@pytest.mark.parametrize('stages',range(2,10))
def test_collocation_source_and_order_identities(family,stages):
    method=Collocation(family,stages);tab=method.tableau
    source=json.loads(ORACLE.read_text())['schemes'][f'{family}{stages}']
    for name in ('a','b','c','companion_b'):
        assert np.allclose(getattr(tab,name),source[name],atol=4e-10,rtol=2e-11),name
    assert np.all(tab.b>0) and np.all(np.diff(tab.c)>0)
    # Collocation C(s) and quadrature B(p) imply the nonlinear order; these
    # identities avoid trying to measure order 18 below a float64 solve floor.
    for k in range(1,stages+1):
        assert np.allclose(tab.a@tab.c**(k-1),tab.c**k/k,atol=2e-14,rtol=2e-13)
    for k in range(1,method.order+1):
        assert tab.b@tab.c**(k-1)==pytest.approx(1/k,abs=2e-14)
    nodes=np.r_[0.,tab.c]
    for k in range(1,method.metadata.embedded_order+1):
        assert tab.companion_b@nodes**(k-1)==pytest.approx(1/k,abs=2e-11)
    degree=method.metadata.embedded_order
    assert abs(tab.companion_b@nodes**degree-1/(degree+1))>1e-8
    if family=='gauss':
        assert np.allclose(np.diag(tab.b)@tab.a+tab.a.T@np.diag(tab.b),np.outer(tab.b,tab.b),atol=2e-14)
        assert np.allclose(tab.a+tab.a[::-1,::-1],tab.b[None,:],atol=2e-14)
    else:assert np.array_equal(tab.a[-1],tab.b)
    with pytest.raises(ValueError):tab.a[0,0]=0


@pytest.mark.parametrize('family',['gauss','radau'])
@pytest.mark.parametrize('stages',[4,5,7,9])
def test_propagated_linear_order_and_stability(family,stages):
    method=Collocation(family,stages);tab=method.tableau
    coefficient=np.ones(stages)
    for k in range(1,method.order+1):
        scaled=float(tab.b@coefficient)*math.factorial(k)
        assert scaled==pytest.approx(1.,rel=3e-8)
        coefficient=tab.a@coefficient
    def amplification(z):return 1+z*(tab.b@np.linalg.solve(np.eye(stages)-z*tab.a,np.ones(stages)))
    for real in (-.1,-1.,-100.):
        for imag in (0.,1.,20.):assert abs(amplification(complex(real,imag)))<=1+2e-12
    if family=='radau':assert abs(amplification(-1e9))<1e-7
    else:assert abs(abs(amplification(10j))-1)<1e-12


@pytest.mark.parametrize('family,stages',[('gauss',4),('gauss',5),('radau',4),('radau',5)])
def test_main_and_companion_rates_on_nonautonomous_polynomial(family,stages):
    method=Collocation(family,stages);p=method.order;q=method.estimate_order
    # Distinguish main and estimator orders with an exactly integrable force;
    # no solver floor or reference integration contaminates this rate test.
    tab=method.tableau;main=[];errors=[]
    for h in (.4,.2,.1):
        values=(h*tab.c)**p
        main.append(abs(h*(tab.b@values)-h**(p+1)/(p+1)))
        values=(h*tab.c)**(q-1)
        errors.append(abs(h*(tab.b@values-tab.companion_b[1:]@values)))
    assert math.log(main[0]/main[1],2)==pytest.approx(p+1,abs=1e-8)
    assert math.log(errors[0]/errors[1],2)==pytest.approx(q,abs=1e-8)


@pytest.mark.parametrize('family',['gauss','radau'])
def test_actual_step_adaptive_restart_gradients_and_input_ownership(family):
    method=Collocation(family,4)
    q=torch.tensor([1.],dtype=torch.float64,requires_grad=True)
    k=torch.tensor(2.,dtype=torch.float64,requires_grad=True)
    rhs=tensor_rhs(lambda t,y:-k*y)
    initial=TensorState(q)
    first=solve(method,initial,.3,rhs,dt=.1,solver_opts=OPTIONS)
    continued=solve(method,None,.5,rhs,restart=first.checkpoint,solver_opts=OPTIONS)
    assert continued.successful and initial.t==0 and q.item()==1
    assert continued.state.y.item()==pytest.approx(math.exp(-1),abs=1e-10)
    dq,dk=torch.autograd.grad(continued.state.y.sum(),(q,k))
    assert dq.item()==pytest.approx(math.exp(-1),abs=1e-9)
    assert dk.item()==pytest.approx(-.5*math.exp(-1),abs=1e-9)
    adaptive=solve(method,TensorState(q.detach()),1.,tensor_rhs(lambda t,y:-2*y),dt=.5,
                   adaptive=True,rtol=1e-6,atol=1e-9,solver_opts=OPTIONS)
    assert adaptive.successful and adaptive.rejected_steps>0
    assert adaptive.state.y.item()==pytest.approx(math.exp(-2),abs=1e-7)


def test_existing_three_stage_maps_are_preserved():
    for family,name in (('gauss','Gauss-Legendre 3'),('radau','Radau IIA s=3')):
        rhs=tensor_rhs(lambda t,y:-y+y*y*.1+math.sin(t))
        state=TensorState(torch.tensor([.7],dtype=torch.float64),t=.3)
        factory=Collocation(family,3)(state,.1,rhs,solver_opts=OPTIONS)
        registered=getIntegrator(name)(state,.1,rhs,solver_opts=OPTIONS)
        assert torch.allclose(factory.state.y,registered.state.y,atol=1e-13)
        assert torch.allclose(factory.error.y,registered.error.y,atol=1e-13)


@pytest.mark.parametrize('family',['gauss','radau'])
@pytest.mark.parametrize('stages',range(2,10))
def test_nonlinear_actual_steps_match_independent_julia_newton(family,stages):
    source=json.loads(ORACLE.read_text())['schemes'][f'{family}{stages}']
    for nonaut,label in ((False,'autonomous'),(True,'nonautonomous')):
        def force(t,y):
            return torch.stack((-y[0]+y[1]**2+(.1*math.sin(t) if nonaut else 0),
                -.4*y[1]+.2*y[0]**2+(.1*math.cos(t) if nonaut else 0)))
        result=Collocation(family,stages)(TensorState(torch.tensor([.7,-.4],dtype=torch.float64),t=.3),
            .1,tensor_rhs(force),solver_opts=OPTIONS)
        assert result.step_status.tolerance_converged
        assert np.allclose(result.state.y.detach().numpy(),source[label]['state'],atol=2e-13,rtol=0)
        assert np.allclose(result.error.y.detach().numpy(),source[label]['error'],atol=5e-13,rtol=0)


@pytest.mark.parametrize('family,stages',[('other',4),('gauss',1),('radau',10),('gauss',True)])
def test_invalid_factory_options(family,stages):
    with pytest.raises(ValueError):Collocation(family,stages)


def test_real_schur_blocks_match_full_coupled_spatial_solve_and_cache():
    # Known real Schur decomposition with a complex pair, an upper coupling
    # and a real singleton; no optional SciPy dependency in this backend test.
    t=torch.tensor([[.3,.4,.2],[-.1,.3,.1],[0.,0.,.5]],dtype=torch.float64)
    q,_=torch.linalg.qr(torch.tensor([[1.,2.,3.],[3.,2.,1.],[1.,3.,2.]],dtype=torch.float64))
    a=q@t@q.T
    j=torch.tensor([[-2.,1.,0.],[.3,-3.,.4],[0.,.2,-4.]],dtype=torch.float64)
    state=BlockState(tuple(TensorState(torch.ones(3,dtype=torch.float64)) for _ in range(3)))
    context=dict(tableau_a=a.numpy(),dt=.2);builds=[]
    def build(state,context):
        builds.append(state)
        def shifted(shift,v):return torch.linalg.solve(torch.eye(3,dtype=v.dtype)-shift*j.to(v.dtype),v)
        return shifted,lambda v:j@v
    preconditioner=stage_schur_preconditioner(build,lambda matrix:(t,q))
    vector=torch.linspace(-1.,1.,9,dtype=torch.float64)
    expected=torch.linalg.solve(torch.eye(9,dtype=torch.float64)-.2*torch.kron(a.contiguous(),j),vector)
    assert torch.allclose(preconditioner(vector,state,context),expected,atol=2e-14,rtol=1e-13)
    assert torch.allclose(preconditioner(2*vector,state,context),2*expected,atol=2e-14,rtol=1e-13)
    assert len(builds)==1
    with pytest.raises(ValueError,match='stage-major'):preconditioner(vector[:-1],state,context)
    bad=stage_schur_preconditioner(build,lambda matrix:(t,2*q))
    with pytest.raises(ValueError,match='orthogonal'):bad(vector,state,context)
