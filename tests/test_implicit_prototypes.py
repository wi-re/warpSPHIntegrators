"""Unregistered s=3 numerical gates and stage-aware preconditioning."""
from pathlib import Path
import sys
import math
import numpy as np
import pytest
import torch
from warpSPHIntegrators import TensorState, tensor_rhs, get_scheme
from warpSPHIntegrators.state import BlockState
from warpSPHIntegrators.solvers import stage_block_preconditioner
import warpSPHIntegrators.testing as testing
from warpSPHIntegrators.analysis.stability import rk_stability_function
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'benchmarks'))
from collocation_prototypes import prototype_tableau,prototype_scheme

OPTS=dict(matvec='jvp',tol=1e-12,newton_tol=1e-11)


@pytest.mark.parametrize('family',['gauss3','radau3'])
def test_collocation_conditions_stability_and_stiff_accuracy(family):
    a,b,c,_=prototype_tableau(family)
    for q in range(3):
        np.testing.assert_allclose(a@c**q,c**(q+1)/(q+1),atol=2e-14)
    order=6 if family=='gauss3' else 5
    for q in range(order):
        assert b@c**q==pytest.approx(1/(q+1),abs=2e-14)
    tab=prototype_tableau(family)
    for real in (0.,-.1,-1.,-10.,-100.):
        for imag in (0.,1.,10.,100.):
            assert abs(rk_stability_function(tab,complex(real,imag)))<=1+2e-12
    if family=='radau3':
        np.testing.assert_allclose(a[-1],b,atol=2e-14)
        assert abs(rk_stability_function(tab,-1e6))<1e-5
    else:
        np.testing.assert_allclose(b[:,None]*a+b[None,:]*a.T,np.outer(b,b),atol=2e-14)
        np.testing.assert_allclose(a+a[::-1,::-1],np.broadcast_to(b,a.shape),atol=2e-14)
        np.testing.assert_allclose(c+c[::-1],1.,atol=2e-14)
        assert not np.allclose(a,a.T)
        assert abs(abs(rk_stability_function(tab,100j))-1)<2e-12


@pytest.mark.parametrize('family,order',[('gauss3',6),('radau3',5)])
def test_prototype_order_on_nonlinear_flow_without_claiming_a_companion(family,order):
    errors=[]
    for dt in (.25,.125,.0625):
        state=TensorState(torch.ones(1,dtype=torch.float64))
        for _ in range(round(.5/dt)):
            result=prototype_scheme(family)(state,dt,tensor_rhs(lambda t,y:y*y),solver_opts=OPTS)
            assert result.step_status.successful and result.error is None
            state=result.state
        errors.append(abs(state.y.item()-2))
    assert math.log(errors[-2]/errors[-1],2)>order-.4,(family,errors)


def test_stage_block_preconditioner_solves_the_expected_per_dof_blocks():
    n,s=5,3;a=prototype_tableau('radau3').a
    diagonal=-torch.arange(1,s*n+1,dtype=torch.float64).reshape(s,n)
    block=BlockState(tuple(TensorState(torch.ones(n,dtype=torch.float64)) for _ in range(s)))
    context=dict(dt=.2,tableau_a=a)
    vector=torch.linspace(-1,1,s*n,dtype=torch.float64)
    prec=stage_block_preconditioner(diagonal)
    actual=prec(vector,block,context).reshape(s,n)
    for j in range(n):
        matrix=torch.eye(s,dtype=torch.float64)-.2*torch.tensor(a)*diagonal[:,j].unsqueeze(0)
        assert torch.allclose(matrix@actual[:,j],vector.reshape(s,n)[:,j],atol=1e-12)
    with pytest.raises(ValueError,match='shape'):
        stage_block_preconditioner(torch.ones(n+1))(vector,block,context)


@pytest.mark.parametrize('family',['gauss3','radau3'])
def test_stage_coupling_reduces_krylov_cost_without_changing_solution(family):
    n=32;rates=torch.logspace(0,4,n,dtype=torch.float64)
    state=TensorState(torch.ones(n,dtype=torch.float64));f=tensor_rhs(lambda t,y:-rates*y)
    opts=dict(matvec='jvp',tol=1e-10,newton_tol=1e-7,gmres_maxiter=300,max_iterations=8)
    cold=prototype_scheme(family)(state,.01,f,solver_opts=opts)
    pre=prototype_scheme(family)(state,.01,f,solver_opts=dict(opts,preconditioner=stage_block_preconditioner(-rates)))
    assert pre.step_status.tolerance_converged
    assert torch.allclose(cold.state.y,pre.state.y,atol=2e-9,rtol=2e-9)
    assert pre.solver_diagnostics[0].gmres_iterations<cold.solver_diagnostics[0].gmres_iterations/5


def test_implicit_first_stage_warm_guess_preserves_answer_and_stage_context():
    scheme=get_scheme('SDIRK2');f=tensor_rhs(lambda t,y:-3*y)
    state=TensorState(torch.ones(1,dtype=torch.float64))
    first=scheme(state,.1,f,solver_opts=OPTS)
    contexts=[]
    def identity(v,state,context):contexts.append(dict(context));return v
    cold=scheme(first.state,.1,f,solver_opts=OPTS)
    warm=scheme(first.state,.1,f,warmStart=first.stages[0],solver_opts=dict(OPTS,preconditioner=identity))
    assert torch.allclose(cold.state.y,warm.state.y,atol=1e-13,rtol=1e-13)
    assert {c['stage_index'] for c in contexts}=={0,1}
    assert all(c['stage_coefficient']>0 and c['dt']==.1 for c in contexts)
    with pytest.raises(ValueError,match='implicit first stage'):
        get_scheme('TR-BDF2')(state,.1,f,warmStart=first.stages[0])


@pytest.mark.parametrize('family',['gauss3','radau3'])
def test_prototype_state_and_rhs_parameter_gradients_match_finite_difference(family):
    def advance(y0,rate,track):
        y=torch.tensor([y0],dtype=torch.float64,requires_grad=track)
        k=torch.tensor(rate,dtype=torch.float64,requires_grad=track)
        result=prototype_scheme(family)(TensorState(y),.1,tensor_rhs(lambda t,y:-k*y),solver_opts=dict(OPTS,preconditioner=stage_block_preconditioner((-k).reshape(1))))
        return result.state.y.sum(),y,k
    out,y,k=advance(1.,2.,True);gy,gk=torch.autograd.grad(out,(y,k))
    eps=1e-4
    finite=(advance(1.,2.+eps,False)[0]-advance(1.,2.-eps,False)[0])/(2*eps)
    assert gk.item()==pytest.approx(finite.item(),rel=1e-7,abs=1e-10)
    assert gy.item()==pytest.approx(math.exp(-.2),rel=2e-7)
