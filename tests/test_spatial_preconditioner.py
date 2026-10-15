"""Shifted spatial preconditioning against direct solves and true Newton residuals."""
import sys
from pathlib import Path
import numpy as np
import pytest
import torch
from warpSPHIntegrators import TensorState, tensor_rhs, get_scheme
from warpSPHIntegrators.state import BlockState
from warpSPHIntegrators.solvers import stage_shifted_preconditioner
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'benchmarks'))
from collocation_prototypes import prototype_scheme,prototype_tableau


def torch_factory(j,counts=None):
    def build(state,context):
        if counts is not None:counts.append(context)
        complex_j=j.to(torch.complex128)
        factors={}
        def solve(shift,vector):
            key=complex(shift.item())
            if key not in factors:
                factors[key]=torch.linalg.lu_factor(torch.eye(len(j),dtype=vector.dtype)-shift*complex_j)
            return torch.linalg.lu_solve(*factors[key],vector[:,None])[:,0]
        return solve
    return build


@pytest.mark.parametrize('family',['gauss3','radau3'])
def test_shifted_stage_transform_matches_full_spatial_block(family):
    n=7;s=3
    j=-torch.diag(torch.arange(1,n+1,dtype=torch.float64))
    j+=.2*(torch.eye(n,dtype=torch.float64).roll(1,0)+torch.eye(n,dtype=torch.float64).roll(-1,0))
    a=torch.tensor(prototype_tableau(family).a)
    state=BlockState(tuple(TensorState(torch.ones(n,dtype=torch.float64)) for _ in range(s)))
    context=dict(dt=.1,tableau_a=a);v=torch.linspace(-1,1,s*n,dtype=torch.float64);calls=[]
    pre=stage_shifted_preconditioner(torch_factory(j,calls))
    expected=torch.linalg.solve(torch.eye(s*n,dtype=torch.float64)-.1*torch.kron(a,j),v)
    assert torch.allclose(pre(v,state,context),expected,atol=1e-12,rtol=1e-12)
    pre(v,state,context);assert len(calls)==1
    pre(v,state,dict(context));assert len(calls)==2  # one cache per Newton context


def test_stage_transform_rejects_bad_shifted_operator_and_singular_basis():
    state=BlockState((TensorState(torch.ones(1,dtype=torch.float64)),)*2)
    pre=stage_shifted_preconditioner(lambda state,context:lambda shift,v:v.imag)
    with pytest.raises(ValueError,match='complex vector'):
        pre(torch.ones(2,dtype=torch.float64),state,dict(dt=.1,tableau_a=np.eye(2)))
    with pytest.raises(ValueError,match='ill-conditioned'):
        stage_shifted_preconditioner(torch_factory(torch.eye(1)))(torch.ones(2,dtype=torch.float64),state,
                dict(dt=.1,tableau_a=np.array([[1.,1.],[0.,1.]])))


@pytest.mark.parametrize('family',['gauss3','radau3'])
def test_preconditioned_flow_and_gradients_match_dense_linear_reference(family):
    n=5;j=-torch.diag(torch.arange(1,n+1,dtype=torch.float64))
    j+=.1*torch.ones(n,n,dtype=torch.float64)
    y=torch.linspace(.5,1,n,dtype=torch.float64,requires_grad=True)
    method=prototype_scheme(family)
    result=method(TensorState(y),.1,tensor_rhs(lambda t,y:j@y),solver_opts={
        'matvec':'jvp','tol':1e-12,'newton_tol':1e-9,
        'preconditioner':stage_shifted_preconditioner(torch_factory(j))})
    assert result.step_status.tolerance_converged
    a,b,_,_=method.blockTableau;s=len(b)
    matrix=torch.eye(s*n,dtype=torch.float64)-.1*torch.kron(torch.tensor(a),j)
    stage=torch.linalg.solve(matrix,y.repeat(s)).reshape(s,n)
    exact=y+.1*torch.sum(torch.tensor(b)[:,None]*(stage@j.T),dim=0)
    assert torch.allclose(result.state.y,exact,atol=1e-11,rtol=1e-11)
    actual=torch.autograd.grad(result.state.y.sum(),y,retain_graph=True)[0]
    expected=torch.autograd.grad(exact.sum(),y)[0]
    assert torch.allclose(actual,expected,atol=1e-10,rtol=1e-10)


@pytest.mark.parametrize('mode',['spatial_lu','spatial_ilu','shifted_lu','shifted_ilu'])
def test_sparse_spatial_reference_passes_the_previously_failed_diffusion_case(mode):
    pytest.importorskip('scipy')
    from implicit_scale import case
    from types import SimpleNamespace
    result=case(SimpleNamespace(n=1024,problem='diffusion',scheme='radau3',mode=mode,
                               steps=3,repeats=1,dt=.05,warm=False))
    assert result['failures']==0 and result['steps_attempted']==3
    assert max(result['krylov_iterations'])<=120
    assert result['factor_metrics']['factorizations']>0
