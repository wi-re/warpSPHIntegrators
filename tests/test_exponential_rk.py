"""Pinned exponential programs, time augmentation, full AD and bounded actions."""
import json,math
from pathlib import Path
import pytest
import torch
from warpSPHIntegrators import ExponentialRK,TensorState,tensor_rhs,SemilinearRHS,solve
from warpSPHIntegrators.exponential_rk import SEMILINEAR,ORDERS,operator_function
SOURCE=json.loads((Path(__file__).parents[1]/'scripts/coefficients/wave11/exponential.json').read_text())
RECORDS=[r for r in SOURCE['records'] if r['method'] in ORDERS and (
    (r['method'] in SEMILINEAR and not r['krylov_branch']) or (r['method'] not in SEMILINEAR and r['augmented_time']))]

def rhs(method,problem='forced',parameter=1.):
    L=torch.tensor(SOURCE['linear_matrix'],dtype=torch.float64)*parameter
    def N(t,y):
        value=torch.stack((.1*y[0]*y[1],.2*y[0]**2))
        if problem=='forced':value=value+y.new_tensor([.2*math.sin(t),.1*math.cos(t)])
        if problem=='manufactured':
            target=y.new_tensor([1.,.4])*math.exp(-t)
            value=value-target-L@target-torch.stack((.1*target[0]*target[1],.2*target[0]**2))
        return value
    if method in SEMILINEAR:return SemilinearRHS(tensor_rhs(lambda t,y:L@y),tensor_rhs(N))
    return tensor_rhs(lambda t,y:L@y+N(t,y))

@pytest.mark.parametrize('row',RECORDS,ids=lambda r:f"{r['method']}-{r['problem']}-{r['h']}")
def test_actual_source_maps(row):
    method=row['method'];initial=row['initial'][1:] if row['augmented_time'] else row['initial']
    out=ExponentialRK(method)(TensorState(torch.tensor(initial,dtype=torch.float64),t=row['t']),row['h'],rhs(method,row['problem']))
    expected=row['output'][1:] if row['augmented_time'] else row['output']
    assert out.step_status.successful
    assert out.state.y.tolist()==pytest.approx(expected,abs=3e-12,rel=0)
    if row['error'] is not None:
        expected=row['error'][1:] if row['augmented_time'] else row['error']
        assert out.error.y.tolist()==pytest.approx(expected,abs=3e-12,rel=0)

@pytest.mark.parametrize('method',ORDERS)
def test_nonautonomous_nonlinear_order(method):
    errors=[]
    for n in (4,8,16):
        out=solve(ExponentialRK(method),TensorState(torch.tensor([1.,.4],dtype=torch.float64)),1.,rhs(method,'manufactured'),dt=1/n)
        assert out.successful
        errors.append((out.state.y-torch.tensor([1.,.4],dtype=torch.float64)*math.exp(-1)).abs().max().item())
    assert math.log2(errors[-2]/errors[-1])>ORDERS[method]-.15,(method,errors)

@pytest.mark.parametrize('method',['ETDRK4','HochOst4','Exp4','EPIRK5P1','Exprb32','Exprb43'])
def test_complete_state_and_operator_parameter_gradients(method):
    def run(q,k,grad=False):
        y=torch.tensor([q,.4],dtype=torch.float64,requires_grad=grad);p=torch.tensor(k,dtype=torch.float64,requires_grad=grad)
        out=ExponentialRK(method)(TensorState(y,t=.3),.2,rhs(method,parameter=p))
        return out.state.y.sum(),y,p
    val,y,p=run(1.,1.,True);dy,dp=torch.autograd.grad(val,(y,p));eps=1e-5
    assert dy[0].item()==pytest.approx(((run(1+eps,1)[0]-run(1-eps,1)[0])/(2*eps)).item(),rel=2e-6,abs=1e-8)
    assert dp.item()==pytest.approx(((run(1,1+eps)[0]-run(1,1-eps)[0])/(2*eps)).item(),rel=2e-6,abs=1e-8)

@pytest.mark.parametrize('method',['ETDRK3','ETDRK4','HochOst4','Exprb43'])
def test_linear_limit_zero_rates_and_stiff_scalar(method):
    y=torch.tensor([1.,.4,0.,0.],dtype=torch.float64)
    if method in SEMILINEAR:f=SemilinearRHS(tensor_rhs(lambda t,y:-1000*y),tensor_rhs(lambda t,y:torch.zeros_like(y)))
    else:f=tensor_rhs(lambda t,y:-1000*y)
    out=ExponentialRK(method)(TensorState(y),.1,f)
    assert out.step_status.successful
    assert out.state.y.abs().max()<2e-12
    assert torch.equal(out.state.y[2:],y[2:])


def test_source_failures_are_retained_and_failed_method_is_unavailable():
    rows=SOURCE['refinements']
    assert next(r for r in rows if r['method']=='EPIRK5s3' and r['augmented_time'])['order']<2.1
    assert next(r for r in rows if r['method']=='HochOst4' and r['krylov_branch'])['order']<3.2
    assert next(r for r in rows if r['method']=='HochOst4' and not r['krylov_branch'])['order']>3.9
    with pytest.raises(ValueError):ExponentialRK('EPIRK5s3')

@pytest.mark.parametrize('method',['ETDRK4','Exprb32','Exprb43'])
def test_adaptive_rejection_and_restart(method,tmp_path):
    if not ExponentialRK(method).has_error_estimate:return
    from warpSPHIntegrators import save_checkpoint,load_checkpoint
    f=rhs(method,'manufactured');scheme=ExponentialRK(method)
    first=solve(scheme,TensorState(torch.tensor([1.,.4],dtype=torch.float64)),.5,f,dt=.5,adaptive=True,rtol=1e-8,atol=1e-10)
    assert first.successful and first.rejected_steps>0
    path=tmp_path/'checkpoint.pt';save_checkpoint(first.checkpoint,path)
    resumed=solve(scheme,None,1.,f,restart=load_checkpoint(path))
    assert resumed.successful
    assert (resumed.state.y-torch.tensor([1.,.4],dtype=torch.float64)*math.exp(-1)).abs().max()<1e-6


def test_large_matrix_free_action_correct_transpose_gradient_and_budget_failure():
    n=80;L=torch.diag(torch.linspace(-1,-3,n,dtype=torch.float64))+torch.diag(torch.full((n-1,),.1,dtype=torch.float64),1)
    def f():return SemilinearRHS(tensor_rhs(lambda t,y:L@y),tensor_rhs(lambda t,y:torch.zeros_like(y)))
    y=torch.linspace(.2,1,n,dtype=torch.float64,requires_grad=True)
    out=ExponentialRK('ETDRK4',phi_dim=30)(TensorState(y),.2,f())
    assert out.step_status.successful
    ref=torch.matrix_exp(.2*L)@y
    assert torch.allclose(out.state.y,ref,rtol=0,atol=2e-12)
    grad=torch.autograd.grad(out.state.y.sum(),y)[0]
    assert torch.allclose(grad,torch.matrix_exp(.2*L).T@torch.ones_like(y),rtol=0,atol=2e-11)
    failed=ExponentialRK('ETDRK4',phi_dim=2)(TensorState(y.detach()),.2,f())
    assert not failed.step_status.successful
    p=torch.tensor(1.,dtype=y.dtype,requires_grad=True)
    with pytest.raises(ValueError,match='operator derivatives'):
        ExponentialRK('ETDRK4')(TensorState(y),.2,SemilinearRHS(tensor_rhs(lambda t,y:p*(L@y)),tensor_rhs(lambda t,y:torch.zeros_like(y))))
