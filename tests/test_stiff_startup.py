"""Order-preserving stiff BDF bootstrap, failure propagation and reset."""
import math
import pytest
import torch
from warpSPHIntegrators import TensorState, tensor_rhs, get_scheme, solve
from warpSPHIntegrators.state import StepHistory
import warpSPHIntegrators.testing as testing
from warpSPHIntegrators.solvers import SolveResult

OPTS=dict(matvec='jvp',tol=1e-12,newton_tol=1e-10)


def trajectory(name,h,end=1.,startup=None,initial=1.):
    state=TensorState(torch.tensor([initial],dtype=torch.float64));history=StepHistory(get_scheme(name).metadata.history_length)
    f=tensor_rhs(lambda t,y:-100*(y-math.sin(4*t))+4*math.cos(4*t))
    peak=abs(initial)
    for _ in range(round(end/h)):
        r=get_scheme(name)(state,h,f,history=history,startup=startup,solver_opts=OPTS)
        if startup != 'explicit':
            assert r.step_status.successful and not r.step_status.precision_floor
        state,history=r.state,r.history
        peak=max(peak,abs(state.y.item()))
    return state,history,peak


@pytest.mark.parametrize('name',['BDF3','BDF4','BDF5'])
@pytest.mark.parametrize('initial',[0.,1.])
def test_implicit_bootstrap_retains_order_on_rate_100_forcing_and_transient(name,initial):
    errors=[]
    for h in (.025,.0125,.00625):
        state,_,_=trajectory(name,h,initial=initial)
        errors.append(abs(state.y.item()-math.sin(4)))
    measured=math.log(errors[-2]/errors[-1],2)
    assert measured>=get_scheme(name).order-.2,(name,errors,measured)


@pytest.mark.parametrize('name',['BDF3','BDF4','BDF5'])
def test_implicit_starter_stays_bounded_where_explicit_startup_amplifies(name):
    implicit,history,peak=trajectory(name,.1)
    explicit,_,old_peak=trajectory(name,.1,startup='explicit')
    assert peak<2 and old_peak>100
    assert abs(implicit.y.item()-math.sin(4))<.02
    assert len(history)==get_scheme(name).metadata.history_length


def test_startup_failure_diagnostics_reach_runner_and_capacity_is_validated():
    class Fail:
        def solve(self,step,y0,norm=None,**opts):return SolveResult(y0,False,0)
    initial=TensorState(torch.ones(1,dtype=torch.float64));f=tensor_rhs(lambda t,y:-y)
    r=solve('BDF5',initial,1.,f,dt=.1,solver=Fail(),raise_on_failure=False)
    assert not r.successful and r.accepted_steps==0 and len(r.history)==0
    assert initial.y.item()==1 and initial.t==0
    with pytest.raises(ValueError,match='capacity'):
        get_scheme('BDF5')(initial,.1,f,history=StepHistory(1))
    with pytest.raises(ValueError,match='startup'):
        get_scheme('BDF3')(initial,.1,f,startup='be')


def test_changed_dt_restarts_before_formula_in_single_step_calls():
    f=tensor_rhs(lambda t,y:-y);scheme=get_scheme('BDF4')
    state=TensorState(torch.ones(1,dtype=torch.float64));hist=StepHistory(3)
    for _ in range(4):
        r=scheme(state,.1,f,history=hist,solver_opts=OPTS);state,hist=r.state,r.history
    changed=scheme(state,.05,f,history=hist,solver_opts=OPTS)
    cold=scheme(state,.05,f,solver_opts=OPTS)
    assert torch.equal(changed.state.y,cold.state.y)
    assert len(changed.history)==1


def test_bootstrap_gradient_and_outer_lifecycle_are_preserved():
    from test_run import StatefulTensor
    y=torch.ones(1,dtype=torch.float64,requires_grad=True)
    rate=torch.tensor(3.,dtype=torch.float64,requires_grad=True)
    initial=StatefulTensor(y)
    f=tensor_rhs(lambda t,y:-rate*y)
    r=solve('BDF5',initial,.5,f,dt=.05,solver_opts=OPTS)
    assert len(r.state.counters)==r.accepted_steps and initial.counters==[]
    gy,gp=torch.autograd.grad(r.state.y.sum(),(y,rate))
    assert gy.item()==pytest.approx(math.exp(-1.5),rel=.001)
    assert gp.item()==pytest.approx(-.5*math.exp(-1.5),rel=.005)
