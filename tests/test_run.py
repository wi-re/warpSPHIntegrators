"""Run-level ownership, acceptance, history, control and failure contracts."""
from dataclasses import dataclass, field
import math
import pytest
import torch
from warpSPHIntegrators import (TensorState, tensor_rhs, solve, getIntegrator,
    RunError, StepHistory, IntegrationResult, SolveDiagnostics,
    SolveResult, testing)


def state(value=1., t=0.):
    return TensorState(torch.tensor([value],dtype=torch.float64),t=t)


def test_fixed_final_time_ownership_callback_and_backward_time():
    initial = state()
    records=[]
    def callback(record):
        records.append(record.result.state.t)
        record.result.state.y.fill_(100)  # callback owns a copy
    r=solve('RK4',initial,1.,tensor_rhs(lambda t,y:-y),dt=.3,callback=callback)
    assert r.state.t == 1.
    assert records == pytest.approx([.3,.6,.9,1.])
    assert initial.y.item() == 1.
    assert r.state.y.item() == pytest.approx(math.exp(-1),rel=1e-4)
    back=solve('RK4',r.state,0.,tensor_rhs(lambda t,y:-y),dt=.05)
    assert back.state.y.item() == pytest.approx(1.,rel=1e-4)
    assert back.state.t == 0.


@pytest.mark.parametrize('name', ['BDF2','Adams-Bashforth 3','SSPRKMSVS32'])
def test_history_threading_and_reset_before_clipped_final_step(name):
    f=tensor_rhs(lambda t,y:-y)
    r=solve(name,state(),.43,f,dt=.1,allow_stagnation=True)
    assert r.state.t == .43
    assert len(r.history) == 1  # final .03 step starts from cleared history
    assert r.history.latest.dt == pytest.approx(.03)
    assert r.state.y.item() == pytest.approx(math.exp(-.43),abs=.002)


class Transaction:
    def __init__(self):self.effects=[];self.commits=0;self.rollbacks=0
    def begin(self,state):return len(self.effects)
    def rollback(self,token):del self.effects[token:];self.rollbacks+=1
    def commit(self,token,result):self.commits+=1


@dataclass
class StatefulTensor(TensorState):
    counters: list = field(default_factory=list)
    def initialize(self,dt,**kwargs):
        self.counters.append(dt)
        return self
    def initializeNewState(self,*args,**kwargs):
        return StatefulTensor(self.state.initializeNewState(),self.t,self.counters)


def test_rejected_attempts_rollback_lifecycle_external_effects_and_callbacks():
    tx=Transaction();original=StatefulTensor(torch.ones(1,dtype=torch.float64)); callbacks=[]
    def rhs(t,y):
        tx.effects.append(t)
        return -2*y
    r=solve('Dormand-Prince 5(4)',original,1.,tensor_rhs(rhs),dt=1.,adaptive=True,
            rtol=1e-8,atol=1e-10,transaction=tx,callback=lambda record:callbacks.append(record.result.state.t))
    assert r.rejected_steps > 0
    assert tx.rollbacks == r.rejected_steps
    assert tx.commits == len(callbacks) == r.accepted_steps
    assert len(r.state.counters) == r.accepted_steps
    assert original.counters == [] and original.y.item()==1.
    assert len(tx.effects) < 7*r.attempts
    assert r.state.y.item() == pytest.approx(math.exp(-2),rel=3e-8)


def test_bad_rhs_options_rollback_once_and_preserve_caller():
    tx=Transaction()
    def rhs(t,y):
        tx.effects.append(t)
        raise ValueError('caller bug')
    original=state()
    with pytest.raises(ValueError,match='caller bug'):
        solve('RK4',original,1.,tensor_rhs(rhs),dt=.1,transaction=tx)
    assert tx.effects == [] and tx.rollbacks==1 and original.y.item()==1.


class FailedSolver:
    def solve(self,step,y0,norm=None,**opts):return SolveResult(y0,False,1)


def test_custom_solver_failure_without_diagnostics_is_observable_and_uncommitted():
    initial=state(); f=tensor_rhs(lambda t,y:-y)
    single=getIntegrator('Backward Euler (implicit)')(initial,.1,f,solver=FailedSolver())
    assert not single.step_status.successful
    tx=Transaction()
    with pytest.raises(RunError) as info:
        solve('Backward Euler (implicit)',initial,1.,f,dt=.1,solver=FailedSolver(),transaction=tx)
    assert info.value.result.accepted_steps==0
    assert info.value.result.state.t==0.
    assert tx.rollbacks==1 and tx.commits==0


def test_stagnation_is_distinct_from_tolerance_and_can_be_opted_into():
    class FloorSolver:
        def solve(self,step,y0,norm=None,**opts):
            return SolveResult(step(y0),True,1,SolveDiagnostics(termination='stagnation'))
    f=tensor_rhs(lambda t,y:y*0)
    with pytest.raises(RunError,match='precision_floor'):
        solve('Backward Euler (implicit)',state(),.1,f,dt=.1,solver=FloorSolver())
    r=solve('Backward Euler (implicit)',state(),.1,f,dt=.1,solver=FloorSolver(),allow_stagnation=True)
    assert r.successful
    assert not IntegrationResult(state(),solver_diagnostics=SolveDiagnostics(termination='stagnation')).step_status.tolerance_converged


@pytest.mark.parametrize('options', [{'max_attempts':2},{'max_rejections':2},{'dt_min':.2}])
def test_nonfinite_adaptive_attempts_are_bounded_and_never_committed(options):
    r=solve('Dormand-Prince 5(4)',state(),1.,tensor_rhs(lambda t,y:y*float('inf')),
            dt=1.,adaptive=True,raise_on_failure=False,**options)
    assert not r.successful and r.accepted_steps==0 and r.state.t==0.
    assert r.rejected_steps>0


def test_adaptive_rejects_no_estimate_history_and_invalid_tolerances():
    for name in ('RK4','BDF2'):
        with pytest.raises(ValueError):solve(name,state(),1.,tensor_rhs(lambda t,y:-y),dt=.1,adaptive=True)
    for tolerance in ({'missing':1e-5},float('nan'),-1):
        with pytest.raises(ValueError):solve('RK4',state(),1.,tensor_rhs(lambda t,y:-y),dt=.1,rtol=tolerance)


def test_accepted_trajectory_gradients_survive_snapshotting():
    y=torch.tensor([1.],dtype=torch.float64,requires_grad=True)
    rate=torch.tensor(.4,dtype=torch.float64,requires_grad=True)
    r=solve('RK4',TensorState(y),.5,tensor_rhs(lambda t,y,rate:rate*y,rate=rate),dt=.01)
    gy,gp=torch.autograd.grad(r.state.y.sum(),(y,rate))
    assert gy.item()==pytest.approx(math.exp(.2),rel=1e-8)
    assert gp.item()==pytest.approx(.5*math.exp(.2),rel=1e-8)


def test_field_tolerances_and_advanced_protocol():
    p=testing.oscillator_problem()
    r=solve('Dormand-Prince 5(4)',p.initial(),.5,p.rhs,dt=.2,adaptive=True,
            rtol={'x':1e-8,'u':1e-8},atol={'x':1e-10,'u':1e-10})
    assert r.state.t==.5 and r.successful


@pytest.mark.parametrize('name', ['ARK3(2)4L[2]SA','Gauss-Legendre 2','BDF2',
                                 'Adams-Moulton 2 (implicit)','IMEX Euler'])
def test_failed_custom_solver_propagates_through_other_implicit_drivers(name):
    r=solve(name,state(),.2,tensor_rhs(lambda t,y:-y),dt=.1,solver=FailedSolver(),raise_on_failure=False)
    startup = int(bool(getIntegrator(name).metadata.history_length))
    assert not r.successful and r.accepted_steps==startup and r.state.t==startup*.1
    assert len(r.history)==startup


def test_reuse_only_on_equal_steps_and_output_limits():
    calls=[]
    def rhs(t,y):calls.append(t);return -y
    f=tensor_rhs(rhs)
    reused=solve('Dormand-Prince 5(4)',state(),.25,f,dt=.1)
    saved=len(calls);calls.clear()
    cold=solve('Dormand-Prince 5(4)',state(),.25,f,dt=.1,reuse=False)
    assert saved==20 and len(calls)==21  # one equal-step FSAL reuse; clipped step is cold
    assert torch.equal(reused.state.y,cold.state.y)
    r=solve('RK4',state(),1.,f,dt=.1,max_steps=2,raise_on_failure=False)
    assert r.termination=='max_steps' and r.accepted_steps==2
    original=state(float('nan'))
    r=solve('RK4',original,1.,f,dt=.1,raise_on_failure=False)
    assert r.termination=='nonfinite_input' and r.attempts==0
