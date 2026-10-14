"""Source-equivalent irregular-grid Adams, adaptive memory and safe restart."""
import json
import math
from pathlib import Path
from dataclasses import dataclass

import numpy as np
import pytest
import torch

from warpSPHIntegrators import (VariableAdams, adams_weights, TensorState, tensor_rhs,
    solve, save_checkpoint, load_checkpoint, StepHistory, HistoryEntry, snapshot_state, getIntegrator)
from warpSPHIntegrators import (BaseState, integrated, copied, constant, reference_state, testing)

ORACLE=Path(__file__).resolve().parents[1]/'scripts/coefficients/wave10/variable_adams.json'


@pytest.mark.parametrize('order',[3,4,5])
@pytest.mark.parametrize('grid',['constant','irregular'])
def test_variable_coefficients_match_pinned_divided_difference_helpers(order,grid):
    source=json.loads(ORACLE.read_text())['records'][f'p{order}_{grid}']
    nodes=(np.asarray(source['times'])-source['times'][-1])/source['h']
    ab=adams_weights(nodes)
    assert np.allclose(np.r_[ab,0.],source['ab'],atol=3e-14)
    low=np.r_[0.,adams_weights(nodes[1:])]
    assert np.allclose(np.r_[ab-low,0.],source['ab_error'],atol=3e-14)
    predictor=adams_weights(nodes[1:])
    assert np.allclose(np.r_[0.,predictor,0.],source['abm_predictor'],atol=3e-14)
    am=adams_weights(np.r_[nodes[1:],1.])
    higher=adams_weights(np.r_[nodes,1.])
    assert np.allclose(np.r_[0.,am],source['abm'],atol=3e-14)
    assert np.allclose(higher-np.r_[0.,am],source['abm_source_error'],atol=3e-14)
    for k in range(order):assert ab@nodes**k==pytest.approx(1/(k+1),abs=2e-13)


def force(t,y):return -y+.2*y*y-.2*math.exp(-2*t)  # exact y=exp(-t)


@pytest.mark.parametrize('order',[3,4,5])
@pytest.mark.parametrize('kind',['ab','abm'])
def test_nonlinear_nonautonomous_order_on_alternating_step_sizes(order,kind):
    errors=[]
    for count in (12,24,48):
        method=VariableAdams(order,kind);state=TensorState(torch.ones(1,dtype=torch.float64));history=None
        for i in range(count):
            h=(.5 if i%2==0 else 1.5)/count
            result=method(state,h,tensor_rhs(force),history=history)
            assert result.step_status.successful
            state,history=result.state,result.history
        assert len(history)==order-1
        errors.append(abs(state.y.item()-math.exp(-1)))
    measured=math.log(errors[-2]/errors[-1],2)
    assert measured==pytest.approx(order,abs=.35),(errors,measured)


@pytest.mark.parametrize('kind',['ab','abm'])
def test_adaptive_trials_reject_transactionally_and_tolerance_refines_accuracy(kind):
    errors=[];counts=[]
    for tolerance in (1e-4,1e-6,1e-8):
        exponents=[]
        result=solve(VariableAdams(3,kind),TensorState(torch.ones(1,dtype=torch.float64)),2.,
            tensor_rhs(force),dt=1.,adaptive=True,rtol=tolerance,atol=tolerance*.01,
            callback=lambda record:exponents.append(record.result.estimate_order))
        assert result.successful and result.rejected_steps>0
        assert 1<=len(result.history)<=2 and result.history.context['method']=='variable_adams.v1'
        assert (3 if kind=='ab' else 4) in exponents
        errors.append(abs(result.state.y.item()-math.exp(-2)));counts.append(result.accepted_steps)
    assert errors[0]>errors[1]>errors[2] and counts[0]<counts[1]<counts[2]
    assert errors[-1]<1e-5


def test_startup_estimator_exponent_and_warm_history_survive_ratio_changes():
    method=VariableAdams(3,'abm');state=TensorState(torch.ones(1,dtype=torch.float64));history=None
    seen=[]
    for h in (.05,.1,.07,.1):
        result=method(state,h,tensor_rhs(force),history=history)
        seen.append(result.estimate_order)
        state,history=result.state,result.history
    assert seen==[2,3,4,4]
    assert [e.dt for e in history]==[.07,.1]
    reset=method(state,.01,tensor_rhs(force),history=history)
    assert reset.estimate_order==2 and len(reset.history)==1
    explicit=method(state,.1,tensor_rhs(force),history=history,restart_history=True)
    assert explicit.estimate_order==2 and len(explicit.history)==1


@pytest.mark.parametrize('kind',['ab','abm'])
def test_fixed_and_adaptive_portable_restart_preserve_variable_history(tmp_path,kind):
    method=VariableAdams(4,kind);rhs=tensor_rhs(force)
    first=solve(method,TensorState(torch.ones(1,dtype=torch.float64)),.3,rhs,dt=.05)
    path=tmp_path/'adams.pt';save_checkpoint(first.checkpoint,path)
    loaded=load_checkpoint(path,map_location='cpu')
    assert loaded.history.context==first.history.context
    resumed=solve(method,None,.65,rhs,restart=loaded)
    whole=solve(method,TensorState(torch.ones(1,dtype=torch.float64)),.65,rhs,dt=.05)
    assert torch.allclose(resumed.state.y,whole.state.y,atol=2e-14)
    adaptive=solve(method,TensorState(torch.ones(1,dtype=torch.float64)),.5,rhs,dt=.2,
        adaptive=True,rtol=1e-6,atol=1e-9)
    save_checkpoint(adaptive.checkpoint,path)
    a=solve(method,None,1.,rhs,restart=load_checkpoint(path))
    b=solve(method,None,1.,rhs,restart=adaptive.checkpoint)
    assert torch.allclose(a.state.y,b.state.y,atol=2e-14)


def test_input_history_and_parameter_gradients_across_restart():
    def forward(value,tracking):
        q=torch.ones(1,dtype=torch.float64,requires_grad=tracking)
        k=torch.tensor(value,dtype=torch.float64,requires_grad=tracking)
        method=VariableAdams(4,'abm');rhs=tensor_rhs(lambda t,y:-k*y)
        initial=TensorState(q)
        first=solve(method,initial,.2,rhs,dt=.025)
        saved=snapshot_state(first.history)
        result=solve(method,None,.5,rhs,restart=first.checkpoint)
        assert initial.t==0 and q.item()==1
        assert all(torch.equal(a.update.dy,b.update.dy) for a,b in zip(first.history,saved))
        return result.state.y.sum(),q,k
    output,q,k=forward(2.,True);dq,dk=torch.autograd.grad(output,(q,k));eps=1e-5
    finite=(forward(2+eps,False)[0]-forward(2-eps,False)[0])/(2*eps)
    assert dq.item()==pytest.approx(output.item(),rel=1e-10)
    assert dk.item()==pytest.approx(finite.item(),rel=2e-7,abs=1e-9)


def test_foreign_or_discontinuous_histories_reinitialize_before_consumption():
    method=VariableAdams(3);state=TensorState(torch.ones(1,dtype=torch.float64))
    foreign=StepHistory(2,[HistoryEntry(-.2,.1,None),HistoryEntry(-.1,.1,None)])
    result=method(state,.1,tensor_rhs(force),history=foreign)
    assert result.estimate_order==2 and len(result.history)==1
    valid=result.history
    first=method(result.state,.1,tensor_rhs(force),history=valid)
    discontinuous=TensorState(first.state.y,t=1.)
    fresh=method(discontinuous,.1,tensor_rhs(force),history=first.history)
    assert fresh.estimate_order==2 and len(fresh.history)==1


@pytest.mark.parametrize('order,kind,ratio',[(2,'ab',4),(6,'abm',4),(3,'wrong',4),(3,'ab',.5),(True,'ab',4)])
def test_invalid_configuration(order,kind,ratio):
    with pytest.raises(ValueError):VariableAdams(order,kind,ratio)


def test_registered_fifth_order_pece_matches_source_weights_and_two_rhs_cost():
    source=json.loads(ORACLE.read_text())['fixed_abm5'];h=.1;t=.4
    values=[];entries=[]
    for i in range(5):
        state=TensorState(torch.tensor([math.exp(-i*h)],dtype=torch.float64),t=i*h)
        update,_=tensor_rhs(force)(state,h);values.append(update.dy)
        if i<4:entries.append(HistoryEntry(i*h,h,update))
    state=TensorState(torch.tensor([math.exp(-t)],dtype=torch.float64),t=t)
    predicted=state.y+h*sum(b*v for b,v in zip(source['predictor'],reversed(values)))
    endpoint=force(t+h,predicted)
    expected=state.y+h*sum(b*v for b,v in zip(source['corrector'],[endpoint]+list(reversed(values))[:-1]))
    calls=[]
    def counted(t,y):calls.append(t);return force(t,y)
    result=getIntegrator('abm5')(state,h,tensor_rhs(counted),history=StepHistory(4,entries))
    assert torch.allclose(result.state.y,expected,atol=1e-14)
    assert len(calls)==2 and result.error is None and result.history.maxlen==4


@pytest.mark.parametrize('name',['ab3','abm4','abm5','am3'])
def test_constant_coefficient_adams_reset_before_a_changed_step(name):
    method=getIntegrator(name);state=TensorState(torch.ones(1,dtype=torch.float64));history=StepHistory(5)
    for _ in range(6):
        result=method(state,.05,tensor_rhs(force),history=history)
        state,history=result.state,result.history
    changed=method(state,.03,tensor_rhs(force),history=history)
    cold=method(state,.03,tensor_rhs(force))
    assert torch.allclose(changed.state.y,cold.state.y,atol=1e-14)
    assert len(changed.history)==1


def test_ramped_startup_estimates_have_the_reported_local_orders():
    errors=[[],[]]
    for h in (.1,.05,.025):
        method=VariableAdams(5,'abm');rhs=tensor_rhs(lambda t,y:y)
        first=method(TensorState(torch.ones(1,dtype=torch.float64)),h,rhs)
        second=method(TensorState(torch.tensor([math.exp(h)],dtype=torch.float64),t=h),h,rhs,history=first.history)
        assert (first.estimate_order,second.estimate_order)==(2,3)
        errors[0].append(abs(first.error.y.item()));errors[1].append(abs(second.error.y.item()))
    for q,values in zip((2,3),errors):assert math.log(values[-2]/values[-1],2)==pytest.approx(q,abs=.12)


@dataclass
class CacheFields(BaseState):
    x: torch.Tensor=integrated('dxdt',tags=('position',))
    u: torch.Tensor=integrated('dudt',tags=('velocity',))
    e: torch.Tensor=integrated('dedt',tags=('quantity',))
    uid: torch.Tensor=constant()
    density: torch.Tensor=copied()


@dataclass
class CacheSystem(testing.ParticleSystem):
    state: CacheFields=reference_state()
    def initializeNewState(self,*args,**kwargs):return CacheSystem(self.state.initializeNewState(),self.t)
    def preprocess(self,initial,dt,**kwargs):self.state.density=self.state.x.square();return self


def cache_force(state,dt,**kwargs):
    s=state.state
    return testing.ParticleUpdate(s.u,-s.x,s.e+1),None


def test_variable_history_uid_changes_reset_and_copied_fields_use_corrected_endpoint():
    initial=CacheSystem(CacheFields(torch.ones(2,dtype=torch.float64),torch.zeros(2,dtype=torch.float64),
        torch.zeros(2,dtype=torch.float64),torch.tensor([1,2]),torch.zeros(2,dtype=torch.float64)))
    method=VariableAdams(3);state=initial;history=None
    for h in (.05,.06,.07,.08):
        result=method(state,h,cache_force,history=history);state,history=result.state,result.history
        assert torch.equal(state.state.density,state.state.x.square())
    assert torch.equal(initial.state.density,torch.zeros(2,dtype=torch.float64))
    changed=snapshot_state(state);changed.state.uid=changed.state.uid.flip(0)
    result=method(changed,.06,cache_force,history=history)
    assert result.estimate_order==2 and len(result.history)==1
    assert torch.equal(history.context['uid'],torch.tensor([1,2]))


def test_history_context_is_explicitly_owned_and_portable_context_rejects_objects(tmp_path):
    entry=HistoryEntry(0.,.1,None)
    history=StepHistory(3,[entry],context={'method':'test'})
    assert history.pushed(HistoryEntry(.1,.2,None),preserve_dt=True,context=history.context).context==history.context
    assert len(history.pushed(HistoryEntry(.1,.2,None)))==1
    assert history.pushed(HistoryEntry(.1,.1,None)).context is None
    result=solve(VariableAdams(3),TensorState(torch.ones(1,dtype=torch.float64)),.3,tensor_rhs(force),dt=.05)
    result.checkpoint.history.context['bad']=object()
    with pytest.raises(TypeError,match='primitive'):save_checkpoint(result.checkpoint,tmp_path/'invalid.pt')


@pytest.mark.parametrize('changed',['integrated','uid'])
def test_warm_initialize_cannot_silently_invalidate_history(changed):
    initial=CacheSystem(CacheFields(torch.ones(2,dtype=torch.float64),torch.zeros(2,dtype=torch.float64),
        torch.zeros(2,dtype=torch.float64),torch.tensor([1,2]),torch.zeros(2,dtype=torch.float64)))
    method=VariableAdams(3);state=initial;history=None
    for _ in range(3):
        result=method(state,.05,cache_force,history=history);state,history=result.state,result.history
    @dataclass
    class ModifyingSystem(CacheSystem):
        def initializeNewState(self,*args,**kwargs):return ModifyingSystem(self.state.initializeNewState(),self.t)
        def initialize(self,dt,**kwargs):
            if changed=='integrated':self.state.x=self.state.x+1
            else:self.state.uid.copy_(self.state.uid.flip(0))
            return self
    edited=ModifyingSystem(state.state,state.t)
    with pytest.raises(ValueError,match='Warm Adams initialize'):
        method(edited,.05,cache_force,history=history)
    assert torch.equal(edited.state.uid,torch.tensor([1,2]))


def test_backward_variable_history_and_restart_are_consistent():
    method=VariableAdams(4,'abm');rhs=tensor_rhs(force)
    first=solve(method,TensorState(torch.ones(1,dtype=torch.float64)),-.3,rhs,dt=.025)
    result=solve(method,None,-.5,rhs,restart=first.checkpoint)
    assert result.successful and result.state.y.item()==pytest.approx(math.exp(.5),abs=1e-6)
