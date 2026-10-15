"""Owned algorithm selection, cache transitions and adaptive child control."""
import math
import pytest
import torch
from warpSPHIntegrators.wrappers import AlgorithmSwitch
from warpSPHIntegrators.schemes import NDF, AdaptiveNDF, VariableAdams, Extrapolation
from warpSPHIntegrators import TensorState, tensor_rhs, solve, get_scheme
from warpSPHIntegrators.state import StepHistory
from warpSPHIntegrators.runtime import save_checkpoint, load_checkpoint, snapshot_state

OPTIONS=dict(matvec='jvp',tol=1e-15,newton_tol=1e-10)


@pytest.mark.parametrize('methods',[('RK4','Implicit Midpoint'),('dormandPrince','trbdf2'),
    (NDF(3),'dormandPrince'),(VariableAdams(4,'abm'),'dormandPrince'),
    (Extrapolation('midpoint',6),'RK4')])
def test_fixed_trajectory_matches_manual_choice_and_cache_resets(methods):
    def selector(state,previous):return 0 if state.t<.3 or state.t>=.5 else 1
    switch=AlgorithmSwitch(methods,selector,'manual-v1');rhs=tensor_rhs(lambda t,y:-y+.2*y*y-.2*math.exp(-2*t))
    state=TensorState(torch.ones(1,dtype=torch.float64));manual=snapshot_state(state)
    history=None;memory=StepHistory(switch.metadata.history_length);last=None;previous_dt=None
    for i in range(40):
        h=.025;index=selector(manual,last);child=switch.methods[index]
        if last!=index:memory=StepHistory(memory.maxlen)
        if child.metadata.history_requires_constant_dt and previous_dt!=h:memory=StepHistory(memory.maxlen)
        expected=child(manual,h,rhs,history=memory,solver_opts=OPTIONS)
        result=switch(state,h,rhs,history=history,solver_opts=OPTIONS)
        assert result.step_status.successful and torch.equal(result.state.y,expected.state.y)
        assert result.history.context['index']==index
        assert result.estimate_order==(expected.estimate_order if expected.estimate_order is not None else child.metadata.estimate_order)
        if isinstance(child,NDF) and last!=index:assert result.quadrature.bootstrap
        state,history=result.state,result.history;manual=expected.state
        memory=expected.history if expected.history is not None else StepHistory(memory.maxlen)
        last,previous_dt=index,h
    assert abs(state.y.item()-math.exp(-1))<1e-4


def test_previous_selection_only_advances_with_accepted_steps_and_no_fsal_reuse():
    observations=[];calls=[]
    def selector(state,previous):observations.append((state.t,previous));return 0 if previous is None else 1-previous
    switch=AlgorithmSwitch(('dormandPrince','trbdf2'),selector,'alternate-v1');records=[]
    result=solve(switch,TensorState(torch.ones(1,dtype=torch.float64)),1.,
        tensor_rhs(lambda t,y:(calls.append(t) or -y)),dt=.8,adaptive=True,rtol=1e-6,atol=1e-9,
        solver_opts=OPTIONS,callback=records.append)
    assert result.successful and result.rejected_steps>0
    assert [r.result.history.context['index'] for r in records]==[i%2 for i in range(len(records))]
    for r in records:
        assert r.result.estimate_order==(5 if r.result.history.context['index']==0 else 3)
    assert observations[0][1] is None and observations[1][1] is None
    assert not switch.fsal and not switch.metadata.dense_output


def test_adaptive_ndf_child_order_policy_portable_restart_and_maximum_step(tmp_path):
    switch=AlgorithmSwitch(('dormandPrince',AdaptiveNDF()),lambda s,p:0 if s.t<.05 else 1,'ramp-v1')
    rhs=tensor_rhs(lambda t,y:-y);records=[]
    first=solve(switch,TensorState(torch.ones(1,dtype=torch.float64)),4.,rhs,dt=.001,
        dt_max=.05,adaptive=True,rtol=1e-7,atol=1e-9,solver_opts=OPTIONS,callback=records.append)
    assert first.successful and switch.order is None and max(r.dt for r in records)<=.05
    orders=[r.result.history.context['child_context']['last_order'] for r in records if r.result.history.context['index']==1]
    assert min(orders)==1 and max(orders)>=3
    path=tmp_path/'switch.pt';save_checkpoint(first.checkpoint,path)
    a=solve(switch,None,5.,rhs,restart=load_checkpoint(path),solver_opts=OPTIONS)
    b=solve(switch,None,5.,rhs,restart=first.checkpoint,solver_opts=OPTIONS)
    assert torch.equal(a.state.y,b.state.y) and abs(a.state.y.item()-math.exp(-5))<1e-7
    changed=AlgorithmSwitch(('dormandPrince',AdaptiveNDF()),lambda s,p:1,'ramp-v2')
    with pytest.raises(ValueError,match='scheme'):
        solve(changed,None,5.,rhs,restart=first.checkpoint)


def test_constant_history_invalidates_before_changed_step_and_failed_switch_keeps_accepted_memory():
    from test_run import FailedSolver
    switch=AlgorithmSwitch(('BDF3','trbdf2'),lambda s,p:0 if s.t<.4 else 1,'stiff-v1')
    rhs=tensor_rhs(lambda t,y:-y)
    run=solve(switch,TensorState(torch.ones(1,dtype=torch.float64)),.3,rhs,dt=.025,solver_opts=OPTIONS)
    changed=switch(run.state,.04,rhs,history=run.history,solver_opts=OPTIONS)
    cold=get_scheme('BDF3')(run.state,.04,rhs,solver_opts=OPTIONS)
    assert torch.equal(changed.state.y,cold.state.y)
    initial=snapshot_state(run.state);initial.t=.4
    before=snapshot_state(run.history)
    failed=switch(initial,.025,rhs,history=run.history,solver=FailedSolver())
    assert not failed.step_status.successful and failed.history.context==before.context
    assert len(failed.history)==len(before)


def test_invalid_or_mutating_selector_rolls_back_and_caller_is_owned():
    from test_run import Transaction
    for selector in (lambda s,p:True,lambda s,p:-1,lambda s,p:2,
                     lambda s,p:(setattr(s.state,'y',s.y+1) or 0)):
        switch=AlgorithmSwitch(('dormandPrince','trbdf2'),selector,'bad-v1')
        initial=TensorState(torch.ones(1,dtype=torch.float64));tx=Transaction()
        with pytest.raises(ValueError):
            solve(switch,initial,.1,tensor_rhs(lambda t,y:-y),dt=.01,transaction=tx,solver_opts=OPTIONS)
        assert tx.commits==0 and tx.rollbacks==1 and initial.y.item()==1 and initial.t==0


def test_explicit_restart_foreign_context_and_capacity_guards():
    seen=[]
    switch=AlgorithmSwitch((NDF(3),'RK4'),lambda s,p:(seen.append(p) or 0),'ndf-v1')
    rhs=tensor_rhs(lambda t,y:-y)
    run=solve(switch,TensorState(torch.ones(1,dtype=torch.float64)),.3,rhs,dt=.025,solver_opts=OPTIONS)
    result=switch(run.state,.025,rhs,history=run.history,restart_history=True,solver_opts=OPTIONS)
    assert seen[-1] is None and result.quadrature.bootstrap
    foreign=snapshot_state(run.history);foreign.context['method']='foreign'
    result=switch(run.state,.025,rhs,history=foreign,solver_opts=OPTIONS)
    assert seen[-1] is None and result.quadrature.bootstrap
    with pytest.raises(ValueError,match='capacity'):
        switch(run.state,.025,rhs,history=StepHistory(1))
    with pytest.raises(ValueError,match='estimate'):
        solve(switch,run.state,.5,rhs,dt=.025,adaptive=True)


def test_full_state_lifecycle_copied_fields_and_parameter_gradient():
    from dataclasses import dataclass
    from test_ndf import System,system,mechanical
    @dataclass
    class Consumer(System):
        extrapolation_compatible=True
        def initializeNewState(self,*args,**kwargs):return Consumer(self.state.initializeNewState(),self.t,self.initialized,self.finalized)
        def finalize(self,initial,*args,**kwargs):self.finalized=initial.finalized+1;return self
    initial=Consumer(system().state)
    switch=AlgorithmSwitch(('RK4',Extrapolation()),lambda s,p:0 if s.t<.2 else 1,'quantity-v1')
    result=solve(switch,initial,.4,mechanical,dt=.025)
    assert result.successful and result.state.initialized==result.state.finalized==16
    assert initial.initialized==initial.finalized==0
    assert torch.equal(result.state.state.density,result.state.state.x.square()+result.state.state.e)
    parameter=torch.tensor(2.,dtype=torch.float64,requires_grad=True)
    switch=AlgorithmSwitch(('dormandPrince','trbdf2'),lambda s,p:0 if s.t<.2 else 1,'gradient-v1')
    result=solve(switch,TensorState(torch.ones(1,dtype=torch.float64)),.4,
        tensor_rhs(lambda t,y:-parameter*y),dt=.005,solver_opts=OPTIONS)
    derivative=torch.autograd.grad(result.state.y.sum(),parameter)[0]
    assert derivative.item()==pytest.approx(-.4*math.exp(-.8),rel=2e-5)


@pytest.mark.parametrize('method',['BDF3',NDF(3),VariableAdams(4,'abm')])
def test_state_value_and_layout_changes_restart_numeric_history(method):
    switch=AlgorithmSwitch((method,'RK4'),lambda s,p:0,'state-guard-v1')
    rhs=tensor_rhs(lambda t,y:-y)
    run=solve(switch,TensorState(torch.ones(2,dtype=torch.float64)),.3,rhs,dt=.025,solver_opts=OPTIONS)
    before=snapshot_state(run.history)
    for change in ('values','shape','dtype'):
        current=snapshot_state(run.state)
        if change=='values':current.state.y=current.y+1
        if change=='shape':current.state.y=current.y.reshape(1,2)
        if change=='dtype':current.state.y=current.y.float()
        a=switch(current,.025,rhs,history=run.history,solver_opts=OPTIONS)
        b=switch.methods[0](current,.025,rhs,solver_opts=OPTIONS)
        assert torch.equal(a.state.y,b.state.y)
    assert torch.equal(run.history.context['current'],before.context['current'])


def test_backward_switching_nonautonomous_time_and_no_estimate_children():
    switch=AlgorithmSwitch(('RK4',Extrapolation()),lambda s,p:0 if s.t>-.2 else 1,'backward-v1')
    result=solve(switch,TensorState(torch.ones(1,dtype=torch.float64)),-.4,
        tensor_rhs(lambda t,y:-y+.2*y*y-.2*math.exp(-2*t)),dt=.025)
    assert result.successful and abs(result.state.y.item()-math.exp(.4))<1e-7
    assert switch.order is None and not switch.has_error_estimate


def test_nonfinite_trials_do_not_advance_policy_history_or_external_state():
    from test_run import Transaction
    observations=[];tx=Transaction()
    switch=AlgorithmSwitch(('dormandPrince','trbdf2'),lambda s,p:(observations.append(p) or 0),'nonfinite-v1')
    result=solve(switch,TensorState(torch.ones(1,dtype=torch.float64)),.1,
        tensor_rhs(lambda t,y:y*float('nan')),dt=.1,adaptive=True,max_attempts=3,
        transaction=tx,raise_on_failure=False)
    assert not result.successful and observations==[None]*3
    assert result.accepted_steps==tx.commits==0 and tx.rollbacks==3
    assert result.history.context is None and result.state.y.item()==1


def test_selected_child_policy_failure_rolls_back_before_commit():
    from test_run import Transaction
    class Invalid(NDF):
        def next_step_size(self,result,h,proposal):
            result.history.context['constant_steps']=999
            return float('nan')
    switch=AlgorithmSwitch((Invalid(2),'dormandPrince'),lambda s,p:0,'invalid-child-v1');tx=Transaction()
    initial=TensorState(torch.ones(1,dtype=torch.float64))
    with pytest.raises(ValueError,match='Child adaptive policy'):
        solve(switch,initial,.1,tensor_rhs(lambda t,y:-y),dt=.001,adaptive=True,
            transaction=tx,solver_opts=OPTIONS)
    assert tx.commits==0 and tx.rollbacks==1 and initial.t==0


def test_uid_edits_restart_numeric_history_and_gradients_survive_method_transitions():
    from test_ndf import system,mechanical
    switch=AlgorithmSwitch((NDF(3),'RK4'),lambda s,p:0,'uid-v1')
    run=solve(switch,system(),.2,mechanical,dt=.025,solver_opts=OPTIONS)
    edited=snapshot_state(run.state);edited.state.uid=edited.state.uid.flip(0)
    result=switch(edited,.025,mechanical,history=run.history,solver_opts=OPTIONS)
    assert result.quadrature.bootstrap
    parameter=torch.tensor(2.,dtype=torch.float64,requires_grad=True)
    switch=AlgorithmSwitch(('dormandPrince',NDF(3)),lambda s,p:0 if s.t<.1 else 1,'ndf-gradient-v1')
    result=solve(switch,TensorState(torch.ones(1,dtype=torch.float64)),.4,
        tensor_rhs(lambda t,y:-parameter*y),dt=.005,solver_opts=OPTIONS)
    derivative=torch.autograd.grad(result.state.y.sum(),parameter)[0]
    assert derivative.item()==pytest.approx(-.4*math.exp(-.8),rel=2e-5)


@pytest.mark.parametrize('options',[dict(methods=('RK4',)),dict(methods=('missing','RK4')),
    dict(selector=None),dict(policy_id='')])
def test_invalid_switch_configuration(options):
    with pytest.raises((ValueError,TypeError)):
        AlgorithmSwitch(**{**dict(methods=('RK4','trbdf2'),selector=lambda s,p:0,policy_id='test-v1'),**options})
