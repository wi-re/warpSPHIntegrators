"""Corrected startup/source warm maps, owned mechanics history and work gates."""
import json
import math
from dataclasses import dataclass
from pathlib import Path

import pytest
import torch

from warpSPHIntegrators.schemes import ImprovedNystrom, Nystrom
from warpSPHIntegrators.schemes.families.improved_nystrom import ImprovedNystromWeights
from warpSPHIntegrators import solve
from warpSPHIntegrators.wrappers import AlgorithmSwitch
from warpSPHIntegrators.state import StepHistory
from test_nystrom import initial, manufactured, Fields, System, system

SOURCE=json.loads((Path(__file__).parents[1]/'scripts/coefficients/wave12/irkn_audit.json').read_text())
CORRECTED=Path(__file__).parents[1]/'scripts/coefficients/wave12/irkn_corrected.json'


def force(problem='autonomous',parameter=1.,calls=None):
    import warpSPHIntegrators.testing as testing
    def rhs(state,dt,**kwargs):
        s=state.state
        if calls is not None:calls.append((state.t,s.x.clone()))
        a=-parameter*s.x if problem=='autonomous' else torch.full_like(s.x,state.t) if problem=='time_only' else -s.x+.1*state.t
        return testing.ParticleUpdate(s.u,a,torch.zeros_like(s.e)),None
    return rhs


@pytest.mark.parametrize('order',[3,4])
@pytest.mark.parametrize('problem',['autonomous','time_only','manufactured'])
def test_corrected_seed_then_actual_source_warm_kernel_trajectories(order,problem):
    for expected in SOURCE['records'][f'IRKN{order}_{problem}_reseeded_control']['runs']:
        n=expected['steps'];result=solve(ImprovedNystrom(order,'declared'),initial(v=[.7]),1.,force(problem),dt=1/n)
        assert result.successful
        assert result.state.state.x.item()==pytest.approx(expected['q'],rel=0,abs=2e-13)
        assert result.state.state.u.item()==pytest.approx(expected['v'],rel=0,abs=2e-13)


@pytest.mark.parametrize('order',[3,4])
def test_nonlinear_time_dependent_full_trajectory_order(order):
    errors=[]
    for h in [.1,.05,.025,.0125]:
        result=solve(ImprovedNystrom(order,'declared'),initial(),1.,manufactured('DPRKN6'),dt=h)
        errors.append(max(abs(result.state.state.x.item()-math.e),abs(result.state.state.u.item()-math.e)))
    measured=math.log(errors[-2]/errors[-1],2)
    assert order-.2<measured<order+.3,(errors,measured)


@pytest.mark.parametrize('order',[3,4])
def test_nonlinear_corrected_source_trajectory_at_every_accepted_step(order):
    import warpSPHIntegrators.testing as testing
    source=json.loads(CORRECTED.read_text())
    def rhs(state,dt,**kwargs):
        s=state.state;target=math.exp(.7*state.t)
        a=-4*s.x-.2*s.x**3+4.49*target+.2*target**3
        return testing.ParticleUpdate(s.u,a,torch.zeros_like(s.e)),None
    for expected in source['records'][f'IRKN{order}']:
        rows=[]
        result=solve(ImprovedNystrom(order,'declared'),initial(v=[.7]),1.,rhs,dt=expected['h'],
            callback=lambda record:rows.append(record.result.state))
        assert result.successful and len(rows)==len(expected['trajectory'])
        for actual,reference in zip(rows,expected['trajectory']):
            assert actual.t==reference['t']
            assert actual.state.x.tolist()==pytest.approx(reference['q'],rel=0,abs=2e-13)
            assert actual.state.u.tolist()==pytest.approx(reference['v'],rel=0,abs=2e-13)


@pytest.mark.parametrize('order',[3,4])
def test_cached_trajectory_gradients_match_finite_difference(order):
    def forward(q,k,tracking=False):
        start=initial(q=[q],v=[.2]);start.state.x.requires_grad_(tracking)
        parameter=torch.tensor(k,dtype=torch.float64,requires_grad=tracking)
        result=solve(ImprovedNystrom(order),start,.4,force(parameter=parameter),dt=.05)
        return result.state.state.x.sum(),start.state.x,parameter
    value,q,k=forward(.7,2.,True);dq,dk=torch.autograd.grad(value,(q,k));eps=1e-5
    for gradient,plus,minus in [(dq,forward(.7+eps,2.)[0],forward(.7-eps,2.)[0]),
                                (dk,forward(.7,2.+eps)[0],forward(.7,2.-eps)[0])]:
        assert gradient.item()==pytest.approx(((plus-minus)/(2*eps)).item(),rel=2e-7,abs=1e-9)


@pytest.mark.parametrize('order',[3,4])
@pytest.mark.parametrize('change',['dt','value','time','uid','layout','dtype','parameter','foreign'])
def test_invalid_history_restarts_before_consumption(order,change):
    from warpSPHIntegrators.runtime import snapshot_state
    method=ImprovedNystrom(order,'declared');rhs=force()
    state=system();state=__import__('warpSPHIntegrators').testing.ParticleSystem(state.state)
    result=method(state,.05,rhs);next_state=snapshot_state(result.state);h=.05;history=result.history;kwargs={}
    if change=='dt':h=.025
    elif change=='value':next_state.state.x+=.01
    elif change=='time':next_state.t+=.01
    elif change=='uid':next_state.state.uid=next_state.state.uid.flip(0)
    elif change=='layout':
        next_state.state.x=next_state.state.x[:,None];next_state.state.u=next_state.state.u[:,None]
    elif change=='dtype':
        for key in ['x','u','e']:setattr(next_state.state,key,getattr(next_state.state,key).float())
    elif change=='parameter':rhs=force(parameter=2.);kwargs['restart_history']=True
    elif change=='foreign':history=StepHistory(1,history.entries,context={'method':'other'})
    actual=method(next_state,h,rhs,history=history,**kwargs);cold=method(next_state,h,rhs)
    assert torch.equal(actual.state.state.x,cold.state.state.x)
    assert torch.equal(actual.state.state.u,cold.state.state.u)
    assert result.history.context['t']==.05


@pytest.mark.parametrize('order',[3,4])
def test_fixed_work_and_checkpoint_continuation(order):
    method=ImprovedNystrom(order,'declared');calls=[];rhs=force(calls=calls)
    first=solve(method,initial(),.5,rhs,dt=.0625)
    # Cold IRKN3/4 costs 5/6 full RHS calls; warm steps cost 2/3.
    assert len(calls)==order+2+7*(order-1)
    resumed=solve(method,None,1.,rhs,restart=first.checkpoint)
    direct=solve(method,initial(),1.,force(),dt=.0625)
    assert torch.equal(resumed.state.state.x,direct.state.state.x)
    assert torch.equal(resumed.state.state.u,direct.state.state.u)
    with pytest.raises(ValueError,match='estimate|adaptive|Adaptive'):
        solve(method,initial(),1.,rhs,dt=.1,adaptive=True)


@dataclass
class IRKNSystem(System):
    irkn_compatible=True
    def initializeNewState(self,*args,**kwargs):
        return IRKNSystem(self.state.initializeNewState(),self.t,self.initialized,self.finalized)
    def finalize(self,initial,dt,auxes,updates,weights,**kwargs):
        assert isinstance(weights,ImprovedNystromWeights)
        assert len(updates)==len(weights.position)==len(weights.velocity)==len(weights.quantity)
        q=initial.state.x+dt*weights.initial_velocity*initial.state.u
        if weights.previous_velocity is not None:q=q+dt*weights.previous_velocity_weight*weights.previous_velocity
        q=q+dt*dt*sum(w*k.dudt for w,k in zip(weights.position,updates))
        v=initial.state.u+dt*sum(w*k.dudt for w,k in zip(weights.velocity,updates))
        assert torch.allclose(self.state.x,q,rtol=0,atol=1e-13)
        assert torch.allclose(self.state.u,v,rtol=0,atol=1e-13)
        self.finalized=initial.finalized+1
        return self


@pytest.mark.parametrize('order',[3,4])
def test_full_lifecycle_passive_quantity_endpoint_copy_and_failure_rollback(order):
    start=IRKNSystem(system().state);method=ImprovedNystrom(order);rhs=force()
    result=solve(method,start,.2,rhs,dt=.05)
    assert result.successful and result.state.initialized==result.state.finalized==4
    assert start.initialized==start.finalized==0
    assert torch.equal(result.state.state.density,result.state.state.x.square()+result.state.state.e)
    assert torch.equal(result.state.state.e,start.state.e)
    warm=method(start,.05,rhs)
    def broken(state,dt,**kwargs):
        k,aux=rhs(state,dt)
        k.dudt=k.dudt*float('nan')
        return k,aux
    failed=method(warm.state,.05,broken,history=warm.history,restart_history=True)
    assert not failed.step_status.successful and failed.state.finalized==1
    assert warm.history.context['t']==.05
    run=solve(method,warm.state,.1,broken,dt=.05,history=warm.history,raise_on_failure=False)
    assert not run.successful and run.accepted_steps==0


@pytest.mark.parametrize('order',[3,4])
def test_switch_owns_child_history_and_cold_starts_transitions(order):
    switch=AlgorithmSwitch((ImprovedNystrom(order,'declared'),Nystrom('Nystrom4VelocityIndependent','declared')),
        lambda s,previous:0 if s.t<.15 else 1,'irkn-test')
    result=solve(switch,initial(),.3,force(),dt=.05)
    assert result.successful and result.accepted_steps==6


def test_structure_guards_and_custom_finalize_opt_in():
    import warpSPHIntegrators.testing as testing
    def rhs(state,dt,**kwargs):
        s=state.state;return testing.ParticleUpdate(s.u,-s.x-.1*s.u.detach(),torch.zeros_like(s.e)),None
    with pytest.raises(ValueError,match='independent acceleration'):ImprovedNystrom()(initial(),.1,rhs)
    with pytest.raises(TypeError,match='irkn_compatible'):ImprovedNystrom()(system(),.1,force())


@pytest.mark.parametrize('order',[3,4])
def test_result_stage_and_quadrature_mutations_do_not_alias_cached_memory(order):
    method=ImprovedNystrom(order,'declared');rhs=force();result=method(initial(),.0625,rhs)
    baseline=method(result.state,.0625,rhs,history=result.history)
    result.stages[-1].update.dudt.add_(99.)
    assert torch.equal(method(result.state,.0625,rhs,history=result.history).state.state.x,baseline.state.state.x)
    warm=method(baseline.state,.0625,rhs,history=baseline.history)
    old=baseline.history.context['velocity'].clone()
    warm.quadrature.previous_velocity.add_(99.)
    assert torch.equal(baseline.history.context['velocity'],old)


@pytest.mark.parametrize('kwargs',[{'order':2},{'order':True},{'structure':'bad'},{'structure_atol':-1}])
def test_invalid_configuration(kwargs):
    with pytest.raises(ValueError):ImprovedNystrom(**kwargs)
