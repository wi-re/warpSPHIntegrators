"""Sampling, continuous event localization and resumable run contracts."""
import math
import pytest
import torch
from warpSPHIntegrators import (solve, TensorState, tensor_rhs, Event,
    save_checkpoint, load_checkpoint, snapshot_state, testing)


def state(value=1.,t=0.):
    return TensorState(torch.tensor([value],dtype=torch.float64),t=t)


def test_dense_sampling_does_not_change_accepted_grid():
    f=tensor_rhs(lambda t,y:-y)
    base=solve('Dormand-Prince 5(4)',state(),1.,f,dt=.2)
    sampled=solve('Dormand-Prince 5(4)',state(),1.,f,dt=.2,sample_times=[0.,.13,.3,.77,1.])
    assert torch.equal(base.state.y,sampled.state.y)
    assert base.attempts==sampled.attempts
    assert [s.t for s in sampled.samples]==[0.,.13,.3,.77,1.]
    for s in sampled.samples:
        assert s.state.t==s.t
        assert s.state.y.item()==pytest.approx(math.exp(-s.t),abs=3e-7)
    sampled.samples[-1].state.y.fill_(9)
    assert sampled.state.y.item()<1.


def test_fixed_grid_sampling_rounding_and_unsupported_requests():
    f=tensor_rhs(lambda t,y:-y)
    base=solve('RK4',state(),.7,f,dt=.1)
    sampled=solve('RK4',state(),.7,f,dt=.1,sample_times=[.3,.6,.7])
    assert torch.equal(base.state.y,sampled.state.y)
    assert [s.t for s in sampled.samples]==[0.,.3,.6,.7]
    with pytest.raises(ValueError,match='Off-grid'):
        solve('RK4',state(),1.,f,dt=.1,sample_times=[.15])
    with pytest.raises(ValueError,match='Event localization'):
        solve('RK4',state(),1.,f,dt=.1,events=[Event(lambda t,s:s.y.item()-.5)])


def test_terminal_event_recomputes_and_rolls_back_full_trial():
    from test_run import StatefulTensor, Transaction
    tx=Transaction(); original=StatefulTensor(torch.ones(1,dtype=torch.float64))
    def rhs(t,y):tx.effects.append(t);return -y
    r=solve('Dormand-Prince 5(4)',original,2.,tensor_rhs(rhs),dt=.2,
            events=[Event(lambda t,s:s.y.item()-.5,'half',terminal=True)],transaction=tx)
    assert r.termination=='event'
    assert r.state.t==pytest.approx(math.log(2),abs=2e-7)
    assert r.state.y.item()==pytest.approx(.5,abs=1e-10)
    assert tx.rollbacks==r.rejected_steps>0
    assert tx.commits==r.accepted_steps==len(r.state.counters)
    assert max(tx.effects)<=r.state.t+1e-10
    assert original.counters==[]


@pytest.mark.parametrize('backward',[False,True])
def test_event_direction_order_and_no_duplicate_endpoint(backward):
    f=tensor_rhs(lambda t,y:torch.ones_like(y))
    a,b=(1.,0.) if backward else (0.,1.)
    events=[Event(lambda t,s:s.y.item()-.5,'up',direction=1),
            Event(lambda t,s:s.y.item()-.5,'down',direction=-1),
            Event(lambda t,s:s.y.item()-.25,'quarter')]
    r=solve('Dormand-Prince 5(4)',state(a,a),b,f,dt=.25,events=events)
    assert [e.name for e in r.events]==(['up','quarter'] if backward else ['quarter','up'])
    assert r.state.t==b


def test_initial_terminal_event_and_invalid_event():
    f=tensor_rhs(lambda t,y:-y)
    r=solve('Dormand-Prince 5(4)',state(),1.,f,dt=.1,events=[Event(lambda t,s:t,terminal=True)])
    assert r.termination=='event' and r.attempts==0
    with pytest.raises(TypeError,match='Event objects'):
        solve('Dormand-Prince 5(4)',state(),1.,f,dt=.1,events=[lambda t,s:t])


@pytest.mark.parametrize('name,adaptive',[('RK4',False),('Dormand-Prince 5(4)',True),('BDF2',False)])
def test_restart_preserves_trajectory_history_and_controller(tmp_path,name,adaptive):
    f=tensor_rhs(lambda t,y:-y)
    opts=dict(dt=.1,adaptive=adaptive,rtol=1e-8,atol=1e-10,allow_stagnation=True)
    complete=solve(name,state(),1.,f,**opts)
    seen=[]
    def stop(record):
        seen.append(record.result.state.t)
        return len(seen)<4
    part=solve(name,state(),1.,f,callback=stop,**opts)
    cp=part.checkpoint
    resumed=solve(name,None,1.,f,restart=cp)
    assert torch.equal(complete.state.y,resumed.state.y)
    assert part.accepted_steps+resumed.accepted_steps==complete.accepted_steps
    path=tmp_path/'run.pt';save_checkpoint(cp,path)
    restored=load_checkpoint(path,map_location='cpu')
    disk=solve(name,None,1.,f,restart=restored)
    assert torch.equal(complete.state.y,disk.state.y)
    assert len(cp.history)==len(restored.history)
    cp.state.y.fill_(0)
    assert part.state.y.item()>0


def test_checkpoint_preserves_gradients_in_memory_and_detaches_on_disk(tmp_path):
    y=torch.ones(1,dtype=torch.float64,requires_grad=True)
    f=tensor_rhs(lambda t,y:y)
    part=solve('RK4',TensorState(y),1.,f,dt=.1,callback=lambda r:False)
    resumed=solve('RK4',None,1.,f,restart=part.checkpoint)
    assert torch.autograd.grad(resumed.state.y.sum(),y)[0].item()==pytest.approx(math.e,rel=1e-6)
    path=tmp_path/'run.pt';save_checkpoint(part.checkpoint,path)
    assert not load_checkpoint(path).state.y.requires_grad
    assert not snapshot_state(part.checkpoint,detach=True).state.y.requires_grad
    with pytest.raises(ValueError,match='scheme'):
        solve('Midpoint',None,1.,f,restart=part.checkpoint)
    with pytest.raises(ValueError,match='state=None'):
        solve('RK4',state(),1.,f,restart=part.checkpoint)


def test_portable_checkpoint_requires_tensor_contract(tmp_path):
    p=testing.oscillator_problem()
    r=solve('RK4',p.initial(),.1,p.rhs,dt=.1)
    with pytest.raises(TypeError,match='plain TensorState'):
        save_checkpoint(r.checkpoint,tmp_path/'unsupported.pt')
