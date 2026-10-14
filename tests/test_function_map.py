"""Direct maps are next values at t+h, with owned full-state lifecycle."""
from dataclasses import dataclass
import json
import math
from pathlib import Path
import pytest
import torch
from warpSPHIntegrators import (FunctionMap,StateMap,MapTransition,tensor_map,TensorState,tensor_rhs,
    solve,save_checkpoint,load_checkpoint)

SOURCE=json.loads((Path(__file__).parents[1]/'scripts/coefficients/wave13/function_map.json').read_text())


@pytest.mark.parametrize('key',SOURCE['records'])
def test_both_source_map_branches_with_an_explicit_scaled_wrapper(key):
    r=SOURCE['records'][key];calls=[]
    def mapping(t,y,h):
        calls.append((t,y.detach().clone(),h))
        value=torch.stack((.9*y[0]+.1*y[1]+.2*math.sin(t),-.2*y[0]+.8*y[1]+.1*math.cos(t)))
        if r['nonlinear']:value=value+torch.stack((.05*y[0]*y[1],.1*y[0]**2))
        return y+h*value if r['scaled'] else value
    initial=TensorState(torch.tensor(r['initial'],dtype=torch.float64),t=r['t'])
    result=FunctionMap()(initial,r['h'],tensor_map(mapping))
    assert result.state.y.tolist()==pytest.approx(r['state'],rel=0,abs=2e-15)
    assert result.state.t==r['t']+r['h'] and len(calls)==r['nf']==1
    assert calls[0][0]==r['calls'][0]['t'] and calls[0][1].tolist()==r['initial']
    assert result.error is None and result.stages[0].update is None
    assert isinstance(result.transition,MapTransition) and result.quadrature is None
    assert result.transition.input_state.t==r['t'] and result.transition.evaluation_time==result.state.t
    assert initial.y.tolist()==r['initial'] and initial.t==r['t']


@pytest.mark.parametrize('shape',[(),(3,),(2,3)])
def test_known_affine_recurrence_parameters_gradients_and_portable_restart(shape,tmp_path):
    value=torch.ones(shape,dtype=torch.float64,requires_grad=True)
    coefficient=torch.tensor(.9,dtype=torch.float64,requires_grad=True)
    mapping=tensor_map(lambda t,y,h,a:y*a,a=coefficient)
    result=solve(FunctionMap(),TensorState(value),2.,mapping,dt=.1)
    assert result.successful and result.accepted_steps==20
    expected=.9**20
    assert torch.allclose(result.state.y,torch.full_like(value,expected),atol=2e-15,rtol=0)
    dy,da=torch.autograd.grad(result.state.y.sum(),(value,coefficient))
    assert torch.allclose(dy,torch.full_like(value,expected),atol=2e-15,rtol=0)
    assert da.item()==pytest.approx(value.numel()*20*.9**19,rel=2e-14)
    path=tmp_path/'map.pt';save_checkpoint(result.checkpoint,path)
    a=solve(FunctionMap(),None,3.,mapping,restart=load_checkpoint(path))
    b=solve(FunctionMap(),None,3.,mapping,restart=result.checkpoint)
    assert torch.equal(a.state.y,b.state.y)


def test_backward_exact_flow_map_and_no_continuous_order_or_adaptive_claim():
    method=FunctionMap();records=[]
    result=solve(method,TensorState(torch.ones(1,dtype=torch.float64)),-.4,
        tensor_map(lambda t,y,h:y*math.exp(-h)),dt=.025,callback=records.append)
    assert result.successful and result.state.y.item()==pytest.approx(math.exp(.4),rel=0,abs=2e-15)
    assert method.order is None and not method.fsal and not method.has_error_estimate
    assert method.metadata.stage_count==1 and len(records)==16
    with pytest.raises(ValueError,match='estimate'):
        solve(method,result.state,1.,tensor_map(lambda t,y,h:y),dt=.1,adaptive=True)
    with pytest.raises(TypeError,match='StateMap'):
        method(result.state,.1,tensor_rhs(lambda t,y:-y))


def consumer():
    from test_ndf import System,system
    @dataclass
    class Consumer(System):
        map_compatible=True
        def initializeNewState(self,*args,**kwargs):return Consumer(self.state.initializeNewState(),self.t,self.initialized,self.finalized)
        def finalize(self,initial,dt,auxes,updates,transition,**kwargs):
            assert updates==[None] and isinstance(transition,MapTransition)
            assert transition.evaluation_time==self.t
            self.finalized=initial.finalized+1;return self
    return Consumer(system().state)


def rotation_heat(s,h,**kwargs):
    q,v=s.state.x,s.state.u;decay=math.exp(-.2*h);c=math.cos(h);sn=math.sin(h)
    qnew=decay*(c*q+sn*v);vnew=decay*(c*v-sn*q)
    s.state.e=s.state.e+(q.square()+v.square()-qnew.square()-vnew.square())/2
    s.state.x,s.state.u=qnew,vnew
    return s,{'mapped':True}


def test_coupled_map_known_rotation_heat_budget_and_endpoint_lifecycle():
    initial=consumer();result=solve(FunctionMap(),initial,.4,StateMap(rotation_heat),dt=.025)
    assert result.successful and result.state.initialized==result.state.finalized==16
    assert initial.initialized==initial.finalized==0
    s=result.state.state
    assert s.x.tolist()==pytest.approx([math.exp(-.08)*math.cos(.4)]*2,rel=0,abs=2e-15)
    assert s.u.tolist()==pytest.approx([-math.exp(-.08)*math.sin(.4)]*2,rel=0,abs=2e-15)
    assert s.e.tolist()==pytest.approx([.5*(1-math.exp(-.16))]*2,rel=0,abs=2e-15)
    assert torch.allclose((s.x.square()+s.u.square())/2+s.e,torch.full_like(s.e,.5),atol=2e-15,rtol=0)
    assert torch.equal(s.density,s.x.square()+s.e)
    assert torch.equal(initial.state.e,torch.zeros(2,dtype=torch.float64))


def test_map_is_called_once_nonfinite_transition_never_finalizes_or_commits():
    from test_run import Transaction
    calls=[];initial=consumer();tx=Transaction()
    def bad(s,h,**kwargs):calls.append(s.t);s.state.e=s.state.e*float('nan');return s
    result=solve(FunctionMap(),initial,.1,StateMap(bad),dt=.1,transaction=tx,raise_on_failure=False)
    assert not result.successful and calls==[.1] and result.accepted_steps==0
    assert result.state.finalized==0 and initial.initialized==0
    assert tx.commits==0 and tx.rollbacks==1


@pytest.mark.parametrize('mode',['shape','dtype','time','uid','return'])
def test_invalid_map_returns_preserve_caller_and_roll_back(mode):
    from test_run import Transaction
    initial=consumer();tx=Transaction()
    def bad(s,h,**kwargs):
        if mode=='shape':s.state.e=s.state.e[:1]
        if mode=='dtype':s.state.e=s.state.e.float()
        if mode=='time':s.t+=1
        if mode=='uid':s.state.uid=s.state.uid.flip(0)
        if mode=='return':return 1.
        return s
    with pytest.raises((ValueError,TypeError)):
        solve(FunctionMap(),initial,.1,StateMap(bad),dt=.1,transaction=tx)
    assert initial.initialized==0 and initial.t==0 and tx.rollbacks==1 and tx.commits==0


def test_lifecycle_cannot_change_map_numerics_and_raw_hooks_require_opt_in():
    from test_ndf import system
    with pytest.raises(TypeError,match='map_compatible'):
        FunctionMap()(system(),.1,StateMap(rotation_heat))
    original=consumer();kind=type(original)
    @dataclass
    class Projecting(kind):
        def initializeNewState(self,*args,**kwargs):return Projecting(self.state.initializeNewState(),self.t,self.initialized,self.finalized)
        def finalize(self,*args,**kwargs):self.state.e=self.state.e+1;return self
    with pytest.raises(ValueError,match='finalize must preserve'):
        FunctionMap()(Projecting(original.state),.1,StateMap(rotation_heat))
    @dataclass
    class Changing(kind):
        def initializeNewState(self,*args,**kwargs):return Changing(self.state.initializeNewState(),self.t,self.initialized,self.finalized)
        def preprocess(self,*args,**kwargs):self.state.x=self.state.x+1;return self
    with pytest.raises(ValueError,match='input preprocess'):
        FunctionMap()(Changing(original.state),.1,StateMap(rotation_heat))


@pytest.mark.parametrize('mapping',[lambda t,y,h:0,lambda t,y,h:y[:1],lambda t,y,h:y.float()])
def test_tensor_map_validates_result_contract(mapping):
    with pytest.raises((ValueError,TypeError)):
        FunctionMap()(TensorState(torch.ones(2,dtype=torch.float64)),.1,tensor_map(mapping))
