"""Source neighbor defects, accepted order control and owned Adams continuation."""
import json
import math
from pathlib import Path
import numpy as np
import pytest
import torch
from warpSPHIntegrators.schemes import AdaptiveAdams, VariableAdams
from warpSPHIntegrators import TensorState, tensor_rhs, solve
from warpSPHIntegrators.runtime import save_checkpoint, load_checkpoint, snapshot_state
from warpSPHIntegrators.state import StepHistory
from warpSPHIntegrators.schemes.families.adaptive_adams import adams_order_choice
from warpSPHIntegrators.schemes.coefficients.adams_control import _defect_weights, _upper_weights
SOURCE=json.loads((Path(__file__).parents[1]/'scripts/coefficients/wave10/adaptive_adams.json').read_text())

@pytest.mark.parametrize('record',SOURCE['choices'])
def test_mature_choice_matches_source_branch_with_declared_lower_bound(record):
 assert adams_order_choice(record['order'],record['error'],record['lower_errors'],record['upper_error'])==max(3,record['source_next_order'])

@pytest.mark.parametrize('record',SOURCE['defects'])
def test_neighbor_weights_match_actual_source_helpers(record):
 p=record['order'];n=p+3;nodes=(np.array(record['times'][::-1])-record['times'][-1])/record['h']
 for key,q in [('main',p),('lower1',p-1),('lower2',p-2)]:
  weights=_defect_weights(nodes,q);actual=np.zeros(n)
  actual[:p+1]=weights[1:][::-1];actual[p+1]=weights[0]
  assert actual==pytest.approx(record[key],abs=1e-14,rel=0)
 w=_upper_weights(nodes,p);actual=np.zeros(n);actual[:p+1]=w[1:][::-1];actual[p+2]=w[0]
 assert actual==pytest.approx(record['upper'],abs=1e-14,rel=0)

@pytest.mark.parametrize('maximum',[3,4,5])
def test_orders_mature_one_at_a_time_and_estimator_follows_actual_step(maximum):
 records=[];r=solve(AdaptiveAdams(maximum),TensorState(torch.ones(2,dtype=torch.float64)),4.,
  tensor_rhs(lambda t,y:-y),dt=.01,adaptive=True,rtol=1e-8,atol=1e-10,callback=records.append)
 assert r.successful and r.rejected_steps>0
 actual=[x.result.history.context['last_order'] for x in records]
 assert min(actual)==3 and max(actual)==maximum
 assert all(3<=x.result.history.context['order']<=maximum and x.error_norm<=1 for x in records)
 for record in records:
  c=record.result.history.context
  assert abs(c['order']-c['last_order'])<=1
  assert record.result.estimate_order in (2,3,c['last_order']+1)
  if c['order']!=c['last_order']:assert c['age']==0
 assert abs(r.state.y[0].item()-math.exp(-4))<2e-7

@pytest.mark.parametrize('tol',[1e-4,1e-6,1e-8])
def test_nonlinear_time_forcing_has_known_accuracy(tol):
 r=solve(AdaptiveAdams(),TensorState(torch.ones(1,dtype=torch.float64)),4.,
  tensor_rhs(lambda t,y:-y+.2*y*y-.2*math.exp(-2*t)),dt=.1,adaptive=True,rtol=tol,atol=tol*.01)
 assert r.successful and abs(r.state.y.item()-math.exp(-4))<50*tol


def test_fixed_calls_match_existing_pece_without_extra_rhs_or_lifecycle():
 adaptive=AdaptiveAdams();fixed=VariableAdams(3);a=b=TensorState(torch.ones(1,dtype=torch.float64));ha=hb=None
 for _ in range(10):
  calls=[0,0]
  def first(t,y):calls[0]+=1;return -y
  def second(t,y):calls[1]+=1;return -y
  ra=adaptive(a,.02,tensor_rhs(first),history=ha);rb=fixed(b,.02,tensor_rhs(second),history=hb)
  assert torch.equal(ra.state.y,rb.state.y) and calls[0]==calls[1]
  assert ra.history.context['order']==3
  a,b,ha,hb=ra.state,rb.state,ra.history,rb.history
 assert calls==[3,3]


def test_portable_restart_and_input_history_are_owned(tmp_path):
 method=AdaptiveAdams();rhs=tensor_rhs(lambda t,y:-y)
 first=solve(method,TensorState(torch.ones(1,dtype=torch.float64)),4.,rhs,dt=.01,
  adaptive=True,rtol=1e-8,atol=1e-10,callback=lambda r:r.result.history.context['order']<5)
 assert first.termination=='callback' and first.history.context['order']==5
 before=snapshot_state(first.history)
 path=tmp_path/'adams.pt';save_checkpoint(first.checkpoint,path)
 a=solve(method,None,4.,rhs,restart=first.checkpoint)
 b=solve(method,None,4.,rhs,restart=load_checkpoint(path))
 assert torch.equal(a.state.y,b.state.y) and a.history.context['order']==b.history.context['order']
 assert torch.equal(first.history.context['current'],before.context['current'])
 assert len(first.history)==len(before)


def test_controller_uses_field_norm_and_maximum_dt():
 calls=[]
 class Observe(AdaptiveAdams):
  def adapt_step(self,result,h,proposal,**kwargs):
   assert kwargs['norm_estimate'](result.error)==pytest.approx(kwargs['error_norm'])
   calls.append(kwargs['error_norm'])
   return super().adapt_step(result,h,proposal,**kwargs)
 rows=[];r=solve(Observe(),TensorState(torch.ones(2,dtype=torch.float64)),1.,tensor_rhs(lambda t,y:-y),
  dt=.001,adaptive=True,rtol={'y':1e-7},atol={'y':1e-9},dt_max=.02,callback=rows.append)
 assert len(calls)==r.accepted_steps and max(x.dt for x in rows)<=.02


def test_policy_failure_rolls_back_before_external_commit():
 from test_run import Transaction
 class Broken(AdaptiveAdams):
  def adapt_step(self,result,*args,**kwargs):
   result.history.context['order']=5
   raise RuntimeError('broken policy')
 tx=Transaction();state=TensorState(torch.ones(1,dtype=torch.float64))
 with pytest.raises(RuntimeError,match='broken policy'):
  solve(Broken(),state,1.,tensor_rhs(lambda t,y:-y),dt=.01,adaptive=True,transaction=tx)
 assert tx.commits==0 and tx.rollbacks==1 and state.y.item()==1

@pytest.mark.parametrize('change',['values','time','dtype','shape','explicit','ratio'])
def test_changed_trajectory_restarts_before_using_history(change):
 method=AdaptiveAdams();rhs=tensor_rhs(lambda t,y:-y)
 run=solve(method,TensorState(torch.ones(1,dtype=torch.float64)),2.,rhs,dt=.01,
  adaptive=True,rtol=1e-8,atol=1e-10)
 state=snapshot_state(run.state);history=snapshot_state(run.history);kwargs={};h=run.last_dt
 if change=='values':state.y.add_(.1)
 elif change=='time':state.t+=.1
 elif change=='dtype':state=TensorState(state.y.float(),state.t)
 elif change=='shape':state=TensorState(state.y.repeat(2),state.t)
 elif change=='explicit':kwargs['restart_history']=True
 elif change=='ratio':h*=10
 result=method(state,h,rhs,history=history,**kwargs)
 assert result.history.context['last_order']==3 and result.history.context['age']==0
 assert result.estimate_order==2 and len(result.history)==1
 assert len(history)==len(run.history)


def test_full_state_quantities_gradients_and_owned_hooks():
 from dataclasses import dataclass
 from test_variable_adams import CacheFields,CacheSystem
 import warpSPHIntegrators.testing as testing
 @dataclass
 class Consumer(CacheSystem):
  initialized:int=0
  finalized:int=0
  def initializeNewState(self,*args,**kwargs):return Consumer(self.state.initializeNewState(),self.t,self.initialized,self.finalized)
  def initialize(self,dt,**kwargs):self.initialized+=1;return self
  def finalize(self,initial,*args,**kwargs):
   self.finalized=initial.finalized+1
   return super().finalize(initial,*args,**kwargs)
 def mechanical(state,dt,**kwargs):
  s=state.state
  return testing.ParticleUpdate(s.u,-s.x-.4*s.u,.4*s.u.square()),None
 initial=Consumer(CacheFields(torch.ones(2,dtype=torch.float64),torch.zeros(2,dtype=torch.float64),
  torch.zeros(2,dtype=torch.float64),torch.tensor([1,2]),torch.zeros(2,dtype=torch.float64)))
 before=snapshot_state(initial);rows=[]
 r=solve(AdaptiveAdams(),initial,1.,mechanical,dt=.01,adaptive=True,
  rtol={'x':1e-7,'u':1e-7,'e':1e-8},atol=1e-10,callback=rows.append)
 assert r.successful
 s=r.state.state
 assert float(abs((s.x*s.x+s.u*s.u)/2+s.e-.5).max())<2e-6
 assert r.state.initialized==r.state.finalized==r.accepted_steps
 assert initial.initialized==initial.finalized==0
 assert torch.equal(s.density,s.x.square())
 assert torch.equal(initial.state.x,before.state.x)
 assert all(x.error_norm<=1 for x in rows)
 y=torch.tensor([1.],dtype=torch.float64,requires_grad=True);rate=torch.tensor(.7,dtype=torch.float64,requires_grad=True)
 out=solve(AdaptiveAdams(),TensorState(y),1.,tensor_rhs(lambda t,y:-rate*y),dt=.01)
 gy,gr=torch.autograd.grad(out.state.y.sum(),(y,rate))
 assert gy.item()==pytest.approx(math.exp(-.7),abs=1e-6,rel=0)
 assert gr.item()==pytest.approx(-math.exp(-.7),abs=2e-6,rel=0)


def test_backward_nonautonomous_and_zero_error_no_spurious_order_increase():
 r=solve(AdaptiveAdams(),TensorState(torch.ones(1,dtype=torch.float64)),-1.,tensor_rhs(lambda t,y:-y),
  dt=.01,adaptive=True,rtol=1e-8,atol=1e-10)
 assert abs(r.state.y.item()-math.e)<2e-6
 rows=[];zero=solve(AdaptiveAdams(),TensorState(torch.tensor([1e20],dtype=torch.float64)),1.,
  tensor_rhs(lambda t,y:y*0),dt=.01,adaptive=True,callback=rows.append)
 assert zero.state.y.item()==1e20 and all(x.result.history.context['order']==3 for x in rows)


def test_lower_order_decision_and_order_hold_policy():
 method=AdaptiveAdams();rhs=tensor_rhs(lambda t,y:-y)
 r=solve(method,TensorState(torch.ones(1,dtype=torch.float64)),4.,rhs,dt=.01,adaptive=True,
  rtol=1e-8,atol=1e-10)
 result=method(r.state,r.last_dt,rhs,history=r.history);c=result.history.context;p=c['order']
 assert p>3
 c['age']=100
 next_h=method.adapt_step(result,abs(r.last_dt),2*abs(r.last_dt),error_norm=.9,norm_estimate=lambda e:1e-6)
 assert c['order']==p-1 and c['age']==0 and next_h==abs(r.last_dt)


def test_algorithm_switch_delegates_accepted_order_policy():
 from warpSPHIntegrators.wrappers import AlgorithmSwitch
 method=AlgorithmSwitch((AdaptiveAdams(),'Dormand-Prince 5(4)'),lambda s,p:0,'adams')
 result=solve(method,TensorState(torch.ones(1,dtype=torch.float64)),4.,tensor_rhs(lambda t,y:-y),
  dt=.01,adaptive=True,rtol=1e-8,atol=1e-10)
 assert result.successful and result.history.context['child_context']['order']==5

@pytest.mark.parametrize('kwargs',[{'max_order':2},{'max_order':6},{'max_order':True},{'max_step_ratio':.5}])
def test_invalid_factory_options(kwargs):
 with pytest.raises(ValueError):AdaptiveAdams(**kwargs)


def test_history_capacity_type_foreign_and_prior_step_guards():
 method=AdaptiveAdams();state=TensorState(torch.ones(1,dtype=torch.float64));rhs=tensor_rhs(lambda t,y:-y)
 with pytest.raises(ValueError,match='capacity'):method(state,.1,rhs,history=StepHistory(1))
 with pytest.raises(TypeError):method(state,.1,rhs,history={})
 from warpSPHIntegrators.state import StageResult
 with pytest.warns(RuntimeWarning):method(state,.1,rhs,priorStep=StageResult())
 foreign=VariableAdams(3)(state,.1,rhs,history=StepHistory(5))
 result=method(foreign.state,.1,rhs,history=foreign.history)
 assert result.estimate_order==2 and len(result.history)==1

@pytest.mark.parametrize('error',[True,-1.,float('inf'),float('nan')])
def test_invalid_normalized_errors(error):
 with pytest.raises(ValueError):adams_order_choice(3,error)
