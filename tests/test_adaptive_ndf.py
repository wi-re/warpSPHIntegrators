"""Accepted order changes, normalized candidates, restart and rollback."""
import json
import math
from pathlib import Path
import pytest
import torch
from warpSPHIntegrators.schemes import AdaptiveNDF, NDF
from warpSPHIntegrators import TensorState, tensor_rhs, solve
from warpSPHIntegrators.runtime import save_checkpoint, load_checkpoint, snapshot_state
from warpSPHIntegrators.schemes.families.adaptive_ndf import ndf_order_proposals
SOURCE=json.loads((Path(__file__).parents[1]/'scripts/coefficients/wave10/ndf_controller.json').read_text())
OPTIONS=dict(matvec='jvp',tol=1e-15,newton_tol=1e-10)

@pytest.mark.parametrize('record',SOURCE['records'])
def test_candidate_ratios_match_independent_source_accept_controller(record):
 p=record['order']
 candidates=ndf_order_proposals(p,record['error'],record['lower_error'],record['upper_error'])
 selected,ratio=max(candidates,key=lambda c:c[1])
 # Apply the source steady deadband to isolate proposal math from our policy.
 q=1/ratio
 if .9<=q<=1.2:ratio=1.
 assert selected==record['next_order']
 assert ratio==pytest.approx(record['next_h_ratio'],rel=3e-15)

@pytest.mark.parametrize('maximum',[1,2,3,4,5])
def test_mature_order_promotions_respect_bounds_and_tolerance(maximum):
 records=[]
 r=solve(AdaptiveNDF(maximum),TensorState(torch.ones(2,dtype=torch.float64)),4.,tensor_rhs(lambda t,y:-y),
  dt=.001,adaptive=True,rtol=1e-7,atol=1e-9,solver_opts=OPTIONS,callback=records.append)
 assert r.successful
 selected=[x.result.history.context['order'] for x in records]
 actual=[x.result.history.context['last_order'] for x in records]
 assert min(actual)==1 and max(actual)==maximum and all(1<=p<=maximum for p in selected)
 assert all(abs(a-b)<=1 for a,b in zip(actual,selected))
 assert abs(r.state.y[0].item()-math.exp(-4))<(.0001 if maximum==1 else 2e-6)
 assert all(x.error_norm<=1 for x in records)

@pytest.mark.parametrize('tolerance',[1e-4,1e-6,1e-8])
def test_nonlinear_nonautonomous_refinement_and_work(tolerance):
 rhs=tensor_rhs(lambda t,y:-y+.2*y*y-.2*math.exp(-2*t))
 r=solve(AdaptiveNDF(),TensorState(torch.ones(1,dtype=torch.float64)),4.,rhs,dt=.1,
  adaptive=True,rtol=tolerance,atol=tolerance*.01,solver_opts=OPTIONS)
 assert r.successful and abs(r.state.y.item()-math.exp(-4))<30*tolerance


def test_portable_dynamic_order_checkpoint_matches_in_memory_continuation(tmp_path):
 method=AdaptiveNDF();rhs=tensor_rhs(lambda t,y:-y)
 first=solve(method,TensorState(torch.ones(1,dtype=torch.float64)),4.,rhs,dt=.001,
  adaptive=True,rtol=1e-7,atol=1e-9,solver_opts=OPTIONS,
  callback=lambda r:not (r.result.state.t>=.5 and r.result.history.context['phase']=='warm'
                        and r.result.history.context['order']>=3))
 assert first.history.context['order']>=3
 before=snapshot_state(first.history)
 path=tmp_path/'adaptive.pt';save_checkpoint(first.checkpoint,path)
 a=solve(method,None,2.,rhs,restart=load_checkpoint(path),solver_opts=OPTIONS)
 b=solve(method,None,2.,rhs,restart=first.checkpoint,solver_opts=OPTIONS)
 assert torch.equal(a.state.y,b.state.y)
 assert a.history.context['order']==b.history.context['order']
 assert torch.equal(first.history.context['differences'],before.context['differences'])


def test_rejected_trials_and_failed_policy_do_not_change_accepted_controller_memory():
 from test_run import Transaction
 class Invalid(AdaptiveNDF):
  def adapt_step(self,result,h,proposal,**kwargs):
   result.history.context['order']=5
   raise RuntimeError('policy failure')
 tx=Transaction();initial=TensorState(torch.ones(1,dtype=torch.float64))
 with pytest.raises(RuntimeError,match='policy failure'):
  solve(Invalid(),initial,.1,tensor_rhs(lambda t,y:-y),dt=.001,adaptive=True,transaction=tx,solver_opts=OPTIONS)
 assert tx.commits==0 and tx.rollbacks==1 and initial.y.item()==1


def test_controller_uses_runner_field_norm_and_respects_dt_max():
 seen=[]
 class Observe(AdaptiveNDF):
  def adapt_step(self,result,h,proposal,**kwargs):
   seen.append(kwargs['norm_estimate'](result.error))
   assert seen[-1]==pytest.approx(kwargs['error_norm'])
   return super().adapt_step(result,h,proposal,**kwargs)
 records=[]
 r=solve(Observe(),TensorState(torch.ones(2,dtype=torch.float64)),1.,tensor_rhs(lambda t,y:-y),
  dt=.001,dt_max=.01,adaptive=True,rtol={'y':1e-6},atol={'y':1e-9},solver_opts=OPTIONS,callback=records.append)
 assert r.successful and len(seen)==r.accepted_steps and max(x.dt for x in records)<=.01


def test_order_changes_can_reduce_and_zero_defects_do_not_raise_order():
 method=AdaptiveNDF();r=solve(method,TensorState(torch.ones(1,dtype=torch.float64)),1.,tensor_rhs(lambda t,y:-y),
  dt=.001,adaptive=True,rtol=1e-7,atol=1e-9,solver_opts=OPTIONS)
 p=r.history.context['order'];state,history=r.state,r.history
 for _ in range(p+1):
  result=NDF(p)(state,r.last_dt,tensor_rhs(lambda t,y:-y),history=history,solver_opts=OPTIONS)
  state,history=result.state,result.history
 result.history.context['constant_steps']=100
 norm_calls=[]
 def norm(error):norm_calls.append(error);return 1e-12 if len(norm_calls)==1 else 1e6
 method.adapt_step(result,r.last_dt,r.last_dt,error_norm=.9,norm_estimate=norm)
 assert result.history.context['order']==p-1
 records=[]
 zero=solve(method,TensorState(torch.ones(2,dtype=torch.float64)),1.,tensor_rhs(lambda t,y:y*0),dt=.01,
  adaptive=True,callback=records.append)
 assert zero.successful and all(x.result.history.context['order']==1 for x in records)


def test_foreign_history_explicit_restart_and_insufficient_capacity():
 from warpSPHIntegrators.state import StepHistory
 method=AdaptiveNDF()
 with pytest.raises(ValueError,match='capacity'):
  method(TensorState(torch.ones(1,dtype=torch.float64)),.1,tensor_rhs(lambda t,y:-y),history=StepHistory(1))
 first=solve(method,TensorState(torch.ones(1,dtype=torch.float64)),1.,tensor_rhs(lambda t,y:-y),dt=.001,
  adaptive=True,rtol=1e-6,solver_opts=OPTIONS)
 result=method(first.state,.01,tensor_rhs(lambda t,y:-y),history=first.history,restart_history=True,solver_opts=OPTIONS)
 assert result.history.context['last_order']==1
 foreign=solve(NDF(3),TensorState(torch.ones(1,dtype=torch.float64)),.2,tensor_rhs(lambda t,y:-y),dt=.01,solver_opts=OPTIONS)
 result=method(foreign.state,.01,tensor_rhs(lambda t,y:-y),history=StepHistory(5,foreign.history.entries,context=foreign.history.context),solver_opts=OPTIONS)
 assert result.history.context['last_order']==1


def test_full_state_quantity_tolerances_and_owned_lifecycle():
 from test_ndf import system,mechanical
 records=[];initial=system()
 result=solve(AdaptiveNDF(),initial,1.,mechanical,dt=.001,adaptive=True,
  rtol={'x':1e-6,'u':1e-5,'e':1e-7},atol=1e-9,solver_opts=OPTIONS,callback=records.append)
 reference=solve(NDF(5),system(),1.,mechanical,dt=.005,solver_opts=OPTIONS)
 assert result.successful and reference.successful
 for field in ('x','u','e'):
  assert torch.allclose(getattr(result.state.state,field),getattr(reference.state.state,field),atol=5e-5,rtol=0)
 assert result.state.initialized==result.state.finalized==result.accepted_steps
 assert initial.initialized==initial.finalized==0
 assert torch.equal(result.state.state.density,result.state.state.x.square()+result.state.state.e)
 assert all(r.error_norm<=1 for r in records)


def test_parameter_gradient_through_order_transitions():
 def forward(value,tracking=False):
  parameter=torch.tensor(value,dtype=torch.float64,requires_grad=tracking)
  result=solve(AdaptiveNDF(),TensorState(torch.ones(1,dtype=torch.float64)),1.,
   tensor_rhs(lambda t,y:-parameter*y-.05*y**3),dt=.001,adaptive=True,
   rtol=1e-8,atol=1e-10,solver_opts=OPTIONS)
  return result.state.y.sum(),parameter,result.history.context['order']
 value,parameter,order=forward(2.,True);gradient=torch.autograd.grad(value,parameter)[0]
 finite=(forward(2.+1e-4)[0]-forward(2.-1e-4)[0])/2e-4
 assert order>=3 and gradient.item()==pytest.approx(finite.item(),rel=2e-5,abs=1e-8)


def test_backward_adaptation_and_invalid_policy_return_rollback():
 from test_run import Transaction
 result=solve(AdaptiveNDF(),TensorState(torch.ones(1,dtype=torch.float64)),-1.,tensor_rhs(lambda t,y:-y),
  dt=.001,adaptive=True,rtol=1e-7,atol=1e-9,solver_opts=OPTIONS)
 assert result.successful and abs(result.state.y.item()-math.e)<3e-6
 class Invalid(AdaptiveNDF):
  def adapt_step(self,*args,**kwargs):return float('nan')
 tx=Transaction()
 with pytest.raises(ValueError,match='positive finite'):
  solve(Invalid(),TensorState(torch.ones(1,dtype=torch.float64)),.1,tensor_rhs(lambda t,y:-y),
   dt=.001,adaptive=True,transaction=tx,solver_opts=OPTIONS)
 assert tx.commits==0 and tx.rollbacks==1


@pytest.mark.parametrize('value',[-1.,float('nan'),float('inf'),True])
def test_invalid_candidate_errors(value):
 with pytest.raises(ValueError):ndf_order_proposals(3,value)

@pytest.mark.parametrize('options',[dict(max_order=0),dict(max_order=6),dict(max_order=True),dict(max_step_ratio=3)])
def test_bad_adaptive_configuration(options):
 with pytest.raises(ValueError):AdaptiveNDF(**options)
