"""Source recurrence steps, time consistency, stage bounds and run contracts."""
import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch
from dataclasses import dataclass

from warpSPHIntegrators import (StabilizedRK, stabilized_coefficients, stabilized_interval,
    TensorState,tensor_rhs,solve,save_checkpoint,load_checkpoint,
    BaseState,integrated,copied,reference_state,testing)

SOURCE=json.loads((Path(__file__).parents[1]/'scripts/coefficients/wave11/stabilized.json').read_text())
FAMILIES=['rkl1','rkl2','rkg1','rkg2']


@pytest.mark.parametrize('key',list(SOURCE['records']))
def test_recurrence_and_stage_times_match_unchanged_source_steps(key):
    r=SOURCE['records'][key];family,stages,kind=key.split('_');times=[]
    def force(t,y):
        times.append(t);forced=kind=='forced'
        return torch.stack((-2*y[0]+.1*y[1]**2+(.2*math.sin(t) if forced else 0),
            -.5*y[1]+.2*y[0]**2+(.1*math.cos(t) if forced else 0)))
    initial=TensorState(torch.tensor(r['initial'],dtype=torch.float64),t=r['t'])
    result=StabilizedRK(family.lower(),int(stages[1:]))(initial,r['h'],tensor_rhs(force))
    assert result.step_status.successful
    assert result.state.y.tolist()==pytest.approx(r['state'],abs=1e-13)
    assert result.error.y.tolist()==pytest.approx(r['euler_difference'],abs=1e-13)
    assert times==pytest.approx(r['rhs_times'],abs=1e-13)
    assert len(times)==r['rhs_calls']==len(result.stages)
    assert torch.equal(initial.y,torch.tensor(r['initial'],dtype=torch.float64)) and initial.t==r['t']


@pytest.mark.parametrize('family',FAMILIES)
@pytest.mark.parametrize('stages',[3,5,21])
def test_nonlinear_nonautonomous_order_and_reported_difference_exponent(family,stages):
    errors=[];estimates=[];method=StabilizedRK(family,stages)
    rhs=tensor_rhs(lambda t,y:-y+.2*y*y-.2*math.exp(-2*t))
    for h in [.1,.05,.025]:
        result=solve(method,TensorState(torch.ones(1,dtype=torch.float64)),1.,rhs,dt=h)
        assert result.successful
        errors.append(abs(result.state.y.item()-math.exp(-1)))
        trial=method(TensorState(torch.ones(1,dtype=torch.float64)),h,rhs)
        estimates.append(abs(trial.error.y.item()))
        assert trial.estimate_order==2
    assert math.log(errors[-2]/errors[-1],2)==pytest.approx(method.order,abs=.08)
    assert math.log(estimates[-2]/estimates[-1],2)==pytest.approx(2,abs=.08)


def gegenbauer(s,x):
    previous2=1.;previous=3*x
    for j in range(2,s+1):previous2,previous=previous,(2*j+1)/j*x*previous-(j+1)/j*previous2
    return previous


@pytest.mark.parametrize('family',FAMILIES)
@pytest.mark.parametrize('stages',[3,9,51])
def test_known_stability_polynomial_and_conservative_negative_real_interval(family,stages):
    s=stages
    if family=='rkl1':w1=2/(s*(s+1));b=1.;a=0.
    elif family=='rkl2':w1=4/(s*s+s-2);b=(s*s+s-2)/(2*s*(s+1));a=1-b
    elif family=='rkg1':w1=4/(s*(s+3));b=2/((s+1)*(s+2));a=0.
    else:w1=6/((s+4)*(s-1));b=4*(s-1)*(s+4)/(3*s*(s+1)*(s+2)*(s+3));a=1-b*(s+1)*(s+2)/2
    method=StabilizedRK(family,s)
    for z in np.linspace(-stabilized_interval(family,s),0,21):
        x=1+w1*z
        expected=a+b*(np.polynomial.legendre.Legendre.basis(s)(x) if family.startswith('rkl') else gegenbauer(s,x))
        result=method(TensorState(torch.ones(1,dtype=torch.float64)),1.,tensor_rhs(lambda t,y:z*y))
        assert result.state.y.item()==pytest.approx(expected,abs=3e-12)
        assert abs(result.state.y.item())<=1+3e-12
    coeff=stabilized_coefficients(family,s)
    assert coeff.c[-1]==pytest.approx(1,abs=1e-13)
    assert sum(coeff.b)==pytest.approx(1,abs=1e-13)
    with pytest.raises(TypeError):coeff.b[0]=0


@pytest.mark.parametrize('family',FAMILIES)
def test_spectral_stage_selection_caps_and_adaptive_portable_restart(family,tmp_path):
    method=StabilizedRK(family,lambda_max=100.)
    for h in [.01,.1,1.]:
        s=method.stage_count(h)
        assert stabilized_interval(family,s)>=h*100
        if s>(3 if family!='rkg1' else 2):
            assert stabilized_interval(family,s-(2 if family.startswith('rkl') else 1))<h*100
    with pytest.raises(ValueError,match='max_stages'):
        StabilizedRK(family,lambda_max=1e12,max_stages=9).stage_count(1.)
    rhs=tensor_rhs(lambda t,y:-y+.2*y*y-.2*math.exp(-2*t))
    errors=[]
    for tolerance in [1e-3,1e-6]:
        run=solve(method,TensorState(torch.ones(1,dtype=torch.float64)),1.,rhs,dt=.5,
            adaptive=True,rtol=tolerance,atol=tolerance*.01)
        assert run.successful and run.rejected_steps>0
        errors.append(abs(run.state.y.item()-math.exp(-1)))
    assert errors[1]<errors[0]/10
    first=solve(method,TensorState(torch.ones(1,dtype=torch.float64)),.2,rhs,dt=.025)
    path=tmp_path/'stabilized.pt';save_checkpoint(first.checkpoint,path)
    a=solve(method,None,.4,rhs,restart=load_checkpoint(path))
    b=solve(method,None,.4,rhs,restart=first.checkpoint)
    assert torch.equal(a.state.y,b.state.y)


@pytest.mark.parametrize('family',FAMILIES)
def test_parameter_gradient_and_coupled_components(family):
    def forward(value,tracking=False):
        k=torch.tensor(value,dtype=torch.float64,requires_grad=tracking)
        y=torch.tensor([1.,.2],dtype=torch.float64,requires_grad=tracking)
        rhs=tensor_rhs(lambda t,y:torch.stack((-k*y[0]+.1*y[1]**2,-.4*y[1]+.2*y[0])))
        result=solve(StabilizedRK(family,5),TensorState(y),.4,rhs,dt=.05)
        return result.state.y.sum(),y,k
    value,y,k=forward(2.,True);dy,dk=torch.autograd.grad(value,(y,k));eps=1e-5
    finite=(forward(2+eps)[0]-forward(2-eps)[0])/(2*eps)
    assert torch.isfinite(dy).all()
    assert dk.item()==pytest.approx(finite.item(),rel=1e-6,abs=1e-9)


@pytest.mark.parametrize('options',[
    dict(family='unknown'),dict(stages=True),dict(stages=0),dict(family='rkl1',stages=4),
    dict(lambda_max=-1),dict(lambda_max=float('nan')),dict(lambda_max=True),dict(lambda_max='bad'),
    dict(stages=5,lambda_max=1),dict(max_stages=2),dict(stages=11,max_stages=9)])
def test_invalid_stabilized_configuration(options):
    with pytest.raises(ValueError):StabilizedRK(**options)


@dataclass
class Fields(BaseState):
    x: torch.Tensor=integrated('dxdt',tags=('position',))
    u: torch.Tensor=integrated('dudt',tags=('velocity',))
    e: torch.Tensor=integrated('dedt',tags=('quantity',))
    density: torch.Tensor=copied()


@dataclass
class System(testing.ParticleSystem):
    state: Fields=reference_state()
    finalized: int=0
    def initializeNewState(self,*args,**kwargs):return System(self.state.initializeNewState(),self.t,self.finalized)
    def preprocess(self,initial,dt,**kwargs):self.state.density=self.state.x.square()+self.state.e;return self
    def finalize(self,initial,dt,auxes,updates,weights,**kwargs):
        assert len(auxes)==len(updates)==len(weights)
        assert sum(weights)==pytest.approx(1,abs=1e-13)
        self.finalized=initial.finalized+1
        return self


@pytest.mark.parametrize('family',FAMILIES)
def test_full_quantity_coupling_lifecycle_copied_endpoint_and_failed_trial(family):
    initial=System(Fields(torch.ones(2,dtype=torch.float64),torch.zeros(2,dtype=torch.float64),
        torch.zeros(2,dtype=torch.float64),torch.zeros(2,dtype=torch.float64)))
    def force(state,dt,**kwargs):
        s=state.state
        return testing.ParticleUpdate(s.u,-s.x+.1*s.e,s.e+.1*s.x*s.u),None
    method=StabilizedRK(family,5)
    result=solve(method,initial,.2,force,dt=.05)
    assert result.successful and result.state.finalized==4 and initial.finalized==0
    assert torch.equal(result.state.state.density,result.state.state.x.square()+result.state.state.e)
    assert result.state.state.e.abs().max()>1e-4
    assert torch.equal(initial.state.density,torch.zeros(2,dtype=torch.float64))
    def broken(state,dt,**kwargs):
        update,_=force(state,dt)
        if state.t>=.05:update.dedt=update.dedt*float('nan')
        return update,None
    failed=solve(method,initial,.2,broken,dt=.05,raise_on_failure=False)
    assert not failed.successful and failed.accepted_steps==0
    assert failed.state.finalized==0 and initial.finalized==0


@pytest.mark.parametrize('family',FAMILIES)
@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
def test_cpu_dtype_and_backward_time(family,dtype):
    result=solve(StabilizedRK(family,5),TensorState(torch.ones(3,dtype=dtype)),-.1,
        tensor_rhs(lambda t,y:-y),dt=.005)
    assert result.successful and result.state.y.dtype==dtype
    assert torch.allclose(result.state.y,torch.full((3,),math.exp(.1),dtype=dtype),atol=2e-4,rtol=0)
