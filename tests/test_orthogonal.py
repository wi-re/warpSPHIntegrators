"""Actual source recurrences in increment coordinates and known full-state gates."""
import json
import math
from pathlib import Path
from dataclasses import dataclass

import numpy as np
import pytest
import torch

from warpSPHIntegrators.schemes import OrthogonalRK
from warpSPHIntegrators import TensorState, tensor_rhs, solve
from warpSPHIntegrators.runtime import save_checkpoint, load_checkpoint
import warpSPHIntegrators.testing as testing
from warpSPHIntegrators.wrappers import AlgorithmSwitch, Strang
from test_nystrom import System, system
from warpSPHIntegrators.schemes.coefficients.orthogonal import orthogonal_coefficients, recurrence_program, available_stages, sizing_bound, _parameters

SOURCE=json.loads((Path(__file__).parents[1]/'scripts/coefficients/wave11/orthogonal_stabilized.json').read_text())
ROWS=[r for r in SOURCE['records'] if r['translated']]
METHODS=['ROCK2','ROCK4','SERK2','RKMC2']


@pytest.mark.parametrize('row',ROWS,ids=lambda r:f"{r['method']}-{r['requested_stages']}-{r['problem']}")
def test_source_translated_maps_defects_and_every_physical_stage(row):
    calls=[]
    def rhs(t,y):
        kind=row['problem'];calls.append((t,y.clone()))
        if kind=='zero':return torch.zeros_like(y)
        return torch.stack((-2*y[0]+.1*y[0]*y[1]+(.2*math.sin(t) if kind=='forced' else 0),
            -3*y[1]+.2*y[0]**2+(.1*math.cos(t) if kind=='forced' else 0)))
    state=TensorState(torch.tensor(row['initial'],dtype=torch.float64),t=row['t'])
    method=OrthogonalRK(row['method'],row['requested_stages']);result=method(state,row['h'],tensor_rhs(rhs))
    assert result.step_status.successful
    assert result.state.y.tolist()==pytest.approx(row['output'],rel=0,abs=2e-13)
    assert result.error.y.tolist()==pytest.approx(row['error'],rel=0,abs=2e-13)
    expected=row['calls'][1:] if row['method']=='SERK2' else row['calls']
    assert len(calls)==len(expected)==method.metadata.stage_count
    for actual,reference in zip(calls,expected):
        assert actual[0]==pytest.approx(reference['t'],rel=0,abs=2e-14)
        assert actual[1].tolist()==pytest.approx(reference['state'],rel=0,abs=2e-13)
    reconstructed=state.y+row['h']*sum(w*s.update.dy for w,s in zip(result.quadrature,result.stages))
    assert torch.allclose(reconstructed,result.state.y,rtol=0,atol=2e-13)
    if row['problem']=='zero':assert torch.equal(result.state.y,state.y) and torch.count_nonzero(result.error.y)==0


@pytest.mark.parametrize('method',METHODS)
def test_nonlinear_nonautonomous_order(method):
    errors=[]
    for h in [.2,.1,.05,.025]:
        result=solve(OrthogonalRK(method,10),TensorState(torch.ones(1,dtype=torch.float64)),1.,
            tensor_rhs(lambda t,y:-y+.2*y*y-.2*math.exp(-2*t)),dt=h)
        assert result.successful
        errors.append(abs(result.state.y.item()-math.exp(-1)))
    assert math.log(errors[-2]/errors[-1],2)>=OrthogonalRK(method).order-.2,(method,errors)


@pytest.mark.parametrize('method',METHODS)
def test_stability_through_the_declared_sizing_bound(method):
    stages=available_stages(method)
    for s in [stages[0],stages[min(3,len(stages)-1)],stages[len(stages)//2],stages[-1]]:
        z=-np.linspace(0,sizing_bound(method,s),2048)
        output,_=recurrence_program(method,s,_parameters(method,s),lambda v,endpoint=False:z*(1+v),np.zeros_like(z))
        assert np.max(np.abs(1+output))<=1+2e-10,(method,s,np.max(np.abs(1+output)))
        configured=OrthogonalRK(method,lambda_max=sizing_bound(method,s))
        assert configured.stages_for(1.)<=s
    with pytest.raises(ValueError,match='max_stages'):
        OrthogonalRK(method,lambda_max=1e12).stages_for(1.)


@pytest.mark.parametrize('method',METHODS)
def test_state_and_parameter_gradients(method):
    def forward(q,k,tracking=False):
        y=torch.tensor([q],dtype=torch.float64,requires_grad=tracking)
        parameter=torch.tensor(k,dtype=torch.float64,requires_grad=tracking)
        result=solve(OrthogonalRK(method,10),TensorState(y),.25,
            tensor_rhs(lambda t,y:-parameter*y-.3*y*y+.2*t),dt=.0625)
        return result.state.y.sum(),y,parameter
    value,y,k=forward(.7,2.,True);dy,dk=torch.autograd.grad(value,(y,k));eps=1e-5
    assert dy.item()==pytest.approx(((forward(.7+eps,2.)[0]-forward(.7-eps,2.)[0])/(2*eps)).item(),rel=3e-7,abs=1e-9)
    assert dk.item()==pytest.approx(((forward(.7,2.+eps)[0]-forward(.7,2.-eps)[0])/(2*eps)).item(),rel=3e-7,abs=1e-9)


def test_pinned_high_stage_instability_counterexamples_are_excluded_or_bounded():
    for row in SOURCE['stability_counterexamples']:
        name=row['method'];s=row['requested_stages'];z=row['linear_z']
        output,_=recurrence_program(name,s,_parameters(name,s),lambda q,endpoint=False:z*(1+q),0.)
        assert abs(row['output'][0])>1.001 and abs(1+output)>1.001
        if name=='ROCK4':
            with pytest.raises(ValueError):OrthogonalRK(name,s)
        else:assert sizing_bound(name,s)<abs(z)


@pytest.mark.parametrize('method',METHODS)
def test_rk_moments_and_component_quadrature(method):
    s=10;n=s+1;rows=[]
    def force(offset,*,endpoint=False):
        rows.append(offset.copy());unit=np.zeros(n);unit[len(rows)-1]=1.
        return unit
    b,error=recurrence_program(method,s,_parameters(method,s),force,np.zeros(n))
    a=np.array(rows);c=a.sum(axis=1)
    assert b.sum()==pytest.approx(1.,abs=2e-12)
    assert b@c==pytest.approx(.5,abs=2e-12)
    assert error.sum()==pytest.approx(0.,abs=2e-12)
    if method=='ROCK4':
        assert [b@(c*c),b@a@c,b@(c**3),b@(c*(a@c)),b@a@(c*c),b@a@a@c]==pytest.approx(
            [1/3,1/6,1/4,1/8,1/12,1/24],abs=2e-12)
        companion=b+error
        assert [companion.sum(),companion@c,companion@(c*c),companion@a@c]==pytest.approx([1,.5,1/3,1/6],abs=2e-12)


@pytest.mark.parametrize('method',METHODS)
def test_adaptive_rejection_accuracy_refinement_and_portable_continuation(method,tmp_path):
    rhs=tensor_rhs(lambda t,y:-4*y+.2*y*y+3*math.exp(-t)-.2*math.exp(-2*t));errors=[]
    scheme=OrthogonalRK(method,lambda_max=10.)
    for tol in [1e-3,1e-6]:
        first=solve(scheme,TensorState(torch.ones(1,dtype=torch.float64)),.5,rhs,dt=.5,
            adaptive=True,rtol=tol,atol=tol*.01)
        assert first.successful
        path=tmp_path/'orthogonal.pt';save_checkpoint(first.checkpoint,path)
        resumed=solve(scheme,None,1.,rhs,restart=load_checkpoint(path))
        same=solve(scheme,None,1.,rhs,restart=first.checkpoint)
        assert resumed.successful and torch.equal(resumed.state.y,same.state.y)
        if tol==1e-6:assert first.rejected_steps>0
        errors.append(abs(resumed.state.y.item()-math.exp(-1)))
    assert errors[-1]<errors[0]/8,(method,errors)


@dataclass
class RecurrenceSystem(System):
    def initializeNewState(self,*args,**kwargs):
        return RecurrenceSystem(self.state.initializeNewState(),self.t,self.initialized,self.finalized)
    def finalize(self,initial,dt,auxes,updates,weights,**kwargs):
        assert len(auxes)==len(updates)==len(weights)
        for key,rate in [('x','dxdt'),('u','dudt'),('e','dedt')]:
            expected=getattr(initial.state,key)+dt*sum(w*getattr(k,rate) for w,k in zip(weights,updates))
            assert torch.allclose(getattr(self.state,key),expected,rtol=0,atol=2e-12)
        self.finalized=initial.finalized+1
        return self


@pytest.mark.parametrize('method',METHODS)
def test_coupled_quantities_lifecycle_copied_endpoint_and_failed_trial_ownership(method):
    start=RecurrenceSystem(system().state);start.state.u=torch.ones_like(start.state.u);start.state.e=torch.ones_like(start.state.e)
    def rhs(current,dt,**kwargs):
        s=current.state;target=math.exp(current.t)
        return testing.ParticleUpdate(s.u,-s.x+.1*s.e*s.e+2*target-.1*target*target,
            .3*s.e+.1*s.x*s.u+.7*target-.1*target*target),None
    result=solve(OrthogonalRK(method),start,.25,rhs,dt=.0625)
    assert result.successful and result.state.finalized==result.state.initialized==4
    assert start.finalized==start.initialized==0
    assert torch.equal(result.state.state.density,result.state.state.x.square()+result.state.state.e)
    assert max(float((getattr(result.state.state,key)-math.exp(.25)).abs().max()) for key in ['x','u','e'])<1e-3
    def broken(current,dt,**kwargs):
        update,aux=rhs(current,dt)
        if current.t>=.02:update.dudt=update.dudt*float('nan')
        return update,aux
    failed=solve(OrthogonalRK(method),start,.25,broken,dt=.0625,raise_on_failure=False)
    assert not failed.successful and failed.accepted_steps==0 and failed.state.finalized==0


@pytest.mark.parametrize('method',METHODS)
def test_switch_and_physical_composition(method):
    from test_operator_splitting import error
    composed=Strang(OrthogonalRK(method),'RK4')
    errors=[error(composed,h) for h in [.1,.05,.025]]
    assert math.log(errors[-2]/errors[-1],2)==pytest.approx(2,abs=.15)
    switch=AlgorithmSwitch((OrthogonalRK(method),'RK4'),lambda s,p:0 if s.t<.1 else 1,'orthogonal-'+method)
    result=solve(switch,TensorState(torch.ones(1,dtype=torch.float64)),.25,tensor_rhs(lambda t,y:-y),dt=.0625)
    assert result.successful


@pytest.mark.parametrize('method',METHODS)
@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
def test_zero_rates_are_bitwise_passive_even_with_large_offsets(method,dtype):
    y=torch.tensor([1e8,1.,1e-8],dtype=dtype)
    result=solve(OrthogonalRK(method),TensorState(y),1.,tensor_rhs(lambda t,y:torch.zeros_like(y)),dt=.125)
    assert result.successful and torch.equal(result.state.y,y) and result.state.y.dtype==dtype


@pytest.mark.parametrize('options',[dict(method='bad'),dict(stages=True),dict(stages=2),dict(stages=1001),
    dict(stages=10,lambda_max=1.),dict(lambda_max=-1),dict(lambda_max=float('inf')),dict(max_stages=0),
    dict(method='ROCK4',stages=133),dict(method='ROCK4',max_stages=152)])
def test_invalid_configuration(options):
    with pytest.raises(ValueError):OrthogonalRK(**options)
