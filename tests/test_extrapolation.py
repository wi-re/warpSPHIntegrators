"""Pinned explicit extrapolation, full-state affine steps and consumer behavior."""
from dataclasses import dataclass
import json
import math
from pathlib import Path
import numpy as np
import pytest
import torch
from warpSPHIntegrators import (Extrapolation,ExtrapolationQuadrature,extrapolation_coefficients,
    TensorState,tensor_rhs,solve,getIntegrator,Strang,ConservativeDissipativeRHS,
    save_checkpoint,load_checkpoint)

SOURCE=json.loads((Path(__file__).parents[1]/'scripts/coefficients/wave13/extrapolation.json').read_text())
METHODS=[Extrapolation('euler',p) for p in range(2,9)]+[Extrapolation('midpoint',p) for p in range(2,11,2)]


@pytest.mark.parametrize('key',SOURCE['records'])
def test_source_main_error_coefficients_and_nonautonomous_times(key):
    r=SOURCE['records'][key];forced=key.endswith('forced')
    def force(t,y):
        return torch.stack((-2*y[0]+.1*y[1]**2+(.2*math.sin(t) if forced else 0),
            -.5*y[1]+.2*y[0]**2+(.1*math.cos(t) if forced else 0)))
    method=Extrapolation(r['base'],r['order']);calls=[]
    def rhs(t,y):calls.append(t);return force(t,y)
    result=method(TensorState(torch.tensor(r['initial'],dtype=torch.float64),t=r['t']),r['h'],tensor_rhs(rhs))
    assert result.state.y.tolist()==pytest.approx(r['state'],rel=0,abs=3e-14)
    assert result.error.y.tolist()==pytest.approx(r['error'],rel=0,abs=3e-14)
    sequence,main,companion=extrapolation_coefficients(method.base,method.order)
    if method.base=='midpoint':
        assert main==pytest.approx(r['main_column_weights'],rel=0,abs=2e-14)
        assert companion==pytest.approx(r['comparison_column_weights'],rel=0,abs=2e-14)
    expected=[r['t']]+[r['t']+j*r['h']/n for n in sequence for j in range(1,n)]+[r['t']+r['h']]
    assert calls==pytest.approx(expected,rel=0,abs=1e-15)
    assert len(calls)==method.metadata.stage_count


@pytest.mark.parametrize('method',METHODS,ids=lambda m:m.name)
def test_nonlinear_forced_convergence_and_defect_exponents(method):
    errors=[]
    for count in (1,2,4):
        result=solve(method,TensorState(torch.ones(1,dtype=torch.float64)),1.,
            tensor_rhs(lambda t,y:-2*y+.2*y*y-.2*math.exp(-4*t)),dt=1/count)
        assert result.successful
        errors.append(abs(result.state.y.item()-math.exp(-2)))
    assert math.log(errors[-2]/errors[-1],2)>=method.order-.4,(method,errors)
    estimates=[];q=method.estimate_order
    for h in (.2,.1,.05):
        result=method(TensorState(torch.zeros(1,dtype=torch.float64)),h,tensor_rhs(lambda t,y:torch.full_like(y,t**(q-1))))
        estimates.append(abs(result.error.y.item()))
    assert math.log(estimates[-2]/estimates[-1],2)==pytest.approx(q,rel=0,abs=1e-5)


@pytest.mark.parametrize('method',METHODS,ids=lambda m:m.name)
def test_polynomial_order_conditions_and_exact_zero_rate_offsets(method):
    ns,alpha,comp=extrapolation_coefficients(method.base,method.order)
    power=1 if method.base=='euler' else 2
    for j in range(len(ns)):
        assert sum(a/n**(power*j) for n,a in zip(ns,alpha))==pytest.approx(float(j==0),rel=0,abs=2e-13)
    # Build each independent scalar stability polynomial using the recurrence.
    poly=np.polynomial.Polynomial;z=poly([0,1]);columns=[]
    for n in ns:
        previous=poly([1]);current=previous+z/n
        for _ in range(1,n):
            following=current*(1+z/n) if method.base=='euler' else previous+2*z/n*current
            previous,current=current,following
        columns.append(current)
    combined=sum(a*p for a,p in zip(alpha,columns))
    assert combined.coef[:method.order+1]==pytest.approx([1/math.factorial(j) for j in range(method.order+1)],rel=0,abs=2e-13)
    state=TensorState(torch.tensor([1e12,-1e12],dtype=torch.float64));calls=[]
    def zero(t,y):calls.append(t);return y*0
    result=method(state,.1,tensor_rhs(zero))
    assert torch.equal(result.state.y,state.y) and torch.equal(result.error.y,torch.zeros_like(state.y))
    assert len(calls)==method.metadata.stage_count and state.t==0


@pytest.mark.parametrize('base,order',[('euler',4),('midpoint',6)])
def test_adaptive_refinement_rejections_and_portable_restart(base,order,tmp_path):
    method=Extrapolation(base,order);rhs=tensor_rhs(lambda t,y:-y+.2*y*y-.2*math.exp(-2*t));errors=[]
    for tolerance in (1e-4,1e-7):
        result=solve(method,TensorState(torch.ones(1,dtype=torch.float64)),2.,rhs,dt=2.,
            adaptive=True,rtol=tolerance,atol=tolerance*.01)
        assert result.successful and result.rejected_steps>0
        errors.append(abs(result.state.y.item()-math.exp(-2)))
    assert errors[-1]<errors[0]/10
    first=solve(method,TensorState(torch.ones(1,dtype=torch.float64)),.3,rhs,dt=.1)
    path=tmp_path/'extrapolation.pt';save_checkpoint(first.checkpoint,path)
    a=solve(method,None,.6,rhs,restart=load_checkpoint(path))
    b=solve(method,None,.6,rhs,restart=first.checkpoint)
    assert torch.equal(a.state.y,b.state.y)


@pytest.mark.parametrize('base,order',[('euler',4),('midpoint',6)])
def test_parameter_gradients_and_physical_composition(base,order):
    parameter=torch.tensor(2.,dtype=torch.float64,requires_grad=True)
    result=solve(Extrapolation(base,order),TensorState(torch.ones(1,dtype=torch.float64)),.4,
        tensor_rhs(lambda t,y:-parameter*y),dt=.025)
    gradient=torch.autograd.grad(result.state.y.sum(),parameter)[0]
    assert gradient.item()==pytest.approx(-.4*math.exp(-.8),rel=2e-6)
    split=tensor_rhs(ConservativeDissipativeRHS(lambda t,y:y,lambda t,y:-2*y))
    result=solve(Strang(Extrapolation(base,order),'RK4'),TensorState(torch.ones(1,dtype=torch.float64)),.4,split,dt=.025)
    assert result.successful and abs(result.state.y.item()-math.exp(-.4))<1e-7


def test_order_two_main_map_equivalence_is_recorded_without_registration():
    state=TensorState(torch.tensor([.7,-.4],dtype=torch.float64),t=.3)
    rhs=tensor_rhs(lambda t,y:-y+y.square()+math.sin(t))
    reference=getIntegrator('Midpoint')(state,.1,rhs)
    for base in ('euler','midpoint'):
        result=Extrapolation(base,2)(state,.1,rhs)
        assert torch.allclose(result.state.y,reference.state.y,atol=1e-15,rtol=0)


def test_full_state_affine_quadrature_lifecycle_and_nonfinite_failure():
    from test_ndf import System,system,mechanical
    @dataclass
    class Consumer(System):
        extrapolation_compatible=True
        def initializeNewState(self,*args,**kwargs):return Consumer(self.state.initializeNewState(),self.t,self.initialized,self.finalized)
        def finalize(self,initial,dt,auxes,updates,quadrature,**kwargs):
            assert isinstance(quadrature,ExtrapolationQuadrature)
            assert len(quadrature.weights)==len(updates)==len(auxes)
            for field,key in [('x','dxdt'),('u','dudt'),('e','dedt')]:
                value=getattr(initial.state,field)
                expected=value+sum(dt*w*getattr(k,key) for w,k in zip(quadrature.weights,updates))
                assert torch.allclose(getattr(self.state,field),expected,atol=1e-14,rtol=0)
            self.finalized=initial.finalized+1;return self
    initial=Consumer(system().state)
    result=solve(Extrapolation(),initial,.4,mechanical,dt=.025)
    assert result.successful and result.state.finalized==result.state.initialized==16
    assert initial.finalized==initial.initialized==0
    assert torch.equal(result.state.state.density,result.state.state.x.square()+result.state.state.e)
    bad=solve(Extrapolation(),initial,.1,lambda s,h,**kw:(type(mechanical(s,h)[0])(*[torch.full_like(s.state.x,float('nan'))]*3),None),
        dt=.1,raise_on_failure=False)
    assert not bad.successful and bad.state.finalized==0 and bad.state.t==0


@pytest.mark.parametrize('options',[dict(base='other'),dict(order=True),dict(base='euler',order=1),
    dict(base='euler',order=9),dict(base='midpoint',order=3),dict(base='midpoint',order=12)])
def test_bad_configurations(options):
    with pytest.raises(ValueError):Extrapolation(**options)


def test_custom_finalize_requires_opt_in_and_cannot_project_values():
    from test_ndf import System,system,mechanical
    with pytest.raises(TypeError,match='extrapolation_compatible'):
        Extrapolation()(system(),.01,mechanical)
    @dataclass
    class Projecting(System):
        extrapolation_compatible=True
        def initializeNewState(self,*args,**kwargs):return Projecting(self.state.initializeNewState(),self.t)
        def finalize(self,*args,**kwargs):self.state.e=self.state.e+1;return self
    original=Projecting(system().state)
    with pytest.raises(ValueError,match='finalize must preserve'):
        Extrapolation()(original,.01,mechanical)
    assert torch.equal(original.state.e,torch.zeros(2,dtype=torch.float64))


def test_backward_nonautonomous_steps_retain_time_nodes():
    result=solve(Extrapolation('midpoint',6),TensorState(torch.ones(1,dtype=torch.float64)),-1.,
        tensor_rhs(lambda t,y:-y+.2*y*y-.2*math.exp(-2*t)),dt=.025)
    assert result.successful and result.state.y.item()==pytest.approx(math.e,rel=0,abs=2e-9)
