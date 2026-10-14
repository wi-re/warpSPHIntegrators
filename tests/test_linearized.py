"""Exact full-JVP Rosenbrock source maps and full nonlinear adjoints."""
import json
import math
from pathlib import Path
from dataclasses import dataclass, field

import pytest
import torch

from warpSPHIntegrators import Rosenbrock, TensorState, tensor_rhs, solve, rosenbrock_tableau
from warpSPHIntegrators import (RosenbrockQuadrature, flatten_integrated, save_checkpoint,
    load_checkpoint, testing, AlgorithmSwitch, Strang)
from test_nystrom import System, system

SOURCE=json.loads((Path(__file__).parents[1]/'scripts/coefficients/wave11/rosenbrock.json').read_text())
METHODS=tuple(SOURCE['records'])
OPTIONS=dict(gmres_tol=1e-13,gmres_maxiter=30)


@pytest.mark.parametrize('method',METHODS)
@pytest.mark.parametrize('index',range(4))
def test_actual_pinned_step_body_maps_and_raw_defects(method,index):
    row=SOURCE.get('corrected_time_records',{}).get(method,SOURCE['records'][method])[index];forced=row['forced']
    def f(t,y):
        return torch.stack((-2*y[0]+.1*y[0]*y[1]+(.2*math.sin(t) if forced else 0),
            -3*y[1]+.2*y[0]**2+(.1*math.cos(t) if forced else 0)))
    def ft(t,y):return torch.tensor([.2*math.cos(t),-.1*math.sin(t)],dtype=y.dtype) if forced else torch.zeros_like(y)
    state=TensorState(torch.tensor(row['initial'],dtype=torch.float64),t=row['t'])
    result=Rosenbrock(method)(state,row['h'],tensor_rhs(f),f_t=tensor_rhs(ft),solver_opts=OPTIONS)
    assert result.step_status.successful
    assert result.state.y.tolist()==pytest.approx(row['output'],rel=0,abs=2e-13)
    if row['error'] is not None:assert result.error.y.tolist()==pytest.approx(row['error'],rel=0,abs=2e-13)
    else:assert result.error is None


@pytest.mark.parametrize('method',METHODS)
def test_nonautonomous_correction_equals_autonomous_time_augmentation(method):
    def dynamics(t,y):
        t=torch.as_tensor(t,dtype=y.dtype,device=y.device)
        return torch.stack((-2*y[0]+.1*y[0]*y[1]+.2*torch.sin(t),
            -3*y[1]+.2*y[0]**2+.1*torch.cos(t)))
    start=torch.tensor([1.,.4],dtype=torch.float64);t=.3;h=.2
    def augmented(t,z):return torch.cat((torch.ones_like(z[:1]),dynamics(z[0],z[1:])))
    reference=Rosenbrock(method)(TensorState(torch.cat((torch.tensor([t],dtype=start.dtype),start))),h,
        tensor_rhs(augmented),solver_opts=OPTIONS)
    partial=tensor_rhs(lambda clock,y:torch.tensor([.2*math.cos(clock),-.1*math.sin(clock)],dtype=y.dtype))
    actual=Rosenbrock(method)(TensorState(start,t=t),h,tensor_rhs(dynamics),f_t=partial,solver_opts=OPTIONS)
    assert actual.step_status.successful and reference.step_status.successful
    assert torch.allclose(actual.state.y,reference.state.y[1:],rtol=0,atol=2e-13)
    if actual.error is not None:assert torch.allclose(actual.error.y,reference.error.y[1:],rtol=0,atol=2e-13)


def test_six_stage_coefficients_match_source_exactly():
    tab=rosenbrock_tableau('RosenbrockW6S4OS');source=SOURCE['tableaus']['RosenbrockW6S4OS']
    for key in ['A','C','c','d','b']:
        assert json.loads(json.dumps(getattr(tab,key)))==source[key]
    assert tab.gamma==source['gamma'] and tab.btilde is None


def test_scalar_stability_limits_and_oscillatory_counterexamples():
    import numpy as np
    def stability(method,z):
        t=rosenbrock_tableau(method);a=np.array(t.A);c=np.array(t.C);b=np.array(t.b);g=t.gamma
        return 1+b@np.linalg.solve((1-g*z)*np.eye(len(b))-g*c-g*z*a,g*z*np.ones(len(b)))
    g=1/(2+math.sqrt(2))
    for z in [-.1,-1.,-100.,1j,10j]:
        assert stability('Rosenbrock23',z)==pytest.approx((1+(1-2*g)*z)/(1-g*z)**2,rel=0,abs=2e-14)
    # For R23, |den(ix)|²-|num(ix)|²=g^4*x^4, and its sole pole is 1/g>0.
    assert 2*g*g-(1-2*g)**2==pytest.approx(0,abs=2e-16)
    assert abs(stability('Rosenbrock23',-1e8))<1e-6
    assert stability('Rosenbrock32',-1e8)==pytest.approx((2+2*math.sqrt(2))/3,rel=1e-6)
    assert abs(stability('Rosenbrock32',1j))>1.001
    assert abs(stability('RosenbrockW6S4OS',1j))>1.00003
    assert abs(stability('RosenbrockW6S4OS',-1e8))<1e-6


@pytest.mark.parametrize('method',METHODS)
def test_nonlinear_nonautonomous_order(method):
    errors=[]
    for h in [.1,.05,.025]:
        result=solve(Rosenbrock(method),TensorState(torch.ones(1,dtype=torch.float64)),1.,
            tensor_rhs(lambda t,y:-y+.2*y*y-.2*math.exp(-2*t)),dt=h,solver_opts=OPTIONS)
        assert result.successful
        errors.append(abs(result.state.y.item()-math.exp(-1)))
    assert math.log(errors[-2]/errors[-1],2)>=Rosenbrock(method).order-.2,(method,errors)


@pytest.mark.parametrize('method',METHODS)
def test_full_state_and_parameter_gradients_include_jacobian_derivative(method):
    def forward(q,k,tracking=False):
        y=torch.tensor([q],dtype=torch.float64,requires_grad=tracking)
        parameter=torch.tensor(k,dtype=torch.float64,requires_grad=tracking)
        result=solve(Rosenbrock(method),TensorState(y),.25,
            tensor_rhs(lambda t,y:-parameter*y-.3*y*y+.2*t),dt=.0625,solver_opts=OPTIONS)
        assert result.successful
        return result.state.y.sum(),y,parameter
    loss,y,k=forward(.7,2.,True);dy,dk=torch.autograd.grad(loss,(y,k));eps=1e-5
    assert dy.item()==pytest.approx(((forward(.7+eps,2.)[0]-forward(.7-eps,2.)[0])/(2*eps)).item(),rel=3e-7,abs=1e-9)
    assert dk.item()==pytest.approx(((forward(.7,2.+eps)[0]-forward(.7,2.-eps)[0])/(2*eps)).item(),rel=3e-7,abs=1e-9)


@pytest.mark.parametrize('method',['Rosenbrock23','Rosenbrock32'])
def test_corrected_time_force_is_exact_and_source_defect_is_only_order_two(method):
    source=SOURCE['time_force_source'][method]
    g=SOURCE['tableaus'][method]['gamma']
    for row in source:
        assert row['defect'][0]==pytest.approx(row['h']**2*(1-g)/6,rel=0,abs=1e-17)
        if method=='Rosenbrock32':assert row['output'][0]-row['exact'][0]==pytest.approx(row['defect'][0])
        result=Rosenbrock(method)(TensorState(torch.zeros(1,dtype=torch.float64)),row['h'],
            tensor_rhs(lambda t,y:torch.full_like(y,t)),solver_opts=OPTIONS)
        assert result.state.y.item()==pytest.approx(row['h']**2/2,rel=0,abs=1e-16)
        assert abs(result.error.y.item())<1e-16


@pytest.mark.parametrize('method',['Rosenbrock23','Rosenbrock32'])
def test_time_polynomial_defect_has_declared_exponent_three(method):
    errors=[]
    for h in [.2,.1,.05]:
        result=Rosenbrock(method)(TensorState(torch.zeros(1,dtype=torch.float64)),h,
            tensor_rhs(lambda t,y:torch.full_like(y,t*t)),solver_opts=OPTIONS)
        errors.append(abs(result.error.y.item()))
    assert errors[0]>1e-6 and math.log(errors[-2]/errors[-1],2)==pytest.approx(3,abs=1e-10)


@pytest.mark.parametrize('method',['Rosenbrock23','Rosenbrock32'])
def test_adaptive_rejection_tolerance_refinement_and_portable_restart(method,tmp_path):
    rhs=tensor_rhs(lambda t,y:-y+.2*y*y-.2*math.exp(-2*t));errors=[]
    for tol in [1e-3,1e-8]:
        first=solve(Rosenbrock(method),TensorState(torch.ones(1,dtype=torch.float64)),.5,rhs,
            dt=.5,adaptive=True,rtol=tol,atol=tol*.01,solver_opts=OPTIONS)
        path=tmp_path/'rosen.pt';save_checkpoint(first.checkpoint,path)
        resumed=solve(Rosenbrock(method),None,1.,rhs,restart=load_checkpoint(path),solver_opts=OPTIONS)
        same=solve(Rosenbrock(method),None,1.,rhs,restart=first.checkpoint,solver_opts=OPTIONS)
        assert resumed.successful and torch.equal(resumed.state.y,same.state.y)
        if tol==1e-8:assert first.rejected_steps>0
        errors.append(abs(resumed.state.y.item()-math.exp(-1)))
    assert errors[-1]<errors[0]/10 and errors[-1]<2e-5,(method,errors)


@dataclass
class RosenSystem(System):
    rosenbrock_compatible=True
    evaluations: list=field(default_factory=list)
    def initializeNewState(self,*args,**kwargs):
        return RosenSystem(self.state.initializeNewState(),self.t,self.initialized,self.finalized,self.evaluations)
    def postprocess(self,current,dt,aux,**kwargs):
        self.evaluations.append(current.t);return self
    def finalize(self,initial,dt,auxes,updates,weights,**kwargs):
        assert isinstance(weights,RosenbrockQuadrature)
        assert len(weights.weights)==len(weights.increments)==len(updates)==len(auxes)
        expected=flatten_integrated(initial)+sum(w*k for w,k in zip(weights.weights,weights.increments))
        assert torch.allclose(flatten_integrated(self),expected,atol=1e-13,rtol=0)
        # JVP/time probes use isolated bases, so only the physical stages and
        # endpoint may contribute to the actual step's lifecycle accumulator.
        assert len(initial.evaluations)-len(self.evaluations)<=1
        self.evaluations=initial.evaluations.copy();self.finalized=initial.finalized+1
        return self


@pytest.mark.parametrize('method',METHODS)
def test_coupled_quantity_quadrature_lifecycle_probe_isolation_and_endpoint(method):
    start=RosenSystem(system().state);start.state.u=torch.ones_like(start.state.u);start.state.e=torch.ones_like(start.state.e)
    def rhs(current,dt,**kwargs):
        s=current.state;target=math.exp(current.t)
        return testing.ParticleUpdate(s.u,-s.x+.1*s.e*s.e+2*target-.1*target*target,
            .3*s.e+.1*s.x*s.u+.7*target-.1*target*target),None
    result=solve(Rosenbrock(method),start,.25,rhs,dt=.0625,solver_opts=OPTIONS)
    assert result.successful and result.state.finalized==result.state.initialized==4
    assert len(result.state.evaluations)==4*Rosenbrock(method).metadata.stage_count
    assert start.evaluations==[] and start.initialized==start.finalized==0
    assert torch.equal(result.state.state.density,result.state.state.x.square()+result.state.state.e)
    assert max(float((getattr(result.state.state,key)-math.exp(.25)).abs().max()) for key in ['x','u','e'])<1e-3


def test_failed_linear_solve_and_nonfinite_forces_cannot_finalize_or_commit():
    start=RosenSystem(system().state)
    def rhs(current,dt,**kwargs):
        s=current.state;return testing.ParticleUpdate(s.u,-torch.tensor([1.,10.],dtype=s.x.dtype)*s.x,.2*s.x*s.u),None
    failed=solve(Rosenbrock(),start,.25,rhs,dt=.125,solver_opts=dict(gmres_tol=1e-14,gmres_maxiter=1),raise_on_failure=False)
    assert not failed.successful and failed.accepted_steps==0 and failed.state.finalized==0
    assert start.evaluations==[]
    def broken(t,y):return torch.full_like(y,float('nan'))
    failed=solve(Rosenbrock(),TensorState(torch.ones(1,dtype=torch.float64)),.25,tensor_rhs(broken),dt=.125,raise_on_failure=False)
    assert not failed.successful and failed.accepted_steps==0


def test_switch_composition_and_nonembedded_adaptive_refusal():
    from test_operator_splitting import error
    method=Strang(Rosenbrock('Rosenbrock32'),'RK4')
    errors=[error(method,h) for h in [.1,.05,.025]]
    assert math.log(errors[-2]/errors[-1],2)==pytest.approx(2,abs=.1)
    switch=AlgorithmSwitch((Rosenbrock(),'dormandPrince'),lambda s,p:0 if s.t<.1 else 1,'rosen')
    result=solve(switch,TensorState(torch.ones(1,dtype=torch.float64)),.25,tensor_rhs(lambda t,y:-y),dt=.0625,solver_opts=OPTIONS)
    assert result.successful
    with pytest.raises(ValueError,match='adaptive|estimate|Adaptive'):
        solve(Rosenbrock('RosenbrockW6S4OS'),TensorState(torch.ones(1,dtype=torch.float64)),1.,tensor_rhs(lambda t,y:-y),dt=.1,adaptive=True)


@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
def test_dtype_and_options_match_layout(dtype):
    result=solve(Rosenbrock(),TensorState(torch.ones(5,dtype=dtype)),.25,tensor_rhs(lambda t,y:-y),dt=.0625,
        solver_opts=dict(gmres_tol=1e-5 if dtype==torch.float32 else 1e-13))
    assert result.successful and result.state.y.dtype==dtype
    assert torch.allclose(result.state.y,torch.full_like(result.state.y,math.exp(-.25)),atol=3e-4,rtol=0)


@pytest.mark.parametrize('options',[dict(solver_opts={'gmres_maxiter':0}),dict(solver_opts={'gmres_tol':-1}),
    dict(ft_eps=0.),dict(f_t='none')])
def test_invalid_controls(options):
    with pytest.raises((ValueError,TypeError)):
        Rosenbrock()(TensorState(torch.ones(1)),.1,tensor_rhs(lambda t,y:-y),**options)
