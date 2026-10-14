"""Pinned NDF warm kernel, variable history, local estimates and safe continuation."""
import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch
from dataclasses import dataclass

from warpSPHIntegrators import (NDF,ndf_difference_transform,TensorState,tensor_rhs,StepHistory,
    HistoryEntry,solve,save_checkpoint,load_checkpoint)
from warpSPHIntegrators import (NDFQuadrature,BaseState,integrated,copied,constant,reference_state,
    testing,JFNKSolver,SolveResult)

SOURCE=json.loads((Path(__file__).parents[1]/'scripts/coefficients/wave10/ndf.json').read_text())
OPTIONS=dict(matvec='jvp',tol=1e-15,newton_tol=1e-10)


@pytest.mark.parametrize('key',list(SOURCE['records']))
def test_warm_kernel_transform_error_and_history_match_pinned_source(key):
    r=SOURCE['records'][key];p=r['order'];state=TensorState(torch.tensor(r['initial'],dtype=torch.float64),t=r['t'])
    differences=torch.tensor(r['input_differences'][:p+2],dtype=torch.float64)
    history=StepHistory(p,[HistoryEntry(r['t']-r['previous_h'],r['previous_h'],None)],
        context=dict(method='ndf.v1',order=p,layout=(('y',(2,),'torch.float64'),),uid=None,
            phase='warm',differences=differences,current=state.y.detach().clone(),constant_steps=p+1))
    forced=key.endswith('forced')
    def force(t,y):return torch.stack((-2*y[0]+.1*y[1]**2+(.2*math.sin(t) if forced else 0),
        -.5*y[1]+.2*y[0]**2+(.1*math.cos(t) if forced else 0)))
    result=NDF(p)(state,r['h'],tensor_rhs(force),history=history,solver_opts=OPTIONS)
    assert result.step_status.successful
    assert np.allclose(ndf_difference_transform(p,r['ratio']),r['difference_transform'],atol=1e-14)
    assert result.state.y.tolist()==pytest.approx(r['state'],rel=0,abs=2e-13)
    assert result.error.y.tolist()==pytest.approx(r['error'],rel=0,abs=2e-13)
    assert torch.allclose(result.history.context['differences'],torch.tensor(r['output_differences'],dtype=torch.float64),atol=2e-13,rtol=0)
    assert torch.equal(history.context['differences'],differences)
    assert NDF(p).beta==pytest.approx(r['beta']) and NDF(p).error_constant==pytest.approx(r['error_constant'])


def force(t,y):return -y+.2*y*y-.2*math.exp(-2*t)


@pytest.mark.parametrize('order',[1,2,3,4,5])
@pytest.mark.parametrize('alternating',[False,True])
def test_nonlinear_nonautonomous_order_and_variable_grid_memory(order,alternating):
    errors=[]
    for count in ([2*(order+1),4*(order+1),8*(order+1)] if alternating else [20,40,80]):
        method=NDF(order);state=TensorState(torch.ones(1,dtype=torch.float64));history=None
        for i in range(count):
            h=(.75 if (i//(order+1))%2==0 else 1.25)/count if alternating else 1/count
            result=method(state,h,tensor_rhs(force),history=history,solver_opts=OPTIONS)
            assert result.step_status.successful
            state,history=result.state,result.history
        assert history.context['phase']=='warm' and len(history)==1
        errors.append(abs(state.y.item()-math.exp(-1)))
    measured=math.log(errors[-2]/errors[-1],2)
    assert measured>=order-.25,(order,alternating,errors,measured)


@pytest.mark.parametrize('order',[1,2,3,4,5])
def test_adaptive_refinement_rejection_and_portable_restart(order,tmp_path):
    method=NDF(order);errors=[];rejections=[]
    for tolerance in [1e-3,1e-5]:
        run=solve(method,TensorState(torch.ones(1,dtype=torch.float64)),2.,tensor_rhs(force),dt=2.,
            adaptive=True,rtol=tolerance,atol=tolerance*.01,solver_opts=OPTIONS)
        assert run.successful
        rejections.append(run.rejected_steps)
        errors.append(abs(run.state.y.item()-math.exp(-2)))
    assert rejections[-1]>0
    assert errors[-1]<errors[0]/3,(order,errors)
    first=solve(method,TensorState(torch.ones(1,dtype=torch.float64)),.3,tensor_rhs(force),dt=.025,solver_opts=OPTIONS)
    path=tmp_path/'ndf.pt';save_checkpoint(first.checkpoint,path)
    a=solve(method,None,.6,tensor_rhs(force),restart=load_checkpoint(path),solver_opts=OPTIONS)
    b=solve(method,None,.6,tensor_rhs(force),restart=first.checkpoint,solver_opts=OPTIONS)
    assert torch.equal(a.state.y,b.state.y)


@pytest.mark.parametrize('order',[1,2,3,4,5])
def test_bootstrap_exponents_work_accounting_and_exact_zero_rate_fields(order):
    method=NDF(order);state=TensorState(torch.tensor([1e12,-1e12],dtype=torch.float64));history=None
    exponents=[]
    for _ in range(order+2):
        calls=[]
        def zero(t,y):calls.append(t);return y*0
        result=method(state,.1,tensor_rhs(zero),history=history)
        diagnostics=result.solver_diagnostics
        diagnostics=diagnostics if isinstance(diagnostics,list) else [diagnostics]
        assert len(calls)==sum(d.rhs_evaluations+d.gmres_iterations for d in diagnostics if d is not None)
        assert torch.equal(result.state.y,state.y)
        exponents.append(result.estimate_order);state,history=result.state,result.history
    assert exponents[:order]==[{1:2,2:3,3:3,4:4,5:5}[order]]*order
    assert exponents[-2:]==[order+1]*2


@pytest.mark.parametrize('order',[2,5])
def test_parameter_gradient_survives_restart_and_history_is_owned(order):
    def forward(value,tracking=False):
        y=torch.ones(1,dtype=torch.float64,requires_grad=tracking)
        k=torch.tensor(value,dtype=torch.float64,requires_grad=tracking)
        rhs=tensor_rhs(lambda t,y:-k*y-.05*y**3);method=NDF(order)
        first=solve(method,TensorState(y),.3,rhs,dt=.025,solver_opts=OPTIONS)
        before=first.history.context['differences'].clone()
        result=solve(method,None,.6,rhs,restart=first.checkpoint,solver_opts=OPTIONS)
        assert torch.equal(before,first.history.context['differences'])
        return result.state.y.sum(),y,k
    output,y,k=forward(2.,True);dy,dk=torch.autograd.grad(output,(y,k));eps=1e-5
    finite=(forward(2+eps)[0]-forward(2-eps)[0])/(2*eps)
    assert dy.item()>0 and dk.item()==pytest.approx(finite.item(),rel=2e-6,abs=1e-9)


def test_fast_alternation_counterexample_is_guarded_by_bootstrap():
    p=5;harmonic=np.cumsum(1/np.arange(1,p+1))
    advance=np.triu(np.ones((p,p)))-NDF(p).beta*np.ones((p,1))*harmonic
    up=np.asarray(ndf_difference_transform(p,5/3)).T;down=np.asarray(ndf_difference_transform(p,3/5)).T
    assert max(abs(np.linalg.eigvals(advance@up@advance@down)))>1.14
    assert max(abs(np.linalg.eigvals(np.linalg.matrix_power(advance,6)@up@np.linalg.matrix_power(advance,6)@down)))<1
    state=TensorState(torch.ones(1,dtype=torch.float64));history=None
    for i in range(40):
        result=NDF(5)(state,(.75 if i%2==0 else 1.25)/40,tensor_rhs(force),history=history,solver_opts=OPTIONS)
        assert result.quadrature.bootstrap  # rapid caller changes use the L-stable one-step path
        state,history=result.state,result.history
    assert abs(state.y.item()-math.exp(-1))<1e-8


def test_adaptive_controller_holds_blocks_and_uses_the_implicit_backbone():
    records=[]
    result=solve(NDF(5),TensorState(torch.ones(1,dtype=torch.float64)),4.,tensor_rhs(force),dt=.2,
        adaptive=True,rtol=1e-6,atol=1e-9,solver_opts=OPTIONS,callback=records.append)
    assert result.successful and sum(not r.result.quadrature.bootstrap for r in records)>=5
    assert records[0].dt==records[4].dt


@pytest.mark.parametrize('order',[1,2,3,4,5])
def test_warm_defect_exponent_on_exact_polynomial_history(order):
    errors=[]
    for h in [.1,.05,.025]:
        times=[-i*h for i in range(order+3)]
        values=[torch.tensor([t**(order+1)/(order+1)],dtype=torch.float64) for t in times]
        d=[]
        for _ in range(order+2):
            values=[a-b for a,b in zip(values,values[1:])];d.append(values[0])
        state=TensorState(torch.zeros(1,dtype=torch.float64))
        history=StepHistory(order,[HistoryEntry(-h,h,None)],context=dict(method='ndf.v1',order=order,
            layout=(('y',(1,),'torch.float64'),),uid=None,phase='warm',constant_steps=order+1,
            differences=torch.stack(d),current=state.y.clone()))
        result=NDF(order)(state,h,tensor_rhs(lambda t,y:torch.full_like(y,t**order)),history=history,solver_opts=OPTIONS)
        errors.append(abs(result.error.y.item()))
    assert math.log(errors[-2]/errors[-1],2)==pytest.approx(order+1,rel=0,abs=.02)


def test_fifth_order_constant_step_main_weights_are_bdf5():
    from warpSPHIntegrators.bdf import getBDFCoefficients
    p=5;method=NDF(p);h=np.cumsum(1/np.arange(1,p+1))
    weights=[float(i==0)+sum((1-method.beta*h[j-1])*(-1)**i*math.comb(j,i)
        for j in range(max(1,i),p+1)) for i in range(p+1)]
    reference,beta=getBDFCoefficients(5)
    assert weights[:-1]==pytest.approx(reference,rel=0,abs=1e-14)
    assert weights[-1]==pytest.approx(0,rel=0,abs=1e-14) and method.beta==pytest.approx(beta)


@pytest.mark.parametrize('order',[1,2,3,4,5])
def test_fixed_step_characteristic_roots_include_negative_real_decay(order):
    from warpSPHIntegrators.bdf import getBDFCoefficients
    weights,beta=getBDFCoefficients(order)
    rho=np.r_[1/beta,-np.asarray(weights)/beta,0.]
    harmonic=sum(1/j for j in range(1,order+1))
    rho=rho-NDF(order).kappa*harmonic*np.asarray(np.poly1d([1,-1])**(order+1))
    for z in -np.logspace(-6,6,61):
        coefficients=rho.astype(complex);coefficients[0]-=z
        assert max(abs(np.roots(coefficients)))<1+1e-10
    theta=np.linspace(0,math.pi,301);r=np.exp(1j*theta)
    boundary=np.polyval(rho,r)/r**(order+1)
    if order==1:
        c=np.cos(theta)
        assert boundary.real==pytest.approx((1-c)*(1-.37*c),rel=0,abs=3e-14)
    elif order==2:
        assert boundary.real==pytest.approx(2/3*(1-np.cos(theta))**3,rel=0,abs=3e-14)
    else:assert boundary.real.min()<-1e-5  # no full A-stability at orders 3–5


def test_method_step_policy_validation_precedes_external_commit():
    from test_run import Transaction
    class InvalidPolicy(NDF):
        def next_step_size(self,result,h,proposal):return float('nan')
    tx=Transaction();state=TensorState(torch.ones(1,dtype=torch.float64))
    with pytest.raises(ValueError,match='next_step_size'):
        solve(InvalidPolicy(1),state,.1,tensor_rhs(lambda t,y:-y),dt=.01,
            adaptive=True,transaction=tx,solver_opts=OPTIONS)
    assert tx.commits==0 and tx.rollbacks==1
    assert state.y.item()==1 and state.t==0


def test_preconditioner_receives_protected_warm_shift_and_solver_default_context():
    rates=torch.logspace(0,3,32,dtype=torch.float64);seen=[]
    def inverse(v,state,context):
        assert context['sentinel']=='preserved'
        seen.append(context['stage_coefficient'])
        return v/(1+context['stage_coefficient']*rates)
    solver=JFNKSolver(matvec='jvp',tol=1e-15,newton_tol=1e-8,preconditioner=inverse,
        preconditioner_context={'sentinel':'preserved','stage_coefficient':999})
    method=NDF(3);state=TensorState(torch.ones_like(rates));history=None
    for _ in range(7):
        result=method(state,.02,tensor_rhs(lambda t,y:-rates*y),history=history,solver=solver)
        assert result.step_status.successful
        state,history=result.state,result.history
    assert seen[-1]==pytest.approx(.02*method.beta)


@dataclass
class Fields(BaseState):
    x: torch.Tensor=integrated('dxdt',tags=('position',))
    u: torch.Tensor=integrated('dudt',tags=('velocity',))
    e: torch.Tensor=integrated('dedt',tags=('quantity',))
    uid: torch.Tensor=constant()
    density: torch.Tensor=copied()


@dataclass
class System(testing.ParticleSystem):
    state: Fields=reference_state()
    ndf_compatible=True
    initialized: int=0
    finalized: int=0
    def initializeNewState(self,*args,**kwargs):return System(self.state.initializeNewState(),self.t,self.initialized,self.finalized)
    def initialize(self,dt,**kwargs):self.initialized+=1;return self
    def preprocess(self,initial,dt,**kwargs):self.state.density=self.state.x.square()+self.state.e;return self
    def finalize(self,initial,dt,auxes,updates,quadrature,**kwargs):
        assert isinstance(quadrature,NDFQuadrature)
        assert len(auxes)==len(updates)==len(quadrature.weights)
        self.finalized=initial.finalized+1
        return self


def system():
    return System(Fields(torch.ones(2,dtype=torch.float64),torch.zeros(2,dtype=torch.float64),
        torch.zeros(2,dtype=torch.float64),torch.tensor([1,2]),torch.zeros(2,dtype=torch.float64)))


def mechanical(state,dt,**kwargs):
    s=state.state
    return testing.ParticleUpdate(s.u,-s.x+.1*s.e,s.e+.1*s.x*s.u),None


def test_coupled_quantities_owned_lifecycle_and_copied_endpoint_fields():
    start=system();result=solve(NDF(3),start,.4,mechanical,dt=.025,solver_opts=OPTIONS)
    assert result.successful and result.state.initialized==result.state.finalized==16
    assert start.initialized==start.finalized==0
    assert torch.equal(result.state.state.density,result.state.state.x.square()+result.state.state.e)
    assert result.state.state.e.abs().max()>1e-4
    assert torch.equal(start.state.density,torch.zeros(2,dtype=torch.float64))


def test_invalid_histories_state_edits_ratio_and_uid_changes_restart_before_use():
    method=NDF(2);rhs=tensor_rhs(force)
    first=solve(method,TensorState(torch.ones(1,dtype=torch.float64)),.3,rhs,dt=.025,solver_opts=OPTIONS)
    for mode in ['state','ratio','explicit','foreign','nonfinite']:
        from warpSPHIntegrators import snapshot_state
        current=snapshot_state(first.state);history=snapshot_state(first.history);h=.025;options={}
        if mode=='state':current.state.y=current.y+1
        if mode=='ratio':h=.2
        if mode=='explicit':options['restart_history']=True
        if mode=='foreign':history.context['method']='other'
        if mode=='nonfinite':history.context['differences'][0]=float('nan')
        changed=method(current,h,rhs,history=history,solver_opts=OPTIONS,**options)
        cold=method(current,h,rhs,solver_opts=OPTIONS)
        assert changed.quadrature.bootstrap and torch.equal(changed.state.y,cold.state.y)
    rich=solve(method,system(),.2,mechanical,dt=.025,solver_opts=OPTIONS)
    rich.state.state.uid=rich.state.state.uid.flip(0)
    reset=method(rich.state,.025,mechanical,history=rich.history,solver_opts=OPTIONS)
    assert reset.quadrature.bootstrap


def test_failed_trial_and_warm_initialize_or_finalize_cannot_corrupt_memory():
    from test_run import FailedSolver
    method=NDF(2);first=solve(method,system(),.2,mechanical,dt=.025,solver_opts=OPTIONS)
    before=first.history.context['differences'].clone()
    failed=method(first.state,.025,mechanical,history=first.history,solver=FailedSolver())
    assert not failed.step_status.successful and torch.equal(failed.history.context['differences'],before)
    @dataclass
    class Modifying(System):
        def initializeNewState(self,*args,**kwargs):return Modifying(self.state.initializeNewState(),self.t,self.initialized,self.finalized)
        def initialize(self,dt,**kwargs):self.state.x=self.state.x+1;return self
    with pytest.raises(ValueError,match='Warm NDF initialize'):
        method(Modifying(first.state.state,first.state.t),.025,mechanical,history=first.history,solver_opts=OPTIONS)
    @dataclass
    class Projecting(System):
        def initializeNewState(self,*args,**kwargs):return Projecting(self.state.initializeNewState(),self.t)
        def finalize(self,initial,dt,*args,**kwargs):self.state.e=self.state.e+1;return self
    with pytest.raises(ValueError,match='finalize must preserve'):
        method(Projecting(system().state),.025,mechanical,solver_opts=OPTIONS)
    assert torch.equal(first.history.context['differences'],before)


@pytest.mark.parametrize('options',[dict(order=True),dict(order=0),dict(order=6),
    dict(max_step_ratio=.5),dict(max_step_ratio=3),dict(max_step_ratio=True)])
def test_invalid_ndf_configuration(options):
    with pytest.raises(ValueError):NDF(**options)
