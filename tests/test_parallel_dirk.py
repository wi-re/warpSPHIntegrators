"""Parallel DIRK's serial map: source, exact order/stability and full-state gates."""
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
import torch

from warpSPHIntegrators.schemes import ParallelDIRK
from warpSPHIntegrators.schemes.coefficients.parallel_dirk import parallel_dirk_tableau
from warpSPHIntegrators import TensorState, tensor_rhs, solve
from warpSPHIntegrators.runtime import save_checkpoint, load_checkpoint
import warpSPHIntegrators.testing as testing
from warpSPHIntegrators.wrappers import AlgorithmSwitch, Strang
from warpSPHIntegrators.analysis.stability import rk_stability_function
from warpSPHIntegrators.solvers import SolveResult, SolveDiagnostics
from test_nystrom import System, system

SOURCE=json.loads((Path(__file__).parents[1]/'scripts/coefficients/wave12/pdirk.json').read_text())
OPTIONS=dict(matvec='jvp',tol=1e-13,newton_tol=1e-8)


@pytest.mark.parametrize('record',SOURCE['records'])
def test_serial_map_matches_both_pinned_source_execution_branches(record):
    forced=record['forced']
    def rhs(t,y):
        return torch.stack((-2*y[0]+.1*y[0]*y[1]+(.2*math.sin(t) if forced else 0),
                            -3*y[1]+.2*y[0]**2+(.1*math.cos(t) if forced else 0)))
    start=TensorState(torch.tensor(record['initial'],dtype=torch.float64),t=record['t'])
    result=ParallelDIRK()(start,record['h'],tensor_rhs(rhs),solver_opts=OPTIONS)
    assert result.step_status.successful
    assert result.state.y.tolist()==pytest.approx(record['output'],rel=0,abs=2e-13)
    assert len(result.stages)==5 and len(result.solver_diagnostics)==4
    assert result.quadrature.tolist()==[-1.,1.5,-1.,1.5,0.]


def test_source_coefficients_independent_groups_and_all_order_four_conditions():
    tab=parallel_dirk_tableau();a,b,c=tab.a,tab.b,tab.c
    assert a.tolist()==SOURCE['A'] and b.tolist()==SOURCE['b'] and c.tolist()==SOURCE['c']
    assert np.allclose(a.sum(axis=1),c)
    assert ParallelDIRK().stage_groups==((0,1),(2,3))
    actual=[sum(b),b@c,b@(c*c),b@a@c,b@(c**3),b@(c*(a@c)),b@a@(c*c),b@a@a@c]
    assert actual==pytest.approx([1.,1/2,1/3,1/6,1/4,1/8,1/12,1/24],rel=0,abs=3e-16)


def test_exact_rational_a_and_l_stability_proof():
    import sympy as s
    z,x=s.symbols('z x',real=True)
    a=s.Matrix([[s.Rational(1,2),0,0,0],[0,s.Rational(2,3),0,0],
        [s.Rational(-5,2),s.Rational(5,2),s.Rational(1,2),0],
        [s.Rational(-5,3),s.Rational(4,3),0,s.Rational(2,3)]])
    b=s.Matrix([-1,s.Rational(3,2),-1,s.Rational(3,2)])
    r=s.factor(1+z*(b.T*(s.eye(4)-z*a).inv()*s.ones(4,1))[0])
    expected=(z+3)*(9*z*z-20*z+12)/((z-2)**2*(2*z-3)**2)
    assert s.simplify(r-expected)==0 and s.limit(r,z,s.oo)==0
    numerator,denominator=r.as_numer_denom()
    defect=s.expand(denominator.subs(z,s.I*x)*denominator.subs(z,-s.I*x)
        -numerator.subs(z,s.I*x)*numerator.subs(z,-s.I*x))
    assert s.factor(defect)==x**6*(16*x*x+119)
    assert set(s.solve(denominator,z))=={s.Rational(3,2),2}  # no poles in closed left half-plane
    for point in [-1.,-10000.,-.2+2j,4j]:
        assert rk_stability_function(parallel_dirk_tableau(),point)==pytest.approx(complex(expected.subs(z,point)))


def test_nonlinear_nonautonomous_order_four():
    errors=[]
    for h in [.1,.05,.025]:
        result=solve(ParallelDIRK(),TensorState(torch.ones(1,dtype=torch.float64)),1.,
            tensor_rhs(lambda t,y:-y+.2*y*y-.2*math.exp(-2*t)),dt=h,solver_opts=OPTIONS)
        assert result.successful
        errors.append(abs(result.state.y.item()-math.exp(-1)))
    assert math.log(errors[-2]/errors[-1],2)==pytest.approx(4,abs=.15),(errors,)


def test_state_and_rhs_parameter_gradients_match_finite_differences():
    def forward(q,k,tracking=False):
        y=torch.tensor([q],dtype=torch.float64,requires_grad=tracking)
        parameter=torch.tensor(k,dtype=torch.float64,requires_grad=tracking)
        result=solve(ParallelDIRK(),TensorState(y),.25,
            tensor_rhs(lambda t,y:-parameter*y-.1*y**2+.2*t),dt=.0625,solver_opts=OPTIONS)
        assert result.successful
        return result.state.y.sum(),y,parameter
    loss,y,k=forward(.7,2.,True);dy,dk=torch.autograd.grad(loss,(y,k));eps=1e-5
    assert dy.item()==pytest.approx(((forward(.7+eps,2.)[0]-forward(.7-eps,2.)[0])/(2*eps)).item(),rel=2e-6)
    assert dk.item()==pytest.approx(((forward(.7,2.+eps)[0]-forward(.7,2.-eps)[0])/(2*eps)).item(),rel=2e-6)


def test_stiff_decay_and_portable_continuation(tmp_path):
    method=ParallelDIRK();rhs=tensor_rhs(lambda t,y:-100*y)
    first=solve(method,TensorState(torch.ones(1,dtype=torch.float64)),.25,rhs,dt=.0625,solver_opts=OPTIONS)
    path=tmp_path/'pdirk.pt';save_checkpoint(first.checkpoint,path)
    resumed=solve(method,None,.5,rhs,restart=load_checkpoint(path),solver_opts=OPTIONS)
    direct=solve(method,TensorState(torch.ones(1,dtype=torch.float64)),.5,rhs,dt=.0625,solver_opts=OPTIONS)
    assert resumed.successful and torch.equal(resumed.state.y,direct.state.y)
    assert abs(resumed.state.y.item())<1e-7
    with pytest.raises(ValueError,match='estimate|adaptive|Adaptive'):
        solve(method,TensorState(torch.ones(1,dtype=torch.float64)),1.,rhs,dt=.1,adaptive=True)


@dataclass
class RKSystem(System):
    def initializeNewState(self,*args,**kwargs):
        return RKSystem(self.state.initializeNewState(),self.t,self.initialized,self.finalized)
    def finalize(self,initial,dt,auxes,updates,weights,**kwargs):
        assert len(auxes)==len(updates)==len(weights)==5
        self.finalized=initial.finalized+1
        assert torch.allclose(self.state.e,initial.state.e+dt*sum(w*k.dedt for w,k in zip(weights,updates)))
        return self


def test_quantity_lifecycle_endpoint_copy_and_failed_trial_ownership():
    start=RKSystem(system().state)
    def rhs(current,dt,**kwargs):
        s=current.state
        return testing.ParticleUpdate(s.u,-s.x+.1*s.e,s.e+.2*s.x*s.u),None
    result=solve(ParallelDIRK(),start,.25,rhs,dt=.0625,solver_opts=OPTIONS)
    assert result.successful and result.state.finalized==result.state.initialized==4
    assert start.finalized==start.initialized==0
    assert torch.equal(start.state.density,torch.zeros_like(start.state.density))
    assert torch.equal(result.state.state.density,result.state.state.x.square()+result.state.state.e)
    class Failed:
        def solve(self,step,guess,norm,**kwargs):
            return SolveResult(guess,False,0,SolveDiagnostics(converged=False,termination='max_iterations'))
    failed=solve(ParallelDIRK(),start,.25,rhs,dt=.0625,solver=Failed(),raise_on_failure=False)
    assert not failed.successful and failed.accepted_steps==0 and failed.state.finalized==0


def test_switch_and_physical_composition_share_the_one_step_contract():
    from test_operator_splitting import damped_split, error
    switched=solve(AlgorithmSwitch((ParallelDIRK(),'RK4'),lambda s,p:0 if s.t<.1 else 1,'pdirk'),
        TensorState(torch.ones(1,dtype=torch.float64)),.25,tensor_rhs(lambda t,y:-y),dt=.0625,solver_opts=OPTIONS)
    assert switched.successful
    method=Strang(ParallelDIRK(),'RK4')
    errors=[error(method,h) for h in [.1,.05,.025]]
    assert math.log(errors[-2]/errors[-1],2)==pytest.approx(2,abs=.1)


@pytest.mark.parametrize('dt',[0.,float('nan'),float('inf'),True])
def test_invalid_step_is_rejected_before_lifecycle(dt):
    with pytest.raises(ValueError):ParallelDIRK()(TensorState(torch.ones(1)),dt,tensor_rhs(lambda t,y:-y))
