"""Pinned source maps, complete RK tree conditions and owned full-stage execution."""
from dataclasses import dataclass
from functools import lru_cache
import importlib.util
import json
import math
from pathlib import Path
import numpy as np
import pytest
import torch
from warpSPHIntegrators.schemes import ExpandedRK
from warpSPHIntegrators.schemes.coefficients.expanded import expanded_rk_methods, expanded_rk_tableau
from warpSPHIntegrators import TensorState, tensor_rhs, solve
from warpSPHIntegrators.wrappers import Strang, ConservativeDissipativeRHS, AlgorithmSwitch
from warpSPHIntegrators.runtime import save_checkpoint, load_checkpoint

ROOT=Path(__file__).parents[1]
SOURCE=json.loads((ROOT/'scripts/coefficients/wave16/oracle.json').read_text())
SCREEN=json.loads((ROOT/'scripts/coefficients/wave16/screen.json').read_text())['screen']
spec=importlib.util.spec_from_file_location('expanded_generator',ROOT/'scripts/coefficients/wave16/generate.py')
generator=importlib.util.module_from_spec(spec);spec.loader.exec_module(generator)
METHODS=expanded_rk_methods()
PAIRS=tuple(name for name in METHODS if ExpandedRK(name).has_error_estimate)


@pytest.mark.parametrize('name',METHODS)
@pytest.mark.parametrize('forced',[False,True])
def test_main_error_and_endpoint_match_pinned_source_kernels(name,forced):
    row=SOURCE['records'][name+('_forced' if forced else '_autonomous')];calls=[]
    def force(t,y):
        calls.append(t)
        return torch.stack((-2*y[0]+.1*y[1]**2+(.2*math.sin(t) if forced else 0),
            -.5*y[1]+.2*y[0]**2+(.1*math.cos(t) if forced else 0)))
    initial=TensorState(torch.tensor(row['initial'],dtype=torch.float64),t=row['t'])
    method=ExpandedRK(name);result=method(initial,row['h'],tensor_rhs(force))
    assert result.state.y.tolist()==pytest.approx(row['state'],rel=0,abs=3e-13)
    if method.has_error_estimate:assert result.error.y.tolist()==pytest.approx(row['error'],rel=0,abs=3e-13)
    else:assert result.error is None
    assert len(calls)==method.metadata.stage_count and calls[-1]==result.state.t
    assert len(result.stages)==len(calls) and result.state.t==row['t']+row['h']
    assert initial.y.tolist()==row['initial'] and initial.t==row['t']


@pytest.mark.parametrize('name',METHODS)
def test_full_rooted_tree_order_conditions_and_immutable_coefficients(name):
    method=ExpandedRK(name);tab=method.tableau;a,b,c=tab
    main=b[0] if isinstance(b,tuple) else b
    assert max(generator.order_defects(a,main,method.order))<1e-10
    assert max(abs(a.sum(1)-c))<1e-10 and np.allclose(a,np.tril(a,-1),atol=0,rtol=0)
    assert np.array_equal(a[-1],main) and main[-1]==0 and c[-1]==1
    if method.has_error_estimate:
        assert max(generator.order_defects(a,b[1],method.embedded_order))<1e-10
        assert method.estimate_order==method.order
    for array in (a,c,*(b if isinstance(b,tuple) else (b,))):
        with pytest.raises(ValueError):array.setflags(write=True)
    with pytest.raises(AttributeError):tab.a=np.zeros_like(a)


@pytest.mark.parametrize('name',METHODS)
def test_accepted_trajectory_matches_source_and_zero_rates_preserve_offsets(name):
    result=solve(ExpandedRK(name),TensorState(torch.ones(1,dtype=torch.float64)),1.,
        tensor_rhs(lambda t,y:-y),dt=.025)
    assert result.successful
    assert result.state.y.item()==pytest.approx(SOURCE['trajectories'][name]['decay']['40'],rel=0,abs=3e-12)
    initial=TensorState(torch.tensor([1e12,-1e12],dtype=torch.float64))
    result=ExpandedRK(name)(initial,.1,tensor_rhs(lambda t,y:y*0))
    assert torch.equal(result.state.y,initial.y)
    if result.error is not None:assert torch.equal(result.error.y,torch.zeros_like(initial.y))


@pytest.mark.parametrize('name',PAIRS)
def test_actual_estimator_exponent_on_a_known_polynomial(name):
    method=ExpandedRK(name);q=method.estimate_order;values=[]
    for h in (.2,.1,.05):
        result=method(TensorState(torch.zeros(1,dtype=torch.float64)),h,
            tensor_rhs(lambda t,y:torch.full_like(y,t**(q-1))))
        values.append(abs(result.error.y.item()))
    assert math.log(values[-2]/values[-1],2)==pytest.approx(q,rel=0,abs=1e-6)


@pytest.mark.parametrize('name',['CarpenterKennedy2N54','ParsaniKetchesonDeconinck3S205','AlJahdaliVor3S165'])
def test_nonlinear_nonautonomous_order_gradients_and_backward_time(name):
    method=ExpandedRK(name);errors=[]
    for count in (2,4,8):
        result=solve(method,TensorState(torch.ones(1,dtype=torch.float64)),1.,
            tensor_rhs(lambda t,y:-2*y+.2*y*y-.2*math.exp(-4*t)),dt=1/count)
        errors.append(abs(result.state.y.item()-math.exp(-2)))
    assert math.log(errors[-2]/errors[-1],2)>method.order-.4,(name,errors)
    parameter=torch.tensor(2.,dtype=torch.float64,requires_grad=True)
    result=solve(method,TensorState(torch.ones(1,dtype=torch.float64)),.4,
        tensor_rhs(lambda t,y:-parameter*y),dt=.025)
    gradient=torch.autograd.grad(result.state.y.sum(),parameter)[0]
    assert gradient.item()==pytest.approx(-.4*math.exp(-.8),rel=2e-6)
    backward=solve(method,result.state,0.,tensor_rhs(lambda t,y:-2*y),dt=.025)
    assert backward.state.y.item()==pytest.approx(1.,rel=0,abs=2e-6)


@pytest.mark.parametrize('name',['RDPK3Sp35','CKLLSRK95_4S'])
def test_adaptive_rejection_portable_restart_composition_and_switching(name,tmp_path):
    method=ExpandedRK(name);rhs=tensor_rhs(lambda t,y:-y);errors=[]
    for tol in (1e-4,1e-7):
        result=solve(method,TensorState(torch.ones(1,dtype=torch.float64)),2.,rhs,dt=2.,adaptive=True,rtol=tol,atol=tol*.01)
        assert result.successful and result.rejected_steps>0
        errors.append(abs(result.state.y.item()-math.exp(-2)))
    assert errors[-1]<errors[0]/10
    first=solve(method,TensorState(torch.ones(1,dtype=torch.float64)),.3,rhs,dt=.025)
    path=tmp_path/'expanded.pt';save_checkpoint(first.checkpoint,path)
    a=solve(method,None,.6,rhs,restart=load_checkpoint(path));b=solve(method,None,.6,rhs,restart=first.checkpoint)
    assert torch.equal(a.state.y,b.state.y)
    physical=tensor_rhs(ConservativeDissipativeRHS(lambda t,y:y,lambda t,y:-2*y))
    composed=solve(Strang(method,'RK4'),TensorState(torch.ones(1,dtype=torch.float64)),.4,physical,dt=.025)
    assert composed.state.y.item()==pytest.approx(math.exp(-.4),rel=0,abs=2e-6)
    switched=solve(AlgorithmSwitch((method,'dormandPrince'),lambda s,p:0 if s.t<.2 else 1,'expanded-v1'),
        TensorState(torch.ones(1,dtype=torch.float64)),.4,rhs,dt=.025)
    assert switched.state.y.item()==pytest.approx(math.exp(-.4),rel=0,abs=2e-6)


def test_source_limitations_are_numerically_screened_and_not_exposed_as_qualified():
    assert len(METHODS)==54 and len(PAIRS)==19 and len(SOURCE['methods'])==61 and not SOURCE['failures']
    assert [len(generator.trees(p)) for p in range(1,6)]==[1,1,2,4,9]
    for name,record in SCREEN.items():
        assert record['qualified']==(name in METHODS)
        if not record['qualified']:
            with pytest.raises(ValueError,match='qualified'):ExpandedRK(name)
    assert not ExpandedRK('CKLLSRK85_4FM_4R').has_error_estimate
    assert SCREEN['CKLLSRK85_4FM_4R']['observed_comparison_order']==1
    for name,problem,exact in [('CKLLSRK54_3C','time_only',.5),('CKLLSRK95_4C','decay',math.exp(-1))]:
        row=SOURCE['trajectories'][name][problem];a,b=abs(row['40']-exact),abs(row['80']-exact)
        assert math.log(a/b,2)==pytest.approx(1.,rel=0,abs=.02)
    for name in ('SHLDDRK52','SHLDDRK64','SHLDDRK_2N_phase1','SHLDDRK_2N_phase2'):
        assert SCREEN[name]['main_order_defects'][0]>5e-8


def test_generated_package_and_screen_records_are_current():
    module,screen=generator.generate()
    assert generator.MODULE.read_text()==module and generator.SCREEN.read_text()==screen


def test_full_state_lifecycle_stage_weights_stage_indices_and_rerun_protocol():
    from test_ndf import System,system
    import warpSPHIntegrators.testing as testing
    @dataclass
    class Consumer(System):
        rerun_seen: bool=False
        rerunRequest: object=None
        weights: tuple=()
        def initializeNewState(self,*args,**kwargs):
            return Consumer(self.state.initializeNewState(),self.t,self.initialized,self.finalized,self.rerun_seen,None,self.weights)
        def initialize(self,h,stageWeights=(),rerun=None,**kwargs):
            self.initialized+=1;self.rerun_seen=rerun is not None;self.weights=stageWeights;return self
        def finalize(self,initial,h,auxes,updates,weights,**kwargs):
            assert tuple(weights)==initial.weights and len(updates)==len(weights)
            self.finalized=initial.finalized+1
            self.rerunRequest=None if initial.rerun_seen else {'ready':True}
            return self
    stages=[]
    def rhs(s,h,stageIndex):
        stages.append(stageIndex);return testing.ParticleUpdate(-s.state.x,-s.state.u,-s.state.e),None
    initial=Consumer(system().state);method=ExpandedRK('CarpenterKennedy2N54')
    result=method(initial,.1,rhs);n=method.metadata.stage_count
    assert stages==list(range(n))*2 and result.state.initialized==2 and result.state.finalized==1
    assert initial.initialized==initial.finalized==0 and initial.rerunRequest is None
    assert torch.equal(result.state.state.density,result.state.state.x.square()+result.state.state.e)
