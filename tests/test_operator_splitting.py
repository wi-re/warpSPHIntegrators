"""Physical split order, structure, lifecycle, ownership and gradients."""
from dataclasses import dataclass,field
import math
import pytest
import torch
from warpSPHIntegrators.wrappers import ConservativeDissipativeRHS, Strang, LieTrotter, compose, LinearExponentialFlow, classify_separable
from warpSPHIntegrators.adapters import SemilinearRHS, RHS, check_contracts
from warpSPHIntegrators import TensorState, tensor_rhs, solve, get_scheme
from warpSPHIntegrators.state import get_reference_state, BaseIntegrationSystem
from warpSPHIntegrators.solvers import SolveResult
import warpSPHIntegrators.testing as testing
from warpSPHIntegrators.runtime import RunError

OPTS=dict(matvec='jvp',tol=1e-12,newton_tol=1e-9)


def damped_split(k=4.,c=.5):
    def conservative(state,dt,**kwargs):
        s=get_reference_state(state)
        return testing.ParticleUpdate(s.u,-k*s.x,s.e*0),None
    def dissipative(state,dt,**kwargs):
        s=get_reference_state(state)
        return testing.ParticleUpdate(s.x*0,-c*s.u,s.e*0),None
    return ConservativeDissipativeRHS(conservative,SemilinearRHS(linear=dissipative))


def error(method,dt):
    p=testing.damped_problem();r=solve(method,p.initial(),1.,damped_split(),dt=dt,solver_opts=OPTS)
    x,u=p.exact(1.);s=get_reference_state(r.state)
    return abs(s.x.item()-x[0])+abs(s.u.item()-u[0])


@pytest.mark.parametrize('method,order',[
    (LieTrotter('Velocity Verlet','Implicit Midpoint'),1),
    (Strang('Velocity Verlet','Implicit Midpoint'),2),
    (Strang('PEFRL','Implicit Midpoint'),2),
    (Strang('RK4','RK4'),2),
    (Strang('Velocity Verlet','Backward Euler (implicit)'),1)])
def test_composition_measures_declared_order_including_be_control(method,order):
    errors=[error(method,h) for h in (.1,.05,.025)]
    assert method.order==order
    assert math.log(errors[-2]/errors[-1],2)==pytest.approx(order,abs=.15)


def test_split_capabilities_additivity_and_tensor_nested_linear_adaptation():
    p=testing.damped_problem();rhs=damped_split();check_contracts(rhs,p.initial(),.1,require_separable=True)
    assert rhs.provides=={'conservative','dissipative'}
    assert classify_separable(rhs._conservative,p.initial()).separable
    assert not classify_separable(p.rhs,p.initial()).separable
    adapted=tensor_rhs(ConservativeDissipativeRHS(lambda t,y:y,SemilinearRHS(linear=lambda t,y:-2*y)))
    assert isinstance(adapted._dissipative,RHS) and 'linear' in adapted._dissipative.provides
    initial=TensorState(torch.ones(1,dtype=torch.float64))
    result=solve(Strang('RK4',LinearExponentialFlow(torch.tensor([[-2.]],dtype=torch.float64))),
                 initial,.5,adapted,dt=.05)
    assert result.state.y.item()==pytest.approx(math.exp(-.5),rel=1e-6)
    assert initial.y.item()==1.


def test_evolving_thermal_quantity_and_hidden_velocity_dependence_are_not_separable():
    state=testing.damped_problem().initial()
    def thermal(s,dt,**kwargs):
        r=get_reference_state(s)
        return testing.ParticleUpdate(r.u,-r.x,r.u),None
    def cubic(s,dt,**kwargs):
        r=get_reference_state(s)
        return testing.ParticleUpdate(r.u,-r.x+r.u**3,r.e*0),None
    assert not classify_separable(thermal,state).separable
    assert not classify_separable(cubic,state).separable  # velocity=0 first probe is insufficient
    assert not classify_separable(lambda s,dt:testing.ParticleUpdate(s.state.u,s.state.u*float('nan'),s.state.e*0),state).separable


def test_structure_errors_precede_initialize_and_invalid_recipes_are_refused():
    state=testing.damped_problem().initial();good=damped_split()
    with pytest.raises(ValueError,match='preserve configuration'):
        Strang()(state,.1,ConservativeDissipativeRHS(good._conservative,good._conservative))
    with pytest.raises(ValueError,match='autonomous'):
        Strang()(state,.1,ConservativeDissipativeRHS(good._conservative,good._dissipative,autonomous=False))
    with pytest.raises(ValueError,match='separable'):
        Strang('Velocity Verlet')(state,.1,ConservativeDissipativeRHS(testing.damped_problem().rhs,good._dissipative))
    with pytest.raises(ValueError,match='one-step'):
        Strang('BDF3','Implicit Midpoint')
    with pytest.raises(ValueError,match='Negative dissipative'):
        compose([('conservative','RK4'),('dissipative','RK4'),('dissipative','RK4')],[1,-.1,1.1])
    with pytest.raises(ValueError,match='palindromic'):
        compose([('conservative','RK4'),('dissipative','RK4')],[1,1],order=2)
    with pytest.raises(ValueError):compose([('conservative','RK4')],[1])


@dataclass
class LifecycleTensor(TensorState):
    composition_compatible=True
    calls:list=field(default_factory=list)
    def initializeNewState(self,*args,**kwargs):
        return LifecycleTensor(self.state.initializeNewState(),self.t,self.calls)
    def initialize(self,dt,*args,**kwargs):self.calls.append(('initialize',dt));return self
    def initialize_substep(self,dt,*args,**kwargs):self.calls.append(('substep',dt));return self
    def finalize(self,initial,dt,returns,updates,weights,*args,**kwargs):
        records=kwargs['composition_substeps']
        assert len(records)==3 and len(weights)==3
        self.calls.append(('finalize',dt));return self


def test_outer_lifecycle_once_per_accepted_step_and_no_caller_mutation():
    state=LifecycleTensor(torch.ones(1,dtype=torch.float64))
    f=tensor_rhs(ConservativeDissipativeRHS(lambda t,y:y,lambda t,y:-2*y))
    r=solve(Strang('RK4','Implicit Midpoint'),state,.2,f,dt=.1,solver_opts=OPTS)
    assert state.calls==[]
    assert [name for name,dt in r.state.calls].count('initialize')==2
    assert [name for name,dt in r.state.calls].count('finalize')==2
    assert [name for name,dt in r.state.calls].count('substep')==6
    assert r.state.t==.2
    # Restart fixed composition retains the bound recipe and caller ownership.
    resumed=solve(Strang('RK4','Implicit Midpoint'),None,.4,f,restart=r.checkpoint,solver_opts=OPTS)
    full=solve(Strang('RK4','Implicit Midpoint'),state,.4,f,dt=.1,solver_opts=OPTS)
    assert torch.equal(resumed.state.y,full.state.y)


def test_custom_finalize_requires_explicit_composition_support():
    @dataclass
    class Unsupported(TensorState):
        def finalize(self,*args,**kwargs):raise AssertionError('must not run')
    f=tensor_rhs(ConservativeDissipativeRHS(lambda t,y:y,lambda t,y:-y))
    with pytest.raises(TypeError,match='composition_compatible'):
        Strang()(Unsupported(torch.ones(1,dtype=torch.float64)),.1,f)


def test_failed_dissipative_solve_commits_no_state_or_lifecycle():
    class Fail:
        def solve(self,step,y0,norm=None,**options):return SolveResult(y0,False,0)
    state=LifecycleTensor(torch.ones(1,dtype=torch.float64))
    f=tensor_rhs(ConservativeDissipativeRHS(lambda t,y:y,lambda t,y:-y))
    with pytest.raises(RunError) as info:
        solve(Strang(dissipative_options={'solver':Fail()}),state,1.,f,dt=.1)
    assert info.value.result.accepted_steps==0 and info.value.result.state.calls==[]
    assert state.calls==[]


def test_exact_dissipative_matrix_has_state_and_parameter_gradients():
    y=torch.ones(1,dtype=torch.float64,requires_grad=True)
    damping=torch.tensor(2.,dtype=torch.float64,requires_grad=True)
    f=tensor_rhs(ConservativeDissipativeRHS(lambda t,y:y,SemilinearRHS(linear=lambda t,y:-damping*y)))
    method=Strang('RK4',LinearExponentialFlow((-damping).reshape(1,1)))
    r=solve(method,TensorState(y),.5,f,dt=.05)
    gy,gc=torch.autograd.grad(r.state.y.sum(),(y,damping))
    assert gy.item()==pytest.approx(math.exp(-.5),rel=1e-6)
    assert gc.item()==pytest.approx(-.5*math.exp(-.5),rel=1e-6)
    # Implicit D also keeps parameter gradients through the IFT.
    d=torch.tensor(2.,dtype=torch.float64,requires_grad=True)
    g=tensor_rhs(ConservativeDissipativeRHS(lambda t,y:y,lambda t,y:-d*y))
    result=solve(Strang('RK4','Implicit Midpoint'),TensorState(y),.5,g,dt=.01,solver_opts=OPTS)
    assert torch.autograd.grad(result.state.y.sum(),d)[0].item()==pytest.approx(-.5*math.exp(-.5),rel=1e-4)


def test_second_order_subflow_need_not_be_symmetric_for_strang_order_two():
    method=Strang('RK4','Midpoint')  # explicit midpoint is order two, not time-reversible
    errors=[error(method,h) for h in (.1,.05,.025)]
    assert math.log(errors[-2]/errors[-1],2)==pytest.approx(2,abs=.15)


def test_precision_floor_opt_in_finishes_every_subflow_before_acceptance():
    from warpSPHIntegrators.solvers import SolveDiagnostics
    class Floor:
        def solve(self,step,y0,norm=None,**opts):
            return SolveResult(step(y0),True,1,SolveDiagnostics(termination='stagnation'))
    calls=[]
    def conservative(t,y):calls.append(t);return y
    rhs=tensor_rhs(ConservativeDissipativeRHS(conservative,lambda t,y:-y))
    result=solve(Strang(dissipative_options={'solver':Floor()}),TensorState(torch.ones(1,dtype=torch.float64)),
                 .1,rhs,dt=.1,allow_stagnation=True)
    assert result.successful and len(calls)==4  # conservative RK4 was not skipped


@pytest.mark.parametrize('dissipative,expected',[('Implicit Midpoint',2),('Backward Euler (implicit)',1)])
def test_burgers_split_order_and_mass_against_an_unsplit_reference(dissipative,expected):
    from pathlib import Path
    import sys
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'benchmarks'))
    from splitting_validation import split_burgers
    state,rhs,_=split_burgers(n=16)
    opts=dict(matvec='jvp',tol=1e-12,newton_tol=1e-7)
    reference=solve('RK4',state,.1,rhs,dt=.0001)
    errors=[]
    for h in (.01,.005,.0025):
        result=solve(Strang('RK4',dissipative),state,.1,rhs,dt=h,solver_opts=opts)
        errors.append(torch.linalg.vector_norm(result.state.y-reference.state.y).item())
        assert abs(float(result.state.y.sum()-state.y.sum()))<1e-10
    assert math.log(errors[-2]/errors[-1],2)==pytest.approx(expected,abs=.15)


def test_nested_time_dependent_part_is_rejected_by_autonomous_composition():
    initial=TensorState(torch.ones(1,dtype=torch.float64))
    rhs=tensor_rhs(ConservativeDissipativeRHS(lambda t,y:y,RHS(linear=lambda t,y:-t*y,autonomous=False)))
    with pytest.raises(ValueError,match='autonomous'):
        Strang()(initial,.1,rhs)


def test_gated_yoshida4_reaches_fourth_order_on_the_known_damped_oscillator():
    from warpSPHIntegrators.wrappers import Yoshida4
    matrix=torch.diag(torch.tensor([0.,-.5,0.],dtype=torch.float64))
    method=Yoshida4('Velocity Verlet',LinearExponentialFlow(matrix))
    errors=[error(method,h) for h in (.2,.1,.05)]
    assert method.order==4
    assert math.log(errors[-2]/errors[-1],2)==pytest.approx(4,abs=.2)
    # The backwards adjoint map is checked within the same explicit bound.
    state=testing.damped_problem().initial();f=damped_split()
    forward=method(state,.1,f,solver_opts=OPTS)
    backward=method(forward.state,-.1,f,solver_opts=OPTS)
    assert torch.allclose(get_reference_state(backward.state).x,get_reference_state(state).x,atol=1e-12)
    assert torch.allclose(get_reference_state(backward.state).u,get_reference_state(state).u,atol=1e-12)


def test_high_order_anti_diffusion_and_nonsymmetric_subflow_guards():
    from warpSPHIntegrators.wrappers import Yoshida4
    with pytest.raises(ValueError,match='symmetric'):
        Yoshida4('Midpoint',LinearExponentialFlow(torch.tensor([[-1.]],dtype=torch.float64)))
    with pytest.raises(ValueError,match='static exact'):
        Yoshida4('Implicit Midpoint','Implicit Midpoint')
    strong=Yoshida4('Implicit Midpoint',LinearExponentialFlow(torch.tensor([[-1000.]],dtype=torch.float64)))
    rhs=tensor_rhs(ConservativeDissipativeRHS(lambda t,y:y,SemilinearRHS(linear=lambda t,y:-1000*y)))
    with pytest.raises(ValueError,match='weak-dissipation'):
        strong(TensorState(torch.ones(1,dtype=torch.float64)),.1,rhs)


def test_oscillator_has_the_exact_contraction_and_second_order_energy_budget():
    from pathlib import Path
    import sys
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'benchmarks'))
    from splitting_validation import oscillator_diagnostics
    report=oscillator_diagnostics()
    geometry={r['method']:r for r in report['geometry']}
    assert geometry['Strang']['defect']<1e-9
    assert geometry['Yoshida4']['defect']<1e-9
    assert geometry['unsplit-RK4']['defect']>1e-4
    assert report['energy_balance_order']==pytest.approx(2,abs=.15)
    assert all(row['final_energy']<row['initial_energy'] for row in report['energy_balance'])


def test_mechanical_structure_probes_preserve_bound_rhs_parameters():
    def conservative(state,dt,*,spring,**kwargs):
        s=get_reference_state(state)
        return testing.ParticleUpdate(s.u,-spring*s.x,s.e*0),None
    def dissipative(state,dt,*,damping,**kwargs):
        s=get_reference_state(state)
        return testing.ParticleUpdate(s.x*0,-damping*s.u,s.e*0),None
    rhs=ConservativeDissipativeRHS(conservative,dissipative)
    method=Strang('Velocity Verlet','RK4',conservative_options={'spring':4.},dissipative_options={'damping':.5})
    result=method(testing.damped_problem().initial(),.1,rhs)
    assert result.step_status.successful
