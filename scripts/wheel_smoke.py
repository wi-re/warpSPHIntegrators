"""Executed outside the checkout by validate_wheel; imports only the installed wheel."""

from pathlib import Path
import importlib
import importlib.metadata
import json
import math
import pkgutil
import sys
import tempfile

import torch
import warpSPHIntegrators as package
from warpSPHIntegrators import (
    TensorState, tensor_rhs, getIntegrator, getPreferredIntegrator, filter_schemes,
    StepHistory, SemilinearRHS, testing, get_reference_state,
    solve, Event, save_checkpoint, load_checkpoint,
    stage_block_preconditioner,
    stage_shifted_preconditioner,
    ConservativeDissipativeRHS, Strang, Yoshida4, LinearExponentialFlow,
    GeneralizedAlpha, Collocation, VariableAdams, StabilizedRK, Nystrom, NystromWeights, NDF, AdaptiveNDF,
    Extrapolation, ExtrapolationQuadrature,
)


prefix = Path(sys.prefix).resolve()
assert Path(package.__file__).resolve().is_relative_to(prefix), package.__file__
assert package.__version__ == importlib.metadata.version('warpSPHIntegrators') == sys.argv[1]
assert (Path(package.__file__).parent / 'py.typed').is_file()
for module in pkgutil.walk_packages(package.__path__, package.__name__ + '.'):
    imported = importlib.import_module(module.name)
    assert Path(imported.__file__).resolve().is_relative_to(prefix), imported.__file__
assert all(hasattr(package, name) for name in package.__all__)
expected = json.loads(Path(sys.argv[2]).read_text())
assert {s.identifier.name for s in filter_schemes()} == {s['identifier'] for s in expected['schemes']}
assert getPreferredIntegrator(4) is getIntegrator('RK4')
y = torch.tensor([1., 2.], dtype=torch.float64)
for scheme in filter_schemes(state_kind='tensor'):
    state = TensorState(y)
    f = tensor_rhs(SemilinearRHS(lambda t, y: -y))
    options = {'s':3} if scheme.metadata.required_options else {}
    history = StepHistory(maxlen=5)
    for _ in range(6):
        result = scheme(state, .001, f, history=history, **options)
        state, history = result.state, result.history or history
    assert torch.allclose(state.y, y * math.exp(-.006), atol=2e-5, rtol=2e-5), scheme.name
# Keep the advanced protocol consumer in the installed-package gate too.
problem = testing.oscillator_problem()
result = getIntegrator('RK4')(problem.initial(), .01, problem.rhs)
assert torch.isfinite(get_reference_state(result.state).x).all()
# The installed runner, dense/event path and portable restart must work together.
rhs = tensor_rhs(lambda t, y: -y)
run = solve('Dormand-Prince 5(4)', TensorState(y), 1., rhs, dt=.3,
            adaptive=True, rtol=1e-8, atol=1e-10, sample_times=[.25,.75])
assert run.state.t == 1. and torch.allclose(run.state.y,y*math.exp(-1),rtol=3e-8)
with tempfile.TemporaryDirectory() as directory:
    path = Path(directory) / 'restart.pt'
    save_checkpoint(run.checkpoint,path)
    resumed = solve('Dormand-Prince 5(4)',None,2.,rhs,restart=load_checkpoint(path),
                    events=[Event(lambda t,s:s.y[0].item()-.2,terminal=True)])
assert resumed.termination == 'event' and abs(resumed.state.t-math.log(5)) < 1e-7
stiff = solve('BDF5',TensorState(torch.ones(1,dtype=torch.float64)),1.,
              tensor_rhs(lambda t,y:-100*y),dt=.1,
              solver_opts={'matvec':'jvp','tol':1e-12,'newton_tol':1e-9})
assert stiff.successful and abs(stiff.state.y.item()) < .001
rates = torch.logspace(0,4,32,dtype=torch.float64)
block = getIntegrator('Radau IIA s=2')(TensorState(torch.ones_like(rates)),.01,
            tensor_rhs(lambda t,y:-rates*y),solver_opts={'matvec':'jvp',
            'tol':1e-10,'newton_tol':1e-7,'preconditioner':stage_block_preconditioner(-rates)})
assert block.step_status.tolerance_converged
shifted = getIntegrator('Radau IIA s=2')(TensorState(torch.ones_like(rates)),.01,
            tensor_rhs(lambda t,y:-rates*y),solver_opts={'matvec':'jvp',
            'tol':1e-10,'newton_tol':1e-7,'preconditioner':stage_shifted_preconditioner(
                lambda state,context:lambda shift,v:v/(1+shift*rates))})
assert shifted.step_status.tolerance_converged and torch.allclose(shifted.state.y,block.state.y,atol=1e-9)
physical = tensor_rhs(ConservativeDissipativeRHS(lambda t,y:y,
                SemilinearRHS(linear=lambda t,y:-2*y)))
split = solve(Strang('RK4',LinearExponentialFlow(torch.tensor([[-2.]],dtype=torch.float64))),
              TensorState(torch.ones(1,dtype=torch.float64)),.5,physical,dt=.05)
assert split.successful and abs(split.state.y.item()-math.exp(-.5))<1e-6
fourth = solve(Yoshida4('Implicit Midpoint',LinearExponentialFlow(torch.tensor([[-2.]],dtype=torch.float64))),
               TensorState(torch.ones(1,dtype=torch.float64)),.5,physical,dt=.05,
               solver_opts={'matvec':'jvp','tol':1e-12,'newton_tol':1e-9})
assert fourth.successful and abs(fourth.state.y.item()-math.exp(-.5))<1e-6
mechanics = testing.oscillator_problem()
alpha = solve(GeneralizedAlpha(.5),mechanics.initial(),.2,mechanics.rhs,dt=.01,
              solver_opts={'matvec':'jvp','tol':1e-12,'newton_tol':1e-7})
continued = solve(GeneralizedAlpha(.5),None,.35,mechanics.rhs,restart=alpha.checkpoint,
                  solver_opts={'matvec':'jvp','tol':1e-12,'newton_tol':1e-7})
assert continued.successful and continued.history.latest.state.t == .35
assert abs(continued.state.state.x.item()-mechanics.exact(.35)[0][0])<2e-4
collocation=solve(Collocation('radau',5),TensorState(torch.ones(1,dtype=torch.float64)),.5,
                  tensor_rhs(lambda t,y:-2*y),dt=.1,
                  solver_opts={'matvec':'jvp','tol':1e-12,'newton_tol':1e-9})
assert collocation.successful and abs(collocation.state.y.item()-math.exp(-1))<1e-10
adams=solve(VariableAdams(4,'abm'),TensorState(torch.ones(1,dtype=torch.float64)),.3,
            tensor_rhs(lambda t,y:-y),dt=.025)
with tempfile.TemporaryDirectory() as directory:
    path=Path(directory)/'adams.pt';save_checkpoint(adams.checkpoint,path)
    restored=solve(VariableAdams(4,'abm'),None,.6,tensor_rhs(lambda t,y:-y),restart=load_checkpoint(path))
assert restored.successful and abs(restored.state.y.item()-math.exp(-.6))<1e-6
cnlf=solve('cnlf2',TensorState(torch.ones(1,dtype=torch.float64)),.2,rhs,dt=.025)
with tempfile.TemporaryDirectory() as directory:
    path=Path(directory)/'cnlf.pt';save_checkpoint(cnlf.checkpoint,path)
    restored=solve('cnlf2',None,.4,rhs,restart=load_checkpoint(path))
assert restored.successful and abs(restored.state.y.item()-math.exp(-.4))<1e-4
stabilized=solve(StabilizedRK('rkg2',lambda_max=100.),TensorState(torch.ones(3,dtype=torch.float64)),.4,
    rhs,dt=.1,adaptive=True,rtol=1e-6,atol=1e-9)
assert stabilized.successful and torch.allclose(stabilized.state.y,torch.full((3,),math.exp(-.4),dtype=torch.float64),atol=2e-6)
quadratures=[]
rkn=solve(Nystrom('FineRKN5'),mechanics.initial(),.2,mechanics.rhs,dt=.025,
    callback=lambda record:quadratures.append(record.result.quadrature))
rkn=solve(Nystrom('FineRKN5'),None,.4,mechanics.rhs,restart=rkn.checkpoint)
assert rkn.successful and isinstance(quadratures[-1],NystromWeights)
assert abs(rkn.state.state.x.item()-mechanics.exact(.4)[0][0])<1e-8
ndf=solve(NDF(3),TensorState(torch.ones(1,dtype=torch.float64)),.3,rhs,dt=.025)
assert ndf.history.latest.update is None and ndf.history.context['phase']=='warm'
with tempfile.TemporaryDirectory() as directory:
    path=Path(directory)/'ndf.pt';save_checkpoint(ndf.checkpoint,path)
    restored=solve(NDF(3),None,.6,rhs,restart=load_checkpoint(path))
assert restored.successful and abs(restored.state.y.item()-math.exp(-.6))<1e-5
adaptive_ndf=solve(AdaptiveNDF(),TensorState(torch.ones(1,dtype=torch.float64)),1.,rhs,
    dt=.001,adaptive=True,rtol=1e-7,atol=1e-9)
assert adaptive_ndf.successful and adaptive_ndf.history.context['order']>=3
with tempfile.TemporaryDirectory() as directory:
    path=Path(directory)/'adaptive-ndf.pt';save_checkpoint(adaptive_ndf.checkpoint,path)
    resumed_ndf=solve(AdaptiveNDF(),None,2.,rhs,restart=load_checkpoint(path))
assert resumed_ndf.successful and abs(resumed_ndf.state.y.item()-math.exp(-2))<2e-6
extrapolation_records=[]
extrapolated=solve(Extrapolation('midpoint',6),TensorState(torch.ones(1,dtype=torch.float64)),1.,rhs,
    dt=.2,adaptive=True,rtol=1e-8,atol=1e-10,callback=extrapolation_records.append)
assert extrapolated.successful and abs(extrapolated.state.y.item()-math.exp(-1))<2e-8
assert isinstance(extrapolation_records[-1].result.quadrature,ExtrapolationQuadrature)
print(f'Installed wheel {package.__version__}: all modules/exports, py.typed, '
      f'{len(filter_schemes(state_kind="tensor"))} tensor drivers, advanced state protocol, '
      'adaptive sampling/events, portable restart, stiff bootstrap, stage preconditioning, physical splitting and generalized-alpha continuation pass')
