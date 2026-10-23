"""Generate independent browser parity fixtures by executing the Python library."""
from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
from warpSPHIntegrators.schemes import IntegrationSchemes
from warpSPHIntegrators.testing import oscillator_problem,forced_problem,damped_problem
from warpSPHIntegrators.solvers.krylov import JFNKSolver
from warpSPHIntegrators.state.fields import get_reference_state

def save_or_check(path,rows):
 if '--check' in sys.argv:
  old=json.loads(path.read_text())
  if len(old)!=len(rows):raise SystemExit(f'{path.name}: fixture count changed')
  for a,b in zip(old,rows):
   if {k:v for k,v in a.items() if k!='y'}!={k:v for k,v in b.items() if k!='y'} or any(abs(x-y)>1e-12 for x,y in zip(a['y'],b['y'])):
    raise SystemExit(f'{path.name}: stale fixture {b["method"]} / {b["problem"]}')
 else:path.write_text(json.dumps(rows,separators=(',',':'))+'\n')

out=[]
for pid,factory in [('oscillator',oscillator_problem),('forced',forced_problem),('damped',damped_problem)]:
 for scheme in IntegrationSchemes:
  if scheme.family.__name__=='Exponential':continue # Requires a structured linear operator.
  prob=factory();state=prob.initial();history=None
  for i in range(6):
   kwargs=dict(s=9,history=history)
   if scheme.implicit:kwargs['solver']=JFNKSolver(tol=1e-13,newton_tol=1e-12)
   if scheme.identifier.name=='ros3p':kwargs.update(gmres_tol=1e-13,f_t='none' if pid!='forced' else 'fd',ft_eps=1e-7)
   result=scheme(state,.025,prob.rhs,**kwargs);state=result.state;history=result.history
  ref=get_reference_state(state)
  out.append(dict(problem=pid,method=scheme.identifier.name,dt=.025,T=.15,y=[*ref.x.tolist(),*ref.u.tolist()]))
path=ROOT/'web/tests/python-reference.json';save_or_check(path,out);print(f'{len(out)} Python run fixtures')
# Parameterized backends and the two structured exponential registry entries.
import torch
from warpSPHIntegrators.schemes import get_scheme
from warpSPHIntegrators.schemes.families import expanded_rk,collocation,nystrom,linearized,extrapolation,parallel_dirk,generalized_alpha,stabilized,exponential_rk,orthogonal
from warpSPHIntegrators.adapters.rhs import SemilinearRHS
from warpSPHIntegrators.testing import ParticleUpdate
catalog=json.loads((ROOT/'web/src/catalog.json').read_text());extra=[]
for item in catalog['schemes']:
 id=item['identifier']
 if item['registered'] and id not in ('etd2rk','exprb32'):continue
 if id.startswith('expanded_'):scheme=expanded_rk.ExpandedRK(id.removeprefix('expanded_'))
 elif id.startswith('collocation_'):
  _,family,stages=id.split('_');scheme=collocation.Collocation(family,int(stages))
 elif id.startswith('nystrom_'):scheme=nystrom.Nystrom(id.removeprefix('nystrom_'))
 elif id.startswith('rosenbrock_'):scheme=linearized.Rosenbrock(id.removeprefix('rosenbrock_'))
 elif id.startswith('extrapolation_'):
  _,base,order=id.split('_');scheme=extrapolation.Extrapolation(base,int(order))
 elif id=='pdirk44':scheme=parallel_dirk.ParallelDIRK()
 elif id=='generalizedAlpha':scheme=generalized_alpha.GeneralizedAlpha(.5)
 elif id.startswith('stabilized_'):scheme=stabilized.StabilizedRK(id.removeprefix('stabilized_'),stages=9)
 elif id.startswith('exprk_'):scheme=exponential_rk.ExponentialRK(id.removeprefix('exprk_'))
 elif id.startswith('orthogonal_'):
  _,method,stages=id.split('_');scheme=orthogonal.OrthogonalRK(method,int(stages))
 elif id in ('etd2rk','exprb32'):scheme=get_scheme(id)
 else:raise ValueError(id)
 prob=oscillator_problem();state=prob.initial();history=None;rhs=prob.rhs
 if item['kind']=='exponential':
  def zero(system,dt,**kwargs):
   ref=get_reference_state(system)
   return ParticleUpdate(torch.zeros_like(ref.x),torch.zeros_like(ref.u),torch.zeros_like(ref.e)),None
  rhs=SemilinearRHS(linear=prob.rhs,nonlinear=zero)
 for i in range(6):
  kwargs=dict(history=history)
  if getattr(scheme,'implicit',False):kwargs['solver']=JFNKSolver(tol=1e-13,newton_tol=1e-12)
  result=scheme(state,.025,rhs,**kwargs);state=result.state;history=result.history
 ref=get_reference_state(state);extra.append(dict(problem='oscillator',method=id,dt=.025,T=.15,y=[*ref.x.tolist(),*ref.u.tolist()]))
save_or_check(ROOT/'web/tests/parameter-reference.json',extra);print(f'{len(extra)} parameterized/structured Python fixtures')
