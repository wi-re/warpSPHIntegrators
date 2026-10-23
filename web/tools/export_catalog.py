"""Export actual source coefficients. Run with the library's Python environment.

No rounded/transcribed method data lives in the browser implementation. Output is
stable and --check detects source drift. This does not modify the Python catalog.
"""
from pathlib import Path
import sys, json, dataclasses, argparse
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
sys.path.insert(0,str(ROOT/'scripts'))
from generate_catalog import snapshot
from warpSPHIntegrators.schemes import IntegrationSchemes
from warpSPHIntegrators.schemes.coefficients.explicit import getButcherTableau
from warpSPHIntegrators.schemes.coefficients.chebyshev import _COEFFS, _recurrence_clock_weights
from warpSPHIntegrators.schemes.coefficients.expanded_data import _METHODS
from warpSPHIntegrators.schemes.coefficients.expanded import expanded_rk_tableau
from warpSPHIntegrators.schemes.coefficients.collocation import collocation_tableau
from warpSPHIntegrators.schemes.coefficients.nystrom import _ORDER, nystrom_tableau
from warpSPHIntegrators.schemes.coefficients.rosenbrock import rosenbrock_tableau
from warpSPHIntegrators.schemes.coefficients.stabilized import stabilized_coefficients
from warpSPHIntegrators.schemes.coefficients.parallel_dirk import parallel_dirk_tableau
from warpSPHIntegrators.schemes.coefficients.extrapolation import extrapolation_coefficients
from warpSPHIntegrators.schemes.coefficients.exponential import PROGRAMS

def clean(x):
    if dataclasses.is_dataclass(x):return clean(dataclasses.asdict(x))
    if isinstance(x,np.ndarray):return x.tolist()
    if isinstance(x,dict):return {k:clean(v) for k,v in x.items()}
    if hasattr(x,'_asdict'):return clean(x._asdict())
    if isinstance(x,(tuple,list)):return [clean(v) for v in x]
    if isinstance(x,np.generic):return x.item()
    return x

def rk(tab):
    d=clean(tab)
    if isinstance(tab.b,tuple):d['b'],d['bhat']=clean(tab.b)
    return d

out=snapshot();byid={s['identifier']:s for s in out['schemes']}
for s in IntegrationSchemes:
    item=byid[s.identifier.name];tab=getattr(s.function,'tableau',None)
    item['registered']=True
    if tab is not None:
        item['kind']='ark' if hasattr(tab,'a_explicit') else 'coupled' if hasattr(tab,'companion_b') else 'rk'
        item['tableau']=rk(tab) if item['kind']=='rk' else clean(tab)
    elif s.identifier.name in ('explicitEuler','tvdRK2','tvdRK3'):
        item['kind']='rk';item['tableau']=rk(getButcherTableau({'explicitEuler':'forwardEuler','tvdRK2':'heunsMethod','tvdRK3':'SSPRK3'}[s.identifier.name]))
    else:item['kind']={'Symplectic':'mechanics','Newmark':'newmark','MultistepImplicit':'history','MultistepExplicit':'history','RelaxedChebyshev':'recurrence','LinearlyImplicit':'ros3p','Exponential':'exponential','IMEX':'imex-history'}[s.family.__name__]
    item['browser']=True

def add(id,name,family,order,kind,**kwargs):
    item=dict(identifier=id,name=name,friendly_name=name,family=family,order=order,kind=kind,registered=False,browser=True,**kwargs)
    out['schemes'].append(item)
for name in _METHODS:
    meta=_METHODS[name]
    add('expanded_'+name,name,'ExpandedRK',meta['order'],'rk',tableau=rk(expanded_rk_tableau(name)),estimate_order=meta['estimate_order'])
for family in ('gauss','radau'):
    for stages in range(2,10):
        add(f'collocation_{family}_{stages}',f'{family.title()} collocation · {stages} stages','Collocation',2*stages-(family=='radau'),'coupled',tableau=clean(collocation_tableau(family,stages)),implicit=True,estimate_order=stages if family=='gauss' else stages+1)
for name,order in _ORDER.items():
    tab=nystrom_tableau(name)
    add('nystrom_'+name,name,'Nystrom',order,'nystrom',tableau=clean(tab),state_kind='second_order',requires_separable=tab.abar is None,estimate_order=tab.estimate_order)
for name in ('Rosenbrock23','Rosenbrock32','RosenbrockW6S4OS'):
    tab=rosenbrock_tableau(name)
    add('rosenbrock_'+name,name,'Rosenbrock',tab.order,'rosenbrock',tableau=clean(tab),implicit=True,estimate_order=3 if name!='RosenbrockW6S4OS' else None)
for base,orders in [('euler',range(2,9)),('midpoint',range(2,11,2))]:
    for order in orders:
        seq,weights,comparison=extrapolation_coefficients(base,order)
        add(f'extrapolation_{base}_{order}',f'{base.title()} extrapolation {order}','Extrapolation',order,'extrapolation',base=base,subdivisions=seq,weights=weights,comparison=comparison,estimate_order=order if base=='euler' or order==2 else order-1)
add('pdirk44','Parallel DIRK 4 (serial stages)','ParallelDIRK',4,'rk',tableau=rk(parallel_dirk_tableau()),implicit=True)
add('generalizedAlpha','Generalized alpha','GeneralizedAlpha',2,'newmark',implicit=True,state_kind='second_order')
for family in ('rkl1','rkl2','rkg1','rkg2'):
    add('stabilized_'+family,family.upper(),'StabilizedRK',int(family[-1]),'recurrence',estimate_order=2)
out['recurrences']={}
for family in ('rkc1','rkc2','rkl2','rkl1','rkg1','rkg2'):
    out['recurrences'][family]={}
    for stages in range(3,66):
        if family in _COEFFS:
            m1,mu,nu,mt,gt=_COEFFS[family](stages)
            c,b=_recurrence_clock_weights(stages,m1,mu,nu,mt,gt)
            data=dict(mu=mu,nu=nu,mtilde=mt,gtilde=gt,c=c,b=b)
        else:data=clean(stabilized_coefficients(family,stages))
        out['recurrences'][family][stages]=data
from warpSPHIntegrators.schemes.families.exponential_rk import ORDERS
for name,order in ORDERS.items():
    add('exprk_'+name,name,'ExponentialRK',order,'exponential',program=name,estimate_order={'Exprb32':3,'Exprb43':4}.get(name))
# Expand the source's affine recurrence using formal independent RHS vectors.
# Each sampled stage is precisely a row of A; final/error offsets give b/bhat.
from warpSPHIntegrators.schemes.families.orthogonal import OrthogonalRK
from warpSPHIntegrators.schemes.coefficients.orthogonal import orthogonal_coefficients, recurrence_program
for name in ('ROCK2','ROCK4','SERK2','RKMC2'):
    seen=set()
    for requested in (10,20,40):
        method=OrthogonalRK(name,stages=requested);stages=method.stages
        if stages in seen:continue
        seen.add(stages);coeff=orthogonal_coefficients(name,stages)
        width=stages+2;rows=[]
        def evaluate(offset,*,endpoint=False):
            rows.append(offset.copy());rate=np.zeros(width);rate[len(rows)-1]=1.;return rate
        final,error=recurrence_program(name,stages,coeff.parameters,evaluate,np.zeros(width))
        count=len(rows)
        tab=dict(a=[row[:count].tolist() for row in rows],b=final[:count].tolist(),bhat=(final-error)[:count].tolist(),c=list(coeff.c))
        if len(tab['c'])<count:tab['c'].append(1.)
        add(f'orthogonal_{name}_{stages}',f'{name} · {stages} stages','OrthogonalRK',method.order,'rk',tableau=tab,estimate_order=method.estimate_order)
# Fixed-target NDF and corrected-startup IRKN are independent history maps.
from warpSPHIntegrators.schemes.families.ndf import NDF
for order in range(1,6):
    method=NDF(order)
    add(f'ndf_{order}',method.name,'NDF',order,'ndf',implicit=True,
        estimate_order=method.estimate_order,history_length=order,kappa=method.kappa,
        beta=method.beta,error_constant=method.error_constant,max_step_ratio=method.max_step_ratio,
        starter={1:'trbdf2',2:'trbdf2',3:'esdirk324l2sa',4:'kencarp4',5:'kencarp5'}[order],
        startup_estimate_order={1:2,2:3,3:3,4:4,5:5}[order])
for order in (3,4):
    add(f'irkn{order}',f'Improved Nyström {order} (corrected startup)','ImprovedNystrom',order,
        'irkn',state_kind='second_order',requires_separable=True,history_length=1)
out['exponentialPrograms']=clean(PROGRAMS)
out['portGaps']=[
 {'family':'AdaptiveAdams / VariableAdams','reason':'Variable-step and variable-order history controllers are not ported. Fixed AB/ABM orders 2–5 are available.'},
 {'family':'AdaptiveNDF','reason':'Fixed-target NDF1–5 with variable-h differences and defect control are available. The separate variable-order AdaptiveNDF controller is not yet ported.'},
 {'family':'OrthogonalRK stage selection','reason':'Source recurrences are expanded at three bounded stage configurations per method. Automatic spectral stage selection and larger source stage counts are not ported.'},
 {'family':'ConservedLagrangian / ProjectedStep','reason':'Custom constrained mechanics and application-defined projection contracts are not ported.'},
 {'family':'State lifecycle / AD','reason':'Browser states use fixed numeric layouts. Arbitrary Python callbacks, autograd graphs, custom lifecycle hooks, checkpoint compatibility and particle reordering are not portable.'}]
text=json.dumps(clean(out),indent=None,separators=(',',':'),allow_nan=False)+'\n'
path=ROOT/'web/src/catalog.json'
if '--check' in sys.argv:
    if not path.exists() or path.read_text()!=text:raise SystemExit('Browser catalog is stale; run web/tools/export_catalog.py')
else:path.write_text(text)
print(f'{len(out["schemes"])} browser configurations, {out["summary"]["scheme_count"]} registered schemes')
