"""Qualified source-expanded RK maps: known accuracy/work and catalog screening."""
import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import platform
import sys
import time
import numpy as np
import torch
from warpSPHIntegrators.schemes import ExpandedRK, filter_schemes
from warpSPHIntegrators.schemes.coefficients.expanded import expanded_rk_methods
from warpSPHIntegrators import TensorState, tensor_rhs, solve, get_scheme

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('expanded_generator',ROOT/'scripts/coefficients/wave16/generate.py')
generator=importlib.util.module_from_spec(spec);spec.loader.exec_module(generator)


def equivalence_screen(methods):
    catalog=[]
    for method in filter_schemes(implicit=False):
        tableau=getattr(method.function,'butcherTableau',None)
        if tableau is None:continue
        b=tableau.b[0] if isinstance(tableau.b,tuple) else tableau.b
        a=np.asarray(tableau.a,dtype=float);b=np.asarray(b,dtype=float);c=np.asarray(tableau.c,dtype=float)
        if a.shape!=(len(b),len(b)):continue
        catalog.append((method.identifier.name,generator.main_map(a,b,c)))
    result={};canonical={}
    for method in methods:
        t=method.tableau;b=t.b[0] if isinstance(t.b,tuple) else t.b
        a,b,c=generator.main_map(t.a,b,t.c);matches=[]
        canonical[method.method]=(a,b,c)
        for name,(aa,bb,cc) in catalog:
            if a.shape==aa.shape and all(np.allclose(x,y,atol=2e-13,rtol=0) for x,y in zip((a,b,c),(aa,bb,cc))):matches.append(name)
        result[method.method]=matches
    groups=[];seen=set()
    for name,values in canonical.items():
        if name in seen:continue
        group=[other for other,other_values in canonical.items()
            if values[0].shape==other_values[0].shape
            and all(np.allclose(x,y,atol=2e-13,rtol=0) for x,y in zip(values,other_values))]
        seen.update(group)
        if len(group)>1:groups.append(sorted(group))
    return dict(fixed_tableaus_compared=len(catalog),canonical_coefficient_matches=result,internal_main_map_groups=groups,
        qualification='Duplicate/unreachable-stage canonical A,b,c equality is sufficient for map equivalence. A nonmatch alone is not a proof against every possible RK representation or non-tableau factory. No duplicate fixed registrations are added.')


def measurements():
    qualified=[ExpandedRK(name) for name in expanded_rk_methods()]
    methods=[*qualified,get_scheme('RK4'),get_scheme('dormandPrince')]
    fixed=[];adaptive=[]
    for n in (64,4096):
        rates=torch.linspace(.1,10,n,dtype=torch.float64)
        for problem in ('reaction','spiral_heat'):
            initial=torch.ones_like(rates) if problem=='reaction' else torch.stack((torch.ones_like(rates),torch.zeros_like(rates),torch.zeros_like(rates)))
            exact=(torch.exp(-rates) if problem=='reaction' else
                torch.stack((math.exp(-.2)*torch.cos(rates),-math.exp(-.2)*torch.sin(rates),torch.full_like(rates,.5*(1-math.exp(-.4))))))
            for method in methods:
                trials=[dict(dt=h) for h in (.1,.05)]
                if n==4096 and method.has_error_estimate:
                    trials+=[dict(dt=.2,adaptive=True,rtol=tol,atol=tol*.01) for tol in (1e-5,1e-8)]
                for options in trials:
                    calls=0
                    def force(t,y):
                        nonlocal calls
                        calls+=1
                        if problem=='reaction':return -rates*y-.05*y**3+.05*torch.exp(-3*rates*t)
                        return torch.stack((rates*y[1]-.2*y[0],-rates*y[0]-.2*y[1],.2*(y[0].square()+y[1].square())))
                    start=time.perf_counter()
                    result=solve(method,TensorState(initial),1.,tensor_rhs(force),**options)
                    row=dict(n=n,problem=problem,method=method.name,source_method=getattr(method,'method',None),
                        **options,successful=result.successful,accepted=result.accepted_steps,rejected=result.rejected_steps,
                        rhs_calls=calls,seconds=time.perf_counter()-start,max_error=float((result.state.y-exact).abs().max()))
                    if problem=='spiral_heat':
                        y=result.state.y
                        row.update(heat_error=float((y[2]-exact[2]).abs().max()),
                            energy_defect=float(((y[0].square()+y[1].square())/2+y[2]-.5).abs().max()))
                    (adaptive if options.get('adaptive') else fixed).append(row)
    return dict(device='cpu',dtype='float64',threads=1,python=sys.version,torch=torch.__version__,platform=platform.platform(),
        cases=fixed,adaptive=adaptive,equivalence_screen=equivalence_screen(qualified),
        qualification='Analytic nonlinear manufactured reaction and damped planar rotations with generated heat. Source-expanded full-stage RK with owned state; no low-storage, source FSAL/controller or SSP/conservation claim. Entry counts include physical endpoint/retries. Same grids/tolerances are not equal-accuracy speed claims; high-order fine results can reach roundoff.',
        source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (
            ROOT/'src/warpSPHIntegrators/schemes/families/expanded_rk.py',ROOT/'src/warpSPHIntegrators/schemes/coefficients/expanded_data.py',
            ROOT/'src/warpSPHIntegrators/schemes/definitions/explicit.py',ROOT/'src/warpSPHIntegrators/runtime/run.py',Path(__file__).resolve())})


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,default=Path('.tmp/phase8/expanded-rk-backend'))
    args=parser.parse_args();torch.set_num_threads(1);report=measurements()
    args.output_dir.mkdir(parents=True,exist_ok=True)
    (args.output_dir/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    assert all(row['successful'] for row in report['cases']+report['adaptive'])
    print('Measured',len(report['cases']),'fixed and',len(report['adaptive']),'adaptive cases;',report['equivalence_screen']['fixed_tableaus_compared'],'fixed tableaus screened')
