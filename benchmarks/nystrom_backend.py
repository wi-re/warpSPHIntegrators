"""Nyström work/accuracy and heat budgets on known vector mechanics problems."""
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
from warpSPHIntegrators import Nystrom,Strang,getIntegrator,solve

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('nystrom_demo',ROOT/'examples/nystrom_demo.py')
demo=importlib.util.module_from_spec(spec);sys.modules[spec.name]=demo;spec.loader.exec_module(demo)
SOURCE=json.loads((ROOT/'scripts/coefficients/wave12/nystrom.json').read_text())


def measurements():
    rows=[];methods=[Nystrom(name,structure='declared') for name in SOURCE['tableaus'] if name!='RKN4']
    baseline=[getIntegrator('RK4'),getIntegrator('dormandPrince')]
    for n in (64,4096):
        for damping in (0.,.4):
            initial,rhs,exact,omega,split=demo.problem(n,damping)
            selected=methods if damping==0 else [m for m in methods if m.tableau.abar is not None]
            selected=[*selected,*baseline]
            if damping:selected.append(Strang(Nystrom('Nystrom4VelocityIndependent',structure='declared'),'RK4'))
            else:selected.append(Nystrom('DPRKN6',structure='probe'))
            for method in selected:
                for h in (.1,.05):
                    calls=0
                    def counted(system,dt,**kwargs):
                        nonlocal calls
                        calls+=1
                        return rhs(system,dt,**kwargs)
                    # Physical splitting reads its declared components; count
                    # both underlying entrypoints rather than a combined call.
                    chosen=counted
                    if method.name=='Strang':
                        from warpSPHIntegrators import ConservativeDissipativeRHS
                        def cons(system,dt,**kwargs):
                            nonlocal calls
                            calls+=1
                            return split._conservative(system,dt,**kwargs)
                        def diss(system,dt,**kwargs):
                            nonlocal calls
                            calls+=1
                            return split._dissipative(system,dt,**kwargs)
                        chosen=ConservativeDissipativeRHS(cons,diss)
                    start=time.perf_counter()
                    result=solve(method,initial(),1.,chosen,dt=h)
                    seconds=time.perf_counter()-start;s=result.state.state;q,v,heat=exact(1.)
                    budget=(s.u.square()+omega.square()*s.x.square())/2+s.e-omega.square()/2
                    rows.append(dict(n=n,damping=damping,method=method.name,structure=getattr(method,'structure',None),
                        dt=h,steps=result.accepted_steps,successful=result.successful,rhs_calls=calls,seconds=seconds,
                        position_error=float((s.x-q).abs().max()),velocity_error=float((s.u-v).abs().max()),
                        heat_error=float((s.e-heat).abs().max()),total_energy_defect=float(budget.abs().max())))
    tab=SOURCE['tableaus']['RKN4'];c=np.r_[0,tab['c']];bp=np.array(tab['bp']);abar=np.array(tab['abar'])
    errors=[]
    for h in (.1,.05,.025,.0125):
        z=-h;r=1+z+z*z/2+z*z*z/12
        errors.append(abs(r**round(1/h)-math.exp(-1)))
    return dict(device='cpu',dtype='float64',threads=1,python=sys.version,torch=torch.__version__,platform=platform.platform(),
        cases=rows,source_rkn4_screen=dict(velocity_third_order_condition=float(bp@abar@c),required=1/6,
            velocity_decay_errors=errors,measured_order=math.log(errors[-2]/errors[-1],2),
            disposition='Only order two for velocity-dependent acceleration; same a,b,bp,c as Nystrom4VelocityIndependent under its restricted contract; excluded from factory'),
        qualification='Analytic underdamped modes and total mechanical-plus-thermal energy; finite horizon and fixed steps. High-order fine-grid measurements can reach rounding. Probe policy includes extra AD/structure RHS calls. No general SPH conservation, symplecticity or equal-accuracy speed claim.',
        source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (
            ROOT/'src/warpSPHIntegrators/nystrom.py',ROOT/'src/warpSPHIntegrators/_nystrom_data.py',
            ROOT/'src/warpSPHIntegrators/splitting.py',ROOT/'src/warpSPHIntegrators/run.py',
            ROOT/'src/warpSPHIntegrators/specs.py',ROOT/'examples/nystrom_demo.py',Path(__file__).resolve())})


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,default=Path('.tmp/phase8/nystrom-backend'))
    args=parser.parse_args();torch.set_num_threads(1);report=measurements()
    args.output_dir.mkdir(parents=True,exist_ok=True)
    (args.output_dir/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print('Measured',len(report['cases']),'known mechanics/thermal cases; RKN4 velocity order',report['source_rkn4_screen']['measured_order'])
