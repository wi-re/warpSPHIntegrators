"""Declared-bound explicit/implicit switching on an analytic varying-stiffness ODE."""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import platform
import sys
import time
import torch
from warpSPHIntegrators import AlgorithmSwitch,NDF,TensorState,tensor_rhs,solve,getIntegrator

ROOT=Path(__file__).resolve().parents[1]


def measurements():
    rows=[]
    for n in (64,4096):
        slow=torch.linspace(.1,2,n,dtype=torch.float64)
        for peak in (100.,1000.,10000.):
            fast=torch.logspace(0,math.log10(peak),n,dtype=torch.float64)
            def choose(state,previous):
                bound=1+(peak-1)*math.sin(math.pi*state.t)**2
                return int(bound>(10 if previous==1 else 30))
            methods=[AlgorithmSwitch(('dormandPrince','trbdf2'),choose,'declared-bound-trbdf-v1'),
                AlgorithmSwitch(('dormandPrince',NDF(3)),choose,'declared-bound-ndf-v1'),
                getIntegrator('dormandPrince'),getIntegrator('trbdf2'),NDF(3)]
            for tolerance in (1e-4,1e-7):
                for method in methods:
                    calls=0;choices=Counter();switches=0;previous=None;gmres=0
                    def force(t,y):
                        nonlocal calls
                        calls+=1;exact=torch.exp(-slow*t)
                        rate=1+(fast-1)*math.sin(math.pi*t)**2
                        return -rate*(y-exact)-slow*exact
                    def precondition(v,state,context):
                        rate=1+(fast-1)*math.sin(math.pi*context['stage_time'])**2
                        return v/(1+context['stage_coefficient']*rate)
                    def accepted(record):
                        nonlocal previous,switches,gmres
                        r=record.result
                        if isinstance(method,AlgorithmSwitch):
                            index=r.history.context['index'];choices[index]+=1
                            if previous is not None and previous!=index:switches+=1
                            previous=index
                        ds=r.solver_diagnostics;ds=ds if isinstance(ds,(tuple,list)) else [ds]
                        gmres+=sum(d.gmres_iterations for d in ds if d is not None)
                    start=time.perf_counter()
                    result=solve(method,TensorState(torch.ones_like(slow)),2.,tensor_rhs(force),dt=.02,
                        dt_max=.1,adaptive=True,rtol=tolerance,atol=tolerance*.01,callback=accepted,
                        solver_opts=dict(matvec='jvp',tol=1e-15,newton_tol=1e-9,preconditioner=precondition))
                    rows.append(dict(n=n,peak=peak,method=method.name,rtol=tolerance,successful=result.successful,
                        accepted=result.accepted_steps,rejected=result.rejected_steps,rhs_calls_including_retries=calls,
                        accepted_gmres_iterations=gmres,selection_counts=dict(sorted(choices.items())),switches=switches,
                        max_error=float((result.state.y-torch.exp(-2*slow)).abs().max()),seconds=time.perf_counter()-start))
    return dict(device='cpu',dtype='float64',threads=1,python=sys.version,torch=torch.__version__,platform=platform.platform(),
        cases=rows,qualification='Manufactured exp(-slow*t) solution with smooth time-varying diagonal stiffness and exact diagonal forward preconditioning. User policy uses a declared spectral bound and hysteresis, not inferred stiffness. RHS counts include retries; GMRES counts cover accepted solves. Same tolerances are not equal-accuracy comparisons. Histories cold-start on actual switches; source CompositeAlgorithm cache transfer is not reproduced.',
        source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (
            ROOT/'src/warpSPHIntegrators/switching.py',ROOT/'src/warpSPHIntegrators/run.py',Path(__file__).resolve())})


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,default=Path('.tmp/phase8/algorithm-switch-backend'))
    args=parser.parse_args();torch.set_num_threads(1);report=measurements()
    args.output_dir.mkdir(parents=True,exist_ok=True)
    (args.output_dir/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    assert all(r['successful'] for r in report['cases'])
    print('Measured',len(report['cases']),'known varying-stiffness cases')
