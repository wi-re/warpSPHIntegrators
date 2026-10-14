"""Backend Adams accuracy/work on known vector reaction and adaptive trajectories."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import time

import torch
from warpSPHIntegrators import VariableAdams, TensorState, tensor_rhs, getIntegrator, StepHistory, solve

ROOT=Path(__file__).resolve().parents[1]


def measurements():
    rows=[]
    methods=[VariableAdams(p,kind) for p in (3,4,5) for kind in ('ab','abm')]
    methods+=[getIntegrator('ab5'),getIntegrator('abm5'),getIntegrator('dormandPrince')]
    for n in (64,16384):
        rates=torch.linspace(.1,5.,n,dtype=torch.float64)
        for alternating in (False,True):
            for method in methods:
                calls=0;state=TensorState(torch.ones(n,dtype=torch.float64));history=StepHistory(5)
                def force(t,y):
                    nonlocal calls
                    calls+=1
                    return -rates*y-.05*y**3+.05*torch.exp(-3*rates*t)
                rhs=tensor_rhs(force);startup=0;start=time.perf_counter()
                for i in range(80):
                    h=(.5 if i%2==0 else 1.5)/80 if alternating else 1/80
                    result=method(state,h,rhs,history=history)
                    if not result.step_status.successful:raise RuntimeError(method.name)
                    startup+=int(method.metadata.history_length>0 and len(result.stages)>3)
                    state,history=result.state,result.history or history
                elapsed=time.perf_counter()-start
                exact=torch.exp(-rates)
                rows.append(dict(method=method.name,n=n,grid='alternating' if alternating else 'constant',
                    steps=80,startup_steps=startup,rhs_calls=calls,
                    max_absolute_error=float((state.y-exact).abs().max()),seconds=elapsed))
    adaptive=[]
    for method in methods[:6]:
        for tolerance in (1e-4,1e-6,1e-8):
            calls=0;exponents=[]
            def force(t,y):
                nonlocal calls
                calls+=1
                return -y+.2*y*y-.2*math.exp(-2*t)
            start=time.perf_counter()
            result=solve(method,TensorState(torch.ones(1,dtype=torch.float64)),2.,tensor_rhs(force),
                dt=1.,adaptive=True,rtol=tolerance,atol=tolerance*.01,
                callback=lambda record:exponents.append(record.result.estimate_order))
            if not result.successful:raise RuntimeError(method.name)
            adaptive.append(dict(method=method.name,rtol=tolerance,steps=result.accepted_steps,
                rejected_steps=result.rejected_steps,rhs_calls=calls,
                absolute_error=abs(result.state.y.item()-math.exp(-2)),seconds=time.perf_counter()-start,
                estimate_exponents=sorted(set(exponents))))
    return dict(device='cpu',dtype='float64',threads=1,vector_cases=rows,adaptive_cases=adaptive,
        qualification='Tolerance controls a local difference estimate, not a guaranteed global error. Fixed-coefficient Adams deliberately restart on alternating h. Variable AB/ABM re-evaluate the start and corrected endpoint, costing 2/3 RHS calls after startup.',
        source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in
            (ROOT/'src/warpSPHIntegrators/variable_adams.py',ROOT/'src/warpSPHIntegrators/run.py',Path(__file__).resolve())})


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,default=Path('.tmp/phase8/variable-adams'))
    args=parser.parse_args();torch.set_num_threads(1);report=measurements()
    args.output_dir.mkdir(parents=True,exist_ok=True)
    (args.output_dir/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print('Measured',len(report['vector_cases']),'vector cases and',len(report['adaptive_cases']),'adaptive cases')
