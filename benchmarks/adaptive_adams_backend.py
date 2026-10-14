"""Adaptive Adams order/accuracy/work on manufactured reaction and damped rotation."""
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
from warpSPHIntegrators import AdaptiveAdams,VariableAdams,TensorState,tensor_rhs,solve,getIntegrator
ROOT=Path(__file__).resolve().parents[1]


def measurements():
    rows=[]
    methods=[AdaptiveAdams(p) for p in (3,4,5)]+[VariableAdams(p,'abm') for p in (3,4,5)]+[getIntegrator('dormandPrince')]
    for n in (64,4096):
        rates=torch.linspace(.1,3.,n,dtype=torch.float64)
        for problem in ('reaction','spiral_heat'):
            initial=torch.ones(n,dtype=torch.float64) if problem=='reaction' else torch.stack((torch.ones_like(rates),torch.zeros_like(rates),torch.zeros_like(rates)))
            exact=torch.exp(-4*rates) if problem=='reaction' else torch.stack((math.exp(-.8)*torch.cos(4*rates),-math.exp(-.8)*torch.sin(4*rates),torch.full_like(rates,.5*(1-math.exp(-1.6)))))
            for method in methods:
                for tol in (1e-5,1e-8):
                    calls=0;counts=Counter();transitions=[];previous=None
                    def force(t,y):
                        nonlocal calls
                        calls+=1
                        if problem=='reaction':return -rates*y+.05*y*y-.05*torch.exp(-2*rates*t)
                        q,v,heat=y
                        return torch.stack((rates*v-.2*q,-rates*q-.2*v,.2*(q*q+v*v)))
                    def observe(record):
                        nonlocal previous
                        c=record.result.history.context or {};p=c.get('last_order',method.order)
                        counts[p]+=1
                        if previous!=p:transitions.append(dict(t=record.result.state.t,order=p));previous=p
                    start=time.perf_counter()
                    result=solve(method,TensorState(initial.clone()),4.,tensor_rhs(force),dt=.02,
                        adaptive=True,rtol=tol,atol=tol*.01,callback=observe)
                    elapsed=time.perf_counter()-start
                    assert result.successful
                    values=result.state.y
                    row=dict(n=n,problem=problem,method=method.name,rtol=tol,atol=tol*.01,
                        successful=result.successful,accepted=result.accepted_steps,rejected=result.rejected_steps,
                        rhs_entries=calls,seconds=elapsed,max_error=float((values-exact).abs().max()),
                        accepted_order_counts=dict(counts),order_transitions=transitions)
                    if problem=='spiral_heat':row.update(heat_error=float((values[2]-exact[2]).abs().max()),
                        total_energy_defect=float(((values[0]**2+values[1]**2)/2+values[2]-.5).abs().max()))
                    rows.append(row)
    cpuinfo=Path('/proc/cpuinfo')
    model=next((line.split(':',1)[1].strip() for line in cpuinfo.read_text().splitlines() if line.startswith('model name')),platform.processor()) if cpuinfo.exists() else platform.processor()
    return dict(cpu_model=model,device='cpu',dtype='float64',threads=1,python=sys.version,torch=torch.__version__,platform=platform.platform(),
        cases=rows,qualification='Known nonlinear forced reaction and damped rotation/thermal budget, T=4, rates .1–3. RHS entries include rejected attempts. Equal tolerances are not equal-accuracy speed comparisons. No stiff/SPH or complete source-controller equivalence claim.',
        source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (Path(__file__).resolve(),ROOT/'src/warpSPHIntegrators/adaptive_adams.py',ROOT/'src/warpSPHIntegrators/variable_adams.py',ROOT/'src/warpSPHIntegrators/run.py')})


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,default=ROOT/'.tmp/phase8/adaptive-adams-backend')
    args=parser.parse_args();torch.set_num_threads(1);report=measurements()
    args.output_dir.mkdir(parents=True,exist_ok=True);(args.output_dir/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print('Measured',len(report['cases']),'known adaptive consumer cases')
