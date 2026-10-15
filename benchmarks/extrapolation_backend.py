"""Fixed/adaptive extrapolation work on known reaction and rotation vectors."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import sys
import time
import torch
from warpSPHIntegrators.schemes import Extrapolation
from warpSPHIntegrators import TensorState, tensor_rhs, solve, get_scheme

ROOT=Path(__file__).resolve().parents[1]


def measurements():
    methods=[Extrapolation('euler',p) for p in (4,6,8)]+[Extrapolation('midpoint',p) for p in (4,6,8,10)]
    methods+=[get_scheme('RK4'),get_scheme('dormandPrince')]
    rows=[];adaptive=[]
    for n in (64,4096):
        rates=torch.linspace(.1,10,n,dtype=torch.float64)
        for problem in ('reaction','rotation'):
            initial=torch.ones_like(rates) if problem=='reaction' else torch.stack((torch.ones_like(rates),torch.zeros_like(rates)))
            exact=torch.exp(-rates) if problem=='reaction' else torch.stack((torch.cos(rates),torch.sin(rates)))
            for method in methods:
                trials=[dict(dt=h) for h in (.2,.1,.05)]
                if n==4096 and method.has_error_estimate:
                    trials+=[dict(dt=.2,adaptive=True,rtol=tol,atol=tol*.01) for tol in (1e-5,1e-8,1e-11)]
                for options in trials:
                    calls=0
                    def force(t,y):
                        nonlocal calls
                        calls+=1
                        if problem=='reaction':return -rates*y-.05*y**3+.05*torch.exp(-3*rates*t)
                        return torch.stack((-rates*y[1],rates*y[0]))
                    start=time.perf_counter()
                    result=solve(method,TensorState(initial),1.,tensor_rhs(force),**options)
                    row=dict(n=n,problem=problem,method=method.name,**options,successful=result.successful,
                        accepted=result.accepted_steps,rejected=result.rejected_steps,rhs_calls=calls,
                        max_error=float((result.state.y-exact).abs().max()),seconds=time.perf_counter()-start)
                    (adaptive if options.get('adaptive') else rows).append(row)
    return dict(device='cpu',dtype='float64',threads=1,python=sys.version,torch=torch.__version__,platform=platform.platform(),
        cases=rows,adaptive=adaptive,qualification='Analytic nonlinear manufactured reaction and real planar rotation vectors. Finite horizon, fixed target orders and serial columns. RHS counts include refresh/retries. No internal parallelism, low-storage, symplecticity or universal work advantage; high-order fine grids can reach floating-point floors. Compare actual errors, not tolerances alone.',
        source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (
            ROOT/'src/warpSPHIntegrators/schemes/families/extrapolation.py',ROOT/'src/warpSPHIntegrators/runtime/run.py',Path(__file__).resolve())})


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,default=Path('.tmp/phase8/extrapolation-backend'))
    args=parser.parse_args();torch.set_num_threads(1);report=measurements()
    args.output_dir.mkdir(parents=True,exist_ok=True)
    (args.output_dir/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    assert all(r['successful'] for r in report['cases']+report['adaptive'])
    print('Measured',len(report['cases']),'fixed and',len(report['adaptive']),'adaptive cases')
