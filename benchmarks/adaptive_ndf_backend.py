"""Adaptive NDF order/work behavior on understood nonlinear and stiff vectors."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import platform
import sys
import time
import torch
from warpSPHIntegrators.schemes import AdaptiveNDF, NDF
from warpSPHIntegrators import TensorState, tensor_rhs, solve, get_scheme

ROOT=Path(__file__).resolve().parents[1]


def measurements():
    rows=[]
    for n in (64,4096):
        for problem in ('reaction','stiff_decay'):
            nonlinear=problem=='reaction';end=4. if nonlinear else 1.
            rates=torch.linspace(.1,10,n,dtype=torch.float64) if nonlinear else torch.logspace(0,4,n,dtype=torch.float64)
            methods=[AdaptiveNDF(),NDF(5),get_scheme('KenCarp5')]
            if nonlinear:methods.append(get_scheme('dormandPrince'))
            for tolerance in (1e-4,1e-6,1e-8):
                for method in methods:
                    calls=0;gmres=0;warm=0;orders=Counter();records=0
                    def force(t,y):
                        nonlocal calls
                        calls+=1
                        return -rates*y-.05*y**3+.05*torch.exp(-3*rates*t) if nonlinear else -rates*y
                    def precondition(v,state,context):
                        diagonal=rates+.15*state.y.square() if nonlinear else rates
                        return v/(1+context['stage_coefficient']*diagonal)
                    def accepted(record):
                        nonlocal gmres,warm,records
                        records+=1;result=record.result
                        ds=result.solver_diagnostics
                        ds=ds if isinstance(ds,(list,tuple)) else [ds]
                        gmres+=sum(d.gmres_iterations for d in ds if d is not None)
                        if isinstance(method,(NDF,AdaptiveNDF)):
                            warm+=int(not result.quadrature.bootstrap)
                            orders[result.history.context.get('last_order',method.order)]+=1
                    start=time.perf_counter()
                    result=solve(method,TensorState(torch.ones_like(rates)),end,tensor_rhs(force),dt=.001,
                        adaptive=True,rtol=tolerance,atol=tolerance*.01,callback=accepted,
                        solver_opts=dict(matvec='jvp',tol=1e-15,newton_tol=1e-10,preconditioner=precondition))
                    rows.append(dict(n=n,problem=problem,method=method.name,rtol=tolerance,
                        successful=result.successful,accepted=result.accepted_steps,rejected=result.rejected_steps,
                        warm_steps=warm,actual_target_order_counts=dict(sorted(orders.items())),
                        accepted_gmres_iterations=gmres,rhs_calls_including_retries=calls,
                        max_error=float((result.state.y-torch.exp(-rates*end)).abs().max()),
                        seconds=time.perf_counter()-start))
    return dict(device='cpu',dtype='float64',threads=1,python=sys.version,torch=torch.__version__,platform=platform.platform(),
        cases=rows,qualification='Analytic manufactured nonlinear reaction and diagonal stiff decay with exact diagonal forward preconditioning. RHS counts include retries; accepted GMRES counts omit rejected solves. Weighted local-defect tolerances do not promise final global errors or equal-accuracy speed. This accepted-order policy uses source proposal mathematics with different maturity, hold, bootstrap and retry semantics from QNDF.',
        source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (
            ROOT/'src/warpSPHIntegrators/schemes/families/adaptive_ndf.py',ROOT/'src/warpSPHIntegrators/schemes/families/ndf.py',
            ROOT/'src/warpSPHIntegrators/runtime/run.py',Path(__file__).resolve())})


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,default=Path('.tmp/phase8/adaptive-ndf-backend'))
    args=parser.parse_args();torch.set_num_threads(1);report=measurements()
    args.output_dir.mkdir(parents=True,exist_ok=True)
    (args.output_dir/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    assert all(r['successful'] for r in report['cases'])
    print('Measured',len(report['cases']),'adaptive vector cases')
