"""NDF work/accuracy on known nonlinear reaction and stiff decay vectors."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import platform
import sys
import time

import numpy as np
import torch
from warpSPHIntegrators.schemes import NDF
from warpSPHIntegrators import TensorState, tensor_rhs, solve, get_scheme
from warpSPHIntegrators.schemes.coefficients.ndf import ndf_difference_transform

ROOT=Path(__file__).resolve().parents[1]


def measurements():
    rows=[]
    methods=[NDF(p) for p in range(1,6)]+[get_scheme(n) for n in ('BDF3','BDF5','KenCarp5')]
    for n in (64,4096):
        for problem in ('reaction','stiff_decay'):
            rates=torch.linspace(.1,10,n,dtype=torch.float64) if problem=='reaction' else torch.logspace(0,4,n,dtype=torch.float64)
            nonlinear=problem=='reaction'
            for method in methods:
                for grid in ('constant','blocks'):
                    for count in (24,48):
                        calls=0;gmres=0;bootstraps=0;converged=True
                        def force(t,y):
                            nonlocal calls
                            calls+=1
                            return -rates*y-.05*y**3+.05*torch.exp(-3*rates*t) if nonlinear else -rates*y
                        def precondition(v,state,context):
                            diagonal=rates+.15*state.y.square() if nonlinear else rates
                            return v/(1+context['stage_coefficient']*diagonal)
                        options=dict(matvec='jvp',tol=1e-15,newton_tol=1e-9,preconditioner=precondition)
                        state=TensorState(torch.ones_like(rates));history=None;start=time.perf_counter()
                        for i in range(count):
                            h=(.75 if i//6%2==0 else 1.25)/count if grid=='blocks' else 1/count
                            # Fixed BDF's warm driver predates protected context;
                            # supply its derivative weight explicitly. DIRK starters
                            # and NDF always replace this with their actual shift.
                            beta=60/137 if method.name=='BDF5' else 6/11 if method.name=='BDF3' else 1.
                            result=method(state,h,tensor_rhs(force),history=history,
                                solver_opts={**options,'preconditioner_context':{'stage_coefficient':beta*h}})
                            converged=converged and result.step_status.tolerance_converged
                            ds=result.solver_diagnostics
                            ds=ds if isinstance(ds,(tuple,list)) else [ds]
                            gmres+=sum(d.gmres_iterations for d in ds if d is not None)
                            bootstraps+=int(result.quadrature.bootstrap if isinstance(method,NDF) else method.metadata.history_length>0 and len(result.stages)>1)
                            state,history=result.state,result.history
                        rows.append(dict(n=n,problem=problem,method=method.name,grid=grid,steps=count,
                            seconds=time.perf_counter()-start,rhs_calls=calls,gmres_iterations=gmres,
                            bootstrap_steps=bootstraps,tolerance_converged=converged,
                            finite=bool(torch.isfinite(state.y).all()),t=state.t,
                            max_error=float((state.y-torch.exp(-rates*state.t)).abs().max())))
    adaptive=[]
    for p in range(1,6):
        for tolerance in (1e-3,1e-5,1e-7):
            calls=0;records=[]
            def force(t,y):
                nonlocal calls
                calls+=1
                return -y+.2*y*y-.2*math.exp(-2*t)
            start=time.perf_counter()
            result=solve(NDF(p),TensorState(torch.ones(1,dtype=torch.float64)),2.,tensor_rhs(force),dt=2.,
                adaptive=True,rtol=tolerance,atol=tolerance*.01,
                solver_opts=dict(matvec='jvp',tol=1e-15,newton_tol=1e-10),callback=records.append)
            adaptive.append(dict(order=p,rtol=tolerance,seconds=time.perf_counter()-start,rhs_calls=calls,
                accepted=result.accepted_steps,rejected=result.rejected_steps,successful=result.successful,
                warm_steps=sum(not r.result.quadrature.bootstrap for r in records),
                final_error=abs(result.state.y.item()-math.exp(-2))))
    p=5;harmonic=np.cumsum(1/np.arange(1,p+1))
    advance=np.triu(np.ones((p,p)))-NDF(p).beta*np.ones((p,1))*harmonic
    up=np.asarray(ndf_difference_transform(p,5/3)).T;down=np.asarray(ndf_difference_transform(p,3/5)).T
    return dict(device='cpu',dtype='float64',threads=1,python=sys.version,torch=torch.__version__,platform=platform.platform(),
        cases=rows,adaptive=adaptive,alternating_step_counterexample=dict(order=5,ratio=5/3,
            per_step_spectral_radius=float(max(abs(np.linalg.eigvals(advance@up@advance@down)))),
            six_step_block_spectral_radius=float(max(abs(np.linalg.eigvals(np.linalg.matrix_power(advance,6)@up@np.linalg.matrix_power(advance,6)@down))))),
        qualification='Fixed target orders with quasi-constant accepted-step blocks. Nonlinear analytic manufactured reaction and diagonal stiff decay, finite horizon, JVP and exact diagonal forward preconditioning. Equal-grid work, not equal-accuracy speed or SPH/DAE claims. Local defects do not guarantee final global tolerances. No variable-order QNDF controller equivalence.',
        source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (
            ROOT/'src/warpSPHIntegrators/schemes/families/ndf.py',ROOT/'src/warpSPHIntegrators/runtime/run.py',ROOT/'src/warpSPHIntegrators/schemes/definitions/dirk.py',
            ROOT/'src/warpSPHIntegrators/runtime/checkpoint.py',Path(__file__).resolve())})


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,default=Path('.tmp/phase8/ndf-backend'))
    args=parser.parse_args();torch.set_num_threads(1);report=measurements()
    args.output_dir.mkdir(parents=True,exist_ok=True)
    (args.output_dir/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    assert all(r['finite'] and r['tolerance_converged'] for r in report['cases'])
    assert all(r['successful'] for r in report['adaptive'])
    print('Measured',len(report['cases']),'vector cases and',len(report['adaptive']),'adaptive refinements')
