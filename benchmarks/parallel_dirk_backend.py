"""Serial PDIRK44 work/accuracy versus existing implicit methods on known modes."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import torch
from warpSPHIntegrators.schemes import ParallelDIRK
from warpSPHIntegrators import TensorState, tensor_rhs, solve, get_scheme
from warpSPHIntegrators.solvers import stage_block_preconditioner


def measurements():
    rows=[]
    for n in (64,4096):
        for stiff in (False,True):
            rates=torch.logspace(0,4 if stiff else .5,n,dtype=torch.float64)
            for method in (ParallelDIRK(),get_scheme('sdirk2'),get_scheme('Radau IIA s=2'),get_scheme('ESDIRK4(3)6L[2]SA')):
                for h in (.125,.0625):
                    calls=0;gmres=0;newton_rhs=0
                    def force(t,y):
                        nonlocal calls
                        calls+=1
                        target=torch.exp(-rates*t)
                        return -rates*y if stiff else -rates*y+.2*y*y-.2*target*target
                    def record(step):
                        nonlocal gmres,newton_rhs
                        diagnostics=step.result.solver_diagnostics
                        for d in diagnostics if isinstance(diagnostics,(list,tuple)) else (diagnostics,):
                            if d is not None:gmres+=d.gmres_iterations;newton_rhs+=d.rhs_evaluations
                    start=time.perf_counter()
                    result=solve(method,TensorState(torch.ones(n,dtype=torch.float64)),1.,tensor_rhs(force),dt=h,
                        solver_opts=dict(matvec='jvp',tol=1e-12,newton_tol=1e-7,gmres_maxiter=100,
                            preconditioner=lambda v,state,context:v/(1+context['stage_coefficient']*rates)) if method.name!='Radau IIA s=2' else
                        dict(matvec='jvp',tol=1e-12,newton_tol=1e-7,gmres_maxiter=100,
                            preconditioner=stage_block_preconditioner(-rates)),
                        callback=record,raise_on_failure=False)
                    rows.append(dict(method=method.name,n=n,problem='stiff_decay' if stiff else 'nonlinear_forced',
                        dt=h,successful=result.successful,termination=result.termination,reached_time=result.state.t,
                        steps=result.accepted_steps,rhs_calls=calls,gmres_iterations=gmres,
                        diagnostic_rhs_evaluations=newton_rhs,seconds=time.perf_counter()-start,
                        max_absolute_error=float((result.state.y-torch.exp(-rates*result.state.t)).abs().max())))
    root=Path(__file__).resolve().parents[1]
    return dict(device='cpu',dtype='float64',threads=1,torch=torch.__version__,cases=rows,
        qualification='First-order ODEs with independent rates 1–sqrt(10) or 1–10000. Serial PDIRK stage execution; no stage-threading speed claim. Shifted diagonal preconditioning (block for Radau). Actual RHS entries include refresh/probes; solver diagnostics are reported separately. Finite horizon and fixed steps, not equal-error comparisons.',
        source_sha256={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [
            root/'src/warpSPHIntegrators/schemes/families/parallel_dirk.py',root/'src/warpSPHIntegrators/schemes/definitions/dirk.py',Path(__file__).resolve()]})


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();torch.set_num_threads(1);report=measurements()
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(report,indent=2)+'\n')
    print('Measured',len(report['cases']),'implicit first-order ODE consumers')
