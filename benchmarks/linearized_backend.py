"""Full-JVP Rosenbrock work on known nonlinear reaction and stiff decay modes."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import torch
from warpSPHIntegrators import Rosenbrock, TensorState, tensor_rhs, solve, getIntegrator


def measurements():
    rows=[]
    for n in (64,4096):
        for stiff in (False,True):
            rates=torch.logspace(0,4 if stiff else .5,n,dtype=torch.float64)
            for method in [*[Rosenbrock(m) for m in ['Rosenbrock23','Rosenbrock32','RosenbrockW6S4OS']],
                           getIntegrator('ESDIRK4(3)6L[2]SA')]:
                for h in (.125,.0625):
                    calls=0;iterations=0;diagnostic_rhs=0
                    def force(t,y):
                        nonlocal calls
                        calls+=1;target=torch.exp(-rates*t)
                        return -rates*y if stiff else -rates*y+.2*y*y-.2*target*target
                    def record(step):
                        nonlocal iterations,diagnostic_rhs
                        for d in step.result.solver_diagnostics:
                            if d is not None:iterations+=d.gmres_iterations;diagnostic_rhs+=d.rhs_evaluations
                    options=dict(gmres_tol=1e-12,gmres_maxiter=100,matvec='jvp',newton_tol=1e-7,
                        preconditioner=lambda v,state,context:v/(1+context['stage_coefficient']*rates))
                    start=time.perf_counter()
                    result=solve(method,TensorState(torch.ones(n,dtype=torch.float64)),1.,tensor_rhs(force),
                        dt=h,solver_opts=options,callback=record,raise_on_failure=False)
                    rows.append(dict(method=method.name,n=n,problem='stiff_decay' if stiff else 'nonlinear_forced',dt=h,
                        successful=result.successful,reached_time=result.state.t,termination=result.termination,
                        steps=result.accepted_steps,rhs_calls=calls,gmres_iterations=iterations,
                        diagnostic_rhs_evaluations=diagnostic_rhs,seconds=time.perf_counter()-start,
                        max_absolute_error=float((result.state.y-torch.exp(-rates*result.state.t)).abs().max()),
                        max_absolute_state=float(result.state.y.abs().max())))
    root=Path(__file__).resolve().parents[1]
    return dict(device='cpu',dtype='float64',threads=1,torch=torch.__version__,cases=rows,
        qualification='Known independent rates 1–sqrt(10) or 1–10000; fixed steps and horizon. Shifted diagonal preconditioner for every method. Actual RHS counts include isolated JVP/time probes and endpoint refresh; diagnostics counted separately. A successful stage solve is not proof of stability or accuracy: Rosenbrock32 can amplify the stiff modes. No equal-error or native-Warp speed claim.',
        source_sha256={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [
            root/'src/warpSPHIntegrators/linearized.py',root/'src/warpSPHIntegrators/_rosenbrock_data.py',
            root/'src/warpSPHIntegrators/jfnk.py',Path(__file__).resolve()]})


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();torch.set_num_threads(1);report=measurements()
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(report,indent=2)+'\n')
    print('Measured',len(report['cases']),'full-JVP Rosenbrock/implicit consumer cases')
