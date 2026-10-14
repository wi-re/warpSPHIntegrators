"""Known-solution accuracy/work comparison for the first wave-9 registrations."""
import argparse
import json
import math
from pathlib import Path
import time
import torch
from warpSPHIntegrators import TensorState,tensor_rhs,getIntegrator


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,default=Path('.tmp/phase8/firk-accuracy'))
    args=parser.parse_args();args.output_dir.mkdir(parents=True,exist_ok=True);torch.set_num_threads(1)
    rows=[]
    for name in ('Gauss-Legendre 2','Gauss-Legendre 3','Radau IIA s=2','Radau IIA s=3'):
        for problem in ('oscillator','nonlinear'):
            errors=[]
            for h in ((.2,.1,.05) if problem=='oscillator' else (.2,.1,.05,.025)):
                state=TensorState(torch.tensor([1.,0.] if problem=='oscillator' else [1.],dtype=torch.float64))
                calls=[0]
                def f(t,y):
                    calls[0]+=1
                    return torch.stack((y[1],-4*y[0])) if problem=='oscillator' else -3*y**3
                rhs=tensor_rhs(f);iterations=0;start=time.perf_counter()
                end=2. if problem=='oscillator' else .4
                for _ in range(round(end/h)):
                    result=getIntegrator(name)(state,h,rhs,solver_opts=dict(matvec='jvp',tol=1e-13,newton_tol=1e-9))
                    if not result.step_status.tolerance_converged:raise RuntimeError(f'{name} {problem} did not converge')
                    iterations+=sum(d.gmres_iterations for d in result.solver_diagnostics)/3 if name in ('Gauss-Legendre 3','Radau IIA s=3') else sum(d.gmres_iterations for d in result.solver_diagnostics)/2
                    state=result.state
                seconds=time.perf_counter()-start
                exact=torch.tensor([math.cos(4),-2*math.sin(4)] if problem=='oscillator' else [1/math.sqrt(1+6*end)],dtype=torch.float64)
                error=float((state.y-exact).abs().sum());errors.append(error)
                rows.append(dict(method=name,problem=problem,h=h,endpoint=end,error=error,rhs_calls=calls[0],krylov_iterations=iterations,seconds=seconds))
            rows[-1]['measured_order']=math.log(errors[-2]/errors[-1],2)
    report=dict(device='cpu',dtype='float64',threads=1,solver_opts=dict(matvec='jvp',tol=1e-13,newton_tol=1e-9),rows=rows)
    (args.output_dir/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    for row in rows:print(row)


if __name__=='__main__':main()
