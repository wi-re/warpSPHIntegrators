"""Known nonlinear/oscillatory accuracy and full RHS work for backend factories."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import time

import torch
from warpSPHIntegrators import Collocation, TensorState, tensor_rhs

ROOT=Path(__file__).resolve().parents[1]


def measurements():
    rows=[]
    for family in ('gauss','radau'):
        for stages in (2,3,4,5,7,9):
            method=Collocation(family,stages)
            for problem in ('oscillator','nonlinear'):
                results=[]
                for h in (.5,.25,.125):
                    end=2.;calls=0
                    state=TensorState(torch.tensor([1.,0.] if problem=='oscillator' else [1.],dtype=torch.float64))
                    def force(t,y):
                        nonlocal calls
                        calls+=1
                        return torch.stack((y[1],-16*y[0])) if problem=='oscillator' else -3*y**3
                    start=time.perf_counter();iterations=0;floors=0
                    for _ in range(round(end/h)):
                        result=method(state,h,tensor_rhs(force),solver_opts=dict(matvec='jvp',tol=1e-13,newton_tol=1e-9))
                        if not result.step_status.successful:raise RuntimeError((family,stages,problem,h,result.step_status))
                        floors+=int(result.step_status.precision_floor)
                        iterations+=result.solver_diagnostics[0].gmres_iterations/stages
                        state=result.state
                    exact=torch.tensor([math.cos(8),-4*math.sin(8)] if problem=='oscillator' else [1/math.sqrt(13)],dtype=torch.float64)
                    error=float((state.y-exact).abs().sum())
                    results.append(dict(h=h,error=error,rhs_calls=calls,krylov_iterations=iterations,
                        precision_floor_steps=floors,seconds=time.perf_counter()-start))
                pairs=[(a,b) for a,b in zip(results,results[1:]) if min(a['error'],b['error'])>1e-11
                       and not a['precision_floor_steps'] and not b['precision_floor_steps']]
                measured=None if not pairs else math.log(pairs[-1][0]['error']/pairs[-1][1]['error'],2)
                rows.append(dict(family=family,stages=stages,propagated_order=method.order,problem=problem,
                    measured_order=measured,qualification='Observational strict-solve accuracy/work run. Precision-floor steps are recorded; no order fit for affected/floored pairs. Coarse grids may be preasymptotic.',results=results))
    return dict(device='cpu',dtype='float64',threads=1,rows=rows,
        source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in
                      (ROOT/'src/warpSPHIntegrators/collocation.py',Path(__file__).resolve())})


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,default=Path('.tmp/phase8/collocation-accuracy'))
    args=parser.parse_args();torch.set_num_threads(1);report=measurements()
    args.output_dir.mkdir(parents=True,exist_ok=True)
    (args.output_dir/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print('Measured',len(report['rows']),'consumer/refinement configurations')
