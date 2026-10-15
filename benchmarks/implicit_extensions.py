"""Known reaction and stiff-decay consumers for MEBDF2/CNLF2."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import torch
from warpSPHIntegrators import TensorState, tensor_rhs, solve

ROOT=Path(__file__).resolve().parents[1]


def measurements():
    rows=[]
    for n in (64,1024):
        for stiff in (False,True):
            rate=100. if stiff else 1.
            for name in ('mebdf2','cnlf2','sdirk2','trapezoidal'):
                for h in (.1,.05):
                    calls=0;iterations=0
                    def force(t,y):
                        nonlocal calls
                        calls+=1
                        return -rate*y if stiff else -y+.2*y*y-.2*torch.exp(torch.tensor(-2*t,dtype=y.dtype))
                    def record(step):
                        nonlocal iterations
                        diagnostics=step.result.solver_diagnostics
                        for diag in diagnostics if isinstance(diagnostics,(tuple,list)) else (diagnostics,):
                            if diag is not None:iterations+=diag.gmres_iterations
                    start=time.perf_counter()
                    result=solve(name,TensorState(torch.ones(n,dtype=torch.float64)),1.,tensor_rhs(force),dt=h,
                        callback=record,raise_on_failure=False,
                        solver_opts=dict(matvec='jvp',tol=1e-15,newton_tol=1e-8))
                    exact=torch.exp(torch.tensor(-rate,dtype=torch.float64)).item()
                    rows.append(dict(method=name,n=n,problem='stiff_decay' if stiff else 'nonlinear_forced',
                        dt=h,successful=result.successful,termination=result.termination,
                        reached_time=result.state.t,steps=result.accepted_steps,
                        rhs_calls=calls,gmres_iterations=iterations,seconds=time.perf_counter()-start,
                        max_absolute_error=float((result.state.y-exact).abs().max())))
    return dict(device='cpu',dtype='float64',threads=1,cases=rows,
        qualification='Finite horizon and constant steps; actual RHS entrypoint calls. CNLF2 retains a parasitic stiff mode and is not a stiff-decay recommendation.',
        source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (
            ROOT/'src/warpSPHIntegrators/schemes/definitions/dirk.py',ROOT/'src/warpSPHIntegrators/drivers/history/imex.py',Path(__file__).resolve())})


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,default=Path('.tmp/phase8/implicit-extensions'))
    args=parser.parse_args();torch.set_num_threads(1);report=measurements()
    args.output_dir.mkdir(parents=True,exist_ok=True)
    (args.output_dir/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print('Measured',len(report['cases']),'known-consumer cases')
