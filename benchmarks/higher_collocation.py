"""Backend-only higher-stage scaling on understood reaction/diffusion fields."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import statistics
import subprocess
import sys
import time

import torch
from warpSPHIntegrators.schemes import Collocation
from implicit_scale import problem
from spatial_preconditioners import spatial_preconditioner

ROOT=Path(__file__).resolve().parents[1]


def case(args):
    torch.set_num_threads(1)
    state,rhs,diagonal=problem(args.n,args.problem)
    diffusion=.2 if args.problem=='reaction' else .02*args.n**2
    method=Collocation(args.family,args.stages);metrics={}
    opts=dict(matvec='jvp',tol=1e-10,newton_tol=1e-6,gmres_maxiter=120,
              gmres_restart=30,max_iterations=6,
              preconditioner=spatial_preconditioner(diagonal,diffusion,'schur_ilu',metrics))
    baseline=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    elapsed=[];iterations=[];calls=[];failures=[]
    for i in range(3):
        start=time.perf_counter();result=method(state,.01,rhs,solver_opts=opts)
        elapsed.append(time.perf_counter()-start)
        if not result.step_status.tolerance_converged:failures.append(result.step_status.termination)
        diagnostics=result.solver_diagnostics[0]
        iterations.append(diagnostics.gmres_iterations/args.stages)
        # Solver counters are stage-scaled and exclude the refreshed block and
        # optional null stage. Include that endpoint work for this comparison.
        calls.append(diagnostics.rhs_evaluations+diagnostics.gmres_iterations+args.stages+
                     method.metadata.estimate_extra_stages)
        state=result.state
    return dict(family=args.family,stages=args.stages,order=method.order,problem=args.problem,n=args.n,
        failures=failures,iterations=iterations,rhs_calls=calls,
        median_seconds=statistics.median(elapsed[1:]),
        additional_peak_mib=(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss-baseline)/1024,
        factorization=metrics)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case',action='store_true');parser.add_argument('--family',default='gauss')
    parser.add_argument('--stages',type=int,default=4);parser.add_argument('--problem',default='diffusion')
    parser.add_argument('--n',type=int,default=1024);parser.add_argument('--workers',type=int,default=3)
    parser.add_argument('--output-dir',type=Path,default=Path('.tmp/phase8/higher-collocation'))
    args=parser.parse_args()
    if args.case:print(json.dumps(case(args)));return
    args.output_dir.mkdir(parents=True,exist_ok=True)
    env=os.environ.copy()
    env.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',CUDA_VISIBLE_DEVICES='')
    configurations=[(family,stages,kind,n) for family in ('gauss','radau') for stages in (3,4,5,7,9)
                    for kind in ('reaction','diffusion') for n in (64,256,1024)]
    def launch(config):
        family,stages,kind,n=config
        command=[sys.executable,str(Path(__file__).resolve()),'--case','--family',family,
                 '--stages',str(stages),'--problem',kind,'--n',str(n)]
        process=subprocess.run(command,env=env,text=True,capture_output=True,check=True)
        return json.loads(process.stdout)
    records=[]
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for result in pool.map(launch,configurations):
            records.append(result)
            print(f'{result["family"]} s={result["stages"]} {result["problem"]} N={result["n"]}: '
                  f'{max(result["iterations"]):g} iterations; failures={result["failures"]}',flush=True)
    for row in records:
        baseline=next(r for r in records if r['family']==row['family'] and r['stages']==3 and
                      r['problem']==row['problem'] and r['n']==row['n'])
        row['rhs_ratio_per_stage']=statistics.median(row['rhs_calls'])/row['stages']/(statistics.median(baseline['rhs_calls'])/3)
        row['passed']=not row['failures'] and max(row['iterations'])<=120 and row['additional_peak_mib']<=256 and row['rhs_ratio_per_stage']<=3
    report=dict(device='cpu',dtype='float64',numerical_threads=1,parallel_cases=args.workers,
        contract='Each step meets tolerance; <=120 Krylov iterations; <=256 MiB additional RSS; <=3x s=3 RHS work per stage. Timings include concurrent-case contention.',
        qualification='Concrete backend stencil assembly and SciPy benchmark-only real Schur / shifted ILU; no frontend dependency. Stage count is parameterized; source coefficients verified for every supported count.',
        passed=all(r['passed'] for r in records),cases=records,
        source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in
                      (ROOT/'src/warpSPHIntegrators/schemes/families/collocation.py',Path(__file__).resolve())})
    (args.output_dir/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    if not report['passed']:raise SystemExit('Higher-stage backend scale gate failed; inspect report')


if __name__=='__main__':main()
