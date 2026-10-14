"""Known diffusion modes, nonlinear accuracy and adaptive work for stabilized RK."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import resource
import subprocess
import sys
import time

import torch
from warpSPHIntegrators import StabilizedRK,TensorState,tensor_rhs,solve

ROOT=Path(__file__).resolve().parents[1]
FAMILIES=('rkl1','rkl2','rkg1','rkg2')


def diffusion(n,family):
    torch.set_num_threads(1)
    x=torch.arange(1,n+1,dtype=torch.float64)/(n+1);dx=1/(n+1)
    low=torch.sin(math.pi*x);high=torch.sin(n*math.pi*x)
    low_rate=4/dx**2*math.sin(math.pi/(2*(n+1)))**2
    high_rate=4/dx**2*math.sin(n*math.pi/(2*(n+1)))**2
    exact=low*math.exp(-low_rate*.1)+.001*high*math.exp(-high_rate*.1)
    method=StabilizedRK(family,lambda_max=4/dx**2,max_stages=4096);calls=0
    def force(t,y):
        nonlocal calls
        calls+=1
        return (torch.cat((y[1:],y.new_zeros(1)))+torch.cat((y.new_zeros(1),y[:-1]))-2*y)/dx**2
    baseline=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss;start=time.perf_counter()
    result=solve(method,TensorState(low+.001*high),.1,tensor_rhs(force),dt=.01)
    seconds=time.perf_counter()-start
    peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    error=float((result.state.y-exact).abs().max())
    extra_mib=(peak-baseline)/1024
    passed=result.successful and error<(.02 if method.order==1 else .001) and extra_mib<=256
    return dict(n=n,family=family,stages=method.stage_count(.01),dt=.01,t_end=.1,
        successful=result.successful,rhs_calls=calls,seconds=seconds,max_absolute_error=error,
        additional_peak_rss_mib=extra_mib,forward_euler_stability_steps=math.ceil(.1*high_rate/2),
        gate_passed=passed)


def trajectory(family,h,tolerance=None):
    calls=0;method=StabilizedRK(family,5)
    def force(t,y):
        nonlocal calls
        calls+=1
        return -y+.2*y*y-.2*math.exp(-2*t)
    start=time.perf_counter()
    result=solve(method,TensorState(torch.ones(1,dtype=torch.float64)),1.,tensor_rhs(force),
        dt=h,adaptive=tolerance is not None,rtol=tolerance,atol=None if tolerance is None else tolerance*.01)
    return dict(family=family,dt=h,rtol=tolerance,successful=result.successful,
        steps=result.accepted_steps,rejected_steps=result.rejected_steps,rhs_calls=calls,
        absolute_error=abs(result.state.y.item()-math.exp(-1)),seconds=time.perf_counter()-start)


def measurements():
    env={**os.environ,'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1','CUDA_VISIBLE_DEVICES':''}
    def child(case):
        n,family=case
        result=subprocess.run([sys.executable,str(Path(__file__).resolve()),'--case',str(n),family],
            cwd=ROOT,env=env,capture_output=True,text=True,check=True)
        return json.loads(result.stdout)
    cases=[(n,family) for n in (64,1024,4096) for family in FAMILIES]
    with ThreadPoolExecutor(max_workers=3) as pool:scale=list(pool.map(child,cases))
    torch.set_num_threads(1)
    accuracy=[trajectory(family,h) for family in FAMILIES for h in (.1,.05,.025)]
    adaptive=[trajectory(family,.5,tol) for family in FAMILIES for tol in (1e-3,1e-5,1e-7)]
    return dict(device='cpu',dtype='float64',threads_per_case=1,concurrent_scale_cases=3,
        python=sys.version,torch=torch.__version__,platform=platform.platform(),
        scale=scale,accuracy=accuracy,adaptive=adaptive,
        scale_gate=dict(sizes=[64,1024,4096],additional_peak_rss_mib=256,
            max_error_order1=.02,max_error_order2=.001,all_passed=all(r['gate_passed'] for r in scale)),
        qualification='Fresh-process diffusion RSS subtracts pre-run high-water RSS. Euler comparison counts stability-limited steps, not equal-accuracy work. Supplied negative-real spectral bound; no nonnormal/complex-spectrum guarantee. Euler differences control local error; source norm/s filtering and FSAL are not reproduced. Stored stages/autograd graphs are not a low-storage execution path.',
        source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (
            ROOT/'src/warpSPHIntegrators/stabilized.py',Path(__file__).resolve())})


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case',nargs=2,metavar=('N','FAMILY'))
    parser.add_argument('--output-dir',type=Path,default=Path('.tmp/phase8/stabilized-backend'))
    args=parser.parse_args()
    if args.case:print(json.dumps(diffusion(int(args.case[0]),args.case[1])))
    else:
        report=measurements();args.output_dir.mkdir(parents=True,exist_ok=True)
        (args.output_dir/'report.json').write_text(json.dumps(report,indent=2)+'\n')
        print('Measured 12 fresh-process diffusion, 12 nonlinear accuracy and 12 adaptive cases')
        if not report['scale_gate']['all_passed']:raise SystemExit('Scale gate failed')
