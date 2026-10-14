"""Reproducible Phase 6 solver gate; each configuration runs in a fresh process.

Run python benchmarks/implicit_scale.py --output-dir .tmp/phase6/scale
Defaults compare 2/3-stage Gauss/Radau and SDIRK2 on nonlinear reaction and
mesh-refined reaction-diffusion. CPU float64, one Torch thread, fixed dt.
The predeclared registration gate requires every stage-block s=3 step to meet
its tolerance, <=120 Krylov iterations/step, <=3x matching s=2 median work and
time, and <=256 MiB additional peak process RSS through N=1024. Failure keeps
wave 9 gated. Peak RSS includes Torch/native memory, not only Python allocations.
"""
import argparse
import hashlib
from pathlib import Path
import json
import math
import platform
import os
import resource
import statistics
import subprocess
import sys
import time

import torch
from warpSPHIntegrators import (TensorState,tensor_rhs,getIntegrator,
    stage_block_preconditioner,flatten_integrated)
from collocation_prototypes import prototype_scheme


def problem(n,kind):
    rates=torch.logspace(0,4,n,dtype=torch.float64)
    diffusion=.2 if kind=='reaction' else .02*n*n
    x=torch.arange(n,dtype=torch.float64)/n
    y=1+.2*torch.sin(2*math.pi*x)+.1*torch.cos(46*math.pi*x)
    def rhs(t,y):
        return -rates*(y+.01*y**3)+diffusion*(y.roll(1)-2*y+y.roll(-1))+math.cos(t)
    def diagonal(state,context):
        if hasattr(state,'states'):
            return torch.stack([-rates*(1+.03*s.y**2)-2*diffusion for s in state.states])
        return -rates*(1+.03*state.y**2)-2*diffusion
    return TensorState(y),tensor_rhs(rhs),diagonal


def case(args):
    torch.set_num_threads(1)
    initial,rhs,diagonal=problem(args.n,args.problem)
    scheme=prototype_scheme(args.scheme) if args.scheme.endswith('3') else getIntegrator({
        'gauss2':'Gauss-Legendre 2','radau2':'Radau IIA s=2','sdirk2':'SDIRK2'}[args.scheme])
    stages=3 if args.scheme.endswith('3') else 2
    opts=dict(matvec='jvp',tol=1e-10,newton_tol=1e-6,gmres_maxiter=120,gmres_restart=30,max_iterations=6)
    if args.mode=='block':
        opts['preconditioner']=stage_block_preconditioner(diagonal)
    elif args.mode=='diagonal':
        def prec(v,state,context):
            d=diagonal(state,context)
            if hasattr(state,'states'):
                a=torch.as_tensor(context['tableau_a'],dtype=v.dtype).diagonal()
                inverse=1/(1-args.dt*a[:,None]*d)
            else:
                inverse=1/(1-context['stage_coefficient']*d)
            return v*inverse.reshape(-1)
        opts['preconditioner']=prec
    counters=[];times=[];fails=0;work=[];solutions=[];residuals=[];newtons=[]
    # Initialize native thread pools before the measured baseline. Each case is
    # a fresh process; ru_maxrss is its actual high-water mark, not a Python proxy.
    torch.linalg.solve(torch.eye(3,dtype=torch.float64),torch.ones(3,dtype=torch.float64))
    baseline=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024
    for _ in range(args.repeats):
        state=TensorState(initial.y.clone());prior=None
        count=[0]
        def counted(system,dt,**kwargs):
            count[0]+=1
            return rhs(system,dt,**kwargs)
        for step in range(args.steps):
            keywords=dict(solver_opts=opts)
            if args.warm and prior is not None:
                keywords['warmStart']=prior[0] if args.scheme=='sdirk2' else prior
            before=count[0];start=time.perf_counter()
            result=scheme(state,args.dt,counted,**keywords)
            elapsed=time.perf_counter()-start
            diags=[d for d in result.solver_diagnostics if d is not None]
            good=result.step_status.successful and all(d.termination=='tolerance' for d in diags)
            fails+=int(not good)
            times.append(elapsed);work.append(count[0]-before)
            # Block diagnostics historically scale Krylov iterations by s;
            # actual iterations and measured RHS calls are recorded separately.
            counters.append(sum(d.gmres_iterations for d in diags)/(1 if args.scheme=='sdirk2' else stages))
            newtons.append(sum(d.rhs_evaluations for d in diags)/(1 if args.scheme=='sdirk2' else stages))
            residuals.append(max(d.residual for d in diags))
            state=result.state;prior=result.stages
            if not good:break
        solutions.append(state.y.tolist())
    peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024
    return dict(scheme=args.scheme,problem=args.problem,n=args.n,unknowns=stages*args.n,
        mode=args.mode,warm=args.warm,steps_attempted=len(times),failures=fails,
        median_seconds=statistics.median(times),total_seconds=sum(times),
        median_rhs_calls=statistics.median(work),rhs_calls=work,krylov_iterations=counters,
        newton_map_evaluations=newtons,residuals=residuals,
        peak_rss_mib=peak,baseline_peak_rss_mib=baseline,additional_peak_rss_mib=max(0,peak-baseline),
        final_solution=solutions[-1])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,default=Path('.tmp/phase6/scale'))
    parser.add_argument('--sizes',type=int,nargs='+',default=[64,256,1024])
    parser.add_argument('--steps',type=int,default=3)
    parser.add_argument('--repeats',type=int,default=2)
    parser.add_argument('--dt',type=float,default=.05)
    parser.add_argument('--case',action='store_true')
    parser.add_argument('--scheme',default='radau3');parser.add_argument('--problem',default='reaction')
    parser.add_argument('--mode',default='block');parser.add_argument('--n',type=int,default=64)
    parser.add_argument('--warm',action='store_true')
    args=parser.parse_args()
    if any(n<2 for n in args.sizes) or args.steps<1 or args.repeats<1 or not math.isfinite(args.dt) or args.dt<=0:
        parser.error('positive dt/step/repeat controls and sizes >=2 required')
    if args.case:
        print(json.dumps(case(args)));return
    args.output_dir.mkdir(parents=True,exist_ok=True)
    env=os.environ.copy()
    for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
        env[name]='1'
    env['CUDA_VISIBLE_DEVICES']=''
    configs=[(s,m,w) for s in ('gauss2','radau2','gauss3','radau3')
             for m,w in (('none',False),('diagonal',False),('block',False),('block',True))]
    configs += [('sdirk2','none',False),('sdirk2','diagonal',False),('sdirk2','diagonal',True)]
    results=[]
    for kind in ('reaction','diffusion'):
        for n in args.sizes:
            for scheme,mode,warm in configs:
                command=[sys.executable,str(Path(__file__).resolve()),'--case','--scheme',scheme,
                    '--problem',kind,'--mode',mode,'--n',str(n),'--dt',str(args.dt),
                    '--steps',str(args.steps),'--repeats',str(args.repeats)]
                if warm:command.append('--warm')
                process=subprocess.run(command,env=env,text=True,capture_output=True)
                if process.returncode:
                    raise RuntimeError(process.stderr)
                record=json.loads(process.stdout);results.append(record)
                print(f'{kind} N={n} {scheme}/{mode}/warm={warm}: failures={record["failures"]}, '
                      f'krylov={max(record["krylov_iterations"]):g}, {record["median_seconds"]:.3f}s',flush=True)
                (args.output_dir/'cases.json').write_text(json.dumps(results,indent=2)+'\n')
    gate=[]
    for record in results:
        if record['scheme'] not in ('gauss3','radau3') or record['mode']!='block' or record['warm']:
            continue
        base=next(r for r in results if r['scheme']==record['scheme'].replace('3','2') and r['n']==record['n']
                  and r['problem']==record['problem'] and r['mode']=='block' and not r['warm'])
        ratio=record['median_seconds']/base['median_seconds'];work=record['median_rhs_calls']/base['median_rhs_calls']
        passed=(record['failures']==0 and max(record['krylov_iterations'])<=120 and
                ratio<=3 and work<=3 and record['additional_peak_rss_mib']<=256 and base['failures']==0)
        gate.append(dict(scheme=record['scheme'],n=record['n'],problem=record['problem'],passed=passed,
                         time_ratio=ratio,work_ratio=work,failures=record['failures']))
    passed=bool(gate) and all(r['passed'] for r in gate) and max(args.sizes)>=1024
    for row in results:
        row['failure_rate']=row['failures']/row['steps_attempted']
        values=row.pop('final_solution')
        row['solution_sha256']=hashlib.sha256(json.dumps(values).encode()).hexdigest()
        row['max_abs_final_solution']=max(abs(v) for v in values)
    report=dict(python=sys.version,torch=torch.__version__,platform=platform.platform(),
        cpu_model=next(line.split(':',1)[1].strip() for line in Path('/proc/cpuinfo').read_text().splitlines() if line.startswith('model name')),
        measurement_git_head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        source_state='Phase 6 working tree, before implementation commits',device='cpu',dtype='float64',threads=1,
        sizes=args.sizes,dt=args.dt,steps=args.steps,repeats=args.repeats,
        memory_definition='fresh-process ru_maxrss, baseline after native thread-pool initialization; additional peak includes first-use AD/library allocation',
        definition='all block/cold s=3 cases: zero tolerance failures, <=120 Krylov iterations per step, '
        '<=3x corresponding s=2 time and RHS work, <=256 MiB additional peak RSS, through N>=1024',
        passed=passed,gate=gate,results=results)
    (args.output_dir/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(f'Wave 9 scale gate passed: {passed}')


if __name__=='__main__':main()
