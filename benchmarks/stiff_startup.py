"""Phase 6 BDF startup comparison and order evidence on a stiff transient.

phi(t)=sin(4t), y'=-rate*(y-phi)+phi', y(0)=1. Exact solution is
phi(t)+exp(-rate*t). Default target rate=100, final time=1. BE warm-up is
an accuracy-loss control, not an available BDF startup mode.
"""
import argparse
import json
import math
from pathlib import Path
import platform
import sys
import time

import torch
from warpSPHIntegrators import TensorState,tensor_rhs,getIntegrator,StepHistory


def trajectory(name,h,rate,mode,end,frequency=4.):
    method=getIntegrator(name);needed=method.metadata.history_length
    state=TensorState(torch.ones(1,dtype=torch.float64));history=StepHistory(needed)
    f=tensor_rhs(lambda t,y:-rate*(y-math.sin(frequency*t))+frequency*math.cos(frequency*t))
    options=dict(matvec='jvp',tol=1e-12,newton_tol=1e-10)
    points=[];work=0;failures=0;floors=0;begin=time.perf_counter()
    def counted(system,dt,**kwargs):
        nonlocal work
        work+=1;return f(system,dt,**kwargs)
    for i in range(round(end/h)):
        scheme=getIntegrator('BDF1') if mode=='BE-control' and i<needed else method
        result=scheme(state,h,counted,history=history,
                      startup='explicit' if mode=='explicit' else 'implicit',solver_opts=options)
        failures+=int(not result.step_status.successful);floors+=int(result.step_status.precision_floor)
        state,history=result.state,result.history
        exact=math.sin(frequency*state.t)+math.exp(-rate*state.t)
        points.append(dict(t=state.t,y=state.y.item(),error=abs(state.y.item()-exact)))
    return dict(scheme=name,h=h,rate=rate,startup=mode,seconds=time.perf_counter()-begin,
        failures=failures,precision_floors=floors,rhs_calls=work,
        peak_amplitude=max([1.]+[abs(p['y']) for p in points]),
        final_error=points[-1]['error'],points=points)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,default=Path('.tmp/phase6/startup'))
    parser.add_argument('--rate',type=float,default=100.)
    parser.add_argument('--end',type=float,default=1.)
    parser.add_argument('--frequency',type=float,default=4.)
    parser.add_argument('--steps',type=float,nargs='+',default=[.1,.05,.025,.0125,.00625])
    args=parser.parse_args();torch.set_num_threads(1)
    if not all(math.isfinite(v) for v in (args.rate,args.end,args.frequency,*args.steps)) or args.rate<=0 or args.end<=0 or any(h<=0 or not math.isclose(round(args.end/h)*h,args.end) for h in args.steps):
        parser.error('positive rate/end/steps on a uniform endpoint grid required')
    args.output_dir.mkdir(parents=True,exist_ok=True)
    rows=[trajectory(name,h,args.rate,mode,args.end,args.frequency) for name in ('BDF3','BDF4','BDF5')
          for mode in ('implicit','explicit','BE-control') for h in args.steps]
    orders=[]
    for name in ('BDF3','BDF4','BDF5'):
        for mode in ('implicit','explicit','BE-control'):
            grid=sorted([r for r in rows if r['scheme']==name and r['startup']==mode],key=lambda r:r['h'],reverse=True)
            orders.append(dict(scheme=name,startup=mode,
                measured_order=math.log(grid[-2]['final_error']/grid[-1]['final_error'])/math.log(grid[-2]['h']/grid[-1]['h'])))
    report=dict(python=sys.version,torch=torch.__version__,platform=platform.platform(),
        device='cpu',dtype='float64',threads=1,rate=args.rate,end=args.end,steps=args.steps,
        problem=f"phi=sin({args.frequency}t), y'= -rate*(y-phi)+phi', y(0)=1",frequency=args.frequency,orders=orders,results=rows)
    (args.output_dir/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(orders,indent=2))


if __name__=='__main__':main()
