"""Known forced oscillator: corrected history accuracy and actual full-RHS work."""
import argparse
import json
import math
from pathlib import Path
import platform
import time

import torch
from warpSPHIntegrators import ImprovedNystrom, Nystrom, getIntegrator, solve, testing


def measurements():
    rows=[]
    methods=[ImprovedNystrom(p,'declared') for p in (3,4)]+[
        Nystrom('Nystrom4VelocityIndependent','declared'),getIntegrator('RK4'),getIntegrator('dormandPrince')]
    for n in (64,4096):
        omega=torch.linspace(.5,3,n,dtype=torch.float64)
        def exact(t):
            return torch.cos(omega*t)+.7*torch.sin(omega*t),omega*(-torch.sin(omega*t)+.7*torch.cos(omega*t))
        for nonlinear in (False,True):
            for method in methods:
                for h in (.125,.0625,.03125):
                    q,v=exact(0.)
                    state=testing.ParticleSystem(testing.ParticleState(q,v,torch.zeros_like(q),torch.ones_like(q)))
                    calls=0
                    def rhs(current,dt,**kwargs):
                        nonlocal calls
                        calls+=1;s=current.state
                        a=-omega.square()*s.x
                        if nonlinear:
                            target,_=exact(current.t)
                            a=a-.2*s.x**3+.2*target**3
                        return testing.ParticleUpdate(s.u,a,torch.zeros_like(s.e)),None
                    start=time.perf_counter();result=solve(method,state,1.,rhs,dt=h);seconds=time.perf_counter()-start
                    q,v=exact(1.)
                    rows.append(dict(n=n,nonlinear=nonlinear,method=method.name,dt=h,seconds=seconds,
                        rhs_calls=calls,successful=result.successful,steps=result.accepted_steps,
                        position_error=float((result.state.state.x-q).abs().max()),
                        velocity_error=float((result.state.state.u-v).abs().max())))
    return dict(device='cpu',dtype='float64',threads=1,torch=torch.__version__,platform=platform.platform(),cases=rows,
        qualification='Analytic independent vector modes, linear and nonlinear forced restoring acceleration. Declared velocity independence; probe overhead excluded. Warm IRKN3/4 need 2/3 full RHS calls, startup 5/6. Fixed grids and finite horizon; no symplecticity, damping, SPH conservation or general speed claim.')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();torch.set_num_threads(1);report=measurements()
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(report,indent=2)+'\n')
    print('Measured',len(report['cases']),'known oscillator trajectories')
