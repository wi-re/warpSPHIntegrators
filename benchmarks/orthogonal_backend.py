"""Periodic heat modes: known smoothing, constant mode, spectral sizing and cost."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import time

import torch
from warpSPHIntegrators import OrthogonalRK, StabilizedRK, TensorState, tensor_rhs, solve, getIntegrator


def measurements():
    rows=[]
    for n in [64,1024,4096]:
        nu=.001;rho=4*nu*n*n;x=torch.arange(n,dtype=torch.float64)/n
        modes=[torch.ones_like(x),.25*torch.cos(2*torch.pi*x),.1*torch.cos(14*torch.pi*x),.005*torch.cos(n*torch.pi*x)]
        rates=[0.,4*nu*n*n*math.sin(math.pi/n)**2,
                4*nu*n*n*math.sin(7*math.pi/n)**2,rho]
        y=sum(modes);end=.1
        exact=sum(mode*math.exp(-rate*end) for mode,rate in zip(modes,rates))
        configurations=[]
        for requested in [.01,.005]:
            for name in ['ROCK2','ROCK4','SERK2','RKMC2']:
                configurations.append((OrthogonalRK(name,lambda_max=rho),requested,requested))
            for name in ['rkl2','rkg2']:
                configurations.append((StabilizedRK(name,lambda_max=rho),requested,requested))
        configurations.append((getIntegrator('RK4'),min(.01,.9*2.785/rho),.01))
        for method,h,requested in configurations:
            calls=0
            def force(t,y):
                nonlocal calls
                calls+=1
                return nu*n*n*(torch.roll(y,1)+torch.roll(y,-1)-2*y)
            start=time.perf_counter()
            result=solve(method,TensorState(y),end,tensor_rhs(force),dt=h,raise_on_failure=False)
            rows.append(dict(method=method.name,n=n,requested_dt=requested,actual_dt=h,
                stages=method.stages_for(h) if isinstance(method,OrthogonalRK) else None,
                successful=result.successful,steps=result.accepted_steps,reached_time=result.state.t,
                rhs_calls=calls,seconds=time.perf_counter()-start,
                max_absolute_error=float((result.state.y-exact).abs().max()),
                mean_drift=float(abs(result.state.y.mean()-y.mean())),
                extrema=[float(result.state.y.min()),float(result.state.y.max())]))
    root=Path(__file__).resolve().parents[1]
    return dict(device='cpu',dtype='float64',threads=1,torch=torch.__version__,cases=rows,
        qualification='Symmetric negative-real periodic Laplacian, Fourier constant/slow/mid/Nyquist modes, exact semi-discrete reference. Includes stage selection and first-use coefficient preparation. RK4 chooses a smaller spectrally stable dt when needed; comparisons are not equal-error. No nonlinear SSP/TVD or SPH conservation claim.',
        source_sha256={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [
            root/'src/warpSPHIntegrators/orthogonal.py',root/'src/warpSPHIntegrators/_orthogonal_program.py',
            root/'src/warpSPHIntegrators/_orthogonal_data.py',Path(__file__).resolve()]})


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();torch.set_num_threads(1);report=measurements()
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(report,indent=2)+'\n')
    print('Measured',len(report['cases']),'known heat-mode cases')
