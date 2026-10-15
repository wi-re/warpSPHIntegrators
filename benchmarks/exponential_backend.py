"""Known reaction–diffusion forcing: split versus full-Jacobian exponential cost."""
import argparse,json,math,time,hashlib
from pathlib import Path
import torch
from warpSPHIntegrators.schemes import ExponentialRK
from warpSPHIntegrators import TensorState, tensor_rhs, solve, get_scheme
from warpSPHIntegrators.adapters import SemilinearRHS

def measurements():
    rows=[]
    for n in (16,128,1024):
        x=torch.arange(n,dtype=torch.float64)/n;initial=1+.2*torch.cos(2*torch.pi*x)
        def L(y):return .0001*n*n*(torch.roll(y,1)+torch.roll(y,-1)-2*y)
        for h in (.1,.05):
            methods=[ExponentialRK(m) for m in ('ETDRK2','ETDRK3','ETDRK4','HochOst4')]
            if n==16:methods += [ExponentialRK(m) for m in ('Exprb32','Exprb43','EPIRK5P1')]
            methods += [get_scheme('etd2rk')]
            for method in methods:
                counts=dict(full=0,linear=0)
                def linear(t,y):counts['linear']+=1;return L(y)
                def nonlinear(t,y):
                    counts['full']+=1;target=initial*math.exp(-t)
                    return .1*y*y-target-L(target)-.1*target*target
                split=SemilinearRHS(tensor_rhs(linear),tensor_rhs(nonlinear))
                start=time.perf_counter();result=solve(method,TensorState(initial),.5,split,dt=h,raise_on_failure=False)
                rows.append(dict(method=method.name,n=n,dt=h,successful=result.successful,
                    seconds=time.perf_counter()-start,steps=result.accepted_steps,counts=counts,
                    max_error=float((result.state.y-initial*math.exp(-.5)).abs().max())))
    root=Path(__file__).resolve().parents[1]
    return dict(device='cpu',dtype='float64',threads=1,torch=torch.__version__,cases=rows,
        qualification='Manufactured nonlinear time-dependent periodic reaction–diffusion, exact semi-discrete exp(-t) reference. N=16 uses dense full derivatives; N>=128 uses constant-L matrix-free Arnoldi. Wall time includes first-use operator preparation and quadrature. Fixed h comparisons have unequal errors; no universal speed claim.',
        source_sha256={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__).resolve(),root/'src/warpSPHIntegrators/schemes/families/exponential_rk.py',root/'src/warpSPHIntegrators/schemes/coefficients/exponential.py']})
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();torch.set_num_threads(1)
    report=measurements();a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,indent=2)+'\n');print(len(report['cases']),'cases')
