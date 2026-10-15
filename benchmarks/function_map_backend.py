"""Direct analytic maps and rate integrators on known spiral/heat behavior."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import sys
import time
import torch
from warpSPHIntegrators.schemes import FunctionMap
from warpSPHIntegrators.adapters import tensor_map
from warpSPHIntegrators import TensorState, tensor_rhs, solve, get_scheme

ROOT=Path(__file__).resolve().parents[1]


def measurements():
    rows=[];recurrences=[]
    for n in (64,4096):
        omega=torch.linspace(.1,10,n,dtype=torch.float64);gamma=torch.linspace(.1,1,n,dtype=torch.float64)
        initial=torch.stack((torch.ones_like(omega),torch.zeros_like(omega),torch.zeros_like(omega)))
        exact=torch.stack((torch.exp(-gamma)*torch.cos(omega),-torch.exp(-gamma)*torch.sin(omega),.5*(1-torch.exp(-2*gamma))))
        for method in (FunctionMap(),get_scheme('RK4'),get_scheme('dormandPrince')):
            for h in (.2,.1,.05):
                calls=0
                def mapping(t,y,dt):
                    nonlocal calls
                    calls+=1;c=torch.cos(omega*dt);s=torch.sin(omega*dt);decay=torch.exp(-gamma*dt)
                    q=decay*(c*y[0]+s*y[1]);v=decay*(c*y[1]-s*y[0])
                    heat=y[2]+(y[0].square()+y[1].square()-q.square()-v.square())/2
                    return torch.stack((q,v,heat))
                def force(t,y):
                    nonlocal calls
                    calls+=1
                    return torch.stack((omega*y[1]-gamma*y[0],-omega*y[0]-gamma*y[1],gamma*(y[0].square()+y[1].square())))
                chosen=tensor_map(mapping) if isinstance(method,FunctionMap) else tensor_rhs(force)
                start=time.perf_counter();result=solve(method,TensorState(initial),1.,chosen,dt=h)
                y=result.state.y;budget=(y[0].square()+y[1].square())/2+y[2]-.5
                rows.append(dict(n=n,method=method.name,dt=h,successful=result.successful,
                    steps=result.accepted_steps,map_calls=calls if isinstance(method,FunctionMap) else 0,
                    rhs_calls=0 if isinstance(method,FunctionMap) else calls,seconds=time.perf_counter()-start,
                    max_error=float((y-exact).abs().max()),max_heat_error=float((y[2]-exact[2]).abs().max()),
                    max_energy_defect=float(budget.abs().max())))
        a=torch.linspace(.8,.99,n,dtype=torch.float64);b=torch.linspace(.01,.1,n,dtype=torch.float64)
        for h in (.2,.1,.05):
            calls=0
            def affine(t,y,dt):
                nonlocal calls
                calls+=1;return a*y+b*torch.exp(torch.tensor(-t,dtype=y.dtype))
            result=solve(FunctionMap(),TensorState(torch.ones_like(a)),1.,tensor_map(affine),dt=h)
            count=round(1/h);r=torch.exp(torch.tensor(-h,dtype=a.dtype));delta=torch.log(a/r)
            total=torch.where(delta.abs()<1e-12,torch.full_like(delta,count),torch.expm1(count*delta)/torch.expm1(delta))
            reference=a**count+b*r**count*total
            recurrences.append(dict(n=n,dt=h,map_calls=calls,steps=result.accepted_steps,successful=result.successful,
                max_error=float((result.state.y-reference).abs().max())))
    return dict(device='cpu',dtype='float64',threads=1,python=sys.version,torch=torch.__version__,platform=platform.platform(),
        cases=rows,affine_recurrences=recurrences,qualification='Analytic damped planar rotation with generated heat and exact affine discrete recurrences. Map calls return complete states and cost more than a derivative evaluation; entry counts alone are not speed comparisons. The analytic map is exact for this understood model, not a generic exact ODE solver. No adaptive/error/order claim for direct maps.',
        source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (
            ROOT/'src/warpSPHIntegrators/schemes/families/function_map.py',ROOT/'src/warpSPHIntegrators/runtime/run.py',
            ROOT/'src/warpSPHIntegrators/state/specs.py',Path(__file__).resolve())})


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,default=Path('.tmp/phase8/function-map-backend'))
    args=parser.parse_args();torch.set_num_threads(1);report=measurements()
    args.output_dir.mkdir(parents=True,exist_ok=True)
    (args.output_dir/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    assert all(r['successful'] for r in report['cases']+report['affine_recurrences'])
    assert all(r['max_error']<1e-12 for r in report['cases'] if r['map_calls'])
    assert all(r['max_error']<1e-12 for r in report['affine_recurrences'])
    print('Measured',len(report['cases']),'spiral/heat and',len(report['affine_recurrences']),'affine-map cases')
