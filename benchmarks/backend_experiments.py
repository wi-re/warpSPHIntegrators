"""Cost and memory experiments on understood tensor problems; no promotion claims."""
import argparse,json,math,time,hashlib
from pathlib import Path
import numpy as np
import torch
from warpSPHIntegrators.schemes import ExpandedRK
from warpSPHIntegrators import TensorState, tensor_rhs, get_scheme
from warpSPHIntegrators.experiments import *

def measurements():
    rows=[]
    for n in (16,1024):
        for order in (3,4):
            calls=0;y=torch.ones(n,dtype=torch.float64)
            def f(t,y):
                nonlocal calls
                calls+=1;return -y+.1*y*y-.1*torch.exp(-2*t)
            start=time.perf_counter();value=y
            for j in range(16):value=taylor_step(f,j/16,value,1/16,order)
            rows.append(dict(kind='Taylor',n=n,order=order,rhs_calls=calls,seconds=time.perf_counter()-start,error=float((value-math.exp(-1)).abs().max())))
    for n in (16,4096):
        initial=torch.ones(n,dtype=torch.float64)
        # Slow periodic FFT convolution is a dense nonlocal operation; fast
        # diagonal relaxation is local. Both share exact manufactured exp(-t).
        spectral=1/(1+torch.arange(n//2+1,dtype=torch.float64)**2)
        def smoothing(y):return torch.fft.irfft(spectral*torch.fft.rfft(y),n=n)
        for ratio in (4,16):
            for mode in ('multirate','all_micro'):
                counts=dict(slow=0,fast=0)
                def slow(t,y):counts['slow']+=1;return .1*smoothing(y)+initial*math.exp(-t)
                def fast(t,y):counts['fast']+=1;return -2.1*y
                start=time.perf_counter();value=initial
                from warpSPHIntegrators.experiments.tensor import _rk4
                for j in range(16):
                    if mode=='multirate':value=multirate_step(slow,fast,j/16,value,1/16,ratio)
                    else:
                        for k in range(ratio):value=_rk4(lambda t,y:slow(t,y)+fast(t,y),(j+k/ratio)/16,value,1/(16*ratio))
                rows.append(dict(kind=mode,n=n,ratio=ratio,counts=counts,seconds=time.perf_counter()-start,error=float((value-math.exp(-1)).abs().max())))
    for base in ('euler','midpoint'):
        for order in (2,4,6):
            calls=0;value=torch.ones(2,dtype=torch.float64)
            def f(t,y):
                nonlocal calls
                calls+=1;return -y+.1*y*y-.1*torch.exp(-2*t)
            start=time.perf_counter()
            for j in range(8):value=implicit_extrapolation_step(f,j/8,value,1/8,order,base)
            rows.append(dict(kind='implicit_extrapolation',base=base,order=order,rhs_calls=calls,seconds=time.perf_counter()-start,error=float((value-math.exp(-1)).abs().max())))
    # Actual live CPU allocation high water from profiler memory deltas; excludes
    # preallocated input, allocator reservation, profiler/RSS and external memory.
    for n in (4096,262144):
        y=torch.ones(n,dtype=torch.float64)
        f=lambda t,y:-y-.1*y*y+.2*t
        for mode in ('full_stage','native_tensor'):
            with torch.no_grad():
                with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU],profile_memory=True) as prof:
                    result=ExpandedRK()(TensorState(y),.1,tensor_rhs(f)).state.y if mode=='full_stage' else low_storage_ck54_step(f,0.,y,.1)
            events=sorted((e for e in prof.events() if e.self_cpu_memory_usage),key=lambda e:e.time_range.start)
            live=peak=0
            for e in events:live+=e.self_cpu_memory_usage;peak=max(peak,live)
            rows.append(dict(kind='allocation',n=n,mode=mode,peak_tracked_cpu_bytes=peak,result_norm=float(result.norm())))
    # High-order monomial reconstruction is poorly conditioned even on fixed
    # grids. This is a warning against extending the current cardinal backend
    # blindly; it is not a failure claim about source divided differences.
    conditioning=[]
    for p in range(3,13):
        nodes=-np.arange(p,dtype=float)
        conditioning.append(dict(order=p,monomial_condition=float(np.linalg.cond(np.vander(nodes,increasing=True)))))
    root=Path(__file__).resolve().parents[1]
    return dict(cases=rows,conditioning=conditioning,torch=torch.__version__,device='cpu',dtype='float64',threads=1,
        qualification='Tensor-only prototypes. Time includes AD dispatch and all per-step dense preparations. Profiler peak is live tracked CPU allocations, not process RSS/device reservation. Native tensor recurrence omits full-state stages/finalize/AD contracts; no production speed or universal memory ratio claim.',
        source_sha256={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__).resolve(),root/'src/warpSPHIntegrators/experiments/tensor.py']})
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();torch.set_num_threads(1)
    r=measurements();a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(r,indent=2)+'\n');print(len(r['cases']),'experiments')
