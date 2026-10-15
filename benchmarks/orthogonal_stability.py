"""Offline spectral screening: dense grids plus polynomial critical points.

Critical points are recovered from a Chebyshev interpolation on the candidate
interval, then evaluated by the original recurrence. This is a numerical gate,
not an exact-arithmetic interval proof or nonlinear/nonnormal stability claim.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.fft import dct

from warpSPHIntegrators._orthogonal_program import (available_stages,sizing_bound,
    recurrence_program,_parameters)


def scan():
    rows=[]
    for name in ['ROCK2','ROCK4','SERK2','RKMC2']:
        for stages in available_stages(name):
            bound=sizing_bound(name,stages);parameters=_parameters(name,stages)
            def evaluate(z):
                return 1+recurrence_program(name,stages,parameters,
                    lambda q,endpoint=False:z*(1+q),np.zeros_like(z))[0]
            # All source RKMC counts are sampled; the table-driven families
            # additionally have every interior critical point checked.
            z=-np.linspace(0,bound,257 if name=='RKMC2' else 4097)
            values=evaluate(z);critical_max=None;critical_count=0
            if name!='RKMC2':
                count=2*(stages+1);x=np.cos(np.pi*(np.arange(count)+.5)/count)
                coefficients=dct(evaluate((x-1)*bound/2),type=2)/count;coefficients[0]/=2
                roots=np.polynomial.chebyshev.chebroots(np.polynomial.chebyshev.chebder(coefficients[:stages+1]))
                points=roots.real[(abs(roots.imag)<1e-8)&(abs(roots.real)<1)]
                critical_count=len(points)
                critical_max=float(np.max(np.abs(evaluate((np.r_[-1.,1.,points]-1)*bound/2))))
            maximum=max(float(np.max(np.abs(values))),critical_max or 0.)
            assert maximum<=1+1e-8,(name,stages,maximum)
            rows.append(dict(method=name,stages=stages,bound=bound,
                grid_points=len(z),max_abs_grid=float(np.max(np.abs(values))),
                min_grid_value=float(np.min(values)),critical_points=critical_count,
                max_abs_critical=critical_max))
    root=Path(__file__).resolve().parents[1]
    return dict(numpy=np.__version__,cases=rows,
        qualification='All 46 ROCK2 / 48 ROCK4 / 11 SERK2 / 998 RKMC2 supported configurations. ROCK/SERK include numerical polynomial critical points; RKMC is grid sampled. ROCK4 source counts 133/152 excluded after actual-source interior instability controls. ROCK4/SERK sizing includes .95/.9 margins. Numerical normal negative-real screening, not an exact arithmetic, nonnormal or nonlinear stability certificate.',
        source_sha256={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [
            root/'src/warpSPHIntegrators/_orthogonal_program.py',root/'src/warpSPHIntegrators/_orthogonal_data.py',Path(__file__).resolve()]})


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();report=scan();args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+'\n');print('Screened',len(report['cases']),'supported stage configurations')
