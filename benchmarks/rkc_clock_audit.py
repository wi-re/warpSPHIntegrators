"""Reproduce the pre-fix RKC/RKL kernel and compare consistent stage clocks.

Historical numerical functions/coefficients are compiled from a pinned local
Git revision against the current shared state helpers. This is a kernel audit,
not a recreation of the complete historical Python/dependency environment.
"""
import argparse
import ast
import dataclasses
import hashlib
import json
import math
from pathlib import Path
import subprocess
import types

import torch
from torch.profiler import record_function
from warpSPHIntegrators import TensorState, tensor_rhs, get_scheme
import warpSPHIntegrators.drivers.stabilized.chebyshev as rkc
from warpSPHIntegrators.state import get_reference_state
from warpSPHIntegrators.state.fields import field_behavior
from warpSPHIntegrators.state import IntegrationResult, StageResult
from warpSPHIntegrators.state.lifecycle import initializeSystem, finalizeSystem, updateStep, reject_prior_step
from warpSPHIntegrators.state import updateStateEuler

ROOT=Path(__file__).resolve().parents[1]
BASELINE='f6c0edf'


def baseline_module():
    source=subprocess.check_output(['git','show',BASELINE+':src/warpSPHIntegrators/rkc.py'],cwd=ROOT,text=True)
    selected=[];assignments={'_COEFFS','_NAMES','_K','_SMIN','RKC1','RKC2','RKL2'}
    for node in ast.parse(source).body:
        if isinstance(node,ast.FunctionDef):selected.append(node)
        elif isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id in assignments for t in node.targets):
            selected.append(node)
    module=types.ModuleType('_historical_rkc_kernel')
    module.__dict__.update({name:globals()[name] for name in ['dataclasses','record_function',
        'initializeSystem','finalizeSystem','updateStep','updateStateEuler','reject_prior_step',
        'get_reference_state','field_behavior','IntegrationResult','StageResult']})
    exec(compile(ast.Module(selected,type_ignores=[]),'<pinned-rkc-kernel>','exec'),module.__dict__)
    return module,hashlib.sha256(source.encode()).hexdigest()


def measurements():
    old,old_hash=baseline_module();rows=[]
    for name in ['RKC1','RKC2','RKL2']:
        for version,method in [('before',getattr(old,name)),('after',get_scheme(name))]:
            for problem in ['time_force','nonlinear_forced']:
                errors=[]
                for n in [20,40,80]:
                    state=TensorState(torch.zeros(1,dtype=torch.float64) if problem=='time_force' else torch.ones(1,dtype=torch.float64))
                    force=(lambda t,y:torch.full_like(y,t)) if problem=='time_force' else (lambda t,y:-y+.2*y*y-.2*math.exp(-2*t))
                    for _ in range(n):state=method(state,1/n,tensor_rhs(force),s=5).state
                    exact=.5 if problem=='time_force' else math.exp(-1)
                    errors.append(abs(state.y.item()-exact))
                order=math.log2(errors[-2]/errors[-1]) if min(errors[-2:])>1e-12 else None
                rows.append(dict(method=name,version=version,problem=problem,steps=[20,40,80],errors=errors,order=order))
    return dict(baseline_revision=BASELINE,baseline_source_sha256=old_hash,
        current_source_sha256=hashlib.sha256(Path(rkc.__file__).read_bytes()).hexdigest(),cases=rows,
        qualification='Actual historical coefficient/driver functions with current shared state helpers; scalar forcing and nonlinear exact exp(-t) references. Null order when component errors reach the roundoff floor. No full historical environment or performance comparison.')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();torch.set_num_threads(1);report=measurements()
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(report,indent=2)+'\n')
    print('Recorded',len(report['cases']),'before/after known-problem refinements')
