"""Measure the package driver against the independent dense structural consumer."""
import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
import time

import torch
from warpSPHIntegrators.schemes import GeneralizedAlpha
from warpSPHIntegrators import get_scheme, solve
from warpSPHIntegrators.state import get_reference_state
import warpSPHIntegrators.testing as testing

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('alpha_consumer',ROOT/'examples/structural_modes.py')
demo=importlib.util.module_from_spec(spec);sys.modules[spec.name]=demo;spec.loader.exec_module(demo)
OPTIONS=dict(matvec='jvp',tol=1e-12,newton_tol=1e-7)


def run(model,method,h,end):
    initial=model.initial()
    state=testing.ParticleSystem(testing.ParticleState(initial.q.clone(),initial.v.clone(),
        torch.zeros_like(initial.q),torch.ones_like(initial.q)))
    rhs_calls=0
    def force(system,dt,**kwargs):
        nonlocal rhs_calls
        rhs_calls+=1
        s=get_reference_state(system)
        return testing.ParticleUpdate(s.u,model.acceleration(s.x,s.u),s.e*0),None
    diagnostics=[]
    def observe(record):diagnostics.append(record.result.solver_diagnostics)
    start=time.perf_counter()
    result=solve(method,state,end,force,dt=h,save_steps=True,callback=observe,solver_opts=OPTIONS)
    seconds=time.perf_counter()-start
    if not result.successful:raise RuntimeError(result.termination)
    states=[]
    for sample in result.samples:
        s=get_reference_state(sample.state)
        states.append(demo.ModeState(s.x,s.u,model.acceleration(s.x,s.u),sample.t))
    return states,result,dict(seconds=seconds,steps=result.accepted_steps,
        gmres_iterations=sum(d.gmres_iterations for d in diagnostics),
        rhs_evaluations=rhs_calls,
        solver_reported_rhs_evaluations=sum(d.rhs_evaluations for d in diagnostics),
        all_tolerance=all(d.termination=='tolerance' for d in diagnostics))


def measurements():
    model=demo.two_mass_problem();behavior=[]
    for rho in (0.,.5,.8,1.):
        actual,result,work=run(model,GeneralizedAlpha(rho),.1,12.)
        dense=demo.trajectory(model,.1,12.,rho)
        parity=max(float((a.q-b.q).abs().max()) for a,b in zip(actual,dense))
        velocity=max(float((a.v-b.v).abs().max()) for a,b in zip(actual,dense))
        acceleration=float((result.history.latest.state.acceleration-dense[-1].a).abs().max())
        assert parity<1e-8 and velocity<1e-7 and acceleration<1e-6 and work['all_tolerance'], (rho,parity,velocity,acceleration,work)
        behavior.append(dict(rho_inf=rho,max_displacement_difference=parity,
            max_velocity_difference=velocity,endpoint_algorithmic_acceleration_difference=acceleration,
            modal_energy_retained=(model.modal_energy(actual[-1])/model.modal_energy(actual[0])).tolist(),**work))
    _,_,newmark=run(model,get_scheme('Newmark'),.1,12.)
    order=[];resolved=demo.two_mass_problem(slow=1.,fast=3.)
    for rho in (0.,.5,1.):
        errors=[];grid=(.1,.05,.025,.0125,.00625)
        for h in grid:
            states,_,_=run(resolved,GeneralizedAlpha(rho),h,2.)
            q,v=resolved.exact(2.)
            errors.append(float(torch.linalg.vector_norm(states[-1].q-q)+torch.linalg.vector_norm(states[-1].v-v)))
        measured=math.log(errors[-2]/errors[-1],2)
        assert abs(measured-2)<.08
        order.append(dict(rho_inf=rho,dt=grid,errors=errors,measured_order=measured))
    return dict(dtype='float64',device='cpu',threads=1,solver_options=OPTIONS,behavior=behavior,
        registered_newmark_work=newmark,order=order,
        qualification='Same physical consumer, timestep and JVP solver tolerances; generalized-alpha solves augmented state/acceleration and makes two RHS evaluations per map. Tiny filtered modal energies reflect numerical floors.',
        source_sha256={str(path.relative_to(ROOT)):hashlib.sha256(path.read_bytes()).hexdigest() for path in (
            ROOT/'src/warpSPHIntegrators/schemes/families/generalized_alpha.py',ROOT/'examples/structural_modes.py',Path(__file__).resolve())})


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,default=Path('.tmp/phase8/generalized-alpha'))
    args=parser.parse_args();torch.set_num_threads(1);report=measurements()
    args.output_dir.mkdir(parents=True,exist_ok=True)
    (args.output_dir/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
