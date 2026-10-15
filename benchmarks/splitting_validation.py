"""Phase 7 library consumers: split oscillator, periodic Burgers, CFL and merging.

The scalar-field PDE uses TensorState; its transported value is not tagged as a
particle coordinate. No SPH separability/conservation claim is inferred from it.
"""
import argparse
import json
import math
from pathlib import Path
import time

import torch
from warpSPHIntegrators.wrappers import ConservativeDissipativeRHS, Strang, LieTrotter, LinearExponentialFlow, compose, classify_separable, Yoshida4
from warpSPHIntegrators.adapters import SemilinearRHS
from warpSPHIntegrators import TensorState, tensor_rhs, solve, get_scheme
import warpSPHIntegrators.testing as testing
from warpSPHIntegrators.state import get_reference_state

OPTS=dict(matvec='jvp',tol=1e-12,newton_tol=1e-7)


def split_burgers(n=32,nu=.1):
    spacing=1/n;eye=torch.eye(n,dtype=torch.float64)
    matrix=nu*(eye.roll(1,0)-2*eye+eye.roll(-1,0))/spacing**2
    def convection(t,y):return -(y.square().roll(-1)-y.square().roll(1))/(4*spacing)
    def diffusion(t,y):return matrix@y
    rhs=tensor_rhs(ConservativeDissipativeRHS(convection,SemilinearRHS(linear=diffusion)))
    initial=TensorState(.2+.1*torch.sin(2*math.pi*torch.arange(n,dtype=torch.float64)/n))
    return initial,rhs,matrix


def run(method,state,rhs,h,end):
    start=time.perf_counter()
    result=solve(method,state,end,rhs,dt=h,solver_opts=OPTS,raise_on_failure=False)
    return result,time.perf_counter()-start


def split_oscillator(k=4.,c=.5):
    def conservative(state,dt,**kwargs):
        s=get_reference_state(state)
        return testing.ParticleUpdate(s.u,-k*s.x,s.e*0),None
    def dissipative(state,dt,**kwargs):
        s=get_reference_state(state)
        return testing.ParticleUpdate(s.x*0,-c*s.u,s.e*0),None
    return ConservativeDissipativeRHS(conservative,SemilinearRHS(linear=dissipative))


def oscillator_diagnostics():
    problem=testing.damped_problem();rhs=split_oscillator()
    exact_d=LinearExponentialFlow(torch.diag(torch.tensor([0.,-.5,0.],dtype=torch.float64)))
    methods=[('LieTrotter',LieTrotter('Velocity Verlet','Implicit Midpoint')),
             ('Strang',Strang('Velocity Verlet',exact_d)),
             ('Yoshida4',Yoshida4('Velocity Verlet',exact_d))]
    orders=[];geometry=[];energy=[]
    for name,method in methods:
        rows=[]
        for h in ((.1,.05,.025) if name=='LieTrotter' else (.2,.1,.05)):
            result,seconds=run(method,problem.initial(),rhs,h,1.)
            if not result.successful:raise RuntimeError(f'{name} oscillator run failed')
            q,v=problem.exact(1.);s=get_reference_state(result.state)
            rows.append(dict(h=h,error=abs(float(s.x)-q[0])+abs(float(s.u)-v[0]),seconds=seconds))
        orders.append(dict(method=name,order=method.order,
            measured_order=math.log(rows[-2]['error']/rows[-1]['error'],2),rows=rows))
    # Linear oscillator: images of the two phase-space basis vectors give its
    # Jacobian directly, without finite-difference or nonlinear probing error.
    for name,method in methods[1:]+[('unsplit-RK4',get_scheme('RK4'))]:
        columns=[]
        for q,v in ((1.,0.),(0.,1.)):
            state=problem.initial();s=get_reference_state(state)
            s.x=torch.tensor([q],dtype=torch.float64);s.u=torch.tensor([v],dtype=torch.float64)
            result=method(state,.5,rhs if name!='unsplit-RK4' else problem.rhs,solver_opts=OPTS)
            if not result.step_status.successful:raise RuntimeError('Geometry subsolve failed')
            s=get_reference_state(result.state);columns.append(torch.stack((s.x[0],s.u[0])))
        determinant=float(torch.linalg.det(torch.stack(columns,dim=1)))
        geometry.append(dict(method=name,h=.5,determinant=determinant,
            expected=math.exp(-.25),defect=abs(determinant-math.exp(-.25))))
    method=methods[1][1]
    for h in (.1,.05,.025):
        state=problem.initial();s=get_reference_state(state)
        initial_energy=float((s.u.square()+4*s.x.square()).sum()/2);work=0.
        for _ in range(round(1/h)):
            old=get_reference_state(state).u.clone()
            result=method(state,h,rhs,solver_opts=OPTS)
            if not result.step_status.successful:raise RuntimeError('Energy subsolve failed')
            state=result.state;s=get_reference_state(state)
            work+=.5*h*float((old.square()+s.u.square()).sum())/2
        final_energy=float((s.u.square()+4*s.x.square()).sum()/2)
        energy.append(dict(h=h,initial_energy=initial_energy,final_energy=final_energy,
            trapezoidal_dissipation=work,balance_defect=abs(final_energy-initial_energy+work)))
    measured=math.log(energy[-2]['balance_defect']/energy[-1]['balance_defect'],2)
    return dict(orders=orders,geometry=geometry,energy_balance=energy,
                energy_balance_order=measured,separability=classify_separable(rhs._conservative,problem.initial()).separable)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,default=Path('.tmp/phase7/benchmarks'))
    args=parser.parse_args();torch.set_num_threads(1);args.output_dir.mkdir(parents=True,exist_ok=True)
    state,rhs,matrix=split_burgers()
    reference,_=run(get_scheme('RK4'),state,rhs,.0001,.1)
    errors=[]
    for kind,method in [('LieTrotter',LieTrotter('RK4','Implicit Midpoint')),
                        ('Strang-midpoint',Strang('RK4','Implicit Midpoint')),
                        ('Strang-exp',Strang('RK4',LinearExponentialFlow(matrix))),
                        ('Strang-BE-control',Strang('RK4','Backward Euler (implicit)'))]:
        rows=[]
        for h in (.01,.005,.0025):
            result,seconds=run(method,state,rhs,h,.1)
            if not result.successful:raise RuntimeError(f'{kind} stopped before the order endpoint: {result.termination}')
            error=float(torch.linalg.vector_norm(result.state.y-reference.state.y))
            rows.append(dict(h=h,error=error,seconds=seconds,termination=result.termination,
                             mass_error=abs(float(result.state.y.sum()-state.y.sum()))))
        measured=math.log(rows[-2]['error']/rows[-1]['error'],2)
        errors.append(dict(method=kind,order=method.order,measured_order=measured,rows=rows))
    stiff,rhs,stiff_matrix=split_burgers(n=64,nu=1.)
    cfl=[]
    for method in [('explicit',Strang('RK4','Forward Euler')),
                   ('implicit',Strang('RK4','Implicit Midpoint')),
                   ('exponential',Strang('RK4',LinearExponentialFlow(stiff_matrix)))]:
        for h in (.001,.005,.01):
            result,seconds=run(method[1],stiff,rhs,h,.05)
            cfl.append(dict(method=method[0],h=h,explicit_step_ratio=h/(1/(2*64**2)),
                successful=result.successful,bounded=result.successful and float(result.state.y.abs().max())<=2*float(stiff.y.abs().max()),termination=result.termination,
                max_amplitude=float(result.state.y.abs().max()),seconds=seconds))
    # Fixed-step exact D merging demonstration. It changes lifecycle boundaries,
    # so it is a mathematical benchmark, not an automatic run/checkpoint policy.
    count=[0]
    def operator(state):count[0]+=1;return torch.tensor([[-2.]],dtype=torch.float64)
    f=tensor_rhs(ConservativeDissipativeRHS(lambda t,y:y*y,SemilinearRHS(linear=lambda t,y:-2*y)))
    d=LinearExponentialFlow(operator);initial=TensorState(torch.ones(1,dtype=torch.float64));n=20
    normal,_=run(Strang('RK4',d),initial,f,.01,.2);normal_flows=count[0];count[0]=0
    parts=[('dissipative',d)];coefficients=[.5/n]
    for i in range(n):
        parts.extend([('conservative',get_scheme('RK4')),('dissipative',d)])
        coefficients.extend([1/n,(.5 if i==n-1 else 1)/n])
    merged=compose(parts,coefficients,order=2,name='Merged exact-flow demonstration')
    result=merged(initial,.2,f)
    merging=dict(steps=n,unmerged_d_flows=normal_flows,merged_d_flows=count[0],
                 final_difference=float((result.state.y-normal.state.y).abs().max()))
    report=dict(dtype='float64',device='cpu',threads=1,oscillator=oscillator_diagnostics(),
                burgers_order=errors,viscous_cfl=cfl,merging=merging)
    (args.output_dir/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
