"""Reproduce the README oscillator measurements and compact stability figures.

CPU float64, one numerical thread; fixed h=.2, T=20; initial (x,v,Q)=(1,0,0).
The analytic reference has x''+.4*x'+x=0 and Q'=.4*v^2. RHS entries count
actual Python calls, including solver probes; GMRES is reported separately.
Wall times describe this small consumer, not equal-accuracy backend rankings.
"""
import hashlib
import json
from pathlib import Path
import platform
import runpy
import sys
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from warpSPHIntegrators import TensorState, tensor_rhs, getIntegrator, solve
from warpSPHIntegrators.stability import rk_stability_function

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT/'docs/assets'
DEMO = runpy.run_path(str(ROOT/'examples/damped_oscillator.py'))


def oscillator():
    rows=[]
    fig, axes=plt.subplots(2, 2, figsize=(11, 7.2), layout='constrained')
    times=np.linspace(0, 20, 1001)
    reference=np.stack([DEMO['exact'](t).numpy() for t in times])
    axes[0,0].plot(times, reference[:,0], 'k--', label='Exact', lw=2)
    axes[0,1].plot(reference[:,0], reference[:,1], 'k--', lw=2)
    axes[1,0].plot(times, .5*(reference[:,0]**2+reference[:,1]**2), 'k--', label='Exact mechanical')
    axes[1,0].plot(times, reference[:,2], color='black', ls=':', label='Exact heat')
    for name in ('Forward Euler', "Heun's Method (2nd order)", 'RK4', 'Dormand-Prince 5(4)', 'TR-BDF2'):
        method=getIntegrator(name);calls=0;gmres=0
        def force(t,y):
            nonlocal calls
            calls+=1
            return DEMO['rates'](t,y)
        def observed(record):
            nonlocal gmres
            gmres+=sum(d.gmres_iterations for d in (record.result.solver_diagnostics or ()) if d is not None)
        options={'solver_opts':dict(matvec='jvp',tol=1e-12,newton_tol=1e-7)} if method.implicit else {}
        start=time.perf_counter()
        result=solve(method,TensorState(torch.tensor([1.,0.,0.],dtype=torch.float64)),20.,
            tensor_rhs(force),dt=.2,save_steps=True,callback=observed,**options)
        elapsed=time.perf_counter()-start
        assert result.successful and result.accepted_steps==100
        ts=np.array([s.t for s in result.samples]);ys=np.stack([s.state.y.detach().numpy() for s in result.samples])
        expected=np.stack([DEMO['exact'](t).numpy() for t in ts])
        mechanical=.5*(ys[:,0]**2+ys[:,1]**2);budget=mechanical+ys[:,2]-.5
        label={'Heun\'s Method (2nd order)':'Heun RK2','Dormand-Prince 5(4)':'DP5'}.get(name,name)
        line,=axes[0,0].plot(ts,ys[:,0],label=label,lw=1.1)
        color=line.get_color()
        axes[0,1].plot(ys[:,0],ys[:,1],color=color,lw=1)
        axes[1,0].plot(ts,mechanical,color=color,lw=1)
        axes[1,1].semilogy(ts,np.maximum(abs(budget),1e-16),color=color,lw=1,label=label)
        rows.append(dict(method=name,dt=.2,end=20.,steps=result.accepted_steps,rhs_entries=calls,
            gmres_iterations=gmres,seconds=elapsed,
            max_position_velocity_error=float(abs(ys[:,:2]-expected[:,:2]).max()),
            max_heat_error=float(abs(ys[:,2]-expected[:,2]).max()),
            max_total_energy_defect=float(abs(budget).max())))
    axes[0,0].set(title='Displacement: decay and phase',xlabel='Time',ylabel='x');axes[0,0].legend(fontsize=8,ncol=2)
    axes[0,1].set(title='Phase portrait: inward spiral',xlabel='x',ylabel='v')
    axes[1,0].set(title='Mechanical energy becomes heat',xlabel='Time',ylabel='Energy');axes[1,0].legend(fontsize=8)
    axes[1,1].set(title='Error in mechanical energy + heat = 0.5',xlabel='Time',ylabel='Absolute budget defect')
    for ax in axes.flat:ax.grid(alpha=.2)
    fig.suptitle("Damped oscillator: x'' + 0.4 x' + x = 0; fixed h = 0.2",fontsize=14)
    fig.savefig(ASSETS/'readme-damped-oscillator.png',dpi=160);plt.close(fig)
    return rows


def stability():
    real=np.linspace(-6,2,321);imag=np.linspace(-4,4,321)
    z=real[None,:]+1j*imag[:,None]
    fig,axes=plt.subplots(2,3,figsize=(11,7.2),layout='constrained')
    names=['Forward Euler','RK4','Dormand-Prince 5(4)','Backward Euler (implicit)','Implicit Midpoint','Radau IIA s=3']
    for ax,name in zip(axes.flat,names):
        method=getIntegrator(name);f=method.function
        tableau=getattr(f,'butcherTableau',getattr(f,'dirkTableau',None))
        if tableau is None:
            from warpSPHIntegrators import Collocation
            tableau=Collocation('radau',stages=3).tableau
        a=np.asarray(tableau.a);b=np.asarray(tableau.b[0] if isinstance(tableau.b,tuple) else tableau.b);s=len(b)
        # Batched solves reproduce the package's scalar stability function.
        matrices=np.eye(s)-z[...,None,None]*a
        with np.errstate(divide='ignore',invalid='ignore'):
            regular=abs(np.linalg.det(matrices))>1e-14
            stages=np.zeros(z.shape+(s,),dtype=complex)
            stages[regular]=np.linalg.solve(matrices[regular],np.ones((int(regular.sum()),s,1)))[...,0]
            r=1+z*np.einsum('...i,i->...',stages,b)
            r[~regular]=complex(float('inf'),0)
        for point in (-2.3+.7j,-.3+2.1j):
            k=np.linalg.solve(np.eye(s)-point*a,np.ones(s));rr=1+point*b@k
            assert abs(rr-rk_stability_function(tableau,point))<1e-11
        stable=(abs(r)<=1+1e-12).astype(float)
        ax.contourf(real,imag,stable,levels=[-.5,.5,1.5],colors=['#ffffff','#b7dfca'])
        ax.contour(real,imag,stable,levels=[.5],colors=['#1b7950'],linewidths=.9)
        ax.axhline(0,color='#555',lw=.5);ax.axvline(0,color='#555',lw=.5)
        ax.set(title=name,xlabel='Re(hλ)',ylabel='Im(hλ)',aspect='equal')
    fig.suptitle("Absolute stability: green means |R(hλ)| ≤ 1; axes share the same scale",fontsize=13)
    fig.savefig(ASSETS/'readme-stability-regions.png',dpi=160);plt.close(fig)


def update_readme(rows):
    """Publish measurements beside the equations and figures they describe."""
    lines=['| Method | RHS entries | GMRES iterations | Max motion error | Max heat error | Max total-energy defect |',
           '|---|---:|---:|---:|---:|---:|']
    for row in rows:
        lines.append(f"| {row['method']} | {row['rhs_entries']:,} | {row['gmres_iterations']:,} | "
                     f"{row['max_position_velocity_error']:.3g} | {row['max_heat_error']:.3g} | "
                     f"{row['max_total_energy_defect']:.3g} |")
    path=ROOT/'README.md';content=path.read_text()
    start='[readme:oscillator-results:start]: #';end='[readme:oscillator-results:end]: #'
    before,remainder=content.split(start,1);_,after=remainder.split(end,1)
    path.write_text(before+start+'\n\n'+'\n'.join(lines)+'\n\n'+end+after)


if __name__=='__main__':
    torch.set_num_threads(1);ASSETS.mkdir(parents=True,exist_ok=True)
    rows=oscillator();stability()
    report=dict(device='cpu',dtype='float64',threads=1,python=sys.version,torch=torch.__version__,
        numpy=np.__version__,platform=platform.platform(),processor=platform.processor(),
        cpu_model=next((line.split(':',1)[1].strip() for line in Path('/proc/cpuinfo').read_text().splitlines() if line.startswith('model name')),'unknown'),
        cases=rows,qualification='Fixed grid and analytic reference; errors are maxima over saved samples. RHS entries include probes; GMRES counts are additional diagnostics, not additional direct RHS entries. Small CPU timings are not equal-accuracy speed rankings.',
        source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (Path(__file__).resolve(),ROOT/'examples/damped_oscillator.py')})
    target=ROOT/'docs/development/evidence/readme-oscillator.json'
    target.write_text(json.dumps(report,indent=2)+'\n')
    update_readme(rows)
    print(json.dumps(report,indent=2))
