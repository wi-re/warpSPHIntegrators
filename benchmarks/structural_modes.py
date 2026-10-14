"""Known structural consumer, damping/phase/order/work and source controls."""
import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
import time
import torch

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('structural_demo',ROOT/'examples/structural_modes.py')
demo=importlib.util.module_from_spec(spec);sys.modules[spec.name]=demo;spec.loader.exec_module(demo)


def error_at_endpoint(model,states):
    q,v=model.exact(states[-1].t)
    return float(torch.linalg.vector_norm(states[-1].q-q)+torch.linalg.vector_norm(states[-1].v-v))


def order_measurements():
    # Both modes resolved here. The coarse behavior sweep deliberately leaves
    # the fast mode underresolved; it cannot be used as an order measurement.
    model=demo.two_mass_problem(slow=1.,fast=3.)
    rows=[]
    for rho in (0.,.5,1.):
        for reset in (False,True):
            errors=[];grids=(.1,.05,.025,.0125,.00625)
            for h in grids:
                states=demo.trajectory(model,h,2.,rho,reset_acceleration=reset)
                errors.append(error_at_endpoint(model,states))
            rows.append(dict(rho_inf=rho,reset_acceleration=reset,h=list(grids),errors=errors,
                measured_order=math.log(errors[-2]/errors[-1],2)))
    return rows


def measurements():
    model=demo.two_mass_problem();h=.1;end=12.;before=model.modal_energy(model.initial());rows=[];traces={}
    for rho in (1.,.8,.5,0.):
        start=time.perf_counter();states=demo.trajectory(model,h,end,rho);seconds=time.perf_counter()-start
        traces[f'rho={rho:g}']=states
        z=torch.stack([model.modes.T@s.q for s in states]);v=torch.stack([model.modes.T@s.v for s in states])
        energy=torch.stack([model.modal_energy(s) for s in states]);t=torch.tensor([s.t for s in states],dtype=torch.float64)
        exact=model.amplitudes[0]*torch.cos(model.frequencies[0]*t)
        # Low-mode phase from q+i*v/omega at the endpoint; no phase-wrap ambiguity
        # at this short horizon. Amplitude comes from modal energy.
        phase=math.atan2(-float(v[-1,0]/model.frequencies[0]),float(z[-1,0]))
        phase_error=math.atan2(math.sin(phase-end),math.cos(phase-end))
        rows.append(dict(rho_inf=rho,dt=h,endpoint=end,omega_dt=[.1,4.],
            modal_energy_retained=(energy[-1]/before).tolist(),low_mode_max_displacement_error=float((z[:,0]-exact).abs().max()),
            low_mode_phase_error=phase_error,dense_solves=len(states)-1,seconds=seconds))
    start=time.perf_counter();baseline=demo.registered_newmark_trajectory(model,h,end);seconds=time.perf_counter()-start
    traces['Registered Newmark']=baseline
    parity=max(float((a.q-b.q).abs().max()) for a,b in zip(traces['rho=1'],baseline))
    baseline_record=dict(modal_energy_retained=(model.modal_energy(baseline[-1])/before).tolist(),
        max_displacement_difference_from_rho1=parity,seconds=seconds,
        qualification='Registered matrix-free nonlinear solver versus dense linear prototype; timings are not a fair backend speed comparison')
    spectral=[]
    for rho in (0.,.5,.8,1.):
        mus=torch.logspace(-3,6,150,dtype=torch.float64).tolist()
        radii=[float(torch.linalg.eigvals(demo.amplification_matrix(mu,rho)).abs().max()) for mu in mus]
        spectral.append(dict(rho_inf=rho,omega_dt=mus,spectral_radius=radii,
            high_frequency_radius=radii[-1],max_radius=max(radii)))
    am,af,_,_=demo.alpha_parameters(1.)
    pinned_beta=(.5+af-am)**2/4
    source_beta_control=dict(rho_inf=1.,pinned_beta=pinned_beta,classical_beta=.25,
        radius_at_omega_dt_4=float(torch.linalg.eigvals(demo.amplification_matrix(4.,1.,beta_override=pinned_beta)).abs().max()),
        qualification='Isolates the cached beta expression; this is not a full run of the upstream solver')
    return dict(dtype='float64',device='cpu',threads=1,dt=h,endpoint=end,
        model=dict(masses=[1.,1.],ground_spring=1.,coupling_spring=799.5,frequencies=[1.,40.],modal_amplitudes=[1.,.05]),
        behavior=rows,registered_newmark=baseline_record,order=order_measurements(),spectral=spectral,
        source_beta_control=source_beta_control),model,traces


def plot(model,traces,report,path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    plt.rcParams.update({'font.size':10})
    fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained')
    times=np.array([s.t for s in next(iter(traces.values()))])
    axes[0,0].plot(times,np.cos(times),'k--',label='Exact slow mode')
    for label,states in traces.items():
        if label in ('rho=0.8','Registered Newmark'):continue
        modal=torch.stack([model.modes.T@s.q for s in states]).numpy()
        energy=torch.stack([model.modal_energy(s) for s in states]).numpy()
        axes[0,0].plot(times,modal[:,0],label=label)
        axes[0,1].semilogy(times,np.maximum(energy[:,1]/energy[0,1],1e-28),label=label)
    axes[0,0].set(title='Slow motion stays resolved',xlabel='Time [s]',ylabel='Slow modal displacement')
    axes[0,1].axhline(1,color='black',linestyle='--',label='Exact undamped energy')
    axes[0,1].set(title='Fast-mode energy is filtered',xlabel='Time [s]',ylabel='Fast energy / initial fast energy')
    for row in report['spectral']:
        axes[1,0].semilogx(row['omega_dt'],row['spectral_radius'],label=f"rho={row['rho_inf']:g}")
    axes[1,0].set(title='Amplification approaches the requested rho',xlabel='omega * dt',ylabel='Augmented spectral radius',ylim=(-.03,1.03))
    for row in report['order']:
        if row['rho_inf']!=.5:continue
        label='Carry acceleration (order 2)' if not row['reset_acceleration'] else 'Reset acceleration (order 1)'
        axes[1,1].loglog(row['h'],row['errors'],'o-',label=label)
    axes[1,1].set(title='Resolved refinement catches memory mistakes',xlabel='dt [s]',ylabel='Endpoint displacement + velocity error')
    for ax in axes.flat:ax.grid(True,alpha=.25);ax.legend(fontsize=8)
    fig.suptitle('Two unit masses: 1 rad/s motion + 40 rad/s vibration; dt=0.1 s',fontsize=14)
    fig.savefig(path,dpi=160);plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,default=Path('.tmp/phase8/structural-demo'))
    parser.add_argument('--figure',type=Path,default=ROOT/'docs/assets/structural-modes.png')
    args=parser.parse_args();torch.set_num_threads(1);report,model,traces=measurements()
    report['consumer_sha256']=hashlib.sha256((ROOT/'examples/structural_modes.py').read_bytes()).hexdigest()
    args.output_dir.mkdir(parents=True,exist_ok=True);args.figure.parent.mkdir(parents=True,exist_ok=True)
    (args.output_dir/'report.json').write_text(json.dumps(report,indent=2)+'\n');plot(model,traces,report,args.figure)
    print(json.dumps({k:v for k,v in report.items() if k!='spectral'},indent=2))


if __name__=='__main__':main()
