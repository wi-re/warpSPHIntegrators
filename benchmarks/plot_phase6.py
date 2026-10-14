"""Render saved Phase 6 CPU measurements; performs no numerical reruns."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--startup',type=Path,default=Path('.tmp/phase6/startup/report.json'))
    parser.add_argument('--scale',type=Path,default=Path('.tmp/phase6/scale/report.json'))
    parser.add_argument('--output',type=Path,default=Path('docs/assets/phase6-implicit-gates.png'))
    args=parser.parse_args()
    startup=json.loads(args.startup.read_text());scale=json.loads(args.scale.read_text())
    fig,axes=plt.subplots(1,3,figsize=(14,4.3),layout='constrained')
    colors={'BDF3':'#2465a1','BDF4':'#bc582b','BDF5':'#33834b'}
    for name in colors:
        for mode,style in [('implicit','-'),('explicit','--')]:
            row=next(r for r in startup['results'] if r['scheme']==name and r['startup']==mode and r['h']==.1)
            axes[0].semilogy([p['t'] for p in row['points']],[p['error'] for p in row['points']],
                color=colors[name],linestyle=style,label=f'{name}, {mode}')
    axes[0].set(title='Stiff bootstrap: rate 100, dt=0.1',xlabel='Time',ylabel='Absolute error')
    axes[0].legend(fontsize=7,ncol=2);axes[0].grid(alpha=.25)
    for scheme,color in [('gauss3','#2465a1'),('radau3','#bc582b')]:
        for kind,style in [('reaction','-'),('diffusion','--')]:
            rows=[r for r in scale['results'] if r['scheme']==scheme and r['problem']==kind and r['mode']=='block' and not r['warm']]
            rows.sort(key=lambda r:r['n'])
            axes[1].loglog([r['n'] for r in rows],[max(r['krylov_iterations']) for r in rows],
                style+'o',color=color,label=f'{scheme}, {kind}')
            axes[2].loglog([r['n'] for r in rows],[r['median_seconds'] for r in rows],
                style+'o',color=color,label=f'{scheme}, {kind}')
            for r in rows:
                if r['failures']:
                    axes[1].plot(r['n'],max(r['krylov_iterations']),'x',markersize=12,color='black')
                    axes[2].plot(r['n'],r['median_seconds'],'x',markersize=12,color='black')
    axes[1].axhline(120,color='gray',linestyle=':',label='Work bar: 120')
    axes[1].set(title='Local stage-block preconditioner',xlabel='Degrees of freedom N',ylabel='Max Krylov iterations / step')
    axes[1].legend(fontsize=7);axes[1].grid(alpha=.25,which='both')
    axes[2].set(title='CPU float64, one thread',xlabel='Degrees of freedom N',ylabel='Median step wall time (seconds)')
    axes[2].legend(fontsize=7);axes[2].grid(alpha=.25,which='both')
    fig.suptitle('Phase 6: stiff startup passes; wave 9 scale gate fails (× = solve failure)',fontsize=12)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(args.output,dpi=180);plt.close(fig)


if __name__=='__main__':main()
