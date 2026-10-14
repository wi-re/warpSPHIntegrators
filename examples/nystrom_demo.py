"""Known damped oscillator with mechanical energy converted into heat.

Run: python examples/nystrom_demo.py --method FineRKN5 --n 64 --dt .05
The exact underdamped solution provides position, velocity and heat references.
Velocity-independent methods require --damping 0 (passive heat).
"""
import argparse
import math
import torch
from warpSPHIntegrators import Nystrom,testing,solve,ConservativeDissipativeRHS


def problem(n=1,damping=.4):
    omega=torch.linspace(1.,4.,n,dtype=torch.float64)
    def initial():
        return testing.ParticleSystem(testing.ParticleState(
            x=torch.ones(n,dtype=torch.float64),u=torch.zeros(n,dtype=torch.float64),
            e=torch.zeros(n,dtype=torch.float64),m=torch.ones(n,dtype=torch.float64)))
    def conservative(system,dt,**kwargs):
        s=system.state
        return testing.ParticleUpdate(s.u,-omega.square()*s.x,torch.zeros_like(s.e)),None
    def dissipative(system,dt,**kwargs):
        s=system.state
        return testing.ParticleUpdate(torch.zeros_like(s.x),-damping*s.u,damping*s.u.square()),None
    def rhs(system,dt,**kwargs):
        s=system.state
        return testing.ParticleUpdate(s.u,-omega.square()*s.x-damping*s.u,damping*s.u.square()),None
    def exact(t):
        frequency=(omega.square()-damping*damping/4).sqrt();decay=math.exp(-damping*t/2)
        q=decay*(torch.cos(frequency*t)+damping/(2*frequency)*torch.sin(frequency*t))
        v=-decay*omega.square()/frequency*torch.sin(frequency*t)
        heat=omega.square()/2-(v.square()+omega.square()*q.square())/2
        return q,v,heat
    return initial,rhs,exact,omega,ConservativeDissipativeRHS(conservative,dissipative)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--method',default='FineRKN5')
    parser.add_argument('--n',type=int,default=1)
    parser.add_argument('--dt',type=float,default=.05)
    parser.add_argument('--damping',type=float,default=.4)
    args=parser.parse_args()
    if args.n<1 or not 0<=args.damping<2:parser.error('n>=1 and 0<=damping<2 required')
    initial,rhs,exact,omega,_=problem(args.n,args.damping)
    result=solve(Nystrom(args.method),initial(),1.,rhs,dt=args.dt)
    q,v,heat=exact(1.);s=result.state.state
    print(f'{args.method}: {result.accepted_steps} steps, {args.n} oscillators')
    print('maximum position/velocity/heat error:',*(float((a-b).abs().max()) for a,b in zip((s.x,s.u,s.e),(q,v,heat))))
    print('maximum total-energy defect:',float(((s.u.square()+omega.square()*s.x.square())/2+s.e-omega.square()/2).abs().max()))


if __name__=='__main__':main()
