"""Pure recurrence programs shared by coefficient inspection and execution.

All states here are increments from the initial state. The callback returns
h*F(y0+increment) and marks the final physical refresh with endpoint=True.
This exact coordinate change preserves zero-rate fields without affine drift.
"""
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

CAPS={'ROCK2':200,'ROCK4':142,'SERK2':250,'RKMC2':1000}
MINIMUM={'ROCK2':3,'ROCK4':5,'SERK2':10,'RKMC2':3}


def available_stages(method):
    if method not in CAPS:raise ValueError(f'Choose from {tuple(CAPS)}')
    from warpSPHIntegrators.schemes.coefficients.orthogonal_data import TABLEAUS
    offset=2 if method=='ROCK2' else 4 if method=='ROCK4' else 0
    # Pinned ROCK4 counts 133/152 have unstable interior bands near z=-8.
    # Retain their raw source data/oracle, but omit them from supported selection.
    return tuple(s+offset for s in TABLEAUS[method]['ms'] if s+offset<=CAPS[method] and (method,s+offset)!=('ROCK4',133))


def sizing_bound(method,stages):
    """Conservative inversion of the pinned stage selector, checked numerically.

    These supplied spectral bounds concern negative-real linear spectra, not
    nonnormal operators or arbitrary nonlinear stability guarantees.
    """
    if method=='ROCK2':return .811*(stages-1)**2-1.5
    if method=='ROCK4':return .95*(.353*(stages-1)**2-3)
    if method=='SERK2':return .9*.8*(stages-1)**2
    if method=='RKMC2':return ((stages-1+.8306782178712795)/1.8547887825836553)**(1/.533871357807877)
    raise ValueError('Unknown orthogonal recurrence')


def _parameters(method,stages):
    from warpSPHIntegrators.schemes.coefficients.orthogonal_data import TABLEAUS
    tab=TABLEAUS[method]
    offset=2 if method=='ROCK2' else 4 if method=='ROCK4' else 0
    index=tab['ms'].index(stages-offset)
    if method in ('ROCK2','ROCK4'):
        degree=tab['ms'][index];start=sum(2*m-1 for m in tab['ms'][:index])
        recurrence=tuple(tab['recf'][start:start+2*degree-1])
        finishing=(tab['fp1'][index],tab['fp2'][index]) if method=='ROCK2' else tuple(
            tuple(tab[key][index]) for key in ('fpa','fpb','fpbe'))
        return recurrence,finishing
    if method=='SERK2':
        start=sum(m+1 for m in tab['ms'][:index])
        return tuple(tab['Bᵢ'][start:start+stages+1])
    return tuple(tab['roots'][index])


def recurrence_program(method,stages,parameters,evaluate,zero):
    """Execute the source recurrence in increments, without dense RK expansion."""
    if method in ('ROCK2','ROCK4'):
        recurrence,finishing=parameters
        f0=evaluate(zero);old=zero;current=recurrence[0]*f0
        for mu,kappa in zip(recurrence[1::2],recurrence[2::2]):
            new=mu*evaluate(current)+(1+kappa)*current-kappa*old
            old,current=current,new
        if method=='ROCK2':
            fp1,fp2=finishing
            first=evaluate(current);second=evaluate(current+fp1*first)
            error=fp2*(second-first)
            output=current+fp1*(first+second)+error
            evaluate(output,endpoint=True)
        else:
            a,b,embedded=finishing
            k1=evaluate(current);k2=evaluate(current+a[0]*k1)
            k3=evaluate(current+a[1]*k1+a[2]*k2)
            k4=evaluate(current+a[3]*k1+a[4]*k2+a[5]*k3)
            rates=(k1,k2,k3,k4)
            output=current
            for weight,rate in zip(b,rates):output=output+weight*rate
            endpoint=evaluate(output,endpoint=True)
            error=embedded[4]*endpoint
            for low,high,rate in zip(embedded,b,rates):error=error+(low-high)*rate
        return output,error
    if method=='SERK2':
        weights=parameters;alpha=2.5/(stages*stages);degree=stages//10
        old=current=zero;total=zero;index=1
        for block in range(10):
            new=current+alpha*evaluate(current)
            total=total+weights[index]*new;index+=1
            old,current=current,new
            for j in range(2,degree+1):
                new=2*current-old+2*alpha*evaluate(current)
                total=total+weights[index]*new;index+=1
                old,current=current,new
        endpoint=evaluate(total,endpoint=True)
        return total,total-endpoint
    if method!='RKMC2':raise ValueError('Unknown orthogonal recurrence')
    w0,w1=parameters;told=1.;tcurrent=w0;bold=.5;bcurrent=1/(1+w0)
    f0=evaluate(zero);old=zero;current=bcurrent*w1*f0
    for j in range(2,stages+1):
        tnew=2*w0*tcurrent-told;bnew=1/(1+tnew)
        mu=2*w0*bnew/bcurrent;nu=-bnew/bold;mutilde=2*w1*bnew/bcurrent
        new=mu*current+nu*old+mutilde*(evaluate(current)-bcurrent*f0)
        if j<stages:
            old,current=current,new;told,tcurrent=tcurrent,tnew;bold,bcurrent=bcurrent,bnew
    gamma=bcurrent/(2*stages*w1);delta=-bcurrent/(2*(stages-2)*w1)
    output=(gamma/bnew)*new+(delta/bold)*old+bcurrent*f0
    endpoint=evaluate(output,endpoint=True)
    return output,(-output+endpoint)/10


@dataclass(frozen=True)
class OrthogonalCoefficients:
    method: str
    stages: int
    parameters: tuple
    c: tuple
    weights: tuple
    error_weights: tuple


@lru_cache(maxsize=64)
def orthogonal_coefficients(method,stages):
    if type(stages) is not int or stages not in available_stages(method):
        raise ValueError('Use an available orthogonal-polynomial stage count')
    parameters=_parameters(method,stages);n=stages+1;nodes=[]
    def evaluate(offset,*,endpoint=False):
        nodes.append(1. if endpoint else float(np.sum(offset)))
        rate=np.zeros(n);rate[len(nodes)-1]=1.
        return rate
    weights,error=recurrence_program(method,stages,parameters,evaluate,np.zeros(n))
    if len(nodes)!=n:raise AssertionError('Source stage count does not match recurrence')
    # Propagate the first-order clock as a scalar too, avoiding accumulation in
    # a sum of many individually rounded coefficient entries at large counts.
    nodes=[]
    def clock(offset,*,endpoint=False):
        nodes.append(1. if endpoint else float(offset));return 1.
    recurrence_program(method,stages,parameters,clock,0.)
    return OrthogonalCoefficients(method,stages,parameters,tuple(nodes),tuple(map(float,weights)),tuple(map(float,error)))
