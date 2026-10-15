"""Dense <=64-entry operator/AD experiments and a matrix-free multirate kernel."""
import math
from fractions import Fraction
import numpy as np
import torch


def _check(y,h,*,dense=False):
    if not isinstance(y,torch.Tensor) or y.ndim!=1 or not y.is_floating_point() or not y.numel():raise ValueError('Nonempty floating vector required')
    if isinstance(h,bool) or not math.isfinite(h) or h==0:raise ValueError('Finite nonzero step required')
    if dense and y.numel()>64:raise ValueError('Dense experiment is limited to 64 entries')


def taylor_step(f,t,y,h,order=3):
    """Total derivatives through time augmentation, nested forward AD, order 3/4.

    f must accept tensor time and support torch.func.jvp including nested AD.
    This has a rapidly growing AD/RHS cost; no finite-difference derivative or
    substitution of time-independent partial derivatives is performed.
    """
    _check(y,h)
    if order not in (3,4):raise ValueError('Taylor order must be 3 or 4')
    z=torch.cat((y.new_tensor([t]),y))
    def field(z):return torch.cat((torch.ones_like(z[:1]),f(z[0],z[1:])))
    derivative=field;out=z
    for k in range(1,order+1):
        out=out+(h**k/math.factorial(k))*derivative(z)
        previous=derivative
        derivative=lambda z,previous=previous:torch.func.jvp(previous,(z,),(field(z),))[1]
    return out[1:]


def operator_step(A,t,y,h,method='magnus4'):
    """Dense time-dependent linear ODE, exponential midpoint/Magnus4/Cayley2.

    A(t) is an explicit matrix; all state/operator derivatives are retained.
    Magnus4 includes the noncommuting commutator, tested on rotating eigenvectors.
    Cayley preserves norms for skew matrices, but does not L-damp negative modes.
    """
    _check(y,h,dense=True)
    if method=='midpoint':return torch.matrix_exp(h*A(t+h/2))@y
    if method=='cayley2':
        M=.5*h*A(t+h/2);I=torch.eye(y.numel(),dtype=y.dtype,device=y.device)
        return torch.linalg.solve(I-M,(I+M)@y)
    if method!='magnus4':raise ValueError('Choose midpoint, magnus4 or cayley2')
    a=A(t+h*(.5-math.sqrt(3)/6));b=A(t+h*(.5+math.sqrt(3)/6))
    omega=.5*h*(a+b)-math.sqrt(3)*h*h/12*(a@b-b@a)
    return torch.matrix_exp(omega)@y


def _rk4(f,t,y,h):
    a=f(t,y);b=f(t+h/2,y+h*a/2);c=f(t+h/2,y+h*b/2);d=f(t+h,y+h*c)
    return y+h*(a+2*b+2*c+d)/6


def multirate_step(slow,fast,t,y,h,ratio=8):
    """Order-two slow forcing with an even number of fast RK4 microsteps.

    First half freezes slow(t,y); the midpoint slow evaluation is extrapolated
    as 2*S_mid-S_start for the second half. Costs two slow + 4*ratio fast calls.
    It needs a stable fast microstep and smooth slow forcing; no stiff/SSP claim.
    """
    _check(y,h)
    if type(ratio) is not int or ratio<2 or ratio%2:raise ValueError('Even microstep ratio >=2 required')
    s0=slow(t,y);value=y;micro=h/ratio
    for j in range(ratio//2):value=_rk4(lambda time,z:fast(time,z)+s0,t+j*micro,value,micro)
    s1=slow(t+h/2,value);forcing=2*s1-s0
    for j in range(ratio//2,ratio):value=_rk4(lambda time,z:fast(time,z)+forcing,t+j*micro,value,micro)
    return value


def interaction_picture_step(L,nonlinear,t,y,h):
    """Dense RK4 interaction picture, matching the source RKIP coordinate map.

    Uses exp(+/-c*h*L); negative exponential actions can overflow for diffusion.
    No source adaptive tableau/cache/FSAL policy is reproduced.
    """
    _check(y,h,dense=True)
    def transformed(time,v):
        offset=time-t
        return torch.matrix_exp(-offset*L)@nonlinear(time,torch.matrix_exp(offset*L)@v)
    return torch.matrix_exp(h*L)@_rk4(transformed,t,y,h)


def _weights(sequence,power):
    points=[Fraction(1,n**power) for n in sequence];weights=[]
    for i,x in enumerate(points):
        value=Fraction(1)
        for j,z in enumerate(points):
            if j!=i:value*=-z/(x-z)
        weights.append(float(value))
    return weights


def implicit_extrapolation_step(f,t,y,h,order=4,base='euler'):
    """Dense frozen-Jacobian linear corrections, not nonlinear backward Euler.

    Euler uses geometric subdivisions and cancels inverse-count powers.
    Midpoint uses 4,8,... subdivisions and the pinned Deuflhard unsmoothed
    recurrence, cancelling even powers. Time is augmented before freezing J.
    This prototype retains all unrolled graphs and does not reproduce source
    order/work controllers, divergence detector, mass matrices or stage lifecycle.
    """
    _check(y,h,dense=True)
    if type(order) is not int or not 2<=order<=6 or base not in ('euler','midpoint') or base=='midpoint' and order%2:
        raise ValueError('Euler order 2–6 or even midpoint order 2–6 required')
    z=torch.cat((y.new_tensor([t]),y))
    def field(z):return torch.cat((torch.ones_like(z[:1]),f(z[0],z[1:])))
    J=torch.func.jacrev(field)(z);I=torch.eye(len(z),dtype=y.dtype,device=y.device);f0=field(z)
    ns=[2**j for j in range(order)] if base=='euler' else [4*(j+1) for j in range(order//2)]
    columns=[]
    for n in ns:
        micro=h/n;M=I-micro*J;previous=z;current=z+torch.linalg.solve(M,micro*f0)
        for _ in range(1,n):
            rate=field(current)
            following=current+torch.linalg.solve(M,micro*rate) if base=='euler' else 2*current-previous+2*torch.linalg.solve(M,micro*rate-(current-previous))
            previous,current=current,following
        columns.append(current-z)
    out=z
    for w,col in zip(_weights(ns,1 if base=='euler' else 2),columns):out=out+w*col
    return out[1:]


def nordsieck_predict(jets,shift=1.):
    """Pascal translation of scaled polynomial derivatives (new owned tensor)."""
    if jets.ndim!=2 or jets.shape[0]<2:raise ValueError('jets must be (order+1, entries)')
    return torch.stack([sum((math.comb(k,j)*shift**(k-j)*jets[k] for k in range(j,len(jets))),torch.zeros_like(jets[0])) for j in range(len(jets))])


def nordsieck_rescale(jets,ratio):
    if isinstance(ratio,bool) or not math.isfinite(ratio) or ratio<=0:raise ValueError('Positive finite step ratio required')
    return torch.stack([ratio**j*jet for j,jet in enumerate(jets)])


def nordsieck_adams_step(f,t,jets,h,*,tol=1e-12,maxiter=10):
    """Warm fixed-grid Adams corrector in Nordsieck coordinates, orders 1–12.

    Input jets are h^j/j! times polynomial derivatives at t. Dense Newton solves
    the endpoint condition; the source AN5/JVODE limited functional iteration,
    startup/interpolants, rejection rewind and order controller are distinct.
    This is a representation/operator prototype, not those complete algorithms.
    """
    _check(jets[0],h,dense=True);p=len(jets)-1
    if not 1<=p<=12:raise ValueError('Warm Adams order 1–12 required')
    polynomial=np.polynomial.Polynomial([1.])
    for j in range(1,p):polynomial*=np.polynomial.Polynomial([1.,1/j])
    primitive=polynomial.integ();constant=float(primitive(0)-primitive(-1))
    coeff=jets.new_tensor([constant,*primitive.coef[1:]])
    predicted=nordsieck_predict(jets);value=predicted[0]
    I=torch.eye(value.numel(),dtype=value.dtype,device=value.device)
    for _ in range(maxiter):
        residual=value-predicted[0]-constant*(h*f(t+h,value)-predicted[1])
        if bool(torch.linalg.vector_norm(residual)<=tol*max(1.,float(torch.linalg.vector_norm(value)))):break
        J=torch.func.jacrev(lambda z:f(t+h,z))(value)
        value=value-torch.linalg.solve(I-constant*h*J,residual)
    else:raise RuntimeError('Warm Nordsieck correction failed')
    correction=h*f(t+h,value)-predicted[1]
    return predicted+coeff[:,None]*correction


def amf_matrix(factors,shift):
    """Ordered product prod(I-shift*J_i), preserving noncommuting factor order.

    Source W is minus this product divided by shift. Replacing I-shift*sum(J_i)
    changes the numerical method by O(shift²); it is not a free exact solve.
    """
    if not factors:raise ValueError('At least one square factor required')
    first=factors[0];I=torch.eye(first.shape[0],dtype=first.dtype,device=first.device);product=I
    if len(I)>64:raise ValueError('Dense AMF experiment is limited to 64 entries')
    for matrix in factors:
        if matrix.shape!=I.shape:raise ValueError('Factors must have matching square layouts')
        product=product@(I-shift*matrix)
    return product


def low_storage_ck54_step(f,t,y,h):
    """Five-stage Carpenter–Kennedy 2N recurrence, tensor-only no-grad experiment.

    Coefficients are recovered algebraically from the qualified source map.
    Two persistent vectors; arbitrary RHS temporaries are outside that count.
    Full-stage SPH finalization/rerun and differentiable runs must retain data and
    are deliberately excluded instead of claiming a memory saving there.
    """
    _check(y,h)
    if torch.is_grad_enabled():raise ValueError('Low-storage experiment requires torch.no_grad()')
    from ..expanded_rk import expanded_rk_tableau
    tab=expanded_rk_tableau('CarpenterKennedy2N54')
    a=np.asarray(tab.A if hasattr(tab,'A') else tab.a);b=np.asarray(tab.b);s=len(b)-1
    rows=list(a[:s])+[b];beta=[];alpha=[];previous=None
    for i in range(s):
        delta=rows[i+1]-rows[i];beta.append(delta[i])
        alpha.append(0. if i==0 else delta[i-1]/(beta[i]*previous[i-1]))
        previous=delta/beta[i]
    value=y.clone();residual=torch.zeros_like(y)
    for i in range(s):
        rate=f(t+h*float(tab.c[i]),value)
        residual.mul_(float(alpha[i])).add_(rate,alpha=h)
        value.add_(residual,alpha=float(beta[i]))
    return value


def irkc_step(implicit,explicit,t,y,h,stages=4,*,tol=1e-12,maxiter=12):
    """Dense IRKC split recurrence with checked nonlinear stage solves.

    Translation of the pinned constant-cache recurrence at a supplied degree.
    This requires nonlinear implicit solves, rather than just frozen-J GMRES.
    No source spectral estimator, stage selector, embedded controller or cache.
    The pinned map fails its advertised second-order gate on the retained split
    nonlinear/time-dependent control; it is an audit specimen, not a recommendation.
    Raises on failed solves (the pinned kernel comments out those checks).
    """
    _check(y,h,dense=True)
    if type(stages) is not int or not 2<=stages<=50:raise ValueError('IRKC stages must be 2–50')
    w0=1+2/(13*stages*stages);temp=w0*w0-1;root=math.sqrt(temp);theta=stages*math.log(w0+root)
    w1=math.sinh(theta)*temp/(math.cosh(theta)*stages*root-w0*math.sinh(theta))
    b0=1/(4*w0*w0);b1=1/w0;mu1=w1*b1
    fi0=implicit(t,y);fe0=explicit(t,y);I=torch.eye(len(y),dtype=y.dtype,device=y.device)
    def solve(anchor,c):
        value=anchor
        for _ in range(maxiter):
            residual=value-anchor-h*mu1*implicit(t+c*h,value)
            if bool(residual.norm()<=tol*max(1.,float(value.norm()))):return value
            J=torch.func.jacrev(lambda z:implicit(t+c*h,z))(value)
            value=value-torch.linalg.solve(I-h*mu1*J,residual)
        raise RuntimeError('IRKC nonlinear stage failed')
    old=y;current=solve(y+h*mu1*fe0,mu1);cold=0.;cprev=mu1
    T0,T1=1.,w0;d0,d1=0.,1.;dd0,dd1=0.,0.;fiold=fi0
    for j in range(2,stages+1):
        T=2*w0*T1-T0;d=2*w0*d1+2*T1-d0;dd=2*w0*dd1+4*d1-dd0
        b=dd/(d*d);mu=2*w0*b/b1;nu=-b/b0;ms=mu*w1/w0;ns=-(1-T1*b1)*ms
        c=mu*cprev+nu*cold+ms+ns
        fiprev=implicit(t+cprev*h,current);feprev=explicit(t+cprev*h,current)
        anchor=y+mu*(current-y)+nu*(old-y)+h*(ms*feprev+ns*fe0+(ns-(1-mu-nu)*mu1)*fi0-nu*mu1*fiold)
        following=solve(anchor,c)
        old,current=current,following;fiold=fiprev;cold,cprev=cprev,c;b0,b1=b1,b
        T0,T1=T1,T;d0,d1=d1,d;dd0,dd1=dd1,dd
    return current


def exponential_adams2_step(L,nonlinear,t,y,h,previous_nonlinear=None):
    """Pinned fixed-step ETD2 warm map and exponential-Euler startup.

    Returns (new tensor, current N) for explicit caller-owned fixed-step history.
    This differs from the one-step ETDRK2 map. Changing h, parameters or identity
    requires discarding history. Full-state ownership/controller is not provided.
    """
    _check(y,h,dense=True)
    from ..exponential_rk import operator_function
    current=nonlinear(t,y);cache={};Z=h*L
    value=operator_function(['phi',0,1.],Z,cache)@y+h*(operator_function(['phi',1,1.],Z,cache)@current)
    if previous_nonlinear is not None:value=value+h*(operator_function(['phi',2,1.],Z,cache)@(current-previous_nonlinear))
    return value,current
