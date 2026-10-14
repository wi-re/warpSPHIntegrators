"""Two masses, two understood modes: python examples/structural_modes.py.

The generalized-alpha step here is a dense LINEAR demo prototype, outside the
registry. It carries algorithmic acceleration, which is not the physical RHS
acceleration at an accepted endpoint. No production restart/adaptive/SPH contract
is implied. Classical parameters agree with PETSc TSALPHA2 after changing its
new-state alpha convention to the old-state weights used below.
"""
from dataclasses import dataclass
import math
import torch
from warpSPHIntegrators import getIntegrator,get_reference_state,testing


@dataclass(frozen=True)
class ModeState:
    q: torch.Tensor
    v: torch.Tensor
    a: torch.Tensor  # algorithmic acceleration; carry it across accepted steps
    t: float = 0.


@dataclass(frozen=True)
class LinearStructure:
    mass: torch.Tensor
    stiffness: torch.Tensor
    damping: torch.Tensor
    modes: torch.Tensor
    frequencies: torch.Tensor
    amplitudes: torch.Tensor

    def acceleration(self,q,v):
        return torch.linalg.solve(self.mass,-self.stiffness@q-self.damping@v)

    def initial(self):
        q=self.modes@self.amplitudes;v=torch.zeros_like(q)
        return ModeState(q,v,self.acceleration(q,v))

    def exact(self,t):
        # This reference is for the constructed undamped, unit-mass model.
        z=self.amplitudes*torch.cos(self.frequencies*t)
        zdot=-self.amplitudes*self.frequencies*torch.sin(self.frequencies*t)
        return self.modes@z,self.modes@zdot

    def modal_energy(self,state):
        z=self.modes.T@state.q;zdot=self.modes.T@state.v
        return (zdot.square()+self.frequencies.square()*z.square())/2


def two_mass_problem(slow=1.,fast=40.,fast_amplitude=.05):
    if not all(math.isfinite(x) for x in (slow,fast,fast_amplitude)) or slow<=0 or fast<=slow or fast_amplitude<=0:
        raise ValueError('finite positive amplitudes and 0 < slow < fast required')
    modes=torch.tensor([[1.,1.],[1.,-1.]],dtype=torch.float64)/math.sqrt(2)
    frequencies=torch.tensor([slow,fast],dtype=torch.float64)
    return LinearStructure(torch.eye(2,dtype=torch.float64),
        modes@torch.diag(frequencies.square())@modes.T,torch.zeros((2,2),dtype=torch.float64),
        modes,frequencies,torch.tensor([1.,fast_amplitude],dtype=torch.float64))


def alpha_parameters(rho_inf):
    if isinstance(rho_inf,bool) or not math.isfinite(rho_inf) or not 0<=rho_inf<=1:
        raise ValueError('rho_inf must be finite and in [0,1]')
    am=(2*rho_inf-1)/(rho_inf+1);af=rho_inf/(rho_inf+1)
    gamma=.5-am+af;beta=(1-am+af)**2/4
    return am,af,beta,gamma


def linear_alpha_step(state,model,dt,rho_inf,*,reset_acceleration=False,beta_override=None):
    """Classical fixed-step generalized-alpha, one dense acceleration solve.

    The two keyword controls are deliberately incorrect variants for diagnosing
    history/parameter mistakes; ordinary consumers leave both at their defaults.
    This prototype has zero external forcing and linear constant M/C/K only.
    """
    if isinstance(dt,bool) or not math.isfinite(dt) or dt<=0:raise ValueError('positive finite dt required')
    am,af,beta,gamma=alpha_parameters(rho_inf)
    if beta_override is not None:
        if not math.isfinite(beta_override) or beta_override<=0:raise ValueError('positive finite beta required')
        beta=beta_override
    q,v=state.q,state.v
    a=model.acceleration(q,v) if reset_acceleration else state.a
    q_predict=q+dt*v+dt**2*(.5-beta)*a
    v_predict=v+dt*(1-gamma)*a
    q_weighted=af*q+(1-af)*q_predict
    v_weighted=af*v+(1-af)*v_predict
    operator=(1-am)*model.mass+(1-af)*gamma*dt*model.damping+(1-af)*beta*dt**2*model.stiffness
    right=-am*(model.mass@a)-model.damping@v_weighted-model.stiffness@q_weighted
    a_next=torch.linalg.solve(operator,right)
    return ModeState(q_predict+beta*dt**2*a_next,v_predict+gamma*dt*a_next,a_next,float(state.t+dt))


def trajectory(model,dt,end,rho_inf,**controls):
    if not math.isfinite(end) or end<=0 or not math.isfinite(dt) or dt<=0 or not math.isclose(round(end/dt)*dt,end,abs_tol=1e-12):
        raise ValueError('positive fixed dt must divide the positive endpoint')
    state=model.initial();states=[state]
    for _ in range(round(end/dt)):
        state=linear_alpha_step(state,model,dt,rho_inf,**controls)
        if not all(torch.isfinite(value).all() for value in (state.q,state.v,state.a)):
            raise RuntimeError('Structural trajectory became nonfinite')
        states.append(state)
    return states


def registered_newmark_trajectory(model,dt,end):
    # Exercise the actual package driver on the same physical force/model.
    initial=model.initial();n=initial.q.numel()
    state=testing.ParticleSystem(testing.ParticleState(initial.q.clone(),initial.v.clone(),
            torch.zeros(n,dtype=torch.float64),torch.ones(n,dtype=torch.float64)))
    def rhs(system,dt,**kwargs):
        s=get_reference_state(system)
        return testing.ParticleUpdate(s.u,model.acceleration(s.x,s.u),s.e*0),None
    states=[initial];method=getIntegrator('Newmark')
    for _ in range(round(end/dt)):
        result=method(state,dt,rhs,solver_opts=dict(matvec='jvp',tol=1e-12,newton_tol=1e-7))
        if not result.step_status.tolerance_converged:raise RuntimeError('Registered Newmark solve did not reach tolerance')
        state=result.state;s=get_reference_state(state)
        states.append(ModeState(s.x,s.u,model.acceleration(s.x,s.u),state.t))
    return states


def generalized_alpha_trajectory(model,dt,end,rho_inf=.5):
    """Exercise the package's nonlinear protocol driver on the dense toy model."""
    from warpSPHIntegrators import GeneralizedAlpha, solve
    initial=model.initial();n=initial.q.numel()
    state=testing.ParticleSystem(testing.ParticleState(initial.q.clone(),initial.v.clone(),
            torch.zeros_like(initial.q),torch.ones_like(initial.q)))
    def rhs(system,dt,**kwargs):
        s=get_reference_state(system)
        return testing.ParticleUpdate(s.u,model.acceleration(s.x,s.u),s.e*0),None
    result=solve(GeneralizedAlpha(rho_inf),state,end,rhs,dt=dt,save_steps=True,
                 solver_opts=dict(matvec='jvp',tol=1e-12,newton_tol=1e-7))
    if not result.successful:raise RuntimeError(f'Generalized-alpha run failed: {result.termination}')
    # Samples contain physical fields. The final checkpoint also owns the
    # algorithmic acceleration needed to continue this damped trajectory.
    states=[initial]
    for sample in result.samples:
        if sample.t==initial.t:continue
        s=get_reference_state(sample.state)
        states.append(ModeState(s.x,s.u,model.acceleration(s.x,s.u),sample.t))
    return states,result


def amplification_matrix(mu,rho_inf,**controls):
    """Map on [q,h*v,h^2*a], using h=1; includes the algorithmic mode."""
    if not math.isfinite(mu) or mu<0:raise ValueError('finite nonnegative mu required')
    model=LinearStructure(torch.ones((1,1),dtype=torch.float64),torch.tensor([[mu**2]],dtype=torch.float64),
        torch.zeros((1,1),dtype=torch.float64),torch.ones((1,1),dtype=torch.float64),
        torch.tensor([mu],dtype=torch.float64),torch.ones(1,dtype=torch.float64))
    columns=[]
    for basis in torch.eye(3,dtype=torch.float64):
        out=linear_alpha_step(ModeState(basis[:1],basis[1:2],basis[2:3]),model,1.,rho_inf,**controls)
        columns.append(torch.cat((out.q,out.v,out.a)))
    return torch.stack(columns,dim=1)


def run_example():
    model=two_mass_problem();steps=trajectory(model,.1,12.,.5)
    before=model.modal_energy(steps[0]);after=model.modal_energy(steps[-1])
    ratios=after/before
    assert ratios[0]>.98 and ratios[1]<.01
    return ratios


if __name__=='__main__':
    ratios=run_example()
    print(f'Slow/fast modal energy retained: {ratios.tolist()}')
