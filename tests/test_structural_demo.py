"""Understood structural consumer: exact modes, damping, order and memory."""
import importlib.util
import math
from pathlib import Path
import sys
import pytest
import torch

spec=importlib.util.spec_from_file_location('structural_example',Path(__file__).resolve().parents[1]/'examples/structural_modes.py')
demo=importlib.util.module_from_spec(spec);sys.modules[spec.name]=demo;spec.loader.exec_module(demo)


def test_two_mass_model_has_the_declared_exact_modes_and_energy():
    model=demo.two_mass_problem()
    assert torch.allclose(model.stiffness@model.modes,model.modes@torch.diag(model.frequencies.square()),atol=1e-12)
    assert torch.allclose(model.modes.T@model.modes,torch.eye(2,dtype=torch.float64),atol=1e-15)
    expected=model.modal_energy(model.initial())
    for time in (.13,1.,12.):
        q,v=model.exact(time);state=demo.ModeState(q,v,model.acceleration(q,v),time)
        assert torch.allclose(model.modal_energy(state),expected,atol=1e-12)


def test_rho_one_matches_the_actual_registered_newmark_trajectory():
    model=demo.two_mass_problem()
    prototype=demo.trajectory(model,.1,2.,1.)
    registered=demo.registered_newmark_trajectory(model,.1,2.)
    assert max(float((a.q-b.q).abs().max()) for a,b in zip(prototype,registered))<1e-9
    assert max(float((a.v-b.v).abs().max()) for a,b in zip(prototype,registered))<1e-9


def test_weighted_equilibrium_and_caller_ownership_hold():
    model=demo.two_mass_problem();state=model.initial()
    before=[a.clone() for a in (state.q,state.v,state.a)]
    am,af,_,_=demo.alpha_parameters(.5)
    next_state=demo.linear_alpha_step(state,model,.1,.5)
    force=model.stiffness@(af*state.q+(1-af)*next_state.q)
    residual=model.mass@(am*state.a+(1-am)*next_state.a)+force
    assert torch.linalg.vector_norm(residual)<1e-10
    assert all(torch.equal(a,b) for a,b in zip((state.q,state.v,state.a),before))
    # Algorithmic acceleration is not a fresh endpoint physical acceleration.
    assert torch.linalg.vector_norm(next_state.a-model.acceleration(next_state.q,next_state.v))>1.


@pytest.mark.parametrize('rho',[0.,.5,1.])
def test_resolved_refinement_is_second_order_and_reset_control_detects_loss(rho):
    model=demo.two_mass_problem(slow=1.,fast=3.);grid=(.025,.0125,.00625)
    for reset in (False,True):
        errors=[]
        for h in grid:
            state=demo.trajectory(model,h,2.,rho,reset_acceleration=reset)[-1]
            q,v=model.exact(2.);errors.append(float(torch.linalg.vector_norm(state.q-q)+torch.linalg.vector_norm(state.v-v)))
        order=math.log(errors[-2]/errors[-1],2)
        expected=1 if reset and rho<1 else 2
        assert order==pytest.approx(expected,abs=.08)


@pytest.mark.parametrize('rho',[0.,.5,.8,1.])
def test_augmented_amplification_is_bounded_and_has_the_requested_limit(rho):
    for mu in torch.logspace(-3,3,40,dtype=torch.float64).tolist():
        radius=float(torch.linalg.eigvals(demo.amplification_matrix(mu,rho)).abs().max())
        assert radius<=1+1e-8
    limit=float(torch.linalg.eigvals(demo.amplification_matrix(1e6,rho)).abs().max())
    assert limit==pytest.approx(rho,abs=2e-4)


def test_underresolved_fast_mode_is_suppressed_with_small_slow_mode_loss():
    ratios=demo.run_example()
    assert ratios[0]>.999 and ratios[1]<1e-15


def test_classical_parameters_and_pinned_beta_negative_control_are_distinct():
    assert demo.alpha_parameters(1.)==(.5,.5,.25,.5)
    assert demo.alpha_parameters(0.)==(-1.,0.,1.,1.5)
    radius=float(torch.linalg.eigvals(demo.amplification_matrix(4.,1.,beta_override=.0625)).abs().max())
    assert radius>5  # cached expression fails a simple high-frequency anchor
    for rho in (-.1,1.1,float('nan'),float('inf'),True):
        with pytest.raises(ValueError):demo.alpha_parameters(rho)
    for h in (0,-1,float('nan')):
        with pytest.raises(ValueError):demo.linear_alpha_step(demo.two_mass_problem().initial(),demo.two_mass_problem(),h,.5)


def test_dense_prototype_keeps_initial_state_and_stiffness_gradients():
    def forward(value,track):
        k=torch.tensor(value,dtype=torch.float64,requires_grad=track)
        q=torch.ones(1,dtype=torch.float64,requires_grad=track);v=q*0
        model=demo.LinearStructure(torch.eye(1,dtype=torch.float64),k.reshape(1,1),torch.zeros((1,1),dtype=torch.float64),
            torch.eye(1,dtype=torch.float64),torch.tensor([2.],dtype=torch.float64),q)
        state=demo.ModeState(q,v,model.acceleration(q,v))
        for _ in range(5):state=demo.linear_alpha_step(state,model,.1,.5)
        return state.q.sum()+state.v.sum(),q,k
    output,q,k=forward(4.,True);gq,gk=torch.autograd.grad(output,(q,k));eps=1e-5
    finite=(forward(4.+eps,False)[0]-forward(4.-eps,False)[0])/(2*eps)
    assert math.isfinite(gq.item())
    assert gk.item()==pytest.approx(finite.item(),rel=1e-7,abs=1e-9)
