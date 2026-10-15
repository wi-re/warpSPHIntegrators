"""Known behavior and explicit limits of bounded tensor backend experiments."""
import math
import pytest
import torch
from warpSPHIntegrators.experiments import *
D=torch.float64

def integrate(step,f,y,n):
    h=1/n
    for j in range(n):y=step(f,j*h,y,h)
    return y

@pytest.mark.parametrize('order',[3,4])
def test_total_taylor_derivatives_include_time_order_and_full_gradients(order):
    f=lambda t,y:-y+.1*y*y-.1*torch.exp(-2*t)
    errors=[]
    for n in (4,8,16):errors.append(abs(integrate(lambda f,t,y,h:taylor_step(f,t,y,h,order),f,torch.ones(1,dtype=D),n).item()-math.exp(-1)))
    assert math.log2(errors[-2]/errors[-1])>order-.15
    y=torch.tensor([.7],dtype=D,requires_grad=True);p=torch.tensor(2.,dtype=D,requires_grad=True)
    def run(y,p):return taylor_step(lambda t,z:-p*z-.1*z*z+.2*torch.sin(t),.3,y,.1,order).sum()
    out=run(y,p);dy,dp=torch.autograd.grad(out,(y,p));eps=1e-5
    assert dy.item()==pytest.approx(((run(y.detach()+eps,p.detach())-run(y.detach()-eps,p.detach()))/(2*eps)).item(),rel=1e-8)
    assert dp.item()==pytest.approx(((run(y.detach(),p.detach()+eps)-run(y.detach(),p.detach()-eps))/(2*eps)).item(),rel=1e-8)

@pytest.mark.parametrize('method,order',[('midpoint',2),('magnus4',4),('cayley2',2)])
def test_time_dependent_noncommuting_rotating_operator(method,order):
    J=torch.tensor([[0.,-1.],[1.,0.]],dtype=D);diag=torch.diag(torch.tensor([-1.,-3.],dtype=D))
    def R(t):return torch.matrix_exp(t*J)
    def A(t):return R(t)@diag@R(t).T+J
    initial=torch.tensor([1.,.4],dtype=D);exact=R(1.)@(torch.matrix_exp(diag)@initial);errors=[]
    for n in (8,16,32):
        value=initial
        for j in range(n):value=operator_step(A,j/n,value,1/n,method)
        errors.append(float((value-exact).abs().max()))
    assert math.log2(errors[-2]/errors[-1])>order-.15,(method,errors)
    if method=='cayley2':
        value=initial
        for j in range(100):value=operator_step(lambda t:J,j*.1,value,.1,method)
        assert value.norm().item()==pytest.approx(initial.norm().item(),abs=1e-13)

@pytest.mark.parametrize('ratio',[2,8,32])
def test_multirate_second_order_noncommuting_time_dependent_splits(ratio):
    A=torch.tensor([[-1.,.3],[0.,-2.]],dtype=D);B=torch.tensor([[0.,0.],[.2,-1.]],dtype=D);amplitude=torch.tensor([1.,.4],dtype=D)
    slow=lambda t,y:A@y+(-torch.exp(-torch.as_tensor(t))*amplitude-(A+B)@(amplitude*math.exp(-t)))
    fast=lambda t,y:B@y
    errors=[]
    for n in (8,16,32):
        value=amplitude
        for j in range(n):value=multirate_step(slow,fast,j/n,value,1/n,ratio)
        errors.append(float((value-amplitude*math.exp(-1)).abs().max()))
    assert math.log2(errors[-2]/errors[-1])>1.9
    with pytest.raises(ValueError):multirate_step(slow,fast,0.,amplitude,.1,3)


def test_interaction_picture_order_linear_exactness_and_diffusion_overflow_limit():
    L=torch.tensor([[-2.,.3],[.1,-3.]],dtype=D);amplitude=torch.tensor([1.,.4],dtype=D)
    def nonlinear(t,y):
        target=amplitude*math.exp(-t)
        return .1*y*y-target-L@target-.1*target*target
    errors=[]
    for n in (4,8,16):
        value=amplitude
        for j in range(n):value=interaction_picture_step(L,nonlinear,j/n,value,1/n)
        errors.append(float((value-amplitude*math.exp(-1)).abs().max()))
    assert math.log2(errors[-2]/errors[-1])>3.9
    zero=lambda t,y:torch.zeros_like(y)
    assert torch.allclose(interaction_picture_step(L,zero,0.,amplitude,.3),torch.matrix_exp(.3*L)@amplitude,atol=1e-15,rtol=0)
    # Source coordinate map needs exp(-c*h*L); a very stiff negative L overflows.
    assert not torch.isfinite(interaction_picture_step(-torch.eye(2,dtype=D)*10000,zero,0.,amplitude,.1)).all()

@pytest.mark.parametrize('base,order',[(b,p) for b in ('euler','midpoint') for p in (2,4,6)])
def test_frozen_jacobian_extrapolation_nonlinear_time_order_and_full_ad(base,order):
    f=lambda t,y:-y+.1*y*y-.1*torch.exp(-2*t)
    errors=[]
    for n in (4,8,16):
        out=integrate(lambda f,t,y,h:implicit_extrapolation_step(f,t,y,h,order,base),f,torch.ones(1,dtype=D),n)
        errors.append(abs(out.item()-math.exp(-1)))
    assert math.log2(errors[-2]/errors[-1])>order-.4,(base,order,errors)
    y=torch.tensor([.7],dtype=D,requires_grad=True);p=torch.tensor(2.,dtype=D,requires_grad=True)
    def run(y,p):return implicit_extrapolation_step(lambda t,z:-p*z-.1*z*z+.2*torch.sin(t),.3,y,.1,order,base).sum()
    dy,dp=torch.autograd.grad(run(y,p),(y,p));eps=1e-5
    assert dy.item()==pytest.approx(((run(y.detach()+eps,p.detach())-run(y.detach()-eps,p.detach()))/(2*eps)).item(),rel=1e-7)
    assert dp.item()==pytest.approx(((run(y.detach(),p.detach()+eps)-run(y.detach(),p.detach()-eps))/(2*eps)).item(),rel=1e-7)

@pytest.mark.parametrize('order',range(1,13))
def test_nordsieck_pascal_translation_rescaling_and_polynomial_exactness(order):
    jets=torch.tensor([[1/(j+1)] for j in range(order+1)],dtype=D)
    shifted=nordsieck_predict(jets)
    assert torch.allclose(nordsieck_predict(shifted,-1.),jets,atol=5e-10,rtol=0)
    assert torch.allclose(nordsieck_rescale(nordsieck_rescale(jets,.5),2.),jets,atol=0,rtol=0)
    # Exact degree-order polynomial in physical time, warm scaled jets at zero.
    h=.1;scaled=torch.stack([h**j*row for j,row in enumerate(jets)])
    def derivative(t,y):return torch.ones_like(y)*sum(j*t**(j-1)/(j+1) for j in range(1,order+1))
    updated=nordsieck_adams_step(derivative,0.,scaled,h)
    expected=sum(h**j/(j+1) for j in range(order+1))
    assert updated[0,0].item()==pytest.approx(expected,rel=0,abs=2e-15)

@pytest.mark.parametrize('order',[3,5,8,12])
def test_warm_nordsieck_order_from_exact_history_and_owned_failure(order):
    errors=[]
    for h in (.5,.25,.125):
        jets=torch.tensor([[(-h)**j/math.factorial(j)] for j in range(order+1)],dtype=D)
        before=jets.clone();out=nordsieck_adams_step(lambda t,y:-y,0.,jets,h)
        assert torch.equal(jets,before)
        errors.append(abs(out[0,0].item()-math.exp(-h)))
    usable=[math.log2(a/b) for a,b in zip(errors,errors[1:]) if b>1e-14]
    if usable:assert usable[-1]>order+.3,(order,errors)
    with pytest.raises(RuntimeError):nordsieck_adams_step(lambda t,y:y*y,0.,torch.ones(3,1,dtype=D),10.,maxiter=1)


def test_amf_noncommuting_product_and_approximation_order():
    a=torch.tensor([[-2.,1.],[0.,-1.]],dtype=D);b=torch.tensor([[-1.,0.],[.4,-3.]],dtype=D);I=torch.eye(2,dtype=D)
    for h in (.2,.1):
        P=amf_matrix((a,b),h)
        assert torch.allclose(P,(I-h*a)@(I-h*b),rtol=0,atol=0)
        assert torch.allclose(P-(I-h*(a+b)),h*h*(a@b),rtol=0,atol=3e-16)
        assert not torch.allclose(P,amf_matrix((b,a),h),rtol=0,atol=1e-10)
    for function in [lambda:operator_step(lambda t:torch.eye(65,dtype=D),0.,torch.ones(65,dtype=D),.1),
                     lambda:implicit_extrapolation_step(lambda t,y:-y,0.,torch.ones(65,dtype=D),.1)]:
        with pytest.raises(ValueError,match='64'):function()


def test_native_two_register_map_matches_expansion_and_rejects_ad():
    from warpSPHIntegrators import ExpandedRK,TensorState,tensor_rhs
    y=torch.linspace(.1,1.,16,dtype=D);f=lambda t,y:-y-.1*y*y+.2*t
    reference=ExpandedRK()(TensorState(y),.1,tensor_rhs(f))
    with torch.no_grad():value=low_storage_ck54_step(f,0.,y,.1)
    assert torch.allclose(value,reference.state.y,rtol=0,atol=3e-16)
    with pytest.raises(ValueError,match='no_grad'):low_storage_ck54_step(f,0.,y,.1)


@pytest.mark.parametrize('stages',[2,4,10])
def test_pinned_irkc_first_order_counterexample_and_failed_stage(stages):
    initial=torch.tensor([1.,.4],dtype=D)
    implicit=lambda t,y:-2*y-.1*y*y
    explicit=lambda t,y:y+.1*initial*initial*math.exp(-2*t)
    errors=[]
    for n in (8,16,32):
        value=initial
        for j in range(n):value=irkc_step(implicit,explicit,j/n,value,1/n,stages)
        errors.append(float((value-initial*math.exp(-1)).abs().max()))
    assert .8<math.log2(errors[-2]/errors[-1])<1.2,(stages,errors)
    with pytest.raises(RuntimeError):irkc_step(lambda t,y:y*y,lambda t,y:y,0.,initial,1.,stages,maxiter=1)


def test_actual_pinned_nordsieck_coefficients_and_irkc_step_bodies():
    import json
    from pathlib import Path
    import numpy as np
    source=json.loads((Path(__file__).parents[1]/'scripts/coefficients/final-backlog/source_controls.json').read_text())
    for row in source['nordsieck']:
        jets=torch.tensor(row['jets'],dtype=D)
        assert torch.allclose(nordsieck_predict(jets),torch.tensor(row['predicted'],dtype=D),atol=2e-12,rtol=0)
        assert torch.allclose(nordsieck_rescale(jets,.5),torch.tensor(row['rescaled'],dtype=D),atol=0,rtol=0)
        polynomial=np.polynomial.Polynomial([1.])
        for j in range(1,row['order']):polynomial*=np.polynomial.Polynomial([1.,1/j])
        primitive=polynomial.integ();constant=primitive(0)-primitive(-1)
        expected=np.r_[1.,primitive.coef[1:]/constant]
        assert np.allclose(expected,row['l'],atol=1e-11,rtol=1e-13)
    initial=torch.tensor([1.,.4],dtype=D)
    for row in source['irkc_steps']:
        out=irkc_step(lambda t,y:-2*y-.1*y*y,lambda t,y:y+.1*initial**2*math.exp(-2*t),0.,initial,row['h'],row['stages'])
        assert out.tolist()==pytest.approx(row['output'],rel=0,abs=1e-12)
    assert all(.8<r['order']<1.2 for r in source['irkc_refinements'])


def test_two_step_etd2_startup_warm_order_and_parameter_gradients():
    L=torch.tensor([[-2.]],dtype=D);start=torch.ones(1,dtype=D);errors=[]
    def nonlinear(t,y):return .1*y*y+math.exp(-t)-.1*math.exp(-2*t)
    for n in (16,32,64):
        value=start;previous=None
        for j in range(n):value,previous=exponential_adams2_step(L,nonlinear,j/n,value,1/n,previous)
        errors.append(abs(value.item()-math.exp(-1)))
    assert math.log2(errors[-2]/errors[-1])>1.9
    p=torch.tensor(2.,dtype=D,requires_grad=True)
    value,_=exponential_adams2_step(-p.reshape(1,1),nonlinear,0.,start,.1)
    grad=torch.autograd.grad(value.sum(),p)[0];eps=1e-5
    a=exponential_adams2_step(-(p.detach()+eps).reshape(1,1),nonlinear,0.,start,.1)[0]
    b=exponential_adams2_step(-(p.detach()-eps).reshape(1,1),nonlinear,0.,start,.1)[0]
    assert grad.item()==pytest.approx(((a-b)/(2*eps)).item(),rel=1e-8)
