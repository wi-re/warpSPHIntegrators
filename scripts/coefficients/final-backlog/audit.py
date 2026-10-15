"""Record individual pinned-source outcomes for the remaining backend candidates."""
import argparse,hashlib,json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('source',type=Path);p.add_argument('--output',type=Path,default=Path(__file__).with_name('audit.json'));args=p.parse_args()
root=args.source
rows=[]
def add(names,wave,status,package,files,reason,condition):
    paths=[f'lib/{package}/src/{f}' for f in files]
    sources={f:hashlib.sha256((root/f).read_bytes()).hexdigest() for f in paths}
    for name in names:
        assert any(name in (root/f).read_text() for f in paths),name
        rows.append(dict(name=name,wave=wave,status=status,source_sha256=sources,reason=reason,revisit_condition=condition))
add(['VCABM'],10,'deferred','OrdinaryDiffEqAdamsBashforthMoulton',['adams_bashforth_moulton_perform_step.jl','adams_bashforth_moulton_caches.jl','algorithms.jl'],
    'Existing AdaptiveAdams implements owned order-3–5 PECE and source neighbor decisions. Full order-1–12 startup/FSAL/rejection policy is a distinct controller. Warm 1–12 polynomial operations are now prototyped; current monomial/cardinal reconstruction becomes ill-conditioned at high orders. Extending the integer range would make an unqualified order claim, especially with DP5 startup above order six.',
    'Stable divided-difference/Nordsieck history with high-order startup, accepted/rejected transition oracle and measured equal-error benefit beyond the bounded backend.')
add(['QNDF','QNDF1','QNDF2'],10,'deferred','OrdinaryDiffEqBDF',['bdf_perform_step.jl','controllers.jl','algorithms.jl'],
    'NDF/AdaptiveNDF already implement source coefficient/difference mathematics with explicit package retry and discontinuity resets. Upstream consecutive-failure counts, Newton refresh/mass-matrix policy and source cache mutations are not equivalent to those controllers. Keep their source names unavailable rather than implying exact policy equivalence.',
    'A separate solver-cache invalidation/refresh protocol plus complete failure/retry/discontinuity trajectory oracle; no frontend dependency.')
add(['NordsieckBDF'],10,'deferred','OrdinaryDiffEqBDF',['algorithms.jl','bdf_perform_step.jl','controllers.jl'],
    'The source exists and propagates accepted Nordsieck arrays, with nonlinear tolerance budget and Jacobian reuse tied to local error. Pascal/rescale representation is qualified; the tensor warm Adams prototype does not implement BDF or its source controller. A new production BDF alias would hide this distinction.',
    'Native owned BDF corrector/history plus tolerance-budgeted Newton and cache-refresh control, benchmarked against AdaptiveNDF on stiff problems.')
add(['AN5','JVODE','JVODE_Adams','JVODE_BDF'],10,'prototype','OrdinaryDiffEqNordsieck',['algorithms.jl','nordsieck_perform_step.jl','nordsieck_utils.jl','controllers.jl'],
    'Actual source coefficient/Pascal/rescale kernels are independently checked at orders 1–12; bounded warm corrector prototype has known polynomial/AD/order gates. Source AN5 uses Tsit5 interpolant startup and at most three functional iterations; JVODE adds mutable order/wait/rewind policy. Neither complete runtime is advertised. JVODE_BDF only adds BDF handling in order-change memory while sharing Adams coefficient preparation; do not label the warm Adams map a BDF port.',
    'Owned startup/rewind/order state and independent full trajectories through failure and step changes; stiff BDF needs its own verified corrector.')
add(['AMF'],11,'prototype','OrdinaryDiffEqAMF',['amf.jl','operators.jl','linsolve.jl'],
    'Approximate matrix factorization is an ordered product of split Jacobian factors, not Adams. The dense product prototype checks noncommuting order and O(shift²) difference from the exact Rosenbrock matrix. Replacing the exact solve can alter method order; existing exact-JVP Rosenbrock does not silently accept approximate factors.',
    'Structured factor update/solve/transpose/gradient ownership, W-method order gates and measured benefit; generic RHS.linear alone is insufficient.')
add(['Tsit5DA','HybridExplicitImplicitRK'],11,'excluded','OrdinaryDiffEqRosenbrock',['algorithms.jl','rosenbrock_perform_step.jl'],
    'Hybrid explicit/linear-implicit DAE-aware tableau family with mass-matrix semantics. The ODE Tsitouras 5(4) method already exists; this is not another plain explicit tableau alias.',
    'A separately requested mass-matrix/DAE problem and consistent initialization contract.')
add(['ImplicitEulerExtrapolation','ImplicitEulerBarycentricExtrapolation','ImplicitDeuflhardExtrapolation','ImplicitHairerWannerExtrapolation'],13,'prototype','OrdinaryDiffEqExtrapolation',['algorithms.jl','extrapolation_perform_step.jl','controllers.jl'],
    'Dense frozen-Jacobian Euler/midpoint column prototypes now have nonlinear time order and complete AD gates at orders 2/4/6. They solve linear corrections, not nonlinear backward-Euler substeps. Production nonlinear/linear solve budgets, source divergence detector, adaptive order/work controls and full-state operator quadrature remain distinct.',
    'Matrix-free checked linear stages and lifecycle/quadrature ownership, then equal-error spatial cost evidence; source controller names remain unavailable.')
add(['RKIP'],13,'prototype','OrdinaryDiffEqRKIP',['algorithms.jl','rkip_perform_step.jl','rkip_cache.jl'],
    'Source is Runge–Kutta in the interaction picture, not Krylov-implicit Rosenbrock. Dense RK4 coordinate prototype is order-four and exact on linear problems. exp(-c*h*L) can overflow for stiff negative diffusion, unlike forward-only ETDRK maps.',
    'A reversible/skew-operator consumer and bounded forward/backward exponential action contract; source adaptive/tableau/cache handling then needs separate gates.')
add(['IRKC'],13,'excluded','OrdinaryDiffEqStabilizedIRK',['algorithms.jl','irkc_perform_step.jl'],
    'Actual pinned constant-cache recurrence with independent dense nonlinear solves reproduces the tensor translation but converges near order one on the retained nonlinear/time split control at degrees 2/4/10. Source also comments out nonlinear-failure checks. Retain as an audit specimen, not a second-order production option.',
    'Corrected source or separately derived recurrence with split nonlinear/time order, failure propagation and joint stability-region evidence.')
# Linear/multirate/Taylor are explicitly prototypes in the execution plan. These
# are contract probes rather than claims of equivalence to all source methods.
add(['CayleyEuler','LinearExponential','MagnusAdapt4'],14,'prototype','OrdinaryDiffEqLinear',['algorithms.jl','linear_perform_step.jl'],
    'Dense matrix-ODE contract probes exponential midpoint, commutator Magnus4 and norm-preserving Cayley midpoint on a rotating noncommuting system. CayleyEuler source may describe a different operator action; no source-name equivalence is inferred from these nearby constructions.',
    'Explicit matrix versus matrix-free action/transpose/mass contracts and measured operator preparation costs.')
add(['ExplicitTaylor','ExplicitTaylor2','ExplicitTaylorAdaptiveOrder'],15,'prototype','OrdinaryDiffEqTaylorSeries',['algorithms.jl'],
    'Order-3/4 nested total-JVP prototype includes tensor time and full state/parameter gradients. Measured six/ten RHS invocations per step and nested AD overhead do not justify a production adaptive-order method. Warp custom derivatives need higher-order AD support.',
    'Cost-competitive high-order derivative propagation on the actual RHS backend with explicit nested-AD compatibility.')
add(['MIS','MRAB','MREEF','MRIGARKERK22a','MRIGARKERK22b','MRIGARKERK33a','MRIGARKERK45a'],14,'prototype','OrdinaryDiffEqMultirate',['algorithms.jl','multirate_perform_step.jl','multirate_tableaus.jl'],
    'Generic order-two slow-forcing/fast-RK4 tensor contract is tested on noncommuting time-dependent splits and measured on an FFT-slow/local-fast consumer. It uses two slow evaluations plus four per microstep fast calls. This is not a coefficient port of each named source algorithm; source slow/fast conventions and modified fast forcing remain separate.',
    'An accepted macro/micro clock, interpolation and rollback contract plus equal-error cost win; individual source tableau/controller ports then need their own gates.')
add(['MREIL','MRIGARKESDIRK34a','MRIGARKESDIRK46a','MRIGARKIRK21a'],14,'deferred','OrdinaryDiffEqMultirateImplicit',['algorithms.jl'],
    'The explicit multirate prototype establishes component call accounting but not coupled implicit macro/micro solves, rejection or adjoints. Existing IMEX handles shared clocks; pretending it supplies multirate implicit semantics would be incorrect.',
    'Verified macro/micro solve, forcing interpolation, failed-substep rollback and gradient ownership following the explicit contract probe.')
args.output.write_text(json.dumps(dict(source_tree='bae9b8e5f1d00e51c554648e5094e215164a19f8',candidates=rows),indent=2)+'\n')
print(len(rows),'individual source outcomes')
