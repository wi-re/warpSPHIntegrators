from enum import Enum

# NOTE: this module is deliberately free of torch. It used to carry a
# `@torch.jit.script` decorator on the enum, which is a no-op at the Python level
# (`type(IntegrationSchemeType)` is still `enum.EnumType`) but forced a torch import
# and a TorchScript compilation for an otherwise pure-stdlib module.
class IntegrationSchemeType(Enum):
    forwardEuler = 0
    rungeKutta2 = 1
    heunsMethod = 2
    ralston2nd = 3
    rungeKutta3 = 4
    heunsMethod3rd = 5
    ralston3rd = 6
    wray3rd = 7
    sspRK3 = 8
    rungeKutta4 = 9
    rungeKutta4alt = 10
    nystrom5th = 11
    leapFrog = 12
    symplecticEuler = 13
    velocityVerlet = 14
    pefrl = 15
    vefrl = 16
    epec = 17
    epecModified = 18
    tvdRK3 = 19
    tvdRK2 = 20
    semiImplicitEuler = 21
    explicitEuler = 22
    # Embedded pairs. Bogacki-Shampine and Dormand-Prince are FSAL, so first-stage
    # reuse is exact for them; Cash-Karp is not.
    bogackiShampine = 23
    dormandPrince = 24
    cashKarp = 25
    # Diagonally implicit (NOTES.md S3.6 Phase 2). None implement first-stage reuse.
    backwardEuler = 26
    implicitMidpoint = 27
    trapezoidal = 28
    sdirk2 = 29
    # Explicit linear multistep (NOTES.md S3.6 Phase 1). `steps == order - 1`; none
    # implement first-stage reuse (a different, single-entry-lookback mechanism).
    ab2 = 30
    ab3 = 31
    ab4 = 32
    ab5 = 33
    abm2 = 34
    abm3 = 35
    abm4 = 36
    newmark = 37
    bdf1 = 38
    bdf2 = 39
    imexEuler = 40
    trbdf2 = 41
    bdf3 = 42
    esdirk324l2sa = 43
    esdirk436l2sa = 44
    # Additive (IMEX) Kennedy-Carpenter ARK pairs (NOTES.md S3.9 Phase 5). Each is an
    # explicit + implicit half pair; the combined method is not FSAL, so neither
    # implements first-stage reuse.
    ark324l2sa = 45
    ark436l2sa = 46
    # Higher-order BDF (NOTES.md S3.8 Phase 4). BDF1/2 are A-stable; BDF3-BDF5 are
    # A(alpha)-stable (cone half-angles 86.03 / 73.35 / 51.84 deg).
    bdf4 = 47
    bdf5 = 48
    # Fully implicit (iterated) Adams-Moulton correctors (NOTES.md S3.8 Phase 4),
    # distinct from the ABM PECE predictor-correctors above.
    am2 = 49
    am3 = 50
    am4 = 51
    # Relaxed Chebyshev / Lobatto super-timestepping (NOTES.md S3.13, Phase 12).
    # Explicit, matrix-free, O(s^2) real-axis stability; s is a per-step parameter
    # (pass s= or lambda_max=). None implement first-stage reuse.
    rkc1 = 52
    rkc2 = 53
    rkl2 = 54
    # Rosenbrock-W (linearly implicit, semilinear split) (NOTES.md S3.14, Phase 7).
    # Each stage is one GMRES solve against a frozen operator W (no outer Newton);
    # not stiffly accurate, so no first-stage reuse.
    ros3p = 55
    # Exponential integrator (NOTES.md S3.15, Phase 7): integrates the linear part
    # exactly via the matrix exponential / phi_k, matrix-free through a Krylov
    # approximation of phi_k(hL)v. Not stiffly accurate, so no first-stage reuse.
    etd2rk = 56
    # Exponential Rosenbrock (NOTES.md S3.16, Phase 7): order 3 with an embedded
    # order-2 estimator; freezes the FULL right-hand-side Jacobian (no semilinear
    # split needed), phi_k(hJn)v matrix-free through a Krylov approximation.
    # Exact on linear problems, L-stable for the linear part; not stiffly
    # accurate, so no first-stage reuse.
    exprb32 = 57
    # Coupled (block) fully implicit RK (NOTES.md S3.18, Phase 6): the stage
    # equations are solved as one s-by-s coupled system (fullyimplicit.BlockState),
    # so first-stage reuse does not apply to either. Gauss-Legendre 2 is order 4,
    # A-stable and symplectic; Radau IIA s=2 is order 3, L-stable, stiffly
    # accurate.
    gaussLegendre2 = 58
    radauIia2 = 59
    # IMEX linear multistep (NOTES.md S3.19, Phase 12): a BDF / trapezoidal
    # implicit backbone for the stiff part plus explicit endpoint extrapolation
    # for the smooth part (imexmultistep.py). A plain callable degenerates to
    # the pure-implicit limit (BDF2 / BDF3 / trapezoidal), so no first-stage
    # reuse for any of them (multistep reuse is the history= mechanism).
    sbdf2 = 60
    sbdf3 = 61
    cnab2 = 62
    # Higher-order SSP explicit RK (Phase 12 follow-up, 2026-09-15): Shu's
    # 10-stage order-4 SSP method (measured SSP coefficient 6.0 -- stage 2's
    # constant coefficient 1 - mu/6 is the binding constraint -- with a
    # negative real-axis stability interval of [-13.916, 0]). The bundled
    # SSP/TVD set previously stopped at order 3, exactly where the SSP
    # barrier for explicit RK bites (order 4 needs 5+ stages). Explicit
    # single-step, so no first-stage reuse.
    ssprk104 = 63


# --------------------------------------------------------------------------- #
# Family views of IntegrationSchemeType                                        #
# --------------------------------------------------------------------------- #
#
# IntegrationSchemeType is the single enum that captures every registered
# scheme, but a 64-member flat list does not tell a user which entry is an
# explicit scheme and which is an implicit one. The family enums below are
# thin VIEWS of the same members: every member keeps the same name and the
# same int value as its IntegrationSchemeType counterpart, so all four lookup
# forms below return the same scheme,
#
#     getIntegrator(ExplicitRK.rungeKutta4)
#     getIntegrator(IntegrationSchemeType.rungeKutta4)
#     getIntegrator('RK4')
#     getIntegrator('rungeKutta4')
#
# and the families are exhaustive and disjoint over the 64 members (pinned in
# tests/test_family_enums.py). `SCHEME_FAMILY` maps a scheme's identifier to
# the family enum class it belongs to; IntegrationScheme.family is the same
# fact on the registered scheme object.
class ExplicitRK(Enum):
    """One-step explicit RK-family schemes (no stage solve).

    Classical RK, the SSP and TVD variants, the embedded pairs, and the
    position/velocity (second-order-ODE) baselines `explicitEuler` and
    `nystrom5th`. Driver: the generic RK path (Butcher tableaus in
    `butcher.py`); `epec`/`epecModified` are PECE variants of the same form.
    """
    forwardEuler = IntegrationSchemeType.forwardEuler.value
    explicitEuler = IntegrationSchemeType.explicitEuler.value
    rungeKutta2 = IntegrationSchemeType.rungeKutta2.value
    heunsMethod = IntegrationSchemeType.heunsMethod.value
    ralston2nd = IntegrationSchemeType.ralston2nd.value
    rungeKutta3 = IntegrationSchemeType.rungeKutta3.value
    heunsMethod3rd = IntegrationSchemeType.heunsMethod3rd.value
    ralston3rd = IntegrationSchemeType.ralston3rd.value
    wray3rd = IntegrationSchemeType.wray3rd.value
    sspRK3 = IntegrationSchemeType.sspRK3.value
    ssprk104 = IntegrationSchemeType.ssprk104.value
    rungeKutta4 = IntegrationSchemeType.rungeKutta4.value
    rungeKutta4alt = IntegrationSchemeType.rungeKutta4alt.value
    nystrom5th = IntegrationSchemeType.nystrom5th.value
    bogackiShampine = IntegrationSchemeType.bogackiShampine.value
    dormandPrince = IntegrationSchemeType.dormandPrince.value
    cashKarp = IntegrationSchemeType.cashKarp.value
    epec = IntegrationSchemeType.epec.value
    epecModified = IntegrationSchemeType.epecModified.value
    tvdRK3 = IntegrationSchemeType.tvdRK3.value
    tvdRK2 = IntegrationSchemeType.tvdRK2.value


class Symplectic(Enum):
    """Geometric integrators for the position/velocity (second-order-ODE) form.

    Symplectic Euler and the symplectic splitting family (Leap Frog, Velocity
    Verlet, the Forest-Ruth PEFRL/VEFRL). Symplectic for separable
    Hamiltonians (force depending on position alone) -- with a
    velocity-dependent force the order-2+ members drop to first order
    (NOTES.md, 'Still open').
    """
    semiImplicitEuler = IntegrationSchemeType.semiImplicitEuler.value
    symplecticEuler = IntegrationSchemeType.symplecticEuler.value
    leapFrog = IntegrationSchemeType.leapFrog.value
    velocityVerlet = IntegrationSchemeType.velocityVerlet.value
    pefrl = IntegrationSchemeType.pefrl.value
    vefrl = IntegrationSchemeType.vefrl.value


class RelaxedChebyshev(Enum):
    """Relaxed Chebyshev / Lobatto super-timestepping (`rkc.py`).

    Explicit, matrix-free, real-axis stability interval [-K(s), 0] growing
    O(s^2) in the per-step stage count (pass `s=` or `lambda_max=`): RKC1
    K = 2s^2, RKC2 K = 2(s^2-1)/3, RKL2 K = (s^2+s-2)/2.
    """
    rkc1 = IntegrationSchemeType.rkc1.value
    rkc2 = IntegrationSchemeType.rkc2.value
    rkl2 = IntegrationSchemeType.rkl2.value


class MultistepExplicit(Enum):
    """Explicit linear multistep (Adams family, `multistep.py`).

    Adams-Bashforth 2-5 (pure predictor, one evaluation after startup) and
    the Adams-Bashforth-Moulton 2-4 PECE predictor-correctors (two
    evaluations after startup). All carry state-bearing `StepHistory`.
    """
    ab2 = IntegrationSchemeType.ab2.value
    ab3 = IntegrationSchemeType.ab3.value
    ab4 = IntegrationSchemeType.ab4.value
    ab5 = IntegrationSchemeType.ab5.value
    abm2 = IntegrationSchemeType.abm2.value
    abm3 = IntegrationSchemeType.abm3.value
    abm4 = IntegrationSchemeType.abm4.value


class DIRK(Enum):
    """Diagonally implicit Runge-Kutta (`dirk.py`, JFNK-closed by default).

    Backward Euler, Implicit Midpoint, Trapezoidal, SDIRK2, TR-BDF2 and the
    two ESDIRKs (explicit first stage + L[2] damping). The stiffly accurate
    tableaus with an explicit first stage accept lossless `priorStep` reuse.
    """
    backwardEuler = IntegrationSchemeType.backwardEuler.value
    implicitMidpoint = IntegrationSchemeType.implicitMidpoint.value
    trapezoidal = IntegrationSchemeType.trapezoidal.value
    sdirk2 = IntegrationSchemeType.sdirk2.value
    trbdf2 = IntegrationSchemeType.trbdf2.value
    esdirk324l2sa = IntegrationSchemeType.esdirk324l2sa.value
    esdirk436l2sa = IntegrationSchemeType.esdirk436l2sa.value


class CoupledRK(Enum):
    """Coupled (block) fully implicit RK (`fullyimplicit.py`).

    Non-triangular tableaus (a_ij != 0 for i != j), so the s stage equations
    are one s-by-s JFNK block solve over a BlockState, not s sequential stage
    solves. Gauss-Legendre 2 (order 4, A-stable, symplectic by measurement)
    and Radau IIA s=2 (order 3, L-stable, stiffly accurate).
    """
    gaussLegendre2 = IntegrationSchemeType.gaussLegendre2.value
    radauIia2 = IntegrationSchemeType.radauIia2.value


class MultistepImplicit(Enum):
    """Implicit linear multistep (BDF family, `bdf.py`/`multistep.py`).

    BDF1-BDF5 (A-stable through BDF2, A(alpha) cones 86.03/73.35/51.84 deg
    for BDF3-5) and the JFNK-corrected Adams-Moulton AM2-AM4. State-snapshot
    `StepHistory`; Dormand-Prince cold start until the history is full.
    """
    bdf1 = IntegrationSchemeType.bdf1.value
    bdf2 = IntegrationSchemeType.bdf2.value
    bdf3 = IntegrationSchemeType.bdf3.value
    bdf4 = IntegrationSchemeType.bdf4.value
    bdf5 = IntegrationSchemeType.bdf5.value
    am2 = IntegrationSchemeType.am2.value
    am3 = IntegrationSchemeType.am3.value
    am4 = IntegrationSchemeType.am4.value


class IMEX(Enum):
    """Additive IMEX schemes on an explicit/implicit split (`imex.py` and friends).

    IMEX Euler (one stage), the ARK3/4 Kennedy-Carpenter pairs (ERK + ESDIRK
    halves), and the IMEX linear multistep family SBDF2/SBDF3/CNAB2 (BDF /
    trapezoidal implicit backbone + explicit endpoint extrapolation). A plain
    callable runs the pure-implicit limit; pass an IMEXRHS to activate the
    split.
    """
    imexEuler = IntegrationSchemeType.imexEuler.value
    ark324l2sa = IntegrationSchemeType.ark324l2sa.value
    ark436l2sa = IntegrationSchemeType.ark436l2sa.value
    sbdf2 = IntegrationSchemeType.sbdf2.value
    sbdf3 = IntegrationSchemeType.sbdf3.value
    cnab2 = IntegrationSchemeType.cnab2.value


class LinearlyImplicit(Enum):
    """Rosenbrock-W (linearly implicit, semilinear split, `rosenbrock.py`).

    Each stage is one GMRES solve against a frozen operator W -- no outer
    Newton loop. ROS3P is order 3, A-stable (R(inf) = 1 - sqrt(3)), not
    L-stable.
    """
    ros3p = IntegrationSchemeType.ros3p.value


class Exponential(Enum):
    """Exponential integrators (`exponential.py`).

    Integrate the stiff part through the matrix exponential / phi_k
    functions, applied matrix-free via Krylov (Arnoldi). ETD2RK (order 2)
    needs the `linear` accessor (SemilinearRHS); EXPRB32 (order 3) freezes
    the full RHS Jacobian and runs on plain callables.
    """
    etd2rk = IntegrationSchemeType.etd2rk.value
    exprb32 = IntegrationSchemeType.exprb32.value


class Newmark(Enum):
    """Newmark beta/gamma integration for the position/velocity form (`newmark.py`).

    The average-acceleration (beta=1/4) variant is oscillator-unconditionally
    stable; JFNK-closed by default.
    """
    newmark = IntegrationSchemeType.newmark.value


FAMILY_ENUMS = (
    ExplicitRK,
    Symplectic,
    RelaxedChebyshev,
    MultistepExplicit,
    DIRK,
    CoupledRK,
    MultistepImplicit,
    IMEX,
    LinearlyImplicit,
    Exponential,
    Newmark,
)

#: Maps each IntegrationSchemeType member to the family enum class it belongs
#: to. Exhaustive and disjoint: every member of IntegrationSchemeType appears
#: in exactly one family (pinned in tests/test_family_enums.py).
SCHEME_FAMILY = {
    IntegrationSchemeType(member.value): family
    for family in FAMILY_ENUMS
    for member in family
}
