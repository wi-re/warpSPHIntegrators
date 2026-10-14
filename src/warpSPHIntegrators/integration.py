import functools
import difflib
import warnings

from enum import Enum

from .util import IntegrationScheme, updateStateEuler, updateStateSemiImplicitEuler
from .util import applyStateUpdate, applyPositionUpdate, applyVelocityUpdate, applyQuantityUpdate
from .specs import (
    StateBlend,
    ComponentUpdateSpec,
    PositionUpdateSpec,
    StageResult,
    StepEvaluation,
    IntegrationResult,
    blend_state,
    explicit_step,
    semi_implicit_position_step,
    verlet_position_step,
)
from .protocol import IntegrationSystem, BaseIntegrationSystem
from .fields import (
    BaseState,
    integrated,
    constant,
    copied,
    ephemeral,
    custom,
    tagged,
    reference_state,
    find_tagged_field,
    get_tagged_attr,
    set_tagged_attr,
    get_reference_state,
)
from .euler import integrateExplicitEuler, integrateSemiImplicitEuler
from .butcher import (
    forwardEuler, RungeKutta2, heunsMethod, ralston2nd, RungeKutta3,
    heunsMethod3rd, ralston3rd, Wray3rd, SSPRK3, SSPRK104, SSPRK53, SSPRK63,
    SSPRK73, SSPRK83, SSPRK54, KYKSSPRK42, KYK2014DGSSPRK3S2, SSPRK43, SSPRK432,
    SSPRK932, SSPRK53_2N1, SSPRK53_2N2, SSPRK53_H,
    Tsit5, OwrenZen3, OwrenZen4, OwrenZen5, BS5, Stepanov5, Alshina2, Alshina3,
    Alshina6, MSRK5, MSRK6, SIR54, Ralston4, RKO65, RKM, PSRK4p7q6, PSRK3p6q5,
    PSRK3p5q4,
    Vern6, Vern7, Vern8, Vern9, RKV76IIa, TanYam7, TsitPap8, DP8,
    Feagin10, Feagin12, Feagin14,
    RungeKutta4,
    RungeKutta4alt, Nystrom5th, BogackiShampine, DormandPrince, CashKarp, EPEC,
    EPECmodified,
)
from .verlet import leapFrog, symplecticEuler, velocityVerlet
from .tvd import TVDRK3, TVDRK2
from .ruth import PEFRL, VEFRL
from .dirk import (backwardEuler as implicitBackwardEuler, implicitMidpoint, trapezoidal,
                   SDIRK2, TRBDF2, ESDIRK324L2SA, ESDIRK436L2SA,
                   Kvaerno3, Kvaerno4, Kvaerno5, KenCarp3, KenCarp4, KenCarp47, KenCarp5,
                   KenCarp58, ESDIRK325L2SA, ESDIRK436L2SA2, ESDIRK437L2SA, ESDIRK547L2SA2,
                   ESDIRK54I8L2SA, ESDIRK659L2SA, SSPSDIRK2, ARS222, ARS232, ARS343, ARS443,
                   BHR553, Cash4, CFNLIRK3, Hairer4, Hairer42, SFSDIRK4, SFSDIRK5, SFSDIRK6,
                   SFSDIRK7, SFSDIRK8)
from .fullyimplicit import gaussLegendre2, radauIia2
from .ark import ARK324L2SA, ARK436L2SA
from .newmark import newmark
from .multistep import AB2, AB3, AB4, AB5, ABM2, ABM3, ABM4, AM2, AM3, AM4
from .ssprkmsvs import SSPRKMSVS32, SSPRKMSVS43
from .bdf import BDF1, BDF2, BDF3, BDF4, BDF5
from .imex import IMEXEuler
from .imexmultistep import SBDF2, SBDF3, CNAB2
from .rkc import RKC1, RKC2, RKL2
from .rosenbrock import integrateROS3P
from .exponential import integrateETD2RK, integrateEXPRB32
from .reuse import step_reuse_analysis, step_reuse_order, supports_step_reuse, is_fsal

semiImplicitEuler = lambda state, dt, f, *args, **kwargs: integrateSemiImplicitEuler(state, dt, f, *args, **kwargs)
explicitEuler = lambda state, dt, f, *args, **kwargs: integrateExplicitEuler(state, dt, f, *args, **kwargs)

IntegrationSchemes = []

from .util import IntegrationSchemeType

# The `dissipation` flag is set to False exactly for the symplectic schemes (the
# Verlet family and the Forest-Ruth variants), which conserve a shadow Hamiltonian
# and so do not drift in energy secularly, and True for the plain Runge-Kutta
# schemes, which do. That is the only reading of the flag consistent with the values
# that were already recorded, and it removes the copy-paste inconsistencies noted in
# NOTES.md 2.13 (symplecticEuler was True while leapFrog was False; PEFRL was False
# while VEFRL was True).
#
# The former `nonLagrangian` flag agreed with `dissipation` for every registered
# scheme, carried no information, and nothing read it, so it was removed
# (housekeeping, 2026-09-15).

IntegrationSchemes.append(IntegrationScheme(forwardEuler, 'Forward Euler', IntegrationSchemeType.forwardEuler, 1, True))
IntegrationSchemes.append(IntegrationScheme(RungeKutta2,  'Midpoint',      IntegrationSchemeType.rungeKutta2, 2, True))
IntegrationSchemes.append(IntegrationScheme(heunsMethod,  'Heun\'s Method (2nd order)', IntegrationSchemeType.heunsMethod, 2, True))
IntegrationSchemes.append(IntegrationScheme(ralston2nd,   'Ralston\'s Method (2nd order)', IntegrationSchemeType.ralston2nd, 2, True))
IntegrationSchemes.append(IntegrationScheme(RungeKutta3,  'RK3',           IntegrationSchemeType.rungeKutta3, 3, True))
IntegrationSchemes.append(IntegrationScheme(heunsMethod3rd, 'Heun\'s Method (3rd order)', IntegrationSchemeType.heunsMethod3rd, 3, True))
IntegrationSchemes.append(IntegrationScheme(ralston3rd,   'Ralston\'s Method (3rd order)', IntegrationSchemeType.ralston3rd, 3, True))
IntegrationSchemes.append(IntegrationScheme(Wray3rd,      'Wray\'s Method (3rd order)', IntegrationSchemeType.wray3rd, 3, True))
IntegrationSchemes.append(IntegrationScheme(SSPRK3,       'SSP RK3',       IntegrationSchemeType.sspRK3, 3, True))
IntegrationSchemes.append(IntegrationScheme(SSPRK104,     'SSPRK(10,4)',   IntegrationSchemeType.ssprk104, 4, True))
# The OrdinaryDiffEq wave-1 SSP tableaus (SCHEME_EXPANSION_PLAN, 2026-10-12);
# display names are ODE's verbatim.
IntegrationSchemes.append(IntegrationScheme(SSPRK53,      'SSPRK53',       IntegrationSchemeType.ssprk53, 3, True))
IntegrationSchemes.append(IntegrationScheme(SSPRK63,      'SSPRK63',       IntegrationSchemeType.ssprk63, 3, True))
IntegrationSchemes.append(IntegrationScheme(SSPRK73,      'SSPRK73',       IntegrationSchemeType.ssprk73, 3, True))
IntegrationSchemes.append(IntegrationScheme(SSPRK83,      'SSPRK83',       IntegrationSchemeType.ssprk83, 3, True))
IntegrationSchemes.append(IntegrationScheme(SSPRK54,      'SSPRK54',       IntegrationSchemeType.ssprk54, 4, True))
IntegrationSchemes.append(IntegrationScheme(KYKSSPRK42,   'KYKSSPRK42',    IntegrationSchemeType.kykssprk42, 2, True))
IntegrationSchemes.append(IntegrationScheme(KYK2014DGSSPRK3S2, 'KYK2014DGSSPRK_3S2', IntegrationSchemeType.kyk2014dgssprk3s2, 2, True))
# The OrdinaryDiffEq wave-2 SSP tableaus (SCHEME_EXPANSION_PLAN, 2026-10-12);
# display names are ODE's verbatim. SSPRK43/SSPRK432 are 3(2) embedded pairs
# (order-3 main, order-2 estimate); SSPRK932 is fixed-step order 3.
IntegrationSchemes.append(IntegrationScheme(SSPRK43,      'SSPRK43',       IntegrationSchemeType.ssprk43, 3, True))
IntegrationSchemes.append(IntegrationScheme(SSPRK432,     'SSPRK432',      IntegrationSchemeType.ssprk432, 3, True))
IntegrationSchemes.append(IntegrationScheme(SSPRK932,     'SSPRK932',      IntegrationSchemeType.ssprk932, 3, True))
# The OrdinaryDiffEq wave-5 SSP tableaus (SCHEME_EXPANSION_PLAN, 2026-10-13);
# display names are ODE's verbatim. Higueras & Roldan low-storage third-order
# SSP variants, screened and confirmed to be distinct one-step maps from
# SSPRK53 (NOTES.md S3.25).
IntegrationSchemes.append(IntegrationScheme(SSPRK53_2N1,  'SSPRK53_2N1',   IntegrationSchemeType.ssprk53_2n1, 3, True))
IntegrationSchemes.append(IntegrationScheme(SSPRK53_2N2,  'SSPRK53_2N2',   IntegrationSchemeType.ssprk53_2n2, 3, True))
IntegrationSchemes.append(IntegrationScheme(SSPRK53_H,    'SSPRK53_H',     IntegrationSchemeType.ssprk53_h, 3, True))
IntegrationSchemes.append(IntegrationScheme(RungeKutta4,  'RK4',           IntegrationSchemeType.rungeKutta4, 4, True))
IntegrationSchemes.append(IntegrationScheme(RungeKutta4alt, 'RK4 (alternative)', IntegrationSchemeType.rungeKutta4alt, 4, True))
IntegrationSchemes.append(IntegrationScheme(Nystrom5th,   'Nystrom 5th order', IntegrationSchemeType.nystrom5th, 5, True))
IntegrationSchemes.append(IntegrationScheme(BogackiShampine, 'Bogacki-Shampine 3(2)', IntegrationSchemeType.bogackiShampine, 3, True))
IntegrationSchemes.append(IntegrationScheme(DormandPrince, 'Dormand-Prince 5(4)', IntegrationSchemeType.dormandPrince, 5, True))
IntegrationSchemes.append(IntegrationScheme(CashKarp,     'Cash-Karp 5(4)', IntegrationSchemeType.cashKarp, 5, True))
# The OrdinaryDiffEq wave-6 tableaus (SCHEME_EXPANSION_PLAN, 2026-10-13):
# Tsit5 (Tsitouras 5(4), ODE's recommended default) and the LowOrderRK
# workhorses. Tsit5 / OwrenZen3 / OwrenZen4 / OwrenZen5 / BS5 / Stepanov5 /
# Alshina2 are (p, p-1) embedded pairs on the generic RK driver (the main
# weight vector propagates, the embedded one is the local error estimate);
# the rest are fixed-step plain tableaus. All one-step explicit and
# dissipative: plain Runge-Kutta, no shadow-Hamiltonian conservation.
IntegrationSchemes.append(IntegrationScheme(Tsit5,        'Tsit5 5(4)',    IntegrationSchemeType.tsit5, 5, True))
IntegrationSchemes.append(IntegrationScheme(OwrenZen3,    'OwrenZen3 3(2)', IntegrationSchemeType.owrenzen3, 3, True))
IntegrationSchemes.append(IntegrationScheme(OwrenZen4,    'OwrenZen4 4(3)', IntegrationSchemeType.owrenzen4, 4, True))
IntegrationSchemes.append(IntegrationScheme(OwrenZen5,    'OwrenZen5 5(4)', IntegrationSchemeType.owrenzen5, 5, True))
IntegrationSchemes.append(IntegrationScheme(BS5,          'BS5 5(4)',      IntegrationSchemeType.bs5, 5, True))
IntegrationSchemes.append(IntegrationScheme(Stepanov5,    'Stepanov5 5(4)', IntegrationSchemeType.stefanov5, 5, True))
IntegrationSchemes.append(IntegrationScheme(Alshina2,     'Alshina2 2(1)', IntegrationSchemeType.arushina2, 2, True))
IntegrationSchemes.append(IntegrationScheme(Alshina3,     'Alshina3',      IntegrationSchemeType.arushina3, 3, True))
IntegrationSchemes.append(IntegrationScheme(Alshina6,     'Alshina6',      IntegrationSchemeType.arushina6, 6, True))
IntegrationSchemes.append(IntegrationScheme(MSRK5,        'MSRK5',         IntegrationSchemeType.msrk5, 5, True))
IntegrationSchemes.append(IntegrationScheme(MSRK6,        'MSRK6',         IntegrationSchemeType.msrk6, 6, True))
IntegrationSchemes.append(IntegrationScheme(SIR54,        'SIR54',         IntegrationSchemeType.sir54, 5, True))
IntegrationSchemes.append(IntegrationScheme(Ralston4,     'Ralston4',      IntegrationSchemeType.raston4, 4, True))
IntegrationSchemes.append(IntegrationScheme(RKO65,        'RKO65',         IntegrationSchemeType.rko65, 5, True))
IntegrationSchemes.append(IntegrationScheme(RKM,          'RKM',           IntegrationSchemeType.rkm, 3, True))
IntegrationSchemes.append(IntegrationScheme(PSRK4p7q6,    'PSRK4p7q6',     IntegrationSchemeType.psrk4p7q6, 4, True))
IntegrationSchemes.append(IntegrationScheme(PSRK3p6q5,    'PSRK3p6q5',     IntegrationSchemeType.psrk3p6q5, 3, True))
IntegrationSchemes.append(IntegrationScheme(PSRK3p5q4,    'PSRK3p5q4',     IntegrationSchemeType.psrk3p5q4, 3, True))
# The OrdinaryDiffEq wave-7 high-order tableaus (SCHEME_EXPANSION_PLAN,
# 2026-10-13): the Verner family, Tsitouras's high-order embedded pairs
# (TanYam7 7(6), TsitPap8 8(7)), Dormand-Prince 8, and the Feagin 10/12/14
# symmetric methods. Coefficients from ODE v7.8.1's caches, cross-checked
# one step against the real ODE solvers (NOTES.md 3.27). The embedded pairs
# (Vern6-9, RKV76IIa, TanYam7, TsitPap8) store `b` as a (main, embedded)
# tuple; DP8 and the Feagin methods are fixed-step plain tableaus (DP8's ODE
# estimate is a dual 5th+3rd-order companion, not a (p-1) solution). FSAL:
# Vern6 only. All one-step explicit and dissipative.
IntegrationSchemes.append(IntegrationScheme(Vern6,        'Vern6 6(5)',    IntegrationSchemeType.vern6, 6, True))
IntegrationSchemes.append(IntegrationScheme(Vern7,        'Vern7 7(6)',    IntegrationSchemeType.vern7, 7, True))
IntegrationSchemes.append(IntegrationScheme(Vern8,        'Vern8 8(7)',    IntegrationSchemeType.vern8, 8, True))
IntegrationSchemes.append(IntegrationScheme(Vern9,        'Vern9 9(8)',    IntegrationSchemeType.vern9, 9, True))
IntegrationSchemes.append(IntegrationScheme(RKV76IIa,     'RKV76IIa 7(6)', IntegrationSchemeType.rkv76iia, 7, True))
IntegrationSchemes.append(IntegrationScheme(TanYam7,      'TanYam7 7(6)',  IntegrationSchemeType.tanyam7, 7, True))
IntegrationSchemes.append(IntegrationScheme(TsitPap8,     'TsitPap8 8(7)', IntegrationSchemeType.tsitpap8, 8, True))
IntegrationSchemes.append(IntegrationScheme(DP8,          'DP8',           IntegrationSchemeType.dp8, 8, True))
IntegrationSchemes.append(IntegrationScheme(Feagin10,     'Feagin10',      IntegrationSchemeType.feagin10, 10, True))
IntegrationSchemes.append(IntegrationScheme(Feagin12,     'Feagin12',      IntegrationSchemeType.feagin12, 12, True))
IntegrationSchemes.append(IntegrationScheme(Feagin14,     'Feagin14',      IntegrationSchemeType.feagin14, 14, True))
IntegrationSchemes.append(IntegrationScheme(leapFrog, 'Leap Frog', IntegrationSchemeType.leapFrog, 2, False))
IntegrationSchemes.append(IntegrationScheme(symplecticEuler, 'Symplectic Euler', IntegrationSchemeType.symplecticEuler, 2, False))
IntegrationSchemes.append(IntegrationScheme(velocityVerlet, 'Velocity Verlet', IntegrationSchemeType.velocityVerlet, 2, False))
IntegrationSchemes.append(IntegrationScheme(PEFRL, 'PEFRL', IntegrationSchemeType.pefrl, 4, False))
IntegrationSchemes.append(IntegrationScheme(VEFRL, 'VEFRL', IntegrationSchemeType.vefrl, 4, False))
IntegrationSchemes.append(IntegrationScheme(EPEC, 'EPEC', IntegrationSchemeType.epec, 2, True))
IntegrationSchemes.append(IntegrationScheme(EPECmodified, 'EPEC Modified', IntegrationSchemeType.epecModified, 2, True))
IntegrationSchemes.append(IntegrationScheme(TVDRK3, 'TVD RK3', IntegrationSchemeType.tvdRK3, 3, True))
IntegrationSchemes.append(IntegrationScheme(TVDRK2, 'TVD RK2', IntegrationSchemeType.tvdRK2, 2, True))
# Semi-implicit (symplectic) Euler is first order, not second: it is one force
# evaluation per step, and measures 1.0 (see NOTES.md 2.13).
IntegrationSchemes.append(IntegrationScheme(semiImplicitEuler, 'Semi-Implicit Euler', IntegrationSchemeType.semiImplicitEuler, 1, False))
IntegrationSchemes.append(IntegrationScheme(explicitEuler, 'Explicit Euler', IntegrationSchemeType.explicitEuler, 1, True))

# ---- Diagonally implicit (NOTES.md S3.6 Phase 2) -------------------------- #
# JFNK is the default nonlinear solve, so these flags describe the converged method
# rather than an uncorrected fixed-count Picard iterate. Implicit midpoint is the
# symplectic Gauss-Legendre s=1 method once its stage equation is solved; the other
# implicit schemes retain their ordinary dissipative classifications.
IntegrationSchemes.append(IntegrationScheme(
    implicitBackwardEuler, 'Backward Euler (implicit)', IntegrationSchemeType.backwardEuler, 1, True,
    implicit=True, steps=1, stiffly_accurate=True, stability='L'))
IntegrationSchemes.append(IntegrationScheme(
    implicitMidpoint, 'Implicit Midpoint', IntegrationSchemeType.implicitMidpoint, 2, False,
    implicit=True, steps=1, stiffly_accurate=False, stability='A'))
IntegrationSchemes.append(IntegrationScheme(
    trapezoidal, 'Trapezoidal (Crank-Nicolson)', IntegrationSchemeType.trapezoidal, 2, True,
    implicit=True, steps=1, stiffly_accurate=True, stability='A'))
IntegrationSchemes.append(IntegrationScheme(
    SDIRK2, 'SDIRK2', IntegrationSchemeType.sdirk2, 2, True,
    implicit=True, steps=1, stiffly_accurate=True, stability='L'))
IntegrationSchemes.append(IntegrationScheme(
    TRBDF2, 'TR-BDF2', IntegrationSchemeType.trbdf2, 2, True,
    implicit=True, steps=1, stiffly_accurate=True, stability='L'))
IntegrationSchemes.append(IntegrationScheme(
    ESDIRK324L2SA, 'ESDIRK3(2)4L[2]SA', IntegrationSchemeType.esdirk324l2sa, 3, True,
    implicit=True, steps=1, stiffly_accurate=True, stability='L'))
IntegrationSchemes.append(IntegrationScheme(
    ESDIRK436L2SA, 'ESDIRK4(3)6L[2]SA', IntegrationSchemeType.esdirk436l2sa, 4, True,
    implicit=True, steps=1, stiffly_accurate=True, stability='L'))
# ---- Wave 8: the remaining ODE v7.8.1 DIRK tableaus (NOTES.md 3.28) -------- #
# All on the JFNK DIRK driver. Coefficients from the ODE v7.8.1 tableaus,
# verified one step against the real ODE solvers (NOTES.md 3.28). Stability
# strings from a left-half-plane stability-function grid (same file): the
# sub-A set is L-stable except kvaerno4 (A-stable, R(inf) ~ 0.72) and
# sspsdirk2 (A-stable, R(inf) -> 1); the ARS family, bhr553, cash4, hairer4,
# hairer42 and cfnlirk3 are L-stable too; the sfstdirk4-8 family has only a
# bounded real-axis stability region (no A(alpha) sector), so it registers
# stability=None like the other bounded-region schemes.
IntegrationSchemes.append(IntegrationScheme(
    Kvaerno3, 'Kvaerno3', IntegrationSchemeType.kvaerno3, 3, True,
    implicit=True, steps=1, stiffly_accurate=True, stability='L'))
IntegrationSchemes.append(IntegrationScheme(
    Kvaerno4, 'Kvaerno4', IntegrationSchemeType.kvaerno4, 4, True,
    implicit=True, steps=1, stiffly_accurate=True, stability='A'))
IntegrationSchemes.append(IntegrationScheme(
    Kvaerno5, 'Kvaerno5', IntegrationSchemeType.kvaerno5, 5, True,
    implicit=True, steps=1, stiffly_accurate=True, stability='L'))
IntegrationSchemes.append(IntegrationScheme(
    KenCarp3, 'KenCarp3', IntegrationSchemeType.kencarp3, 3, True,
    implicit=True, steps=1, stiffly_accurate=True, stability='L'))
IntegrationSchemes.append(IntegrationScheme(
    KenCarp4, 'KenCarp4', IntegrationSchemeType.kencarp4, 4, True,
    implicit=True, steps=1, stiffly_accurate=True, stability='L'))
IntegrationSchemes.append(IntegrationScheme(
    KenCarp47, 'KenCarp47', IntegrationSchemeType.kencarp47, 4, True,
    implicit=True, steps=1, stiffly_accurate=True, stability='L'))
IntegrationSchemes.append(IntegrationScheme(
    KenCarp5, 'KenCarp5', IntegrationSchemeType.kencarp5, 5, True,
    implicit=True, steps=1, stiffly_accurate=True, stability='L'))
IntegrationSchemes.append(IntegrationScheme(
    KenCarp58, 'KenCarp58', IntegrationSchemeType.kencarp58, 5, True,
    implicit=True, steps=1, stiffly_accurate=True, stability='L'))
IntegrationSchemes.append(IntegrationScheme(
    ESDIRK325L2SA, 'ESDIRK3(2)5L[2]SA', IntegrationSchemeType.esdirk325l2sa, 3, True,
    implicit=True, steps=1, stiffly_accurate=True, stability='L'))
IntegrationSchemes.append(IntegrationScheme(
    ESDIRK436L2SA2, 'ESDIRK4(3)6L[2]SA2', IntegrationSchemeType.esdirk436l2sa2, 4, True,
    implicit=True, steps=1, stiffly_accurate=True, stability='L'))
IntegrationSchemes.append(IntegrationScheme(
    ESDIRK437L2SA, 'ESDIRK4(3)7L[2]SA', IntegrationSchemeType.esdirk437l2sa, 4, True,
    implicit=True, steps=1, stiffly_accurate=True, stability='L'))
IntegrationSchemes.append(IntegrationScheme(
    ESDIRK547L2SA2, 'ESDIRK5(4)7L[2]SA2', IntegrationSchemeType.esdirk547l2sa2, 5, True,
    implicit=True, steps=1, stiffly_accurate=True, stability='L'))
IntegrationSchemes.append(IntegrationScheme(
    ESDIRK54I8L2SA, 'ESDIRK5(4)I8L[2]SA', IntegrationSchemeType.esdirk54i8l2sa, 5, True,
    implicit=True, steps=1, stiffly_accurate=True, stability='L'))
IntegrationSchemes.append(IntegrationScheme(
    ESDIRK659L2SA, 'ESDIRK6(5)9L[2]SA', IntegrationSchemeType.esdirk659l2sa, 6, True,
    implicit=True, steps=1, stiffly_accurate=True, stability='L'))
IntegrationSchemes.append(IntegrationScheme(
    SSPSDIRK2, 'SSPSDIRK2', IntegrationSchemeType.sspsdirk2, 2, False,
    # Two implicit-midpoint half-steps: symplectic, R(infinity) = +1.
    implicit=True, steps=1, stiffly_accurate=False, stability='A'))
IntegrationSchemes.append(IntegrationScheme(
    ARS222, 'ARS222', IntegrationSchemeType.ars222, 2, True,
    implicit=True, steps=1, stiffly_accurate=True, stability='L'))
IntegrationSchemes.append(IntegrationScheme(
    ARS232, 'ARS232', IntegrationSchemeType.ars232, 2, True,
    implicit=True, steps=1, stiffly_accurate=True, stability='L'))
IntegrationSchemes.append(IntegrationScheme(
    ARS343, 'ARS343', IntegrationSchemeType.ars343, 3, True,
    implicit=True, steps=1, stiffly_accurate=True, stability='L'))
IntegrationSchemes.append(IntegrationScheme(
    ARS443, 'ARS443', IntegrationSchemeType.ars443, 3, True,
    implicit=True, steps=1, stiffly_accurate=True, stability='L'))
IntegrationSchemes.append(IntegrationScheme(
    BHR553, 'BHR553', IntegrationSchemeType.bhr553, 3, True,
    implicit=True, steps=1, stiffly_accurate=True, stability='L'))
IntegrationSchemes.append(IntegrationScheme(
    Cash4, 'Cash4', IntegrationSchemeType.cash4, 4, True,
    implicit=True, steps=1, stiffly_accurate=False, stability='L'))
IntegrationSchemes.append(IntegrationScheme(
    CFNLIRK3, 'CFNLIRK3', IntegrationSchemeType.cfnlirk3, 3, True,
    implicit=True, steps=1, stiffly_accurate=True, stability='L'))
IntegrationSchemes.append(IntegrationScheme(
    Hairer4, 'Hairer4', IntegrationSchemeType.hairer4, 4, True,
    implicit=True, steps=1, stiffly_accurate=True, stability='L'))
IntegrationSchemes.append(IntegrationScheme(
    Hairer42, 'Hairer42', IntegrationSchemeType.hairer42, 4, True,
    implicit=True, steps=1, stiffly_accurate=True, stability='L'))
IntegrationSchemes.append(IntegrationScheme(
    SFSDIRK4, 'SFSDIRK4', IntegrationSchemeType.sfstdirk4, 4, True,
    implicit=True, steps=1, stiffly_accurate=False))
IntegrationSchemes.append(IntegrationScheme(
    SFSDIRK5, 'SFSDIRK5', IntegrationSchemeType.sfstdirk5, 4, True,
    implicit=True, steps=1, stiffly_accurate=False))
IntegrationSchemes.append(IntegrationScheme(
    SFSDIRK6, 'SFSDIRK6', IntegrationSchemeType.sfstdirk6, 4, True,
    implicit=True, steps=1, stiffly_accurate=False))
IntegrationSchemes.append(IntegrationScheme(
    SFSDIRK7, 'SFSDIRK7', IntegrationSchemeType.sfstdirk7, 4, True,
    implicit=True, steps=1, stiffly_accurate=False))
IntegrationSchemes.append(IntegrationScheme(
    SFSDIRK8, 'SFSDIRK8', IntegrationSchemeType.sfstdirk8, 4, True,
    implicit=True, steps=1, stiffly_accurate=False))
# ---- Coupled (block) fully implicit RK (NOTES.md S3.18, Phase 6) ----------- #
# The stage equations are coupled (a_ij != 0 for i != j), so they are solved as
# one s-by-s block system (fullyimplicit.BlockState) rather than s sequential
# stage solves -- a DIRK driver structurally cannot take these tableaus. Both
# carry a null-stage order-2 companion (the 2-stage-only order-2 pair is
# degenerate, i.e. b itself, for both). Neither implements first-stage reuse
# (no explicit first stage to splice into); pass warmStart=result.stages for
# the block-solve analogue of a good initial guess. Gauss-Legendre 2 is
# symplectic for separable Hamiltonians (pinned by measurement in
# tests/test_hamiltonian.py -- its tableau is NOT A-symmetric), so
# dissipation=False like the Verlet family.
IntegrationSchemes.append(IntegrationScheme(
    gaussLegendre2, 'Gauss-Legendre 2', IntegrationSchemeType.gaussLegendre2, 4, False,
    implicit=True, steps=1, stiffly_accurate=False, stability='A'))
IntegrationSchemes.append(IntegrationScheme(
    radauIia2, 'Radau IIA s=2', IntegrationSchemeType.radauIia2, 3, True,
    implicit=True, steps=1, stiffly_accurate=True, stability='L'))
# Additive (IMEX) Kennedy-Carpenter ARK pairs (NOTES.md S3.9 Phase 5). Each is an
# explicit + implicit half; the combined method is *not* FSAL (the explicit half is
# not stiffly accurate), so `stiffly_accurate=False` and priorStep is rejected. The
# order is the *combined* additive order (3 / 4); pass an IMEXRHS to activate the
# split, or an ordinary RHS for the pure-implicit (ESDIRK) limit.
IntegrationSchemes.append(IntegrationScheme(
    ARK324L2SA, 'ARK3(2)4L[2]SA', IntegrationSchemeType.ark324l2sa, 3, True,
    implicit=True, steps=1, stiffly_accurate=False, stability='L'))
IntegrationSchemes.append(IntegrationScheme(
    ARK436L2SA, 'ARK4(3)6L[2]SA', IntegrationSchemeType.ark436l2sa, 4, True,
    implicit=True, steps=1, stiffly_accurate=False, stability='L'))
IntegrationSchemes.append(IntegrationScheme(
    newmark, 'Newmark', IntegrationSchemeType.newmark, 2, True,
    implicit=True, steps=1, stiffly_accurate=False, stability='A'))
IntegrationSchemes.append(IntegrationScheme(
    BDF1, 'BDF1', IntegrationSchemeType.bdf1, 1, True,
    implicit=True, steps=1, stiffly_accurate=True, stability='L', startup_order=1))
IntegrationSchemes.append(IntegrationScheme(
    BDF2, 'BDF2', IntegrationSchemeType.bdf2, 2, True,
    implicit=True, steps=1, stiffly_accurate=True, stability='A', startup_order=1))
IntegrationSchemes.append(IntegrationScheme(
    BDF3, 'BDF3', IntegrationSchemeType.bdf3, 3, True,
    implicit=True, steps=2, stiffly_accurate=True, stability='A(alpha)', startup_order=3))
# BDF4/5: same JFNK-closed multistep residual, one state snapshot per past step.
# A(alpha), not A-stable -- the stability region is a lobe whose closest approach
# to the negative real axis is 73.35 deg (BDF4) and 51.84 deg (BDF5) from it, so
# stiff oscillatory modes outside those cones are not damped (tests/test_bdf.py).
IntegrationSchemes.append(IntegrationScheme(
    BDF4, 'BDF4', IntegrationSchemeType.bdf4, 4, True,
    implicit=True, steps=3, stiffly_accurate=True, stability='A(alpha)', startup_order=4))
IntegrationSchemes.append(IntegrationScheme(
    BDF5, 'BDF5', IntegrationSchemeType.bdf5, 5, True,
    implicit=True, steps=4, stiffly_accurate=True, stability='A(alpha)', startup_order=5))
IntegrationSchemes.append(IntegrationScheme(
    IMEXEuler, 'IMEX Euler', IntegrationSchemeType.imexEuler, 1, True,
    implicit=True, steps=1, stiffly_accurate=True, stability='A'))
# ---- IMEX linear multistep (NOTES.md S3.19, Phase 12) ---------------------- #
# BDF / trapezoidal implicit backbone for the stiff part, explicit endpoint
# extrapolation for the smooth part (imexmultistep.py). Pass an IMEXRHS to
# activate the split; an ordinary RHS runs the pure-implicit limit (BDF2 /
# BDF3 / the trapezoidal rule), exactly as the ARK pairs and IMEX Euler do.
# `steps` is the history depth (order - 1), the BDF family's convention; cold
# calls bootstrap from Dormand-Prince 5(4). `stability` describes the implicit
# backbone (the pure-implicit limit) -- the explicit part carries the matching
# AB-like real-axis restriction, measured in tests/test_imexmultistep.py.
# priorStep is refused: multistep reuse is the history= mechanism, not
# first-stage splicing.
IntegrationSchemes.append(IntegrationScheme(
    SBDF2, 'SBDF2', IntegrationSchemeType.sbdf2, 2, True,
    implicit=True, steps=1, stiffly_accurate=True, stability='L', startup_order=2))
IntegrationSchemes.append(IntegrationScheme(
    SBDF3, 'SBDF3', IntegrationSchemeType.sbdf3, 3, True,
    implicit=True, steps=2, stiffly_accurate=True, stability='A(alpha)', startup_order=3))
IntegrationSchemes.append(IntegrationScheme(
    CNAB2, 'CNAB2', IntegrationSchemeType.cnab2, 2, True,
    implicit=True, steps=1, stiffly_accurate=True, stability='A', startup_order=2))

# ---- Explicit linear multistep (NOTES.md S3.6 Phase 1) -------------------- #
# `dissipation=True` for all seven, measured directly (max relative energy error on
# `oscillator` grows ~7-9x over an 8x-longer run for every one) and consistent with
# NOTES.md S3.6's own citation (Tang 1993: no linear multistep method is symplectic
# for a general Hamiltonian). `startup_order` equals each scheme's own `order`, not a
# capped value: the Dormand-Prince 5(4) starter these schemes bootstrap from (S3.7
# pain point 1) has order 5, at or above every one of these, so none of them lose
# order to a low-quality cold start the way a self-starting Euler bootstrap would --
# confirmed empirically (`tests/test_multistep.py`), not assumed from the starter's
# order alone.
IntegrationSchemes.append(IntegrationScheme(
    AB2, 'Adams-Bashforth 2', IntegrationSchemeType.ab2, 2, True,
    implicit=False, steps=1, startup_order=2))
IntegrationSchemes.append(IntegrationScheme(
    AB3, 'Adams-Bashforth 3', IntegrationSchemeType.ab3, 3, True,
    implicit=False, steps=2, startup_order=3))
IntegrationSchemes.append(IntegrationScheme(
    AB4, 'Adams-Bashforth 4', IntegrationSchemeType.ab4, 4, True,
    implicit=False, steps=3, startup_order=4))
IntegrationSchemes.append(IntegrationScheme(
    AB5, 'Adams-Bashforth 5', IntegrationSchemeType.ab5, 5, True,
    implicit=False, steps=4, startup_order=5))
IntegrationSchemes.append(IntegrationScheme(
    ABM2, 'Adams-Bashforth-Moulton 2 (PECE)', IntegrationSchemeType.abm2, 2, True,
    implicit=False, steps=1, startup_order=2))
IntegrationSchemes.append(IntegrationScheme(
    ABM3, 'Adams-Bashforth-Moulton 3 (PECE)', IntegrationSchemeType.abm3, 3, True,
    implicit=False, steps=2, startup_order=3))
IntegrationSchemes.append(IntegrationScheme(
    ABM4, 'Adams-Bashforth-Moulton 4 (PECE)', IntegrationSchemeType.abm4, 4, True,
    implicit=False, steps=3, startup_order=4))

# ---- SSP multistep (NOTES.md S3.23, SCHEME_EXPANSION_PLAN wave 3) ---------- #
# Shu's strong-stability-preserving multistep methods (ssprkmsvs.py): convex
# combinations of forward-Euler substeps across past states. SSPRKMSVS32 is
# 3-step / order 2 / SSP C = 1/2, SSPRKMSVS43 is 4-step / order 3 / SSP C = 1/3.
# One right-hand-side evaluation per step once past the cold start (the past
# derivatives are reused from the history). `dissipation=True` as for the Adams
# family (an explicit linear multistep method, so not symplectic for a general
# Hamiltonian -- Tang 1993). Fixed-step only: no embedded error estimate, so
# `result.error is None` and adaptive stepping is refused by construction.
IntegrationSchemes.append(IntegrationScheme(
    SSPRKMSVS32, 'SSPRKMSVS32', IntegrationSchemeType.ssprkmsvs32, 2, True,
    implicit=False, steps=2, startup_order=2))
IntegrationSchemes.append(IntegrationScheme(
    SSPRKMSVS43, 'SSPRKMSVS43', IntegrationSchemeType.ssprkmsvs43, 3, True,
    implicit=False, steps=3, startup_order=3))

# ---- Fully implicit Adams-Moulton (NOTES.md S3.8 Phase 4) ------------------ #
# The AM corrector formula solved to convergence with JFNK, as distinct from the
# uniterated PECE corrector above. AM2 is the trapezoidal rule, so it inherits
# A-stability; AM3/AM4 have only a bounded stability region around the origin
# (no named class to advertise). History carries the past *derivatives* (the
# `update` of each entry); a matching Adams-Bashforth predictor is the default
# initial guess (`predictor=False` starts the solve from the known part).
IntegrationSchemes.append(IntegrationScheme(
    AM2, 'Adams-Moulton 2 (implicit)', IntegrationSchemeType.am2, 2, True,
    implicit=True, steps=1, stiffly_accurate=False, stability='A', startup_order=2))
IntegrationSchemes.append(IntegrationScheme(
    AM3, 'Adams-Moulton 3 (implicit)', IntegrationSchemeType.am3, 3, True,
    implicit=True, steps=2, stiffly_accurate=False, startup_order=3))
IntegrationSchemes.append(IntegrationScheme(
    AM4, 'Adams-Moulton 4 (implicit)', IntegrationSchemeType.am4, 4, True,
    implicit=True, steps=3, stiffly_accurate=False, startup_order=4))

# ---- Stabilised explicit super-timestepping (NOTES.md S3.13, Phase 12) ---- #
# RKC/RKL trade a variable number `s` of cheap right-hand-side evaluations for a
# real-axis stability interval that grows like O(s^2), so the step size is set by
# the physics rather than the stiffest diffusive eigenvalue. They are fully explicit
# (no solve, no linear-RHS requirement -- the full `f` is evaluated at every stage),
# one-step, and dissipative. `stability` is left None: the interval is `[-K(s), 0]`
# with K depending on the per-step stage count (pass `s=` or `lambda_max=`), so there
# is no single region to advertise. RKL2 is Meyer/Balsara/Aslam 2014; RKC2 is the same
# construction with Chebyshev in place of Legendre; RKC1 is the order-1 limit.
IntegrationSchemes.append(IntegrationScheme(
    RKC1, 'RKC1', IntegrationSchemeType.rkc1, 1, True,
    implicit=False, steps=1, stiffly_accurate=False, stability=None))
IntegrationSchemes.append(IntegrationScheme(
    RKC2, 'RKC2', IntegrationSchemeType.rkc2, 2, True,
    implicit=False, steps=1, stiffly_accurate=False, stability=None))
IntegrationSchemes.append(IntegrationScheme(
    RKL2, 'RKL2', IntegrationSchemeType.rkl2, 2, True,
    implicit=False, steps=1, stiffly_accurate=False, stability=None))

# ---- Rosenbrock-W (NOTES.md S3.14, Phase 7) --------------------------------- #
# Linearly-implicit Rosenbrock-W on the semilinear split f = L·y + N: each of the
# three stages is ONE GMRES solve against the frozen operator W (the Jacobian of f
# at the step start, or the linear part), so there is no outer Newton loop. ROS3P
# is order 3, A-stable but *not* L-stable (R(inf) = 1 - sqrt(3) ~= -0.732: stiff
# modes are damped to |1 - sqrt(3)| per step, not killed). It is *not* stiffly
# accurate (the last stage is not the step), so priorStep is rejected. Pass a
# SemilinearRHS to activate the split; the frozen operator W is selected with
# `w=` ('jvp' exact / 'fd' finite-difference / 'linear' the L part only -- the
# latter is order 3 only when f is linear in the state, first order otherwise).
IntegrationSchemes.append(IntegrationScheme(
    integrateROS3P, 'ROS3P', IntegrationSchemeType.ros3p, 3, True,
    implicit=True, steps=1, stiffly_accurate=False, stability='A'))


# ---- Exponential integrator (NOTES.md S3.15, Phase 7) ---------------------- #
# ETD2RK: a two-stage, order-2 exponential Runge-Kutta method for the semilinear
# split f = L·y + N. The linear part L is integrated EXACTLY through the matrix
# exponential and the entire phi_k functions, applied matrix-free as
# phi_k(hL)v via a Krylov (Arnoldi) approximation; the nonlinear remainder N is
# quadratured (two evaluations per step). It is *explicit* in the stage sense
# (no stage solves for its own value), so `implicit=False`, and it is L-stable
# for the linear part (e^{hL} damps stiff modes to zero as h|lambda| -> inf). It
# is *not* stiffly accurate (the last stage is not the step), so priorStep is
# rejected. Pass a SemilinearRHS (or an RHS providing `linear`); a plain
# callable has no linear part and is rejected before the solve.
IntegrationSchemes.append(IntegrationScheme(
    integrateETD2RK, 'ETD2RK', IntegrationSchemeType.etd2rk, 2, True,
    implicit=False, steps=1, stiffly_accurate=False, stability='L'))


# ---- Exponential Rosenbrock (NOTES.md S3.16, Phase 7) ---------------------- #
# EXPRB32: a two-stage, order-3 exponential Rosenbrock method (Hochbrueck,
# Ostermann & Schweitzer 2009) with an embedded order-2 estimator. Where ETD2RK
# needs the semilinear split f = L·y + N, it freezes the FULL right-hand-side
# Jacobian Jn = Df(tn, yn) and carries the stiffness through the phi_k(hJn)
# functions, applied matrix-free via Krylov (two builds per step, three with the
# non-autonomous f_t term). It is *explicit* in the stage sense (Jn is never
# inverted), exact on linear problems, L-stable for the linear part (scalar
# stability function e^{h lambda}), and *not* stiffly accurate (the stage U2 is
# not the step), so priorStep is rejected. It runs on plain callables -- no
# `linear` part is needed; the frozen operator is selected with `w=` ('jvp'
# exact / 'fd' finite-difference) and the time derivative with `f_t=`.
IntegrationSchemes.append(IntegrationScheme(
    integrateEXPRB32, 'EXPRB32', IntegrationSchemeType.exprb32, 3, True,
    implicit=False, steps=1, stiffly_accurate=False, stability='L'))


# --------------------------------------------------------------------------- #
# Reuse metadata + guard                                                       #
# --------------------------------------------------------------------------- #

_warned_reuse = set()


def _with_reuse_guard(scheme: IntegrationScheme) -> IntegrationScheme:
    """Attach the reuse analysis to a scheme and warn (once) if reuse costs order.

    The warning is deliberately not an error: trading a known order loss for half the
    right-hand-side evaluations is a legitimate choice, and one the SPH literature
    makes routinely. The caller is only entitled to know which trade they are making.
    """
    analysis = step_reuse_analysis(scheme)
    inner = scheme.function

    @functools.wraps(inner)
    def guarded(state, dt, f, *args, **kwargs):
        if kwargs.get('priorStep') is not None and scheme.name not in _warned_reuse:
            if analysis.order is None:
                _warned_reuse.add(scheme.name)
                warnings.warn(
                    f"{scheme.name}: {analysis.reason}. The supplied priorStep will be ignored.",
                    RuntimeWarning, stacklevel=2)
            elif analysis.order < scheme.order:
                _warned_reuse.add(scheme.name)
                warnings.warn(
                    f"{scheme.name}: first-stage reuse (priorStep) drops the convergence order "
                    f"from {scheme.order} to {analysis.order}, because {analysis.reason}. "
                    f"This halves the right-hand-side cost and may still be the right trade; "
                    f"use integrators.supports_step_reuse(scheme) to check beforehand, or pick "
                    f"an FSAL scheme (Bogacki-Shampine 3(2), Dormand-Prince 5(4)) for lossless reuse.",
                    RuntimeWarning, stacklevel=2)
        return inner(state, dt, f, *args, **kwargs)

    # functools.wraps does not carry non-function attributes set on the original.
    if hasattr(inner, 'butcherTableau'):
        guarded.butcherTableau = inner.butcherTableau
    if hasattr(inner, 'dirkTableau'):
        guarded.dirkTableau = inner.dirkTableau
    if hasattr(inner, 'blockTableau'):
        guarded.blockTableau = inner.blockTableau
    if hasattr(inner, 'arkTableau'):
        guarded.arkTableau = inner.arkTableau
    return scheme._replace(function=guarded, reuse_order=analysis.order, fsal=analysis.fsal)


IntegrationSchemes[:] = [_with_reuse_guard(scheme) for scheme in IntegrationSchemes]


def getPreferredScheme(order):
    if order == 1:
        return semiImplicitEuler
    elif order == 2:
        return symplecticEuler
    elif order == 3:
        return TVDRK3
    elif order == 4:
        return RungeKutta4
    elif order == 5:
        return Nystrom5th
    else:
        raise ValueError(f"No scheme for order {order}")


def getIntegrator(integrator):
    """Look a scheme up by display name, enum member, or enum member name.

    `integrator` may be a display name ('RK4'), an enum member name
    ('rungeKutta4'), an `IntegrationSchemeType` member, or a member of any of
    the family enums (`ExplicitRK`, `DIRK`, ... in `enums.FAMILY_ENUMS`) —
    the family members keep the same names and values, so every form
    resolves to the same registered scheme.
    """
    for scheme in IntegrationSchemes:
        if (scheme.name == integrator
                or scheme.identifier == integrator
                or scheme.identifier.name == integrator
                or (isinstance(integrator, Enum)
                    and scheme.identifier.value == integrator.value)):
            return scheme
    if isinstance(integrator, str):
        close = difflib.get_close_matches(
            integrator,
            [scheme.name for scheme in IntegrationSchemes]
            + [member.name for member in IntegrationSchemeType],
            n=3)
        hint = f" Did you mean: {', '.join(close)}?" if close else ""
    else:
        hint = (" (valid: a display name, an enum member name, an "
                "IntegrationSchemeType member, or a family-enum member)")
    raise ValueError(f"Unknown integrator {integrator!r}.{hint}")


def getIntegrationEnum(integrator):
    return getIntegrator(integrator).identifier
