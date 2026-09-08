import functools
import warnings

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
    forwardEuler, RungeKutta2, midPoint, heunsMethod, ralston2nd, RungeKutta3,
    heunsMethod3rd, ralston3rd, Wray3rd, SSPRK3, RungeKutta4, RungeKutta4alt,
    Nystrom5th, BogackiShampine, DormandPrince, CashKarp, EPEC, EPECmodified,
)
from .verlet import leapFrog, symplecticEuler, velocityVerlet
from .tvd import TVDRK3, TVDRK2
from .ruth import PEFRL, VEFRL
from .dirk import backwardEuler as implicitBackwardEuler, implicitMidpoint, trapezoidal, SDIRK2
from .newmark import newmark
from .multistep import AB2, AB3, AB4, AB5, ABM2, ABM3, ABM4
from .bdf import BDF1, BDF2
from .imex import IMEXEuler
from .reuse import step_reuse_analysis, step_reuse_order, supports_step_reuse, is_fsal

semiImplicitEuler = lambda state, dt, f, *args, **kwargs: integrateSemiImplicitEuler(state, dt, f, *args, **kwargs)
explicitEuler = lambda state, dt, f, *args, **kwargs: integrateExplicitEuler(state, dt, f, *args, **kwargs)

IntegrationSchemes = []

from .util import IntegrationSchemeType

# The two boolean flags are (dissipation, nonLagrangian).
#
# `dissipation` is set to False exactly for the symplectic schemes (the Verlet
# family and the Forest-Ruth variants), which conserve a shadow Hamiltonian and so
# do not drift in energy secularly, and True for the plain Runge-Kutta schemes,
# which do. That is the only reading of the flag consistent with the values that
# were already recorded, and it removes the copy-paste inconsistencies noted in
# NOTES.md 2.13 (symplecticEuler was True while leapFrog was False; PEFRL was False
# while VEFRL was True).
#
# `nonLagrangian` now agrees with `dissipation` for every registered scheme, i.e. it
# carries no information. Nothing in the codebase reads either flag. Keep them for
# API compatibility, but `nonLagrangian` is a candidate for removal.

IntegrationSchemes.append(IntegrationScheme(forwardEuler, 'Forward Euler', IntegrationSchemeType.forwardEuler, 1, True, True))
IntegrationSchemes.append(IntegrationScheme(RungeKutta2,  'Midpoint',      IntegrationSchemeType.rungeKutta2, 2, True, True))
IntegrationSchemes.append(IntegrationScheme(heunsMethod,  'Heun\'s Method (2nd order)', IntegrationSchemeType.heunsMethod, 2, True, True))
IntegrationSchemes.append(IntegrationScheme(ralston2nd,   'Ralston\'s Method (2nd order)', IntegrationSchemeType.ralston2nd, 2, True, True))
IntegrationSchemes.append(IntegrationScheme(RungeKutta3,  'RK3',           IntegrationSchemeType.rungeKutta3, 3, True, True))
IntegrationSchemes.append(IntegrationScheme(heunsMethod3rd, 'Heun\'s Method (3rd order)', IntegrationSchemeType.heunsMethod3rd, 3, True, True))
IntegrationSchemes.append(IntegrationScheme(ralston3rd,   'Ralston\'s Method (3rd order)', IntegrationSchemeType.ralston3rd, 3, True, True))
IntegrationSchemes.append(IntegrationScheme(Wray3rd,      'Wray\'s Method (3rd order)', IntegrationSchemeType.wray3rd, 3, True, True))
IntegrationSchemes.append(IntegrationScheme(SSPRK3,       'SSP RK3',       IntegrationSchemeType.sspRK3, 3, True, True))
IntegrationSchemes.append(IntegrationScheme(RungeKutta4,  'RK4',           IntegrationSchemeType.rungeKutta4, 4, True, True))
IntegrationSchemes.append(IntegrationScheme(RungeKutta4alt, 'RK4 (alternative)', IntegrationSchemeType.rungeKutta4alt, 4, True, True))
IntegrationSchemes.append(IntegrationScheme(Nystrom5th,   'Nystrom 5th order', IntegrationSchemeType.nystrom5th, 5, True, True))
IntegrationSchemes.append(IntegrationScheme(BogackiShampine, 'Bogacki-Shampine 3(2)', IntegrationSchemeType.bogackiShampine, 3, True, True))
IntegrationSchemes.append(IntegrationScheme(DormandPrince, 'Dormand-Prince 5(4)', IntegrationSchemeType.dormandPrince, 5, True, True))
IntegrationSchemes.append(IntegrationScheme(CashKarp,     'Cash-Karp 5(4)', IntegrationSchemeType.cashKarp, 5, True, True))
IntegrationSchemes.append(IntegrationScheme(leapFrog, 'Leap Frog', IntegrationSchemeType.leapFrog, 2, False, False))
IntegrationSchemes.append(IntegrationScheme(symplecticEuler, 'Symplectic Euler', IntegrationSchemeType.symplecticEuler, 2, False, False))
IntegrationSchemes.append(IntegrationScheme(velocityVerlet, 'Velocity Verlet', IntegrationSchemeType.velocityVerlet, 2, False, False))
IntegrationSchemes.append(IntegrationScheme(PEFRL, 'PEFRL', IntegrationSchemeType.pefrl, 4, False, False))
IntegrationSchemes.append(IntegrationScheme(VEFRL, 'VEFRL', IntegrationSchemeType.vefrl, 4, False, False))
IntegrationSchemes.append(IntegrationScheme(EPEC, 'EPEC', IntegrationSchemeType.epec, 2, True, True))
IntegrationSchemes.append(IntegrationScheme(EPECmodified, 'EPEC Modified', IntegrationSchemeType.epecModified, 2, True, True))
IntegrationSchemes.append(IntegrationScheme(TVDRK3, 'TVD RK3', IntegrationSchemeType.tvdRK3, 3, True, True))
IntegrationSchemes.append(IntegrationScheme(TVDRK2, 'TVD RK2', IntegrationSchemeType.tvdRK2, 2, True, True))
# Semi-implicit (symplectic) Euler is first order, not second: it is one force
# evaluation per step, and measures 1.0 (see NOTES.md 2.13).
IntegrationSchemes.append(IntegrationScheme(semiImplicitEuler, 'Semi-Implicit Euler', IntegrationSchemeType.semiImplicitEuler, 1, False, False))
IntegrationSchemes.append(IntegrationScheme(explicitEuler, 'Explicit Euler', IntegrationSchemeType.explicitEuler, 1, True, True))

# ---- Diagonally implicit (NOTES.md S3.6 Phase 2) -------------------------- #
# JFNK is the default nonlinear solve, so these flags describe the converged method
# rather than an uncorrected fixed-count Picard iterate. Implicit midpoint is the
# symplectic Gauss-Legendre s=1 method once its stage equation is solved; the other
# implicit schemes retain their ordinary dissipative classifications.
IntegrationSchemes.append(IntegrationScheme(
    implicitBackwardEuler, 'Backward Euler (implicit)', IntegrationSchemeType.backwardEuler, 1, True, True,
    implicit=True, steps=1, stiffly_accurate=True, stability='L'))
IntegrationSchemes.append(IntegrationScheme(
    implicitMidpoint, 'Implicit Midpoint', IntegrationSchemeType.implicitMidpoint, 2, False, False,
    implicit=True, steps=1, stiffly_accurate=False, stability='A'))
IntegrationSchemes.append(IntegrationScheme(
    trapezoidal, 'Trapezoidal (Crank-Nicolson)', IntegrationSchemeType.trapezoidal, 2, True, True,
    implicit=True, steps=1, stiffly_accurate=True, stability='A'))
IntegrationSchemes.append(IntegrationScheme(
    SDIRK2, 'SDIRK2', IntegrationSchemeType.sdirk2, 2, True, True,
    implicit=True, steps=1, stiffly_accurate=True, stability='L'))
IntegrationSchemes.append(IntegrationScheme(
    newmark, 'Newmark', IntegrationSchemeType.newmark, 2, True, True,
    implicit=True, steps=1, stiffly_accurate=False, stability='A'))
IntegrationSchemes.append(IntegrationScheme(
    BDF1, 'BDF1', IntegrationSchemeType.bdf1, 1, True, True,
    implicit=True, steps=1, stiffly_accurate=True, stability='L', startup_order=1))
IntegrationSchemes.append(IntegrationScheme(
    BDF2, 'BDF2', IntegrationSchemeType.bdf2, 2, True, True,
    implicit=True, steps=1, stiffly_accurate=True, stability='A', startup_order=1))
IntegrationSchemes.append(IntegrationScheme(
    IMEXEuler, 'IMEX Euler', IntegrationSchemeType.imexEuler, 1, True, True,
    implicit=True, steps=1, stiffly_accurate=True, stability='A'))

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
    AB2, 'Adams-Bashforth 2', IntegrationSchemeType.ab2, 2, True, True,
    implicit=False, steps=1, startup_order=2))
IntegrationSchemes.append(IntegrationScheme(
    AB3, 'Adams-Bashforth 3', IntegrationSchemeType.ab3, 3, True, True,
    implicit=False, steps=2, startup_order=3))
IntegrationSchemes.append(IntegrationScheme(
    AB4, 'Adams-Bashforth 4', IntegrationSchemeType.ab4, 4, True, True,
    implicit=False, steps=3, startup_order=4))
IntegrationSchemes.append(IntegrationScheme(
    AB5, 'Adams-Bashforth 5', IntegrationSchemeType.ab5, 5, True, True,
    implicit=False, steps=4, startup_order=5))
IntegrationSchemes.append(IntegrationScheme(
    ABM2, 'Adams-Bashforth-Moulton 2 (PECE)', IntegrationSchemeType.abm2, 2, True, True,
    implicit=False, steps=1, startup_order=2))
IntegrationSchemes.append(IntegrationScheme(
    ABM3, 'Adams-Bashforth-Moulton 3 (PECE)', IntegrationSchemeType.abm3, 3, True, True,
    implicit=False, steps=2, startup_order=3))
IntegrationSchemes.append(IntegrationScheme(
    ABM4, 'Adams-Bashforth-Moulton 4 (PECE)', IntegrationSchemeType.abm4, 4, True, True,
    implicit=False, steps=3, startup_order=4))


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
    """Look a scheme up by display name, enum member, or enum member name."""
    for scheme in IntegrationSchemes:
        if (scheme.name == integrator
                or scheme.identifier == integrator
                or scheme.identifier.name == integrator):
            return scheme
    raise ValueError(f"Unknown integrator {integrator}")


def getIntegrationEnum(integrator):
    return getIntegrator(integrator).identifier
