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
