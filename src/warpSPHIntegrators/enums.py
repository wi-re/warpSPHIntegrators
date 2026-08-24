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
