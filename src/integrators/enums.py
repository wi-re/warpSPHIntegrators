from enum import Enum
import torch


@torch.jit.script
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
    
    