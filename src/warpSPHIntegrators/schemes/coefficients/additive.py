"""Validated additive RK coefficient definitions; no numerical driver dependency."""
from typing import NamedTuple
import numpy as np

class AdditiveTableau(NamedTuple):
    """An additive (IMEX) Runge-Kutta tableau: one explicit + one implicit half.

    Both halves share the node vector ``c``. ``a_explicit`` is strictly lower
    triangular (its stages are explicit), ``a_implicit`` is lower triangular with a
    possibly nonzero diagonal (those stages are solved implicitly). ``b_*`` propagate,
    ``d_*`` are the embedded (lower-order) weights for the error estimate.
    """

    c: np.ndarray
    a_explicit: np.ndarray
    a_implicit: np.ndarray
    b_explicit: np.ndarray
    b_implicit: np.ndarray
    d_explicit: np.ndarray
    d_implicit: np.ndarray


def getARKTableau(scheme: str) -> AdditiveTableau:
    if scheme == 'ARK324L2SA':
        # Kennedy-Carpenter ARK(2,3)4L[2]SA: 4 stages, implicit half order 3,
        # explicit half order 2, L[2]-stable. SUNDIALS ARKODE v7.9.0
        # (ARKODE_ARK324L2SA_{ERK,DIRK}_4_2_3). The implicit half (a_implicit,
        # b_implicit) is identical in a/b/c to the registered ESDIRK3(2)4L[2]SA;
        # only the embedded d differs (the ARK pair shares one d across both halves).
        g = 0.4358665215084589994160194511935568425293
        c = np.array([0.0,
                      0.8717330430169179988320389023871136850586,
                      0.6,
                      1.0])
        a_explicit = np.array([
            [0.0, 0.0, 0.0, 0.0],
            [0.8717330430169179988320389023871136850586, 0.0, 0.0, 0.0],
            [0.52758901197630041156180797140291790433,
             0.07241098802369958843819202859708209566999, 0.0, 0.0],
            [0.3990960076760701320627260736092142797856,
             -0.437557654613519443722846363831022571942,
             1.038461646937449311660120290221808292156, 0.0],
        ])
        a_implicit = np.array([
            [0.0, 0.0, 0.0, 0.0],
            [g, g, 0.0, 0.0],
            [0.2576482460664272457999960162840797092643,
             -0.09351476757488624521601546747763655179361, g, 0.0],
            [0.1876410243467238251612921441668043913795,
             -0.5952974735769549480478230275858851737782,
             0.9717899277217721234705114322255239398694, g],
        ])
        b = np.array([0.1876410243467238251612921441668043913795,
                      -0.5952974735769549480478230275858851737782,
                      0.9717899277217721234705114322255239398694,
                      g])
        d = np.array([0.2147402862233891404862383406484193714659,
                      -0.4851622638849390928209050808398155895845,
                      0.86872500252038755116621237682951240796,
                      0.4016969751411624011684543633618838101586])
        # Both halves carry the same propagation and embedded weights (verified in
        # the .def entries), but they are stored separately: the update applies b_exp
        # to f_exp and b_implicit to f_imp independently.
        return AdditiveTableau(c=c, a_explicit=a_explicit, a_implicit=a_implicit,
                               b_explicit=b, b_implicit=b.copy(),
                               d_explicit=d, d_implicit=d.copy())
    elif scheme == 'ARK436L2SA':
        # Kennedy-Carpenter ARK(3,4)6L[2]SA: 6 stages, implicit half order 4,
        # explicit half order 3, L[2]-stable. SUNDIALS ARKODE v7.9.0
        # (ARKODE_ARK436L2SA_{ERK,DIRK}_6_3_4), in exact rational form. The
        # implicit half is a *different* 6-stage ESDIRK design from the registered
        # standalone ESDIRK4(3)6L[2]SA (different nodes: 83/250, 31/50, 17/20 vs
        # (2-sqrt(2))/4, 5/8, 26/25), so its coefficients are carried here in full.
        c = np.array([0.0, 1.0 / 2.0, 83.0 / 250.0, 31.0 / 50.0, 17.0 / 20.0, 1.0])
        a_explicit = np.array([
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.5, 0.0, 0.0, 0.0, 0.0, 0.0],
            [13861.0 / 62500.0, 6889.0 / 62500.0, 0.0, 0.0, 0.0, 0.0],
            [-116923316275.0 / 2393684061468.0,
             -2731218467317.0 / 15368042101831.0,
             9408046702089.0 / 11113171139209.0, 0.0, 0.0, 0.0],
            [-451086348788.0 / 2902428689909.0,
             -2682348792572.0 / 7519795681897.0,
             12662868775082.0 / 11960479115383.0,
             3355817975965.0 / 11060851509271.0, 0.0, 0.0],
            [647845179188.0 / 3216320057751.0,
             73281519250.0 / 8382639484533.0,
             552539513391.0 / 3454668386233.0,
             3354512671639.0 / 8306763924573.0,
             4040.0 / 17871.0, 0.0],
        ])
        a_implicit = np.array([
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [1.0 / 4.0, 1.0 / 4.0, 0.0, 0.0, 0.0, 0.0],
            [8611.0 / 62500.0, -1743.0 / 31250.0, 1.0 / 4.0, 0.0, 0.0, 0.0],
            [5012029.0 / 34652500.0, -654441.0 / 2922500.0,
             174375.0 / 388108.0, 1.0 / 4.0, 0.0, 0.0],
            [15267082809.0 / 155376265600.0, -71443401.0 / 120774400.0,
             730878875.0 / 902184768.0, 2285395.0 / 8070912.0, 1.0 / 4.0, 0.0],
            [82889.0 / 524892.0, 0.0, 15625.0 / 83664.0, 69875.0 / 102672.0,
             -2260.0 / 8211.0, 1.0 / 4.0],
        ])
        b = np.array([82889.0 / 524892.0, 0.0, 15625.0 / 83664.0, 69875.0 / 102672.0,
                      -2260.0 / 8211.0, 1.0 / 4.0])
        d = np.array([4586570599.0 / 29645900160.0, 0.0, 178811875.0 / 945068544.0,
                      814220225.0 / 1159782912.0, -3700637.0 / 11593932.0,
                      61727.0 / 225920.0])
        return AdditiveTableau(c=c, a_explicit=a_explicit, a_implicit=a_implicit,
                               b_explicit=b, b_implicit=b.copy(),
                               d_explicit=d, d_implicit=d.copy())
    else:
        raise ValueError(f"Unknown ARK scheme {scheme}")
