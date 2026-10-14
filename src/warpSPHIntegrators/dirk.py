"""Diagonally implicit Runge-Kutta driver (NOTES.md S3.1, S3.6, Phase 2).

Reuses the explicit-RK machinery `butcher.py` already has for the parts that do not
change: accumulating a stage's explicit contribution `sum_{j<i} a_ij k_j`
(`updateStateEuler`, the same helper the explicit path uses), the final b-weighted
update and finalize (`_weighted_update`, `finalizeSystem`), and embedded-pair error
estimation (`_error_estimate`). The only new piece is closing each stage's diagonal
term through a `NonlinearSolver` (`JFNKSolver` by default, with fixed-count Picard
available as an explicit low-overhead override): a DIRK stage equation
`Y_i = y^n + dt*sum_{j<=i} a_ij k_j(Y_j)`, with
`k_i = f(t_i, Y_i)`, is exactly the fixed-point form `NonlinearSolver.solve` expects
once the explicit part is folded into a `base_state` and the diagonal term is
expressed as a function of the still-unknown `Y_i`.

The tableaus below are all verified against their order conditions (symbolic
Taylor-model expansion of one step against the exact Taylor solution) and, for
the L-stable ones, their stability functions, not just smoke-tested: an embedded
pair's low-order weights are easy to transcribe wrong in a way an order
measurement would not obviously catch. The ESDIRK324/ESDIRK436 tableaus and
TR-BDF2's embedded pair are the Kennedy-Carpenter coefficients as published in
SUNDIALS ARKODE v7.9.0 (`src/arkode/arkode_butcher_dirk.def`); TR-BDF2's `a`/`b`
match that entry exactly, and its embedded `d` is the entry's published
third-order pair, so the estimate is O(dt^3) -- the propagated branch's own true
local error.
"""

from typing import Optional

import numpy as np
from torch.profiler import record_function

from .butcher import _error_estimate, _weighted_update, butcherTableau
from .fields import get_reference_state, integrated_field_names, state_difference, state_norm
from .history import HistoryEntry
from .jfnk import JFNKSolver
from .solvers import record_solve_diagnostics, FixedPointSolver, NonlinearSolver
from .specs import IntegrationResult, StageResult
from .util import (finalizeSystem, initializeSystem, reject_prior_step,
                  unpack_prior_step, updateStateEuler, updateStep)


def getDIRKTableau(scheme: str) -> butcherTableau:
    if scheme == 'backwardEuler':
        return butcherTableau(a=np.array([[1.0]]), b=np.array([1.0]), c=np.array([1.0]))
    elif scheme == 'implicitMidpoint':
        return butcherTableau(a=np.array([[0.5]]), b=np.array([1.0]), c=np.array([0.5]))
    elif scheme == 'trapezoidal':
        # Lobatto IIIA-2 / Crank-Nicolson: explicit first stage (a11 = 0, c1 = 0), one
        # implicit stage. Exercises the a_ii == 0 branch below on a real tableau.
        return butcherTableau(
            a=np.array([[0.0, 0.0], [0.5, 0.5]]),
            b=np.array([0.5, 0.5]),
            c=np.array([0.0, 1.0]),
        )
    elif scheme == 'SDIRK2':
        # Ellsiepen's L-stable 2-stage SDIRK2, gamma = 1 - sqrt(2)/2. Verified against
        # the order-2 condition sum(b*c) = 1/2 by hand: with this gamma,
        # 2*gamma - gamma**2 == 1/2 exactly (gamma solves gamma**2 - 2*gamma + 1/2 = 0).
        gamma = 1.0 - np.sqrt(2.0) / 2.0
        return butcherTableau(
            a=np.array([[gamma, 0.0], [1.0 - gamma, gamma]]),
            b=np.array([1.0 - gamma, gamma]),
            c=np.array([gamma, 1.0]),
        )
    elif scheme == 'TRBDF2':
        # TR-BDF2: a trapezoidal substep at gamma = 2-sqrt(2), followed by a
        # variable-step BDF2 endpoint solve. Written as an SDIRK tableau, both
        # implicit diagonals equal (1-gamma)/(2-gamma), making it L-stable and
        # stiffly accurate. sum(b*c) = 1/2 verifies the order-2 condition.
        # The embedded pair is SUNDIALS ARKODE's ARKODE_TRBDF2_3_3_2 published
        # (2, 3) pair (the entry's `d` vector): the estimate is O(dt^3), i.e.
        # the propagated branch's own true local error, and it satisfies
        # sum(d) = 1, d.c = 1/2, d.c^2 = 1/3 exactly.
        gamma = 2.0 - np.sqrt(2.0)
        diagonal = (1.0 - gamma) / (2.0 - gamma)
        first_weight = 1.0 / (2.0 * (2.0 - gamma))
        b_embed = np.array([
            (1.0 - np.sqrt(2.0) / 4.0) / 3.0,
            (1.0 + 3.0 * np.sqrt(2.0) / 4.0) / 3.0,
            gamma / 6.0,
        ])
        return butcherTableau(
            a=np.array([
                [0.0, 0.0, 0.0],
                [gamma / 2.0, gamma / 2.0, 0.0],
                [first_weight, first_weight, diagonal],
            ]),
            b=(np.array([first_weight, first_weight, diagonal]), b_embed),
            c=np.array([0.0, gamma, 1.0]),
        )
    elif scheme == 'ESDIRK324L2SA':
        # Kennedy-Carpenter 4-stage ESDIRK3(2)4L[2]SA (order 3 / embedded 2,
        # A- and L-stable, stiffly accurate), as published in SUNDIALS ARKODE
        # v7.9.0, `src/arkode/arkode_butcher_dirk.def` (ARKODE_ESDIRK324L2SA).
        # Explicit first stage (a11 = 0, c1 = 0). Order, embedded order, and
        # A/L stability verified by symbolic Taylor expansion, numeric
        # local-error rates, and exact stability-function sweeps (NOTES.md S3.6).
        g = 0.4358665215084589994160194511935568425293
        return butcherTableau(
            a=np.array([
                [0.0, 0.0, 0.0, 0.0],
                [g, g, 0.0, 0.0],
                [0.2576482460664272457999960162840797092643,
                 -0.09351476757488624521601546747763655179361,
                 g, 0.0],
                [0.1876410243467238251612921441668043913795,
                 -0.5952974735769549480478230275858851737782,
                 0.9717899277217721234705114322255239398694,
                 g],
            ]),
            b=(np.array([0.1876410243467238251612921441668043913795,
                         -0.5952974735769549480478230275858851737782,
                         0.9717899277217721234705114322255239398694,
                         g]),
               np.array([0.1088966176158644541561307380704960821824,
                         -0.9153258118707127534816380978168183454991,
                         1.271273597302152167844715894135642876535,
                         0.5351555969526961314807914656106793867813])),
            c=np.array([0.0,
                        0.8717330430169179988320389023871136850586,
                        0.6,
                        1.0]),
        )
    elif scheme == 'ESDIRK436L2SA':
        # Kennedy-Carpenter 6-stage ESDIRK4(3)6L[2]SA (order 4 / embedded 3,
        # A- and L-stable, stiffly accurate), as published in SUNDIALS ARKODE
        # v7.9.0, `src/arkode/arkode_butcher_dirk.def` (ARKODE_ESDIRK436L2SA),
        # in its exact radical/rational form. Explicit first stage. The
        # off-diagonal A[i][0] entries are derived exactly as in that file:
        # A[i][0] = c[i] - sum_{j>=1} A[i][j]. Order, embedded order, and
        # A/L stability verified as for ESDIRK324L2SA (NOTES.md S3.6).
        sqrt2 = np.sqrt(2.0)
        b0 = (1181.0 - 987.0 * sqrt2) / 13782.0
        b2 = 47.0 * (-267.0 + 1783.0 * sqrt2) / 273343.0
        b3 = -16.0 * (-22922.0 + 3525.0 * sqrt2) / 571953.0
        b4 = -15625.0 * (97.0 + 376.0 * sqrt2) / 90749876.0
        b5 = 1.0 / 4.0
        b_main = np.array([b0, b0, b2, b3, b4, b5])
        b_embed = np.array([
            -480923228411.0 / 4982971448372.0,
            -480923228411.0 / 4982971448372.0,
            6709447293961.0 / 12833189095359.0,
            3513175791894.0 / 6748737351361.0,
            -498863281070.0 / 6042575550617.0,
            2077005547802.0 / 8945017530137.0,
        ])
        A11 = 1.0 / 4.0
        A21 = (1.0 - sqrt2) / 8.0
        A22 = 1.0 / 4.0
        A31 = (5.0 - 7.0 * sqrt2) / 64.0
        A32 = 7.0 * (1.0 + sqrt2) / 32.0
        A33 = 1.0 / 4.0
        A41 = -(13796.0 + 54539.0 * sqrt2) / 125000.0
        A42 = (506605.0 + 132109.0 * sqrt2) / 437500.0
        A43 = 166.0 * (-97.0 + 376.0 * sqrt2) / 109375.0
        A44 = 1.0 / 4.0
        c = np.array([0.0, 0.5, (2.0 - sqrt2) / 4.0, 5.0 / 8.0, 26.0 / 25.0, 1.0])
        return butcherTableau(
            a=np.array([
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [c[1] - A11, A11, 0.0, 0.0, 0.0, 0.0],
                [c[2] - A21 - A22, A21, A22, 0.0, 0.0, 0.0],
                [c[3] - A31 - A32 - A33, A31, A32, A33, 0.0, 0.0],
                [c[4] - A41 - A42 - A43 - A44, A41, A42, A43, A44, 0.0],
                [b_main[0], b_main[1], b_main[2], b_main[3], b_main[4], b_main[5]],
            ]),
            b=(b_main, b_embed),
            c=c,
        )
    elif scheme == 'Kvaerno3':
        # Kvaerno's 4-stage L-stable ESDIRK3(2) (order 3 / embedded 2, A- and
        # L-stable, stiffly accurate), Kværnø, BIT 44(3):489-502 (2004);
        # coefficients from the ODE v7.8.1 ESDIRKIMEXTableau (NOTES.md S3.28).
        # ODE stores c[1] = gamma (the SDIRK diagonal) while row 2 sums to
        # 2*gamma; the row sum is the consistent node, so c[1] = 2*gamma here
        # (ODE's non-autonomous execution evaluates stage 2 at the stored, wrong
        # time). Embedded pair = b - btilde per butcher.py's ODE-conversion rule.
        g = 0.4358665215
        return butcherTableau(
            a=np.array([
                [0.0, 0.0, 0.0, 0.0],
                [g, g, 0.0, 0.0],
                [0.490563388419108, 0.073570090080892, g, 0.0],
                [0.308809969973036, 1.490563388254106, -1.235239879727145, g],
            ]),
            b=(np.array([0.308809969973036, 1.490563388254106, -1.235239879727145, g]),
               np.array([0.127056551526964, 2.90755668642732, -2.90634628095429, 0.871733043])),
            c=np.array([0.0, 0.871733043, 1.0, 1.0]),
        )
    elif scheme == 'Kvaerno4':
        # Kvaerno's 5-stage ESDIRK4(3) (order 4 / embedded 3, A-stable with
        # R(inf) ~ 0.718 -- NOT L-stable, unlike its 3- and 5-stage siblings),
        # Kværnø, BIT 44(3):489-502 (2004); coefficients from the ODE v7.8.1
        # ESDIRKIMEXTableau. Same c[1] = 2*gamma storage-bug correction as
        # Kvaerno3 (NOTES.md S3.28).
        g = 0.4358665215
        return butcherTableau(
            a=np.array([
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [g, g, 0.0, 0.0, 0.0],
                [0.140737774731968, -0.108365551378832, g, 0.0, 0.0],
                [0.102399400616089, -0.376878452267324, 0.838612530151233, g, 0.0],
                [0.157024897860995, 0.117330441357768, 0.61667803039168, -0.326899891110444, g],
            ]),
            b=(np.array([0.157024897860995, 0.117330441357768, 0.61667803039168, -0.326899891110444, g]),
               np.array([0.211650395105901, 0.6115393349828601, 0.39474353063212697, -1.089666303720888, 0.871733043])),
            c=np.array([0.0, 0.871733043, 0.468238744853136, 1.0, 1.0]),
        )
    elif scheme == 'Kvaerno5':
        # Kvaerno's 7-stage L-stable ESDIRK5(4) (order 5 / embedded 4, A- and
        # L-stable, stiffly accurate), Kværnø, BIT 44(3):489-502 (2004);
        # coefficients from the ODE v7.8.1 ESDIRKIMEXTableau. Same c[1] = 2*gamma
        # storage-bug correction (NOTES.md S3.28). c[2] = 1.2303... > 1.
        return butcherTableau(
            a=np.array([
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.26, 0.26, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.13, 0.8403332099679081, 0.26, 0.0, 0.0, 0.0, 0.0],
                [0.22371961478320504, 0.476755323197997, -0.06470895363112615, 0.26, 0.0, 0.0, 0.0],
                [0.16648564323248322, 0.1045001884159172, 0.03631482272098715, -0.13090704451073998, 0.26, 0.0, 0.0],
                [0.13855640231268224, 0.0, -0.04245337201752043, 0.02446657898003141, 0.6194303907248068, 0.26, 0.0],
                [0.13659751177640292, 0.0, -0.05496908796538376, -0.04118626728321046, 0.629933048990164, 0.06962479448202728, 0.26],
            ]),
            b=(np.array([0.13659751177640292, 0.0, -0.05496908796538376, -0.04118626728321046,
                         0.629933048990164, 0.06962479448202728, 0.26]),
               np.array([0.1346386212401236, 0.0, -0.06748480391324709, -0.10683911354645234,
                         0.6404357072555212, -0.12075041103594543, 0.52])),
            c=np.array([0.0, 0.52, 1.230333209967908, 0.895765984350076, 0.436393609858648, 1.0, 1.0]),
        )
    elif scheme == 'KenCarp3':
        # Kennedy's 4-stage ESDIRK3(2)4L[2]SA for convection-diffusion-reaction
        # (order 3 / embedded 2, A- and L-stable, stiffly accurate), Kennedy, NASA
        # TM-2001-210878 (2001). Its (a, b, c) core is bit-identical to the
        # SUNDIALS ARKODE ESDIRK324L2SA registered above; only the embedded pair
        # differs (Kennedy's btilde vs ARKODE's published `d`). Coefficients from
        # the ODE v7.8.1 ESDIRKIMEXTableau (NOTES.md S3.28).
        g = 0.435866521508459
        return butcherTableau(
            a=np.array([
                [0.0, 0.0, 0.0, 0.0],
                [g, g, 0.0, 0.0],
                [0.2576482460664272, -0.09351476757488625, g, 0.0],
                [0.18764102434672383, -0.595297473576955, 0.9717899277217721, g],
            ]),
            b=(np.array([0.18764102434672383, -0.595297473576955, 0.9717899277217721, g]),
               np.array([0.16054176247005852, -0.7054326832689708, 1.0748548529231567, 0.4700360678757556])),
            c=np.array([0.0, 0.871733043016918, 0.6, 1.0]),
        )
    elif scheme == 'KenCarp4':
        # Kennedy's 6-stage ESDIRK4(3)6L[2]SA (order 4 / embedded 3, A- and
        # L-stable, stiffly accurate), Kennedy, NASA TM-2001-210878 (2001);
        # coefficients from the ODE v7.8.1 ESDIRKIMEXTableau (NOTES.md S3.28).
        return butcherTableau(
            a=np.array([
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.25, 0.25, 0.0, 0.0, 0.0, 0.0],
                [0.137776, -0.055776, 0.25, 0.0, 0.0, 0.0],
                [0.14463686602698217, -0.22393190761334475, 0.4492950415863626, 0.25, 0.0, 0.0],
                [0.09825878328356477, -0.5915442428196704, 0.8101210538282996, 0.283164405707806, 0.25, 0.0],
                [0.15791629516167136, 0.0, 0.18675894052400077, 0.6805652953093346, -0.27524053099500667, 0.25],
            ]),
            b=(np.array([0.15791629516167136, 0.0, 0.18675894052400077, 0.6805652953093346,
                         -0.27524053099500667, 0.25]),
               np.array([0.16112078956013054, 0.0, 0.18431268938732132, 0.6590852193897474,
                         -0.23129366292643425, 0.22677496458923513])),
            c=np.array([0.0, 0.5, 0.332, 0.62, 0.85, 1.0]),
        )
    elif scheme == 'KenCarp47':
        # Kennedy-Carpenter's 7-stage ESDIRK4(3)7L[2]SA (order 4 / embedded 3,
        # A- and L-stable, stiffly accurate), Kennedy & Carpenter, ANM
        # 136:183-205 (2019); coefficients from the ODE v7.8.1
        # ESDIRKIMEXTableau (NOTES.md S3.28).
        return butcherTableau(
            a=np.array([
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.1235, 0.1235, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.14907768747653863, 0.14907768747653863, 0.1235, 0.0, 0.0, 0.0, 0.0],
                [0.12483442871739439, 0.12483442871739439, -0.03816885743478878, 0.1235, 0.0, 0.0, 0.0],
                [-0.07303194030218091, -0.07303194030218091, -0.2434356871601467, 0.3409995677645085, 0.1235, 0.0, 0.0],
                [-0.15296500088128806, -0.15296500088128806, 0.07220562047433587, 0.40430630248551713, 0.4059180788027232, 0.1235, 0.0],
                [0.0, 0.0, 0.5161107283174237, -0.1460635639385708, 0.23473048589019332, 0.27172234973095377, 0.1235],
            ]),
            b=(np.array([0.0, 0.0, 0.5161107283174237, -0.1460635639385708, 0.23473048589019332,
                         0.27172234973095377, 0.1235]),
               np.array([0.0, 0.0, 0.5146997104754991, -0.14038892081600224, 0.23274089307804527,
                         0.26799831726245776, 0.12495])),
            c=np.array([0.0, 0.247, 0.42165537495307726, 0.335, 0.075, 0.7, 1.0]),
        )
    elif scheme == 'KenCarp5':
        # Kennedy's 8-stage ESDIRK5(4)8L[2]SA (order 5 / embedded 4, A- and
        # L-stable, stiffly accurate), Kennedy, NASA TM-2001-210878 (2001);
        # coefficients from the ODE v7.8.1 ESDIRKIMEXTableau (NOTES.md S3.28).
        return butcherTableau(
            a=np.array([
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.205, 0.205, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.1025, -0.047570415551619845, 0.205, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.07389944079200692, 0.0, -0.08074895409950329, 0.205, 0.0, 0.0, 0.0, 0.0],
                [0.299218118308015, 0.0, 2.4638206661140414, -2.0480387844220567, 0.205, 0.0, 0.0, 0.0],
                [0.14689238442881303, 0.0, 0.11740332879881549, -0.221701968002454, -0.007593745225174481, 0.205, 0.0, 0.0],
                [0.17845729560319554, 0.0, 1.0197467452199207, -0.22154535039396367, -0.03612491620526532, -0.5455337742238872, 0.205, 0.0],
                [-0.09554858675139874, 0.0, 0.0, 2.3386928037652464, -0.14043175608247527, -2.070587707956559, 0.7628752470251866, 0.205],
            ]),
            b=(np.array([-0.09554858675139874, 0.0, 0.0, 2.3386928037652464, -0.14043175608247527,
                         -2.070587707956559, 0.7628752470251866, 0.205]),
               np.array([-0.09152020869778875, 0.0, 0.0, 2.270222727530718, -0.12071532907943695,
                         -1.9969388194685913, 0.746184871625375, 0.19276675808972416])),
            c=np.array([0.0, 0.41, 0.25992958444838016, 0.19815048669250362, 0.92, 0.24, 0.6, 1.0]),
        )
    elif scheme == 'KenCarp58':
        # Kennedy-Carpenter's 8-stage ESDIRK5(4)8L[2]SA (order 5 / embedded 4,
        # A- and L-stable, stiffly accurate), Kennedy & Carpenter, ANM
        # 136:183-205 (2019); coefficients from the ODE v7.8.1
        # ESDIRKIMEXTableau (NOTES.md S3.28).
        return butcherTableau(
            a=np.array([
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.2222222222222222, 0.2222222222222222, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.26824595137478835, 0.26824595137478835, 0.2222222222222222, 0.0, 0.0, 0.0, 0.0, 0.0],
                [-0.057945592237231995, -0.057945592237231995, 0.008938396816283733, 0.2222222222222222, 0.0, 0.0, 0.0, 0.0],
                [-0.043305287723547685, -0.043305287723547685, -0.034013891077568637, 0.25515937270676026, 0.2222222222222222, 0.0, 0.0, 0.0],
                [0.13179599023759678, 0.13179599023759678, -0.03237672627786233, 0.12385474427672251, 0.14270777930372408, 0.2222222222222222, 0.0, 0.0],
                [0.3093228210043426, 0.3093228210043426, -0.6829199272336792, -0.05882275614969546, -0.04130861383349944, 0.8971834329859666, 0.2222222222222222, 0.0],
                [0.0, 0.0, 0.1736625357358126, 0.2547916626081235, 0.2419017684509479, 0.30740485830222825, -0.19998304731933453, 0.2222222222222222],
            ]),
            b=(np.array([0.0, 0.0, 0.1736625357358126, 0.2547916626081235, 0.2419017684509479,
                         0.30740485830222825, -0.19998304731933453, 0.2222222222222222]),
               np.array([0.0, 0.0, 0.2846008545189181, 0.2543501680694674, 0.24477598774188264,
                         0.21573019452909847, -0.2568088433801624, 0.25735163852079573])),
            c=np.array([0.0, 0.4444444444444444, 0.7587141249717989, 0.11526943456404197, 0.3567571284043185, 0.72, 0.955, 1.0]),
        )
    elif scheme == 'ESDIRK325L2SA':
        # Kennedy-Carpenter's optimized 5-stage ESDIRK3(2)5L[2]SA (order 3 /
        # embedded 2, A- and L-stable, stiffly accurate), Kennedy & Carpenter,
        # ANM 146:221-244 (2019); coefficients from the ODE v7.8.1
        # ESDIRKIMEXTableau (NOTES.md S3.28).
        return butcherTableau(
            a=np.array([
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.225, 0.225, 0.0, 0.0, 0.0],
                [0.2715990257669732, 0.2715990257669732, 0.225, 0.0, 0.0],
                [0.22374368670764583, 0.22374368670764583, -0.07248737341529163, 0.225, 0.0],
                [0.17554550212940523, 0.17554550212940523, -0.34685820002600626, 0.7707671957671958, 0.225],
            ]),
            b=(np.array([0.17554550212940523, 0.17554550212940523, -0.34685820002600626, 0.7707671957671958, 0.225]),
               np.array([0.16673978224275124, 0.16673978224275124, -0.41550821600102555, 0.8446723789910678, 0.23735627252445513])),
            c=np.array([0.0, 0.45, 0.7681980515339464, 0.6, 1.0]),
        )
    elif scheme == 'ESDIRK436L2SA2':
        # Kennedy-Carpenter's optimized 6-stage ESDIRK4(3)6L[2]SA (order 4 /
        # embedded 3, A- and L-stable, stiffly accurate), Kennedy & Carpenter,
        # ANM 146:221-244 (2019); coefficients from the ODE v7.8.1
        # ESDIRKIMEXTableau (NOTES.md S3.28). c[4] = 16/15 > 1.
        return butcherTableau(
            a=np.array([
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.248, 0.248, 0.0, 0.0, 0.0, 0.0],
                [-0.05136248173426378, -0.05136248173426378, 0.248, 0.0, 0.0, 0.0],
                [-0.0852828542665107, -0.0852828542665107, 0.5339373380758116, 0.248, 0.0, 0.0],
                [-0.6896242166092131, -0.6896242166092131, 1.5044770187873386, 0.6736944913541645, 0.248, 0.0],
                [-0.0024736542509845756, -0.0024736542509845756, 0.3581348706213477, 0.4966739745362785, -0.09786153665565701, 0.248],
            ]),
            b=(np.array([-0.0024736542509845756, -0.0024736542509845756, 0.3581348706213477, 0.4966739745362785,
                         -0.09786153665565701, 0.248]),
               np.array([-0.0831759014778839, -0.0831759014778839, 0.4924309168997777, 0.5167635540019172,
                         -0.07191728852078486, 0.22907462057485772])),
            c=np.array([0.0, 0.496, 0.14527503653147242, 0.6113716295427901, 1.0469230769230768, 1.0]),
        )
    elif scheme == 'ESDIRK437L2SA':
        # Kennedy-Carpenter's optimized 7-stage ESDIRK4(3)7L[2]SA (order 4 /
        # embedded 3, A- and L-stable, stiffly accurate), Kennedy & Carpenter,
        # ANM 146:221-244 (2019); coefficients from the ODE v7.8.1
        # ESDIRKIMEXTableau (NOTES.md S3.28).
        return butcherTableau(
            a=np.array([
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.125, 0.125, 0.0, 0.0, 0.0, 0.0, 0.0],
                [-0.02588834764831844, -0.02588834764831844, 0.125, 0.0, 0.0, 0.0, 0.0],
                [0.33838834764831843, 0.33838834764831843, -0.30177669529663687, 0.125, 0.0, 0.0, 0.0],
                [-0.3592453618381594, -0.3592453618381594, 0.9365078600463644, 0.3536318936123176, 0.125, 0.0, 0.0],
                [0.23361061091244562, 0.23361061091244562, -0.0433153738101898, 0.01903274535895701, 0.13841061297554788, 0.125, 0.0],
                [-0.40085161500960825, -0.40085161500960825, 0.9391524145239087, 0.5185422838949312, 0.7755100321672022, -0.5565015005668256, 0.125],
            ]),
            b=(np.array([-0.40085161500960825, -0.40085161500960825, 0.9391524145239087, 0.5185422838949312,
                         0.7755100321672022, -0.5565015005668256, 0.125]),
               np.array([-0.24210689376668587, -0.24210689376668587, 0.6587096818817366, 0.5004777357240691,
                         0.7607872310157866, -0.5714751468025062, 0.1357142857142857])),
            c=np.array([0.0, 0.25, 0.07322330470336312, 0.5, 0.6966490299823633, 0.7063492063492064, 1.0]),
        )
    elif scheme == 'ESDIRK547L2SA2':
        # Kennedy-Carpenter's optimized 7-stage ESDIRK5(4)7L[2]SA (order 5 /
        # embedded 4, A- and L-stable, stiffly accurate), Kennedy & Carpenter,
        # ANM 146:221-244 (2019); coefficients from the ODE v7.8.1
        # ESDIRKIMEXTableau (NOTES.md S3.28). ODE stores c[3] = 49/453 while row
        # 4 sums to 0.1388101983002833; with the row sum b.c = 1/2 exactly, so
        # the row sum is the consistent node and the stored value is an ODE
        # typo (its non-autonomous execution evaluates stage 4 at the wrong
        # time).
        return butcherTableau(
            a=np.array([
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.184, 0.184, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.22210764773832475, 0.22210764773832475, 0.184, 0.0, 0.0, 0.0, 0.0],
                [-0.014049475381926283, -0.014049475381926283, -0.01709085093586415, 0.184, 0.0, 0.0, 0.0],
                [-0.40838859254931464, -0.40838859254931464, 0.16646399821362964, 1.1662718062895745, 0.184, 0.0, 0.0],
                [-0.5392907235588114, -0.5392907235588114, -0.242234428845425, 1.4888806111225146, 0.5563122281913184, 0.184, 0.0],
                [-0.03946606910974026, -0.03946606910974026, 0.27263649025024267, 0.4321651725202882, 0.3524160862328891, -0.16228561078393952, 0.184],
            ]),
            b=(np.array([-0.03946606910974026, -0.03946606910974026, 0.27263649025024267, 0.4321651725202882,
                         0.3524160862328891, -0.16228561078393952, 0.184]),
               np.array([-0.08068946656664744, -0.08068946656664744, 0.18289968461343553, 0.517013873766231,
                         0.4265997313024543, -0.10432685796713498, 0.13919250141830897])),
            c=np.array([0.0, 0.368, 0.6282152954766494, 0.1388101983002833, 0.6999586194045747, 0.9083769633507853, 1.0]),
        )
    elif scheme == 'ESDIRK54I8L2SA':
        # Kennedy-Carpenter's 8-stage ESDIRK5(4)I8L[2]SA (order 5 / embedded 4,
        # A- and L-stable, stiffly accurate; the intermediate-stage variant of
        # the 5th-order pair), Kennedy & Carpenter, ANM 146:221-244 (2019);
        # coefficients from the ODE v7.8.1 ESDIRKIMEXTableau (NOTES.md S3.28).
        return butcherTableau(
            a=np.array([
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.25, 0.25, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.30177669529663687, 0.30177669529663687, 0.25, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.1914039030905011, 0.1914039030905011, -0.10280780618100219, 0.25, 0.0, 0.0, 0.0, 0.0],
                [0.28240705587394466, 0.28240705587394466, -0.04126691018786242, 0.026452798439973117, 0.25, 0.0, 0.0, 0.0],
                [0.23835518790251256, 0.2552834985916606, -0.19539385777416787, 0.018765913020685827, 0.11298925825930888, 0.25, 0.0, 0.0],
                [0.12371171764236197, -0.30513942215186096, -5.475399871109475, 2.9975825630620987, 9.742587241281745, -6.333342228724871, 0.25, 0.0],
                [0.13369730938460478, -0.20386397876250517, -4.263238079373237, 2.4818742436204877, 7.606707292513777, -4.985159097468383, -0.02001768991474364, 0.25],
            ]),
            b=(np.array([0.13369730938460478, -0.20386397876250517, -4.263238079373237, 2.4818742436204877,
                         7.606707292513777, -4.985159097468383, -0.02001768991474364, 0.25]),
               np.array([0.15344432088458565, -0.0035866795399671636, -1.866126975782635, 1.462035021517263,
                         3.3828969265768274, -2.3190589319290202, -0.5539912956439024, 0.7443876139168488])),
            c=np.array([0.0, 0.5, 0.8535533905932737, 0.53, 0.8, 0.68, 1.0, 1.0]),
        )
    elif scheme == 'ESDIRK659L2SA':
        # Kennedy-Carpenter's 9-stage ESDIRK6(5)9L[2]SA (order 6 / embedded 5,
        # stiffly accurate), Kennedy & Carpenter, ANM 146:221-244 (2019);
        # coefficients from the ODE v7.8.1 ESDIRKIMEXTableau (NOTES.md S3.28).
        # NOTE: ODE's own docstring flags this tableau with "STABILITY ISSUES"
        # (SciML/OrdinaryDiffEq.jl#1933); the probe's 89-degree A-cone grid and
        # real-axis sweep found no violation on the tested ranges, but it is
        # the one of the family whose A-stability is not certified.
        return butcherTableau(
            a=np.array([
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.2222222222222222, 0.2222222222222222, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.1111111111111111, -0.051566846126417175, 0.2222222222222222, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.08243662514910446, -0.2529078790095889, 0.4580891117002715, 0.2222222222222222, 0.0, 0.0, 0.0, 0.0, 0.0],
                [-0.06982516858049691, -0.7933123004931639, 0.9883018015038463, 0.5676134453475923, 0.2222222222222222, 0.0, 0.0, 0.0, 0.0],
                [0.11511510302793021, 0.32068272496205846, -0.2808105937575744, -0.1585095783016084, -0.007884594291318341, 0.2222222222222222, 0.0, 0.0, 0.0],
                [-0.025616385303347644, 0.2652452871344536, -0.3973244335372646, -0.058235695424658944, -0.013834018407705677, 0.09725563562997894, 0.2222222222222222, 0.0, 0.0],
                [-0.3346904986115499, -0.9223373180833085, 0.8027404838515502, 0.9070977714643121, -0.016263252234954912, -0.304522257436572, 0.6157528488283008, 0.2222222222222222, 0.0],
                [0.0, 0.0, 0.0, 0.4180208363576131, 0.5401802121965142, 0.11134394759380895, 0.1931979255143056, -0.4849651438844641, 0.2222222222222222],
            ]),
            b=(np.array([0.0, 0.0, 0.0, 0.4180208363576131, 0.5401802121965142, 0.11134394759380895,
                         0.1931979255143056, -0.4849651438844641, 0.2222222222222222]),
               np.array([-0.8059655150219478, 0.0, -1.1009693680941892, 1.2934361343398642, -0.4182532992328589,
                         -0.5071073556433957, 2.069584052538532, 0.4854901286726704, -0.016214777558674842])),
            c=np.array([0.0, 0.4444444444444444, 0.28176648720691616, 0.5098400800620093, 0.915, 0.21081528386170978, 0.08971261231367791, 0.97, 1.0]),
        )
    elif scheme == 'SSPSDIRK2':
        # Ketcheson-Macdonald-Gottlieb's 2-stage SDIRK with the maximum SSP
        # region (order 2, gamma = 1/4, A-stable on the real axis with
        # R(inf) = 1 -- NOT L-stable), Ketcheson, Macdonald & Gottlieb, ANM
        # 59(2):373-392 (2009). Coefficients from the ODE v7.8.1
        # ESDIRKIMEXTableau (NOTES.md S3.28). ODE stores c = [1, 1] (an
        # execution artifact of its explicit_fsallast handling); the consistent
        # abscissae are the row sums [1/4, 3/4], and ODE's non-autonomous
        # execution evaluates both stages at the stored (wrong) times.
        return butcherTableau(
            a=np.array([
                [0.25, 0.0],
                [0.5, 0.25],
            ]),
            b=np.array([0.5, 0.5]),
            c=np.array([0.25, 0.75]),
        )
    elif scheme == 'ARS222':
        # Ascher-Ruuth-Spiteri's 3-stage ESDIRK2 (order 2, stiffly accurate;
        # the implicit half of ODE's ARS222 IMEX method -- its Ae/be explicit
        # part is unused on pure problems), Ascher, Ruuth & Spiteri, ANM
        # 25(2-3):151-167 (1997); coefficients from the ODE v7.8.1 tableau
        # (NOTES.md S3.28). The first stage is free (a[0] = 0, b[0] = 0):
        # the effective method is the 2-stage SDIRK with
        # gamma = 1 - 1/sqrt(2) = 0.29289321881345254.
        return butcherTableau(
            a=np.array([
                [0.0, 0.0, 0.0],
                [0.0, 0.29289321881345254, 0.0],
                [0.0, 0.7071067811865475, 0.29289321881345254],
            ]),
            b=np.array([0.0, 0.7071067811865475, 0.29289321881345254]),
            c=np.array([0.0, 0.29289321881345254, 1.0]),
        )
    elif scheme == 'ARS232':
        # Ascher-Ruuth-Spiteri's 3-stage ESDIRK2 (order 2, stiffly accurate),
        # Ascher, Ruuth & Spiteri, ANM 25(2-3):151-167 (1997); coefficients
        # from the ODE v7.8.1 tableau (NOTES.md S3.28). ODE's ARS232Tableau
        # carries the SAME implicit (Ai, bi, c) as ARS222 -- only the Ae/be
        # explicit part differs (delta = -2*sqrt(2)/3), and it is unused on
        # pure problems, so this is the implicit half of ARS(2,3,2).
        return butcherTableau(
            a=np.array([
                [0.0, 0.0, 0.0],
                [0.0, 0.29289321881345254, 0.0],
                [0.0, 0.7071067811865475, 0.29289321881345254],
            ]),
            b=np.array([0.0, 0.7071067811865475, 0.29289321881345254]),
            c=np.array([0.0, 0.29289321881345254, 1.0]),
        )
    elif scheme == 'ARS343':
        # Ascher-Ruuth-Spiteri's 4-stage ESDIRK3 (order 3, stiffly accurate;
        # the implicit half of ODE's ARS343 IMEX method), Ascher, Ruuth &
        # Spiteri, ANM 25(2-3):151-167 (1997); coefficients from the ODE
        # v7.8.1 tableau (NOTES.md S3.28).
        return butcherTableau(
            a=np.array([
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.435866521508459, 0.0, 0.0],
                [0.0, 0.2820667392457705, 0.435866521508459, 0.0],
                [0.0, 1.20849664917601, -0.644363170684469, 0.435866521508459],
            ]),
            b=np.array([0.0, 1.20849664917601, -0.644363170684469, 0.435866521508459]),
            c=np.array([0.0, 0.435866521508459, 0.7179332607542295, 1.0]),
        )
    elif scheme == 'ARS443':
        # Ascher-Ruuth-Spiteri's 5-stage ESDIRK3 (order 3, stiffly accurate;
        # the implicit half of ODE's ARS443 IMEX method, gamma = 1/2),
        # Ascher, Ruuth & Spiteri, ANM 25(2-3):151-167 (1997); coefficients
        # from the ODE v7.8.1 tableau (NOTES.md S3.28).
        return butcherTableau(
            a=np.array([
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.5, 0.0, 0.0, 0.0],
                [0.0, 0.16666666666666666, 0.5, 0.0, 0.0],
                [0.0, -0.5, 0.5, 0.5, 0.0],
                [0.0, 1.5, -1.5, 0.5, 0.5],
            ]),
            b=np.array([0.0, 1.5, -1.5, 0.5, 0.5]),
            c=np.array([0.0, 0.5, 0.6666666666666666, 0.5, 1.0]),
        )
    elif scheme == 'BHR553':
        # Boscarino-Russo's 5-stage ESDIRK3 (order 3, stiffly accurate; the
        # BHR(5,5,3)* tableau, implicit half of ODE's BHR553 IMEX method),
        # Boscarino & Russo, SIAM J. Sci. Comput. 31(3):1926-1945 (2009);
        # coefficients from the ODE v7.8.1 tableau (NOTES.md S3.28), as
        # stored (12 digits). Stage 4 sits at c[3] = 1.5, well past the step
        # end (as in ODE) -- see test_stage_times_stay_within_the_step.
        return butcherTableau(
            a=np.array([
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.43586652150846, 0.43586652150846, 0.0, 0.0, 0.0],
                [0.43586652150846, 0.0, 0.43586652150846, 0.0, 0.0],
                [0.523600775834581, 0.0, 0.540532702656959, 0.43586652150846, 0.0],
                [0.369394442791758, 0.0, 0.36286338557874, -0.168124349878957, 0.43586652150846],
            ]),
            b=np.array([0.369394442791758, 0.0, 0.36286338557874, -0.168124349878957, 0.43586652150846]),
            c=np.array([0.0, 0.87173304301692, 0.87173304301692, 1.5, 1.0]),
        )
    elif scheme == 'Cash4':
        # Cash's 5-stage SDIRK4 (order 4; the last row IS the main b, but
        # ODE stores c[-1] 7.8e-12 past 1, so the driver's 1e-12 stiff
        # accuracy predicate -- and this registration -- record it as NOT
        # stiffly accurate), Hindmarsh et al. (SUNDIALS), ACM Trans. Math.
        # Software 31(3):363-396 (2005); coefficients from the ODE v7.8.1
        # tableau (NOTES.md S3.28), as stored (12 digits) except the last
        # b weight, nudged by 7.8e-12 so sum(b) == 1.0 exactly (ODE's
        # stored weights sum to 1 + 7.8e-12). Implicit first stage
        # (a[0,0] = gamma); the c row is the row sums, not ODE's stored c
        # (which carries ~1.1e-11 transcription noise).
        return butcherTableau(
            a=np.array([
                [0.435866521508, 0.0, 0.0, 0.0, 0.0],
                [-1.1358665215, 0.435866521508, 0.0, 0.0, 0.0],
                [1.08543330679, -0.721299828287, 0.435866521508, 0.0, 0.0],
                [0.416349501547, 0.190984004184, -0.118643265417, 0.435866521508, 0.0],
                [0.896869652944, 0.0182725272734, -0.0845900310706, -0.266418670647, 0.435866521508],
            ]),
            b=(np.array([0.896869652944, 0.0182725272734, -0.0845900310706, -0.266418670647, 0.4358665215002]),
               np.array([1.0170473729779999, 0.006797775398400003, -0.1424360382338, -0.7531421531429999, 0.8717330430082])),
            c=np.array([0.435866521508, -0.6999999999919999, 0.8000000000109999, 0.924556761822, 1.0000000000078]),
        )
    elif scheme == 'CFNLIRK3':
        # Calvo-DeFrutos-Novo's 4-stage linearly-implicit ESDIRK3 (order 3,
        # stiffly accurate), Calvo, De Frutos & Novo, ANM 37(4):535-549
        # (2001); coefficients from the ODE v7.8.1 tableau (NOTES.md
        # S3.28), as stored (12 digits). ODE registers CFNLIRK3 as a split
        # (IMEX) method; on a pure problem the effective step is the
        # implicit-side DIRK, whose coefficients match ARS343's to the 12
        # stored digits.
        return butcherTableau(
            a=np.array([
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.43586652150846, 0.0, 0.0],
                [0.0, 0.28206673924576997, 0.43586652150846, 0.0],
                [0.0, 1.2084966491760127, -0.6443631706844728, 0.43586652150846],
            ]),
            b=np.array([0.0, 1.2084966491760127, -0.6443631706844728, 0.43586652150846]),
            c=np.array([0.0, 0.43586652150846, 0.71793326075423, 1.0]),
        )
    elif scheme == 'Hairer4':
        # Hairer-Wanner's 5-stage SDIRK4 (order 4, stiffly accurate, (4,3)
        # embedded pair), Hairer & Wanner, Solving Ordinary Differential
        # Equations II, Springer (1996); coefficients from the ODE v7.8.1
        # tableau (NOTES.md S3.28). Implicit first stage (a[0,0] = 1/4).
        return butcherTableau(
            a=np.array([
                [0.25, 0.0, 0.0, 0.0, 0.0],
                [0.5, 0.25, 0.0, 0.0, 0.0],
                [0.34, -0.04, 0.25, 0.0, 0.0],
                [0.2727941176470588, -0.05036764705882353, 0.027573529411764705, 0.25, 0.0],
                [1.0416666666666667, -1.0208333333333333, 7.8125, -7.083333333333333, 0.25],
            ]),
            b=(np.array([1.0416666666666667, -1.0208333333333333, 7.8125, -7.083333333333333, 0.25]),
               np.array([0.8541666666666667, -1.8645833333333333, 8.59375, -7.083333333333333, 0.5])),
            c=np.array([0.25, 0.75, 0.55, 0.5, 1.0]),
        )
    elif scheme == 'Hairer42':
        # Hairer-Wanner's 5-stage SDIRK4 (order 4, stiffly accurate, (4,3)
        # embedded pair; the second of the book's two 4th-order 5-stage
        # tableaus), Hairer & Wanner, Solving Ordinary Differential
        # Equations II, Springer (1996); coefficients from the ODE v7.8.1
        # tableau (NOTES.md S3.28). Implicit first stage (a[0,0] = 4/15).
        return butcherTableau(
            a=np.array([
                [0.26666666666666666, 0.0, 0.0, 0.0, 0.0],
                [0.5, 0.26666666666666666, 0.0, 0.0, 0.0],
                [0.3541539528432732, -0.054153952843273234, 0.26666666666666666, 0.0, 0.0],
                [0.08515494131138653, -0.06484332287891555, 0.07915325296404206, 0.26666666666666666, 0.0],
                [2.1001157005669326, -0.7677800284445977, 2.3998163610800263, -2.998818699869028, 0.26666666666666666],
            ]),
            b=(np.array([2.1001157005669326, -0.7677800284445977, 2.3998163610800263, -2.998818699869028, 0.26666666666666666]),
               np.array([1.3149671967466712, -1.3896807085929184, 2.4096240396949127, -1.8682438611819994, 0.5333333333333333])),
            c=np.array([0.26666666666666666, 0.7666666666666667, 0.5666666666666667, 0.36613153806317966, 1.0]),
        )
    elif scheme == 'SFSDIRK4':
        # Ferracina-Spijker's 4-stage SFSDIRK (order 4, NOT stiffly
        # accurate -- the c nodes stop at 0.885), Ferracina & Spijker, ANM
        # 58(11):1675-1686 (2008); coefficients from the ODE v7.8.1 tableau
        # (NOTES.md S3.28), as stored (12 digits). Implicit first stage;
        # the c row is the row sums, not ODE's stored c (which carries
        # ~9e-12 transcription noise).
        return butcherTableau(
            a=np.array([
                [0.097961082941, 0.0, 0.0, 0.0],
                [0.262318069183, 0.097961082941, 0.0, 0.0],
                [0.230169419019, 0.294466719347, 0.097961082941, 0.0],
                [0.210562684389, 0.26938288828, 0.307008634881, 0.097961082941],
            ]),
            b=np.array([0.222119403264, 0.282060762166, 0.236881213175, 0.258938621395]),
            c=np.array([0.097961082941, 0.36027915212399997, 0.622597221307, 0.884915290491]),
        )
    elif scheme == 'SFSDIRK5':
        # Ferracina-Spijker's 5-stage SFSDIRK (order 4, NOT stiffly
        # accurate), Ferracina & Spijker, ANM 58(11):1675-1686 (2008);
        # coefficients from the ODE v7.8.1 tableau (NOTES.md S3.28), as
        # stored (12 digits) except the last b weight, nudged by 1e-12 so
        # sum(b) == 1.0 exactly. ODE's own alg_order records 4 (its
        # docstring's
        # 'Method of order 5' names the stage count, not the order; the
        # order is pinned by test_dirk_scheme_reaches_its_claimed_order).
        # Implicit first stage.
        return butcherTableau(
            a=np.array([
                [0.078752939968, 0.0, 0.0, 0.0, 0.0],
                [0.222465723027, 0.078752939968, 0.0, 0.0, 0.0],
                [0.2031923617, 0.230847263068, 0.078752939968, 0.0, 0.0],
                [0.188022704389, 0.191735630027, 0.209922288451, 0.078752939968, 0.0],
                [0.188025114093, 0.191739898281, 0.20990760186, 0.252726086329, 0.078752939968],
            ]),
            b=np.array([0.192143833571, 0.200935182974, 0.205799262036, 0.20055384464, 0.20056787677899998]),
            c=np.array([0.078752939968, 0.301218662995, 0.512792564736, 0.668433562835, 0.921151640531]),
        )
    elif scheme == 'SFSDIRK6':
        # Ferracina-Spijker's 6-stage SFSDIRK (order 4, NOT stiffly
        # accurate), Ferracina & Spijker, ANM 58(11):1675-1686 (2008);
        # coefficients from the ODE v7.8.1 tableau (NOTES.md S3.28), as
        # stored (12 digits). Implicit first stage.
        return butcherTableau(
            a=np.array([
                [0.067410767219, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.194216850802, 0.067410767219, 0.0, 0.0, 0.0, 0.0],
                [0.194216850802, 0.199861501713, 0.067410767219, 0.0, 0.0, 0.0],
                [0.162188551749, 0.16690234333, 0.145120313717, 0.067410767219, 0.0, 0.0],
                [0.1651768185, 0.169977460026, 0.150227711763, 0.181214258555, 0.067410767219, 0.0],
                [0.1651768185, 0.169977460026, 0.150227711763, 0.181214258555, 0.199861501713, 0.067410767219],
            ]),
            b=np.array([0.16895417046, 0.173864595628, 0.156683775305, 0.157643002581, 0.173864725004, 0.168989731022]),
            c=np.array([0.067410767219, 0.261627618021, 0.461489119734, 0.541621976015, 0.734007016063, 0.933868517776]),
        )
    elif scheme == 'SFSDIRK7':
        # Ferracina-Spijker's 7-stage SFSDIRK (order 4, NOT stiffly
        # accurate), Ferracina & Spijker, ANM 58(11):1675-1686 (2008);
        # coefficients from the ODE v7.8.1 tableau (NOTES.md S3.28), as
        # stored (12 digits) except the last b weight, nudged by 1e-12 so
        # sum(b) == 1.0 exactly. Implicit first stage.
        return butcherTableau(
            a=np.array([
                [0.056879041592, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.172205581756, 0.056879041592, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.135485903539, 0.135485903539, 0.056879041592, 0.0, 0.0, 0.0, 0.0],
                [0.133962606568, 0.133962606568, 0.170269437596, 0.056879041592, 0.0, 0.0, 0.0],
                [0.133962606568, 0.133962606568, 0.170269437596, 0.172205581756, 0.056879041592, 0.0, 0.0],
                [0.138004377067, 0.133084723451, 0.152274237527, 0.15400575717, 0.15400575717, 0.056879041592, 0.0],
                [0.13943366564, 0.134719607258, 0.145910607076, 0.147569765489, 0.147569765489, 0.165009008641, 0.056879041592],
            ]),
            b=np.array([0.138370770799, 0.134572540279, 0.150642940425, 0.152355910489, 0.152355910489, 0.132951737506, 0.1387501900130001]),
            c=np.array([0.056879041592, 0.229084623348, 0.32785084867, 0.495073692324, 0.66727927408, 0.788253893977, 0.937091461185]),
        )
    elif scheme == 'SFSDIRK8':
        # Ferracina-Spijker's 8-stage SFSDIRK (order 4, NOT stiffly
        # accurate), Ferracina & Spijker, ANM 58(11):1675-1686 (2008);
        # coefficients from the ODE v7.8.1 tableau (NOTES.md S3.28), as
        # stored (12 digits) except the last b weight, nudged by 1e-12 so
        # sum(b) == 1.0 exactly. Implicit first stage.
        return butcherTableau(
            a=np.array([
                [0.050353353407, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.147724666662, 0.050353353407, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.114455029802, 0.114455029802, 0.050353353407, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.114147680771, 0.114147680771, 0.14732797782, 0.050353353407, 0.0, 0.0, 0.0, 0.0],
                [0.114163314686, 0.114163314686, 0.147259379853, 0.14765588399, 0.050353353407, 0.0, 0.0, 0.0],
                [0.114163314686, 0.114163314686, 0.147259379853, 0.14765588399, 0.147724666662, 0.050353353407, 0.0, 0.0],
                [0.118472990244, 0.118472990244, 0.128349529304, 0.128695117609, 0.12875506777, 0.12875506777, 0.050353353407, 0.0],
                [0.118472990244, 0.118472990244, 0.128349529304, 0.128695117609, 0.12875506777, 0.12875506777, 0.147724666662, 0.050353353407],
            ]),
            b=np.array([0.117592883046, 0.117592883046, 0.132211234288, 0.13256722045, 0.132628974356, 0.132293123539, 0.117556840638, 0.1175568406369999]),
            c=np.array([0.050353353407, 0.198078020069, 0.279263413011, 0.425976692769, 0.573595246622, 0.721319913284, 0.801854116348, 0.94957878301]),
        )
    else:
        raise ValueError(f"Unknown DIRK scheme {scheme}")


def _dirk_reuses_first_stage(tableau: butcherTableau) -> bool:
    """True iff this DIRK tableau consumes a reused ``k0`` losslessly.

    Both halves of the (implicit) FSAL condition must hold:

    * the first stage is explicit (``a[0,0] == 0``, ``c[0] == 0``), so it needs no
      solve and can consume a ready-made derivative; and
    * the tableau is stiffly accurate (``c[-1] == 1``, ``a[-1] == b``), so the
      previous step's last stage IS ``y^{n+1}`` and its converged derivative is
      exactly ``f(t^{n+1}, y^{n+1})`` -- the value this step's first stage needs.

    Only then is the reuse exact (no order cost). This is derived from the tableau
    itself so the call-time decision cannot drift from the registered
    ``stiffly_accurate`` metadata; ``reuse.dirk_reuse_analysis`` reports the same
    property for callers that only have the scheme.
    """
    a = np.asarray(tableau.a, dtype=float)
    c = np.asarray(tableau.c, dtype=float)
    b = tableau.b[0] if isinstance(tableau.b, tuple) else tableau.b
    b = np.asarray(b, dtype=float)
    explicit_first = abs(a[0, 0]) < 1e-12 and abs(c[0]) < 1e-12
    stiffly_accurate = abs(c[-1] - 1.0) < 1e-12 and np.allclose(a[-1], b, atol=1e-12)
    return bool(explicit_first and stiffly_accurate)


def _default_norm(rtol: float, atol: float):
    def norm(y_new, y_old):
        return state_norm(state_difference(y_new, y_old), rtol, atol, reference=y_old)
    return norm


def DIRK(initialState, dt, f, tableau: butcherTableau, *args,
        name: str = 'DIRK', solver: Optional[NonlinearSolver] = None, **kwargs):
    """One DIRK step. `tableau.a`'s diagonal may be nonzero; those stages solve implicitly.

    First-stage reuse (`priorStep`) is implemented for the tableaus that satisfy both
    halves of the (implicit) FSAL condition -- an explicit first stage
    (`a[0,0] == 0`, `c[0] == 0`) *and* stiff accuracy (`c[-1] == 1`, `a[-1] == b`):
    Trapezoidal, TR-BDF2, ESDIRK3(2)4L[2]SA and ESDIRK4(3)6L[2]SA. For those, the
    previous step's last stage IS `y^{n+1}`, so `stages[-1].update` is exactly the
    `f(t^{n+1}, y^{n+1})` this step's explicit first stage needs; the reuse costs no
    order and saves one (cheap, explicit) RHS evaluation per step. Every other DIRK
    tableau -- SDIRK2's implicit first stage, the single-stage backward Euler and
    implicit midpoint -- cannot consume a reused `k0`, so it rejects `priorStep` with
    the same warning every other non-reuse scheme in this library gives.

    `history=`, `rtol=`/`atol=` (the default Picard-convergence norm's tolerances, only
    used when `solver_opts={'tol': ...}` requests early exit -- the default fixed
    2-iteration schedule ignores them), and `solver_opts={}` (forwarded to
    `NonlinearSolver.solve` as `**opts`) are all optional.

    `norm` (below) is always this module's own Hairer-Wanner weighted-RMS
    (`fields.state_norm`, `_default_norm`) -- its own documented convention is
    "< 1.0 means converged". Note that this driver does *not* inject a
    `newton_tol` into `solver_opts`: `JFNKSolver`'s `newton_tol` falls back to
    its own `tol` (1e-8, the GMRES linear-solve tolerance), which float64
    solves reach and float32 ones exit through the stagnation floor instead
    (see `jfnk.py`'s `JFNKSolver.solve` docstring on `newton_tol`); a caller
    that wants the norm's own "< 1.0" scale as the Newton threshold passes
    `solver_opts={'newton_tol': 1.0}` explicitly.
    """
    verbose = bool(kwargs.get('verbose', False))
    solver = solver or JFNKSolver()
    history = kwargs.pop('history', None)
    priorStep = kwargs.pop('priorStep', None)
    # Reuse is valid for exactly the stiffly-accurate tableaus with an explicit first
    # stage (`_dirk_reuses_first_stage`). For every other DIRK tableau the reused k0
    # would be stale (last stage is not y^{n+1}) or unusable (the first stage is
    # implicit), so reject it the same way every other non-reuse scheme does.
    if priorStep is not None and not _dirk_reuses_first_stage(tableau):
        reject_prior_step(name, priorStep)
    solver_opts = kwargs.get('solver_opts', {})
    norm = _default_norm(kwargs.get('rtol', 1e-3), kwargs.get('atol', 1e-6))

    with record_function("[Integration] DIRK"):
        initializeSystem(initialState, dt, *args, **kwargs)
        ks, rs = [], []
        solver_diagnostics = []
        stage_state = None

        for i, c_i in enumerate(tableau.c):
            with record_function(f"[Integration] DIRK: stage {i}"):
                a_row = tableau.a[i, :i]
                a_ii = tableau.a[i, i]
                t_i = float(initialState.t + c_i * dt)

                base_state = initialState.initializeNewState(*args, **kwargs)
                for j, a_ij in enumerate(a_row):
                    if a_ij != 0:
                        base_state = updateStateEuler(base_state, ks[j], a_ij * dt, copyState=False, **kwargs)
                base_state.t = t_i

                if a_ii == 0:
                    # An explicit stage inside an otherwise-implicit tableau (e.g.
                    # trapezoidal's first stage): no solve needed, k_i = f(t_i, base).
                    if i == 0 and priorStep is not None:
                        # Stiffly accurate + explicit first stage: the previous step's
                        # last stage IS f(t^{n+1}, y^{n+1}), exactly this stage's value,
                        # so consume it instead of re-evaluating. Saves one (cheap,
                        # explicit) RHS evaluation per step at no order cost.
                        k_i, r_i = unpack_prior_step(priorStep, verbose)
                    else:
                        k_i, r_i = updateStep(initialState, base_state, dt, f, *args, **kwargs)
                    stage_state = base_state
                    solver_diagnostics.append(None)
                else:
                    box = {}

                    def step_fn(Y, base_state=base_state, a_ii=a_ii, t_i=t_i, box=box):
                        Y.t = t_i
                        k, r = updateStep(initialState, Y, dt, f, *args, **kwargs)
                        # Also remember `Y` itself, not just `(k, r)`: `updateStep`'s
                        # `preprocess` call ran *on this object*, in place, so it is the
                        # only thing downstream that is carrying this stage's `copied`
                        # fields (density, pressure, ...). The `y_new` this function
                        # returns is a fresh clone built by `updateStateEuler`
                        # (`copyState=True`) that never saw `preprocess` at all.
                        box['k'], box['r'], box['Y'] = k, r, Y
                        return updateStateEuler(base_state, k, a_ii * dt, copyState=True, **kwargs)

                    y0 = base_state.initializeNewState(*args, **kwargs)
                    y0.t = t_i
                    if verbose:
                        print(f"[Integrator] DIRK stage {i}: solving Y = base + {a_ii:.4f}*dt*f(Y) at t={t_i:.4f}")
                    solve_result = solver.solve(step_fn, y0, norm, **solver_opts)
                    solver_diagnostics.append(record_solve_diagnostics(solve_result))
                    # `box['Y']`/`box['k']` come from the *last* `step_fn` call, i.e. the
                    # last RHS evaluation of this stage -- the same "buffer the last
                    # evaluation ran on" convention `butcher.RungeKuttaB` uses for its
                    # own `lastStageState`. `stage_state = base + a_ii*dt*k_i` holds by
                    # construction (that is what `step_fn` just returned), so `(k_i,
                    # box['Y'])` satisfy the stage equation the way a converged Picard
                    # iterate is supposed to, not an approximation one iteration behind.
                    if isinstance(solver, JFNKSolver) or 'k' not in box:
                        # Finite-difference Krylov probes use isolated cloned states,
                        # so re-evaluate the converged stage on this driver's buffer.
                        # That gives copied fields their standard last-stage lifecycle.
                        stage_state = base_state
                        solved_ref = get_reference_state(solve_result.y)
                        stage_ref = get_reference_state(stage_state)
                        for field_name in integrated_field_names(solve_result.y):
                            setattr(stage_ref, field_name, getattr(solved_ref, field_name))
                        stage_state.t = t_i
                        k_i, r_i = updateStep(initialState, stage_state, dt, f, *args, **kwargs)
                    else:
                        k_i, r_i = box['k'], box['r']
                        stage_state = box['Y']

                ks.append(k_i)
                rs.append(r_i)

        lastStageState = stage_state
        stages = [StageResult(aux=r, update=k) for r, k in zip(rs, ks)]

        def _next_history():
            if history is None:
                return None
            entry = HistoryEntry(t=float(initialState.t), dt=dt, update=ks[-1], aux=rs[-1])
            return history.pushed(entry)

        with record_function("[Integration] DIRK: Update"):
            if not isinstance(tableau.b, tuple):
                new_state = _weighted_update(initialState, ks, tableau.b, dt, *args, **kwargs)
                new_state.t = float(initialState.t + dt)
                finalizeSystem(new_state, initialState, dt, rs, ks, tableau.b,
                               *args, lastStageSystem=lastStageState, **kwargs)
                return IntegrationResult(
                    state=new_state, stages=stages, history=_next_history(),
                    solver_diagnostics=solver_diagnostics)

            b_main, b_embedded = tableau.b
            new_state = _weighted_update(initialState, ks, b_main, dt, *args, **kwargs)
            new_state.t = float(initialState.t + dt)
            error = _error_estimate(initialState, ks, b_main, b_embedded, dt, *args, **kwargs)
            finalizeSystem(new_state, initialState, dt, rs, ks, b_main,
                           *args, lastStageSystem=lastStageState, **kwargs)
            return IntegrationResult(
                state=new_state, stages=stages, error=error, history=_next_history(),
                solver_diagnostics=solver_diagnostics)


def dirkScheme(tableau_name: str, **tableau_kwargs):
    """Build a DIRK scheme callable that carries its tableau, mirroring `butcherScheme`.

    Exposes the tableau as `.dirkTableau`, not `.butcherTableau`. The distinction is
    load-bearing: `reuse._tableau_of` looks for `.butcherTableau` to run the
    *explicit*-scheme substitution analysis, whose argument assumes a stage is a plain
    function of already-known states -- not true for an implicit stage, whose value
    depends on `dt` through the very solve reuse would try to skip. Carrying the
    tableau under a distinct name keeps `reuse.step_reuse_analysis` off that analysis;
    it instead dispatches to `reuse.dirk_reuse_analysis`, which decides reuse on the
    DIRK terms (stiff accuracy + explicit first stage) rather than the explicit-tableau
    formula.
    """
    tableau = getDIRKTableau(tableau_name)

    def scheme(state, dt, f, *args, **kwargs):
        return DIRK(state, dt, f, tableau, *args, name=tableau_name, **kwargs)

    scheme.__name__ = tableau_name
    scheme.dirkTableau = tableau
    return scheme


backwardEuler = dirkScheme('backwardEuler')
implicitMidpoint = dirkScheme('implicitMidpoint')
trapezoidal = dirkScheme('trapezoidal')
SDIRK2 = dirkScheme('SDIRK2')
TRBDF2 = dirkScheme('TRBDF2')
ESDIRK324L2SA = dirkScheme('ESDIRK324L2SA')
ESDIRK436L2SA = dirkScheme('ESDIRK436L2SA')
# Wave 8 (SCHEME_EXPANSION_PLAN, 2026-10-13): the remaining ODE v7.8.1
# OrdinaryDiffEqSDIRK tableaus (NOTES.md S3.28). SDIRK22 is deliberately absent:
# its ODE constructor is a stub (returns Trapezoid), so there is nothing to
# transcribe.
Kvaerno3 = dirkScheme('Kvaerno3')
Kvaerno4 = dirkScheme('Kvaerno4')
Kvaerno5 = dirkScheme('Kvaerno5')
KenCarp3 = dirkScheme('KenCarp3')
KenCarp4 = dirkScheme('KenCarp4')
KenCarp47 = dirkScheme('KenCarp47')
KenCarp5 = dirkScheme('KenCarp5')
KenCarp58 = dirkScheme('KenCarp58')
ESDIRK325L2SA = dirkScheme('ESDIRK325L2SA')
ESDIRK436L2SA2 = dirkScheme('ESDIRK436L2SA2')
ESDIRK437L2SA = dirkScheme('ESDIRK437L2SA')
ESDIRK547L2SA2 = dirkScheme('ESDIRK547L2SA2')
ESDIRK54I8L2SA = dirkScheme('ESDIRK54I8L2SA')
ESDIRK659L2SA = dirkScheme('ESDIRK659L2SA')
SSPSDIRK2 = dirkScheme('SSPSDIRK2')
ARS222 = dirkScheme('ARS222')
ARS232 = dirkScheme('ARS232')
ARS343 = dirkScheme('ARS343')
ARS443 = dirkScheme('ARS443')
BHR553 = dirkScheme('BHR553')
Cash4 = dirkScheme('Cash4')
CFNLIRK3 = dirkScheme('CFNLIRK3')
Hairer4 = dirkScheme('Hairer4')
Hairer42 = dirkScheme('Hairer42')
SFSDIRK4 = dirkScheme('SFSDIRK4')
SFSDIRK5 = dirkScheme('SFSDIRK5')
SFSDIRK6 = dirkScheme('SFSDIRK6')
SFSDIRK7 = dirkScheme('SFSDIRK7')
SFSDIRK8 = dirkScheme('SFSDIRK8')
