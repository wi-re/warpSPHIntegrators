from warpSPHIntegrators.schemes.coefficients.tableau import ButcherTableau as butcherTableau


import numpy as np


def getButcherTableau(scheme, alpha = 1/2, beta = 2/3):
    if scheme == 'forwardEuler':
        return butcherTableau(
            a = np.array([[0]]),
            b = np.array([1]),
            c = np.array([0])
        )
    elif scheme == 'generic2nd':
        return butcherTableau(
            a = np.array([[0, 0], [alpha, 0]]),
            b = np.array([1 - 1/(2*alpha), 1/(2*alpha)]),
            c = np.array([0, alpha])
        )
    elif scheme == 'midpoint':
        return butcherTableau(
            a = np.array([[0, 0], [1/2, 0]]),
            b = np.array([0, 1]),
            c = np.array([0, 1/2])
        )
    elif scheme == 'heunsMethod':
        return butcherTableau(
            a = np.array([[0, 0], [1, 0]]),
            b = np.array([1/2, 1/2]),
            c = np.array([0, 1])
        )
    elif scheme == 'ralston':
        return butcherTableau(
            a = np.array([[0, 0], [2/3, 0]]),
            b = np.array([1/4, 3/4]),
            c = np.array([0, 2/3])
        )
    elif scheme == 'generic3rd':
        return butcherTableau(
            a = np.array([[0, 0, 0],
                          [alpha, 0, 0],
                          [beta / alpha * (beta - 3 * alpha * (1-alpha)) / (3 * alpha - 2), - beta/alpha *(beta - alpha) / (3 *alpha - 2), 0]]),
            b = np.array([1 - (3 * alpha + 3 * beta -2) / ( 6 * alpha * beta), (3 * beta - 2) / (6 * alpha * (beta - alpha)), (2 -3 *alpha / (6 * beta * (beta - alpha)))]),
            c = np.array([0, alpha, beta])
        )
    elif scheme == 'RK3':
        return butcherTableau(
            a = np.array([[0, 0, 0],
                          [1/2, 0, 0],
                          [-1, 2, 0]]),
            b = np.array([1/6, 2/3, 1/6]),
            c = np.array([0, 1/2, 1])
        )
    elif scheme == 'Heun3':
        return butcherTableau(
            a = np.array([[0, 0, 0],
                          [1/3, 0, 0],
                          [0, 2/3, 0]]),
            b = np.array([1/4, 0, 3/4]),
            c = np.array([0, 1/3, 2/3])
        )
    elif scheme == 'ralston3':
        return butcherTableau(
            a = np.array([[0, 0, 0],
                          [1/2, 0, 0],
                          [0, 3/4, 0]]),
            b = np.array([2/9, 1/3, 4/9]),
            c = np.array([0, 1/2, 3/4])
        )
    elif scheme == 'Wray3':
        return butcherTableau(
            a = np.array([[0, 0, 0],
                          [8/15, 0, 0],
                          [1/4, 5/12, 0]]),
            b = np.array([1/4, 0, 3/4]),
            c = np.array([0, 8/15, 2/3])
        )
    elif scheme == 'SSPRK3':
        return butcherTableau(
            a = np.array([[0, 0, 0],
                          [1, 0, 0],
                          [1/4, 1/4, 0]]),
            b = np.array([1/6, 1/6, 2/3]),
            c = np.array([0, 1, 1/2])
        )
    elif scheme == 'SSPRK104':
        # Shu's 10-stage order-4 SSP method (Shu, "Total variation diminishing
        # Runge-Kutta time discretizations", J. Comput. Phys. 169 (2001)
        # 208-228, Sec. 4.2): a 5+5 stage cascade, full step (c[-1] = 1).
        # Measured: order 4 on all three test problems, exact SSP coefficient
        # 6.0 (stage 2's constant coefficient 1 - mu/6 is the binding
        # constraint), and a negative real-axis stability interval of
        # [-13.916, 0] (OrdinaryDiffEq.jl records 13.917). Row sums == c;
        # tests/test_tvd.py pins the SSP boundary and the real-axis interval.
        return butcherTableau(
            a = np.array([[0,       0,       0,       0,       0,       0,       0,       0,       0,       0],
                          [1/6,     0,       0,       0,       0,       0,       0,       0,       0,       0],
                          [1/6,     1/6,     0,       0,       0,       0,       0,       0,       0,       0],
                          [1/6,     1/6,     1/6,     0,       0,       0,       0,       0,       0,       0],
                          [1/6,     1/6,     1/6,     1/6,     0,       0,       0,       0,       0,       0],
                          [1/15,    1/15,    1/15,    1/15,    1/15,    0,       0,       0,       0,       0],
                          [1/15,    1/15,    1/15,    1/15,    1/15,    1/6,     0,       0,       0,       0],
                          [1/15,    1/15,    1/15,    1/15,    1/15,    1/6,     1/6,     0,       0,       0],
                          [1/15,    1/15,    1/15,    1/15,    1/15,    1/6,     1/6,     1/6,     0,       0],
                          [1/15,    1/15,    1/15,    1/15,    1/15,    1/6,     1/6,     1/6,     1/6,     0]]),
            b = np.array([1/10, 1/10, 1/10, 1/10, 1/10, 1/10, 1/10, 1/10, 1/10, 1/10]),
            c = np.array([0, 1/6, 1/3, 1/2, 2/3, 1/3, 1/2, 2/3, 5/6, 1])
        )
    # The OrdinaryDiffEq wave-1 SSP tableaus (SCHEME_EXPANSION_PLAN, 2026-10-12):
    # Ruuth's globally optimized methods SSPRK53/63/73/83 (order 3) and SSPRK54
    # (order 4) -- Ruuth, "Global optimization of explicit strong-stability-
    # preserving Runge-Kutta methods", Mathematics of Computation 75.253 (2006)
    # 183-207 -- plus the Kubatko-Yeager-Ketcheson DG-optimized methods
    # KYKSSPRK42 and KYK2014DGSSPRK_3S2 (order 2) -- "Optimal strong-stability-
    # preserving Runge-Kutta time discretizations for discontinuous Galerkin
    # methods", J. Sci. Comput. 60 (2014) 313-344. Coefficients transcribed from
    # OrdinaryDiffEq v7.8.1 (OrdinaryDiffEqSSPRK/ssprk_caches.jl alphas/betas,
    # ssprk_perform_step.jl recurrences), converted from the Shu-Osher form to
    # Butcher form (stage states as (u_i - u^n)/dt; c = row sums of a, which
    # match ODE's stored c to 1e-13). All are plain single-step methods, none
    # FSAL, so no first-stage reuse. tests/test_tvd.py pins each SSP boundary
    # and real-axis interval.
    elif scheme == 'SSPRK53':
        # Measured: order 3, SSP coefficient 2.6506 (ODE publishes 2.65),
        # real-axis interval [-6.214, 0]. Row sums == c.
        return butcherTableau(
            a = np.array([[0.0, 0.0, 0.0, 0.0, 0.0],
                          [0.377268915331368, 0.0, 0.0, 0.0, 0.0],
                          [0.377268915331368, 0.377268915331368, 0.0, 0.0, 0.0],
                          [0.24299522053739583, 0.24299522053739583, 0.242995220537396, 0.0, 0.0],
                          [0.15358906769512648, 0.15358906769512648, 0.1535890676951266, 0.23845893284629, 0.0]]),
            b = np.array([0.20673402086480447, 0.20673402086480447, 0.11709725184184414, 0.18180256012013943, 0.287632146308408]),
            c = np.array([0.0, 0.377268915331368, 0.754537830662736, 0.7289856616121877, 0.6992261359316696])
        )
    elif scheme == 'SSPRK63':
        # Measured: order 3, SSP coefficient 3.5184 (ODE publishes 3.518),
        # real-axis interval [-7.213, 0]. Row sums == c.
        return butcherTableau(
            a = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                          [0.284220721334261, 0.0, 0.0, 0.0, 0.0, 0.0],
                          [0.284220721334261, 0.284220721334261, 0.0, 0.0, 0.0, 0.0],
                          [0.284220721334261, 0.284220721334261, 0.284220721334261, 0.0, 0.0, 0.0],
                          [0.1487128616603831, 0.12071378576592966, 0.12071378576592966, 0.12071378576593, 0.0, 0.0],
                          [0.1487128616603831, 0.12071378576592966, 0.12071378576592966, 0.12071378576593, 0.284220721334261, 0.0]]),
            b = np.array([0.16974662234923632, 0.14609361068522916, 0.10197638641686799, 0.10197638641686826, 0.24010349706589984, 0.2401034970659]),
            c = np.array([0.0, 0.284220721334261, 0.568441442668522, 0.852662164002783, 0.5108542189581724, 0.7950749402924334])
        )
    elif scheme == 'SSPRK73':
        # Measured: order 3, SSP coefficient 4.2879 (ODE publishes 4.2879; ODE
        # stores alpha73/alpha76 at 6 digits, so its coefficients carry that
        # slack), real-axis interval [-8.888, 0]. Row sums == c.
        return butcherTableau(
            a = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                          [0.233213863663009, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                          [0.233213863663009, 0.233213863663009, 0.0, 0.0, 0.0, 0.0, 0.0],
                          [0.233213863663009, 0.233213863663009, 0.233213863663009, 0.0, 0.0, 0.0, 0.0],
                          [0.19007802386584471, 0.19007802386584471, 0.19007802386584471, 0.190078023865845, 0.0, 0.0, 0.0],
                          [0.16930787981247397, 0.0958849178781437, 0.0958849178781437, 0.09588491787814384, 0.117644805593912, 0.0, 0.0],
                          [0.16930787981247397, 0.0958849178781437, 0.0958849178781437, 0.09588491787814384, 0.117644805593912, 0.233213863663009, 0.0]]),
            b = np.array([0.17698931516532443, 0.11239171983253872, 0.11239171983253872, 0.08435964663410883, 0.10350401760632937, 0.205181790464579, 0.205181790464579]),
            c = np.array([0.0, 0.233213863663009, 0.466427727326018, 0.699641590989027, 0.7603120954633792, 0.5746074390408172, 0.8078213027038262])
        )
    elif scheme == 'SSPRK83':
        # Measured: order 3, SSP coefficient 5.1071 (ODE publishes 5.107),
        # real-axis interval [-10.986, 0]. Row sums == c.
        return butcherTableau(
            a = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                          [0.195804015330143, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                          [0.195804015330143, 0.195804015330143, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                          [0.195804015330143, 0.195804015330143, 0.195804015330143, 0.0, 0.0, 0.0, 0.0, 0.0],
                          [0.195804015330143, 0.195804015330143, 0.195804015330143, 0.195804015330143, 0.0, 0.0, 0.0, 0.0],
                          [0.1132986712473457, 0.11213375462167291, 0.11213375462167291, 0.11213375462167291, 0.112133754621673, 0.0, 0.0, 0.0],
                          [0.11364964986110604, 0.11165673643345277, 0.11165673643345277, 0.11165673643345277, 0.11165673643345285, 0.194971062960412, 0.0, 0.0],
                          [0.14221023579187003, 0.14091014951805214, 0.12047209837964422, 0.07283978741985271, 0.07283978741985275, 0.12719027290864166, 0.127733653231944, 0.0]]),
            b = np.array([0.14221023579187003, 0.14091014951805214, 0.12047209837964422, 0.07283978741985271, 0.07283978741985275, 0.12719027290864166, 0.127733653231944, 0.195804015330143]),
            c = np.array([0.0, 0.195804015330143, 0.391608030660286, 0.587412045990429, 0.783216061320572, 0.5618336897340375, 0.7552476585553292, 0.8041959846698575])
        )
    elif scheme == 'SSPRK54':
        # Measured: order 4, SSP coefficient 1.7693 (all-stage upwind model; the
        # final map alone is convex to 1.8611), real-axis interval [-5.331, 0].
        # ODE's published 1.508 is the paper's coefficient, which the library's
        # stage-map convexity measurement does not reproduce. Row sums == c.
        return butcherTableau(
            a = np.array([[0.0, 0.0, 0.0, 0.0, 0.0],
                          [0.39175222657189, 0.0, 0.0, 0.0, 0.0],
                          [0.2176690962611688, 0.368410593050371, 0.0, 0.0, 0.0],
                          [0.0826920866578106, 0.13995850219189537, 0.251891774271694, 0.0, 0.0],
                          [0.06796628363711477, 0.11503469850463158, 0.20703489859738564, 0.544974750228521, 0.0]]),
            b = np.array([0.1468118760847866, 0.2484829094449762, 0.10425883033198097, 0.2744389009013507, 0.226007483236906]),
            c = np.array([0.0, 0.39175222657189, 0.5860796893115398, 0.47454236312139997, 0.935010630967653])
        )
    elif scheme == 'KYKSSPRK42':
        # Measured: order 2, SSP coefficient 2.2838 (all-stage upwind model),
        # real-axis interval [-4.954, 0]. ODE's published 2.459 matches the
        # FINAL-map-only convexity CFL (2.4595), not the stage maps, which are
        # the binding constraint here. Row sums == c.
        return butcherTableau(
            a = np.array([[0.0, 0.0, 0.0, 0.0],
                          [0.406584463657504, 0.0, 0.0, 0.0],
                          [0.24606229845682182, 0.246062298456822, 0.0, 0.0],
                          [0.25901120328686006, 0.24537398664540927, 0.405447122055692, 0.0]]),
            b = np.array([0.20997096222437225, 0.1833284972503433, 0.30292539407757624, 0.303775146447707]),
            c = np.array([0.0, 0.406584463657504, 0.4921245969136438, 0.9098323119879613])
        )
    elif scheme == 'KYK2014DGSSPRK_3S2':
        # Measured: order 2, SSP coefficient 1.8939, real-axis interval
        # [-4.267, 0]. ODE's published 0.8417 is the paper's DG-context
        # coefficient, not the upwind-model stage-map convexity CFL. Row sums
        # == c.
        return butcherTableau(
            a = np.array([[0.0, 0.0, 0.0],
                          [0.528005024856522, 0.0, 0.0],
                          [0.4818821386339936, 0.481882138633993, 0.0]]),
            b = np.array([0.3384803991135379, 0.3156535616530465, 0.345866039233415]),
            c = np.array([0.0, 0.528005024856522, 0.9637642772679866])
        )
    # The OrdinaryDiffEq wave-2 SSP tableaus (SCHEME_EXPANSION_PLAN, 2026-10-12):
    # the embedded 3(2) pairs SSPRK43 and SSPRK432 (order-3 main / order-2
    # estimate, 4 stages) and the high-stage SSPRK932 (order 3, 9 stages,
    # fixed-step only here). Coefficients transcribed from OrdinaryDiffEq v7.8.1
    # (OrdinaryDiffEqSSPRK/ssprk_perform_step.jl recurrences), which are already
    # in Butcher form here (exact rationals; c = row sums of a).
    #   SSPRK43   -- Kraaijevanger, "Contractivity of Runge-Kutta methods", BIT
    #                31.3 (1991) 482-528 (main); Conde, Fekete & Shadid,
    #                arXiv:1806.08693 (embedded pair).
    #   SSPRK432  -- Gottlieb, Ketcheson & Shu, "Strong Stability Preserving
    #                Runge-Kutta and Multistep Time Discretizations", World
    #                Scientific (2011), Example 6.1 (low-storage 3(2) variant).
    #   SSPRK932  -- same GKS (2011) book; ODE's docstring suggests SSPRK43
    #                instead. Registered FIXED-STEP only: ODE's adaptive
    #                estimate (uprev + 6*u6 + 6*dt*k)/7 - u_final has leading
    #                term (5/7)*dt*u'(t) (the 1/7, 6/7, 6/7 coefficients sum
    #                to 13/7 -- utilde is not an embedded solution), i.e.
    #                O(dt) instead of the O(dt^3) a 3(2) pair would need, so
    #                the library does not register it as an error estimate.
    # All three are non-FSAL (no first-stage reuse). tests/test_tvd.py pins each
    # SSP boundary and real-axis interval; the 3(2) pairs get estimate-rate and
    # adaptive tests via tests/test_embedded.py.
    elif scheme == 'SSPRK43':
        # 3(2) pair. Measured: order 3 (estimate O(h^3)), SSP coefficient 2.0
        # (ODE publishes 2.0), real-axis interval [-5.149, 0]. b = (b_main,
        # b_embed) with b_embed = (1/4, 1/4, 1/4, 1/4). Row sums == c.
        return butcherTableau(
            a = np.array([[0.0, 0.0, 0.0, 0.0],
                          [1/2, 0.0, 0.0, 0.0],
                          [1/2, 1/2, 0.0, 0.0],
                          [1/6, 1/6, 1/6, 0.0]]),
            b = (np.array([1/6, 1/6, 1/6, 1/2]),
                 np.array([1/4, 1/4, 1/4, 1/4])),
            c = np.array([0.0, 1/2, 1.0, 1/2])
        )
    elif scheme == 'SSPRK432':
        # 3(2) pair, low-storage variant of SSPRK43 (same a, c, b_main).
        # Measured: order 3 (estimate O(h^3)), SSP coefficient 2.0 (ODE
        # publishes 2.0), real-axis interval [-5.149, 0]. b_embed =
        # (1/3, 1/3, 1/3, 0). Row sums == c.
        return butcherTableau(
            a = np.array([[0.0, 0.0, 0.0, 0.0],
                          [1/2, 0.0, 0.0, 0.0],
                          [1/2, 1/2, 0.0, 0.0],
                          [1/6, 1/6, 1/6, 0.0]]),
            b = (np.array([1/6, 1/6, 1/6, 1/2]),
                 np.array([1/3, 1/3, 1/3, 0.0])),
            c = np.array([0.0, 1/2, 1.0, 1/2])
        )
    elif scheme == 'SSPRK932':
        # Order-3, 9 stages, FIXED-STEP (no embedded b: see group comment).
        # Measured: order 3, SSP coefficient 6.0 (ODE publishes 6.0), real-axis
        # interval [-13.290, 0]. A 6-stage dt/6 cascade followed by three
        # recovery stages. Row sums == c.
        return butcherTableau(
            a = np.array([[0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0],
                          [1/6,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0],
                          [1/6,  1/6,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0],
                          [1/6,  1/6,  1/6,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0],
                          [1/6,  1/6,  1/6,  1/6,  0.0,  0.0,  0.0,  0.0,  0.0],
                          [1/6,  1/6,  1/6,  1/6,  1/6,  0.0,  0.0,  0.0,  0.0],
                          [1/6,  1/15, 1/15, 1/15, 1/15, 1/15, 0.0,  0.0,  0.0],
                          [1/6,  1/15, 1/15, 1/15, 1/15, 1/15, 1/6,  0.0,  0.0],
                          [1/6,  1/15, 1/15, 1/15, 1/15, 1/15, 1/6,  1/6,  0.0]]),
            b = np.array([1/6, 1/15, 1/15, 1/15, 1/15, 1/15, 1/6, 1/6, 1/6]),
            c = np.array([0.0, 1/6, 1/3, 1/2, 2/3, 5/6, 1/2, 2/3, 5/6])
        )
    # The OrdinaryDiffEq wave-5 SSP tableaus (SCHEME_EXPANSION_PLAN, 2026-10-13):
    # Higueras & Roldan's new third-order low-storage SSP methods (arXiv:1809.04807).
    # Screened before registration: all three are algebraically DISTINCT one-step
    # maps from SSPRK53 (different stages and final b-row; one-step-map diff
    # O(dt^4) ~ 1.5e-4..4.6e-3 per unit dt^4, not ~1e-16), so they are their own
    # methods, not storage rearrangements of it (NOTES.md S3.25). ODE's storage
    # saving (1-2 temporaries) does not apply to this driver, which builds each
    # stage state fresh from the a-row. Coefficients transcribed from
    # OrdinaryDiffEq v7.8.1 (ssprk_caches.jl alphas/betas,
    # ssprk_perform_step.jl recurrences), converted to Butcher form as in wave 1
    # (row sums == ODE's stored c to 1e-15; order-3 conditions to <=5e-15).
    # None are FSAL.
    elif scheme == 'SSPRK53_2N1':
        # Measured: order 3, SSP coefficient 2.1808 (ODE publishes 2.18),
        # real-axis interval [-5.527, 0]. Row sums == c. c_3 = 1.0053 > 1: stage
        # 3 evaluates just past the step end, as ODE does.
        return butcherTableau(
            a = np.array([[0.0, 0.0, 0.0, 0.0, 0.0],
                          [0.443568244942995, 0.0, 0.0, 0.0, 0.0],
                          [0.443568244942995, 0.291111420073766, 0.0, 0.0, 0.0],
                          [0.443568244942995, 0.291111420073766, 0.270612601278217, 0.0, 0.0],
                          [0.1901117921952908, 0.12476933240758091, 0.11598361065328995, 0.110577759392786, 0.0]]),
            b = np.array([0.1901117921952908, 0.12476933240758091, 0.11598361065328995, 0.110577759392786, 0.458557505351052]),
            c = np.array([0.0, 0.443568244942995, 0.734679665016762, 1.005292266294979, 0.541442494648948])
        )
    elif scheme == 'SSPRK53_2N2':
        # Measured: order 3, SSP coefficient 2.1487 (ODE publishes 2.148),
        # real-axis interval [-7.263, 0] -- the largest of the four despite the
        # smallest SSP coefficient. Row sums == c.
        return butcherTableau(
            a = np.array([[0.0, 0.0, 0.0, 0.0, 0.0],
                          [0.465388589249323, 0.0, 0.0, 0.0, 0.0],
                          [0.465388589249323, 0.465388589249323, 0.0, 0.0, 0.0],
                          [0.14783400776685549, 0.14783400776685549, 0.124745797313998, 0.0, 0.0],
                          [0.14783400776685549, 0.14783400776685549, 0.124745797313998, 0.465388589249323, 0.0]]),
            b = np.array([0.14114733153392192, 0.14114733153392192, 0.11910342333890192, 0.4443386098445868, 0.154263303748666]),
            c = np.array([0.0, 0.465388589249323, 0.930777178498646, 0.42041381284771, 0.885802402097033])
        )
    elif scheme == 'SSPRK53_H':
        # Measured: order 3, SSP coefficient 2.6506 (ODE publishes 2.65),
        # real-axis interval [-6.213, 0] (the same as SSPRK53's to measurement
        # tolerance). Shares SSPRK53's first two nodes (c1, c2) but has distinct
        # later stages. Row sums == c.
        return butcherTableau(
            a = np.array([[0.0, 0.0, 0.0, 0.0, 0.0],
                          [0.377268915331368, 0.0, 0.0, 0.0, 0.0],
                          [0.377268915331368, 0.377268915331368, 0.0, 0.0, 0.0],
                          [0.26081197914449766, 0.26081197914449766, 0.260811979144498, 0.0, 0.0],
                          [0.21915343633198697, 0.11709725184184361, 0.11709725184184376, 0.169383144652957, 0.0]]),
            b = np.array([0.21915343633198697, 0.11709725184184361, 0.11709725184184376, 0.169383144652957, 0.377268915331368]),
            c = np.array([0.0, 0.377268915331368, 0.754537830662737, 0.782435937433493, 0.622731084668631])
        )
    elif scheme == 'RK4':
        return butcherTableau(
            a = np.array([[0, 0, 0, 0],
                          [1/2, 0, 0, 0],
                          [0, 1/2, 0, 0],
                          [0, 0, 1, 0]]),
            b = np.array([1/6, 1/3, 1/3, 1/6]),
            c = np.array([0, 1/2, 1/2, 1])
        )
    elif scheme == 'RK4alt':
        return butcherTableau(
            a = np.array([[0, 0, 0, 0],
                          [1/3, 0, 0, 0],
                          [-1/3, 1, 0, 0],
                          [1, -1, 1, 0]]),
            b = np.array([1/8, 3/8, 3/8, 1/8]),
            c = np.array([0, 1/3, 2/3, 1])
        )
    elif scheme == 'Nystrom5':
        return butcherTableau(
            a = np.array([[0,       0,      0,     0, 0,0],
                          [1/3,     0,      0,     0, 0, 0],
                          [4/25, 6/25,      0,     0, 0, 0],
                          [1/4,     -3,  15/4,     0, 0, 0],
                          [2/27,  10/9, -50/81, 8/81, 0, 0],
                          [2/25, 12/25,   2/15, 8/75, 0 ,0]]),
            b = np.array([23/192, 0, 125/192,  0, -27/64, 125/192]),
            c = np.array([0, 1/3, 2/5, 1, 2/3, 4/5])
        )
    # The OrdinaryDiffEq wave-6 tableaus (SCHEME_EXPANSION_PLAN, 2026-10-13): the
    # fixed-step workhorses of LowOrderRK (the embedded pairs follow below).
    # Coefficients are the exact Float64s ODE stores in its ConstantCache
    # (extracted with a local Julia probe, transcribed by script, zero hand
    # entry), and every tableau reproduces ODE's own perform_step bit-for-bit in
    # a one-step cross-check (max|diff| = 0.0 vs solve(..., adaptive=false)).
    # Plain tableaus on the generic RK driver.
    elif scheme == 'Alshina3':
        # Alshina et al. (2008) "3rd order, 3-stage method with optimal
        # parameters". Registered fixed-step: ODE's b2tilde estimate is not a
        # valid embedded branch (its implied map is not a solution). Measured:
        # order 3, SSP coefficient 1.0, real-axis interval [-2.512, 0].
        # Row sums == c.
        return butcherTableau(
            a = np.array([
                          [0.0, 0.0, 0.0],
                          [0.5, 0.0, 0.0],
                          [0.0, 0.75, 0.0],
            ]),
            b = np.array([0.2222222222222222, 0.3333333333333333, 0.4444444444444444]),
            c = np.array([0.0, 0.5, 0.75])
        )
    elif scheme == 'Alshina6':
        # Alshina et al. (2008) "6th order, 7-stage method with optimal
        # parameters". Non-monotone but in-range nodes (c5 = 0.27639, c6 =
        # 0.72361). Measured: order 6, SSP coefficient 0.0, real-axis interval
        # [-2.904, 0]. Row sums == c.
        return butcherTableau(
            a = np.array([
                          [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                          [0.5714285714285714, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                          [1.0267857142857142, -0.3125, 0.0, 0.0, 0.0, 0.0, 0.0],
                          [0.9349206349206349, 0.2777777777777778, -0.35555555555555557, 0.0, 0.0, 0.0, 0.0],
                          [0.18002567144208434, 0.14737940683961614, 0.016070163293314288, -0.0670820393249937, 0.0, 0.0, 0.0],
                          [-0.07979765856603138, 0.025290591998992584, -0.3516326202266813, 0.3207294901687516, 0.8090169943749475, 0.0, 0.0],
                          [0.4988599356197352, -0.8633499941930429, 1.6778122846668349, -1.2682372542187894, -0.42705098312484235, 1.381966011250105, 0.0],
            ]),
            b = np.array([0.08333333333333333, 0.0, 0.0, 0.0, 0.4166666666666667, 0.4166666666666667, 0.08333333333333333]),
            c = np.array([0.0, 0.5714285714285714, 0.7142857142857143, 0.8571428571428571, 0.276393202250021, 0.7236067977499789, 1.0])
        )
    elif scheme == 'Ralston4':
        # Ralston (1962) "4th order Ralston method with minimum truncation
        # error for 4-stage explicit Runge-Kutta". Measured: order 4, SSP
        # coefficient 0.7209, real-axis interval [-2.784, 0]. Row sums == c.
        return butcherTableau(
            a = np.array([
                          [0.0, 0.0, 0.0, 0.0],
                          [0.4, 0.0, 0.0, 0.0],
                          [0.29697760924775357, 0.15875964497103584, 0.0, 0.0],
                          [0.21810038822592004, -3.0509651486929306, 3.8328647604670105, 0.0],
            ]),
            b = np.array([0.1747602822626904, -0.551480662878733, 1.2055355993965235, 0.17118478121951902]),
            c = np.array([0.0, 0.4, 0.4557372542187894, 1.0])
        )
    elif scheme == 'RKO65':
        # Tsitouras' Runge-Kutta-Oliver 6-stage 5th-order method, with a FREE
        # first stage: k1 = f(t + (2/3) dt, y^n), so the first row of a is all
        # zero and the row sums are NOT c (structural, as in ODE). Measured:
        # order 5, SSP coefficient 0.0, real-axis interval [-4.164, 0].
        return butcherTableau(
            a = np.array([
                          [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                          [0.16666666666666666, 0.0, 0.0, 0.0, 0.0, 0.0],
                          [-1.875, 2.625, 0.0, 0.0, 0.0, 0.0],
                          [-9.0, 10.714285714285714, -0.7142857142857143, 0.0, 0.0, 0.0],
                          [-3.0, 3.9150857142857145, -0.13028571428571428, 0.0152, 0.0, 0.0],
                          [0.0, 0.3236842105263158, 2.5, 0.15, -1.9736842105263157, 0.0],
            ]),
            b = np.array([0.0, 0.40601503759398494, 1.5238095238095237, 0.05555555555555555, -1.0964912280701755, 0.1111111111111111]),
            c = np.array([0.6666666666666666, 0.16666666666666666, 0.75, 1.0, 0.8, 1.0])
        )
    elif scheme == 'RKM':
        # Mead-Renaut 6-stage 1-2-1-1-1 low-storage method, "designed to have
        # good stability properties when applied to pseudospectral
        # discretizations". ODE's docstring targets that setting; on general
        # problems the order conditions give order 3, which is what registers.
        # Measured: order 3, SSP coefficient 0.7531, real-axis interval
        # [-3.224, 0]. Row sums == c.
        return butcherTableau(
            a = np.array([
                          [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                          [0.167266187050662, 0.0, 0.0, 0.0, 0.0, 0.0],
                          [0.0, 0.484574582244783, 0.0, 0.0, 0.0, 0.0],
                          [0.0, 0.0, 0.536909403373491, 0.0, 0.0, 0.0],
                          [0.0, 0.0, 0.0, 0.082069535961948, 0.0, 0.0],
                          [0.0, 0.0, 0.0, 0.0, 0.853923000035347, 0.0],
            ]),
            b = np.array([-0.028289441132839, 0.46396891856471, -0.434414348751899, 0.693796229087598, 0.0, 0.30493864223243]),
            c = np.array([0.0, 0.167266187050662, 0.484574582244783, 0.536909403373491, 0.082069535961948, 0.853923000035347])
        )
    elif scheme == 'MSRK5':
        # Stepanov (arXiv:2202.08443, Fig. 3) 5th-order method, 9 stages =
        # 8 + FSAL (fixed-step). Measured: order 5, SSP coefficient 0.3226,
        # real-axis interval [-4.366, 0]. Row sums == c.
        return butcherTableau(
            a = np.array([
                          [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                          [0.08888888888888889, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                          [0.03333333333333333, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                          [0.05, 0.0, 0.15, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                          [0.5, 0.0, -1.875, 1.875, 0.0, 0.0, 0.0, 0.0, 0.0],
                          [-0.08148148148148149, 0.0, 0.5111111111111111, -0.07407407407407407, 0.17777777777777778, 0.0, 0.0, 0.0, 0.0],
                          [0.046296296296296294, 0.0, 0.4861111111111111, -0.27314814814814814, -0.9259259259259259, 1.5, 0.0, 0.0, 0.0],
                          [0.2421875, 0.0, -1.6881696428571429, 2.080357142857143, 1.5450892857142857, -1.7075892857142858, 0.478125, 0.0, 0.0],
                          [0.06359649122807018, 0.0, 0.0, 0.2894736842105263, 0.07407407407407407, 0.275, 0.21052631578947367, 0.08732943469785576, 0.0],
            ]),
            b = np.array([0.06359649122807018, 0.0, 0.0, 0.2894736842105263, 0.07407407407407407, 0.275, 0.21052631578947367, 0.08732943469785576, 0.0]),
            c = np.array([0.0, 0.08888888888888889, 0.13333333333333333, 0.2, 0.5, 0.5333333333333333, 0.8333333333333334, 0.95, 1.0])
        )
    elif scheme == 'MSRK6':
        # Stepanov (arXiv:2202.08443, Table 4) 6th-order method, 9 stages =
        # 8 + FSAL (fixed-step). Measured: order 6, SSP coefficient 0.1009,
        # real-axis interval [-4.164, 0]. Row sums == c.
        return butcherTableau(
            a = np.array([
                          [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                          [0.07142857142857142, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                          [0.0, 0.14285714285714285, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                          [0.05357142857142857, 0.0, 0.16071428571428573, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                          [0.4027777777777778, 0.0, -1.4583333333333333, 1.5555555555555556, 0.0, 0.0, 0.0, 0.0, 0.0],
                          [-0.30357142857142855, 0.0, 1.6607142857142858, -1.1428571428571428, 0.42857142857142855, 0.0, 0.0, 0.0, 0.0],
                          [0.14504373177842567, 0.0, -0.9948979591836735, 1.6058673469387754, -0.7024416909620991, 0.8035714285714286, 0.0, 0.0, 0.0],
                          [0.19155336771370526, 0.0, 1.5777074542897327, -2.491610017190186, 2.9752988748241913, -1.842573058290358, 0.5896233786529145, 0.0, 0.0],
                          [0.06584362139917696, 0.0, 0.0, 0.31438458660680885, 0.17666666666666667, 0.19761316872427984, 0.19761316872427984, 0.04787878787878788, 0.0],
            ]),
            b = np.array([0.06584362139917696, 0.0, 0.0, 0.31438458660680885, 0.17666666666666667, 0.19761316872427984, 0.19761316872427984, 0.04787878787878788, 0.0]),
            c = np.array([0.0, 0.07142857142857142, 0.14285714285714285, 0.21428571428571427, 0.5, 0.6428571428571429, 0.8571428571428571, 1.0, 1.0])
        )
    elif scheme == 'SIR54':
        # Kovalnogov et al. "5th order method suited for SIR-type epidemic
        # models", 8 stages = 7 + FSAL (fixed-step): ODE's btilde estimate is
        # not a valid embedded branch (sums to 7.85, not 0), so only the
        # propagated weights register. Measured: order 5, SSP coefficient 0.0,
        # real-axis interval [-3.512, 0]. Row sums == c.
        return butcherTableau(
            a = np.array([
                          [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                          [0.22499185714559425, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                          [0.08915336103206711, 0.23896021111688856, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                          [1.3050223986305804, -5.501775490886399, 5.1411313952931605, 0.0, 0.0, 0.0, 0.0, 0.0],
                          [1.7623401292754435, -7.4480042847020265, 6.708799990212466, -0.03413773303571631, 0.0, 0.0, 0.0, 0.0],
                          [1.8741291161981875, -7.9146797771177075, 7.079854670922641, -0.024494533823827536, -0.01480947617929343, 0.0, 0.0, 0.0],
                          [0.09869714985516642, 0.0010072934687415066, 0.49511836687375954, 3.937670696186872, -13.739542408717368, 10.20704890233283, 0.0, 0.0],
                          [0.09869714985516642, 0.0010072934687415066, 0.49511836687375954, 3.937670696186872, -13.739542408717368, 10.20704890233283, 0.0, 0.0],
            ]),
            b = np.array([0.09869714985516642, 0.0010072934687415066, 0.49511836687375954, 3.937670696186872, -13.739542408717368, 10.20704890233283, 0.0, 0.0]),
            c = np.array([0.0, 0.22499185714559425, 0.32811357214895565, 0.9443783030373425, 0.9889981017501664, 1.0, 1.0, 1.0])
        )
    elif scheme == 'PSRK4p7q6':
        # Aubry-Chartier (1998) pseudo-symplectic method, 6 stages (fixed-
        # step); the largest real-axis interval of the wave, [-11.41, 0].
        # Measured: order 4, SSP coefficient 0.0. Row sums == c.
        return butcherTableau(
            a = np.array([
                          [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                          [0.23593376536652, 0.0, 0.0, 0.0, 0.0, 0.0],
                          [0.34750735658424, -0.13561935398346, 0.0, 0.0, 0.0, 0.0],
                          [-0.20592852403227, 1.89179076622108, -0.89775024478958, 0.0, 0.0, 0.0],
                          [-0.09435493281455, 1.75617141223762, -0.96707850476948, 0.06932825997989, 0.0, 0.0],
                          [0.14157883255197, -1.17039696277833, 1.30579112376331, -2.20354136855289, 2.92656837501595, 0.0],
            ]),
            b = np.array([0.07078941627598, 0.87808570611881, -0.44887512239479, -0.44887512239479, 0.87808570611881, 0.07078941627598]),
            c = np.array([0.0, 0.23593376536652, 0.21188800260078, 0.78811199739923, 0.76406623463348, 1.0])
        )
    elif scheme == 'PSRK3p6q5':
        # Aubry-Chartier (1998) pseudo-symplectic method, 5 stages (fixed-
        # step). Measured: order 3, SSP coefficient 0.0, real-axis interval
        # [-3.548, 0]. Row sums == c.
        return butcherTableau(
            a = np.array([
                          [0.0, 0.0, 0.0, 0.0, 0.0],
                          [0.13502027922909, 0.0, 0.0, 0.0, 0.0],
                          [-0.47268213605237, 1.05980250415419, 0.0, 0.0, 0.0],
                          [-1.21650460595689, 2.16217630216753, -0.37234592426536, 0.0, 0.0],
                          [0.33274443036387, -0.20882668296587, 1.87865617737921, -1.00257392477721, 0.0],
            ]),
            b = np.array([0.04113894457092, 0.26732123194414, 0.86700906289955, -0.30547139552036, 0.13000215610576]),
            c = np.array([0.0, 0.13502027922909, 0.58712036810182, 0.57332577194528, 1.0])
        )
    elif scheme == 'PSRK3p5q4':
        # Aubry-Chartier (1998) pseudo-symplectic method, 4 stages (fixed-
        # step). Measured: order 3, SSP coefficient 0.0, real-axis interval
        # [-2.784, 0]. Row sums == c.
        return butcherTableau(
            a = np.array([
                          [0.0, 0.0, 0.0, 0.0],
                          [0.375, 0.0, 0.0, 0.0],
                          [0.9166666666666666, -0.6666666666666666, 0.0, 0.0],
                          [-0.08333333333333333, 1.8333333333333333, -0.75, 0.0],
            ]),
            b = np.array([0.1111111111111111, 0.8888888888888888, -0.2222222222222222, 0.2222222222222222]),
            c = np.array([0.0, 0.375, 0.25, 1.0])
        )
    # The OrdinaryDiffEq wave-7 plain high-order tableaus
    # (SCHEME_EXPANSION_PLAN, 2026-10-13): Dormand-Prince 8 and the
    # Feagin 10/12/14 methods. Coefficients from ODE v7.8.1's caches,
    # cross-checked one step for bit-for-bit agreement with the real ODE
    # solvers (NOTES.md 3.27). Row sums == c.
    elif scheme == 'DP8':
        # Dormand-Prince 8th order, 12 stages (ODE's HighOrderRK DP8). Not FSAL: ODE
        # recomputes f fresh each step (the last-row a's != b). ODE's error estimate is a dual
        # 5th+3rd-order companion (no (p-1)=7th-order embedded solution), so it registers
        # plain, like the Feagin methods. Row sums == c (c12 = 1 is the last stage).
        return butcherTableau(
            a = np.array([
                          [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                          [0.05260015195876773, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                          [0.0197250569845379, 0.0591751709536137, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                          [0.02958758547680685, 0.0, 0.08876275643042054, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                          [0.2413651341592667, 0.0, -0.8845494793282861, 0.924834003261792, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                          [0.037037037037037035, 0.0, 0.0, 0.17082860872947386, 0.12546768756682242, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                          [0.037109375, 0.0, 0.0, 0.17025221101954405, 0.06021653898045596, -0.017578125, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                          [0.03709200011850479, 0.0, 0.0, 0.17038392571223998, 0.10726203044637328, -0.015319437748624402, 0.008273789163814023, 0.0, 0.0, 0.0, 0.0, 0.0],
                          [0.6241109587160757, 0.0, 0.0, -3.3608926294469414, -0.868219346841726, 27.59209969944671, 20.154067550477894, -43.48988418106996, 0.0, 0.0, 0.0, 0.0],
                          [0.47766253643826434, 0.0, 0.0, -2.4881146199716677, -0.590290826836843, 21.230051448181193, 15.279233632882423, -33.28821096898486, -0.020331201708508627, 0.0, 0.0, 0.0],
                          [-0.9371424300859873, 0.0, 0.0, 5.186372428844064, 1.0914373489967295, -8.149787010746927, -18.52006565999696, 22.739487099350505, 2.4936055526796523, -3.0467644718982196, 0.0, 0.0],
                          [2.273310147516538, 0.0, 0.0, -10.53449546673725, -2.0008720582248625, -17.9589318631188, 27.94888452941996, -2.8589982771350235, -8.87285693353063, 12.360567175794303, 0.6433927460157636, 0.0],
            ]),
            b = np.array([0.054293734116568765, 0.0, 0.0, 0.0, 0.0, 4.450312892752409, 1.8915178993145003, -5.801203960010585, 0.3111643669578199, -0.1521609496625161, 0.20136540080403034, 0.04471061572777259]),
            c = np.array([0.0, 0.05260015195876773, 0.0789002279381516, 0.1183503419072274, 0.2816496580927726, 0.3333333333333333, 0.25, 0.30769230769230765, 0.651282051282051, 0.5999999999999982, 0.857142857142858, 1.0000000000000007])
        )
    elif scheme == 'Feagin10':
        # Feagin (1980s) 10th order, 17 stages (ODE's Feagin10). Symmetric method;
        # the interior c-duplicates are its split-stage structure. No embedded branch in
        # ODE's cache, so it registers plain. Row sums == c.
        return butcherTableau(
            a = np.array([
                          [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                          [0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                          [-0.9151765613752915, 1.4545344021782731, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                          [0.20225919030111816, 0.0, 0.6067775709033545, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                          [0.18402471470864357, 0.0, 0.19796683122719236, -0.07295478473136326, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                          [0.08790073402066813, 0.0, 0.0, 0.41045970252026065, 0.4827137536788665, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                          [0.08597005049024603, 0.0, 0.0, 0.3308859630407222, 0.4896629573094502, -0.07318563750708508, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                          [0.12093044912533372, 0.0, 0.0, 0.0, 0.2601246757582956, 0.032540262154909134, -0.0595780211817361, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                          [0.11085437958039149, 0.0, 0.0, 0.0, 0.0, -0.06057614882550056, 0.3217637056017784, 0.510485725608063, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                          [0.112054414752879, 0.0, 0.0, 0.0, 0.0, -0.14494277590286592, -0.3332697190962567, 0.4992692295568801, 0.5095046089296861, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                          [0.11397678396418598, 0.0, 0.0, 0.0, 0.0, -0.07688133642033569, 0.23952736032439065, 0.3977746623680946, 0.010755895687360746, -0.3277691241640189, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                          [0.07983145282801961, 0.0, 0.0, 0.0, 0.0, -0.052032968680060306, -0.05769541461685489, 0.19478191571210415, 0.14538492318832508, -0.07829427103516708, -0.11450329936109892, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                          [0.9851156101648573, 0.0, 0.0, 0.3308859630407222, 0.4896629573094502, -1.3789648657484357, -0.8611641950276356, 5.784288136375372, 3.2880776198510357, -2.386339050931364, -3.254793424836439, -2.16343541686423, 0.0, 0.0, 0.0, 0.0, 0.0],
                          [0.8950802957716328, 0.0, 0.19796683122719236, -0.07295478473136326, 0.0, -0.8512362396620076, 0.3983201123185333, 3.639372631810356, 1.5482287703983033, -2.122217147040537, -1.5835039854532618, -1.7156160828593627, -0.024403640575012746, 0.0, 0.0, 0.0, 0.0],
                          [-0.9151765613752915, 1.4545344021782731, 0.0, 0.0, -0.7773336436449683, 0.0, -0.0910895662155176, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0910895662155176, 0.7773336436449683, 0.0, 0.0, 0.0],
                          [0.1, 0.0, -0.15717866579977116, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.15717866579977116, 0.0, 0.0],
                          [0.1817813007000953, 0.675, 0.3427581598471898, 0.0, 0.25911121454832275, -0.35827896671795206, -1.0459489594088331, 0.930327845415627, 1.7795095943170811, 0.1, -0.2825475695390441, -0.15932735011997254, -0.14551589464700151, -0.25911121454832275, -0.3427581598471898, -0.675, 0.0],
            ]),
            b = np.array([0.03333333333333333, 0.025, 0.03333333333333333, 0.0, 0.05, 0.0, 0.04, 0.0, 0.1892374781489235, 0.2774291885177432, 0.2774291885177432, 0.1892374781489235, -0.04, -0.05, -0.03333333333333333, -0.025, 0.03333333333333333]),
            c = np.array([0.0, 0.1, 0.5393578408029817, 0.8090367612044727, 0.30903676120447265, 0.9810741902197953, 0.8333333333333334, 0.3540173658568024, 0.8825276619647324, 0.6426157582403226, 0.3573842417596774, 0.11747233803526766, 0.8333333333333333, 0.30903676120447254, 0.5393578408029817, 0.1, 1.0])
        )
    elif scheme == 'Feagin12':
        # Feagin 12th order, 25 stages (ODE's Feagin12). Symmetric; no embedded
        # branch in ODE's cache, so it registers plain. Row sums == c.
        return butcherTableau(
            a = np.array([
                          [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                          [0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                          [-0.21604938271604937, 0.7716049382716049, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                          [0.20833333333333334, 0.0, 0.625, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                          [0.19333333333333333, 0.0, 0.22, -0.08, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                          [0.1, 0.0, 0.0, 0.4, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                          [0.10336447165001048, 0.0, 0.0, 0.12405309452894676, 0.4831711675610329, -0.038753024569476324, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                          [0.12403826143183333, 0.0, 0.0, 0.0, 0.21705063219795848, 0.013745579207596677, -0.06610953172676828, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                          [0.0914774894856883, 0.0, 0.0, 0.0, 0.0, -0.005443485237174697, 0.06807168016884535, 0.40839431558264105, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                          [0.08900136525025511, 0.0, 0.0, 0.0, 0.0, 0.004995282266455323, 0.397918238819829, 0.4279302107525766, -0.0865117637557827, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                          [0.06950876241349076, 0.0, 0.0, 0.0, 0.0, 0.12914694190017645, 1.530736381023113, 0.57787476112914, -0.9512947723210889, -0.40827664296563193, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                          [0.044486140329513583, 0.0, 0.0, 0.0, 0.0, -0.0038047686705696172, 0.01069550640296242, 0.020961624449990432, -0.023314602325932177, 0.0026326598106453697, 0.0031547276897702504, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                          [0.019458881511975546, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 6.785129491718125e-05, -4.297958590492736e-05, 1.7635898226028515e-05, 0.0653866627415027, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                          [0.2068368356642771, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.016679606710415646, -0.008795015632007103, 0.003466754553624639, -0.8612644601057177, 0.9086518820740502, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                          [0.0203926084654484, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0869469392016686, -0.019164963041014983, 0.006556291594936633, 0.09874761281274348, 0.005353646955249961, 0.3011678640109679, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                          [0.2284104339177781, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.4987074007930252, 0.1348411683357245, -0.03874582440558342, -1.2747325747347484, 1.4391636446287717, -0.21400746796799025, 0.9582024177544303, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                          [2.002224776559742, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.067018099615249, 0.6239781360861395, -0.046228368550031144, -8.849732883626496, 7.7425770785085595, -0.5883585192508692, -1.1068373336238064, -0.929529037579204, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                          [3.1378953341207345, 0.0, 0.0, 0.0, 0.0, 0.12914694190017645, 1.530736381023113, 0.57787476112914, 5.420882630551267, 0.2315469260348293, 0.07592929955789135, -12.372997338018651, 9.854558834647696, 0.08591114313704365, -5.652427528626439, -1.9430093524281962, -0.12835260184940453, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                          [1.3836005443219601, 0.0, 0.0, 0.0, 0.0, 0.004995282266455323, 0.397918238819829, 0.4279302107525766, -1.3029910742447577, 0.661292278669377, -0.14455977430695435, -6.965760347317982, 6.6580854323599175, -1.669973751088415, 2.064137023180353, -0.6747439626443065, -0.001156188347949395, -0.005440579086770074, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                          [0.9512362970482877, 0.0, 0.0, 0.0, 0.21705063219795848, 0.013745579207596677, -0.06610953172676828, 0.0, 0.15228169673641445, -0.33774101835759984, -0.019282598163399577, -3.682592696968668, 3.1619787040698206, -0.3704625221068853, -0.05149742003654404, -0.0008296255321201529, 2.798010414192786e-06, 0.041860391641236026, 0.27908425509087736, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                          [0.10336447165001048, 0.0, 0.0, 0.12405309452894676, 0.4831711675610329, -0.038753024569476324, 0.0, -0.43831382036112243, 0.0, -0.21863663372167666, -0.031233476439471924, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.031233476439471924, 0.21863663372167666, 0.43831382036112243, 0.0, 0.0, 0.0, 0.0, 0.0],
                          [0.19333333333333333, 0.0, 0.22, -0.08, 0.0, 0.0, 0.0984256130499316, -0.19641088922305466, 0.0, 0.43645793049306875, 0.06526137216757211, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.06526137216757211, -0.43645793049306875, 0.19641088922305466, -0.0984256130499316, 0.0, 0.0, 0.0, 0.0],
                          [-0.21604938271604937, 0.7716049382716049, 0.0, 0.0, -0.6666666666666666, 0.0, -0.39069646929597845, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.39069646929597845, 0.6666666666666666, 0.0, 0.0, 0.0],
                          [0.2, 0.0, -0.1646090534979424, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1646090534979424, 0.0, 0.0],
                          [1.4717872488111041, 0.7875, 0.4212962962962963, 0.0, 0.2916666666666667, 0.0, 0.34860071762832956, 0.22949954476899484, 5.79046485790482, 0.4185875118565069, 0.307039880222474, -4.687009053506033, 3.1357166559380225, 1.4013482971096571, -5.52931101439499, -0.8531382355080633, 0.10357578037361014, -0.14047441695060095, -0.4185875118565069, -0.22949954476899484, -0.34860071762832956, -0.2916666666666667, -0.4212962962962963, -0.7875, 0.0],
            ]),
            b = np.array([0.023809523809523808, 0.0234375, 0.03125, 0.0, 0.041666666666666664, 0.0, 0.05, 0.05, 0.0, 0.1, 0.07142857142857142, 0.0, 0.13841302368078298, 0.2158726906049313, 0.2438095238095238, 0.2158726906049313, 0.13841302368078298, -0.07142857142857142, -0.1, -0.05, -0.05, -0.041666666666666664, -0.03125, -0.0234375, 0.023809523809523808]),
            c = np.array([0.0, 0.2, 0.5555555555555556, 0.8333333333333334, 0.3333333333333333, 1.0, 0.6718357091705138, 0.2887249411106202, 0.5625, 0.8333333333333334, 0.9476954311791993, 0.05481128768638026, 0.08488805186071653, 0.26557560326464286, 0.5, 0.7344243967353572, 0.9151119481392831, 0.9476954311792003, 0.8333333333333336, 0.28872494111062014, 0.6718357091705138, 0.3333333333333333, 0.5555555555555556, 0.2, 1.0000000000000007])
        )
    elif scheme == 'Feagin14':
        # Feagin 14th order, 35 stages (ODE's Feagin14). Symmetric; no embedded
        # branch in ODE's cache, so it registers plain. Row sums == c.
        return butcherTableau(
            a = np.array([
                          [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                          [0.1111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                          [-0.8333333333333334, 1.3888888888888888, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                          [0.20833333333333334, 0.0, 0.625, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                          [0.19333333333333333, 0.0, 0.22, -0.08, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                          [0.1, 0.0, 0.0, 0.4, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                          [0.10348456163667978, 0.0, 0.0, 0.12206888730640722, 0.4825744903312466, -0.0381409600015607, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                          [0.12438052665409441, 0.0, 0.0, 0.0, 0.2261202821975843, 0.013788588761808088, -0.06722101339966845, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                          [0.09369190656596738, 0.0, 0.0, 0.0, 0.0, -0.00613406843450511, 0.21601982562550306, 0.4236950635157619, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                          [0.08384798124090527, 0.0, 0.0, 0.0, 0.0, -0.01179493671009738, -0.24729902056881264, 0.0978080858367729, 0.21759068924342062, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                          [0.061525535976942825, 0.0, 0.0, 0.0, 0.0, 0.005922327803245033, 0.47032615996384114, 0.299688863848679, -0.2476568775939949, 0.11089502977143768, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                          [0.04197000733627826, 0.0, 0.0, 0.0, 0.0, -0.003179876962662051, 0.806397714906192, 0.0975983126412389, 0.778575578158399, 0.20489042383159942, -1.5626157962746818, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                          [0.04377267822337302, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.006243650275201952, 0.20004309710957732, -0.008053283678049831, 0.021151752806739654, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                          [0.028349925036351455, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.002491632048558174, 0.023013878785459314, -0.003221559566929771, 0.009884425494476646, -0.021301077132888736, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                          [0.343511894290243, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2104519120236274, 1.034274520572304, 0.006003036458644225, 0.8559381250996195, -0.9772350050367669, -0.6600269804792946, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                          [-0.014357400167216807, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.036625327004904, 0.03502549756362137, 0.03609460163621135, -0.02652199675536811, 0.044569901130569814, 0.12434309333135825, 0.004138296932394807, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                          [0.3560324044251203, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.4501927589475626, 0.4305279070837109, 0.5119730290110223, 0.9083036388864043, -1.2392109337193393, -0.6490486616717615, 0.25170890458681927, 0.7799064703455864, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                          [0.013093568740651306, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -9.32053067985114e-05, 0.05053743342622993, 8.04470341944488e-07, 0.0005917260294941712, -4.0161472215455734e-07, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                          [0.0207926484466053, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0005826959188000859, -0.00801700732358816, 4.0384764384713694e-06, 0.08546099980555061, -2.0448648093580423e-06, 0.10532857882443189, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                          [1.4015344979573603, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.23025200098422127, -7.211068404669129, 0.0037290156069483636, -4.7141549572712504, -0.0017636765754534924, 7.641305480386988, 3.5060204365975185, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                          [11.951465069412068, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 7.794809321081759, -56.45013938673258, 0.0912376306930645, -12.73362799254349, -0.039689592190471974, 54.43921418835709, -3.6441163792156925, -0.8045032499105099, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                          [-148.80942650710048, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -91.72952782912564, 707.6561449715983, -1.1056361185748245, 176.13459188381137, 0.49138482421488067, -684.278000449815, 27.991060499839826, 13.193971003028233, 1.2512878128398044, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                          [-9.673079469481968, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -4.469901508585055, 45.51271286909527, -0.07130850861838268, 11.227361406841274, 0.12624437671762273, -43.54393395494833, 0.787174307543059, 0.5322646967446842, 0.42242273399632535, 0.08591312495030672, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                          [-10.06640324470547, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.036625327004904, 0.03502549756362137, 0.03609460163621135, -0.02652199675536811, -6.270889721814641, 48.2079237442563, -0.06944716891361656, 12.68106902048503, 0.011967116896832376, -46.72497649924824, 1.330296133266267, 1.007667875033983, 0.02095120519336651, 0.02101347063312642, 0.009521960144171218, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                          [-409.4780816777437, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2104519120236274, 1.034274520572304, 0.006003036458644225, 0.8559381250996195, -250.51699854744786, 1946.4246665238843, -3.0450388210231036, 490.6263795282817, 1.5664758953127091, -1881.9742899401117, 75.25922247248472, 34.57343569803311, 3.21147679440969, -0.4604080417384144, -0.08707183398418106, -7.393518141583031, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                          [3.433474758535509, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.002491632048558174, 0.023013878785459314, -0.003221559566929771, 0.009884425494476646, 2.162527993779225, -16.269986454645743, -0.12853450212052456, -8.989150426665043, -0.0034859536323202534, 15.793619411333982, -0.574403330914095, -0.3456020390213933, -0.006622414902065851, -0.007777881292422042, -0.0035608419240227493, 4.792825064499308, 0.15372546487306857, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                          [32.30385208719854, 0.0, 0.0, 0.0, 0.0, -0.003179876962662051, 0.806397714906192, 0.0975983126412389, 0.778575578158399, 0.20489042383159942, -1.5626157962746818, 0.0, 16.34298918823106, -154.54455529354362, 1.5697108870333487, 3.2768554508724814, -0.05034892451936532, 153.32115185804167, 7.175681863277205, -2.9403674867530047, -0.06658459460768032, -0.04623460549908437, -0.02041987335856794, -53.35231064387359, -1.3554871471507866, -1.5719627580123274, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                          [-16.64514674863415, 0.0, 0.0, 0.0, 0.0, 0.005922327803245033, 0.47032615996384114, 0.299688863848679, -0.2476568775939949, 0.11089502977143768, 0.0, -0.49171904384622916, -11.47431544272895, 80.25931665762303, -0.38413230398004283, 7.281476674681076, -0.13269938461224837, -81.07998325257307, -1.2503749283562064, 2.592635949695437, -0.30144029834640457, 0.22138446078983234, 0.08275772747718929, 18.99606620406115, 0.2692319464096397, 1.6267482744706654, 0.49171904384622916, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                          [0.08384798124090527, 0.0, 0.0, 0.0, 0.0, -0.01179493671009738, -0.24729902056881264, 0.0978080858367729, 0.21759068924342062, 0.0, 0.13758560676332524, 0.04398702297150467, 0.0, -0.5137008137681933, 0.8263556911513155, 25.701813971981185, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -25.701813971981185, -0.8263556911513155, 0.5137008137681933, -0.04398702297150467, -0.13758560676332524, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                          [0.12438052665409441, 0.0, 0.0, 0.0, 0.2261202821975843, 0.013788588761808088, -0.06722101339966845, 0.0, 0.0, -0.8562389750854283, -1.963375228668589, -0.2323328227241194, 0.0, 4.306607190864534, -2.927229632494655, -82.31316663978589, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 82.31316663978589, 2.927229632494655, -4.306607190864534, 0.2323328227241194, 1.963375228668589, 0.8562389750854283, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                          [0.10348456163667978, 0.0, 0.0, 0.12206888730640722, 0.4825744903312466, -0.0381409600015607, 0.0, -0.5504995253108024, 0.0, -0.7119158115851892, -0.5841296056715514, 0.0, 0.0, 2.1104630812586493, -0.08374947367395721, 5.1002149907232095, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -5.1002149907232095, 0.08374947367395721, -2.1104630812586493, 0.0, 0.5841296056715514, 0.7119158115851892, 0.5504995253108024, 0.0, 0.0, 0.0, 0.0, 0.0],
                          [0.19333333333333333, 0.0, 0.22, -0.08, 0.0, 0.0, 0.10999342558072471, -0.2542970480762702, 0.0, 0.8655707771166943, 3.3241644911409307, 0.0, 0.0, -12.010222331597793, 0.4766014662424932, -29.02430112210364, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 29.02430112210364, -0.4766014662424932, 12.010222331597793, 0.0, -3.3241644911409307, -0.8655707771166943, 0.2542970480762702, -0.10999342558072471, 0.0, 0.0, 0.0, 0.0],
                          [-0.8333333333333334, 1.3888888888888888, 0.0, 0.0, -0.75, 0.0, -0.4925295437180263, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.4925295437180263, 0.75, 0.0, 0.0, 0.0],
                          [0.1111111111111111, 0.0, -0.2222222222222222, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2222222222222222, 0.0, 0.0],
                          [0.28583514038897156, 0.2916666666666667, 0.21875, 0.0, 0.1640625, 0.0, 0.21819435494555667, 0.18039289847869777, 0.0, 0.20571383940484503, 0.24271579158177023, 0.2464657808136293, -3.4499194079089084, 0.22887556216003607, 0.2832905997021514, 3.2108512583776663, -0.2235387773648457, -0.707121157204419, 3.2112334515028707, 1.4095434830966977, -0.15136205344374262, 0.37235057452701426, 0.2529787464063613, -3.2108512583776663, -0.2832905997021514, -0.22887556216003607, -0.2464657808136293, -0.24271579158177023, -0.20571383940484503, -0.18039289847869777, -0.21819435494555667, -0.1640625, -0.21875, -0.2916666666666667, 0.0],
            ]),
            b = np.array([0.017857142857142856, 0.005859375, 0.01171875, 0.0, 0.017578125, 0.0, 0.0234375, 0.029296875, 0.0, 0.03515625, 0.041015625, 0.046875, 0.0, 0.052734375, 0.05859375, 0.064453125, 0.0, 0.10535211357175302, 0.17056134624175218, 0.20622939732935194, 0.20622939732935194, 0.17056134624175218, 0.10535211357175302, -0.064453125, -0.05859375, -0.052734375, -0.046875, -0.041015625, -0.03515625, -0.029296875, -0.0234375, -0.017578125, -0.01171875, -0.005859375, 0.017857142857142856]),
            c = np.array([0.0, 0.1111111111111111, 0.5555555555555555, 0.8333333333333334, 0.3333333333333333, 1.0, 0.669986979272773, 0.2970683842138183, 0.7272727272727273, 0.14015279904218877, 0.7007010397701507, 0.3636363636363637, 0.2631578947368421, 0.039217224665027084, 0.8129175029283767, 0.16666666666666666, 0.9, 0.06412992574519669, 0.20414990928342883, 0.3953503910487606, 0.604649608951239, 0.7958500907165438, 0.9358700742548045, 0.16666666666666524, 0.8129175029284074, 0.03921722466502661, 0.3636363636363684, 0.700701039770156, 0.14015279904218877, 0.2970683842138183, 0.669986979272773, 0.3333333333333333, 0.5555555555555555, 0.1111111111111111, 0.9999999999999998])
        )
    # ---- Embedded pairs -------------------------------------------------- #
    # `b` is a tuple (propagated weights, embedded lower-order weights). The first
    # entry advances the solution; the difference between the two is returned as
    # IntegrationResult.error for step size control.
    elif scheme == 'BogackiShampine':
        # Bogacki-Shampine 3(2), scipy's RK23. FSAL: c[-1] == 1 and a[-1] == b[0],
        # so the last stage of step n IS f(t^{n+1}, y^{n+1}) and can be reused as k0
        # of step n+1 at no cost in order -- 4 stages, 3 effective evaluations.
        return butcherTableau(
            a = np.array([[  0,   0,   0, 0],
                          [1/2,   0,   0, 0],
                          [  0, 3/4,   0, 0],
                          [2/9, 1/3, 4/9, 0]]),
            b = (np.array([2/9, 1/3, 4/9, 0]),
                 np.array([7/24, 1/4, 1/3, 1/8])),
            c = np.array([0, 1/2, 3/4, 1])
        )
    elif scheme == 'DormandPrince':
        # Dormand-Prince 5(4), scipy's RK45 / MATLAB's ode45. Also FSAL:
        # 7 stages, 6 effective evaluations under reuse.
        return butcherTableau(
            a = np.array([
                [          0,            0,           0,         0,            0,       0, 0],
                [       1/5,             0,           0,         0,            0,       0, 0],
                [      3/40,          9/40,           0,         0,            0,       0, 0],
                [     44/45,        -56/15,        32/9,         0,            0,       0, 0],
                [19372/6561,   -25360/2187,  64448/6561,  -212/729,            0,       0, 0],
                [ 9017/3168,       -355/33,  46732/5247,    49/176,  -5103/18656,       0, 0],
                [    35/384,             0,    500/1113,   125/192,   -2187/6784,   11/84, 0]]),
            b = (np.array([35/384, 0, 500/1113, 125/192, -2187/6784, 11/84, 0]),
                 np.array([5179/57600, 0, 7571/16695, 393/640, -92097/339200, 187/2100, 1/40])),
            c = np.array([0, 1/5, 3/10, 4/5, 8/9, 1, 1])
        )
    elif scheme == 'CashKarp':
        # Cash-Karp 5(4). Not FSAL (c[-1] = 7/8), but a well-conditioned embedded
        # pair in 6 stages when Dormand-Prince is more than is needed.
        return butcherTableau(
            a = np.array([
                [          0,        0,          0,             0,        0, 0],
                [        1/5,        0,          0,             0,        0, 0],
                [       3/40,     9/40,          0,             0,        0, 0],
                [       3/10,    -9/10,        6/5,             0,        0, 0],
                [     -11/54,      5/2,     -70/27,         35/27,        0, 0],
                [1631/55296,  175/512, 575/13824, 44275/110592, 253/4096, 0]]),
            b = (np.array([37/378, 0, 250/621, 125/594, 0, 512/1771]),
                 np.array([2825/27648, 0, 18575/48384, 13525/55296, 277/14336, 1/4])),
            c = np.array([0, 1/5, 3/10, 3/5, 1, 7/8])
        )
    # The OrdinaryDiffEq wave-6 embedded pairs (SCHEME_EXPANSION_PLAN, 2026-10-13):
    # b_embed = b_main - btilde (or - bhat for BS5), the difference convention of
    # DiffEqBase's calculate_residuals. Coefficients and the one-step
    # bit-for-bit cross-check as for the fixed-step wave-6 tableaus above.
    elif scheme == 'Tsit5':
        # Tsitouras 5(4), ODE's recommended default for non-stiff problems
        # (free 4th order interpolant). 7 stages = 6 + FSAL. Measured: main
        # order 5, embedded 4, SSP coefficient 0.0, real-axis interval
        # [-3.506, 0]. Row sums == c.
        return butcherTableau(
            a = np.array([
                          [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                          [0.161, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                          [-0.008480655492356989, 0.335480655492357, 0.0, 0.0, 0.0, 0.0, 0.0],
                          [2.8971530571054935, -6.359448489975075, 4.3622954328695815, 0.0, 0.0, 0.0, 0.0],
                          [5.325864828439257, -11.748883564062828, 7.4955393428898365, -0.09249506636175525, 0.0, 0.0, 0.0],
                          [5.86145544294642, -12.92096931784711, 8.159367898576159, -0.071584973281401, -0.028269050394068383, 0.0, 0.0],
                          [0.09646076681806523, 0.01, 0.4798896504144996, 1.379008574103742, -3.290069515436081, 2.324710524099774, 0.0],
            ]),
            b = (np.array([0.09646076681806523, 0.01, 0.4798896504144996, 1.379008574103742, -3.290069515436081, 2.324710524099774, 0.0]),
                 np.array([0.09824077787029101, 0.010816434459656746, 0.4720087724042376, 1.5237195812770048, -3.872426680888636, 2.7827926300289607, -0.015151515151515152])),
            c = np.array([0.0, 0.161, 0.327, 0.9, 0.9800255409045097, 1.0, 1.0])
        )
    elif scheme == 'OwrenZen3':
        # Owren-Zennaro (1992) optimized-interpolation 3(2) (free 3rd order
        # interpolant), 4 stages = 3 + FSAL. Measured: main order 3, embedded
        # 2, SSP coefficient 0.7813, real-axis interval [-2.512, 0].
        # Row sums == c.
        return butcherTableau(
            a = np.array([
                          [0.0, 0.0, 0.0, 0.0],
                          [0.5217391304347826, 0.0, 0.0, 0.0],
                          [-0.18133333333333335, 0.9813333333333333, 0.0, 0.0],
                          [0.2152777777777778, 0.4592013888888889, 0.3255208333333333, 0.0],
            ]),
            b = (np.array([0.2152777777777778, 0.4592013888888889, 0.3255208333333333, 0.0]),
                 np.array([0.3888888888888889, -0.039930555555555525, 0.6510416666666666, 0.0])),
            c = np.array([0.0, 0.5217391304347826, 0.8, 1.0])
        )
    elif scheme == 'OwrenZen4':
        # Owren-Zennaro (1992) optimized-interpolation 4(3) (free 4th order
        # interpolant), 6 stages = 5 + FSAL. Measured: main order 4, embedded
        # 3, SSP coefficient 0.6444, real-axis interval [-2.872, 0].
        # Row sums == c.
        return butcherTableau(
            a = np.array([
                          [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                          [0.16666666666666666, 0.0, 0.0, 0.0, 0.0, 0.0],
                          [0.03214024835646457, 0.26515704894083275, 0.0, 0.0, 0.0, 0.0],
                          [0.6895990230002036, -1.6993690209647874, 1.6568288214939955, 0.0, 0.0, 0.0],
                          [-0.09002509947964493, 0.6817777777777778, -0.2402791551882461, 0.5151931435567799, 0.0, 0.0],
                          [0.08990252172070354, 0.0, 0.4360623278236915, 0.1842858372687918, 0.2897493131868132, 0.0],
            ]),
            b = (np.array([0.08990252172070354, 0.0, 0.4360623278236915, 0.1842858372687918, 0.2897493131868132, 0.0]),
                 np.array([-0.0984318711591439, 0.0, 0.9664084022038568, -0.4474751574183393, 0.5794986263736264, 0.0])),
            c = np.array([0.0, 0.16666666666666666, 0.2972972972972973, 0.6470588235294118, 0.8666666666666667, 1.0])
        )
    elif scheme == 'OwrenZen5':
        # Owren-Zennaro (1992) optimized-interpolation 5(4) (free 5th order
        # interpolant), 8 stages = 7 + FSAL. ODE's perform_step applies a
        # literal k3 to stage 4 (a43 = 1) that its cache does not store; the
        # tableau includes it. Measured: main order 5, embedded 4, SSP
        # coefficient 0.0, real-axis interval [-3.192, 0]. Row sums == c.
        return butcherTableau(
            a = np.array([
                          [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                          [0.16666666666666666, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                          [0.0625, 0.1875, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                          [0.25, -0.75, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                          [-0.75, 3.75, -3.0, 0.5, 0.0, 0.0, 0.0, 0.0],
                          [0.26895043731778423, -0.7084548104956269, 0.8658892128279884, 0.15462307371928363, 0.06184922948771345, 0.0, 0.0, 0.0],
                          [-0.02947695035460993, 0.18500664893617022, 0.4802345261121857, -0.5337849069148937, -0.013090093085106383, 0.7861107753062541, 0.0, 0.0],
                          [0.08783068783068783, 0.0, 0.3006060606060606, 0.22777777777777777, 0.027777777777777776, 0.06218596218596219, 0.2938217338217338, 0.0],
            ]),
            b = (np.array([0.08783068783068783, 0.0, 0.3006060606060606, 0.22777777777777777, 0.027777777777777776, 0.06218596218596219, 0.2938217338217338, 0.0]),
                 np.array([0.2867724867724868, 0.0, -0.6109090909090908, 2.2055555555555557, 0.1388888888888889, -1.607951307951308, 0.5876434676434676, 0.0])),
            c = np.array([0.0, 0.16666666666666666, 0.25, 0.5, 0.5, 0.6428571428571429, 0.875, 1.0])
        )
    elif scheme == 'BS5':
        # Bogacki-Shampine (1996) 5(4), "an efficient Runge-Kutta (4,5) pair",
        # 8 stages = 7 + FSAL. The embedded branch is ODE's bhat (the 4th order
        # companion); ODE's btilde is a separate 5th order estimate for the
        # step controller, not a solution. Measured: main order 5, embedded
        # 4, SSP coefficient 0.5237, real-axis interval [-3.986, 0].
        # Row sums == c.
        return butcherTableau(
            a = np.array([
                          [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                          [0.16666666666666666, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                          [0.07407407407407407, 0.14814814814814814, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                          [0.13338192419825073, -0.47230320699708456, 0.7674927113702624, 0.0, 0.0, 0.0, 0.0, 0.0],
                          [0.22895622895622897, -0.36363636363636365, 0.2937062937062937, 0.5076405076405076, 0.0, 0.0, 0.0, 0.0],
                          [0.026500355113636364, 0.23011363636363635, 0.10772747760052448, 0.1602190777972028, 0.225439453125, 0.0, 0.0, 0.0],
                          [0.18159821692916506, -0.38707982536247293, 0.41288268560074054, 0.6409913191253497, -1.0121439383514517, 1.1637515420586695, 0.0, 0.0],
                          [0.07279265873015874, 0.0, 0.28662437773692473, 0.19513621794871794, 0.008638392857142857, 0.3595655806182122, 0.07724277210884353, 0.0],
            ]),
            b = (np.array([0.07279265873015874, 0.0, 0.28662437773692473, 0.19513621794871794, 0.008638392857142857, 0.3595655806182122, 0.07724277210884353, 0.0]),
                 np.array([0.07513640873015874, 0.0, 0.27624830233206643, 0.21162660256410254, -0.010345982142857145, 0.37009189640768586, 0.07724277210884353, 0.0])),
            c = np.array([0.0, 0.16666666666666666, 0.2222222222222222, 0.42857142857142855, 0.6666666666666666, 0.75, 1.0, 1.0])
        )
    elif scheme == 'Stepanov5':
        # Stepanov (Stepanov2021Embedded5) 5(4), 7 stages = 6 + FSAL.
        # Measured: main order 5, embedded 4, SSP coefficient 0.0, real-axis
        # interval [-3.496, 0]. Row sums == c.
        return butcherTableau(
            a = np.array([
                          [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                          [0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                          [0.0621301775147929, 0.26094674556213016, 0.0, 0.0, 0.0, 0.0, 0.0],
                          [1.6301020408163265, -5.207142857142857, 4.4770408163265305, 0.0, 0.0, 0.0, 0.0],
                          [2.8811124102418746, -9.263950892857142, 7.442850056689342, -0.08501157407407407, 0.0, 0.0, 0.0],
                          [3.2522024583580915, -10.435540069686411, 8.27662100358286, -0.05432098765432099, -0.03896240460022027, 0.0, 0.0],
                          [0.09774793108126441, 0.0, 0.48629574293291106, 1.4567901234567902, -2.904470161107329, 1.8636363636363635, 0.0],
            ]),
            b = (np.array([0.09774793108126441, 0.0, 0.48629574293291106, 1.4567901234567902, -2.904470161107329, 1.8636363636363635, 0.0]),
                 np.array([0.10141093474426807, 0.0, 0.4736707572990759, 1.7234567901234568, -3.775811209439528, 2.5037752314463497, -0.026502504173622706])),
            c = np.array([0.0, 0.2, 0.3230769230769231, 0.9, 0.975, 1.0, 1.0])
        )
    elif scheme == 'Alshina2':
        # Alshina et al. (2008) "2nd order, 2-stage method with optimal
        # parameters", 2(1). The embedded branch is exactly forward Euler
        # (b_main - b_embed = (-3/4, 3/4) = ODE's btilde). Measured: main
        # order 2, embedded 1, SSP coefficient 1.0, real-axis interval
        # [-2.0, 0]. Not FSAL.
        return butcherTableau(
            a = np.array([
                          [0.0, 0.0],
                          [0.6666666666666666, 0.0],
            ]),
            b = (np.array([0.25, 0.75]),
                 np.array([1.0, 0.0])),
            c = np.array([0.0, 0.666666666666666])
        )
    # The OrdinaryDiffEq wave-7 embedded pairs (SCHEME_EXPANSION_PLAN,
    # 2026-10-13): TanYam7 7(6), TsitPap8 8(7), Vern6/7/8/9, RKV76IIa 7(6).
    # b_embed = b_main - btilde (or ODE's bh for RKV76IIa). Coefficients and
    # the one-step bit-for-bit cross-check as for the wave-7 plain tableaus
    # above. Row sums == c.
    elif scheme == 'TanYam7':
        # Tsitouras-Tan (1998) 7th order, 10 stages (ODE's HighOrderRK TanYam7).
        # b_embed = b_main - btilde. Row sums == c.
        return butcherTableau(
            a = np.array([
                          [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                          [0.07816646510113846, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                          [0.029312424412926925, 0.08793727323878078, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                          [0.04396863661939039, 0.0, 0.13190590985817116, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                          [0.7361834837738382, 0.0, -2.8337999624233303, 2.5963565888408913, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                          [-12.062819391370866, 0.0, 48.20838100175243, -38.05863046463434, 2.6851905444372632, 0.0, 0.0, 0.0, 0.0, 0.0],
                          [105.21957276320198, 0.0, -417.92888626241256, 332.3155504499333, -19.827591183572938, 1.2125399024549859, 0.0, 0.0, 0.0, 0.0],
                          [114.67755718631742, 0.0, -455.5612169896097, 362.24095553923144, -21.67190442182809, 1.3189132007137807, -0.0048025566150346555, 0.0, 0.0, 0.0],
                          [115.21334870553768, 0.0, -457.69356568613233, 363.93688218862735, -21.776682078900294, 1.3250670887878468, -0.004518190986768983, -0.0005320269334859959, 0.0, 0.0],
                          [115.18928245800194, 0.0, -457.598022271643, 363.8610256312148, -21.77212754027556, 1.3248804645074317, -0.0045057252106918315, -0.0005330165949429136, 0.0, 0.0],
            ]),
            b = (np.array([0.05126014249744686, 0.0, 0.0, 0.27521638456212627, 0.33696650340710543, 0.18986072244906577, 8.461099418514403, -130.15941672640542, 121.84501355497527, 0.0]),
                 np.array([0.05151786740451655, 0.0, 0.0, 0.2742934740775871, 0.34014440474452484, 0.1777561335073176, 11.167123377986993, -174.7008623676715, 243.69002710995053, -80.0])),
            c = np.array([0.0, 0.07816646510113846, 0.1172496976517077, 0.17587454647756157, 0.4987401101913992, 0.7721216901844854, 0.9911856696047794, 0.9995019582098041, 1.0000000000000016, 0.999999999999948])
        )
    elif scheme == 'TsitPap8':
        # Tsitouras-Papakonstantinou 8th order, 13 stages (ODE's HighOrderRK TsitPap8).
        # Stage 8 sits at c = 1.0071 (just past the step end, as ODE evaluates it).
        # b_embed = b_main - btilde. Row sums == c.
        return butcherTableau(
            a = np.array([
                          [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                          [0.06338028169014084, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                          [0.019438980427336498, 0.08334896544902781, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                          [0.038545479703636615, 0.0, 0.11563643911090984, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                          [0.39436557770112496, 0.0, -1.4818719321673373, 1.4751032536910185, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                          [0.045994489107698204, 0.0, 0.0, 0.23235070626395474, 0.18740822928588133, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                          [0.06005228953244051, 0.0, 0.0, 0.11220383194636775, -0.03357232951906142, 0.016721613445658576, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                          [-1.5733292732086857, 0.0, 0.0, -1.3167087730223663, -11.723515296181773, 9.107825028173872, 6.512820512820513, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                          [-0.48107625624391254, 0.0, 0.0, -6.6506103607463904, -4.530206099782572, 3.894414525020157, 8.634217645525526, 0.009401624788681498, 0.0, 0.0, 0.0, 0.0, 0.0],
                          [-0.7754121446230569, 0.0, 0.0, -7.996604718235832, -6.726558607230182, 5.532184454327406, 10.89757332024991, 0.020091650280045396, -0.039186042680376856, 0.0, 0.0, 0.0, 0.0],
                          [-1.1896363245449992, 0.0, 0.0, -7.128368483301214, -9.53722789710108, 7.574470108980868, 11.267486382070919, 0.051009801223058315, 0.08019413469508256, -0.15819617839847347, 0.0, 0.0, 0.0],
                          [-0.39200039047127266, 0.0, 0.0, 3.916659042493856, -2.8017459289080557, 2.441204566481742, -2.4183655778824718, -0.33943326290032927, 0.19496450383103364, -0.19437176762508154, 0.5930888149805791, 0.0, 0.0],
                          [-1.4847063081291894, 0.0, 0.0, -2.390723588981498, -11.184306772840532, 8.720804667459817, 7.33673830753461, 0.01289874999394761, 0.042583289842657704, -0.05328834487981156, 0.0, 0.0, 0.0],
            ]),
            b = (np.array([0.04441161093250152, 0.0, 0.0, 0.0, 0.0, 0.35395063113733116, 0.2485219684184965, -0.3326913171720666, 1.921248828652836, -2.7317783000882523, 1.4012004409899175, 0.0951361371292365, 0.0]),
                 np.array([0.044484201850329544, 0.0, 0.0, 0.0, 0.0, 0.35502352274458154, 0.24825530158391707, -2.4242252962684616, 1.5999301534099692, -1.8107646286929682, 0.0, 0.0, 2.9872967453726327])),
            c = np.array([0.0, 0.06338028169014084, 0.1027879458763643, 0.15418191881454646, 0.38759689922480617, 0.4657534246575343, 0.15540540540540543, 1.0070921985815597, 0.8761410785614893, 0.9120879120879126, 0.9597315436241611, 0.9999999999999998, 1.0000000000000002])
        )
    elif scheme == 'Vern6':
        # Verner 6th order, 9 stages (ODE's Vern6). FSAL: no separate b in ODE's dump,
        # the propagated weights ARE the last row. b_embed = b_main - btilde. Row sums == c.
        return butcherTableau(
            a = np.array([
                          [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                          [0.06, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                          [0.019239962962962962, 0.07669337037037037, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                          [0.035975, 0.0, 0.107925, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                          [1.3186834152331484, 0.0, -5.042058063628562, 4.220674648395414, 0.0, 0.0, 0.0, 0.0, 0.0],
                          [-41.87259166432751, 0.0, 159.43256216313748, -122.11921356501004, 5.531743066200053, 0.0, 0.0, 0.0, 0.0],
                          [-54.430156935316504, 0.0, 207.06725136501848, -158.61081378459, 6.991816585950242, -0.01859723106220323, 0.0, 0.0, 0.0],
                          [-54.66374178728198, 0.0, 207.95280625538936, -159.2889574744995, 7.018743740796944, -0.018338785905045722, -0.0005119484997882099, 0.0, 0.0],
                          [0.03438957868357036, 0.0, 0.0, 0.25826245556335037, 0.4209371189673537, 4.40539646966931, -176.48311902429865, 172.36413340141507, 0.0],
            ]),
            b = (np.array([0.03438957868357036, 0.0, 0.0, 0.25826245556335037, 0.4209371189673537, 4.40539646966931, -176.48311902429865, 172.36413340141507, 0.0]),
                 np.array([0.025766174401369503, 0.0, 0.0, 0.27769648551650306, 0.3924870463793157, 6.515107530734602, -279.94166192426263, 273.7639380205642, -0.03333333333333333])),
            c = np.array([0.0, 0.06, 0.09593333333333333, 0.1439, 0.4973000000000001, 0.9724999999999779, 0.9995000000000204, 0.9999999999999826, 1.0000000000000138])
        )
    elif scheme == 'Vern7':
        # Verner 7th order, 10 stages (ODE's Vern7). b_embed = b_main - btilde. Row sums == c.
        return butcherTableau(
            a = np.array([
                          [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                          [0.005, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                          [-1.07679012345679, 1.185679012345679, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                          [0.04083333333333333, 0.0, 0.1225, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                          [0.6389139236255726, 0.0, -2.455672638223657, 2.272258714598084, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                          [-2.6615773750187572, 0.0, 10.804513886456137, -8.3539146573962, 0.820487594956657, 0.0, 0.0, 0.0, 0.0, 0.0],
                          [6.067741434696772, 0.0, -24.711273635911088, 20.427517930788895, -1.9061579788166472, 1.006172249242068, 0.0, 0.0, 0.0, 0.0],
                          [12.054670076253203, 0.0, -49.75478495046899, 41.142888638604674, -4.461760149974004, 2.042334822239175, -0.09834843665406107, 0.0, 0.0, 0.0],
                          [10.138146522881808, 0.0, -42.6411360317175, 35.76384003992257, -4.3480228403929075, 2.0098622683770357, 0.3487490460338272, -0.27143900510483127, 0.0, 0.0],
                          [-45.030072034298676, 0.0, 187.3272437654589, -154.02882369350186, 18.56465306347536, -7.141809679295079, 1.3088085781613787, 0.0, 0.0, 0.0],
            ]),
            b = (np.array([0.04715561848627222, 0.0, 0.0, 0.25750564298434153, 0.26216653977412624, 0.15216092656738558, 0.4939969170032485, -0.29430311714032503, 0.08131747232495111, 0.0]),
                 np.array([0.044608606606341174, 0.0, 0.0, 0.26716403785713727, 0.22010183001772932, 0.2188431703143157, 0.2289871705411204, 0.0, 0.0, 0.02029518466335628])),
            c = np.array([0.0, 0.005, 0.10888888888888903, 0.16333333333333333, 0.4554999999999999, 0.6095094489978374, 0.8839999999999997, 0.9249999999999989, 1.0000000000000013, 1.0000000000000102])
        )
    elif scheme == 'Vern8':
        # Verner 8th order, 13 stages (ODE's Vern8). b_embed = b_main - btilde. Row sums == c.
        return butcherTableau(
            a = np.array([
                          [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                          [0.05, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                          [-0.0069931640625, 0.1135556640625, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                          [0.0399609375, 0.0, 0.1198828125, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                          [0.36139756280045754, 0.0, -1.3415240667004928, 1.3701265039000352, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                          [0.049047202797202795, 0.0, 0.0, 0.23509720422144048, 0.18085559298135673, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                          [0.06169289044289044, 0.0, 0.0, 0.11236568314640277, -0.03885046071451367, 0.01979188712522046, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                          [-1.767630240222327, 0.0, 0.0, -62.5, -6.061889377376669, 5.6508231982227635, 65.62169641937624, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                          [-1.1809450665549708, 0.0, 0.0, -41.50473441114321, -4.434438319103725, 4.260408188586133, 43.75364022446172, 0.00787142548991231, 0.0, 0.0, 0.0, 0.0, 0.0],
                          [-1.2814059994414884, 0.0, 0.0, -45.047139960139866, -4.731362069449576, 4.514967016593808, 47.44909557172985, 0.01059228297111661, -0.0057468422638446166, 0.0, 0.0, 0.0, 0.0],
                          [-1.7244701342624853, 0.0, 0.0, -60.92349008483054, -5.951518376222392, 5.556523730698456, 63.98301198033305, 0.014642028250414961, 0.06460408772358203, -0.0793032316900888, 0.0, 0.0, 0.0],
                          [-3.301622667747079, 0.0, 0.0, -118.01127235975251, -10.141422388456112, 9.139311332232058, 123.37594282840426, 4.62324437887458, -3.3832777380682018, 4.527592100324618, -5.828495485811623, 0.0, 0.0],
                          [-3.039515033766309, 0.0, 0.0, -109.26086808941763, -9.290642497400293, 8.43050498176491, 114.20100103783314, -0.9637271342145479, -5.0348840888021895, 5.958130824002923, 0.0, 0.0, 0.0],
            ]),
            b = (np.array([0.04427989419007951, 0.0, 0.0, 0.0, 0.0, 0.3541049391724449, 0.24796921549564377, -15.694202038838085, 25.084064965558564, -31.738367786260277, 22.938283273988784, -0.2361324633071542, 0.0]),
                 np.array([0.04431261522908979, 0.0, 0.0, 0.0, 0.0, 0.35460956423432266, 0.24784804313666528, 4.448134732475783, 19.846886366118735, -23.58162337746562, 0.0, 0.0, -0.36016794372897754])),
            c = np.array([0.0, 0.05, 0.1065625, 0.15984375, 0.3899999999999999, 0.465, 0.155, 0.9430000000000023, 0.9018020417358582, 0.9089999999999987, 0.94, 0.9999999999999929, 1.0000000000000036])
        )
    elif scheme == 'Vern9':
        # Verner 9th order, 16 stages (ODE's Vern9). b_embed = b_main - btilde. Row sums == c.
        return butcherTableau(
            a = np.array([
                          [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                          [0.03462, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                          [-0.03893354388572875, 0.13595789452450918, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                          [0.03638413148954267, 0.0, 0.10915239446862801, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                          [2.0257639143939694, 0.0, -7.638023836496291, 6.173259922102322, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                          [0.05112275589406061, 0.0, 0.0, 0.17708237945550218, 0.0008027762409222536, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                          [0.13160063579752163, 0.0, 0.0, -0.2957276252669636, 0.08781378035642955, 0.6213052975225274, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                          [0.07166666666666667, 0.0, 0.0, 0.0, 0.0, 0.33055335789153195, 0.2427799754418014, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                          [0.071806640625, 0.0, 0.0, 0.0, 0.0, 0.3294380283228177, 0.1165190029271823, -0.034013671875, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                          [0.04836757646340646, 0.0, 0.0, 0.0, 0.0, 0.03928989925676164, 0.10547409458903446, -0.021438652846483126, -0.10412291746271944, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                          [-0.026645614872014785, 0.0, 0.0, 0.0, 0.0, 0.03333333333333333, -0.1631072244872467, 0.03396081684127761, 0.1572319413814626, 0.21522674780318796, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                          [0.03689009248708622, 0.0, 0.0, 0.0, 0.0, -0.1465181576725543, 0.2242577768172024, 0.02294405717066073, -0.0035850052905728597, 0.08669223316444385, 0.43838406519683376, 0.0, 0.0, 0.0, 0.0, 0.0],
                          [-0.4866012215113341, 0.0, 0.0, 0.0, 0.0, -6.304602650282853, -0.2812456182894729, -2.679019236219849, 0.5188156639241577, 1.3653531876033418, 5.8850910885039465, 2.8028087862720628, 0.0, 0.0, 0.0, 0.0],
                          [0.4185367457753472, 0.0, 0.0, 0.0, 0.0, 6.724547581906459, -0.42544428016461133, 3.3432791530012653, 0.6170816631175374, -0.9299661239399329, -6.099948804751011, -3.002206187889399, 0.2553202529443446, 0.0, 0.0, 0.0],
                          [-0.7793740861228848, 0.0, 0.0, 0.0, 0.0, -13.937342538107776, 1.2520488533793563, -14.691500408016868, -0.494705058533141, 2.2429749091462368, 13.367893803828643, 14.396650486650687, -0.79758133317768, 0.4409353709534278, 0.0, 0.0],
                          [2.0580513374668867, 0.0, 0.0, 0.0, 0.0, 22.357937727968032, 0.9094981099755646, 35.89110098240264, -3.442515027624454, -4.865481358036369, -18.909803813543427, -34.26354448030452, 1.2647565216956427, 0.0, 0.0, 0.0],
            ]),
            b = (np.array([0.014611976858423152, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.3915211862331339, 0.23109325002895065, 0.12747667699928525, 0.2246434176204158, 0.5684352689748513, 0.058258715572158275, 0.13643174034822156, 0.030570139830827976, 0.0]),
                 np.array([0.01996996514886773, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.19149930494933, 0.08857071848208439, 0.11405602348659657, 0.2533163805345107, -2.056564386240941, 0.340809679901312, 0.0, 0.0, 0.04834231373823958])),
            c = np.array([0.0, 0.03462, 0.09702435063878043, 0.14553652595817068, 0.5610000000000004, 0.22900791159048503, 0.544992088409515, 0.645, 0.48375, 0.06757, 0.25, 0.6590650618730998, 0.8205999999999994, 0.9011999999999996, 1.0000000000000013, 0.9999999999999996])
        )
    elif scheme == 'RKV76IIa':
        # Rushkamp (RKV76IIa) 7th order, 10 stages (ODE's Verner RKV76IIa). The embedded
        # branch is ODE's bh (the 6th-order solution, sums to 1) directly. Row sums == c.
        return butcherTableau(
            a = np.array([
                          [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                          [0.069, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                          [0.017101449275362317, 0.10089855072463769, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                          [0.04425, 0.0, 0.13275, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                          [0.7353445130709566, 0.0, -2.8301606578569376, 2.595816144785981, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                          [-12.21580485360408, 0.0, 48.82665485823736, -38.556155923199285, 2.7190858300965357, 0.0, 0.0, 0.0, 0.0, 0.0],
                          [108.86141887041765, 0.0, -432.4521181775778, 343.9115281800118, -20.550411359252738, 1.2235824864010403, 0.0, 0.0, 0.0, 0.0],
                          [113.47551318837385, 0.0, -450.8122021555997, 358.51327651900897, -21.450466676484456, 1.2740533186059528, -0.002174193904638423, 0.0, 0.0, 0.0],
                          [115.69962233242326, 0.0, -459.6635446100248, 365.5534717131746, -21.885115863497848, 1.2987181096987215, -5.318700918481884e-05, -0.0030984947647318645, 0.0, 0.0],
                          [124.15439356124647, 0.0, -493.23187133145973, 392.20862193158007, -23.486415642908533, 1.3623229489089075, -0.007051467367205771, 0.0, 0.0, 0.0],
            ]),
            b = (np.array([0.05163520172057869, 0.0, 0.0, 0.2767172535461649, 0.3374175285287151, 0.18844882678109678, 24.54134121634868, -68.81190284469012, 44.416342817764885, 0.0]),
                 np.array([0.050896765836929474, 0.0, 0.0, 0.2793777374763234, 0.3281330142746535, 0.2241721218186151, 0.7874574778015077, 0.0, 0.0, -0.6700371172080292])),
            c = np.array([0.0, 0.069, 0.11800000000000001, 0.177, 0.5010000000000001, 0.7737799115305326, 0.9939999999999798, 0.9979999999999585, 0.9999999999999901, 0.9999999999999742])
        )
    else:
        raise ValueError(f"Unknown scheme {scheme}")
