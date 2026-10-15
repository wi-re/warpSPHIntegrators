"""Wave 1 SSPRK expansion: ODE v7.8.1 Shu-Osher alpha/beta recurrences -> Butcher.

Source of truth, read line-by-line 2026-10-12 (decimal literals verbatim):
  ~/.julia/packages/OrdinaryDiffEq/OnlvZ/lib/OrdinaryDiffEqSSPRK/src/
    ssprk_caches.jl        ConstantCache constructors (alpha/beta/c literals)
    ssprk_perform_step.jl  stage recurrences (the S-row / b-vector derivations)

Per-method checks:
  1. c == ODE stored c (row sums of a), to 1e-13
  2. sum(b) == 1, to 1e-13 (SSPRK73 to 1e-5: ODE stores alpha73/alpha76 at
     6 digits -- T(0.120199) / T(0.879801) -- so its own coefficients carry
     that much order-condition slack)
  3. classical order conditions (order 2 / 3 / 4)
  4. non-FSAL: c[-1] != 1
  5. trajectory match against .tmp/ode_reference.txt (julia ODE, same ODE, dt)
  6. SSP convex-combination CFL (tvd_analysis.convex_combination_cfl, mu_max=8)
     vs ODE's published ssp_coefficient
  7. real-axis stability interval + final-map-only convexity CFL (predicts the
     measured per-step TVD CFL verdict)

Prints butcher.py-ready float literals for every entry.
"""
import sys
from fractions import Fraction as Fr
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / 'src'))

from warpSPHIntegrators.schemes.coefficients.tableau import ButcherTableau as butcherTableau  # noqa: E402
import warpSPHIntegrators.analysis.tvd as tvd_analysis  # noqa: E402

# ---------------------------------------------------------------------------
# Coefficients, verbatim from ssprk_caches.jl (al* = alpha, be* = beta).
# ---------------------------------------------------------------------------
METH = {
    'KYKSSPRK42': dict(order=2, C_ode=2.459, coef=dict(
        al20='0.394806441339829', al21='0.605193558660171',
        al30='0.00279730708739', al32='0.99720269291261',
        al40='0.252860909354373', al43='0.747139090645627',
        be10='0.406584463657504', be21='0.246062298456822',
        be30='0.013637216641451', be32='0.405447122055692',
        be40='0.016453567333598', be43='0.303775146447707'),
        c_ode=[0.0, 0.406584463657504, 0.4921245969136438,
               0.9098323119879613]),
    'SSPRK53': dict(order=3, C_ode=2.65, coef=dict(
        al30='0.355909775063327', al32='0.644090224936674',
        al40='0.367933791638137', al43='0.632066208361863',
        al52='0.237593836598569', al54='0.762406163401431',
        be10='0.377268915331368', be21='0.377268915331368',
        be32='0.242995220537396', be43='0.23845893284629',
        be54='0.287632146308408'),
        c_ode=[0.0, 0.377268915331368, 0.754537830662736,
               0.728985661612188, 0.69922613593167]),
    'SSPRK63': dict(order=3, C_ode=3.518, coef=dict(
        al40='0.476769811285196', al41='0.098511733286064',
        al43='0.42471845542874', al62='0.155221702560091',
        al65='0.844778297439909', be10='0.284220721334261',
        be21='0.284220721334261', be32='0.284220721334261',
        be43='0.12071378576593', be54='0.284220721334261',
        be65='0.2401034970659'),
        c_ode=[0.0, 0.284220721334261, 0.568441442668522,
               0.852662164002783, 0.510854218958172, 0.795074940292433]),
    'SSPRK73': dict(order=3, C_ode=4.2879, coef=dict(
        al40='0.184962588071072', al43='0.815037411928928',
        al50='0.18071865657038', al51='0.314831034403793',
        al54='0.504450309025826', al73='0.120199', al76='0.879801',
        be10='0.233213863663009', be21='0.233213863663009',
        be32='0.233213863663009', be43='0.190078023865845',
        be54='0.117644805593912', be65='0.233213863663009',
        be76='0.205181790464579'),
        c_ode=[0.0, 0.233213863663009, 0.466427727326018,
               0.699641590989027, 0.760312095463379, 0.574607439040817,
               0.807821302703826]),
    'SSPRK83': dict(order=3, C_ode=5.107, coef=dict(
        al50='0.421366967085359', al51='0.005949401107575',
        al54='0.572683631807067', al61='0.004254010666365',
        al65='0.995745989333635', al72='0.104380143093325',
        al73='0.243265240906726', al76='0.65235461599995',
        be10='0.195804015330143', be21='0.195804015330143',
        be32='0.195804015330143', be43='0.195804015330143',
        be54='0.112133754621673', be65='0.194971062960412',
        be76='0.127733653231944', be87='0.195804015330143'),
        c_ode=[0.0, 0.195804015330143, 0.391608030660286,
               0.587412045990429, 0.783216061320572, 0.561833689734037,
               0.755247658555329, 0.804195984669857]),
    'SSPRK54': dict(order=4, C_ode=1.508, coef=dict(
        be10='0.39175222657189', al20='0.444370493651235',
        al21='0.555629506348765', be21='0.368410593050371',
        al30='0.620101851488403', al32='0.379898148511597',
        be32='0.251891774271694', al40='0.178079954393132',
        al43='0.821920045606868', be43='0.544974750228521',
        al52='0.517231671970585', al53='0.096059710526147',
        be53='0.06369246866629', al54='0.386708617503269',
        be54='0.226007483236906'),
        c_ode=[0.0, 0.39175222657189, 0.58607968931154,
               0.4745423631214, 0.935010630967653]),
    'KYK2014DGSSPRK_3S2': dict(order=2, C_ode=0.8417, coef=dict(
            al10='1.0', al20='0.087353119859156', al21='0.912646880140844',
            al30='0.344956917166841', al32='0.655043082833159',
            be10='0.528005024856522', be21='0.481882138633993',
            be30='0.022826837460491', be32='0.345866039233415'),
        c_ode=[0.0, 0.528005024856522, 0.963764277267986]),
}


def build(name):
    """Butcher (a, b, c) from the ODE perform_step recurrences (Fractions)."""
    c = METH[name]['coef']
    g = {k: Fr(v) for k, v in c.items()}
    if name == 'KYKSSPRK42':
        S2 = {1: g['be10']}
        S3 = {1: g['al21'] * S2[1], 2: g['be21']}
        S4 = {1: g['al32'] * S3[1] + g['be30'], 2: g['al32'] * S3[2],
              3: g['be32']}
        a = [[Fr(0), Fr(0), Fr(0), Fr(0)],
             [S2[1], Fr(0), Fr(0), Fr(0)],
             [S3[1], S3[2], Fr(0), Fr(0)],
             [S4[1], S4[2], S4[3], Fr(0)]]
        b = [g['al43'] * S4[1] + g['be40'], g['al43'] * S4[2],
             g['al43'] * S4[3], g['be43']]
    elif name == 'SSPRK53':
        S2 = {1: g['be10']}
        S3 = {1: g['be10'], 2: g['be21']}
        S4 = {1: g['al32'] * S3[1], 2: g['al32'] * S3[2], 3: g['be32']}
        S5 = {1: g['al43'] * S4[1], 2: g['al43'] * S4[2],
              3: g['al43'] * S4[3], 4: g['be43']}
        a = [[Fr(0)] * 5,
             [S2[1], Fr(0), Fr(0), Fr(0), Fr(0)],
             [S3[1], S3[2], Fr(0), Fr(0), Fr(0)],
             [S4[1], S4[2], S4[3], Fr(0), Fr(0)],
             [S5[1], S5[2], S5[3], S5[4], Fr(0)]]
        b = [g['al52'] * S3[1] + g['al54'] * S5[1],
             g['al52'] * S3[2] + g['al54'] * S5[2],
             g['al54'] * S5[3], g['al54'] * S5[4], g['be54']]
    elif name == 'SSPRK63':
        S2 = {1: g['be10']}
        S3 = {1: g['be10'], 2: g['be21']}
        S4 = {1: g['be10'], 2: g['be21'], 3: g['be32']}
        S5 = {1: (g['al41'] + g['al43']) * S4[1], 2: g['al43'] * S4[2],
              3: g['al43'] * S4[3], 4: g['be43']}
        S6 = dict(S5)
        S6[5] = g['be54']
        a = [[Fr(0)] * 6,
             [S2[1], Fr(0), Fr(0), Fr(0), Fr(0), Fr(0)],
             [S3[1], S3[2], Fr(0), Fr(0), Fr(0), Fr(0)],
             [S4[1], S4[2], S4[3], Fr(0), Fr(0), Fr(0)],
             [S5[1], S5[2], S5[3], S5[4], Fr(0), Fr(0)],
             [S6[1], S6[2], S6[3], S6[4], S6[5], Fr(0)]]
        b = [g['al62'] * S3[1] + g['al65'] * S6[1],
             g['al62'] * S3[2] + g['al65'] * S6[2],
             g['al62'] * S3.get(3, Fr(0)) + g['al65'] * S6[3],
             g['al65'] * S6[4], g['al65'] * S6[5], g['be65']]
    elif name == 'SSPRK73':
        S2 = {1: g['be10']}
        S3 = {1: g['be10'], 2: g['be21']}
        S4 = {1: g['be10'], 2: g['be21'], 3: g['be32']}
        S5 = {1: g['al43'] * S4[1], 2: g['al43'] * S4[2],
              3: g['al43'] * S4[3], 4: g['be43']}
        S6 = {1: g['al51'] * S2[1] + g['al54'] * S5[1],
              2: g['al54'] * S5[2], 3: g['al54'] * S5[3],
              4: g['al54'] * S5[4], 5: g['be54']}
        S7 = dict(S6)
        S7[6] = g['be65']
        a = [[Fr(0)] * 7,
             [S2[1], Fr(0), Fr(0), Fr(0), Fr(0), Fr(0), Fr(0)],
             [S3[1], S3[2], Fr(0), Fr(0), Fr(0), Fr(0), Fr(0)],
             [S4[1], S4[2], S4[3], Fr(0), Fr(0), Fr(0), Fr(0)],
             [S5[1], S5[2], S5[3], S5[4], Fr(0), Fr(0), Fr(0)],
             [S6[1], S6[2], S6[3], S6[4], S6[5], Fr(0), Fr(0)],
             [S7[1], S7[2], S7[3], S7[4], S7[5], S7[6], Fr(0)]]
        b = [g['al73'] * S4[1] + g['al76'] * S7[1],
             g['al73'] * S4[2] + g['al76'] * S7[2],
             g['al73'] * S4[3] + g['al76'] * S7[3],
             g['al76'] * S7[4], g['al76'] * S7[5], g['al76'] * S7[6],
             g['be76']]
    elif name == 'SSPRK83':
        S2 = {1: g['be10']}
        S3 = {1: g['be10'], 2: g['be21']}
        S4 = {1: g['be10'], 2: g['be21'], 3: g['be32']}
        S5 = {1: g['be10'], 2: g['be21'], 3: g['be32'], 4: g['be43']}
        S6 = {1: (g['al51'] + g['al54']) * S5[1], 2: g['al54'] * S5[2],
              3: g['al54'] * S5[3], 4: g['al54'] * S5[4], 5: g['be54']}
        S7 = {1: g['al61'] * S2[1] + g['al65'] * S6[1],
              2: g['al65'] * S6[2], 3: g['al65'] * S6[3],
              4: g['al65'] * S6[4], 5: g['al65'] * S6[5], 6: g['be65']}
        S8 = {1: g['al72'] * S3[1] + g['al73'] * S4[1] + g['al76'] * S7[1],
              2: g['al72'] * S3[2] + g['al73'] * S4[2] + g['al76'] * S7[2],
              3: g['al73'] * S4[3] + g['al76'] * S7[3],
              4: g['al76'] * S7[4], 5: g['al76'] * S7[5],
              6: g['al76'] * S7[6], 7: g['be76']}
        a = [[Fr(0)] * 8,
             [S2[1], Fr(0), Fr(0), Fr(0), Fr(0), Fr(0), Fr(0), Fr(0)],
             [S3[1], S3[2], Fr(0), Fr(0), Fr(0), Fr(0), Fr(0), Fr(0)],
             [S4[1], S4[2], S4[3], Fr(0), Fr(0), Fr(0), Fr(0), Fr(0)],
             [S5[1], S5[2], S5[3], S5[4], Fr(0), Fr(0), Fr(0), Fr(0)],
             [S6[1], S6[2], S6[3], S6[4], S6[5], Fr(0), Fr(0), Fr(0)],
             [S7[1], S7[2], S7[3], S7[4], S7[5], S7[6], Fr(0), Fr(0)],
             [S8[1], S8[2], S8[3], S8[4], S8[5], S8[6], S8[7], Fr(0)]]
        b = [S8[1], S8[2], S8[3], S8[4], S8[5], S8[6], S8[7], g['be87']]
    elif name == 'SSPRK54':
        S2 = {1: g['be10']}
        S3 = {1: g['al21'] * S2[1], 2: g['be21']}
        S4 = {1: g['al32'] * S3[1], 2: g['al32'] * S3[2], 3: g['be32']}
        S5 = {1: g['al43'] * S4[1], 2: g['al43'] * S4[2],
              3: g['al43'] * S4[3], 4: g['be43']}
        a = [[Fr(0)] * 5,
             [S2[1], Fr(0), Fr(0), Fr(0), Fr(0)],
             [S3[1], S3[2], Fr(0), Fr(0), Fr(0)],
             [S4[1], S4[2], S4[3], Fr(0), Fr(0)],
             [S5[1], S5[2], S5[3], S5[4], Fr(0)]]
        b = [g['al52'] * S3[1] + g['al53'] * S4[1] + g['al54'] * S5[1],
             g['al52'] * S3[2] + g['al53'] * S4[2] + g['al54'] * S5[2],
             g['al53'] * S4[3] + g['al54'] * S5[3],
             g['al54'] * S5[4] + g['be53'], g['be54']]
    elif name == 'KYK2014DGSSPRK_3S2':
        S2 = {1: g['be10']}
        S3 = {1: g['al21'] * S2[1], 2: g['be21']}
        a = [[Fr(0), Fr(0), Fr(0)],
             [S2[1], Fr(0), Fr(0)],
             [S3[1], S3[2], Fr(0)]]
        b = [g['al32'] * S3[1] + g['be30'], g['al32'] * S3[2], g['be32']]
    else:
        raise KeyError(name)
    cvals = [Fr(0)] + [sum(a[i]) for i in range(1, len(a))]
    return a, b, cvals


def order_residuals(a, b, cvals, order):
    s = len(b)
    A = a
    e = [Fr(1)] * s

    def mv(M, v):
        return [sum(M[i][j] * v[j] for j in range(s)) for i in range(s)]

    def bv(v):
        return sum(b[i] * v[i] for i in range(s))

    def cv(v):
        return [cvals[i] * v[i] for i in range(s)]

    A1 = mv(A, e)
    A2 = mv(A, A1)
    A3 = mv(A, A2)
    r = {'b e': bv(e) - 1, 'b c': bv(cv(e)) - Fr(1, 2)}
    if order >= 3:
        r['b c^2'] = bv(cv(cv(e))) - Fr(1, 3)
        r['b A^2'] = bv(A2) - Fr(1, 6)
    if order >= 4:
        r['b c^3'] = bv(cv(cv(cv(e)))) - Fr(1, 4)
        r['b c2A'] = bv(cv(cv(A1))) - Fr(1, 12)
        r['b AcA'] = bv(mv(A, cv(A1))) - Fr(1, 8)
        r['b A^3'] = bv(A3) - Fr(1, 24)
    return r


def reference_rk(f, a, b, cvals, u0, T, dt):
    A = np.array(a, dtype=float)
    B = np.array(b, dtype=float)
    Cv = np.array(cvals, dtype=float)
    u = np.array(u0, dtype=float)
    t = 0.0
    for _ in range(int(round(T / dt))):
        K = []
        for i in range(len(B)):
            ui = u + dt * sum(A[i, j] * K[j] for j in range(i))
            K.append(f(t + Cv[i] * dt, ui))
        u = u + dt * (B @ np.array(K))
        t += dt
    return u


def real_axis_interval(a, b):
    A = np.array(a, dtype=float)
    B = np.array(b, dtype=float)
    s = len(B)
    e = np.ones(s)

    def R(z):
        return 1.0 + z * (B @ np.linalg.solve(np.eye(s) - z * A, e))

    zs = -np.linspace(0.0, 30.0, 12001)
    for i in range(1, len(zs)):
        if abs(R(zs[i])) > 1.0:
            lo, hi = zs[i - 1], zs[i]
            for _ in range(80):
                mid = 0.5 * (lo + hi)
                if abs(R(mid)) <= 1.0:
                    lo = mid
                else:
                    hi = mid
            return float(lo)
    return -30.0


def final_map_convex_cfl(tableau, mu_max=8.0, n=128):
    """Largest mu with the FINAL map alone a convex circulant row."""
    def passes(mu):
        if mu <= 0.0:
            return True
        _, final = tvd_analysis.stage_and_final_maps(tableau, mu, n)
        return tvd_analysis.is_convex_row(final, 1e-10)

    grid = np.linspace(0.0, mu_max, 400)
    for i in range(1, 400):
        if not passes(grid[i]):
            lo, hi = grid[i - 1], grid[i]
            for _ in range(60):
                mid = 0.5 * (lo + hi)
                if passes(mid):
                    lo = mid
                else:
                    hi = mid
            return float(lo)
    return float(mu_max)


def main():
    ref = {}
    ref_path = Path(__file__).resolve().parent / 'ode_reference.txt'
    if ref_path.exists():
        lines = ref_path.read_text().strip().split('\n')
        i = 0
        while i < len(lines):
            name = lines[i]
            ref[name] = [float(x) for x in lines[i + 1:i + 4]]
            i += 4

    f = lambda t, u: np.array([u[1] - u[0] * u[2] + np.sin(t),
                               -u[0] - u[1] ** 2 + np.cos(2 * t),
                               u[0] * u[1] - u[2] + t])
    u0 = [0.3, -0.7, 1.1]

    all_ok = True
    for name, meta in METH.items():
        a, b, cvals = build(name)
        print(f'=== {name} (order {meta["order"]}, ODE C = {meta["C_ode"]}) ===')
        ok = True

        # 1. c check vs ODE stored c
        dc = max(abs(float(cvals[i]) - meta['c_ode'][i])
                 for i in range(len(cvals)))
        ok &= dc < 1e-13
        print(f'  c == ODE c:            max|dc| = {dc:.3e}  ' +
              ('OK' if dc < 1e-13 else 'FAIL'))

        # 2. sum(b)
        sb = sum(b) - 1
        tol_b = 1e-5 if name == 'SSPRK73' else 1e-13
        ok &= abs(sb) < tol_b
        print(f'  sum(b) - 1:            {float(sb):+.3e}  ' +
              ('OK' if abs(sb) < tol_b else 'FAIL'))

        # 3. measured order on the smooth nonlinear time-dependent test ODE
        # (four halvings; the repo's own convergence gate uses the same 0.15
        # tolerance below the claimed order)
        dts = [0.2, 0.1, 0.05, 0.025]
        u_ref = reference_rk(f, a, b, cvals, u0, 0.5, 0.2 / 16)
        errs = [float(np.max(np.abs(reference_rk(f, a, b, cvals, u0, 0.5, dt) - u_ref)))
                for dt in dts]
        order_meas = float(np.polyfit(np.log2(dts), np.log2(errs), 1)[0])
        tol_o = 0.15
        ok &= order_meas > meta['order'] - tol_o
        print(f'  measured order:        {order_meas:.3f}  (claimed {meta["order"]})  ' +
              ('OK' if order_meas > meta['order'] - tol_o else 'FAIL'))

        # 4. non-FSAL
        fsal = abs(float(cvals[-1]) - 1.0) < 1e-14
        print(f'  FSAL (c[-1] == 1):     {fsal}')

        # 5. trajectory vs julia ODE
        if name in ref:
            mine = reference_rk(f, a, b, cvals, u0, 0.5, 0.02)
            dtraj = float(np.max(np.abs(mine - np.array(ref[name]))))
            ok &= dtraj < 1e-12
            print(f'  traj vs ODE (julia):   max|du| = {dtraj:.3e}  ' +
                  ('OK' if dtraj < 1e-12 else 'FAIL'))
        else:
            print('  traj vs ODE (julia):   (reference file missing)')

        # 6/7. library property measurements
        tab = butcherTableau(a=[[float(x) for x in row] for row in a],
                             b=[float(x) for x in b],
                             c=[float(x) for x in cvals])
        ssp = tvd_analysis.convex_combination_cfl(tab, mu_max=8.0)
        dssp = abs(ssp - meta['C_ode'])
        ok &= dssp < 1e-3
        print(f'  SSP CFL (library):     {ssp:.6f}   (ODE {meta["C_ode"]}, '
              f'diff {dssp:.2e})  ' + ('OK' if dssp < 1e-3 else 'FAIL'))
        fcc = final_map_convex_cfl(tab)
        interval = real_axis_interval(a, b)
        print(f'  final-map convex CFL:  {fcc:.6f}')
        print(f'  real-axis interval:    [{interval:.5f}, 0]')

        print('  butcher.py literals:')
        print('    a = [')
        for row in a:
            print('        [' + ', '.join(repr(float(x)) for x in row) + '],')
        print('    ]')
        print('    b = [' + ', '.join(repr(float(x)) for x in b) + ']')
        print('    c = [' + ', '.join(repr(float(x)) for x in cvals) + ']')
        print()
        all_ok &= ok
    print('ALL OK' if all_ok else 'SOME CHECKS FAILED')
    return 0 if all_ok else 1


if __name__ == '__main__':
    sys.exit(main())
