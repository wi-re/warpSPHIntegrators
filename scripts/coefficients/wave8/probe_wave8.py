"""Wave 8 probe v2: verify every property of the ODE v7.8.1 SDIRK tableaus we
plan to register, by parsing the Julia dump (extract_wave8_out.txt).

Key findings handled here (see NOTES §3.28 for the write-up):
  * ODE stores c2 = gamma (not 2*gamma) for Kvaerno3/4/5: the materializer
    puts the SDIRK diagonal into the second node while row 2 sums to 2*gamma.
    The true node is the row sum; ODE's execution evaluates stage 2 at the
    stored (wrong) time for non-autonomous problems.
  * ODE stores c4 = 49/453 for ESDIRK547L2SA2 while row 4 sums to
    0.13881020330028328; with the row-sum c, b.c = 1/2 exactly, so the row
    sums are the consistent abscissae.
  * SFSDIRK4-8 / Cash4 store 12-significant-digit coefficients: consistency
    holds only to ~1e-11.

Checks per scheme (using the CONSISTENT abscissae c = row sums):
  * |c_rowsum - c_stored|  (the ODE storage mismatch, if any)
  * sum(b) == 1
  * moment conditions b.c^k == 1/(k+1), k = 0..p-1
  * b_embed = b - btilde: sum == 1, moments to p-2  (where btilde present)
  * R(inf) exactly via the det(I - zA + z b 1^T)/det(I - zA) leading terms
  * A(alpha) cone: largest alpha (deg) with |R| <= 1 on the sector around -R+
  * real-axis monotonicity interval: max x >= 0 with |R(-x')| <= 1 on [0, x]

Plus identity checks:
  * KenCarp3 core == library ESDIRK324L2SA core
  * ARS222 implicit side == ARS232 implicit side
  * SDIRK22compact gamma == library SDIRK2 gamma (to 1 ulp)
  * ODE SDIRK2 (gamma=1) reindexed to standard form: R(inf) = -1/2
"""
from pathlib import Path
import re
import sys

import numpy as np

HERE = __file__.rsplit("/", 1)[0]


def parse_dump(path):
    text = open(path).read()
    schemes = {}
    for m in re.finditer(r"^=== (\S+) \((\w+)\) ===\n((?:  \S.*\n?)+)", text, re.M):
        name, kind, body = m.group(1), m.group(2), m.group(3)
        d = {"kind": kind}
        for line in body.strip().splitlines():
            key, val = line.strip().split(" = ", 1)
            d[key] = val
        schemes[name] = d
    return schemes


def pmat(val):
    if val.startswith("Matrix{") or val.startswith("Vector{") or val == "Float64[]":
        return None
    v = val.strip()
    if v.startswith("["):
        rows = v[1:-1].split(";")
        return np.array([[float(x) for x in r.split()] for r in rows])
    return None


def pvec(val):
    if val.startswith("Matrix{") or val.startswith("Vector{") or val == "Float64[]":
        return None
    v = val.strip()
    return np.array([float(x) for x in v[1:-1].split(",")])


def moments(b, c, order):
    worst = 0.0
    for k in range(order):
        worst = max(worst, abs(float(b @ np.power(c, k)) - 1.0 / (k + 1)))
    return worst


def R_of(A, b, z):
    s = A.shape[0]
    M = np.eye(s, dtype=complex) - z * A.astype(complex)
    v = np.linalg.solve(M, np.ones(s, dtype=complex))
    return 1.0 + z * (b.astype(complex) @ v)


def r_inf(A, b):
    """Exact R(inf) from det(I + zB) = prod_i (1 + z*lambda_i(B)) for
    Bnum = e*b^T - A and Bden = -A: R(z) = 1 + z b^T (I - zA)^-1 e
    = det(I - zA + z e b^T)/det(I - zA) (matrix-determinant lemma with
    u = e, v = b -- note e*b^T, NOT b*e^T). Coefficients are the elementary
    symmetric polynomials of the eigenvalues, by convolution.
    |R(inf)| == 0 means L-stable."""
    s = A.shape[0]
    Bnum = np.outer(np.ones(s), b) - A
    Bden = -A

    def coeffs(B):
        lam = np.linalg.eigvals(B)
        c = np.array([1.0 + 0.0j])
        for l in lam:
            c = np.convolve(c, np.array([1.0, l]))
        return c  # c[k] * z^k

    cn, cd = coeffs(Bnum), coeffs(Bden)

    def deg(c):
        scale = max(abs(x) for x in c)
        for k in range(len(c) - 1, -1, -1):
            if abs(c[k]) > 1e-11 * scale:
                return k
        return 0

    dn, dd = deg(cn), deg(cd)
    if dn < dd:
        return 0.0 + 0.0j
    return cn[dn] / cd[dd]


def a_alpha(A, b, max_r=100.0, tol=1e-7):
    best = 0.0
    for th_deg in np.arange(0.0, 90.0, 1.0):
        th = np.deg2rad(th_deg)
        ok = True
        for r in np.geomspace(0.05, max_r, 80):
            if abs(R_of(A, b, r * np.exp(1j * (np.pi - th)))) > 1.0 + tol:
                ok = False
                break
        if ok:
            best = th_deg
        else:
            break
    return float(best)


def real_axis_interval(A, b, x_max=80.0):
    """Largest x with |R(-x')| <= 1 for all x' in [0, x] (0.02 grid, refine)."""
    xs = np.linspace(0.0, x_max, 4001)
    for i, x in enumerate(xs):
        if abs(R_of(A, b, -x)) > 1.0 + 1e-9:
            lo, hi = xs[i - 1], x
            for _ in range(40):
                mid = 0.5 * (lo + hi)
                if abs(R_of(A, b, -mid)) <= 1.0 + 1e-9:
                    lo = mid
                else:
                    hi = mid
            return float(lo)
    return float(x_max)


PLAN = [
    "Kvaerno3", "Kvaerno4", "Kvaerno5",
    "KenCarp3", "KenCarp4", "KenCarp47", "KenCarp5", "KenCarp58",
    "ESDIRK325L2SA", "ESDIRK436L2SA2", "ESDIRK437L2SA", "ESDIRK547L2SA2",
    "ESDIRK54I8L2SA", "ESDIRK659L2SA", "SSPSDIRK2",
    "ARS222", "ARS343", "ARS443", "BHR553", "Cash4", "Hairer4", "Hairer42",
    "SFSDIRK4", "SFSDIRK5", "SFSDIRK6", "SFSDIRK7", "SFSDIRK8",
]

# SSPSDIRK2: the dump's c=[1,1] is an execution artifact (explicit_fsallast,
# const_stage_guess=[0,3]); the published Ketcheson-Macdonald-Gottlieb 2009 form
# has c = row sums = [1/4, 3/4] (row sums of the stored Ai already give this).


def main():
    dump = parse_dump(HERE + "/extract_wave8_out.txt")
    fails = []
    hdr = (f"{'scheme':16s} {'p':>2s} {'s':>2s} {'c_mm':>9s} {'sumb':>9s} {'mom':>9s} "
           f"{'bemb_mom':>9s} {'|R_inf|':>8s} {'A(a)':>6s} {'ReAxis':>9s}")
    print(hdr)
    print("-" * len(hdr))
    for name in PLAN:
        d = dump[name]
        A = pmat(d["Ai"])
        b = pvec(d["bi"])
        c_stored = pvec(d["c"])
        c = A.sum(axis=1)  # consistent abscissae (row sums)
        if name == "SSPSDIRK2":
            c = np.array([0.25, 0.75])
            c_stored = c  # documented artifact; row sums already [1/4, 3/4]
        bt = pvec(d["btilde"])
        p = int(d["order"])
        s = int(d["s"])
        assert A.shape == (s, s) and len(b) == s and len(c) == s
        cmm = 0.0 if name == "SSPSDIRK2" else float(np.abs(c - c_stored).max())
        sb = abs(float(b.sum()) - 1.0)
        mom = moments(b, c, p)
        bemb_mom = float("nan")
        if bt is not None:
            be = b - bt
            bemb_mom = max(abs(float(be.sum()) - 1.0), moments(be, c, p - 1))
        ri = abs(r_inf(A, b))
        alpha = a_alpha(A, b)
        rax = real_axis_interval(A, b)
        print(f"{name:16s} {p:2d} {s:2d} {cmm:9.3e} {sb:9.2e} {mom:9.2e} "
              f"{bemb_mom:9.2e} {ri:8.2e} {alpha:5.1f}d {rax:9.4f}")
        # 12-digit-coefficient families (SFSDIRK*, Cash4) are only consistent
        # to ~1e-11; everything else must be at the 1e-13 level.
        loose = name in ("Cash4", "SFSDIRK4", "SFSDIRK5", "SFSDIRK6", "SFSDIRK7", "SFSDIRK8")
        t_mom, t_sum = (1e-9, 1e-9) if loose else (1e-13, 1e-14)
        ok = sb < t_sum and mom < t_mom
        if bt is not None:
            ok = ok and bemb_mom < t_mom
        if not ok:
            fails.append((name, "moments", sb, mom, bemb_mom))

    print()
    print("=== identity / cross-checks ===")
    sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
    from warpSPHIntegrators.dirk import getDIRKTableau  # noqa: E402

    kc3 = dump["KenCarp3"]
    ref = getDIRKTableau("ESDIRK324L2SA")
    dA = float(np.abs(pmat(kc3["Ai"]) - ref.a).max())
    db = float(np.abs(pvec(kc3["bi"]) - ref.b[0]).max())
    dc = float(np.abs(pvec(kc3["c"]) - ref.c).max())
    print(f"KenCarp3 core vs ESDIRK324L2SA: dA={dA:.3e} db={db:.3e} dc={dc:.3e}")
    if max(dA, db, dc) > 1e-14:
        fails.append(("KenCarp3", "core-identity", dA, db, dc))
    a222, a232 = dump["ARS222"], dump["ARS232"]
    same = (pmat(a222["Ai"]) == pmat(a232["Ai"])).all() \
        and (pvec(a222["bi"]) == pvec(a232["bi"])).all() \
        and (pvec(a222["c"]) == pvec(a232["c"])).all()
    print(f"ARS222 implicit side == ARS232 implicit side: {bool(same)}")
    if not same:
        fails.append(("ARS232", "duplicate-check-failed"))
    g22 = float(dump["SDIRK22compact"]["a"].strip())
    sdirk2 = getDIRKTableau("SDIRK2")
    glib = float(sdirk2.a[0][0])
    ulp = np.abs(np.spacing(glib))
    print(f"SDIRK22compact a={g22!r}  library SDIRK2 gamma={glib!r}  "
          f"diff={abs(g22 - glib):.2e} (1 ulp = {ulp:.2e})")
    if abs(g22 - glib) > 2 * ulp:
        fails.append(("SDIRK22", "gamma-mismatch", g22, glib))
    d2 = dump["SDIRK2"]
    A = pmat(d2["Ai"])
    b = pvec(d2["bi"])
    c = pvec(d2["c"])
    Astd = A[np.array([1, 0]), :][:, np.array([1, 0])]
    bstd = b[np.array([1, 0])]
    cstd = c[np.array([1, 0])]
    rs = float(np.abs(Astd.sum(axis=1) - cstd).max())
    ri = r_inf(Astd, bstd)
    rax = real_axis_interval(Astd, bstd)
    print(f"ODE SDIRK2 standard form: c={cstd.tolist()} rs={rs:.2e} "
          f"R_inf={ri:.4f} ReAxis={rax:.3f} (expect -0.5, not L-stable)")
    if abs(ri + 0.5) > 1e-9:
        fails.append(("ODE-SDIRK2", "R_inf-not-minus-half", ri))

    print()
    if fails:
        print(f"FAILURES ({len(fails)}):")
        for f in fails:
            print("  ", f)
        sys.exit(1)
    print("ALL CHECKS PASS")


if __name__ == "__main__":
    main()
