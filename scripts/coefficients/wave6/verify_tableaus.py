"""Wave 6 verification probe.

Parses the Julia extraction (wave6_julia/extract_out.txt), constructs every
tableau per the ODE perform_step decoding, and verifies:
  1. row sums == c
  2. FSAL structure (last a-row == b_main, c_s == 1, b_s == 0)
  3. numerical order (main map; embedded map for pairs)
  4. embedded-rate: u_main - u_embed ~ O(dt^{q+1})
  5. SSP coefficient (library convex_combination_cfl, mu_max=8)
  6. real-axis stability interval (test_tvd.py method)
"""
from pathlib import Path
import re
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
from types import SimpleNamespace
from warpSPHIntegrators.tvd_analysis import convex_combination_cfl

# ---------------------------------------------------------------- parsing
text = open(str(Path(__file__).resolve().parent / "extract_out.txt")).read()
vals = {}
current = None
for line in text.splitlines():
    line = line.strip()
    if line.startswith("### "):
        current = line[4:].strip()
        vals[current] = {}
    elif " = " in line and current is not None:
        f, v = line.split(" = ", 1)
        if v.startswith("["):
            vals[current][f] = [float(x) for x in v.strip("[]").split(",")]
        else:
            vals[current][f] = float(v)

print("parsed caches:", len(vals))

# ---------------------------------------------------------------- specs
M = {}  # name -> dict(a, b, b_embed or None, c)

def zero(s):
    return [[0.0] * s for _ in range(s)]

def put(a, row, colspec):
    for j, v in colspec:
        a[row][j] = v

# Tsit5: 7 stages (6 + FSAL), 5(4)
t = vals["Tsit5ConstantCacheActual"]
a = zero(7)
put(a, 1, [(0, t["a21"])])
put(a, 2, [(0, t["a31"]), (1, t["a32"])])
put(a, 3, [(0, t["a41"]), (1, t["a42"]), (2, t["a43"])])
put(a, 4, [(0, t["a51"]), (1, t["a52"]), (2, t["a53"]), (3, t["a54"])])
put(a, 5, [(0, t["a61"]), (1, t["a62"]), (2, t["a63"]), (3, t["a64"]), (4, t["a65"])])
put(a, 6, [(0, t["a71"]), (1, t["a72"]), (2, t["a73"]), (3, t["a74"]), (4, t["a75"]), (5, t["a76"])])
c = [0.0, t["c1"], t["c2"], t["c3"], t["c4"], t["c5"], t["c6"]]
b_main = a[6][:]  # last column already 0 (row 6 of the 7x7 matrix)
btilde = [t["btilde%d" % i] for i in range(1, 8)]
M["Tsit5"] = dict(a=a, b=b_main, b_embed=[b_main[i] - btilde[i] for i in range(7)], c=c)

# OwrenZen3: 4 stages (3 + FSAL), 3(2)
t = vals["OwrenZen3ConstantCache"]
a = zero(4)
put(a, 1, [(0, t["a21"])])
put(a, 2, [(0, t["a31"]), (1, t["a32"])])
put(a, 3, [(0, t["a41"]), (1, t["a42"]), (2, t["a43"])])
c = [0.0, t["c1"], t["c2"], 1.0]
b_main = [t["a41"], t["a42"], t["a43"], 0.0]
bt = [t["btilde1"], t["btilde2"], t["btilde3"], 0.0]
M["OwrenZen3"] = dict(a=a, b=b_main, b_embed=[b_main[i] - bt[i] for i in range(4)], c=c)

# OwrenZen4: 6 stages (5 + FSAL), 4(3)
t = vals["OwrenZen4ConstantCache"]
a = zero(6)
put(a, 1, [(0, t["a21"])])
put(a, 2, [(0, t["a31"]), (1, t["a32"])])
put(a, 3, [(0, t["a41"]), (1, t["a42"]), (2, t["a43"])])
put(a, 4, [(0, t["a51"]), (1, t["a52"]), (2, t["a53"]), (3, t["a54"])])
c = [0.0, t["c1"], t["c2"], t["c3"], t["c4"], 1.0]
b_main = [t["a61"], 0.0, t["a63"], t["a64"], t["a65"], 0.0]
bt = [t["btilde1"], 0.0, t["btilde3"], t["btilde4"], t["btilde5"], 0.0]
a[5] = b_main[:]
M["OwrenZen4"] = dict(a=a, b=b_main, b_embed=[b_main[i] - bt[i] for i in range(6)], c=c)

# OwrenZen5: 8 stages (7 + FSAL), 5(4); a43 = 1 implicit in ODE code
t = vals["OwrenZen5ConstantCache"]
a = zero(8)
put(a, 1, [(0, t["a21"])])
put(a, 2, [(0, t["a31"]), (1, t["a32"])])
put(a, 3, [(0, t["a41"]), (1, t["a42"]), (2, 1.0)])
put(a, 4, [(0, t["a51"]), (1, t["a52"]), (2, t["a53"]), (3, t["a54"])])
put(a, 5, [(0, t["a61"]), (1, t["a62"]), (2, t["a63"]), (3, t["a64"]), (4, t["a65"])])
put(a, 6, [(0, t["a71"]), (1, t["a72"]), (2, t["a73"]), (3, t["a74"]), (4, t["a75"]), (5, t["a76"])])
c = [0.0, t["c1"], t["c2"], t["c3"], t["c4"], t["c5"], t["c6"], 1.0]
b_main = [t["a81"], 0.0, t["a83"], t["a84"], t["a85"], t["a86"], t["a87"], 0.0]
bt = [t["btilde1"], 0.0, t["btilde3"], t["btilde4"], t["btilde5"], t["btilde6"], t["btilde7"], 0.0]
a[7] = b_main[:]
M["OwrenZen5"] = dict(a=a, b=b_main, b_embed=[b_main[i] - bt[i] for i in range(8)], c=c)

# BS5: 8 stages (7 + FSAL), 5(4); bhat = difference to the 4th-order embedded
t = vals["BS5ConstantCache"]
a = zero(8)
put(a, 1, [(0, t["a21"])])
put(a, 2, [(0, t["a31"]), (1, t["a32"])])
put(a, 3, [(0, t["a41"]), (1, t["a42"]), (2, t["a43"])])
put(a, 4, [(0, t["a51"]), (1, t["a52"]), (2, t["a53"]), (3, t["a54"])])
put(a, 5, [(0, t["a61"]), (1, t["a62"]), (2, t["a63"]), (3, t["a64"]), (4, t["a65"])])
put(a, 6, [(0, t["a71"]), (1, t["a72"]), (2, t["a73"]), (3, t["a74"]), (4, t["a75"]), (5, t["a76"])])
c = [0.0, t["c1"], t["c2"], t["c3"], t["c4"], t["c5"], 1.0, 1.0]
b_main = [t["a81"], 0.0, t["a83"], t["a84"], t["a85"], t["a86"], t["a87"], 0.0]
bt = [t["bhat1"], 0.0, t["bhat3"], t["bhat4"], t["bhat5"], t["bhat6"], 0.0, 0.0]
a[7] = b_main[:]
M["BS5"] = dict(a=a, b=b_main, b_embed=[b_main[i] - bt[i] for i in range(8)], c=c)

# Stepanov5: 7 stages (6 + FSAL), 5(4)
t = vals["Stepanov5ConstantCache"]
a = zero(7)
put(a, 1, [(0, t["a21"])])
put(a, 2, [(0, t["a31"]), (1, t["a32"])])
put(a, 3, [(0, t["a41"]), (1, t["a42"]), (2, t["a43"])])
put(a, 4, [(0, t["a51"]), (1, t["a52"]), (2, t["a53"]), (3, t["a54"])])
put(a, 5, [(0, t["a61"]), (1, t["a62"]), (2, t["a63"]), (3, t["a64"]), (4, t["a65"])])
c = [0.0, t["c2"], t["c3"], t["c4"], t["c5"], t["c6"], 1.0]
b_main = [t["b1"], 0.0, t["b3"], t["b4"], t["b5"], t["b6"], 0.0]
a[6] = b_main[:]
btilde = [t["btilde%d" % i] for i in range(1, 8)]
M["Stepanov5"] = dict(a=a, b=b_main, b_embed=[b_main[i] - btilde[i] for i in range(7)], c=c)

# SIR54: 8 stages (7 + FSAL), 5(4); b == a7 row
t = vals["SIR54ConstantCache"]
a = zero(8)
put(a, 1, [(0, t["a21"])])
put(a, 2, [(0, t["a31"]), (1, t["a32"])])
put(a, 3, [(0, t["a41"]), (1, t["a42"]), (2, t["a43"])])
put(a, 4, [(0, t["a51"]), (1, t["a52"]), (2, t["a53"]), (3, t["a54"])])
put(a, 5, [(0, t["a61"]), (1, t["a62"]), (2, t["a63"]), (3, t["a64"]), (4, t["a65"])])
put(a, 6, [(0, t["a71"]), (1, t["a72"]), (2, t["a73"]), (3, t["a74"]), (4, t["a75"]), (5, t["a76"])])
c = [0.0, t["c2"], t["c3"], t["c4"], t["c5"], t["c6"], t["c7"], 1.0]
b_main = [t["b1"], t["b2"], t["b3"], t["b4"], t["b5"], t["b6"], 0.0, 0.0]
assert max(abs(b_main[i] - a[6][i]) for i in range(6)) < 1e-15, "SIR54 b != a7 row"
a[7] = b_main[:]
btilde = [t["btilde%d" % i] for i in range(1, 8)]
M["SIR54"] = dict(a=a, b=b_main, b_embed=[b_main[i] - btilde[i] for i in range(7)] + [0.0], c=c)

# Alshina2: 2 stages, 2(1); valid embedded = Euler (b_main - (b1tilde, b2tilde))
t = vals["Alshina2ConstantCache"]
a = [[0.0, 0.0], [t["a21"], 0.0]]
c = [0.0, t["c2"]]
b_main = [t["b1"], t["b2"]]
b_embed = [b_main[0] - t["b1tilde"], b_main[1] - t["b2tilde"]]
M["Alshina2"] = dict(a=a, b=b_main, b_embed=b_embed, c=c)

# Alshina3: 3 stages, 3rd order; ODE estimate utilde = dt*b2tilde*k2 gives
# b_main - (0, b2tilde, 0) with sum 5/9 != 1 -> NOT a valid embedded; fixed-step
t = vals["Alshina3ConstantCache"]
a = [[0.0, 0.0, 0.0], [t["a21"], 0.0, 0.0], [0.0, t["a32"], 0.0]]
c = [0.0, t["c2"], t["c3"]]
b_main = [t["b1"], t["b2"], t["b3"]]
M["Alshina3"] = dict(a=a, b=b_main, b_embed=None, c=c)

# Alshina6: 7 stages, 6th order, fixed-step
t = vals["Alshina6ConstantCache"]
a = zero(7)
put(a, 1, [(0, t["a21"])])
put(a, 2, [(0, t["a31"]), (1, t["a32"])])
put(a, 3, [(0, t["a41"]), (1, t["a42"]), (2, t["a43"])])
put(a, 4, [(0, t["a51"]), (1, t["a52"]), (2, t["a53"]), (3, t["a54"])])
put(a, 5, [(0, t["a61"]), (1, t["a62"]), (2, t["a63"]), (3, t["a64"]), (4, t["a65"])])
put(a, 6, [(0, t["a71"]), (1, t["a72"]), (2, t["a73"]), (3, t["a74"]), (4, t["a75"]), (5, t["a76"])])
c = [0.0, t["c2"], t["c3"], t["c4"], t["c5"], t["c6"], t["c7"]]
b_main = [t["b1"], 0.0, 0.0, 0.0, t["b5"], t["b6"], t["b7"]]
M["Alshina6"] = dict(a=a, b=b_main, b_embed=None, c=c)

# MSRK5: 9 stages (8 + FSAL), 5th order, fixed-step
t = vals["MSRK5ConstantCache"]
a = zero(9)
put(a, 1, [(0, t["a21"])])
put(a, 2, [(0, t["a31"]), (1, t["a32"])])
put(a, 3, [(0, t["a41"]), (2, t["a43"])])
put(a, 4, [(0, t["a51"]), (2, t["a53"]), (3, t["a54"])])
put(a, 5, [(0, t["a61"]), (2, t["a63"]), (3, t["a64"]), (4, t["a65"])])
put(a, 6, [(0, t["a71"]), (2, t["a73"]), (3, t["a74"]), (4, t["a75"]), (5, t["a76"])])
put(a, 7, [(0, t["a81"]), (2, t["a83"]), (3, t["a84"]), (4, t["a85"]), (5, t["a86"]), (6, t["a87"])])
c = [0.0, t["c2"], t["c3"], t["c4"], t["c5"], t["c6"], t["c7"], t["c8"], 1.0]
b_main = [t["b1"], 0.0, 0.0, t["b4"], t["b5"], t["b6"], t["b7"], t["b8"], 0.0]
a[8] = b_main[:]
M["MSRK5"] = dict(a=a, b=b_main, b_embed=None, c=c)

# MSRK6: 9 stages (8 + FSAL), 6th order, fixed-step; row3 uses only k2
t = vals["MSRK6ConstantCache"]
a = zero(9)
put(a, 1, [(0, t["a21"])])
put(a, 2, [(1, t["a32"])])
put(a, 3, [(0, t["a41"]), (2, t["a43"])])
put(a, 4, [(0, t["a51"]), (2, t["a53"]), (3, t["a54"])])
put(a, 5, [(0, t["a61"]), (2, t["a63"]), (3, t["a64"]), (4, t["a65"])])
put(a, 6, [(0, t["a71"]), (2, t["a73"]), (3, t["a74"]), (4, t["a75"]), (5, t["a76"])])
put(a, 7, [(0, t["a81"]), (2, t["a83"]), (3, t["a84"]), (4, t["a85"]), (5, t["a86"]), (6, t["a87"])])
c = [0.0, t["c2"], t["c3"], t["c4"], t["c5"], t["c6"], t["c7"], t["c8"], 1.0]
b_main = [t["b1"], 0.0, 0.0, t["b4"], t["b5"], t["b6"], t["b7"], t["b8"], 0.0]
a[8] = b_main[:]
M["MSRK6"] = dict(a=a, b=b_main, b_embed=None, c=c)

# RKO65: 6 stages, 5th order, fixed-step; first stage at c1 = 2/3 (not t)
t = vals["RKO65ConstantCache"]
a = zero(6)
put(a, 1, [(0, t["\u03b121"])])
put(a, 2, [(0, t["\u03b131"]), (1, t["\u03b132"])])
put(a, 3, [(0, t["\u03b141"]), (1, t["\u03b142"]), (2, t["\u03b143"])])
put(a, 4, [(0, t["\u03b151"]), (1, t["\u03b152"]), (2, t["\u03b153"]), (3, t["\u03b154"])])
put(a, 5, [(1, t["\u03b162"]), (2, t["\u03b163"]), (3, t["\u03b164"]), (4, t["\u03b165"])])
c = [t["c1"], t["c2"], t["c3"], t["c4"], t["c5"], t["c6"]]
b_main = [0.0, t["\u03b22"], t["\u03b23"], t["\u03b24"], t["\u03b25"], t["\u03b26"]]
M["RKO65"] = dict(a=a, b=b_main, b_embed=None, c=c)

# RKM: 6 stages, 1-2-1-1-1 low-storage, fixed-step
t = vals["RKMConstantCache"]
a = zero(6)
put(a, 1, [(0, t["\u03b12"])])
put(a, 2, [(1, t["\u03b13"])])
put(a, 3, [(2, t["\u03b14"])])
put(a, 4, [(3, t["\u03b15"])])
put(a, 5, [(4, t["\u03b16"])])
c = [0.0, t["c2"], t["c3"], t["c4"], t["c5"], t["c6"]]
b_main = [t["\u03b21"], t["\u03b22"], t["\u03b23"], t["\u03b24"], 0.0, t["\u03b26"]]
M["RKM"] = dict(a=a, b=b_main, b_embed=None, c=c)

# Ralston4: 4 stages, 4th order, fixed-step
t = vals["Ralston4ConstantCache"]
a = zero(4)
put(a, 1, [(0, t["a21"])])
put(a, 2, [(0, t["a31"]), (1, t["a32"])])
put(a, 3, [(0, t["a41"]), (1, t["a42"]), (2, t["a43"])])
c = [0.0, t["c2"], t["c3"], 1.0]
b_main = [t["b1"], t["b2"], t["b3"], t["b4"]]
M["Ralston4"] = dict(a=a, b=b_main, b_embed=None, c=c)

# PSRK4p7q6: 6 stages, fixed-step
t = vals["PSRK4p7q6ConstantCache"]
a = zero(6)
put(a, 1, [(0, t["a21"])])
put(a, 2, [(0, t["a31"]), (1, t["a32"])])
put(a, 3, [(0, t["a41"]), (1, t["a42"]), (2, t["a43"])])
put(a, 4, [(0, t["a51"]), (1, t["a52"]), (2, t["a53"]), (3, t["a54"])])
put(a, 5, [(0, t["a61"]), (1, t["a62"]), (2, t["a63"]), (3, t["a64"]), (4, t["a65"])])
c = [0.0, t["c2"], t["c3"], t["c4"], t["c5"], t["c6"]]
b_main = [t["b1"], t["b2"], t["b3"], t["b4"], t["b5"], t["b6"]]
M["PSRK4p7q6"] = dict(a=a, b=b_main, b_embed=None, c=c)

# PSRK3p6q5: 5 stages, fixed-step
t = vals["PSRK3p6q5ConstantCache"]
a = zero(5)
put(a, 1, [(0, t["a21"])])
put(a, 2, [(0, t["a31"]), (1, t["a32"])])
put(a, 3, [(0, t["a41"]), (1, t["a42"]), (2, t["a43"])])
put(a, 4, [(0, t["a51"]), (1, t["a52"]), (2, t["a53"]), (3, t["a54"])])
c = [0.0, t["c2"], t["c3"], t["c4"], t["c5"]]
b_main = [t["b1"], t["b2"], t["b3"], t["b4"], t["b5"]]
M["PSRK3p6q5"] = dict(a=a, b=b_main, b_embed=None, c=c)

# PSRK3p5q4: 4 stages, fixed-step
t = vals["PSRK3p5q4ConstantCache"]
a = zero(4)
put(a, 1, [(0, t["a21"])])
put(a, 2, [(0, t["a31"]), (1, t["a32"])])
put(a, 3, [(0, t["a41"]), (1, t["a42"]), (2, t["a43"])])
c = [0.0, t["c2"], t["c3"], t["c4"]]
b_main = [t["b1"], t["b2"], t["b3"], t["b4"]]
M["PSRK3p5q4"] = dict(a=a, b=b_main, b_embed=None, c=c)

# ---------------------------------------------------------------- checks
def rk_step(a, b, c, f, u0, dt, t0):
    s = len(b)
    ks = []
    for i in range(s):
        ui = u0 + dt * sum(a[i][j] * ks[j] for j in range(i))
        ks.append(f(t0 + c[i] * dt, ui))
    return u0 + dt * sum(b[i] * ks[i] for i in range(s)), ks

def problem_f(t, u):
    return np.array([
        -u[1] - 0.5 * u[0] * u[2],
        u[0] + 0.3 * u[1] * u[2] - 0.2 * u[1] ** 2,
        0.1 * u[0] * u[1] + 0.05 * np.cos(3.0 * t) * u[2],
    ])

T_END = 0.5
def reference_solution():
    # high-accuracy reference via dense tiny-step integration of a 7th-order
    # composite: just use very fine RK4 (error ~ (dt)^4 * T; dt = T/20000 -> ~1e-15)
    n = 20000
    dt = T_END / n
    u = np.array([0.3, -0.2, 0.1])
    for i in range(n):
        t = i * dt
        k1 = problem_f(t, u)
        k2 = problem_f(t + 0.5 * dt, u + 0.5 * dt * k1)
        k3 = problem_f(t + 0.5 * dt, u + 0.5 * dt * k2)
        k4 = problem_f(t + dt, u + dt * k3)
        u = u + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
    return u

REF = reference_solution()
print("reference solution:", REF)

def measured_order(spec, nsteps_list=(8, 16, 32, 64, 128)):
    u0 = np.array([0.3, -0.2, 0.1])
    errs = []
    for n in nsteps_list:
        dt = T_END / n
        u = u0.copy()
        for i in range(n):
            u, _ = rk_step(spec["a"], spec["b"], spec["c"], problem_f, u, dt, i * dt)
        errs.append(np.max(np.abs(u - REF)))
    orders = []
    for i in range(len(errs) - 1):
        if errs[i] > 0 and errs[i + 1] > 0:
            orders.append(np.log(errs[i] / errs[i + 1]) / np.log(2))
    return errs, orders

def embedded_rate(spec, nsteps_list=(16, 32, 64, 128)):
    """u_main - u_embed should scale like dt^{q+1} where q = embedded order."""
    u0 = np.array([0.3, -0.2, 0.1])
    errs = []
    for n in nsteps_list:
        dt = T_END / n
        u = u0.copy()
        for i in range(n):
            ui, ks = rk_step(spec["a"], spec["b"], spec["c"], problem_f, u, dt, i * dt)
            ue = u + dt * sum(spec["b_embed"][j] * ks[j] for j in range(len(ks)))
            errs.append(np.max(np.abs(ui - ue)))
            u = ui
        errs = errs[-1:]
    rates = []
    for i in range(len(errs) - 1):
        rates.append(np.log(errs[i] / errs[i + 1]) / np.log(2))
    return errs, rates

def real_axis_left(a, b, c, guess=40.0, npts=20001):
    # test_tvd.py method, state-tracking: us[0]=1, us[i] = 1 + z*sum_j A[i,j]us[j],
    # R = 1 + z * sum_i b_i us_i; left = min z with |R| <= 1+1e-9.
    A = np.array(a, dtype=float)
    B = np.array(b, dtype=float)
    s = len(B)
    amps = np.empty(npts)
    z = np.linspace(-guess, 0.0, npts)
    for idx, zi in enumerate(z):
        us = np.empty(s)
        us[0] = 1.0
        for i in range(1, s):
            us[i] = 1.0 + zi * np.dot(A[i, :i], us[:i])
        amps[idx] = abs(1.0 + zi * np.dot(B, us))
    ok = z[amps <= 1.0 + 1e-9]
    return float(ok.min()) if len(ok) else None

print()
print("%-14s %6s %6s %10s %12s  %-8s %s" % ("method", "stages", "order", "SSP(mu=8)", "real-left", "FSAL", "embedded-rate"))
print("-" * 88)
for name, spec in M.items():
    s = len(spec["b"])
    # row sums
    rs = max(abs(sum(spec["a"][i]) - spec["c"][i]) for i in range(s))
    # b sum
    bs = abs(sum(spec["b"]) - 1.0)
    fsal = (s >= 2 and spec["c"][-1] == 1.0 and spec["b"][-1] == 0.0
            and max(abs(spec["a"][-1][j] - spec["b"][j]) for j in range(s - 1)) < 1e-14)
    errs, orders = measured_order(spec)
    ord_est = orders[-1] if orders else float("nan")
    tab = SimpleNamespace(
        a=np.array(spec["a"], dtype=float),
        b=np.array(spec["b"], dtype=float),
        c=np.array(spec["c"], dtype=float),
    )
    ssp = convex_combination_cfl(tab, mu_max=8.0)
    ra = real_axis_left(spec["a"], spec["b"], spec["c"])
    er = ""
    if spec["b_embed"] is not None:
        eerrs, erates = embedded_rate(spec)
        er = "%.2f" % erates[-1] if erates else "?"
        # also check embedded order conditions numerically: embedded map order
        # (measure order of the embedded solution itself)
    print("%-14s %6d %6.2f %10.4f %12.5f  %-8s %s   (rows %.1e, bsum %.1e)" % (
        name, s, ord_est, ssp, ra if ra is not None else float("inf"), str(fsal).lower(), er, rs, bs))
    # embedded map own order (for pairs): measure convergence of embedded solution
    if spec["b_embed"] is not None:
        em = dict(a=spec["a"], b=spec["b_embed"], c=spec["c"])
        eerrs2, eorders2 = measured_order(em)
        print("%-14s        embedded-map order estimate: %.2f (errs %s)" % (
            name, eorders2[-1], " ".join("%.1e" % x for x in eerrs2)))
        print("%-14s        main errs %s" % ("", " ".join("%.1e" % x for x in errs)))
    else:
        print("%-14s        main errs %s" % ("", " ".join("%.1e" % x for x in errs)))
