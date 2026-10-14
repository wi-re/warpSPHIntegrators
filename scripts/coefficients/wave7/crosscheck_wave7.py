from pathlib import Path
#!/usr/bin/env python3
# Wave-7 one-step cross-check (Python side). Runs each assembled Butcher tableau
# for one step of the SAME 2D nonlinear non-autonomous problem as onestep_wave7.jl
# and compares against ODE's result (onestep_wave7_out.txt). Must agree to ~1e-14.
import json
import math

dt = 0.1
t0 = 0.0
y0 = [0.5, -0.3]


def f(t, y):
    return [y[0] ** 2 * math.cos(t) + y[1] * math.sin(t) + 0.1 * t,
            -y[0] * y[1] + math.exp(-t) * y[0] - 0.5 * y[1] ** 3]


def rk_step(A, c, b):
    s = len(c)
    k = []
    for i in range(s):
        ti = t0 + c[i] * dt
        yi = [y0[d] + dt * sum(A[i][j] * k[j][d] for j in range(i)) for d in range(2)]
        k.append(f(ti, yi))
    return [y0[d] + dt * sum(b[i] * k[i][d] for i in range(s)) for d in range(2)]


# ODE results (name -> (v1, v2))
ode = {}
for line in open(str(Path(__file__).resolve().parent / "onestep_wave7_out.txt")):
    parts = line.split()
    if len(parts) == 3 and parts[1] != "ERROR":
        ode[parts[0]] = (float(parts[1]), float(parts[2]))

tableaus = json.load(open(str(Path(__file__).resolve().parent / "wave7_tableaus.json")))

print(f"{'method':10s} {'|dy| (mine vs ODE)':>22s}  status")
all_ok = True
for name in ["TanYam7", "TsitPap8", "DP8", "Feagin10", "Feagin12", "Feagin14",
             "Vern6", "Vern7", "Vern8", "Vern9", "RKV76IIa"]:
    tb = tableaus[name]
    mine = rk_step(tb["A"], tb["c"], tb["b_main"])
    if name not in ode:
        print(f"{name:10s}  ODE missing")
        all_ok = False
        continue
    ref = ode[name]
    diff = max(abs(mine[0] - ref[0]), abs(mine[1] - ref[1]))
    ok = diff < 1e-12
    if not ok:
        all_ok = False
    status = "OK" if ok else "MISMATCH"
    print(f"{name:10s} {diff:22.3e}  {status}   mine={[f'{v:.15e}' for v in mine]}")

print("\nALL MATCH" if all_ok else "\nSOME MISMATCH")
raise SystemExit(0 if all_ok else 1)
