from pathlib import Path
# Wave 8: exact float64 values from the ODE v7.8.1 dump.
# - Verifies the batch-1+2 tableaus already in dirk.py (a, b, b_embed, c).
# - Emits ready-to-paste Python literals for the batch-3 schemes.
# Convention (butcher.py): b tuple = (b_main, b_embed) with b_embed = b_main - btilde
# (ODE's ESDIRKIMEXTableau btilde field IS the error weight, sum ~ 0; the project's
# b[1] is the lower-order solution's weights). c registered = stored c, except where
# the stored c disagrees with the row sum by > 1e-13 (ODE's stored-c execution bug),
# in which case the row sum (the consistent abscissa) is registered.
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

DUMP = str(Path(__file__).resolve().parent / "extract_wave8_out.txt")

def parse_dump(path):
    schemes = {}
    cur = None
    with open(path) as fh:
        for line in fh:
            line = line.rstrip("\n")
            m = re.match(r"^=== (\S+) \(\w+\) ===$", line.strip())
            if m:
                cur = m.group(1)
                schemes[cur] = {}
                continue
            if cur is None or " = " not in line:
                continue
            key, val = line.strip().split(" = ", 1)
            schemes[cur][key] = val.strip()
    return schemes

def pmat(v):
    if not v.startswith("[") or v.startswith("Matrix{"):
        return None
    body = v[1:-1]
    return [[float(x) for x in row.split()] for row in body.split(";")]

def pvec(v):
    if v.startswith("Float64[]") or not v.startswith("["):
        return None
    body = v[1:-1]
    if not body.strip():
        return None
    return [float(x) for x in body.split(",")]

BATCH12 = ["Kvaerno3", "Kvaerno4", "Kvaerno5", "KenCarp3", "KenCarp4", "KenCarp47",
           "KenCarp5", "KenCarp58", "ESDIRK325L2SA", "ESDIRK436L2SA2", "ESDIRK437L2SA",
           "ESDIRK547L2SA2", "ESDIRK54I8L2SA", "ESDIRK659L2SA", "SSPSDIRK2"]
BATCH3 = ["ARS222", "ARS232", "ARS343", "ARS443", "BHR553", "Cash4", "CFNLIRK3",
          "Hairer4", "Hairer42", "SFSDIRK4", "SFSDIRK5", "SFSDIRK6", "SFSDIRK7", "SFSDIRK8"]

dump = parse_dump(DUMP)

from warpSPHIntegrators.dirk import getDIRKTableau

print("=== VERIFY batch1+2 (registered vs computed) ===")
for name in BATCH12:
    d = dump[name]
    a = pmat(d["Ai"])
    b = pvec(d["bi"])
    bt = pvec(d.get("btilde", "Float64[]"))
    c_stored = pvec(d["c"])
    c_rs = [sum(row) for row in a]
    c_bug = max(abs(x - y) for x, y in zip(c_stored, c_rs))
    c_reg = c_rs if c_bug > 1e-13 else c_stored
    t = getDIRKTableau(name)
    import numpy as np
    da = max(abs(x - y) for r1, r2 in zip(t.a, a) for x, y in zip(r1, r2))
    if isinstance(t.b, tuple):
        db = max(abs(x - y) for x, y in zip(t.b[0], b))
        db_e = max(abs(x - y) for x, y in zip(t.b[1], [bi - ti for bi, ti in zip(b, bt)]))
    else:
        db = max(abs(x - y) for x, y in zip(t.b, b))
        db_e = float("nan")
    dc = max(abs(x - y) for x, y in zip(t.c, c_reg))
    status = "OK" if max(da, db, db_e if db_e == db_e else 0.0, dc) == 0.0 else "CHECK"
    print(f"{name:16s} da={da:.3e} db={db:.3e} db_e={db_e:.3e} dc={dc:.3e} cbug={c_bug:.3e} {status}")

print()
print("=== BATCH 3 LITERALS ===")
def lit_rows(rows):
    return ",\n".join("[" + ", ".join(repr(x) for x in r) + "]" for r in rows)

def lit_vec(v):
    return ", ".join(repr(x) for x in v)

for name in BATCH3:
    d = dump[name]
    a = pmat(d["Ai"])
    b = pvec(d["bi"])
    bt = pvec(d.get("btilde", "Float64[]"))
    c_stored = pvec(d["c"])
    c_rs = [sum(row) for row in a]
    c_bug = max(abs(x - y) for x, y in zip(c_stored, c_rs))
    c_reg = c_rs if c_bug > 1e-13 else c_stored
    s = len(b)
    print(f"    elif scheme == '{name}':  # s={s} cbug={c_bug:.3e} pair={'yes' if bt else 'no'}")
    print(f"        a=np.array([\n{lit_rows(a)},\n        ]),")
    if bt:
        be = [bi - ti for bi, ti in zip(b, bt)]
        print(f"        b=(np.array([{lit_vec(b)}]),\n           np.array([{lit_vec(be)}])),")
    else:
        print(f"        b=np.array([{lit_vec(b)}]),")
    print(f"        c=np.array([{lit_vec(c_reg)}]),")
    print()
