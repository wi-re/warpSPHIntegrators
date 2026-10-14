from pathlib import Path
#!/usr/bin/env python3
# Wave-7: assemble Butcher tableaus (A, c, b_main, b_embed) from the ODE v7.8.1
# cache dump (extract_wave7_out.txt), using ONLY the data plus the Butcher
# consistency constraint (row_sums(A) == c) to resolve field-name indexing.
#
# Naming families observed in the dump:
#   * 2-3 digit a's  (a21, a108): 1-indexed a{row}{col}   [TanYam7, Vern6/7, RKV76IIa]
#   * 4 digit a's    (a0201):      a{rr}{cc}, 1-indexed   [TsitPap8, DP8]  (no col "00")
#   * 4 digit a's    (a0100):      a{rr}{cc}, 0-indexed   [Feagin10/12/14] (has col "00")
# b / btilde / bh are always 1-indexed (b[i] = weight of stage i).
#
# FSAL resolution (data-driven):
#   * no b fields at all        -> b_main = A[-1]        (Vern6: last row stored as a's)
#   * b fields, no last-row a's -> A[-1] = b_main        (DP8: fsallast row == b)
#   * otherwise                 -> not FSAL (b_main from b, A[-1] from a's)
#
# Embedded (b_embed must sum to 1, the (p-1)-order solution weights):
#   * btilde fields (sum to 0, the DIFFERENCE) -> b_embed = b_main - btilde
#   * bh fields (sum to 1, the lower solution) -> b_embed = bh
#   * neither                               -> plain (no embedded branch)
import json
import re
import sys

DUMP = str(Path(__file__).resolve().parent / "extract_wave7_out.txt")


def parse_dump(path):
    methods = {}
    cur = None
    for line in open(path):
        line = line.rstrip("\n")
        m = re.match(r"^=== (\w+) \(", line)
        if m:
            cur = m.group(1)
            methods[cur] = {}
            continue
        m = re.match(r"^  (\w+) = (.+)$", line)
        if m and cur is not None:
            methods[cur][m.group(1)] = m.group(2)
    return methods


def a_field_digits(name):
    """Return the digit string of an a-field ('a21'->'21', 'a0201'->'0201'), or None."""
    if not name.startswith("a"):
        return None
    rest = name[1:]
    if rest.isdigit():
        return rest
    return None


def parse_a(row_col, s):
    """Resolve a 2-3 digit a-name to a 1-indexed (row, col)."""
    for i in range(1, len(row_col)):
        row = int(row_col[:i])
        col = int(row_col[i:])
        if 2 <= row <= s and 1 <= col < row:
            return row, col
    raise ValueError(f"cannot parse a-field with digits {row_col!r} for s={s}")


def classify(fields):
    """Return ('short'|'wide', '1index'|'0index') for the a-field naming."""
    digits = [a_field_digits(n) for n in fields]
    digits = [d for d in digits if d is not None]
    if not digits:
        raise ValueError("no a-fields")
    if all(len(d) in (2, 3) for d in digits):
        return "short"
    if all(len(d) == 4 for d in digits):
        # 0-indexed iff some column field is "00"
        if any(d.endswith("00") for d in digits):
            return ("wide", "0index")
        return ("wide", "1index")
    raise ValueError(f"mixed a-field widths: {[set(len(d) for d in digits)]}")


def parse_a_name(name, s, conv):
    d = a_field_digits(name)
    if d is None:
        return None
    if conv == "short":
        return parse_a(d, s)
    if conv == ("wide", "0index"):
        r, c = int(d[:2]), int(d[2:])
        return r + 1, c + 1
    if conv == ("wide", "1index"):
        r, c = int(d[:2]), int(d[2:])
        return r, c
    raise ValueError(conv)


def b_kind(name):
    m = re.fullmatch(r"btilde(\d+)", name)
    if m:
        return "btilde", int(m.group(1))
    m = re.fullmatch(r"bh(\d+)", name)
    if m:
        return "bh", int(m.group(1))
    m = re.fullmatch(r"b(\d+)", name)
    if m:
        return "b", int(m.group(1))
    return None


def assemble(name, fields, s):
    conv = classify(fields)
    A = [[0.0] * s for _ in range(s)]
    last_row_has_a = False
    for fn, val in fields.items():
        rc = parse_a_name(fn, s, conv)
        if rc is None:
            continue
        row, col = rc
        if row > s:
            continue  # dense-output / extra stage beyond the update tableau (e.g. DP8 rows 14-16)
        if not (1 <= row <= s and 1 <= col <= s):
            raise ValueError(f"{name}: a-field {fn} -> out-of-range {rc}")
        A[row - 1][col - 1] = float(val)
        if row == s:
            last_row_has_a = True

    b_main = [0.0] * s
    btilde = {}
    bh = {}
    has_b = False
    for fn, val in fields.items():
        k = b_kind(fn)
        if k is None:
            continue
        kind, idx = k
        if not (1 <= idx <= s):
            raise ValueError(f"{name}: b-field {fn} index {idx} out of range for s={s}")
        if kind == "b":
            b_main[idx - 1] = float(val)
            has_b = True
        elif kind == "btilde":
            btilde[idx - 1] = float(val)
        elif kind == "bh":
            bh[idx - 1] = float(val)

    fsal = False
    if not has_b:
        # FSAL: propagated weights are the last row (stored as a's)
        if not last_row_has_a:
            raise ValueError(f"{name}: no b fields and no last-row a's")
        b_main = list(A[-1])
        fsal = True
    elif not last_row_has_a:
        # FSAL: last row == b, not stored as a's
        A[-1] = list(b_main)
        fsal = True

    c = [sum(row) for row in A]

    b_embed = None
    if name not in PLAIN:
        if btilde:
            b_embed = [b_main[i] - btilde.get(i, 0.0) for i in range(s)]
        elif bh:
            b_embed = [bh.get(i, 0.0) for i in range(s)]

    return A, c, b_main, b_embed, fsal


def check(name, s, A, c, b_main, b_embed, fields):
    errs = []
    # lower-triangular, zero first row
    for i in range(s):
        if i > 0 and c[i - 1] > c[i] + 1e-12:
            pass  # c not required monotone (RKV76IIa has c1=0, c2=0.069... fine)
        for j in range(i, s):
            if abs(A[i][j]) > 1e-15:
                errs.append(f"A[{i}][{j}] = {A[i][j]} not strictly lower-triangular")
    if any(abs(x) > 1e-15 for x in A[0]):
        errs.append("first row of A not zero")
    if abs(c[0]) > 1e-12:
        errs.append(f"c[0] = {c[0]} != 0")
    sb = sum(b_main)
    if abs(sb - 1.0) > 1e-9:
        errs.append(f"sum(b_main) = {sb} != 1")
    if b_embed is not None:
        se = sum(b_embed)
        if abs(se - 1.0) > 1e-9:
            errs.append(f"sum(b_embed) = {se} != 1")
    # cross-check derived c against the dump's explicit c fields (where present)
    dump_c = {}
    for fn, val in fields.items():
        m = re.fullmatch(r"c(\d+)", fn)
        if m:
            dump_c[int(m.group(1))] = float(val)
    if dump_c:
        cs = [round(x, 12) for x in c]
        for k, v in dump_c.items():
            if k > s:
                continue  # dense-output / extra stage (e.g. DP8 c14-c16)
            if round(v, 12) not in cs:
                errs.append(f"dump c{k} = {v} not found in derived c")
    return errs


STAGES = {
    "TanYam7": 10, "TsitPap8": 13, "DP8": 12,
    "Feagin10": 17, "Feagin12": 25, "Feagin14": 35,
    "Vern6": 9, "Vern7": 10, "Vern8": 13, "Vern9": 16, "RKV76IIa": 10,
}

#:
# Methods with no (p-1)-order embedded solution. DP8 (12-stage, 8th order,
# not FSAL -- ODE recomputes f fresh each step since its last-row a's != b)
# is a dual-error-estimate method: ODE's perform_step combines a 5th-order
# companion (the `er` weights) and a 3rd/4th-order companion (the `btilde`
# weights) for step control. Neither is a (p-1)=7th-order solution, so no
# single b_embed captures its error estimate -- register plain, like the
# Feagin methods and wave-6's SIR54 (NOTES.md 3.27).
PLAIN = {"DP8"}


def main():
    methods = parse_dump(DUMP)
    out = {}
    all_ok = True
    for name, s in STAGES.items():
        fields = methods[name]
        A, c, b_main, b_embed, fsal = assemble(name, fields, s)
        errs = check(name, s, A, c, b_main, b_embed, fields)
        status = "OK" if not errs else "FAIL"
        if errs:
            all_ok = False
        print(f"{name:10s} s={s:2d} fsal={str(fsal):5s} embedded={'yes' if b_embed is not None else 'no ':3s} {status}")
        for e in errs:
            print(f"   ! {e}")
        out[name] = {
            "s": s, "fsal": fsal,
            "A": A, "c": c, "b_main": b_main,
            "b_embed": b_embed,
        }
    with open(str(Path(__file__).resolve().parent / "wave7_tableaus.json"), "w") as f:
        json.dump(out, f, indent=1)
    print(f"\nWrote wave7_tableaus.json ({len(out)} tableaus)")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
