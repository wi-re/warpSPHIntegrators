#!/usr/bin/env python
"""Convergence study of first-stage reuse (``priorStep``) for a given scheme.

Reusing the last stage of step *n* as the first stage (``k0``) of step *n+1* saves one
right-hand-side evaluation per step. It is a standard trick -- CRKSPH, for instance,
does exactly this for its second-order scheme -- but its validity depends on the
tableau, not on the caller.

This script measures the cost: it integrates a problem with an analytic solution at a
sequence of step sizes, with and without reuse, and reports the measured convergence
order for each. The *prediction* it checks against comes from the library itself
(``integrators.step_reuse_order``), so this script is a verification tool for that
prediction rather than a second implementation of it.

Usage
-----
    python scripts/step_reuse_convergence.py                      # default: RK4
    python scripts/step_reuse_convergence.py --scheme 'SSP RK3'
    python scripts/step_reuse_convergence.py --all                # summary table
    python scripts/step_reuse_convergence.py --problem forced --plot reuse.png

Run inside the ``warp`` conda environment.
"""

from __future__ import annotations

import argparse
import math
import warnings

from warpSPHIntegrators import getIntegrator, testing
from warpSPHIntegrators.integration import IntegrationSchemes
from warpSPHIntegrators.reuse import step_reuse_analysis


def study(scheme, problem, dts, T, quiet=False):
    analysis = step_reuse_analysis(scheme)

    if not quiet:
        print(f'Scheme   : {scheme.name} (registered order {scheme.order})')
        print(f'Problem  : {problem.description}')
        print(f'Integrate to T = {T}, float64\n')
        if analysis.order is None:
            print(f'Reuse    : NOT APPLICABLE -- {analysis.reason}\n')
        else:
            verdict = ('lossless (FSAL)' if analysis.fsal
                       else f'loses {scheme.order - analysis.order} order(s)'
                       if analysis.order < scheme.order else 'lossless')
            print(f'Reuse    : {analysis.reason}')
            print(f'           predicted order with reuse: {analysis.order} -- {verdict}\n')

    rows = []
    errs_off, errs_on = [], []
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        for dt in dts:
            e_off = testing.final_error(scheme, problem, dt, T, reuse=False)
            try:
                e_on = testing.final_error(scheme, problem, dt, T, reuse=True)
            except Exception as exc:
                e_on = None
                if not quiet:
                    print(f'  reuse raised {type(exc).__name__}: {exc}')
            errs_off.append(e_off)
            errs_on.append(e_on)
            rows.append((dt, e_off, e_on))

    if not quiet:
        print(f'{"dt":>9}{"error (no reuse)":>20}{"p":>7}'
              f'{"error (reuse)":>18}{"p":>7}{"reuse / no reuse":>19}')
        print('-' * 80)
        for i, (dt, e_off, e_on) in enumerate(rows):
            p_off = (f'{math.log(rows[i-1][1] / e_off) / math.log(rows[i-1][0] / dt):.2f}'
                     if i and e_off > 1e-13 and rows[i - 1][1] > 1e-13 else '--')
            if e_on is None:
                print(f'{dt:>9.5f}{e_off:>20.4e}{p_off:>7}{"ERROR":>18}{"":>7}{"":>19}')
                continue
            p_on = (f'{math.log(rows[i-1][2] / e_on) / math.log(rows[i-1][0] / dt):.2f}'
                    if i and rows[i - 1][2] and e_on > 1e-13 and rows[i - 1][2] > 1e-13 else '--')
            ratio = f'{e_on / e_off:>10.1f}x' if e_off > 0 else ''
            print(f'{dt:>9.5f}{e_off:>20.4e}{p_off:>7}{e_on:>18.4e}{p_on:>7}{ratio:>19}')

    o_off = testing.measured_order(errs_off, dts)
    o_on = testing.measured_order(errs_on, dts)
    if not quiet:
        print('-' * 80)
        print(f'measured order   no reuse: {o_off:.2f}' if o_off else 'measured order   no reuse: n/a')
        if o_on is not None:
            tag = ''
            if analysis.order is not None:
                tag = f'   (predicted {analysis.order}: ' \
                      f'{"MATCH" if abs(o_on - analysis.order) < 0.25 else "MISMATCH"})'
            print(f'                    reuse: {o_on:.2f}{tag}')
        else:
            print('                    reuse: n/a')
        if o_off and o_on and o_on < o_off - 0.25:
            worst = max((on / off) for on, off in zip(errs_on, errs_off) if on and off)
            print(f'\n  >> Reuse costs {o_off - o_on:.1f} orders of convergence here; '
                  f'the error is up to {worst:.0f}x larger at the same step size.')
        elif o_off and o_on:
            print('\n  >> Reuse is safe for this scheme: convergence order is retained.')
    return analysis, o_off, o_on, rows


def plot(rows, scheme, problem, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    dts = [r[0] for r in rows]
    fig, ax = plt.subplots(figsize=(6.5, 5))
    ax.loglog(dts, [r[1] for r in rows], 'o-', label='no reuse')
    if all(r[2] is not None for r in rows):
        ax.loglog(dts, [r[2] for r in rows], 's-', label='priorStep reuse')
    ref = rows[-1][1]
    for p, style in ((scheme.order, '--'), (max(scheme.order - 1, 1), ':')):
        ax.loglog(dts, [ref * (d / dts[-1]) ** p for d in dts], style, color='0.6',
                  lw=1, label=f'$O(\\Delta t^{{{p}}})$')
    ax.set_xlabel('$\\Delta t$')
    ax.set_ylabel('|error| at $T$')
    ax.set_title(f'{scheme.name} -- {problem.name}')
    ax.legend()
    ax.grid(True, which='both', alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    print(f'\nwrote {path}')


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--scheme', default='RK4', help="registered scheme name, e.g. 'SSP RK3'")
    ap.add_argument('--problem', default='oscillator', choices=sorted(testing.PROBLEMS))
    ap.add_argument('--all', action='store_true', help='summary table over every scheme')
    ap.add_argument('-T', type=float, default=2.0, help='integration end time')
    ap.add_argument('--dt', type=float, default=0.1, help='coarsest step size')
    ap.add_argument('--levels', type=int, default=4, help='number of halvings')
    ap.add_argument('--plot', metavar='PATH', help='write a log-log convergence plot')
    args = ap.parse_args()

    problem = testing.PROBLEMS[args.problem]()
    dts = testing.default_step_sizes(args.dt, args.levels)

    if args.all:
        print(f'Problem: {problem.description}   T={args.T}\n')
        print(f'{"scheme":<32}{"order":>6}{"no reuse":>10}{"reuse":>8}'
              f'{"predicted":>11}   note')
        print('-' * 96)
        for scheme in IntegrationSchemes:
            analysis, o_off, o_on, _ = study(scheme, problem, dts, args.T, quiet=True)
            pred = '-' if analysis.order is None else str(analysis.order)
            if analysis.order is None:
                note = 'reuse refused/ignored'
            elif analysis.fsal:
                note = 'FSAL, exact'
            elif o_on and o_off and o_on > o_off - 0.25:
                note = 'lossless'
            elif o_on is not None and o_on < analysis.order - 0.25:
                note = 'DEGRADED + unstable'
            else:
                note = f'DEGRADED to {pred}'
            print(f'{scheme.name:<32}{scheme.order:>6}'
                  f'{(f"{o_off:.2f}" if o_off else "n/a"):>10}'
                  f'{(f"{o_on:.2f}" if o_on else "n/a"):>8}{pred:>11}   {note}')
        return

    scheme = getIntegrator(args.scheme)
    _, _, _, rows = study(scheme, problem, dts, args.T)
    if args.plot:
        plot(rows, scheme, problem, args.plot)


if __name__ == '__main__':
    main()
