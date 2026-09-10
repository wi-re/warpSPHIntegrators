#!/usr/bin/env python
"""TVD classifier for every registered scheme (NOTES S3.11).

Runs the Phase 13 TVD classification over the whole scheme registry:

1. the rigorous SSP (convex-combination) coefficient of each scheme's stage
   maps, from the Fourier stage-map analysis of the 1D periodic first-order
   upwind advection model problem (``tvd_analysis.convex_combination_cfl``),
   for every scheme that exposes its tableau;
2. the measured per-step TVD CFL from a trajectory sweep of the registered
   driver on a step initial condition (``tvd_analysis.measure_tvd_cfl``), for
   every scheme that integrates a first-order system.

The verdict per scheme is the classification recorded in NOTES S3.11; the
table below is the maintained fact the roadmap's Phase 13 gate refers to
(re-run this script to refresh it -- the numbers are measured, not asserted
by name). The Verlet family and Newmark integrate second-order systems, so
the TVD question does not apply to them.

Usage:
    python scripts/tvd_classifier.py [--n 64] [--cfls 0.25,0.5,0.75,1.0,1.5,2.0,3.0,5.0]
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))

from warpSPHIntegrators import testing, tvd_analysis
from warpSPHIntegrators.integration import IntegrationSchemes


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--n', type=int, default=64,
                        help='grid points of the advection model problem')
    parser.add_argument('--cfls', type=str,
                        default='0.25,0.5,0.75,1.0,1.25,1.5,2.0,3.0,5.0',
                        help='ascending CFL grid for the trajectory sweep')
    parser.add_argument('--mu-max', type=float, default=4.0,
                        help='upper bound for the convex-combination scan '
                             '(a pass at mu_max is reported as unconditional)')
    args = parser.parse_args()
    cfls = [float(x) for x in args.cfls.split(',')]
    assert cfls == sorted(cfls), 'CFL grid must be ascending'

    problem = testing.advection_problem(n=args.n, ic='step')
    dt_scale = 1.0 / args.n  # h / c with L = 1, c = 1

    print('TVD classification, model problem: 1D periodic first-order upwind '
          f'advection, n={args.n}, step IC (TV0 = 4)')
    print(f'CFL grid: {cfls}; convex-combination scan to CFL {args.mu_max:g}; '
          'burn-in 10, window 40')
    print()
    header = f'{"scheme":28s} {"ord":>3s} {"ssp CFL":>9s} {"tvd CFL":>9s}  verdict'
    print(header)
    print('-' * len(header))
    verdicts = [tvd_analysis.classify_tvd(
        scheme, problem, cfls, dt_scale, mu_max=args.mu_max)
        for scheme in IntegrationSchemes]
    for v in verdicts:
        ssp = '   n/a' if v.ssp_cfl is None else f'{v.ssp_cfl:9.4g}'
        tvd = '   n/a' if v.tvd_cfl is None else f'{v.tvd_cfl:9.4g}'
        print(f'{v.scheme:28s} {v.order:3d} {ssp} {tvd}  {v.verdict}')
    print()
    print('detail per scheme:')
    for v in verdicts:
        if v.tvd_cfl is not None:
            print(f'  {v.scheme:28s} {v.detail}')


if __name__ == '__main__':
    main()
