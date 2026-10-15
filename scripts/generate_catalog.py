"""Generate registry catalog data and maintained documentation counts.

Run `python scripts/generate_catalog.py` after registration changes; `--check`
fails on stale files. Authored equations, stability qualifications and numerical
source records remain in the family pages.
"""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import re
import sys

from warpSPHIntegrators.schemes import FAMILY_ENUMS, IntegrationSchemes


ROOT = Path(__file__).resolve().parents[1]
FAMILY_PAGES = {
    'ExplicitRK': ('Explicit RK', 'explicit-rk'),
    'Symplectic': ('Symplectic', 'symplectic'),
    'RelaxedChebyshev': ('Relaxed Chebyshev', 'relaxed-chebyshev'),
    'MultistepExplicit': ('Multistep explicit', 'multistep-explicit'),
    'DIRK': ('DIRK', 'dirk'),
    'CoupledRK': ('Coupled RK', 'coupled-rk'),
    'MultistepImplicit': ('Multistep implicit', 'multistep-implicit'),
    'IMEX': ('IMEX', 'imex'),
    'LinearlyImplicit': ('Linearly implicit', 'rosenbrock'),
    'Exponential': ('Exponential', 'exponential'),
    'Newmark': ('Newmark', 'newmark'),
}

# Documentation labels, keyed by the stable public enum spelling. Author names
# and constructions follow the cited family pages and coefficient source notes.
# Keep legacy lookup names in snapshot()['schemes'][...]['name']; a friendly
# label is presentation, not a new get_scheme string alias.
FRIENDLY_NAMES = {
    'rungeKutta2': 'Explicit midpoint (Runge–Kutta 2)',
    'rungeKutta3': 'Classical Runge–Kutta 3',
    'rungeKutta4': 'Classical Runge–Kutta 4',
    'rungeKutta4alt': 'Runge–Kutta 4 (3/8 rule)',
    'sspRK3': 'Shu–Osher SSP Runge–Kutta 3',
    'ssprk104': 'SSP Runge–Kutta (10 stages, order 4)',
    'kykssprk42': 'Kubatko–Yeager–Ketcheson SSP RK2 (4 stages)',
    'kyk2014dgssprk3s2': 'Kubatko–Yeager–Ketcheson DG SSP RK2 (3 stages)',
    'ssprk43': 'SSP Runge–Kutta 3(2) (4 stages, comparison 1)',
    'ssprk432': 'SSP Runge–Kutta 3(2) (4 stages, comparison 2)',
    'ssprk932': 'SSP Runge–Kutta 3 (9 stages)',
    'ssprk53_2n1': 'Higueras–Roldán SSP RK3 (2N1 variant)',
    'ssprk53_2n2': 'Higueras–Roldán SSP RK3 (2N2 variant)',
    'ssprk53_h': 'Higueras–Roldán SSP RK3 (H variant)',
    'nystrom5th': 'Nyström Runge–Kutta 5',
    'bogackiShampine': 'Bogacki–Shampine 3(2)',
    'dormandPrince': 'Dormand–Prince 5(4)',
    'cashKarp': 'Cash–Karp 5(4)',
    'tsit5': 'Tsitouras 5(4)',
    'owrenzen3': 'Owren–Zennaro 3(2)',
    'owrenzen4': 'Owren–Zennaro 4(3)',
    'owrenzen5': 'Owren–Zennaro 5(4)',
    'bs5': 'Bogacki–Shampine 5(4)',
    'stefanov5': 'Stepanov 5(4) (7 stages)',
    'arushina2': 'Alshina–Zaks–Kalitkin 2(1)',
    'arushina3': 'Alshina–Zaks–Kalitkin 3',
    'arushina6': 'Alshina–Zaks–Kalitkin 6',
    'msrk5': 'Stepanov 5 (9 stages)',
    'msrk6': 'Stepanov 6 (9 stages)',
    'sir54': 'Kovalnogov–Simos–Tsitouras 5 (SIR method)',
    'raston4': 'Ralston Runge–Kutta 4',
    'rko65': 'Runge–Kutta–Oliver 5 (6 stages)',
    'rkm': 'Mead–Renaut Runge–Kutta 3',
    'psrk4p7q6': 'Pseudo-symplectic Runge–Kutta 4 (6 stages)',
    'psrk3p6q5': 'Pseudo-symplectic Runge–Kutta 3 (5 stages)',
    'psrk3p5q4': 'Pseudo-symplectic Runge–Kutta 3 (4 stages)',
    'rkv76iia': 'Verner 7(6) (RKV76.IIa)',
    'tanyam7': 'Tanaka–Muramatsu–Yamashita 7(6)',
    'tsitpap8': 'Tsitouras–Papakostas 8(7)',
    'dp8': 'Dormand–Prince 8',
    'leapFrog': 'Leapfrog (synchronized)',
    'pefrl': 'Forest–Ruth 4 (position-first, PEFRL)',
    'vefrl': 'Forest–Ruth 4 (velocity-first, VEFRL)',
    'epec': 'Evaluate–predict–evaluate–correct (midpoint)',
    'epecModified': 'Evaluate–predict–evaluate–correct (Heun)',
    'tvdRK2': 'TVD Runge–Kutta 2 (Shu–Osher)',
    'tvdRK3': 'TVD Runge–Kutta 3 (Shu–Osher)',
    'sdirk2': 'Singly diagonally implicit Runge–Kutta 2',
    'trbdf2': 'Trapezoidal rule–BDF2 (TR-BDF2)',
    'sspsdirk2': 'SSP singly diagonally implicit Runge–Kutta 2',
    'bhr553': 'Boscarino–Russo 3 (BHR553, implicit part)',
    'cash4': 'Cash SDIRK 4(3)',
    'cfnlirk3': 'Calvo–de Frutos–Novo 3 (implicit part)',
    'hairer4': 'Hairer–Wanner SDIRK 4(3) (variant 1)',
    'hairer42': 'Hairer–Wanner SDIRK 4(3) (variant 2)',
    'mebdf2': 'Modified extended backward differentiation 2',
    'ssprkmsvs32': 'SSP multistep (3 steps, order 2)',
    'ssprkmsvs43': 'SSP multistep (4 steps, order 3)',
    'imexEuler': 'Implicit–explicit Euler',
    'sbdf2': 'Semi-implicit backward differentiation 2',
    'sbdf3': 'Semi-implicit backward differentiation 3',
    'cnab2': 'Crank–Nicolson–Adams–Bashforth 2',
    'cnlf2': 'Crank–Nicolson–leapfrog 2',
    'rkc1': 'Runge–Kutta–Chebyshev 1',
    'rkc2': 'Runge–Kutta–Chebyshev 2',
    'rkl2': 'Runge–Kutta–Legendre 2',
    'ros3p': 'Rosenbrock–W 3(2) (ROS3P)',
    'etd2rk': 'Exponential time-differencing Runge–Kutta 2',
    'exprb32': 'Exponential Rosenbrock 3(2)',
    'newmark': 'Newmark (average acceleration by default)',
}
for stages, order in ((5, 3), (6, 3), (7, 3), (8, 3), (5, 4)):
    FRIENDLY_NAMES[f'ssprk{stages}{order}'] = f'SSP Runge–Kutta ({stages} stages, order {order})'
for order in (6, 7, 8, 9):
    FRIENDLY_NAMES[f'vern{order}'] = f'Verner {order}({order-1})'
for order in (10, 12, 14):
    FRIENDLY_NAMES[f'feagin{order}'] = f'Feagin Runge–Kutta {order}'
for order in (3, 4, 5):
    FRIENDLY_NAMES[f'kvaerno{order}'] = f'Kværnø {order}({order-1})'
for code, pair, stages in (('3', '3(2)', 4), ('4', '4(3)', 6), ('47', '4(3)', 7),
                           ('5', '5(4)', 8), ('58', '5(4)', 8)):
    FRIENDLY_NAMES[f'kencarp{code}'] = f'Kennedy–Carpenter {pair} ({stages} stages, KenCarp{code})'
for code, pair, stages, variant in (
        ('324l2sa', '3(2)', 4, ''), ('436l2sa', '4(3)', 6, ''),
        ('325l2sa', '3(2)', 5, ''), ('436l2sa2', '4(3)', 6, ', variant 2'),
        ('437l2sa', '4(3)', 7, ''), ('547l2sa2', '5(4)', 7, ', variant 2'),
        ('54i8l2sa', '5(4)', 8, ', I variant'), ('659l2sa', '6(5)', 9, '')):
    FRIENDLY_NAMES[f'esdirk{code}'] = f'Explicit-first-stage SDIRK {pair} ({stages} stages{variant})'
for code in ('222', '232', '343', '443'):
    FRIENDLY_NAMES[f'ars{code}'] = f'Ascher–Ruuth–Spiteri (ARS{code}, implicit part)'
for stages in range(4, 9):
    FRIENDLY_NAMES[f'sfstdirk{stages}'] = f'Ferracina–Spijker SDIRK 4 ({stages} stages)'
for stages in (2, 3):
    FRIENDLY_NAMES[f'gaussLegendre{stages}'] = f'Gauss–Legendre collocation ({stages} stages)'
    FRIENDLY_NAMES[f'radauIia{stages}'] = f'Radau IIA collocation ({stages} stages)'
for code, pair, stages in (('324l2sa', '3(2)', 4), ('436l2sa', '4(3)', 6)):
    FRIENDLY_NAMES[f'ark{code}'] = f'Kennedy–Carpenter additive RK {pair} ({stages} stages)'
for order in range(1, 6):
    FRIENDLY_NAMES[f'bdf{order}'] = f'Backward differentiation formula {order}'


def snapshot():
    families = []
    for family in FAMILY_ENUMS:
        label, page = FAMILY_PAGES[family.__name__]
        families.append({'name': family.__name__, 'label': label, 'page': page,
                         'count': sum(s.family is family for s in IntegrationSchemes)})
    schemes = []
    for s in IntegrationSchemes:
        meta = asdict(s.metadata)
        for key in ('required_capabilities', 'optional_capabilities', 'required_options'):
            meta[key] = sorted(meta[key])
        schemes.append({'name': s.name,
                        'friendly_name': FRIENDLY_NAMES.get(s.identifier.name, s.name),
                        'identifier': s.identifier.name,
                        'family': s.family.__name__, 'order': s.order,
                        'implicit': s.implicit, 'stability': s.stability,
                        'fsal': s.fsal, 'reuse_order': s.reuse_order, 'steps': s.steps,
                        'startup_order': s.startup_order, **meta})
    return {'summary': {
        'scheme_count': len(schemes), 'family_count': len(families),
        'estimate_count': sum(s.has_error_estimate for s in IntegrationSchemes),
        'max_order': max(s.order for s in IntegrationSchemes),
    }, 'families': families, 'schemes': schemes}


def summary_text(data):
    s = data['summary']
    return (f"**{s['scheme_count']} registered schemes across {s['family_count']} families**, "
            f"with orders through {s['max_order']} and {s['estimate_count']} estimate emitters.")


def region(text, name, body):
    start, end = f'[catalog:{name}:start]: #', f'[catalog:{name}:end]: #'
    if start not in text or end not in text:
        raise ValueError(f'missing catalog region {name}')
    before, after = text.split(start, 1)
    _, tail = after.split(end, 1)
    return before + start + '\n\n' + body.rstrip() + '\n\n' + end + tail


def catalog_markdown(data):
    lines = ['---', 'title: Scheme Catalog', '---', '', '# Scheme catalog', '',
             'Generated by `scripts/generate_catalog.py` from the runtime registry.',
             '', summary_text(data), '',
             'Stored stages include endpoint rows. A driver without a fixed stage count',
             'shows `driver` or its per-step stage options; a nonzero null-stage',
             'companion weight adds one RHS call outside the coupled solve. These are stage counts, not total',
             'RHS evaluations or solver work. The estimate exponent q is the value to',
             'pass to `propose_dt`, and can differ from the propagated order.', '',
             'An empty required-RHS set permits a combined plain callable. IMEX splits',
             'are optional; KenCarp/ARS DIRKs supply their pure implicit tableaus.',
             'Stability labels and reuse qualifications are explained in the family pages.', '',
             'Friendly names spell out authors and method constructions. The enum column',
             'gives the exact family member accepted by `get_scheme`, for example',
             '`get_scheme(ExplicitRK.tsit5)` for Tsitouras 5(4). Family members share',
             'their values with `IntegrationSchemeType`. Existing lookup strings keep',
             'their original spelling; these documentation labels do not add string aliases.', '']
    for family in data['families']:
        lines += [f"## {family['label']} — {family['count']} schemes", '',
                  f"[Equations and qualifications](families/{family['page']}).", '',
                  '| Friendly name | Enum member | Order | Stages | Companion order | Estimate q | Stability | Reuse order | Required RHS / options | State / history |',
                  '|---|---|---:|---|---:|---:|---|---|---|---|']
        for s in data['schemes']:
            if s['family'] != family['name']:
                continue
            stages = str(s['stage_count']) if s['stage_count'] is not None else 'driver'
            if s['estimate_extra_stages']:
                stages += f" + {s['estimate_extra_stages']} companion"
            if s['required_options']:
                stages = 'variable'
            requirements = ', '.join(s['required_capabilities']) or 'combined'
            if s['required_options']:
                requirements += '; ' + ' or '.join(f'`{opt}=`' for opt in s['required_options'])
            reuse = str(s['reuse_order']) if s['reuse_order'] is not None else 'refused'
            if s['fsal']:
                reuse += ' (FSAL)'
            state = 'position/velocity' if s['state_kind'] == 'second_order' else 'tensor'
            if s['requires_separable']:
                state += '; separability assumptions'
            if s['history_length']:
                state += f"; {s['history_length']} previous entries"
            lines.append(f"| {s['friendly_name']} | `{s['family']}.{s['identifier']}` | {s['order']} | {stages} | "
                         f"{s['embedded_order'] if s['embedded_order'] is not None else '—'} | "
                         f"{s['estimate_order'] if s['estimate_order'] is not None else '—'} | "
                         f"{s['stability'] or '—'} | {reuse} | {requirements} | {state} |")
        lines.append('')
    return '\n'.join(lines)


def readme_catalog(data):
    """Compact complete front-page registry, kept beside the authored guidance."""
    lines = []
    geometric = {'Semi-Implicit Euler', 'Symplectic Euler', 'Leap Frog',
                 'Velocity Verlet', 'PEFRL', 'VEFRL'}
    for family in data['families']:
        lines += [f"### {family['label']} ({family['count']})", '',
                  '| Friendly name | Enum member | Order | Stored stages | Estimate | Stability / structure |',
                  '|---|---|---:|---|---|---|']
        for s in data['schemes']:
            if s['family'] != family['name']:
                continue
            stages = str(s['stage_count']) if s['stage_count'] is not None else 'driver'
            if s['required_options']:
                stages = 'variable'
            if s['estimate_extra_stages']:
                stages += f" + {s['estimate_extra_stages']}"
            estimate = (f"{s['order']}({s['embedded_order']}), q={s['estimate_order']}"
                        if s['estimate_order'] is not None else '—')
            if s['fsal']:
                estimate += '; FSAL'
            properties = [s['stability']] if s['stability'] else []
            if properties and s['family'] == 'IMEX':
                properties = [f"implicit part: {s['stability']}"]
            elif properties and s['family'] == 'Exponential':
                properties = [f"linear test: {s['stability']}"]
            elif s['family'] == 'Newmark':
                properties = ['bounded linear oscillator (default parameters)']
            if s['name'] in geometric:
                properties.append('symplectic for separable Hamiltonians')
            elif s['name'] == 'Implicit Midpoint' or s['name'].startswith('Gauss-Legendre'):
                properties.append('symplectic with converged solves')
            elif s['name'] == 'Newmark' or s['name'].startswith('Trapezoidal'):
                properties.append('linear oscillator area preservation')
            if s['name'].startswith(('SSP', 'TVD')):
                properties.append('SSP under step/history bounds' if s['history_length']
                                  else 'SSP under its CFL bound')
            if s['history_length']:
                properties.append('history required')
            lines.append(f"| {s['friendly_name']} | `{s['family']}.{s['identifier']}` | {s['order']} | {stages} | {estimate} | "
                         f"{'; '.join(properties) or '—'} |")
        lines.append('')
    return '\n'.join(lines)


def artifacts(root=ROOT):
    data = snapshot()
    result = {
        Path('docs/src/data/catalog.json'): json.dumps(data, indent=2) + '\n',
        Path('docs/docs/catalog.md'): catalog_markdown(data),
    }
    for page in ('README.md', 'docs/docs/index.md', 'docs/docs/selection.md',
                 'docs/development/usage-reference.md', 'docs/docs/tensor-ode.md', 'docs/docs/quickstart.md'):
        text = (root / page).read_text()
        if '[catalog:summary:start]' in text:
            text = region(text, 'summary', summary_text(data))
        if page == 'README.md' and '[catalog:schemes:start]' in text:
            text = region(text, 'schemes', readme_catalog(data))
        # Included examples are displayed verbatim.
        if page in ('README.md', 'docs/docs/index.md', 'docs/docs/tensor-ode.md', 'docs/docs/quickstart.md'):
            text = re.sub(r'(\[example:include ([^\n]+)\]: #\n```python\n).*?^```',
                          lambda m: m[1] + (root / m[2]).read_text() + '```',
                          text, flags=re.M | re.S)
        if page == 'docs/docs/index.md':
            text = re.sub(r'## The \d+ families',
                          f"## The {data['summary']['family_count']} families", text)
            rows = ['| Family | Enum | Schemes | Page |', '|---|---|---:|---|']
            for family in data['families']:
                rows.append(f"| {family['label']} | `{family['name']}` | {family['count']} | "
                            f"[{family['page']}](/docs/families/{family['page']}) |")
            text = region(text, 'families', '\n'.join(rows))
        if page == 'docs/docs/tensor-ode.md':
            count = sum(s['state_kind'] == 'tensor' for s in data['schemes'])
            text = re.sub(r'supports the \d+ catalog records',
                          f'supports the {count} catalog records', text)
        if page in ('README.md', 'docs/development/usage-reference.md', 'docs/docs/tensor-ode.md'):
            n, e = data['summary']['scheme_count'], data['summary']['estimate_count']
            text = re.sub(r'\b\d+ registered schemes across \d+ families',
                          f"{n} registered schemes across {data['summary']['family_count']} families", text)
            text = re.sub(r'\b(all |contains |captures all )(\d+)( schemes| estimate emitters)',
                          lambda m: m[1] + str(e if 'emitters' in m[3] else n) + m[3], text)
            text = re.sub(r'Runge-Kutta up to \d+th order',
                          f"Runge-Kutta up to {data['summary']['max_order']}th order", text)
            dirks = [s for s in data['schemes'] if s['family'] == 'DIRK']
            dirk_count = len(dirks)
            reuse_count = sum(s['reuse_order'] is not None and s['reuse_order'] >= s['order']
                              for s in dirks)
            text = re.sub(r'\d+ DIRKs including', f'{reuse_count} DIRKs including', text)
            text = re.sub(r'for all \d+ methods', f'for all {dirk_count} methods', text)
            text = re.sub(r'(`DIRK` \| )\d+ methods:',
                          lambda m: m[1] + f'{dirk_count} methods:', text)
            text = re.sub(r'\d+ DIRK schemes through order \d+',
                          f"{dirk_count} DIRK schemes through order {max(s['order'] for s in dirks)}", text)
        result[Path(page)] = text
    api = (root / 'docs/docs/api.md').read_text()
    api = re.sub(r'every registered scheme \(\d+\)',
                 f"every registered scheme ({data['summary']['scheme_count']})", api)
    result[Path('docs/docs/api.md')] = api
    for family in data['families']:
        path = Path(f"docs/docs/families/{family['page']}.md")
        text = (root / path).read_text()
        count = family['count']
        text = re.sub(r'^(# .*? — )\d+ schemes?$',
                      lambda m: m[1] + f"{count} {'scheme' if count == 1 else 'schemes'}",
                      text, count=1, flags=re.M)
        result[path] = text
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    stale = []
    for relative, content in artifacts().items():
        path = ROOT / relative
        if path.exists() and path.read_text() == content:
            continue
        stale.append(str(relative))
        if not args.check:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content)
    if args.check and stale:
        print('Stale catalog files: ' + ', '.join(stale), file=sys.stderr)
        return 1
    print('Catalog is current' if args.check else f'Updated {len(stale)} catalog files')
    return 0


if __name__ == '__main__':
    sys.exit(main())
