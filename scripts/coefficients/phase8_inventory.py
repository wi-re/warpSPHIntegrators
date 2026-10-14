"""Enumerate wave 10/11 algorithm declarations from the pinned monorepo.

This records source names/order declarations, not coefficient verification or
method equivalence. Macro-generated types and public aliases are retained.
Pass the same source root whose complete tree is checked by wave9/oracle.jl.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import tomllib

PACKAGES={
    10:['AdamsBashforthMoulton','BDF','Nordsieck','AMF','IMEXMultistep'],
    11:['StabilizedRK','Rosenbrock','ExponentialRK','Newmark'],
}
DRIVERS={
    'AdamsBashforthMoulton':'history/coefficient/controller work, not plain tableaus',
    'BDF':'history/order/nonlinear-solver work; D-prefixed algorithms include DAEs',
    'Nordsieck':'Nordsieck history/step/order transformations',
    'AMF':'wrapper over another algorithm with approximated matrix factorization',
    'IMEXMultistep':'split history and startup work',
    'StabilizedRK':'recurrence/stage-selection coefficients and stability/cost gate',
    'Rosenbrock':'linearized stage driver; tableau compatibility must be checked per method',
    'ExponentialRK':'phi/Jacobian stage formulas and Krylov error/cost work',
    'Newmark':'second-order residual/weighted-time driver, not a plain RK tableau',
}


def inventory(root):
    output=[]
    for wave,packages in PACKAGES.items():
        for suffix in packages:
            package=root/'lib'/('OrdinaryDiffEq'+suffix)
            if not package.exists():
                output.append(dict(wave=wave,package=package.name,status='absent from pinned source'))
                continue
            files=[p for p in package.joinpath('src').glob('*.jl')
                   if p.name in {'algorithms.jl','alg_utils.jl','amf.jl'}]
            names=set();orders={};hashes={}
            for path in sorted(files):
                source=path.read_text();hashes[str(path.relative_to(root))]=hashlib.sha256(path.read_bytes()).hexdigest()
                if path.name in {'algorithms.jl','amf.jl'}:
                    names.update(re.findall(r'^\s*(?:Base\.@kwdef\s+)?(?:mutable\s+)?struct\s+([A-Za-z]\w*)',source,re.M))
                    for loop in re.findall(r'for Alg in \[([^\]]+)\]',source):
                        names.update(re.findall(r':([A-Za-z]\w*)',loop))
                    names.update(re.findall(r'^\s*\(?\s*:([A-Za-z]\w*)\s*,',source,re.M))
                    # Tuple entries can put the type symbol on its own line.
                    names.update(re.findall(r'^\s*:([A-Za-z]\w*)\s*,',source,re.M))
                    if suffix=='Rosenbrock' and re.search(r'^Tsit5DA\(',source,re.M):names.add('Tsit5DA')
                for name,value in re.findall(r'alg_order\(alg::([A-Za-z]\w*)\)\s*=\s*([^\n]+)',source):
                    orders[name]=value.strip()
            orders={name:value for name,value in orders.items() if name in names}
            output.append(dict(wave=wave,package=package.name,
                version=tomllib.loads((package/'Project.toml').read_text())['version'],
                algorithms=sorted(names),source_order_declarations=orders,
                implementation_class=DRIVERS[suffix],source_sha256=hashes))
    return dict(monorepo_version='7.8.1',monorepo_tree='bae9b8e5f1d00e51c554648e5094e215164a19f8',
        qualification='Source declaration inventory only; complete root tree verified by wave9/oracle.jl',packages=output,
        corrections=[
            'Variable-step coefficients concern step/history ratios; non-autonomous RHS works with fixed-step AB/ABM too.',
            'MEBDF2 means modified extended BDF, not modified-energy BDF.',
            'QNDF refers to quasi-constant timesteps for NDF; it is not shorthand for quasi-Newton.',
            'ABM54 order five does not by its name certify an embedded fourth-order error estimate.',
            'Pinned AMF means Approximate Matrix Factorization for Rosenbrock-W, not Adams-Moulton with a frozen operator.',
            'Tsit5DA is a HybridExplicitImplicitRK instance for mass-matrix DAEs; it is not a generic stiff fifth-order Rosenbrock addition.',
            'ETD2 alone is not the whole exponential family: macro-generated ETDRK/EPIRK/Exprb members exist.',
            'GeneralizedAlpha needs weighted second-order acceleration/time residuals and a parameter contract.',
            'OrdinaryDiffEq CompositeAlgorithm switches algorithms; it is not Phase 7 physical flow composition.',
        ])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source_root',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();data=inventory(args.source_root)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(data,indent=2)+'\n')
    for row in data['packages']:print(row['package'],row.get('algorithms',row['status'] if 'status' in row else []))


if __name__=='__main__':main()
