"""Build a wheel and smoke-test it from a temporary directory in a separate venv.

Default: install dependencies into the fresh environment (network needed).
--cpu-torch selects the CI CPU wheel index before installing the project.
--reuse-dependencies is an offline alternative sharing the existing numerical
runtime; project imports must still come entirely from the venv's installed wheel.
"""

import argparse
import json
import os
import shutil
from pathlib import Path
import subprocess
import sys
import tempfile
import tomllib
import venv
import zipfile


ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--reuse-dependencies', action='store_true')
    group.add_argument('--cpu-torch', action='store_true')
    parser.add_argument('--output-dir', type=Path, default=ROOT / '.tmp/wheel-gate')
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    project = tomllib.loads((ROOT / 'pyproject.toml').read_text())['project']
    version = project['version']
    with tempfile.TemporaryDirectory(prefix='integrators-wheel-') as directory:
        work = Path(directory)
        env = os.environ.copy()
        env.pop('PYTHONPATH', None)
        env.pop('PYTHONHOME', None)
        env.update(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
                   CUDA_VISIBLE_DEVICES='', PIP_DISABLE_PIP_VERSION_CHECK='1',
                   PIP_CACHE_DIR=str(work / 'pip-cache'))
        def run(command, cwd=work):
            subprocess.run(command, cwd=cwd, env=env, check=True)
        # Build from a clean source staging tree. Setuptools' persistent build/lib
        # can otherwise retain deleted flat modules after a package refactor.
        staged=work/'source';staged.mkdir()
        shutil.copytree(ROOT/'src',staged/'src',ignore=shutil.ignore_patterns('__pycache__','*.egg-info'))
        for name in ('pyproject.toml','README.md'):
            shutil.copyfile(ROOT/name,staged/name)
        license_name = project['license']['file']
        shutil.copyfile(ROOT/license_name, staged/license_name)
        for license_file in ROOT.glob('LICEN[CS]E*'):
            if license_file.is_file():shutil.copyfile(license_file,staged/license_file.name)
        run([sys.executable, '-m', 'build', '--wheel', '--no-isolation',
             '--outdir', str(work / 'dist'), str(staged)])
        wheel, = (work / 'dist').glob('*.whl')
        with zipfile.ZipFile(wheel) as archive:
            names = archive.namelist()
            assert 'warpSPHIntegrators/py.typed' in names
            assert 'warpSPHIntegrators/adapters/tensor.py' in names
            expected={str(p.relative_to(ROOT/'src')) for p in (ROOT/'src/warpSPHIntegrators').rglob('*.py')}
            assert {n for n in names if n.endswith('.py')}==expected, 'Wheel contains missing or stale Python modules'
            assert not any('__pycache__' in n or n.endswith('.pyc') for n in names)
            license_member, = [n for n in names if n.endswith(f'.dist-info/licenses/{license_name}')]
            assert archive.read(license_member) == (ROOT/license_name).read_bytes()
        target = work / 'venv'
        venv.EnvBuilder(with_pip=True, system_site_packages=args.reuse_dependencies).create(target)
        python = target / ('Scripts/python.exe' if os.name == 'nt' else 'bin/python')
        if args.cpu_torch:
            run([str(python), '-m', 'pip', 'install', 'torch',
                 '--index-url', 'https://download.pytorch.org/whl/cpu'])
        install = [str(python), '-m', 'pip', 'install']
        if args.reuse_dependencies:
            install += ['--no-deps']
        run(install + [str(wheel)])
        smoke = work / 'wheel_smoke.py'
        smoke.write_text((ROOT / 'scripts/wheel_smoke.py').read_text())
        catalog = work / 'catalog.json'
        catalog.write_text((ROOT / 'docs/src/data/catalog.json').read_text())
        run([str(python), '-I', str(smoke), version, str(catalog)])
        for name in ('test_conserved_lagrangian.py', 'test_projected_map.py'):
            shutil.copyfile(ROOT/'tests'/name, work/name)
        # Run the actual custom-state continuation tests against the installed
        # project. The shared-runtime mode already supplies pytest offline.
        if not args.reuse_dependencies:
            run([str(python), '-m', 'pip', 'install', 'pytest'])
        run([str(python), '-I', '-m', 'pytest', '-q', '--confcutdir', str(work),
             str(work/'test_conserved_lagrangian.py'), str(work/'test_projected_map.py')])
        (output / 'report.json').write_text(json.dumps({
            'version':version, 'wheel':wheel.name, 'python':sys.version,
            'dependency_mode':'shared runtime, isolated project install' if args.reuse_dependencies else 'fresh dependency installation',
            'cwd_outside_checkout':True, 'isolated_python':True, 'result':'passed',
            'license_file':license_name,
            'consumer_maps':'installed factory exports, analytic steps, gradients, custom-state in-memory continuation and portable-save rejection',
        }, indent=2)+'\n')
    print(f'Wheel gate passed; report: {output / "report.json"}')


if __name__ == '__main__':
    main()
