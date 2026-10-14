#!/usr/bin/env python3
"""Run the CPU merge gate and retain timings, failures and environment metadata.

Run in the documented test environment, e.g.:
    python scripts/validate_test_gate.py --mode shards --parallel-shards 3

The shard expressions are the same inventory consumed by CI and collection.
Concurrent shard timings include contention; use --parallel-shards 1 for isolated
measurements. Reports and full subprocess output are retained in --output-dir.
"""

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
import hashlib
from importlib.metadata import version
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

import numpy
import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]


def gate_fingerprint():
    """Identify tested code/config, including new files absent from git diff."""
    digest = hashlib.sha256()
    paths = [ROOT / 'pyproject.toml', ROOT / '.github/test-shards.json',
             ROOT / '.github/workflows/tests.yml', Path(__file__).resolve()]
    paths += list((ROOT / 'src').rglob('*.py')) + list((ROOT / 'tests').rglob('*.py'))
    for path in sorted(paths):
        digest.update(str(path.relative_to(ROOT)).encode() + b'\0')
        digest.update(path.read_bytes() + b'\0')
    return digest.hexdigest()


def case_counts(path):
    return Counter((case.get('classname'), case.get('name'))
                   for case in ET.parse(path).getroot().iter('testcase'))


def run_gate(label, expression, workers, output_dir, env):
    log_path = output_dir / f'{label}.txt'
    xml_path = output_dir / f'{label}.xml'
    command = [sys.executable, '-m', 'pytest', '-q', '-n', str(workers), '--dist=worksteal',
               '--durations=20', f'--junitxml={xml_path}']
    if expression is not None:
        command += ['-m', expression]
    start = time.perf_counter()
    with log_path.open('w') as log:
        process = subprocess.run(command, cwd=ROOT, env=env, stdout=log,
                                 stderr=subprocess.STDOUT, check=False)
    elapsed = time.perf_counter() - start
    record = {'label': label, 'expression': expression, 'command': command,
              'seconds': round(elapsed, 3), 'exit_code': process.returncode,
              'log': str(log_path), 'failures': []}
    if xml_path.exists():
        suites = ET.parse(xml_path).getroot().findall('testsuite')
        for key in ('tests', 'failures', 'errors', 'skipped'):
            record[f'{key}_count'] = sum(int(suite.get(key, 0)) for suite in suites)
        for suite in suites:
            for case in suite.findall('testcase'):
                if case.find('failure') is not None or case.find('error') is not None:
                    record['failures'].append(f'{case.get("classname")}::{case.get("name")}')
    print(f'{label}: exit {process.returncode}, {elapsed:.1f}s ({log_path})', flush=True)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['full', 'shards', 'all'], default='all')
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--parallel-shards', type=int, default=1)
    parser.add_argument('--output-dir', type=Path, default=ROOT / '.tmp/test-gate')
    parser.add_argument('--compare-full', type=Path,
                        help='verify shard case IDs against an existing full-run JUnit report')
    args = parser.parse_args()
    if args.workers < 1 or args.parallel_shards < 1:
        parser.error('worker and shard counts must be positive')
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        env[name] = '1'
    # The merge gate exercises CPU precision even on hosts with a GPU installed.
    env['CUDA_VISIBLE_DEVICES'] = ''
    report = {
        'started_utc': datetime.now(timezone.utc).isoformat(),
        'python': sys.version, 'torch': torch.__version__, 'numpy': numpy.__version__,
        'pytest': pytest.__version__, 'device': 'cpu', 'platform': platform.platform(),
        'package': version('warpSPHIntegrators'),
        'git_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT,
                                             text=True).strip(),
        'gate_sha256': gate_fingerprint(),
        'workers_per_run': args.workers, 'parallel_shards': args.parallel_shards,
        'numerical_threads_per_worker': 1, 'runs': [],
    }
    start = time.perf_counter()
    if args.mode in ('full', 'all'):
        report['runs'].append(run_gate('full', None, args.workers, output_dir, env))
    if args.mode in ('shards', 'all'):
        shards = json.loads((ROOT / '.github/test-shards.json').read_text())
        with ThreadPoolExecutor(max_workers=args.parallel_shards) as pool:
            futures = [pool.submit(run_gate, f'shard-{i+1}', expression,
                                   args.workers, output_dir, env)
                       for i, expression in enumerate(shards)]
            report['runs'].extend(future.result() for future in futures)
        full_path = args.compare_full
        if full_path is None and args.mode == 'all':
            full_path = output_dir / 'full.xml'
        if full_path is not None:
            actual = Counter()
            for i in range(len(shards)):
                xml_path = output_dir / f'shard-{i+1}.xml'
                if xml_path.exists():
                    actual.update(case_counts(xml_path))
            expected = case_counts(full_path)
            report['partition_complete'] = actual == expected and all(
                count == 1 for count in actual.values())
            report['partition_case_count'] = len(actual)
            print(f'Shard partition complete: {report["partition_complete"]} '
                  f'({len(actual)} cases)', flush=True)
    report['seconds'] = round(time.perf_counter() - start, 3)
    path = output_dir / 'report.json'
    path.write_text(json.dumps(report, indent=2) + '\n')
    print(f'Report: {path}', flush=True)
    return int(any(run['exit_code'] != 0 for run in report['runs'])
               or report.get('partition_complete') is False)


if __name__ == '__main__':
    sys.exit(main())
