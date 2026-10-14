"""Regenerate the qualified Nyström coefficient module from its pinned oracle."""
import argparse
import json
from pathlib import Path
import pprint

ROOT=Path(__file__).resolve().parents[3]


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--check',action='store_true')
    args=parser.parse_args()
    data=json.loads(Path(__file__).with_name('nystrom.json').read_text())
    qualified={name:record for name,record in data['tableaus'].items() if name!='RKN4'}
    text=('"""Generated from the pinned wave12/nystrom.jl source constructor oracle.\n\n'
        'RKN4 is excluded: its velocity-dependent map does not achieve order four.\n"""\n\n'
        'SOURCE_TREE = '+repr(data['monorepo_tree'])+'\n\nTABLEAUS = '
        +pprint.pformat(qualified,width=110,sort_dicts=True)+'\n')
    target=ROOT/'src/warpSPHIntegrators/_nystrom_data.py'
    if args.check:
        if target.read_text()!=text:raise SystemExit('Nyström coefficient module is stale')
    else:target.write_text(text)
    print('Qualified Nyström coefficient module matches the pinned oracle')


if __name__=='__main__':main()
