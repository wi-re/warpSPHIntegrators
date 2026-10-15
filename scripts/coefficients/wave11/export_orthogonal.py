"""Export the actual pinned ROCK/SERK arrays and RKMC fixed-stage roots."""
import argparse
import json
from pathlib import Path
import pprint

root=Path(__file__).resolve().parents[3]
source=json.loads(Path(__file__).with_name('orthogonal_stabilized.json').read_text())
text='"""Pinned ROCK/SERK coefficients and RKMC roots; source oracle in wave11."""\n\nTABLEAUS = '+pprint.pformat(source['tableaus'],sort_dicts=True,width=110)+'\n'
target=root/'src/warpSPHIntegrators/schemes/coefficients/orthogonal_data.py'
parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--check',action='store_true');args=parser.parse_args()
if args.check:
    assert target.read_text()==text,'Regenerate pinned orthogonal coefficients'
    print('Orthogonal coefficient export is current')
else:target.write_text(text)
