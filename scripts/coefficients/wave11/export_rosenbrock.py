"""Export the pinned six-stage coefficients without runtime JSON/file dependencies."""
import argparse
import json
from pathlib import Path
import pprint

root=Path(__file__).resolve().parents[3]
source=json.loads((Path(__file__).with_name('rosenbrock.json')).read_text())
table=source['tableaus']['RosenbrockW6S4OS']
data={key:tuple(tuple(row) for row in value) if key in ['A','C'] else
      tuple(value) if isinstance(value,list) else value for key,value in table.items()}
text='"""Pinned Rosenbrock-W6S4OS Rodas-form coefficients; see wave11 source oracle."""\n\nW6S4OS = '+pprint.pformat(data,sort_dicts=False,width=100)+'\n'
target=root/'src/warpSPHIntegrators/_rosenbrock_data.py'
parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--check',action='store_true');args=parser.parse_args()
if args.check:
    assert target.read_text()==text,'Regenerate the pinned Rosenbrock data'
    print('Pinned Rosenbrock coefficient export is current')
else:target.write_text(text)
