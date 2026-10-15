"""Compile formal source-kernel traces into operator-valued RK definitions."""
import argparse,json,math,pprint
from pathlib import Path
import sympy as sp
root=Path(__file__).resolve().parents[3]
source=json.loads(Path(__file__).with_name('exponential_programs.json').read_text())
z=sp.Symbol('Z');phi={}
def parse(x):
    if isinstance(x,str):return sp.Symbol(x)
    if not isinstance(x,list):return sp.Float(x,17)
    op,*a=x
    if op=='phi':
        k,arg=a;c=sp.simplify(parse(arg)/z);assert not c.free_symbols
        key=(int(k),float(c));name=sp.Symbol(f'P{len(phi)}')
        if key not in phi:phi[key]=name
        return phi[key]
    a=list(map(parse,a))
    return {'+':lambda:a[0]+a[1],'-':lambda:a[0]-a[1],'*':lambda:a[0]*a[1],'/':lambda:a[0]/a[1]}[op]()
def emit(x):
    if x==z:return ['z']
    if x in phi.values():
        key=next(k for k,v in phi.items() if v==x);return ['phi',*key]
    if x.is_Number:return float(x)
    if x.is_Add:return ['add',*map(emit,x.args)]
    if x.is_Mul:return ['mul',*map(emit,x.args)]
    if x.is_Pow and x.args[1].is_Integer and x.args[1]>=0:return ['pow',emit(x.args[0]),int(x.args[1])]
    raise ValueError(x)
def compile_expr(x,n):
    expr=sp.expand(parse(x));variables=[sp.Symbol('Y')]+[sp.Symbol(f'F{i}') for i in range(n)]
    coeff=[sp.expand(expr.coeff(v)) for v in variables]
    assert sp.expand(expr-sum(c*v for c,v in zip(coeff,variables)))==0
    assert not (set().union(*(c.free_symbols for c in coeff)) & set(variables))
    return [emit(c) for c in coeff]
programs={}
for method,program in source.items():
    n=len(program['calls']);calls=[]
    for i,call in enumerate(program['calls']):calls.append(dict(c=call['time'],coefficients=compile_expr(call['state'],i)))
    programs[method]=dict(calls=calls,output=compile_expr(program['output'],n),error=None if program['error'] is None else compile_expr(program['error'],n))
text='"""Operator-valued definitions extracted from actual pinned ExpRK kernels."""\nPROGRAMS = '+pprint.pformat(programs,width=115,sort_dicts=True)+'\n'
target=root/'src/warpSPHIntegrators/_exponential_data.py'
p=argparse.ArgumentParser();p.add_argument('--check',action='store_true');args=p.parse_args()
if args.check:assert target.read_text()==text
else:target.write_text(text)
print(f'{len(programs)} source operator programs compiled')
