"""Screen pinned low-storage source maps and emit qualified full-RK coefficients."""
import argparse
from functools import lru_cache
import json
import math
from pathlib import Path
import pprint
import numpy as np

ROOT=Path(__file__).resolve().parents[3]
SOURCE=Path(__file__).with_name('oracle.json')
MODULE=ROOT/'src/warpSPHIntegrators/_expanded_rk_data.py'
SCREEN=Path(__file__).with_name('screen.json')
TOLERANCE=1e-10


@lru_cache(None)
def trees(n):
    if n==1:return ((),)
    prior=[(k,t) for k in range(1,n) for t in trees(k)];out=[]
    def fill(left,start,children):
        if not left:out.append(tuple(children));return
        for i in range(start,len(prior)):
            size,tree=prior[i]
            if size<=left:fill(left-size,i,children+[tree])
    fill(n-1,0,[]);return tuple(out)


def tree_order(tree):return 1+sum(tree_order(child) for child in tree)
def density(tree):return tree_order(tree)*math.prod(density(child) for child in tree)


def order_defects(a,b,order):
    @lru_cache(None)
    def phi(tree):
        return np.prod([a@phi(child) for child in tree],axis=0) if tree else np.ones(len(b))
    return [float(max(abs(b@phi(tree)-1/density(tree)) for tree in trees(k))) for k in range(1,order+1)]


def canonical(a,b,c,error=None,*,endpoint=False):
    """Merge identical earlier stages, preserving actual source time nodes."""
    n=len(b);rows=[];times=[];mapping=[]
    for i in range(n):
        row=np.zeros(n)
        for j in range(i):row[mapping[j]]+=a[i,j]
        match=next((j for j,(old,time) in enumerate(zip(rows,times))
            if abs(time-c[i])<2e-14 and np.allclose(old,row,atol=2e-14,rtol=0)),None)
        if match is None:
            match=len(rows);rows.append(row);times.append(float(c[i]))
        mapping.append(match)
    m=len(rows);aa=np.asarray(rows)[:,:m];bb=np.zeros(m);ee=None if error is None else np.zeros(m)
    for i,j in enumerate(mapping):
        bb[j]+=b[i]
        if ee is not None:ee[j]+=error[i]
    if endpoint:
        # Reconstruct physical endpoint exactly from the main weights. Existing
        # source endpoint rows can differ by recurrence rounding only.
        if abs(times[-1]-1)<2e-14 and bb[-1]==0 and np.allclose(aa[-1],bb,atol=2e-14,rtol=0):
            aa[-1]=bb;times[-1]=1.
        else:
            aa=np.pad(aa,((0,1),(0,1)));aa[-1,:-1]=bb;bb=np.r_[bb,0.]
            times.append(1.)
            if ee is not None:ee=np.r_[ee,0.]
    return aa,bb,np.asarray(times),ee,mapping


def main_map(a,b,c):
    """Remove stages unreachable from the main result for equivalence screening."""
    a,b,c,_,_=canonical(a,b,c)
    needed=set(np.flatnonzero(abs(b)>2e-14).tolist())
    for i in range(len(b)-1,-1,-1):
        if i in needed:needed.update(np.flatnonzero(abs(a[i])>2e-14).tolist())
    indices=sorted(needed)
    return a[np.ix_(indices,indices)],b[indices],c[indices]


def generate():
    source=json.loads(SOURCE.read_text());screen={};data={}
    for name,record in sorted(source['methods'].items()):
        a=np.asarray(record['a']);b=np.asarray(record['b']);c=np.asarray(record['c']);p=record['order']
        defects=order_defects(a,b,p)
        row=dict(order=p,source_method=record['source_method'],phase=record['phase'],
            main_order_defects=defects,time_node_defect=float(max(abs(a.sum(1)-c))),
            initial_scale_defect=max(float(max(abs(np.asarray(record['initial_stage_scales'])-1))),abs(record['initial_output_scale']-1)))
        row['qualified']=bool(max([row['time_node_defect'],row['initial_scale_defect'],*defects])<TOLERANCE)
        error=record['error_weights'];paired=False
        if error is not None:
            error=np.asarray(error)
            row['comparison_defects']=order_defects(a,b-error,p)
            row['observed_comparison_order']=next((i for i,v in enumerate(row['comparison_defects']) if v>TOLERANCE),p)
            paired=(max(row['comparison_defects'][:p-1])<TOLERANCE and abs(record['initial_error_scale'])<TOLERANCE)
            row['source_pair_qualified']=bool(paired)
        screen[name]=row
        if not row['qualified']:continue
        aa,bb,cc,ee,mapping=canonical(a,b,c,error if paired else None,endpoint=True)
        data[name]=dict(order=p,estimate_order=p if paired else None,
            a=tuple(tuple(float(x) for x in row) for row in aa),b=tuple(float(x) for x in bb),
            c=tuple(float(x) for x in cc),error_weights=None if ee is None else tuple(float(x) for x in ee))
        row.update(expanded_stages=len(bb),source_to_expanded=mapping)
    payload=dict(source=source['monorepo_tree'],tolerance=TOLERANCE,screen=screen,
        qualified_methods=len(data),qualified_pairs=sum(r['estimate_order'] is not None for r in data.values()))
    header='"""Generated qualified full-stage expansions; edit wave16/generate.py, not this file."""\n'
    module=header+'_METHODS = '+pprint.pformat(data,width=120,sort_dicts=True)+'\n'
    return module,json.dumps(payload,indent=2)+'\n'


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--check',action='store_true');args=parser.parse_args()
    module,screen=generate()
    if args.check:
        assert MODULE.read_text()==module,'generated package coefficients differ'
        assert SCREEN.read_text()==screen,'screen record differs'
        print('Expanded RK coefficients and source screen are current')
    else:
        MODULE.write_text(module);SCREEN.write_text(screen)
        print('Generated',json.loads(screen)['qualified_methods'],'main maps and',json.loads(screen)['qualified_pairs'],'pairs')
