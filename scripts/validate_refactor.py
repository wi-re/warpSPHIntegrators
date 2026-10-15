"""Check numerical AST preservation across the package refactor against local Git."""
import argparse,ast,hashlib,json,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];BASELINE='7d12f0b'

class NumericalAST(ast.NodeTransformer):
    def visit_Import(self,node):return None
    def visit_ImportFrom(self,node):return None
    def visit_Expr(self,node):
        if isinstance(node.value,ast.Constant) and isinstance(node.value.value,str):return None
        return self.generic_visit(node)


def find(text,name,owner=None):
    tree=ast.parse(text)
    nodes=tree.body if owner is None else next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name==owner).body
    return next(n for n in nodes if isinstance(n,ast.FunctionDef) and n.name==name)

def normalized(node):
    node=NumericalAST().visit(node);node.name='numerical_function'
    return ast.dump(node,include_attributes=False)


def validate():
    rows=[];pairs=[]
    def add(old,target,names):
        pairs.extend((old,name,None,target,name) for name in names)
    add('jfnk','solvers/krylov',['gmres','fd_matvec','jvp_matvec','_implicit_diff_reattach'])
    add('butcher','drivers/runge_kutta',['RungeKuttaB','_RungeKuttaBOnce','_weighted_update','_error_estimate'])
    add('butcher','schemes/coefficients/explicit',['getButcherTableau'])
    add('dirk','schemes/coefficients/implicit',['getDIRKTableau'])
    add('dirk','drivers/implicit/dirk',['DIRK','_dirk_reuses_first_stage','_default_norm'])
    add('fullyimplicit','schemes/coefficients/coupled',['getBlockTableau'])
    add('fullyimplicit','drivers/implicit/coupled',['fullyImplicitBlock'])
    add('ark','schemes/coefficients/additive',['getARKTableau'])
    add('ark','drivers/implicit/ark',['ARK','_additive_weighted_update','_additive_error_estimate'])
    add('rkc','schemes/coefficients/chebyshev',['_rkc1_coeffs','_rkc2_coeffs','_rkl2_coeffs','stage_count','_recurrence_clock_weights'])
    add('rkc','drivers/stabilized/chebyshev',['integrateRelaxedChebyshev','_reference_lincomb'])
    factories={'collocation':('Collocation','implicit'),'variable_adams':('VariableAdams','history'),'adaptive_adams':('AdaptiveAdams','history'),
        'ndf':('NDF','history'),'adaptive_ndf':('AdaptiveNDF','history'),'stabilized':('StabilizedRK','stabilized'),
        'orthogonal':('OrthogonalRK','stabilized'),'nystrom':('Nystrom','mechanics'),'improved_nystrom':('ImprovedNystrom','mechanics'),
        'generalized_alpha':('GeneralizedAlpha','mechanics'),'parallel_dirk':('ParallelDIRK','implicit'),'linearized':('Rosenbrock','implicit'),
        'exponential_rk':('ExponentialRK','exponential'),'extrapolation':('Extrapolation','extrapolation'),
        'expanded_rk':('ExpandedRK','explicit'),'function_map':('FunctionMap','maps')}
    for module,(cls,group) in factories.items():pairs.append((module,'__call__',cls,'drivers/'+group+'/'+module,'step'))
    for old,name,owner,target,newname in pairs:
        source=subprocess.check_output(['git','show',f'{BASELINE}:src/warpSPHIntegrators/{old}.py'],cwd=ROOT,text=True)
        current=(ROOT/'src/warpSPHIntegrators'/f'{target}.py').read_text()
        a=normalized(find(source,name,owner));b=normalized(find(current,newname))
        assert a==b,f'Numerical statements changed: {old}.{owner or ""}.{name} -> {target}.{newname}'
        rows.append(dict(before=f'{old}.{owner+"." if owner else ""}{name}',after=f'{target}.{newname}',numerical_ast_sha256=hashlib.sha256(a.encode()).hexdigest()))
    return dict(baseline=BASELINE,functions=rows,qualification='Exact Python AST comparison after removing import statements and docstrings and normalizing only the outer function name. All numerical statements, constants, calls, arguments and control flow are retained. Binding, registry guards, coefficient ownership and public API changes are gated separately by tests; this is not a whole-program equivalence proof.')

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path);a=p.parse_args();report=validate()
    if a.output:a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,indent=2)+'\n')
    print(len(report['functions']),'numerical functions match the pre-refactor AST')
