"""Validate local supported consumers without running research simulations or CI."""
import argparse,ast,importlib,json,math,os
from pathlib import Path
import sys,time
for name,value in {'NUMBA_CACHE_DIR':'/tmp/integrators-numba-cache','WARP_CACHE_PATH':'/tmp/integrators-warp-cache',
    'MPLCONFIGDIR':'/tmp/integrators-matplotlib','CUDA_VISIBLE_DEVICES':'','OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1'}.items():
    os.environ.setdefault(name,value)

def validate(roots):
    import torch
    from warpSPHBootstrap import bootstrap
    runtime=bootstrap(precision='float64');torch.set_num_threads(1)
    report=dict(device=str(runtime.device),dtype=str(runtime.dtype),cuda_available=torch.cuda.is_available(),imports=[],checks=[])
    files=notebooks=0;bindings=set()
    for root in roots:
        for path in root.rglob('*'):
            if not path.is_file() or path.suffix not in ('.py','.ipynb') or any(p in path.parts for p in ['.git','.tmp','.venv','build','__pycache__']):continue
            raw=path.read_text()
            if 'warpSPHIntegrators' not in raw:continue
            texts=[raw] if path.suffix=='.py' else [''.join(c['source']) for c in json.loads(raw)['cells'] if c['cell_type']=='code']
            if path.suffix=='.py':files+=1
            else:notebooks+=1
            for text in texts:
                if text.lstrip().startswith('%%'):continue
                text='\n'.join('pass' if line.lstrip().startswith(('%','!')) else line for line in text.splitlines())
                try:tree=ast.parse(text)
                except SyntaxError:continue  # IPython magics are inspected separately.
                for node in ast.walk(tree):
                    if isinstance(node,ast.Import):
                        for alias in node.names:
                            if alias.name.startswith('warpSPHIntegrators'):importlib.import_module(alias.name);bindings.add((alias.name,'<module>'))
                    if isinstance(node,ast.ImportFrom) and node.module and node.module.startswith('warpSPHIntegrators'):
                        assert not any(a.name=='*' for a in node.names),str(path)
                        module=importlib.import_module(node.module)
                        for a in node.names:getattr(module,a.name);bindings.add((node.module,a.name))
    report.update(python_files=files,notebooks=notebooks,validated_import_bindings=len(bindings))
    for name in ['warpSPH.configurations.simulationConfig','warpSPH.systems.baseState','warpSPH.systems.compSPH',
        'warpSPH.systems.weaklyCompressible','warpSPH.systems.incompressible','warpSPH.systems.artificialCompressible',
        'warpSPH.runner.runner','warpSPH.io.export']:
        module=importlib.import_module(name);report['imports'].append(dict(module=name,file=module.__file__))
    from warpSPH.configurations.simulationConfig import buildConfig,configurationToDict,dictToConfig
    from warpSPHIntegrators import get_scheme,solve
    from warpSPHIntegrators.schemes import IntegrationSchemeType
    from warpSPHIntegrators.state import deferHostTime
    config,method=buildConfig(device=torch.device('cpu'),dtype=torch.float64,dim=2,nx=8,
        integrationScheme=IntegrationSchemeType.dormandPrince,calibrateNormalization=False)
    stored=configurationToDict(config);decoded=dictToConfig(stored)
    assert decoded.integrationScheme is IntegrationSchemeType.dormandPrince
    assert method.identifier is get_scheme(decoded.integrationScheme).identifier
    report['checks'].append('actual configuration build and serialized enum roundtrip')
    from warpSPH.systems.baseState import BaseParticleState,BaseSystem,BaseSystemUpdate
    x=torch.tensor([[1.,0.],[0.,1.]],dtype=torch.float64);v=torch.zeros_like(x);ones=torch.ones(2,dtype=x.dtype)
    fields=BaseParticleState(positions=x,velocities=v,supports=ones,masses=ones,densities=ones,
        kinds=torch.zeros(2,dtype=torch.int32),materials=torch.zeros(2,dtype=torch.int32),
        UIDs=torch.arange(2),UIDcounter=2)
    initial=BaseSystem(fields)
    def rhs(system,dt,*args,**kwargs):
        return BaseSystemUpdate(system.state.velocities,-system.state.positions),system.adjacency,system.state
    first=solve(IntegrationSchemeType.rungeKutta4,initial,.05,rhs,dt=.025)
    result=solve(IntegrationSchemeType.rungeKutta4,None,.1,rhs,restart=first.checkpoint)
    assert result.successful and torch.allclose(result.state.state.positions,x*math.cos(.1),rtol=0,atol=1e-9)
    assert torch.equal(initial.state.positions,x)
    assert torch.equal(result.state.state.UIDs,fields.UIDs)
    report['checks'].append('actual BaseSystem typed updates, finalize and in-memory restart on oscillator')
    initial.t=torch.tensor(0.,dtype=x.dtype)
    with deferHostTime():clocked=get_scheme(IntegrationSchemeType.rungeKutta4)(initial,.01,rhs)
    assert isinstance(clocked.state.t,torch.Tensor) and abs(clocked.state.t.item()-.01)<1e-15
    report['checks'].append('actual BaseSystem deferred tensor clock hook on CPU')
    report['qualification']='Actual local CPU consumer imports/configuration/state/finalize/restart/clock. Integrator import edges are checked in Python and parseable notebook cells; research notebooks are not executed. No CUDA graph, GPU physics, PESPH or frontend feature integration claim.'
    return report

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--dev-root',type=Path,default=Path.home()/'dev');a=p.parse_args();start=time.perf_counter()
    result=validate([a.dev_root/name for name in ['warpSPH','warpSPHCore','warpSPHCases','warpSPHPlotting']]);result['seconds']=time.perf_counter()-start
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n')
    print(len(result['checks']),'actual consumer behavior checks;',result['validated_import_bindings'],'import bindings')
