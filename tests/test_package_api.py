"""Architectural boundaries, extension contracts and stored-data migration."""
import ast,base64,dataclasses,json,pickle,io
from pathlib import Path
from types import SimpleNamespace
import pytest
import torch
import warpSPHIntegrators as package
from warpSPHIntegrators import get_scheme, solve, TensorState, tensor_rhs
from warpSPHIntegrators.schemes import IntegrationSchemeType, ExplicitRK, DIRK, CoupledRK, IMEX, AdaptiveAdams, SchemeMetadata, IntegrationSchemes
from warpSPHIntegrators.state import Stepper
from warpSPHIntegrators.runtime import load_checkpoint
from warpSPHIntegrators.wrappers import AlgorithmSwitch, Strang, ConservativeDissipativeRHS
ROOT=Path(__file__).parents[1]


def test_root_exposes_only_ordinary_entry_points_and_definitions_are_immutable():
    assert set(package.__all__)=={'get_scheme','solve','TensorState','tensor_rhs'}
    assert not any(hasattr(package,name) for name in ['RungeKutta4','Tsit5','DIRK','getIntegrator','BaseState'])
    assert isinstance(IntegrationSchemes,tuple)
    for identifier in [ExplicitRK.rungeKutta4,DIRK.esdirk436l2sa,CoupledRK.radauIia2,IMEX.ark324l2sa]:
        scheme=get_scheme(identifier)
        assert {scheme: identifier}[get_scheme(identifier)] is identifier
        definition=scheme.definition
        assert dataclasses.is_dataclass(definition)
        with pytest.raises(dataclasses.FrozenInstanceError):definition.name='changed'
        coefficient=definition.tableau.a if hasattr(definition.tableau,'a') else definition.tableau.a_explicit
        with pytest.raises(ValueError):coefficient.flat[0]=42.
        with pytest.raises(ValueError):coefficient.setflags(write=True)


def test_real_old_enum_pickle_and_all_numeric_identities_survive():
    old=json.loads((ROOT/'tests/data/enum-storage-before-refactor.json').read_text())
    assert {v.name:v.value for v in IntegrationSchemeType}==old['enums']
    assert pickle.loads(base64.b64decode(old['pickle'])) is IntegrationSchemeType.dormandPrince
    from warpSPHIntegrators.schemes.enums import IntegrationSchemeType as stored_type
    assert stored_type is IntegrationSchemeType


def test_real_old_portable_adams_checkpoint_continues_identically():
    old=json.loads((ROOT/'tests/data/adams-before-refactor.json').read_text())
    cp=load_checkpoint(io.BytesIO(base64.b64decode(old['checkpoint_base64'])))
    rhs=tensor_rhs(lambda t,y:-y+.1*y*y-.1*torch.exp(y.new_tensor(-2*t)))
    result=solve(AdaptiveAdams(),None,1.,rhs,restart=cp)
    assert result.successful and result.state.t==old['t']
    assert result.state.y.tolist()==pytest.approx(old['output'],rel=0,abs=2e-14)


@dataclasses.dataclass(frozen=True)
class Extension:
    name: str='unregistered extension'
    order: int=4
    implicit: bool=False
    fsal: bool=False
    metadata: object=SchemeMetadata()
    @property
    def identifier(self):return SimpleNamespace(name='test.extension.v1')
    def __call__(self,state,dt,rhs,*args,**kwargs):return get_scheme(ExplicitRK.rungeKutta4)(state,dt,rhs,*args,**kwargs)


def test_structural_extension_runs_switches_and_composes_without_type_registration():
    method=Extension();assert isinstance(method,Stepper) and get_scheme(method) is method
    y=torch.ones(1,dtype=torch.float64);rhs=tensor_rhs(lambda t,y:-y)
    for selected in [method,AlgorithmSwitch((method,ExplicitRK.dormandPrince),lambda state,previous:0,'extension')]:
        result=solve(selected,TensorState(y),.2,rhs,dt=.02)
        assert result.successful and abs(result.state.y.item()-torch.exp(y.new_tensor(-.2)).item())<1e-9
    physical=tensor_rhs(ConservativeDissipativeRHS(lambda t,y:-y,lambda t,y:-2*y))
    result=solve(Strang(method,method),TensorState(y),.2,physical,dt=.02)
    assert result.successful and abs(result.state.y.item()-torch.exp(y.new_tensor(-.6)).item())<1e-8


def test_lower_layers_do_not_import_selection_or_run_orchestration():
    root=ROOT/'src/warpSPHIntegrators'
    for folder in ['state','solvers','drivers','schemes/coefficients']:
        for path in (root/folder).rglob('*.py'):
            for node in ast.walk(ast.parse(path.read_text())):
                if isinstance(node,ast.ImportFrom):
                    assert node.module not in ['warpSPHIntegrators.schemes.registry','warpSPHIntegrators.runtime.run'],str(path)
                    if folder=='schemes/coefficients' and node.module:
                        assert not node.module.startswith('warpSPHIntegrators.drivers'),str(path)
