"""Immutable coupled RK definitions bound to their numerical driver."""
from dataclasses import dataclass
from warpSPHIntegrators.schemes.coefficients.tableau import freeze_tableau
from warpSPHIntegrators.schemes.coefficients.coupled import BlockTableau, getBlockTableau
@dataclass(frozen=True, eq=False)
class CoupledDefinition:
    name: str
    tableau: BlockTableau
    def __post_init__(self):
        object.__setattr__(self,'tableau',freeze_tableau(self.tableau))
    @property
    def __name__(self):return self.name
    @property
    def blockTableau(self):return self.tableau
    def __call__(self,state,dt,f,*args,**kwargs):
        from warpSPHIntegrators.drivers.implicit.coupled import fullyImplicitBlock
        return fullyImplicitBlock(state,dt,f,self.tableau,*args,name=self.name,**kwargs)
def blockScheme(tableau_name,**kwargs):return CoupledDefinition(tableau_name,getBlockTableau(tableau_name))


radauIia2 = blockScheme('radauIia2')
gaussLegendre2 = blockScheme('gaussLegendre2')
gaussLegendre3 = blockScheme('gaussLegendre3')
radauIia3 = blockScheme('radauIia3')
