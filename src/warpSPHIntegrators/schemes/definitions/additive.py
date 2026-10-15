"""Immutable additive RK definitions bound to their numerical driver."""
from dataclasses import dataclass
from warpSPHIntegrators.schemes.coefficients.tableau import freeze_tableau
from warpSPHIntegrators.schemes.coefficients.additive import AdditiveTableau, getARKTableau
@dataclass(frozen=True, eq=False)
class AdditiveDefinition:
    name: str
    tableau: AdditiveTableau
    def __post_init__(self):
        object.__setattr__(self,'tableau',freeze_tableau(self.tableau))
    @property
    def __name__(self):return self.name
    @property
    def arkTableau(self):return self.tableau
    def __call__(self,state,dt,f,*args,**kwargs):
        from warpSPHIntegrators.drivers.implicit.ark import ARK
        return ARK(state,dt,f,self.tableau,*args,name=self.name,**kwargs)
def arkScheme(tableau_name,**kwargs):return AdditiveDefinition(tableau_name,getARKTableau(tableau_name))


ARK324L2SA = arkScheme('ARK324L2SA')
ARK436L2SA = arkScheme('ARK436L2SA')
