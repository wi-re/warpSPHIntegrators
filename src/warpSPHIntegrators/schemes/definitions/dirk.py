


from warpSPHIntegrators.solvers.krylov import JFNKSolver


from dataclasses import dataclass
from warpSPHIntegrators.schemes.coefficients.tableau import freeze_tableau
from warpSPHIntegrators.schemes.coefficients.implicit import getDIRKTableau
@dataclass(frozen=True, eq=False)
class DIRKDefinition:
    name: str
    tableau: object
    default_solver: object=None
    def __post_init__(self):
        object.__setattr__(self,'tableau',freeze_tableau(self.tableau))
    @property
    def __name__(self):return self.name
    @property
    def dirkTableau(self):return self.tableau
    def __call__(self,state,dt,f,*args,**kwargs):
        from warpSPHIntegrators.drivers.implicit.dirk import DIRK
        if self.default_solver is not None and kwargs.get('solver') is None:kwargs['solver']=self.default_solver
        return DIRK(state,dt,f,self.tableau,*args,name=self.name,**kwargs)
def dirkScheme(tableau_name,*,default_solver=None,**kwargs):
    return DIRKDefinition(tableau_name,getDIRKTableau(tableau_name),default_solver)


backwardEuler = dirkScheme('backwardEuler')
implicitMidpoint = dirkScheme('implicitMidpoint')
trapezoidal = dirkScheme('trapezoidal')
SDIRK2 = dirkScheme('SDIRK2')
MEBDF2 = dirkScheme('MEBDF2',default_solver=JFNKSolver(tol=1e-12,newton_tol=1e-7))
TRBDF2 = dirkScheme('TRBDF2')
ESDIRK324L2SA = dirkScheme('ESDIRK324L2SA')
ESDIRK436L2SA = dirkScheme('ESDIRK436L2SA')
# Wave 8 (docs/development/plan-reconciliation-2026-10-23.md, 2026-10-13): the remaining ODE v7.8.1
# OrdinaryDiffEqSDIRK tableaus (docs/research/integrators/notes-3-28.md). SDIRK22 is deliberately absent:
# its ODE constructor is a stub (returns Trapezoid), so there is nothing to
# transcribe.
Kvaerno3 = dirkScheme('Kvaerno3')
Kvaerno4 = dirkScheme('Kvaerno4')
Kvaerno5 = dirkScheme('Kvaerno5')
KenCarp3 = dirkScheme('KenCarp3')
KenCarp4 = dirkScheme('KenCarp4')
KenCarp47 = dirkScheme('KenCarp47')
KenCarp5 = dirkScheme('KenCarp5')
KenCarp58 = dirkScheme('KenCarp58')
ESDIRK325L2SA = dirkScheme('ESDIRK325L2SA')
ESDIRK436L2SA2 = dirkScheme('ESDIRK436L2SA2')
ESDIRK437L2SA = dirkScheme('ESDIRK437L2SA')
ESDIRK547L2SA2 = dirkScheme('ESDIRK547L2SA2')
ESDIRK54I8L2SA = dirkScheme('ESDIRK54I8L2SA')
ESDIRK659L2SA = dirkScheme('ESDIRK659L2SA')
SSPSDIRK2 = dirkScheme('SSPSDIRK2')
ARS222 = dirkScheme('ARS222')
ARS232 = dirkScheme('ARS232')
ARS343 = dirkScheme('ARS343')
ARS443 = dirkScheme('ARS443')
BHR553 = dirkScheme('BHR553')
Cash4 = dirkScheme('Cash4')
CFNLIRK3 = dirkScheme('CFNLIRK3')
Hairer4 = dirkScheme('Hairer4')
Hairer42 = dirkScheme('Hairer42')
SFSDIRK4 = dirkScheme('SFSDIRK4')
SFSDIRK5 = dirkScheme('SFSDIRK5')
SFSDIRK6 = dirkScheme('SFSDIRK6')
SFSDIRK7 = dirkScheme('SFSDIRK7')
SFSDIRK8 = dirkScheme('SFSDIRK8')
