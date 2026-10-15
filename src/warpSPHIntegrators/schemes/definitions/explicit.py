


from dataclasses import dataclass
from warpSPHIntegrators.schemes.coefficients.tableau import freeze_tableau
from warpSPHIntegrators.schemes.coefficients.explicit import getButcherTableau
from warpSPHIntegrators.schemes.coefficients.tableau import ButcherTableau as butcherTableau
@dataclass(frozen=True, eq=False)
class ExplicitDefinition:
    name: str
    tableau: butcherTableau
    def __post_init__(self):
        object.__setattr__(self,'tableau',freeze_tableau(self.tableau))
    @property
    def __name__(self):return self.name
    @property
    def butcherTableau(self):return self.tableau
    def __call__(self,state,dt,f,*args,**kwargs):
        from warpSPHIntegrators.drivers.runge_kutta import RungeKuttaB
        return RungeKuttaB(state,dt,f,self.tableau,*args,**kwargs)
def butcherScheme(tableau_name,**kwargs):
    return ExplicitDefinition(tableau_name,getButcherTableau(tableau_name,**kwargs))


# Evaluate-Predict-Evaluate-Correct (EPEC) scheme based on pySPH code
# is equivalent to the traditional explicit midpoint method
EPEC = butcherScheme('midpoint')
# Modified EPEC scheme based on pySPH code, uses $y^{n+1} = y^n + \frac{\Delta t}{2}\left( F(y^n) + F(y^{n+\frac{1}{2}}) \right)$
EPECmodified = butcherScheme('heunsMethod')


forwardEuler    = butcherScheme('forwardEuler')
RungeKutta2     = butcherScheme('midpoint')
midPoint        = butcherScheme('midpoint')
heunsMethod     = butcherScheme('heunsMethod')
ralston2nd      = butcherScheme('ralston')
RungeKutta3     = butcherScheme('RK3')
heunsMethod3rd  = butcherScheme('Heun3')
ralston3rd      = butcherScheme('ralston3')
Wray3rd         = butcherScheme('Wray3')
SSPRK3           = butcherScheme('SSPRK3')
SSPRK104         = butcherScheme('SSPRK104')
SSPRK53          = butcherScheme('SSPRK53')
SSPRK63          = butcherScheme('SSPRK63')
SSPRK73          = butcherScheme('SSPRK73')
SSPRK83          = butcherScheme('SSPRK83')
SSPRK54          = butcherScheme('SSPRK54')
KYKSSPRK42       = butcherScheme('KYKSSPRK42')
KYK2014DGSSPRK3S2 = butcherScheme('KYK2014DGSSPRK_3S2')
SSPRK43          = butcherScheme('SSPRK43')
SSPRK432         = butcherScheme('SSPRK432')
SSPRK932         = butcherScheme('SSPRK932')
SSPRK53_2N1      = butcherScheme('SSPRK53_2N1')
SSPRK53_2N2      = butcherScheme('SSPRK53_2N2')
SSPRK53_H        = butcherScheme('SSPRK53_H')
RungeKutta4      = butcherScheme('RK4')
RungeKutta4alt  = butcherScheme('RK4alt')
Nystrom5th      = butcherScheme('Nystrom5')
BogackiShampine = butcherScheme('BogackiShampine')
DormandPrince   = butcherScheme('DormandPrince')
CashKarp        = butcherScheme('CashKarp')
# The OrdinaryDiffEq wave-6 tableaus (SCHEME_EXPANSION_PLAN, 2026-10-13):
# Tsit5 and the LowOrderRK workhorses (embedded pairs first, then fixed-step).
Tsit5           = butcherScheme('Tsit5')
OwrenZen3       = butcherScheme('OwrenZen3')
OwrenZen4       = butcherScheme('OwrenZen4')
OwrenZen5       = butcherScheme('OwrenZen5')
BS5             = butcherScheme('BS5')
Stepanov5       = butcherScheme('Stepanov5')
Alshina2        = butcherScheme('Alshina2')
Alshina3        = butcherScheme('Alshina3')
Alshina6        = butcherScheme('Alshina6')
MSRK5           = butcherScheme('MSRK5')
MSRK6           = butcherScheme('MSRK6')
SIR54           = butcherScheme('SIR54')
Ralston4        = butcherScheme('Ralston4')
RKO65           = butcherScheme('RKO65')
RKM             = butcherScheme('RKM')
PSRK4p7q6       = butcherScheme('PSRK4p7q6')
PSRK3p6q5       = butcherScheme('PSRK3p6q5')
PSRK3p5q4       = butcherScheme('PSRK3p5q4')
# The OrdinaryDiffEq wave-7 high-order tableaus (SCHEME_EXPANSION_PLAN,
# 2026-10-13): the Verner family, Tsitouras's high-order pairs, DP8, and the
# Feagin symmetric methods.
Vern6           = butcherScheme('Vern6')
Vern7           = butcherScheme('Vern7')
Vern8           = butcherScheme('Vern8')
Vern9           = butcherScheme('Vern9')
RKV76IIa        = butcherScheme('RKV76IIa')
TanYam7         = butcherScheme('TanYam7')
TsitPap8        = butcherScheme('TsitPap8')
DP8             = butcherScheme('DP8')
Feagin10        = butcherScheme('Feagin10')
Feagin12        = butcherScheme('Feagin12')
Feagin14        = butcherScheme('Feagin14')
