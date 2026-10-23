// Complete, independently resettable demo setups. Never inherit old solver knobs.
export const baseConfig={dt:.04,T:12,backend:'auto',levels:6,stages:9,rtol:1e-6,atol:1e-9,solveTolerance:1e-11,linearSolver:'dense',rhoInfinity:.5,beta:.25,gamma:.5,adaptive:false,stabilityExtent:6,stabilityResolution:512};
export const presets={
 smooth:{problem:'oscillator',mode:'trajectory',methods:['forwardEuler','rungeKutta4','velocityVerlet','gaussLegendre2'],config:{dt:.04,T:12},params:{k:4}},
 stiff:{problem:'stiffRelaxation',mode:'trajectory',methods:['forwardEuler','backwardEuler','trbdf2','radauIia3'],config:{dt:.01,T:2},params:{rate:100,forcing:'sin'}},
 ssp:{problem:'advectionStep',mode:'ssp',methods:['tvdRK3','ssprk104','rungeKutta4','rungeKutta2'],config:{dt:.0125,T:.5},params:{n:64,speed:1}},
 orbit:{problem:'kepler',mode:'geometry',methods:['rungeKutta4','velocityVerlet','pefrl','implicitMidpoint'],config:{dt:.12,T:80}},
 modes:{problem:'structural',mode:'trajectory',methods:['newmark','generalizedAlpha','rungeKutta4','radauIia3'],config:{dt:.05,T:12}},
 history:{problem:'oscillator',mode:'trajectory',methods:['ndf_3','irkn3','irkn4','rungeKutta4'],config:{dt:.05,T:6},params:{k:4}},
 jfnk:{problem:'viscousBurgers',mode:'trajectory',methods:['backwardEuler','trbdf2','ark324l2sa','bdf2'],config:{dt:.01,T:.5,backend:'cpu',linearSolver:'gmres'},params:{n:32,nu:.04}},
};
export const problemSetups={
 oscillator:presets.smooth,stiffRelaxation:presets.stiff,advectionStep:presets.ssp,kepler:presets.orbit,structural:presets.modes,viscousBurgers:presets.jfnk,
 damped:{methods:['rungeKutta4','dormandPrince','trbdf2'],config:{dt:.04,T:12}},
 heated:{methods:['rungeKutta4','dormandPrince','trbdf2'],config:{dt:.04,T:12}},
 forced:{methods:['rungeKutta4','tvdRK3','gaussLegendre2'],config:{dt:.05,T:12}},
 stiffDampedOscillator:{methods:['backwardEuler','trbdf2','radauIia3'],params:{omega:1,c:50},config:{dt:.02,T:3,backend:'cpu'}},
 vanDerPol:{methods:['rungeKutta4','dormandPrince','implicitMidpoint'],config:{dt:.02,T:8}},
 robertson:{methods:['backwardEuler','radauIia3','ros3p'],config:{dt:.001,T:.1,backend:'cpu'}},
 diffusion:{methods:['rungeKutta4','rkc2','trbdf2'],params:{n:32,D:.02},config:{dt:.002,T:.1}},
 advection:{methods:['tvdRK3','rungeKutta4','ssprk104'],params:{n:32,speed:1},config:{dt:.01,T:.5}},
 burgers:{mode:'ssp',methods:['tvdRK3','ssprk104','rungeKutta4'],params:{n:32},config:{dt:.01,T:.25}},
 acoustic:{methods:['rungeKutta4','tvdRK3','dormandPrince'],params:{n:16},config:{dt:.005,T:1}},
 driftOscillator:{methods:['rungeKutta4','dormandPrince','gaussLegendre2'],config:{dt:.02,T:6}},
};
