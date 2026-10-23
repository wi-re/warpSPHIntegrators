import {methodCFL} from './cfl.js';
const root='src/warpSPHIntegrators/';
// Non-tableau maps use their driver-native representation, including history
// indexing and startup. Parameter-dependent values match the current controls.
export function coefficientRecord(method,catalog,config={}){
 const m=method,id=m.identifier,q=m.order,D=catalog.driverCoefficients;
 let representation,coefficients,notes=[],parameters={},source=root+'schemes/coefficients/';
 if(m.tableau){
  representation={rk:'Butcher tableau',coupled:'Coupled RK tableau',ark:'Additive / IMEX tableau',nystrom:'Nyström tableau',rosenbrock:'Rosenbrock tableau'}[m.kind];coefficients=m.tableau;
  notes.push(m.kind==='nystrom'?'a and b weight h² acceleration in position; abar and bp weight h acceleration in velocity. btilde and bptilde are error-difference weights.':m.kind==='rosenbrock'?'A and C use the source Rosenbrock stage convention; btilde is the error-difference vector.':m.kind==='ark'?'Separate explicit/implicit matrices and output weights; d vectors store embedded weights.':'a: stage matrix; b: propagated weights; c: stage times. bhat or companion_b, when present, are embedded solution weights.');
 }else if(m.kind==='recurrence'){
  representation='Stabilized three-term recurrence';parameters={stages:config.stages??9};coefficients=catalog.recurrences[id.replace('stabilized_','')][parameters.stages];
  notes.push('Arrays are indexed 0…s; index 0 is the initial state. Stage 1 uses mtilde[1]. Later stages use mu, nu, mtilde and gtilde; c holds RHS clocks, b the equivalent output weights.');
 }else if(m.kind==='exponential'){
  representation='Exponential operator program';const program=m.program??(id==='etd2rk'?'ETDRK2':'Exprb32');parameters={program};coefficients=catalog.exponentialPrograms[program];
  source+='exponential.py';notes.push('Coefficients are operator expressions in z = hL or hJ, encoded as the source expression tree (add, mul, phi, exp). They are not scalar Butcher weights. calls lists stage clocks/coefficient expressions; output and error give the final combinations.');
 }else if(m.kind==='extrapolation'){
  representation='Richardson extrapolation';coefficients={base:m.base,subdivisions:m.subdivisions,weights:m.weights,comparison:m.comparison};source+='extrapolation.py';notes.push('Combine base-method trajectories using weights. comparison gives the lower-order combination for the error estimate.');
 }else if(m.kind==='newmark'){
  representation='Second-order acceleration map';source=root+'drivers/mechanics/';
  if(id==='generalizedAlpha'){const rho=config.rhoInfinity??.5,am=(2*rho-1)/(rho+1),af=rho/(rho+1);parameters={rhoInfinity:rho};coefficients={alpha_m:am,alpha_f:af,gamma:.5-am+af,beta:(1-am+af)**2/4};source+='generalized_alpha.py';}
  else{parameters={beta:config.beta??.25,gamma:config.gamma??.5};coefficients={alpha_m:0,alpha_f:0,...parameters};source+='newmark.py';}
  notes.push('qₙ₊₁ = qₙ + h vₙ + h²[(½−β)aₙ + βaₙ₊₁]; vₙ₊₁ = vₙ + h[(1−γ)aₙ + γaₙ₊₁]. Generalized alpha evaluates the force at αf yₙ + (1−αf)yₙ₊₁ and balances (1−αm)aₙ₊₁ + αm aₙ.');
 }else if(m.kind==='ndf'){
  representation='Numerical differentiation formula';source=root+'drivers/history/ndf.py';coefficients={order:q,kappa:m.kappa,beta:m.beta,error_constant:m.error_constant,harmonic:Array.from({length:q},(_,i)=>Array.from({length:i+1},(_,j)=>1/(j+1)).reduce((a,b)=>a+b,0)),max_step_ratio:m.max_step_ratio,starter:m.starter,starter_tableau:catalog.schemes.find(s=>s.identifier===m.starter).tableau};
  notes.push('Backward differences are ordered D₁…Dq. Prediction is y + ΣDj; the implicit known term subtracts β ΣHj Dj. Error = error_constant × (accepted endpoint − prediction). The source R(r)U transform rescales differences when h changes.');
 }else if(m.kind==='irkn'){
  representation='Improved Nyström history map';source=root+'drivers/mechanics/improved_nystrom.py';coefficients={c:q===3?[0,.5]:[0,.25,.75],a21:q===3?1/8:1/32,...(q===4?{a32:9/32}:{}),warm_velocity_drift:[1.5,-.5],warm_acceleration_labels:q===3?['aₙ','aₙ₋₁','a₂','a₂,old']:['aₙ','aₙ₋₁','a₂','a₂,old','a₃','a₃,old'],warm_position_weights:q===3?[0,0,5/12,-5/12]:[0,0,7/24,-7/24,1/8,-1/8],warm_velocity_weights:q===3?[2/3,1/3,5/6,-5/6]:[19/18,-1/18,-1/6,1/6,11/18,-11/18],startup:{c:[0,.5,1],a:[[0,0,0],[1/8,0,0],[0,.5,0]],b:[1/6,1/3,0],bp:[1/6,2/3,1/6]}};
  notes.push('Warm position weights multiply h² acceleration, velocity weights h acceleration. Constant h and one accepted history entry are required; otherwise the corrected Nyström starter runs. IRKN3 stage 2 uses the previous initial acceleration; IRKN4 uses the current endpoint acceleration.');
 }else if(m.kind==='ros3p'){
  representation='ROS3P derivative-stage map';coefficients=D.ROS3P;source=root+'drivers/implicit/rosenbrock.py';notes.push('(I − hγJ)kᵢ = f(t+cᵢh, y+hΣAᵢⱼkⱼ) + hJΣGᵢⱼkⱼ + hdᵢfₜ; yₙ₊₁ = yₙ + hΣbᵢkᵢ. btilde gives the error-difference weights.');
 }else if(['history','imex-history'].includes(m.kind)){
  representation='Linear multistep formula';source=root+'drivers/history/';
  if(id.startsWith('ab')){coefficients={predictor:D.AB[q],...(id.startsWith('abm')?{corrector:D.AM[q]}:{})};notes.push('Predictor weights multiply h[fₙ, fₙ₋₁, …]; PECE corrector weights multiply h[f(predicted endpoint), fₙ, …].');source+='adams.py';}
  else if(id.startsWith('am')){coefficients={implicit_derivative:D.AM[q][0],past_derivatives:D.AM[q].slice(1)};notes.push('yₙ₊₁ = yₙ + h[implicit_derivative fₙ₊₁ + Σ past_derivativesⱼ fₙ₋ⱼ].');source+='adams.py';}
  else if(id.startsWith('bdf')||id.startsWith('sbdf')){const [state_weights,beta]=D.BDF[q];coefficients={state_weights,beta,...(id.startsWith('sbdf')?{explicit_extrapolation:q===2?[2,-1]:[3,-3,1]}:{})};notes.push('yₙ₊₁ = Σ state_weightsⱼ yₙ₋ⱼ + hβ fₙ₊₁. SBDF uses implicit fₙ₊₁ plus the explicit extrapolation of past explicit RHSs.');source+=id.startsWith('sbdf')?'imex.py':'bdf.py';}
  else if(id.startsWith('ssprkmsvs')){coefficients=id.endsWith('32')?{state_weights:[.75,0,.25],derivative_weights:[1.5,0,0]}:{state_weights:[16/27,0,0,11/27],derivative_weights:[16/9,0,0,4/9]};notes.push('yₙ₊₁ = Σ state_weightsⱼ yₙ₋ⱼ + h Σ derivative_weightsⱼ fₙ₋ⱼ.');source+='ssp.py';}
  else{coefficients={imexEuler:{state_weights:[1],explicit_weights:[1],implicit_weights:[1,0]},cnab2:{state_weights:[1],explicit_weights:[1.5,-.5],implicit_weights:[.5,.5,0]},cnlf2:{state_weights:[0,1],explicit_weights:[2,0],implicit_weights:[1,0,1]}}[id];source+='imex.py';notes.push('State and explicit weights start at n, then n−1. Implicit weights start at n+1, then n, then n−1. All derivative terms multiply h.');}
  const starter=(id==='cnlf2'||id==='imexEuler')?'IMEX Euler':((id.startsWith('bdf')||id.startsWith('sbdf'))&&q>=3)?{3:'ESDIRK324L2SA',4:'KenCarp4',5:'KenCarp5'}[q]:'Dormand–Prince 5(4)';notes.push('Warm formula shown; browser startup: '+starter+'. A step-size change restarts fixed-step history. SBDF on viscous Burgers uses Dormand–Prince startup.');
 }else if(m.kind==='mechanics'){
  source=root+'drivers/mechanics/';representation='Position / velocity map';
  if(id==='pefrl'||id==='vefrl'){const pe=id==='pefrl',xi=pe?.1786178958448091:.1720865590295143,lambda=pe?-.2123418310626054:-.09156203075515678,chi=pe?-.06626458266981849:-.1616217622107222,symmetric=[xi,chi,1-2*(xi+chi),chi,xi],other=[(1-2*lambda)/2,lambda,lambda,(1-2*lambda)/2];coefficients={xi,lambda,chi,drift:pe?symmetric:other,kick:pe?other:symmetric,first:pe?'drift':'kick'};source+='forest_ruth.py';notes.push('Alternate drift q ← q + h dᵢ v and kick v ← v + h kᵢ a. The RHS clock advances by drift weights.');}
  else if(id==='semiImplicitEuler'){coefficients={kick:[1],drift:[1],first:'kick'};notes.push('Kick v with a(tₙ,yₙ), then drift q with the updated velocity.');source=root+'drivers/explicit/euler.py';}
  else if(id==='symplecticEuler'){coefficients={predictor:.5,acceleration_weight:1,position_velocity_weights:[.5,.5]};notes.push('Source midpoint-predictor map: evaluate a at y + ½h f(t,y); vₙ₊₁ = vₙ + h a; qₙ₊₁ = qₙ + ½h(vₙ + vₙ₊₁).');source+='verlet.py';}
  else{coefficients={position_acceleration:.5,velocity_acceleration_weights:[.5,.5],endpoint_velocity_predictor:id==='velocityVerlet'?.5:0};notes.push('qₙ₊₁ = qₙ + h vₙ + ½h²aₙ; vₙ₊₁ = vₙ + ½h(aₙ + aₙ₊₁). Velocity Verlet evaluates the endpoint RHS with half-kicked velocity; leapfrog uses the old velocity.');source+='verlet.py';}
 }
 if(!coefficients)throw Error('Missing coefficient representation: '+id);
 return structuredClone({schemaVersion:1,application:'warpSPH Integrator Laboratory',method:{identifier:id,name:m.name,family:m.family,kind:m.kind,order:q,embedded_order:m.embedded_order??null,estimate_order:m.estimate_order??null},representation,parameters,coefficients,notes,cfl:methodCFL(m),source});
}
