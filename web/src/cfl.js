// A CFL is a property of a method + spatial operator, not of an ODE method alone.
// Guaranteed SSP constants and finite-model measurements remain separate.
const ssp={forwardEuler:1,explicitEuler:1,heunsMethod:1,tvdRK2:1,tvdRK3:1,sspRK3:1,ssprk43:2,ssprk432:2,ssprk932:6,ssprk104:6,ssprkmsvs32:.5,ssprkmsvs43:1/3};
const upwind={ssprk53:2.6506,ssprk63:3.5184,ssprk73:4.2879,ssprk83:5.1071,ssprk54:1.7693,kykssprk42:2.2838,kyk2014dgssprk3s2:1.8939,ssprk53_2n1:2.1808,ssprk53_2n2:2.1487,ssprk53_h:2.6506};
export function methodCFL(method){
 const id=method.identifier;
 return {ssp:ssp[id]??null,upwind:upwind[id]??null,
  note:id.startsWith('ssprkmsvs')?'SSP bound applies to the warm history formula; the Dormand–Prince starter has a separate stage behavior.':upwind[id]!=null?'Measured all-stage convexity boundary for the library’s periodic upwind model, rounded to four decimals. This is not a general nonlinear SSP guarantee.':ssp[id]!=null?'SSP inheritance: h ≤ C × hFE when forward Euler has the required convex stability property.':'No verified SSP coefficient is recorded for this scheme. Absolute stability and accuracy impose separate requirements.',
  source:upwind[id]!=null?'tests/test_tvd.py; docs/docs/families/explicit-rk.md':id.startsWith('ssprkmsvs')?'src/warpSPHIntegrators/drivers/history/ssp.py':'docs/docs/families/explicit-rk.md'};
}
export function timestepScale(problem){
 const {id,p,y0}=problem,n=p.n;
 if(id.startsWith('advection'))return {label:'Transport CFL',unit:'a h / Δx',rate:Math.abs(p.speed)*n,feLimit:1/(Math.abs(p.speed)*n),note:'Periodic first-order upwind: CFL = 1 is the forward-Euler monotonicity limit.'};
 if(id==='burgers'){const speed=Math.max(...y0.map(Math.abs));return {label:'Initial CFL',unit:'max|u₀| h / Δx',rate:speed*n,feLimit:1/(speed*n),note:'Lax–Friedrichs Burgers: initial-state CFL. The wave speed can change during a run.'};}
 if(id==='diffusion')return {label:'Diffusion ratio',unit:'D h / Δx²',rate:p.D*(n+1)**2,feLimit:1/(2*p.D*(n+1)**2),note:'Centered diffusion: ratio ½ is the forward-Euler maximum-principle limit.'};
 if(id==='acoustic')return {label:'Acoustic CFL',unit:'c h / Δx',rate:Math.abs(p.speed)*n,feLimit:null,note:'Wave propagation ratio. Forward Euler is unstable for this centered acoustic operator; no SSP time-step bound follows.'};
 if(id==='viscousBurgers')return {label:'Diffusion ratio',unit:'ν h / Δx²',rate:p.nu*n*n,feLimit:null,note:'Diffusive part only. Central nonlinear transport and the explicit/implicit split impose separate limits.'};
 return {label:'Relative CFL',unit:null,rate:null,feLimit:null,note:'Not defined for this ODE: there is no spatial grid or transport speed.'};
}
export function timestepIncrement(value){return Number.isFinite(value)&&value>0?10**(Math.floor(Math.log10(value))-1):.001;}
export function nudgeTimestep(value,direction){return Number(Math.min(100,Math.max(1e-8,value+direction*timestepIncrement(value))).toPrecision(14));}
