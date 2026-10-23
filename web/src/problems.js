// Fixed-layout ports of testing.py's toy RHSs. Jv and Jᵀv are analytic;
// no finite-difference Jacobian or automatic differentiation is used here.
export const problemCatalog = [
  ['oscillator','Harmonic oscillator','mechanics',"q″ = −kq",'Exact solution · conservative'],
  ['damped','Damped oscillator','mechanics',"q″ = −kq − cv",'Exact solution · drag'],
  ['heated','Oscillator + heat','mechanics',"q″ = −kq − cv, Q′ = cv²",'Exact solution · energy budget'],
  ['forced','Forced motion','mechanics',"q″ = cos(t)",'Exact solution · stage times'],
  ['kepler','Kepler orbit','mechanics',"q″ = −q / |q|³",'Exact circular orbit · nonlinear geometry'],
  ['stiffDampedOscillator','Stiff damped oscillator','mechanics',"q″ = −ω²q − cv",'Exact solution · all damping regimes'],
  ['structural','Two-mass structural modes','mechanics',"q″ = −Kq, ω = 1, 40",'Exact modal solution · high-frequency damping'],
  ['vanDerPol','Van der Pol oscillator','mechanics',"q″ = μ(1 − q²)v − q",'Nonlinear · refinement reference'],
  ['stiffRelaxation','Stiff relaxation','ode',"y′ = −λ(y − s(t)) + s′(t)",'Exact solution · Prothero–Robinson'],
  ['robertson','Robertson chemistry','ode',"A → B → C",'Stiff · mass and positivity'],
  ['diffusion','Diffusion','pde',"uₜ = D uₓₓ",'Dirichlet · exact semi-discrete modes'],
  ['advection','Upwind advection','pde',"uₜ + a uₓ = 0",'Periodic · exact semi-discrete Fourier modes'],
  ['advectionStep','Advection of a step','pde',"uₜ + a uₓ = 0",'Periodic · TVD and stage overshoots'],
  ['burgers','Inviscid Burgers','pde',"uₜ + (u²/2)ₓ = 0",'Lax–Friedrichs flux · SSP'],
  ['viscousBurgers','Viscous Burgers','pde',"uₜ + (u²/2)ₓ = νuₓₓ",'Central flux · semilinear split'],
  ['acoustic','Linear acoustics','pde',"v′ = −c²Gρ, ρ′ = −Dv",'Skew-adjoint gradient/divergence pair'],
  ['driftOscillator','Drift-field oscillator','ode',"q′ = v, v′ = −q − κr, r′ = (1+q²/2)v − γr",'Coupled drift quantity · invariant'],
].map(([id,name,type,equation,note])=>({id,name,type,equation,note}));
export const defaults={k:4,c:.5,omega:1,mu:2,rate:50,forcing:'tanh',sign:1,n:32,D:.02,nu:.01,speed:1,kappa:.5,gamma:0,mode:4};
const sq=x=>x*x, zeros=n=>Array(n).fill(0);
export const norm=a=>Math.hypot(...a);
export const tv=a=>a.reduce((s,x,i)=>s+Math.abs(x-a[(i+1)%a.length]),0);
export function makeProblem(id,options={}) {
  const p={...defaults,...options};const meta=problemCatalog.find(x=>x.id===id);
  if(!meta)throw Error(`Unknown problem ${id}`);
  if(!Number.isInteger(p.n)||p.n<8||p.n>128)throw Error('Grid size must be an integer from 8 to 128');
  for(const key of ['k','c','omega','mu','rate','D','nu','speed','kappa','gamma'])if(!Number.isFinite(p[key])||p[key]<0)throw Error(`Invalid ${key}`);
  if(p.k<=0||p.omega<=0||p.speed<=0||p.rate<=0)throw Error('k, ω, speed and rate must be positive');
  if(!['tanh','sin','cos'].includes(p.forcing)||![-1,1].includes(p.sign))throw Error('Invalid forcing or sign');
  const n=p.n;let y0,exact=null,energy=null,m=0,separable=false,linear=false,spectral=0;
  const mech=['oscillator','damped','heated','forced','kepler','stiffDampedOscillator','structural','vanDerPol'].includes(id);
  if(mech){m=['kepler','structural'].includes(id)?2:1;y0=id==='kepler'?[1,0,0,1]:id==='structural'?[1.1/Math.SQRT2,.9/Math.SQRT2,0,0]:[1,0];if(id==='heated')y0.push(0);separable=['oscillator','kepler','structural','forced'].includes(id);}
  if(['oscillator','damped','heated','stiffDampedOscillator'].includes(id)){
    const k=id==='stiffDampedOscillator'?sq(p.omega):p.k,c=id==='oscillator'?0:p.c,g=c/2,d=k-g*g;
    exact=t=>{let x,v;const e=Math.exp(-g*t);if(Math.abs(d)<1e-12){x=e*(1+g*t);v=-k*t*e;}else if(d>0){const w=Math.sqrt(d);x=e*(Math.cos(w*t)+g*Math.sin(w*t)/w);v=-k*e*Math.sin(w*t)/w;}else{const w=Math.sqrt(-d),a=-g+w,b=-g-w;x=(-b*Math.exp(a*t)+a*Math.exp(b*t))/(2*w);v=k*(Math.exp(b*t)-Math.exp(a*t))/(2*w);}return id==='heated'?[x,v,.5*k-.5*(k*x*x+v*v)]:[x,v];};
    energy=y=>.5*(k*y[0]*y[0]+y[1]*y[1])+(id==='heated'?y[2]:0);linear=id!=='heated';spectral=Math.max(Math.sqrt(k),c);
  }
  if(id==='forced'){exact=t=>[2-Math.cos(t),Math.sin(t)];linear=true;}
  if(id==='kepler'){exact=t=>[Math.cos(t),Math.sin(t),-Math.sin(t),Math.cos(t)];energy=y=>.5*(y[2]**2+y[3]**2)-1/Math.hypot(y[0],y[1]);spectral=2;}
  if(id==='structural'){exact=t=>{const a=Math.cos(t),b=.1*Math.cos(40*t),v=-Math.sin(t),w=-4*Math.sin(40*t);return [(a+b)/Math.SQRT2,(a-b)/Math.SQRT2,(v+w)/Math.SQRT2,(v-w)/Math.SQRT2];};energy=y=>.5*(y[2]**2+y[3]**2)+.25*((y[0]+y[1])**2+1600*(y[0]-y[1])**2);linear=true;spectral=40;}
  if(id==='stiffRelaxation'){const s=t=>p.forcing==='tanh'?Math.tanh(t):p.forcing==='sin'?Math.sin(t):Math.cos(t);y0=[s(0)];exact=t=>[s(t)];linear=true;spectral=p.rate;}
  if(id==='robertson'){y0=[1,0,0];spectral=1e4;}
  if(id==='driftOscillator'){y0=[1,0,.2];energy=y=>.5*y[1]**2+.5*(1+p.kappa)*y[0]**2+p.kappa*y[0]**4/24+p.kappa*(.2-7/6)*y[0];}
  if(id==='diffusion'){y0=Array.from({length:n},(_,i)=>Math.sin(Math.PI*(i+1)/(n+1))+.5*Math.sin(5*Math.PI*(i+1)/(n+1)));exact=t=>y0.map((_,i)=>Math.sin(Math.PI*(i+1)/(n+1))*Math.exp(-4*p.D*(n+1)**2*sq(Math.sin(Math.PI/(2*(n+1))))*t)+.5*Math.sin(5*Math.PI*(i+1)/(n+1))*Math.exp(-4*p.D*(n+1)**2*sq(Math.sin(5*Math.PI/(2*(n+1))))*t));linear=true;spectral=4*p.D*(n+1)**2;}
  if(id==='advection'||id==='advectionStep'){y0=Array.from({length:n},(_,i)=>id==='advectionStep'?(i<n/2?1:-1):Math.cos(2*Math.PI*i/n)+.5*Math.cos(4*Math.PI*i/n));
    // Discrete Fourier solution of the upwind operator, including step data.
    const modes=Array.from({length:n},(_,k)=>{let re=0,im=0;for(let j=0;j<n;j++){re+=y0[j]*Math.cos(2*Math.PI*k*j/n)/n;im-=y0[j]*Math.sin(2*Math.PI*k*j/n)/n;}return [re,im];});
    exact=t=>y0.map((_,j)=>modes.reduce((s,[re,im],k)=>{const a=2*Math.PI*k/n,decay=Math.exp(p.speed*n*(Math.cos(a)-1)*t),angle=a*j-p.speed*n*Math.sin(a)*t;return s+decay*(re*Math.cos(angle)-im*Math.sin(angle));},0));linear=true;spectral=2*p.speed*n;
  }
  if(id==='burgers')y0=Array.from({length:n},(_,i)=>-Math.sin(2*Math.PI*i/n));
  if(id==='viscousBurgers'){y0=Array.from({length:n},(_,i)=>Math.exp(-((i/n-.5)**2)/(2*.08**2)));spectral=4*p.nu*n*n;}
  if(id==='acoustic'){y0=[...Array.from({length:n},(_,i)=>i/n),...zeros(n),...Array.from({length:n},(_,i)=>1+.001*Math.cos(2*Math.PI*p.mode*i/n))];linear=true;spectral=2*p.speed*n;energy=y=>y.slice(n,2*n).reduce((s,v,i)=>s+.5*v*v+.5*p.speed**2*(y[2*n+i]-1)**2,0);}
  function rhs(t,y,part='full') {
    const z=zeros(y.length);let a,b,c;
    if(mech){for(let j=0;j<m;j++)z[j]=y[m+j];
      if(id==='kepler'){const r=Math.hypot(y[0],y[1]);z[2]=-y[0]/r**3;z[3]=-y[1]/r**3;}
      else if(id==='structural'){z[2]=-800.5*y[0]+799.5*y[1];z[3]=799.5*y[0]-800.5*y[1];}
      else if(id==='forced')z[1]=Math.cos(t);
      else if(id==='vanDerPol')z[1]=p.mu*(1-y[0]**2)*y[1]-y[0];
      else {a=id==='stiffDampedOscillator'?p.omega**2:p.k;b=id==='oscillator'?0:p.c;z[1]=-a*y[0]-b*y[1];if(id==='heated')z[2]=b*y[1]**2;}
    }else if(id==='stiffRelaxation'){a=p.forcing==='tanh'?Math.tanh(t):p.forcing==='sin'?Math.sin(t):Math.cos(t);b=p.forcing==='tanh'?1-a*a:p.forcing==='sin'?Math.cos(t):-Math.sin(t);z[0]=-p.sign*p.rate*(y[0]-a)+b;}
    else if(id==='robertson'){a=.04*y[0];b=1e4*y[1]*y[2];c=3e7*y[1]*y[1];z[0]=-a+b;z[1]=a-b-c;z[2]=c;}
    else if(id==='driftOscillator'){z[0]=y[1];z[1]=-y[0]-p.kappa*y[2];z[2]=(1+.5*y[0]**2)*y[1]-p.gamma*y[2];}
    else if(id==='acoustic'){for(let i=0;i<n;i++){z[i]=y[n+i];z[n+i]=-(p.speed**2)*n*(y[2*n+(i+1)%n]-y[2*n+i]);z[2*n+i]=-n*(y[n+i]-y[n+(i+n-1)%n]);}}
    else {const alpha=Math.max(...y.map(Math.abs));for(let i=0;i<n;i++){const u=y[i],l=id==='diffusion'?(i===0?0:y[i-1]):y[(i+n-1)%n],r=id==='diffusion'?(i===n-1?0:y[i+1]):y[(i+1)%n];
      if(id==='diffusion')z[i]=p.D*(n+1)**2*(l-2*u+r);
      else if(id.startsWith('advection'))z[i]=p.speed*n*(l-u);
      else if(id==='viscousBurgers')z[i]=-.25*n*(r*r-l*l)+p.nu*n*n*(l-2*u+r);
      else z[i]=.25*n*(l*l-r*r)+.5*n*alpha*(l-2*u+r);
    }}
    // Library's ordinary callable IMEX fallback is pure implicit. Only the
    // viscous Burgers helper declares a nontrivial semilinear split.
    if(part==='full')return z;
    if(id==='viscousBurgers'){const l=y.map((u,i)=>p.nu*n*n*(y[(i+n-1)%n]-2*u+y[(i+1)%n]));return part==='implicit'||part==='linear'?l:z.map((u,i)=>u-l[i]);}
    if(part==='explicit')return zeros(y.length);
    return z;
  }
  function jvp(t,y,v,part='full'){
    const z=zeros(y.length);
    if(linear){if(part==='explicit')return z;const a=rhs(t,v),b=rhs(t,zeros(y.length));return a.map((x,i)=>x-b[i]);}
    if(id==='heated'){z[0]=v[1];z[1]=-p.k*v[0]-p.c*v[1];z[2]=2*p.c*y[1]*v[1];}
    else if(id==='kepler'){const r=Math.hypot(y[0],y[1]),d=y[0]*v[0]+y[1]*v[1];z[0]=v[2];z[1]=v[3];z[2]=-v[0]/r**3+3*y[0]*d/r**5;z[3]=-v[1]/r**3+3*y[1]*d/r**5;}
    else if(id==='vanDerPol'){z[0]=v[1];z[1]=(-1-2*p.mu*y[0]*y[1])*v[0]+p.mu*(1-y[0]**2)*v[1];}
    else if(id==='robertson'){const b=1e4*(v[1]*y[2]+y[1]*v[2]),c=6e7*y[1]*v[1];z[0]=-.04*v[0]+b;z[1]=.04*v[0]-b-c;z[2]=c;}
    else if(id==='driftOscillator'){z[0]=v[1];z[1]=-v[0]-p.kappa*v[2];z[2]=y[0]*y[1]*v[0]+(1+.5*y[0]**2)*v[1]-p.gamma*v[2];}
    else if(id==='viscousBurgers'||id==='burgers'){const imax=y.reduce((a,x,i)=>Math.abs(x)>Math.abs(y[a])?i:a,0),alpha=Math.abs(y[imax]),da=Math.sign(y[imax])*v[imax];for(let i=0;i<n;i++){const l=(i+n-1)%n,r=(i+1)%n,L=v[l]-2*v[i]+v[r];const diff=id==='viscousBurgers'?p.nu*n*n*L:.5*n*(da*(y[l]-2*y[i]+y[r])+alpha*L);const adv=.5*n*(y[l]*v[l]-y[r]*v[r]);z[i]=id==='viscousBurgers'?(part==='implicit'||part==='linear'?diff:part==='explicit'?adv:diff+adv):diff+adv;}}
    if(part==='explicit'&&id!=='viscousBurgers')return zeros(y.length);
    return z;
  }
  function vjp(t,y,v,part='full'){return y.map((_,j)=>{const e=zeros(y.length);e[j]=1;return jvp(t,y,e,part).reduce((s,x,i)=>s+x*v[i],0);});}
  function ft(t,y){const z=zeros(y.length);if(id==='forced')z[1]=-Math.sin(t);if(id==='stiffRelaxation'){const s=exact(t)[0],sp=p.forcing==='tanh'?1-s*s:p.forcing==='sin'?Math.cos(t):-Math.sin(t),spp=p.forcing==='tanh'?-2*s*sp:-s;z[0]=p.sign*p.rate*sp+spp;}return z;}
  return {...meta,p,y0,rhs,jvp,vjp,ft,exact,energy,m,mechanics:mech,separable,linear,spectral,grid:meta.type==='pde',feLimit:id.startsWith('advection')?1/(p.speed*n):id==='burgers'?1/n:null};
}
export function compatible(method,problem){
  if(['mechanics','newmark','nystrom','splitting'].includes(method.kind)&&!problem.mechanics)return 'Requires a position/velocity mechanics problem';
  if(method.kind==='nystrom'&&method.requires_separable&&!problem.separable)return 'Requires velocity-independent forces and passive extra fields';
  if(method.kind==='splitting'&&!['oscillator','damped','heated','stiffDampedOscillator','kepler','structural'].includes(problem.id))return 'Requires declared exact physical flows';
  if(method.kind==='exponential'&&(method.identifier==='etd2rk'||['LawsonEuler','NorsettEuler','ETDRK2','ETDRK3','ETDRK4','HochOst4'].includes(method.program))&&!problem.linear&&problem.id!=='viscousBurgers')return 'Requires a declared constant linear operator';
  if(method.kind==='functionmap'&&!problem.exact)return 'Requires a known analytic flow';
  return null;
}
