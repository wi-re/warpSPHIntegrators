import {makeProblem,compatible,norm,tv} from './problems.js';
export const zeros=n=>Array(n).fill(0);
export const add=(a,b,c=1)=>a.map((x,i)=>x+c*b[i]);
export const scale=(a,c)=>a.map(x=>x*c);
export const dot=(a,b)=>a.reduce((s,x,i)=>s+x*b[i],0);
const maxabs=a=>Math.max(...a.map(Math.abs));
function combine(y,h,k,b){let z=y.slice();for(let j=0;j<b.length;j++)if(b[j])z=add(z,k[j],h*b[j]);return z;}
export function denseSolve(A,b){
  const n=b.length,M=A.map((r,i)=>[...r,b[i]]);
  for(let k=0;k<n;k++){let pivot=k;for(let i=k+1;i<n;i++)if(Math.abs(M[i][k])>Math.abs(M[pivot][k]))pivot=i;
    if(Math.abs(M[pivot][k])<1e-25)throw Error('Singular stage Jacobian');
    [M[k],M[pivot]]=[M[pivot],M[k]];
    for(let i=k+1;i<n;i++){const c=M[i][k]/M[k][k];for(let j=k+1;j<=n;j++)M[i][j]-=c*M[k][j];M[i][k]=0;}
  }
  const x=zeros(n);for(let i=n-1;i>=0;i--){let s=M[i][n];for(let j=i+1;j<n;j++)s-=M[i][j]*x[j];x[i]=s/M[i][i];}return x;
}
export function gmres(A,b,{tol=1e-11,maxIterations=192,restart=24}={}){
  const n=b.length;let x=zeros(n),iterations=0,r=b.slice();const target=tol*Math.max(1,norm(b));
  while(norm(r)>target&&iterations<maxIterations){
    const beta=norm(r),V=[scale(r,1/beta)],m=Math.min(restart,n,maxIterations-iterations),H=Array.from({length:m+1},()=>zeros(m)),g=zeros(m+1),cs=[],sn=[];g[0]=beta;let used=0;
    for(let j=0;j<m;j++){
      let w=A(V[j]);iterations++;used=j+1;
      for(let pass=0;pass<2;pass++)for(let i=0;i<=j;i++){const h=dot(w,V[i]);H[i][j]+=h;w=add(w,V[i],-h);}
      H[j+1][j]=norm(w);const breakdown=H[j+1][j]<1e-25;if(!breakdown)V.push(scale(w,1/H[j+1][j]));
      for(let i=0;i<j;i++){const a=H[i][j],b=H[i+1][j];H[i][j]=cs[i]*a+sn[i]*b;H[i+1][j]=-sn[i]*a+cs[i]*b;}
      const d=Math.hypot(H[j][j],H[j+1][j]);if(d<1e-30)throw Error('GMRES breakdown');cs[j]=H[j][j]/d;sn[j]=H[j+1][j]/d;H[j][j]=d;H[j+1][j]=0;
      g[j+1]=-sn[j]*g[j];g[j]*=cs[j];if(Math.abs(g[j+1])<=target||breakdown)break;
    }
    const c=zeros(used);for(let i=used-1;i>=0;i--){let s=g[i];for(let j=i+1;j<used;j++)s-=H[i][j]*c[j];c[i]=s/H[i][i];}
    for(let i=0;i<used;i++)x=add(x,V[i],c[i]);r=add(b,A(x),-1);
  }
  if(norm(r)>target*5)throw Error(`GMRES did not converge (${iterations} iterations; residual ${norm(r).toExponential(2)})`);
  return {x,iterations,residual:norm(r)};
}
export function newton(residual,jvp,initial,opts={},stats={}){
  let y=initial.slice();const tol=opts.solveTolerance??1e-11,max=opts.newtonIterations??12;
  for(let i=0;i<=max;i++){
    const r=residual(y),rn=norm(r);stats.residual=rn;
    if(!Number.isFinite(rn))throw Error('Nonfinite nonlinear residual');
    if(rn<=tol*(1+norm(y))){stats.solveResidual=Math.max(stats.solveResidual||0,rn);return y;}
    if(i===max)break;
    let delta;
    if(opts.linearSolver==='dense'){
      const columns=y.map((_,j)=>{const e=zeros(y.length);e[j]=1;return jvp(y,e);});
      const M=y.map((_,i)=>columns.map(c=>c[i]));delta=denseSolve(M,scale(r,-1));stats.linear=(stats.linear||0)+1;
    }else{const solve=gmres(v=>jvp(y,v),scale(r,-1),{tol:Math.max(1e-14,tol*.1),maxIterations:opts.krylovIterations??256,restart:Math.min(y.length,32)});delta=solve.x;stats.linear=(stats.linear||0)+solve.iterations;}
    stats.newton=(stats.newton||0)+1;let factor=1,candidate=add(y,delta);while(factor>1/64&&norm(residual(candidate))>rn*(1-1e-4*factor)){factor*=.5;candidate=add(y,delta,factor);}y=candidate;
  }throw Error('Newton stage failed to converge; reduce the step or tighten the linear solve');
}
const AB={1:[1],2:[3/2,-1/2],3:[23/12,-16/12,5/12],4:[55/24,-59/24,37/24,-9/24],5:[1901/720,-2774/720,2616/720,-1274/720,251/720]};
const AM={1:[1],2:[.5,.5],3:[5/12,8/12,-1/12],4:[9/24,19/24,-5/24,1/24],5:[251/720,646/720,-264/720,106/720,-19/720]};
const BDF={1:[[1],1],2:[[4/3,-1/3],2/3],3:[[18/11,-9/11,2/11],6/11],4:[[48/25,-36/25,16/25,-3/25],12/25],5:[[300/137,-300/137,200/137,-75/137,12/137],60/137]};
export function createStepper(method,problem,catalog,options={}){
  const issue=compatible(method,problem);if(issue)throw Error(issue);
  const stats={rhs:0,jvp:0,newton:0,linear:0,stageTV:0,stageOvershoot:0,residual:0};
  const initialMin=Math.min(...problem.y0),initialMax=Math.max(...problem.y0);
  const observe=y=>{if(problem.grid){stats.stageTV=Math.max(stats.stageTV,tv(y));stats.stageOvershoot=Math.max(stats.stageOvershoot,Math.max(...y)-initialMax,initialMin-Math.min(...y));}};
  const f=(t,y,part='full')=>{stats.rhs++;observe(y);return problem.rhs(t,y,part);};
  const J=(t,y,v,part='full')=>{stats.jvp++;return problem.jvp(t,y,v,part);};
  const lookup=id=>catalog.schemes.find(s=>s.identifier===id);
  function implicit(base,t,c,part='full',guess=base){return newton(z=>add(add(z,base,-1),f(t,z,part),-c),(z,v)=>add(v,J(t,z,v,part),-c),guess,options,stats);}
  function rk(tab,y,t,h,coupled=false){
    let k=[],error=null;
    if(coupled){
      const s=tab.c.length,n=y.length;
      const seed=f(t,y),guess=Array.from({length:s},(_,i)=>add(y,seed,h*tab.c[i])).flat();
      const stages=newton(z=>{const rates=tab.c.map((c,i)=>f(t+c*h,z.slice(i*n,(i+1)*n)));return tab.a.flatMap((row,i)=>add(z.slice(i*n,(i+1)*n),combine(y,h,rates,row),-1));},
        (z,v)=>{const rates=tab.c.map((c,i)=>J(t+c*h,z.slice(i*n,(i+1)*n),v.slice(i*n,(i+1)*n)));return tab.a.flatMap((row,i)=>add(v.slice(i*n,(i+1)*n),combine(zeros(n),h,rates,row),-1));},guess,options,stats);
      k=tab.c.map((c,i)=>f(t+c*h,stages.slice(i*n,(i+1)*n)));
      if(tab.companion_b){const ext=[f(t,y),...k];error=scale(add(combine(y,h,k,tab.b),combine(y,h,ext,tab.companion_b),-1),1);}
    }else{
      for(let i=0;i<tab.c.length;i++){let stage=combine(y,h,k,tab.a[i].slice(0,i));if(tab.a[i][i])stage=implicit(stage,t+h*tab.c[i],h*tab.a[i][i]);k.push(f(t+h*tab.c[i],stage));}
      if(tab.bhat)error=combine(zeros(y.length),h,k,tab.b.map((b,i)=>b-tab.bhat[i]));
    }
    return {y:combine(y,h,k,tab.b),error};
  }
  function ark(tab,y,t,h){const e=[],im=[];for(let i=0;i<tab.c.length;i++){let z=combine(combine(y,h,e,tab.a_explicit[i].slice(0,i)),h,im,tab.a_implicit[i].slice(0,i));if(tab.a_implicit[i][i])z=implicit(z,t+h*tab.c[i],h*tab.a_implicit[i][i],'implicit');e.push(f(t+h*tab.c[i],z,'explicit'));im.push(f(t+h*tab.c[i],z,'implicit'));}
    return {y:combine(combine(y,h,e,tab.b_explicit),h,im,tab.b_implicit),error:combine(combine(zeros(y.length),h,e,tab.b_explicit.map((x,i)=>x-tab.d_explicit[i])),h,im,tab.b_implicit.map((x,i)=>x-tab.d_implicit[i]))};}
  function mechanics(id,y,t,h){
    const m=problem.m,N=y.length;let z=y.slice();const k0=f(t,y);
    if(id==='semiImplicitEuler'){for(let j=m;j<N;j++)z[j]+=h*k0[j];for(let j=0;j<m;j++)z[j]+=h*z[m+j];}
    else if(id==='symplecticEuler'){const half=add(y,k0,h/2),k=f(t+h/2,half);for(let j=m;j<N;j++)z[j]+=h*k[j];for(let j=0;j<m;j++)z[j]+=h/2*(y[m+j]+z[m+j]);}
    else if(id==='leapFrog'||id==='velocityVerlet'){
      for(let j=0;j<m;j++)z[j]+=h*y[m+j]+h*h/2*k0[m+j];
      if(id==='velocityVerlet')for(let j=m;j<2*m;j++)z[j]+=h/2*k0[j];const k1=f(t+h,z);
      for(let j=m;j<2*m;j++)z[j]=y[j]+h/2*(k0[j]+k1[j]);
      for(let j=2*m;j<N;j++)z[j]=y[j]+(id==='leapFrog'?h/2*(k0[j]+k1[j]):h*k1[j]);
    }else{
      const xi=id==='pefrl'?.1786178958448091:.1720865590295143,lambda=id==='pefrl'?-.2123418310626054:-.09156203075515678,chi=id==='pefrl'?-.06626458266981849:-.1616217622107222;
      const drift=id==='pefrl'?[xi,chi,1-2*(chi+xi),chi,xi]:[(1-2*lambda)/2,lambda,lambda,(1-2*lambda)/2];
      const kick=id==='pefrl'?[(1-2*lambda)/2,lambda,lambda,(1-2*lambda)/2]:[xi,chi,1-2*(chi+xi),chi,xi];
      let clock=t;
      if(id==='pefrl'){for(let i=0;i<5;i++){for(let j=0;j<m;j++)z[j]+=h*drift[i]*z[m+j];clock+=h*drift[i];if(i<4){const k=f(clock,z);for(let j=m;j<N;j++)z[j]+=h*kick[i]*k[j];}}}
      else{for(let i=0;i<5;i++){const k=f(clock,z);for(let j=m;j<N;j++)z[j]+=h*kick[i]*k[j];if(i<4){for(let j=0;j<m;j++)z[j]+=h*drift[i]*z[m+j];clock+=h*drift[i];}}}
    }return {y:z};
  }
  function nystrom(tab,y,t,h){const m=problem.m,N=y.length,ks=[];
    for(let i=0;i<tab.c.length;i++){const z=y.slice();for(let j=0;j<m;j++){z[j]+=tab.c[i]*h*y[m+j];for(let l=0;l<i;l++)z[j]+=h*h*(tab.a[i][l]||0)*ks[l][m+j];}
      if(tab.abar)for(let j=m;j<N;j++)for(let l=0;l<i;l++)z[j]+=h*(tab.abar[i][l]||0)*ks[l][j];ks.push(f(t+tab.c[i]*h,z));}
    const z=y.slice(),error=zeros(N);for(let j=0;j<m;j++){z[j]+=h*y[m+j];for(let i=0;i<ks.length;i++){z[j]+=h*h*tab.b[i]*ks[i][m+j];error[j]+=h*h*(tab.btilde[i]||0)*ks[i][m+j];}}
    for(let j=m;j<N;j++)if(j<2*m||tab.abar)for(let i=0;i<ks.length;i++){z[j]+=h*tab.bp[i]*ks[i][j];error[j]+=h*(tab.bptilde[i]||0)*ks[i][j];}f(t+h,z);return {y:z,error:method.estimate_order?error:null};
  }
  function structural(y,t,h,memory){
    const m=problem.m,rho=options.rhoInfinity??.5,isAlpha=method.identifier==='generalizedAlpha',am=isAlpha?(2*rho-1)/(rho+1):0,af=isAlpha?rho/(rho+1):0,gamma=isAlpha?.5-am+af:options.gamma??.5,beta=isAlpha?(1-am+af)**2/4:options.beta??.25;
    const f0=f(t,y),a0=isAlpha&&memory.acceleration?memory.acceleration:f0.slice(m,2*m);
    function state(a){const z=y.slice();for(let j=0;j<m;j++){z[j]+=h*y[m+j]+h*h*((.5-beta)*a0[j]+beta*a[j]);z[m+j]+=h*((1-gamma)*a0[j]+gamma*a[j]);}return z;}
    const a=newton(a=>{const z=state(a),weighted=y.map((x,i)=>af*x+(1-af)*z[i]),rate=f(t+(1-af)*h,weighted);return a.map((x,j)=>(1-am)*x+am*a0[j]-rate[m+j]);},
      (a,v)=>{const z=state(a),weighted=y.map((x,i)=>af*x+(1-af)*z[i]),dz=zeros(y.length);for(let j=0;j<m;j++){dz[j]=(1-af)*h*h*beta*v[j];dz[m+j]=(1-af)*h*gamma*v[j];}const r=J(t+(1-af)*h,weighted,dz);return v.map((x,j)=>(1-am)*x-r[m+j]);},a0,options,stats);
    let z=state(a);if(y.length>2*m){ // Coupled endpoint trapezoid for first-order quantities.
      const fixed=z.slice();z=newton(w=>{const rate=f(t+h,w);return w.map((x,j)=>j<2*m?x-fixed[j]:x-y[j]-.5*h*(f0[j]+rate[j]));},(w,v)=>{const r=J(t+h,w,v);return v.map((x,j)=>j<2*m?x:x-.5*h*r[j]);},z,options,stats);
    }return {y:z,memory:{...memory,acceleration:a}};
  }
  function recurrence(y,t,h){const family=method.identifier.replace('stabilized_',''),s=options.stages??9,data=catalog.recurrences[family]?.[s];if(!data)throw Error('Choose a supported stage count from 3 to 65');
    const f0=f(t,y);let prev=y.slice(),current=add(y,f0,h*data.mtilde[1]);for(let j=2;j<=s;j++){const rate=f(t+h*data.c[j-1],current);const next=y.map((x,i)=>x+data.mu[j]*(current[i]-x)+data.nu[j]*(prev[i]-x)+h*(data.mtilde[j]*rate[i]+data.gtilde[j]*f0[i]));prev=current;current=next;}return {y:current,error:method.estimate_order?add(current,add(y,f0,h),-1):null};}
  function history(y,t,h,memory){
    const id=method.identifier,q=method.order,old=Math.abs((memory.h??h)-h)<=1e-12*Math.abs(h)?memory.entries??[]:[],entry={y:y.slice(),t,f:f(t,y)},entries=[entry,...old].slice(0,5);
    const wrap=result=>({...result,memory:{entries,h}});let needed=q-1;if(id==='ssprkmsvs32')needed=2;if(id==='ssprkmsvs43')needed=3;
    if(id==='cnlf2')needed=1;
    if(old.length<needed){
      if(id==='cnlf2')return wrap({y:implicit(add(y,f(t,y,'explicit'),h),t+h,h,'implicit')});
      let starter='dormandPrince';if((id.startsWith('bdf')||(id.startsWith('sbdf')&&problem.id!=='viscousBurgers'))&&q>=3)starter={3:'esdirk324l2sa',4:'kencarp4',5:'kencarp5'}[q];
      return wrap(rk(lookup(starter).tableau,y,t,h));
    }
    if(id==='ssprkmsvs32')return wrap({y:y.map((x,i)=>.75*(x+2*h*entry.f[i])+.25*old[1].y[i])});
    if(id==='ssprkmsvs43')return wrap({y:y.map((x,i)=>16/27*(x+3*h*entry.f[i])+11/27*(old[2].y[i]+12/11*h*old[2].f[i]))});
    if(id.startsWith('ab')){const predictor=combine(y,h,entries.map(e=>e.f),AB[q]);if(id.startsWith('abm'))return wrap({y:combine(y,h,[f(t+h,predictor),...entries.map(e=>e.f)],AM[q])});return wrap({y:predictor});}
    if(id.startsWith('am')){const b=AM[q],base=combine(y,h,entries.map(e=>e.f),b.slice(1));return wrap({y:implicit(base,t+h,h*b[0])});}
    if(id.startsWith('bdf')){const [weights,b]=BDF[q],base=combine(zeros(y.length),1,entries.map(e=>e.y),weights);return wrap({y:implicit(base,t+h,h*b)});}
    if(id==='imexEuler')return wrap({y:implicit(add(y,f(t,y,'explicit'),h),t+h,h,'implicit')});
    if(id.startsWith('sbdf')){const [weights,b]=BDF[q],w=q===2?[2,-1]:[3,-3,1];let base=combine(zeros(y.length),1,entries.map(e=>e.y),weights);base=combine(base,h*b,entries.slice(0,q).map(e=>f(e.t,e.y,'explicit')),w);return wrap({y:implicit(base,t+h,h*b,'implicit')});}
    if(id==='cnab2'){let base=add(y,f(t,y,'implicit'),h/2);base=combine(base,h,entries.slice(0,2).map(e=>f(e.t,e.y,'explicit')),AB[2]);return wrap({y:implicit(base,t+h,h/2,'implicit')});}
    if(id==='cnlf2'){let base=add(old[0].y,f(old[0].t,old[0].y,'implicit'),h);base=add(base,f(t,y,'explicit'),2*h);return wrap({y:implicit(base,t+h,h,'implicit')});}
    throw Error(`Unimplemented history method ${id}`);
  }
  function extrapolate(y,t,h){const cols=[];for(const n of method.subdivisions){const dt=h/n;let z=y.slice();if(method.base==='euler'){for(let j=0;j<n;j++)z=add(z,f(t+j*dt,z),dt);}else{let prev=z;z=add(z,f(t,z),dt);for(let j=1;j<n;j++){const next=add(prev,f(t+j*dt,z),2*dt);prev=z;z=next;}}cols.push(z);}
    const z=combine(zeros(y.length),1,cols,method.weights),low=method.subdivisions.length===1?add(y,f(t,y),h):combine(zeros(y.length),1,cols,method.comparison);return {y:z,error:add(z,low,-1)};}
  function rosenbrock(y,t,h){
    const N=y.length,ft=problem.ft(t,y),tab=method.tableau,g=tab?.gamma??(.5+Math.sqrt(3)/6),stages=[];
    const solve=b=>{const r=gmres(v=>add(v,J(t,y,v),-h*g),b,{tol:options.solveTolerance??1e-11,maxIterations:256,restart:Math.min(32,N)});stats.linear+=r.iterations;return r.x;};
    if(method.kind==='ros3p'){
      const A=[[0,0,0],[1,0,0],[1,0,0]],G=[[0,0,0],[-1,0,0],[-g,.5-2*g,0]],clock=[0,1,1],time=[g,g-1,.5-2*g];
      for(let i=0;i<3;i++){const z=combine(y,h,stages,A[i].slice(0,i)),w=combine(zeros(N),1,stages,G[i].slice(0,i));stages.push(solve(add(add(f(t+h*clock[i],z),J(t,y,w),h),ft,h*time[i])));}
      return {y:combine(y,h,stages,[2/3,0,1/3]),error:combine(zeros(N),h,stages,[1/3,-1/3,0])};
    }
    for(let i=0;i<tab.c.length;i++){const z=combine(y,1,stages,tab.A[i].slice(0,i));let b=scale(f(t+h*tab.c[i],z),h*g);b=combine(b,g,stages,tab.C[i].slice(0,i));b=add(b,ft,h*h*g*tab.d[i]);stages.push(solve(b));}
    return {y:combine(y,1,stages,tab.b),error:tab.btilde?combine(zeros(N),1,stages,tab.btilde):null};
  }
  function exponential(y,t,h){
    const semilinear=method.identifier==='etd2rk'||['LawsonEuler','NorsettEuler','ETDRK2','ETDRK3','ETDRK4','HochOst4'].includes(method.program);
    if(semilinear&&!problem.linear&&problem.id!=='viscousBurgers')throw Error('This exponential method requires a declared constant linear operator');
    const ft=problem.ft(t,y),start=semilinear?y:[t,...y],A=v=>semilinear?scale(J(t,y,v,'linear'),h):[0,...scale(add(J(t,y,v.slice(1)),ft,v[0]),h)];
    const rows=zeros(start.length);for(let j=0;j<start.length;j++){const e=zeros(start.length);e[j]=1;A(e).forEach((x,i)=>rows[i]+=Math.abs(x));}const bound=Math.max(...rows);
    const action=(expr,v)=>{if(typeof expr==='number')return scale(v,expr);const [op,...args]=expr;
      if(op==='z')return A(v);
      if(op==='add')return args.reduce((s,e)=>add(s,action(e,v)),zeros(v.length));
      if(op==='mul')return args.reduce((z,e)=>action(e,z),v);
      if(op==='pow'){let z=v;for(let i=0;i<args[1];i++)z=action(args[0],z);return z;}
      if(op==='phi')return phiAction(w=>scale(A(w),args[1]),v,args[0],bound*Math.abs(args[1]),stats);
      if(op==='exp')return phiAction(w=>scale(A(w),args[0]??1),v,0,bound*Math.abs(args[0]??1),stats);
      throw Error(`Unknown exponential expression ${op}`);};
    let programName=method.program??(method.identifier==='etd2rk'?'ETDRK2':'Exprb32');const program=catalog.exponentialPrograms[programName];if(!program)throw Error('Exponential program unavailable');
    const rates=[],combineOps=coeff=>{let z=action(coeff[0],start);for(let i=1;i<coeff.length;i++)if(coeff[i]!==0)z=add(z,action(coeff[i],rates[i-1]));return z;};
    for(const call of program.calls){const z=combineOps(call.coefficients),r=scale(f(semilinear?t+h*call.c:z[0],semilinear?z:z.slice(1)),h);rates.push(semilinear?r:[h,...r]);}
    let z=combineOps(program.output),error=program.error?combineOps(program.error):null;if(!semilinear){z=z.slice(1);if(error)error=error.slice(1);}return {y:z,error};
  }
  function step(y,t,h,memory={}){
    let r;
    switch(method.kind){
      case 'rk':r=rk(method.tableau,y,t,h);break;
      case 'coupled':r=rk(method.tableau,y,t,h,true);break;
      case 'ark':r=ark(method.tableau,y,t,h);break;
      case 'mechanics':r=mechanics(method.identifier,y,t,h);break;
      case 'nystrom':r=nystrom(method.tableau,y,t,h);break;
      case 'newmark':r=structural(y,t,h,memory);break;
      case 'recurrence':r=recurrence(y,t,h);break;
      case 'history':case 'imex-history':r=history(y,t,h,memory);break;
      case 'extrapolation':r=extrapolate(y,t,h);break;
      case 'rosenbrock':case 'ros3p':r=rosenbrock(y,t,h);break;
      case 'exponential':r=exponential(y,t,h);break;
      default:throw Error(`Browser driver not implemented: ${method.kind}`);
    }
    observe(r.y);if(r.y.some(x=>!Number.isFinite(x)||Math.abs(x)>1e18))throw Error('Trajectory diverged; reduce the timestep or choose a stable method');return {...r,memory:r.memory??memory};
  }
  return {step,stats};
}
// Scaled Taylor exponential of an augmented block operator. The upper block of
// exp([[A,I,…],[0,0,I,…],…]) [0,…,v] is phi_k(A)v. Scaling also scales the
// identity blocks, avoiding the cancellation of unscaled negative-real series.
export function phiAction(A,v,k,bound,stats={}){
  const N=v.length,s=Math.max(1,Math.ceil((bound+1)/2));if(s>2048)throw Error('Exponential action budget exceeded; reduce dt');
  let result=zeros(N*(k+1));result.splice(k*N,N,...v);
  const apply=w=>{stats.actions=(stats.actions||0)+1;const z=zeros(w.length),head=A(w.slice(0,N));for(let i=0;i<N;i++)z[i]=head[i]/s;for(let j=0;j<k;j++)for(let i=0;i<N;i++)z[j*N+i]+=w[(j+1)*N+i]/s;return z;};
  for(let j=0;j<s;j++){let term=result.slice(),sum=result.slice(),converged=false;for(let l=1;l<=64;l++){term=scale(apply(term),1/l);sum=add(sum,term);if(norm(term)<1e-15*Math.max(1,norm(sum))){converged=true;break;}}if(!converged)throw Error('Exponential action did not converge');result=sum;}
  return result.slice(0,N);
}
export function solveCPU(method,problem,catalog,config={}){
  const dt=config.dt??.05,T=config.T??10;if(!(dt>0&&T>0&&Number.isFinite(dt)&&Number.isFinite(T)))throw Error('Positive finite timestep and duration required');
  if(config.adaptive&&!method.estimate_order)throw Error('This method has no supported embedded error estimate');
  const runner=createStepper(method,problem,catalog,config),start=performance.now(),trajectory=[],maxSteps=config.maxSteps??100000,rtol=config.rtol??1e-6,atol=config.atol??1e-9;
  let y=(config.y0??problem.y0).slice(),t=0,h=dt,memory={},accepted=0,rejected=0,attempts=0,failure=null;const stride=Math.max(1,Math.ceil(T/dt/600));
  const record=()=>{const exact=problem.exact?.(t);trajectory.push({t,y:y.slice(),energy:problem.energy?.(y)??null,error:exact?norm(add(y,exact,-1))/Math.sqrt(y.length):null,tv:problem.grid?tv(y):null,dt:h});if(trajectory.length>600){const retained=trajectory.filter((_,i)=>i%2===0||i===trajectory.length-1);trajectory.splice(0,trajectory.length,...retained);}};record();
  while(t<T-1e-13*Math.max(1,T)){
    if(++attempts>maxSteps){failure=`Step budget (${maxSteps}) exhausted`;break;}
    // Use a uniform grid fitted to the endpoint. Integer step clocks avoid
    // accumulated time drift corrupting fine-grid convergence measurements.
    const uniformStep=T/Math.ceil(T/dt);h=config.adaptive?Math.min(h,T-t):uniformStep;
    if(h<1e-14*Math.max(1,T)){failure='Timestep underflow';break;}
    let result;try{result=runner.step(y,t,h,memory);}catch(e){if(config.adaptive&&h>1e-12){h*=.25;rejected++;continue;}failure=e.message;break;}
    let ratio=0;
    if(config.adaptive){if(!result.error)throw Error('Driver returned no error estimate');ratio=Math.sqrt(result.error.reduce((s,x,i)=>s+(x/(atol+rtol*Math.max(Math.abs(y[i]),Math.abs(result.y[i]))))**2,0)/y.length);if(!Number.isFinite(ratio))ratio=Infinity;}
    if(!config.adaptive||ratio<=1){y=result.y;t=config.adaptive?Math.min(T,t+h):Math.min(T,(accepted+1)*h);memory=result.memory;accepted++;if(accepted%stride===0||t>=T-1e-13)record();}else rejected++;
    if(config.adaptive)h*=Math.max(.2,Math.min(5,.9*Math.max(ratio,1e-14)**(-1/method.estimate_order)));
  }
  const endReference=problem.exact?.(t),error=endReference?norm(add(y,endReference,-1))/Math.sqrt(y.length):null;
  return {method:method.identifier,name:method.friendly_name??method.name,order:method.order,backend:'CPU · float64',precision:'f64',trajectory,y,t,accepted,rejected,attempts,error,failure,elapsed:performance.now()-start,stats:runner.stats,actualDt:config.adaptive?null:T/Math.ceil(T/dt)};
}
export function convergenceCPU(method,problem,catalog,config={}){
  const levels=config.levels??6;if(!Number.isInteger(levels)||levels<3||levels>10)throw Error('Choose 3–10 refinement levels');
  const base=config.dt??.1,rows=[];let reference=null,referenceUncertainty=0;
  if(!problem.exact){const ref=catalog.schemes.find(s=>s.identifier==='radauIia3'),fine=base/2**(levels+2);const a=solveCPU(ref,problem,catalog,{...config,adaptive:false,dt:fine,maxSteps:200000}),b=solveCPU(ref,problem,catalog,{...config,adaptive:false,dt:fine/2,maxSteps:200000});if(a.failure||b.failure)throw Error(`Reference solve failed: ${a.failure??b.failure}`);reference=b.y;referenceUncertainty=norm(add(a.y,b.y,-1))/Math.sqrt(b.y.length);}
  for(let l=0;l<levels;l++){const dt=base/2**l,r=solveCPU(method,problem,catalog,{...config,adaptive:false,dt}),error=r.failure?null:reference?norm(add(r.y,reference,-1))/Math.sqrt(r.y.length):r.error,previous=rows.at(-1);const h=r.actualDt??dt;const floor=Math.max(2e-13,10*referenceUncertainty,16*Number.EPSILON*(1+r.accepted)*Math.max(1,norm(r.y)/Math.sqrt(r.y.length)),(r.stats.solveResidual??0)*r.accepted);const order=previous?.error>previous?.floor&&error>floor?Math.log(previous.error/error)/Math.log(previous.dt/h):null;rows.push({dt:h,error,order,floor,work:r.stats.rhs+r.stats.linear,time:r.elapsed,failure:r.failure});}
  return {method:method.identifier,rows,numericFloor:Math.max(...rows.map(r=>r.floor)),reference:reference?'Radau IIA 5 · two-grid checked numerical reference':'Analytic semi-discrete solution',referenceUncertainty};
}
export function geometryCPU(method,problem,catalog,config={}){
  if(!problem.mechanics||!problem.separable||problem.y0.length!==2*problem.m)throw Error('Use harmonic oscillator, Kepler, structural or forced mechanics for the canonical form test');
  if(['history','imex-history','newmark'].includes(method.kind))throw Error('The one-step canonical form test does not apply to methods with hidden acceleration/history state');
  const y=problem.y0,m=problem.m,N=y.length,h=config.dt??.1,epsilon=1e-5;const columns=y.map((_,j)=>{const a=y.slice(),b=y.slice();a[j]+=epsilon;b[j]-=epsilon;const A=createStepper(method,problem,catalog,config).step(a,0,h).y,B=createStepper(method,problem,catalog,config).step(b,0,h).y;return scale(add(A,B,-1),1/(2*epsilon));});
  const omega=(a,b)=>a.slice(0,m).reduce((s,x,i)=>s+x*b[m+i]-a[m+i]*b[i],0);let defect=0;for(let i=0;i<N;i++)for(let j=0;j<N;j++){const target=i<m&&j===m+i?1:j<m&&i===m+j?-1:0;defect+=(omega(columns[i],columns[j])-target)**2;}
  return {defect:Math.sqrt(defect),epsilon,dt:h,description:'Frobenius norm of DΦᵀ Ω DΦ − Ω; central differences of the numerical step map.'};
}
export function sensitivityCPU(method,problem,catalog,config={}){
  // Analytic adjoints of the fixed RHS are demonstrated by the continuous
  // forward tangent / adjoint duality, not claimed as AD through a solver.
  const y=problem.y0,v=y.map((_,i)=>Math.cos(i+1)),w=y.map((_,i)=>Math.sin(i+1)),jv=problem.jvp(0,y,v),jtw=problem.vjp(0,y,w),left=dot(w,jv),right=dot(v,jtw);
  const T=config.T??2,h=T/Math.ceil(T/(config.dt??.02)),steps=Math.ceil(T/h),states=[],direction=y.map((_,i)=>i===0?1:0);let z=[...y,...direction];
  const rhs=(t,z)=>[...problem.rhs(t,z.slice(0,y.length)),...problem.jvp(t,z.slice(0,y.length),z.slice(y.length))];
  for(let k=0;k<steps;k++){const t=k*h,a=rhs(t,z),b=rhs(t+h/2,add(z,a,h/2)),c=rhs(t+h/2,add(z,b,h/2)),d=rhs(t+h,add(z,c,h));z=combine(z,h,[a,b,c,d],[1/6,1/3,1/3,1/6]);if(k%Math.max(1,Math.ceil(steps/300))===0)states.push({t:t+h,value:norm(z.slice(y.length))});if(z.some(x=>!Number.isFinite(x)))throw Error('Tangent integration diverged; reduce dt');}
  return {left,right,dualityError:Math.abs(left-right),jvp:jv,vjp:jtw,tangent:z.slice(y.length),states,description:'Analytic Jv / Jᵀv duality at the initial state. Continuous tangent integrated with RK4; this is not differentiation of the selected integrator.'};
}
export function runTask(task,catalog){const problem=makeProblem(task.problem,task.params),methods=task.methods.map(id=>{const m=catalog.schemes.find(s=>s.identifier===id);if(!m)throw Error(`Unknown method ${id}`);return m;});return methods.map(method=>{try{const cfg=task.config??{};if(task.mode==='convergence')return convergenceCPU(method,problem,catalog,cfg);if(task.mode==='geometry')return geometryCPU(method,problem,catalog,cfg);if(task.mode==='sensitivity')return sensitivityCPU(method,problem,catalog,cfg);return solveCPU(method,problem,catalog,cfg);}catch(e){return {method:method.identifier,failure:e.message};}});}
