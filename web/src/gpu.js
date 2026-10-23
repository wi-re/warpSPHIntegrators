import {makeProblem,compatible,norm,tv} from './problems.js';
const f=x=>{if(!Number.isFinite(x))throw Error('Nonfinite shader constant');const s=Number(x).toPrecision(17);return s.includes('.')||s.includes('e')?s:s+'.0';};
const arr=(name,a)=>`const ${name}=array<f32,${a.length}>(${a.map(f).join(',')});`;
export function gpuSupport(method,problem,config={}){
  const issue=compatible(method,problem);if(issue)return issue;
  if(config.adaptive)return 'Adaptive acceptance uses the CPU float64 engine';
  if(['exponential','rosenbrock','extrapolation'].includes(method.kind))return 'This driver currently runs on the CPU float64 engine';
  if(problem.y0.length>64)return 'WebGPU private-array budget is limited to 64 state components';
  if(method.kind==='coupled'&&method.tableau.c.length*problem.y0.length>48)return 'Coupled stages exceed the GPU dense Newton budget (48 unknowns)';
  if(config.linearSolver==='gmres'&&method.implicit)return 'Matrix-free Newton–GMRES is implemented on the CPU engine';
  if(method.kind==='newmark'&&problem.y0.length>2*problem.m)return 'Coupled structural quantities currently use the CPU engine';
  if(!['rk','coupled','ark','mechanics','history','imex-history','recurrence','nystrom','newmark','ros3p'].includes(method.kind))return 'No WebGPU kernel for this driver';
  return null;
}
function problemWGSL(problem){
  const {id,p,m}=problem,N=problem.y0.length,n=p.n;let body='',jac='',ft='';
  const mech=problem.mechanics;
  if(mech){body+=`for(var j=0u;j<${m}u;j++){z[j]=y[j+${m}u];}`;jac+=`for(var j=0u;j<${m}u;j++){z[j]=v[j+${m}u];}`;
    if(id==='kepler'){body+='let r=sqrt(y[0]*y[0]+y[1]*y[1]);z[2]=-y[0]/(r*r*r);z[3]=-y[1]/(r*r*r);';jac+='let r=sqrt(y[0]*y[0]+y[1]*y[1]);let d=y[0]*v[0]+y[1]*v[1];for(var j=0u;j<2u;j++){z[j+2u]=-v[j]/pow(r,3.0)+3.0*y[j]*d/pow(r,5.0);}';}
    else if(id==='structural'){body+='z[2]=-800.5*y[0]+799.5*y[1];z[3]=799.5*y[0]-800.5*y[1];';jac+='z[2]=-800.5*v[0]+799.5*v[1];z[3]=799.5*v[0]-800.5*v[1];';}
    else if(id==='forced'){body+='z[1]=cos(t);';ft='z[1]=-sin(t);';}
    else if(id==='vanDerPol'){body+=`z[1]=${f(p.mu)}*(1.0-y[0]*y[0])*y[1]-y[0];`;jac+=`z[1]=(-1.0-2.0*${f(p.mu)}*y[0]*y[1])*v[0]+${f(p.mu)}*(1.0-y[0]*y[0])*v[1];`;}
    else{const k=id==='stiffDampedOscillator'?p.omega**2:p.k,c=id==='oscillator'?0:p.c;body+=`z[1]=-${f(k)}*y[0]-${f(c)}*y[1];`;jac+=`z[1]=-${f(k)}*v[0]-${f(c)}*v[1];`;if(id==='heated'){body+=`z[2]=${f(c)}*y[1]*y[1];`;jac+=`z[2]=2.0*${f(c)}*y[1]*v[1];`;}}
  }else if(id==='stiffRelaxation'){
    const s=p.forcing==='tanh'?'tanh(t)':p.forcing==='sin'?'sin(t)':'cos(t)',sp=p.forcing==='tanh'?'1.0-s*s':p.forcing==='sin'?'cos(t)':'-sin(t)',spp=p.forcing==='tanh'?'-2.0*s*sp':'-s';body+=`let s=${s};let sp=${sp};z[0]=-${f(p.sign*p.rate)}*(y[0]-s)+sp;`;jac+=`z[0]=-${f(p.sign*p.rate)}*v[0];`;ft+=`let s=${s};let sp=${sp};z[0]=${f(p.sign*p.rate)}*sp+${spp};`;
  }else if(id==='robertson'){body+='let a=0.04*y[0];let b=1e4*y[1]*y[2];let c=3e7*y[1]*y[1];z[0]=-a+b;z[1]=a-b-c;z[2]=c;';jac+='let b=1e4*(v[1]*y[2]+y[1]*v[2]);let c=6e7*y[1]*v[1];z[0]=-0.04*v[0]+b;z[1]=0.04*v[0]-b-c;z[2]=c;';}
  else if(id==='driftOscillator'){body+=`z[0]=y[1];z[1]=-y[0]-${f(p.kappa)}*y[2];z[2]=(1.0+0.5*y[0]*y[0])*y[1]-${f(p.gamma)}*y[2];`;jac+=`z[0]=v[1];z[1]=-v[0]-${f(p.kappa)}*v[2];z[2]=y[0]*y[1]*v[0]+(1.0+0.5*y[0]*y[0])*v[1]-${f(p.gamma)}*v[2];`;}
  else if(id==='acoustic'){body+=`for(var i=0u;i<${n}u;i++){z[i]=y[${n}u+i];z[${n}u+i]=-${f(p.speed**2*n)}*(y[${2*n}u+(i+1u)%${n}u]-y[${2*n}u+i]);z[${2*n}u+i]=-${f(n)}*(y[${n}u+i]-y[${n}u+(i+${n-1}u)%${n}u]);}`;jac=body.replaceAll('y[','v[');}
  else{
    const left=id==='diffusion'?'select(y[i-1u],0.0,i==0u)':`y[(i+${n-1}u)%${n}u]`,right=id==='diffusion'?`select(y[i+1u],0.0,i==${n-1}u)`:`y[(i+1u)%${n}u]`;
    if(id==='burgers'){body+='var alpha=0.0;for(var j=0u;j<N;j++){alpha=max(alpha,abs(y[j]));}';jac+='var alpha=0.0;var da=0.0;for(var j=0u;j<N;j++){if(abs(y[j])>alpha){alpha=abs(y[j]);da=sign(y[j])*v[j];}}';}
    body+=`for(var i=0u;i<N;i++){let l=${left};let r=${right};let u=y[i];`;
    jac+=`for(var i=0u;i<N;i++){let l=${left};let r=${right};let u=y[i];let vl=${left.replaceAll('y[','v[')};let vr=${right.replaceAll('y[','v[')};let vu=v[i];`;
    if(id==='diffusion'){body+=`z[i]=${f(p.D*(n+1)**2)}*(l-2.0*u+r);`;jac+=`z[i]=${f(p.D*(n+1)**2)}*(vl-2.0*vu+vr);`;}
    else if(id.startsWith('advection')){body+=`z[i]=${f(p.speed*n)}*(l-u);`;jac+=`z[i]=${f(p.speed*n)}*(vl-vu);`;}
    else if(id==='viscousBurgers'){body+=`let diffusion=${f(p.nu*n*n)}*(l-2.0*u+r);let advection=${f(.25*n)}*(l*l-r*r);z[i]=select(select(diffusion+advection,diffusion,part==1u),advection,part==2u);`;jac+=`let diffusion=${f(p.nu*n*n)}*(vl-2.0*vu+vr);let advection=${f(.5*n)}*(l*vl-r*vr);z[i]=select(select(diffusion+advection,diffusion,part==1u),advection,part==2u);`;}
    else{body+=`z[i]=${f(.25*n)}*(l*l-r*r)+${f(.5*n)}*alpha*(l-2.0*u+r);`;jac+=`z[i]=${f(.5*n)}*(l*vl-r*vr+da*(l-2.0*u+r)+alpha*(vl-2.0*vu+vr));`;}
    body+='}';jac+='}';
  }
  const split=id==='viscousBurgers'?'':'if(part==2u){return z;}';
  return `fn raw(t:f32,y:Vec,part:u32)->Vec{var z:Vec;${split}${body}return z;}
fn rhs(t:f32,y:Vec,part:u32)->Vec{rhsCount++;observe(y);return raw(t,y,part);}
fn jvp(t:f32,y:Vec,v:Vec,part:u32)->Vec{var z:Vec;${split}${jac}return z;}
fn partial(t:f32)->Vec{var z:Vec;${ft}return z;}`;
}
function rkFunction(name,tab){const S=tab.c.length,explicit=tab.a.every((row,i)=>row[i]===0);
 return `${arr(name+'A',tab.a.flat())}${arr(name+'B',tab.b)}${arr(name+'C',tab.c)}
fn ${name}(y:Vec,t:f32,h:f32)->Vec{var k:array<Vec,${S}>;
for(var i=0u;i<${S}u;i++){var z=y;for(var j=0u;j<i;j++){z=vadd(z,k[j],h*${name}A[i*${S}u+j]);}
${explicit?'':`if(${name}A[i*${S}u+i]!=0.0){z=implicit(z,z,t+h*${name}C[i],h*${name}A[i*${S}u+i],0u);}`}
k[i]=rhs(t+h*${name}C[i],z,0u);}var result=y;for(var j=0u;j<${S}u;j++){result=vadd(result,k[j],h*${name}B[j]);}return result;}`;
}
function implicitWGSL(N){return `
fn implicit(base:Vec,seed:Vec,t:f32,c:f32,part:u32)->Vec{
var z=seed;
for(var iteration=0u;iteration<14u;iteration++){
 let rate=rhs(t,z,part);let r=vadd(vadd(z,base,-1.0),rate,-c);let rn=vnorm(r);residualNorm=rn;
 if(rn<=solveTol*(1.0+vnorm(z))){return z;}
 var mat:array<f32,${N*(N+1)}>;
 for(var col=0u;col<N;col++){var e=Vec();e[col]=1.0;let jv=jvp(t,z,e,part);for(var row=0u;row<N;row++){mat[row*(N+1u)+col]=select(0.0,1.0,row==col)-c*jv[row];}}
 for(var row=0u;row<N;row++){mat[row*(N+1u)+N]=-r[row];}
 for(var col=0u;col<N;col++){var pivot=col;for(var row=col+1u;row<N;row++){if(abs(mat[row*(N+1u)+col])>abs(mat[pivot*(N+1u)+col])){pivot=row;}}
 if(abs(mat[pivot*(N+1u)+col])<1e-20){failure=2.0;return z;}
 for(var j=col;j<=N;j++){let tmp=mat[col*(N+1u)+j];mat[col*(N+1u)+j]=mat[pivot*(N+1u)+j];mat[pivot*(N+1u)+j]=tmp;}
 for(var row=col+1u;row<N;row++){let factor=mat[row*(N+1u)+col]/mat[col*(N+1u)+col];for(var j=col+1u;j<=N;j++){mat[row*(N+1u)+j]-=factor*mat[col*(N+1u)+j];}}}
 var delta:Vec;for(var rev=0u;rev<N;rev++){let row=N-1u-rev;var value=mat[row*(N+1u)+N];for(var j=row+1u;j<N;j++){value-=mat[row*(N+1u)+j]*delta[j];}delta[row]=value/mat[row*(N+1u)+row];}
 linearCount++;newtonCount++;var factor=1.0;var candidate=vadd(z,delta,factor);
 for(var line=0u;line<6u;line++){let rr=vadd(vadd(candidate,base,-1.0),rhs(t,candidate,part),-c);if(vnorm(rr)<=rn*(1.0-1e-4*factor)){break;}factor*=0.5;candidate=vadd(z,delta,factor);}z=candidate;
}failure=2.0;return z;}
`}
function coupledWGSL(tab,N){const S=tab.c.length,Q=N*S;return `${arr('ca',tab.a.flat())}${arr('cb',tab.b)}${arr('cc',tab.c)}
alias Block=array<f32,${Q}>;
fn blockResidual(z:Block,y:Vec,t:f32,h:f32)->Block{var rates:array<Vec,${S}>;for(var i=0u;i<${S}u;i++){var stage:Vec;for(var j=0u;j<N;j++){stage[j]=z[i*N+j];}rates[i]=rhs(t+h*cc[i],stage,0u);}var r:Block;for(var i=0u;i<${S}u;i++){for(var j=0u;j<N;j++){var value=z[i*N+j]-y[j];for(var k=0u;k<${S}u;k++){value-=h*ca[i*${S}u+k]*rates[k][j];}r[i*N+j]=value;}}return r;}
fn advance(y:Vec,t:f32,h:f32,index:u32)->Vec{
 var z:Block;for(var i=0u;i<${S}u;i++){for(var j=0u;j<N;j++){z[i*N+j]=y[j];}}
 for(var iteration=0u;iteration<14u;iteration++){let r=blockResidual(z,y,t,h);var rn=0.0;var zn=0.0;for(var i=0u;i<${Q}u;i++){rn+=r[i]*r[i];zn+=z[i]*z[i];}residualNorm=sqrt(rn);if(sqrt(rn)<=solveTol*(1.0+sqrt(zn))){break;}
 var mat:array<f32,${Q*(Q+1)}>;
 for(var col=0u;col<${Q}u;col++){let block=col/N;let component=col%N;var e=Vec();e[component]=1.0;var stage:Vec;for(var j=0u;j<N;j++){stage[j]=z[block*N+j];}let jv=jvp(t+h*cc[block],stage,e,0u);for(var i=0u;i<${S}u;i++){for(var j=0u;j<N;j++){let row=i*N+j;mat[row*${Q+1}u+col]=select(0.0,1.0,row==col)-h*ca[i*${S}u+block]*jv[j];}}}
 for(var row=0u;row<${Q}u;row++){mat[row*${Q+1}u+${Q}u]=-r[row];}
 for(var col=0u;col<${Q}u;col++){var pivot=col;for(var row=col+1u;row<${Q}u;row++){if(abs(mat[row*${Q+1}u+col])>abs(mat[pivot*${Q+1}u+col])){pivot=row;}}if(abs(mat[pivot*${Q+1}u+col])<1e-20){failure=2.0;return y;}for(var j=col;j<=${Q}u;j++){let tmp=mat[col*${Q+1}u+j];mat[col*${Q+1}u+j]=mat[pivot*${Q+1}u+j];mat[pivot*${Q+1}u+j]=tmp;}for(var row=col+1u;row<${Q}u;row++){let factor=mat[row*${Q+1}u+col]/mat[col*${Q+1}u+col];for(var j=col+1u;j<=${Q}u;j++){mat[row*${Q+1}u+j]-=factor*mat[col*${Q+1}u+j];}}}
 var delta:Block;for(var rev=0u;rev<${Q}u;rev++){let row=${Q-1}u-rev;var value=mat[row*${Q+1}u+${Q}u];for(var j=row+1u;j<${Q}u;j++){value-=mat[row*${Q+1}u+j]*delta[j];}delta[row]=value/mat[row*${Q+1}u+row];}for(var j=0u;j<${Q}u;j++){z[j]+=delta[j];}linearCount++;newtonCount++;if(iteration==13u){failure=2.0;return y;}}
 var result=y;for(var i=0u;i<${S}u;i++){var stage:Vec;for(var j=0u;j<N;j++){stage[j]=z[i*N+j];}result=vadd(result,rhs(t+h*cc[i],stage,0u),h*cb[i]);}return result;
}`;}
export function shaderSource(method,problem,catalog,config={}){
  const support=gpuSupport(method,problem,config);if(support)throw Error(support);
  const N=problem.y0.length,m=problem.m,lookup=id=>catalog.schemes.find(s=>s.identifier===id);let functions='',advance='';
  if(method.kind==='rk'){functions=rkFunction('rkmain',method.tableau);advance='return rkmain(y,t,h);';}
  if(method.kind==='coupled')functions=coupledWGSL(method.tableau,N);
  if(method.kind==='ark'){const tab=method.tableau,S=tab.c.length;functions=[arr('ae',tab.a_explicit.flat()),arr('ai',tab.a_implicit.flat()),arr('be',tab.b_explicit),arr('bi',tab.b_implicit),arr('clock',tab.c)].join('');advance=`var e:array<Vec,${S}>;var im:array<Vec,${S}>;for(var i=0u;i<${S}u;i++){var z=y;for(var j=0u;j<i;j++){z=vadd(vadd(z,e[j],h*ae[i*${S}u+j]),im[j],h*ai[i*${S}u+j]);}if(ai[i*${S}u+i]!=0.0){z=implicit(z,z,t+h*clock[i],h*ai[i*${S}u+i],1u);}e[i]=rhs(t+h*clock[i],z,2u);im[i]=rhs(t+h*clock[i],z,1u);}var z=y;for(var j=0u;j<${S}u;j++){z=vadd(vadd(z,e[j],h*be[j]),im[j],h*bi[j]);}return z;`;}
  if(method.kind==='mechanics'){
    const id=method.identifier;
    if(id==='semiImplicitEuler')advance=`var z=vadd(y,rhs(t,y,0u),h);for(var j=0u;j<${m}u;j++){z[j]=y[j]+h*z[j+${m}u];}return z;`;
    else if(id==='symplecticEuler')advance=`let k0=rhs(t,y,0u);let k1=rhs(t+h/2.0,vadd(y,k0,h/2.0),0u);var z=vadd(y,k1,h);for(var j=0u;j<${m}u;j++){z[j]=y[j]+h/2.0*(y[j+${m}u]+z[j+${m}u]);}return z;`;
    else if(id==='velocityVerlet'||id==='leapFrog')advance=`let k0=rhs(t,y,0u);var z=y;for(var j=0u;j<${m}u;j++){z[j]+=h*y[j+${m}u]+h*h/2.0*k0[j+${m}u];}${id==='velocityVerlet'?`for(var j=${m}u;j<${2*m}u;j++){z[j]+=h/2.0*k0[j];}`:''}let k1=rhs(t+h,z,0u);for(var j=${m}u;j<${2*m}u;j++){z[j]=y[j]+h/2.0*(k0[j]+k1[j]);}for(var j=${2*m}u;j<N;j++){z[j]=y[j]+${id==='leapFrog'?'h/2.0*(k0[j]+k1[j])':'h*k1[j]'};}return z;`;
    else {const xi=id==='pefrl'?.1786178958448091:.1720865590295143,lambda=id==='pefrl'?-.2123418310626054:-.09156203075515678,chi=id==='pefrl'?-.06626458266981849:-.1616217622107222,drift=id==='pefrl'?[xi,chi,1-2*(xi+chi),chi,xi]:[(1-2*lambda)/2,lambda,lambda,(1-2*lambda)/2],kick=id==='pefrl'?[(1-2*lambda)/2,lambda,lambda,(1-2*lambda)/2]:[xi,chi,1-2*(xi+chi),chi,xi];functions=arr('drifts',drift)+arr('kicks',kick);advance='var z=y;var clock=t;';if(id==='pefrl')advance+=`for(var i=0u;i<5u;i++){for(var j=0u;j<${m}u;j++){z[j]+=h*drifts[i]*z[j+${m}u];}clock+=h*drifts[i];if(i<4u){let k=rhs(clock,z,0u);for(var j=${m}u;j<N;j++){z[j]+=h*kicks[i]*k[j];}}}`;else advance+=`for(var i=0u;i<5u;i++){let k=rhs(clock,z,0u);for(var j=${m}u;j<N;j++){z[j]+=h*kicks[i]*k[j];}if(i<4u){for(var j=0u;j<${m}u;j++){z[j]+=h*drifts[i]*z[j+${m}u];}clock+=h*drifts[i];}}`;advance+='return z;';}
  }
  if(method.kind==='nystrom'){const a=method.tableau,S=a.c.length;functions=arr('na',a.a.map(row=>[...row,...Array(S-row.length).fill(0)]).flat())+arr('nc',a.c)+arr('nb',a.b)+arr('np',a.bp)+(a.abar?arr('nv',a.abar.map(row=>[...row,...Array(S-row.length).fill(0)]).flat()):'');advance=`var ks:array<Vec,${S}>;for(var i=0u;i<${S}u;i++){var z=y;for(var j=0u;j<${m}u;j++){z[j]+=h*nc[i]*y[j+${m}u];for(var l=0u;l<i;l++){z[j]+=h*h*na[i*${S}u+l]*ks[l][j+${m}u];}}${a.abar?`for(var j=${m}u;j<N;j++){for(var l=0u;l<i;l++){z[j]+=h*nv[i*${S}u+l]*ks[l][j];}}`:''}ks[i]=rhs(t+h*nc[i],z,0u);}var z=y;for(var j=0u;j<${m}u;j++){z[j]+=h*y[j+${m}u];for(var i=0u;i<${S}u;i++){z[j]+=h*h*nb[i]*ks[i][j+${m}u];}}for(var j=${m}u;j<${a.abar?'N':2*m+'u'};j++){for(var i=0u;i<${S}u;i++){z[j]+=h*np[i]*ks[i][j];}}return z;`;}
  if(method.kind==='recurrence'){const fam=method.identifier.replace('stabilized_',''),S=config.stages??9,data=catalog.recurrences[fam]?.[S];if(!data)throw Error('Unsupported recurrence stage count');functions=arr('rmu',data.mu)+arr('rnu',data.nu)+arr('rmt',data.mtilde)+arr('rgt',data.gtilde)+arr('rc',data.c);advance=`let f0=rhs(t,y,0u);var prev=y;var z=vadd(y,f0,h*rmt[1]);for(var j=2u;j<=${S}u;j++){let rate=rhs(t+h*rc[j-1u],z,0u);var next:Vec;for(var i=0u;i<N;i++){next[i]=y[i]+rmu[j]*(z[i]-y[i])+rnu[j]*(prev[i]-y[i])+h*(rmt[j]*rate[i]+rgt[j]*f0[i]);}prev=z;z=next;}return z;`;}
  if(['history','imex-history'].includes(method.kind)){
    const id=method.identifier,q=method.order,AB={1:[1],2:[1.5,-.5],3:[23/12,-16/12,5/12],4:[55/24,-59/24,37/24,-9/24],5:[1901/720,-2774/720,2616/720,-1274/720,251/720]},AM={2:[.5,.5],3:[5/12,8/12,-1/12],4:[9/24,19/24,-5/24,1/24],5:[251/720,646/720,-264/720,106/720,-19/720]},BDF={1:[[1],1],2:[[4/3,-1/3],2/3],3:[[18/11,-9/11,2/11],6/11],4:[[48/25,-36/25,16/25,-3/25],12/25],5:[[300/137,-300/137,200/137,-75/137,12/137],60/137]};
    let needed=q-1,starter='dormandPrince';if(id==='ssprkmsvs32')needed=2;if(id==='ssprkmsvs43')needed=3;if((id.startsWith('bdf')||(id.startsWith('sbdf')&&problem.id!=='viscousBurgers'))&&q>=3)starter={3:'esdirk324l2sa',4:'kencarp4',5:'kencarp5'}[q];if(id==='cnlf2')needed=1;
    functions='var<private> pastY:array<Vec,5>;var<private> pastF:array<Vec,5>;'+rkFunction('startup',lookup(starter).tableau);
    advance='let current=rhs(t,y,0u);var z=y;';
    const startup=id==='cnlf2'?'implicit(vadd(y,rhs(t,y,2u),h),y,t+h,h,1u)':'startup(y,t,h)';advance+=`if(index<${needed}u){z=${startup};}else{`;
    if(id.startsWith('ab')){functions+=arr('hab',AB[q]);advance+=`for(var j=0u;j<${q}u;j++){var rate=current;if(j>0u){rate=pastF[j-1u];}z=vadd(z,rate,h*hab[j]);}`;if(id.startsWith('abm')){functions+=arr('ham',AM[q]);advance+=`let predicted=rhs(t+h,z,0u);z=vadd(y,predicted,h*ham[0]);for(var j=1u;j<${q}u;j++){var rate=current;if(j>1u){rate=pastF[j-2u];}z=vadd(z,rate,h*ham[j]);}`;}}
    else if(id.startsWith('am')){functions+=arr('ham',AM[q]);advance+=`for(var j=1u;j<${q}u;j++){var rate=current;if(j>1u){rate=pastF[j-2u];}z=vadd(z,rate,h*ham[j]);}z=implicit(z,z,t+h,h*ham[0],0u);`;}
    else if(id.startsWith('bdf')||id.startsWith('sbdf')){const [weights,b]=BDF[q];functions+=arr('hweights',weights);advance+=`var base:Vec;for(var j=0u;j<${q}u;j++){var value=y;if(j>0u){value=pastY[j-1u];}base=vadd(base,value,hweights[j]);}`;if(id.startsWith('sbdf')){functions+=arr('extrap',q===2?[2,-1]:[3,-3,1]);advance+=`for(var j=0u;j<${q}u;j++){var value=y;if(j>0u){value=pastY[j-1u];}base=vadd(base,rhs(t-f32(j)*h,value,2u),h*${f(b)}*extrap[j]);}`;}advance+=`z=implicit(base,base,t+h,h*${f(b)},${id.startsWith('sbdf')?'1':'0'}u);`;}
    else if(id==='ssprkmsvs32')advance+='for(var j=0u;j<N;j++){z[j]=0.75*(y[j]+2.0*h*current[j])+0.25*pastY[1][j];}';
    else if(id==='ssprkmsvs43')advance+='for(var j=0u;j<N;j++){z[j]=16.0/27.0*(y[j]+3.0*h*current[j])+11.0/27.0*(pastY[2][j]+12.0/11.0*h*pastF[2][j]);}';
    else if(id==='imexEuler')advance+='let base=vadd(y,rhs(t,y,2u),h);z=implicit(base,base,t+h,h,1u);';
    else if(id==='cnab2')advance+='var base=vadd(y,rhs(t,y,1u),h/2.0);base=vadd(base,rhs(t,y,2u),1.5*h);base=vadd(base,rhs(t-h,pastY[0],2u),-0.5*h);z=implicit(base,base,t+h,h/2.0,1u);';
    else if(id==='cnlf2')advance+='var base=vadd(pastY[0],rhs(t-h,pastY[0],1u),h);base=vadd(base,rhs(t,y,2u),2.0*h);z=implicit(base,base,t+h,h,1u);';
    advance+='}for(var j=4u;j>0u;j--){pastY[j]=pastY[j-1u];pastF[j]=pastF[j-1u];}pastY[0]=y;pastF[0]=current;return z;';
  }
  if(method.kind==='newmark'){
    const alpha=method.identifier==='generalizedAlpha',rho=config.rhoInfinity??.5,am=alpha?(2*rho-1)/(rho+1):0,af=alpha?rho/(rho+1):0,g=alpha?.5-am+af:config.gamma??.5,b=alpha?(1-am+af)**2/4:config.beta??.25;
    // A modified first-order RHS in acceleration coordinates lets the same dense
    // analytic Newton closure solve the algorithmic acceleration equation.
    functions=`var<private> lastAcceleration:Vec;`;
    advance=`let force0=rhs(t,y,0u);var a0=force0;if(index>0u&&${alpha?'true':'false'}){a0=lastAcceleration;}
var a=a0;for(var it=0u;it<14u;it++){var stage=y;for(var j=0u;j<${m}u;j++){stage[j]+=h*y[j+${m}u]+h*h*((${f(.5-b)})*a0[j+${m}u]+${f(b)}*a[j+${m}u]);stage[j+${m}u]+=h*(${f(1-g)}*a0[j+${m}u]+${f(g)}*a[j+${m}u]);}
var weighted:Vec;for(var j=0u;j<N;j++){weighted[j]=${f(af)}*y[j]+${f(1-af)}*stage[j];}let force=rhs(t+${f(1-af)}*h,weighted,0u);var residual:array<f32,${m}>;var rn=0.0;for(var j=0u;j<${m}u;j++){residual[j]=${f(1-am)}*a[j+${m}u]+${f(am)}*a0[j+${m}u]-force[j+${m}u];rn+=residual[j]*residual[j];}if(sqrt(rn)<solveTol*(1.0+vnorm(a))){lastAcceleration=a;return stage;}
var mat:array<f32,${m*(m+1)}>;for(var col=0u;col<${m}u;col++){var direction=Vec();direction[col]=${f((1-af)*b)}*h*h;direction[col+${m}u]=${f((1-af)*g)}*h;let jv=jvp(t+${f(1-af)}*h,weighted,direction,0u);for(var row=0u;row<${m}u;row++){mat[row*${m+1}u+col]=select(0.0,${f(1-am)},row==col)-jv[row+${m}u];}}
for(var row=0u;row<${m}u;row++){mat[row*${m+1}u+${m}u]=-residual[row];}for(var col=0u;col<${m}u;col++){if(abs(mat[col*${m+1}u+col])<1e-20){failure=2.0;return y;}for(var row=col+1u;row<${m}u;row++){let factor=mat[row*${m+1}u+col]/mat[col*${m+1}u+col];for(var j=col+1u;j<=${m}u;j++){mat[row*${m+1}u+j]-=factor*mat[col*${m+1}u+j];}}}var delta:array<f32,${m}>;for(var rev=0u;rev<${m}u;rev++){let row=${m-1}u-rev;var value=mat[row*${m+1}u+${m}u];for(var j=row+1u;j<${m}u;j++){value-=mat[row*${m+1}u+j]*delta[j];}delta[row]=value/mat[row*${m+1}u+row];}for(var j=0u;j<${m}u;j++){a[j+${m}u]+=delta[j];}linearCount++;newtonCount++;}failure=2.0;return y;`;
  }
  // The fixed two-mode problem has an exact diagonal stiffness representation.
  // Solve its linear acceleration equation there: subtracting two ~800*q forces
  // in physical coordinates cannot satisfy an absolute f32 Newton tolerance.
  if(method.kind==='newmark'&&problem.id==='structural'){
    const alpha=method.identifier==='generalizedAlpha',rho=config.rhoInfinity??.5,am=alpha?(2*rho-1)/(rho+1):0,af=alpha?rho/(rho+1):0,g=alpha?.5-am+af:config.gamma??.5,b=alpha?(1-am+af)**2/4:config.beta??.25;
    functions='var<private> modalAcceleration:array<f32,2>;';
    advance=`let inv=0.7071067811865476;var q=array<f32,2>((y[0]+y[1])*inv,(y[0]-y[1])*inv);var v=array<f32,2>((y[2]+y[3])*inv,(y[2]-y[3])*inv);let stiffness=array<f32,2>(1.0,1600.0);
for(var j=0u;j<2u;j++){var a0=-stiffness[j]*q[j];if(index>0u&&${alpha?'true':'false'}){a0=modalAcceleration[j];}let predicted=q[j]+${f(1-af)}*(h*v[j]+h*h*${f(.5-b)}*a0);let right=-stiffness[j]*predicted-${f(am)}*a0;let diagonal=${f(1-am)}+stiffness[j]*${f((1-af)*b)}*h*h;let a1=right/diagonal;residualNorm=max(residualNorm,abs(diagonal*a1-right));q[j]+=h*v[j]+h*h*(${f(.5-b)}*a0+${f(b)}*a1);v[j]+=h*(${f(1-g)}*a0+${f(g)}*a1);modalAcceleration[j]=a1;}linearCount++;var z:Vec;z[0]=(q[0]+q[1])*inv;z[1]=(q[0]-q[1])*inv;z[2]=(v[0]+v[1])*inv;z[3]=(v[0]-v[1])*inv;return z;`;
  }
  if(method.kind==='ros3p'){
    const g=.5+Math.sqrt(3)/6;functions=`fn frozenSolve(y:Vec,t:f32,h:f32,b:Vec)->Vec{
// Solve the affine frozen-Jacobian stage with the common Newton linear closure.
// A separate dense assembly keeps the linearized stage distinct from a DIRK solve.
var mat:array<f32,${N*(N+1)}>;for(var col=0u;col<N;col++){var e=Vec();e[col]=1.0;let jv=jvp(t,y,e,0u);for(var row=0u;row<N;row++){mat[row*(N+1u)+col]=select(0.0,1.0,row==col)-h*${f(g)}*jv[row];}}for(var row=0u;row<N;row++){mat[row*(N+1u)+N]=b[row];}
for(var col=0u;col<N;col++){if(abs(mat[col*(N+1u)+col])<1e-20){failure=2.0;return y;}for(var row=col+1u;row<N;row++){let factor=mat[row*(N+1u)+col]/mat[col*(N+1u)+col];for(var j=col+1u;j<=N;j++){mat[row*(N+1u)+j]-=factor*mat[col*(N+1u)+j];}}}var x:Vec;for(var rev=0u;rev<N;rev++){let row=N-1u-rev;var value=mat[row*(N+1u)+N];for(var j=row+1u;j<N;j++){value-=mat[row*(N+1u)+j]*x[j];}x[row]=value/mat[row*(N+1u)+row];}linearCount++;return x;}`;
    advance=`let ft=partial(t);let k0=frozenSolve(y,t,h,vadd(rhs(t,y,0u),ft,h*${f(g)}));let stage=vadd(y,k0,h);let f1=rhs(t+h,stage,0u);let k1=frozenSolve(y,t,h,vadd(vadd(f1,jvp(t,y,k0,0u),-h),ft,h*${f(g-1)}));let w=vadd(vscale(k0,${f(-g)}),k1,${f(.5-2*g)});let k2=frozenSolve(y,t,h,vadd(vadd(f1,jvp(t,y,w,0u),h),ft,h*${f(.5-2*g)}));return vadd(vadd(y,k0,h*2.0/3.0),k2,h/3.0);`;
  }
  const initial=arr('initial',problem.y0),obs=problem.grid?`var variation=0.0;for(var j=0u;j<N;j++){variation+=abs(y[j]-y[(j+1u)%N]);overshoot=max(overshoot,max(y[j]-${f(Math.max(...problem.y0))},${f(Math.min(...problem.y0))}-y[j]));}stageTV=max(stageTV,variation);`:'';
  return `// Generated from the exported library coefficients; fixed f32 layout.
const N=${N}u;alias Vec=array<f32,${N}>;
const solveTol=${f(Math.max(config.solveTolerance??1e-6,2e-7))};
var<private> rhsCount:u32;var<private> linearCount:u32;var<private> newtonCount:u32;
var<private> failure:f32;var<private> residualNorm:f32;var<private> stageTV:f32;var<private> overshoot:f32;
fn observe(y:Vec){${obs}}
fn vadd(a:Vec,b:Vec,c:f32)->Vec{var z:Vec;for(var j=0u;j<N;j++){z[j]=a[j]+c*b[j];}return z;}
fn vscale(a:Vec,c:f32)->Vec{var z:Vec;for(var j=0u;j<N;j++){z[j]=a[j]*c;}return z;}
fn vnorm(a:Vec)->f32{var s=0.0;for(var j=0u;j<N;j++){s+=a[j]*a[j];}return sqrt(s);}
${problemWGSL(problem)}${implicitWGSL(N)}${initial}${functions}
${method.kind==='coupled'?'':`fn advance(y:Vec,t:f32,h:f32,index:u32)->Vec{${advance}}`}
struct Job{steps:u32,stride:u32,h:f32,pad:f32};
@group(0) @binding(0) var<storage,read> jobs:array<Job>;
@group(0) @binding(1) var<storage,read_write> output:array<f32>;
fn store(base:u32,sample:u32,t:f32,y:Vec){let offset=base+12u+sample*(N+1u);output[offset]=t;for(var j=0u;j<N;j++){output[offset+1u+j]=y[j];}}
@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) gid:vec3<u32>){if(gid.x>=arrayLength(&jobs)){return;}let job=jobs[gid.x];let base=gid.x*(12u+514u*(N+1u));rhsCount=0u;linearCount=0u;newtonCount=0u;failure=0.0;stageTV=0.0;overshoot=0.0;var y=initial;var count=1u;var completed=0u;store(base,0u,0.0,y);
for(var index=0u;index<job.steps;index++){let next=advance(y,f32(index)*job.h,job.h,index);for(var j=0u;j<N;j++){if(!(abs(next[j])<1e18)){failure=1.0;}}if(failure!=0.0){break;}y=next;completed=index+1u;observe(y);if(completed%job.stride==0u||completed==job.steps){store(base,count,f32(completed)*job.h,y);count++;}}
if(completed>0u&&completed%job.stride!=0u&&completed!=job.steps){store(base,count,f32(completed)*job.h,y);count++;}
output[base]=f32(completed);output[base+1u]=f32(rhsCount);output[base+2u]=f32(newtonCount);output[base+3u]=f32(linearCount);output[base+4u]=stageTV;output[base+5u]=overshoot;output[base+6u]=failure;output[base+7u]=residualNorm;output[base+8u]=f32(count);
}`;
}
export class GPUEngine{
  constructor(device){this.device=device;this.cache=new Map();this.lost=null;device.lost.then(info=>{this.lost=info.message||'WebGPU device lost';this.cache.clear();});}
  static async create(){if(!globalThis.navigator?.gpu)throw Error('WebGPU unavailable; use a secure context and a WebGPU browser');const adapter=await navigator.gpu.requestAdapter();if(!adapter)throw Error('No WebGPU adapter available');const engine=new GPUEngine(await adapter.requestDevice());engine.adapter=adapter.info;return engine;}
  async solve(method,problem,catalog,configs){
    if(this.lost)throw Error(this.lost);const config=configs[0],code=shaderSource(method,problem,catalog,config),device=this.device,key=code;let pipeline=this.cache.get(key);
    if(!pipeline){const module=device.createShaderModule({label:method.name,code});const messages=(await module.getCompilationInfo()).messages.filter(m=>m.type==='error');if(messages.length)throw Error(messages.map(m=>`WGSL ${m.lineNum}: ${m.message}`).join('\n'));pipeline=await device.createComputePipelineAsync({layout:'auto',compute:{module,entryPoint:'main'}});this.cache.set(key,pipeline);if(this.cache.size>24)this.cache.delete(this.cache.keys().next().value);}
    const N=problem.y0.length,recordSize=12+514*(N+1),data=new ArrayBuffer(configs.length*16),view=new DataView(data),jobs=configs.map((c,i)=>{if(!(Number.isFinite(c.T)&&Number.isFinite(c.dt)&&c.T>0&&c.dt>0))throw Error('Positive finite duration and dt required');const steps=Math.ceil(c.T/c.dt);if(steps>100000)throw Error('GPU step budget (100000) exceeded');const h=c.T/steps,stride=Math.max(1,Math.ceil(steps/512));view.setUint32(i*16,steps,true);view.setUint32(i*16+4,stride,true);view.setFloat32(i*16+8,h,true);return {h,steps,stride};});
    let input,output,readback;const start=performance.now();device.pushErrorScope('validation');
    try{
      input=device.createBuffer({size:data.byteLength,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST});output=device.createBuffer({size:recordSize*configs.length*4,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC});readback=device.createBuffer({size:output.size,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});device.queue.writeBuffer(input,0,data);
      const bind=device.createBindGroup({layout:pipeline.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:input}},{binding:1,resource:{buffer:output}}]}),encoder=device.createCommandEncoder(),pass=encoder.beginComputePass();pass.setPipeline(pipeline);pass.setBindGroup(0,bind);pass.dispatchWorkgroups(configs.length);pass.end();encoder.copyBufferToBuffer(output,0,readback,0,output.size);device.queue.submit([encoder.finish()]);await readback.mapAsync(GPUMapMode.READ);const values=new Float32Array(readback.getMappedRange()).slice();readback.unmap();const validation=await device.popErrorScope();if(validation)throw Error(validation.message);
      const elapsed=performance.now()-start;return jobs.map((job,i)=>{const offset=i*recordSize,accepted=values[offset],failure=values[offset+6]===1?'Trajectory diverged; reduce dt':values[offset+6]===2?'GPU Newton solve failed; reduce dt or use CPU float64':null,trajectory=[];for(let k=0;k<values[offset+8];k++){const row=offset+12+k*(N+1),t=values[row],y=Array.from(values.slice(row+1,row+1+N)),ref=problem.exact?.(t);trajectory.push({t,y,error:ref?norm(y.map((x,j)=>x-ref[j]))/Math.sqrt(N):null,energy:problem.energy?.(y)??null,tv:problem.grid?tv(y):null,dt:job.h});}const last=trajectory.at(-1),t=accepted*job.h;return {method:method.identifier,name:method.friendly_name??method.name,order:method.order,backend:'WebGPU · float32',precision:'f32',trajectory,y:last.y,t,accepted,rejected:0,attempts:accepted,error:last.error,elapsed,failure,actualDt:job.h,stats:{rhs:values[offset+1],newton:values[offset+2],linear:values[offset+3],stageTV:values[offset+4],stageOvershoot:values[offset+5],residual:values[offset+7]},timingNote:`Batch wall time includes submission and readback (${configs.length} jobs); compilation excluded.`};});
    }catch(e){await device.popErrorScope().catch(()=>{});throw e;}finally{input?.destroy();output?.destroy();readback?.destroy();}
  }
  destroy(){this.device.destroy();}
}
