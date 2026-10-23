// Absolute RK stability from the actual tableau: R(z)=1+z bᵀ(I-zA)⁻¹1.
const mul=([a,b],[c,d])=>[a*c-b*d,a*d+b*c],sub=([a,b],[c,d])=>[a-c,b-d],add=([a,b],[c,d])=>[a+c,b+d],div=([a,b],[c,d])=>[(a*c+b*d)/(c*c+d*d),(b*c-a*d)/(c*c+d*d)];
export function stabilityValue(method,z,catalog,stages=9){
 if(!['rk','coupled','recurrence'].includes(method.kind))throw Error('Stability map is available for ordinary RK and stabilized recurrence methods');
 if(method.kind==='recurrence'){const fam=method.identifier.replace('stabilized_',''),d=catalog.recurrences[fam][stages];let prev=[1,0],cur=add([1,0],mul([d.mtilde[1],0],z));for(let j=2;j<=stages;j++){const next=add(add([1-d.mu[j]-d.nu[j],0],mul([d.mu[j],0],cur)),add(mul([d.nu[j],0],prev),mul(z,add(mul([d.mtilde[j],0],cur),[d.gtilde[j],0]))));prev=cur;cur=next;}return Math.hypot(...cur);}
 const a=method.tableau?.a,b=method.tableau?.b;if(!a||!b)throw Error('Stability map is available for ordinary RK and stabilized recurrence methods');
 const n=b.length,lower=a.every((r,i)=>r.slice(i+1).every(x=>x===0));let k=[];
 if(lower){for(let i=0;i<n;i++){let base=[1,0];for(let j=0;j<i;j++)base=add(base,mul(z,mul([a[i][j],0],k[j])));k.push(div(base,sub([1,0],mul(z,[a[i][i],0]))));}}
 else{const M=a.map((r,i)=>[...r.map((x,j)=>sub([i===j?1:0,0],mul(z,[x,0]))),[1,0]]);for(let c=0;c<n;c++){let p=c;for(let i=c+1;i<n;i++)if(Math.hypot(...M[i][c])>Math.hypot(...M[p][c]))p=i;if(Math.hypot(...M[p][c])<1e-15)return Infinity;[M[c],M[p]]=[M[p],M[c]];for(let i=c+1;i<n;i++){const q=div(M[i][c],M[c][c]);for(let j=c+1;j<=n;j++)M[i][j]=sub(M[i][j],mul(q,M[c][j]));}}k=Array(n);for(let i=n-1;i>=0;i--){let v=M[i][n];for(let j=i+1;j<n;j++)v=sub(v,mul(M[i][j],k[j]));k[i]=div(v,M[i][i]);}}
 let result=[1,0];for(let i=0;i<n;i++)result=add(result,mul(z,mul([b[i],0],k[i])));return Math.hypot(...result);
}
export function stabilityMap(method,catalog,config={}){
 const width=config.stabilityResolution??512,height=width,extent=config.stabilityExtent??6;
 if(!Number.isInteger(width)||width<128||width>2048)throw Error('Stability resolution must be an integer from 128 to 2048');
 if(!Number.isFinite(extent)||extent<1||extent>50)throw Error('Stability extent must be from 1 to 50');
 const evaluate=compileStability(method,catalog,config.stages??9),values=new Float32Array(width*height);
 for(let j=0;j<height;j++)for(let i=0;i<width;i++)values[j*width+i]=evaluate(-extent+2*extent*i/(width-1),extent-2*extent*j/(height-1));
 return {method:method.identifier,width,height,extent,values,description:'Absolute stability for y′=λy. Shading means |R(hλ)| ≤ 1; it is not an accuracy or SSP certificate.'};
}

// Compile once per map. Pixel evaluation uses scalar complex Horner operations,
// rather than allocating stage vectors or solving a stage matrix at every pixel.
function determinantPolynomial(A){
 const n=A.length;let B=A.map((_,i)=>A.map((_,j)=>i===j?1:0));const c=[1];
 for(let k=1;k<=n;k++){const product=A.map(row=>Array.from({length:n},(_,j)=>row.reduce((sum,x,l)=>sum+x*B[l][j],0)));const ck=-product.reduce((sum,row,i)=>sum+row[i],0)/k;c.push(ck);product.forEach((row,i)=>row[i]+=ck);B=product;}return c;
}
function complexHorner(coefficients,x,y){let re=coefficients.at(-1),im=0;for(let i=coefficients.length-2;i>=0;i--){const next=re*x-im*y+coefficients[i];im=re*y+im*x;re=next;}return [re,im];}
export function compileStability(method,catalog,stages=9){
 if(!['rk','coupled','recurrence'].includes(method.kind))throw Error('Stability map is available for ordinary RK and stabilized recurrence methods');
 let numerator,denominator=[1];
 if(method.kind==='recurrence'){
  const d=catalog.recurrences[method.identifier.replace('stabilized_','')]?.[stages];if(!d)throw Error('Unsupported stabilized stage count');let prev=[1],current=[1,d.mtilde[1]];
  for(let j=2;j<=stages;j++){const next=Array(j+1).fill(0);next[0]=1-d.mu[j]-d.nu[j];current.forEach((x,i)=>{next[i]+=d.mu[j]*x;next[i+1]+=d.mtilde[j]*x;});prev.forEach((x,i)=>next[i]+=d.nu[j]*x);next[1]+=d.gtilde[j];prev=current;current=next;}numerator=current;
 }else{
  const {a,b}=method.tableau??{};if(!a||!b)throw Error('Stability map is available for ordinary RK and stabilized recurrence methods');
  const explicit=a.every((row,i)=>row.slice(i).every(x=>x===0));
  if(explicit){numerator=[1];let power=Array(b.length).fill(1);for(let i=0;i<b.length;i++){numerator.push(b.reduce((s,x,j)=>s+x*power[j],0));power=a.map(row=>row.reduce((s,x,j)=>s+x*power[j],0));}}
  else{denominator=determinantPolynomial(a);numerator=determinantPolynomial(a.map(row=>row.map((x,j)=>x-b[j])));}
 }
 return (x,y)=>{const [nr,ni]=complexHorner(numerator,x,y),[dr,di]=complexHorner(denominator,x,y);return Math.hypot(nr,ni)/Math.hypot(dr,di);};
}
