// Reuse only a source-declared explicit FSAL tableau whose final stage is
// exactly the accepted endpoint. Other families retain their own cache policy.
export function fsalEligible(method){
 const tab=method.tableau;
 return !!method.fsal&&method.kind==='rk'&&!method.implicit&&tab.c[0]===0&&tab.c.at(-1)===1&&tab.a.every((row,i)=>row.slice(i).every(x=>x===0))&&tab.a.at(-1).every((x,j)=>x===tab.b[j]);
}
