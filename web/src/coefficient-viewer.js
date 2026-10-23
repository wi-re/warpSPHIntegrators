const esc=x=>String(x??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'})[c]);
const numeric=x=>typeof x==='number'?String(x):esc(x);
export function downloadCoefficients(record){
 const url=URL.createObjectURL(new Blob([JSON.stringify(record,null,2)+'\n'],{type:'application/json'})),a=document.createElement('a');
 a.href=url;a.download=`${record.method.identifier}-coefficients.json`;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
}
function display(key,value){
 if(Array.isArray(value)&&value.every(x=>typeof x==='number'||x===null))return `<div class="coefficient-block"><h3>${esc(key)}</h3><div class="coefficient-scroll"><table><thead><tr>${value.map((_,i)=>`<th scope="col">${i}</th>`).join('')}</tr></thead><tbody><tr>${value.map(x=>`<td>${numeric(x)}</td>`).join('')}</tr></tbody></table></div></div>`;
 if(Array.isArray(value)&&value.every(row=>Array.isArray(row)&&row.every(x=>typeof x==='number'||x===null))){const width=Math.max(0,...value.map(row=>row.length));return `<div class="coefficient-block"><h3>${esc(key)}</h3><div class="coefficient-scroll"><table><thead><tr><th scope="col">i / j</th>${Array.from({length:width},(_,i)=>`<th scope="col">${i}</th>`).join('')}</tr></thead><tbody>${value.map((row,i)=>`<tr><th scope="row">${i}</th>${row.map(x=>`<td>${numeric(x)}</td>`).join('')}</tr>`).join('')}</tbody></table></div></div>`;}
 if(value&&typeof value==='object')return `<details class="coefficient-block" open><summary>${esc(key)}</summary><pre>${esc(JSON.stringify(value,null,2))}</pre></details>`;
 return `<div class="coefficient-scalar"><span>${esc(key)}</span><code>${esc(value===null?'none':value)}</code></div>`;
}
export function createCoefficientViewer(){
 const dialog=document.createElement('dialog');dialog.className='coefficient-viewer';dialog.setAttribute('aria-labelledby','coefficient-title');
 dialog.innerHTML=`<div class="coefficient-header"><div><span class="eyebrow">SCHEME COEFFICIENTS</span><h2 id="coefficient-title"></h2><p id="coefficient-kind"></p></div><button class="button" id="close-coefficients" aria-label="Close coefficients">✕</button></div><div class="coefficient-content"><div id="coefficient-notes"></div><div id="coefficient-values"></div><details class="coefficient-block"><summary>Full export JSON</summary><pre id="coefficient-json"></pre></details><p class="hint" id="coefficient-source"></p></div><div class="coefficient-footer"><span id="coefficient-status" role="status">Full float64 precision · zero-based indices</span><button class="button" id="copy-coefficients">Copy JSON</button><button class="button" id="download-coefficients">Export JSON ↓</button></div>`;
 document.body.append(dialog);const find=id=>dialog.querySelector('#'+id);let record;
 find('close-coefficients').onclick=()=>dialog.close();dialog.addEventListener('click',event=>{if(event.target===dialog){const r=dialog.getBoundingClientRect();if(event.clientX<r.left||event.clientX>r.right||event.clientY<r.top||event.clientY>r.bottom)dialog.close();}});
 find('download-coefficients').onclick=()=>downloadCoefficients(record);
 find('copy-coefficients').onclick=async()=>{try{await navigator.clipboard.writeText(JSON.stringify(record,null,2)+'\n');find('coefficient-status').textContent='Coefficient JSON copied';}catch{find('coefficient-status').textContent='Clipboard unavailable. Use Export JSON to save the coefficients.';}};
 return {open(value){record=structuredClone(value);find('coefficient-title').textContent=record.method.name;find('coefficient-kind').textContent=record.representation+(Object.keys(record.parameters).length?' · '+Object.entries(record.parameters).map(([k,v])=>`${k} = ${v}`).join(' · '):'');find('coefficient-notes').innerHTML=record.notes.map(note=>`<p>${esc(note)}</p>`).join('');find('coefficient-values').innerHTML=Object.entries(record.coefficients).map(([key,value])=>display(key,value)).join('');find('coefficient-json').textContent=JSON.stringify(record,null,2);find('coefficient-source').textContent='Source: '+record.source;find('coefficient-status').textContent='Full float64 precision · zero-based indices';dialog.showModal();}};
}
