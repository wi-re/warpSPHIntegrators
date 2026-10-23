// Browse the complete coefficient catalog by numerical family.
export const methodCategories=[
 {id:'explicit',name:'Explicit Runge–Kutta',note:'Classical, embedded and low-storage stages',families:['ExplicitRK','ExpandedRK']},
 {id:'implicit',name:'Implicit Runge–Kutta',note:'DIRK, coupled, collocation and IMEX',families:['DIRK','CoupledRK','Collocation','ParallelDIRK','IMEX']},
 {id:'mechanics',name:'Mechanics & structure',note:'Symplectic splitting, Nyström and structural dynamics',families:['Symplectic','Nystrom','ImprovedNystrom','Newmark','GeneralizedAlpha']},
 {id:'multistep',name:'Multistep',note:'Adams, BDF and accepted history',families:['MultistepImplicit','MultistepExplicit','NDF']},
 {id:'stabilized',name:'Stabilized explicit',note:'Chebyshev, Legendre and orthogonal stages',families:['RelaxedChebyshev','StabilizedRK','OrthogonalRK']},
 {id:'rosenbrock',name:'Linearly implicit',note:'Rosenbrock and analytic linear solves',families:['LinearlyImplicit','Rosenbrock']},
 {id:'exponential',name:'Exponential',note:'Linear flows and exponential actions',families:['Exponential','ExponentialRK']},
 {id:'extrapolation',name:'Extrapolation',note:'Euler and midpoint refinement',families:['Extrapolation']}
];
export const categoryFor=method=>methodCategories.find(c=>c.families.includes(method.family));
const escape=x=>String(x??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'})[c]);
export function createMethodPicker({catalog,getContext,onChoose,onPreview,onClose}){
 const dialog=document.createElement('dialog');dialog.id='method-browser';dialog.className='method-browser';dialog.setAttribute('aria-labelledby','method-browser-title');
 dialog.innerHTML=`<div class="picker-header"><div><span class="eyebrow">INTEGRATOR CATALOG</span><h2 id="method-browser-title">Add a method</h2></div><button class="button" id="close-method-browser" aria-label="Close method picker">✕</button></div><div class="picker-tools"><label class="picker-search">Search methods<input id="search" type="search" placeholder="RK4, Gauss, BDF, SSP…" autocomplete="off"></label><label>Family<select id="family"><option value="all">All families</option></select></label><label>Order<select id="order"><option value="all">All orders</option></select></label><button id="reset-method-filters" class="button">Reset filters</button></div><div class="picker-body"><nav id="method-categories" aria-label="Method categories"></nav><div class="picker-results"><div id="picker-result-count" role="status"></div><div id="method-cards"></div></div></div><div class="picker-footer"><span id="picker-instruction"></span><span id="picker-preview" aria-live="polite">Hover or focus a card to inspect its method.</span></div>`;
 document.body.append(dialog);
 const $=id=>dialog.querySelector('#'+id),methods=catalog.schemes;
 for(const family of [...new Set(methods.map(m=>m.family))])$('family').add(new Option(family,family));
 for(const order of [...new Set(methods.map(m=>m.order))].sort((a,b)=>a-b))$('order').add(new Option('Order '+order,order));
 let category='all',replaceId=null,trigger=null,chosenId=null;
 const byId=id=>methods.find(m=>m.identifier===id);
 function reason(m,context){return context.unavailable(m)||(context.methods.includes(m.identifier)&&m.identifier!==replaceId?'Already in comparison':!replaceId&&context.methods.length>=6?'Comparison is full':'');}
 function render(){
  const context=getContext(),query=$('search').value.trim().toLowerCase(),family=$('family').value,order=$('order').value;
  const matches=m=>(family==='all'||m.family===family)&&(order==='all'||m.order===Number(order))&&(!query||[m.name,m.friendly_name,m.identifier,m.family,categoryFor(m)?.name].join(' ').toLowerCase().includes(query));
  const groups=[{id:'all',name:'All methods',note:'Every source configuration',families:[...new Set(methods.map(m=>m.family))]},...methodCategories];
  $('method-categories').innerHTML=groups.map(c=>`<button data-category="${c.id}" aria-pressed="${category===c.id}" title="${escape(c.note)}"><span>${escape(c.name)}</span><small>${methods.filter(m=>c.families.includes(m.family)&&matches(m)).length}</small></button>`).join('');
  const visible=methods.filter(m=>matches(m)&&(category==='all'||categoryFor(m)?.id===category));
  $('picker-result-count').textContent=`${visible.length} configurations · ${visible.filter(m=>!reason(m,context)).length} selectable · ${context.problemName}`;
  $('method-cards').innerHTML=methodCategories.filter(c=>category==='all'||category===c.id).map(c=>{const inCategory=visible.filter(m=>categoryFor(m)?.id===c.id);if(!inCategory.length)return '';return `<section class="picker-category"><h3>${escape(c.name)}<small>${escape(c.note)}</small></h3>${c.families.map(f=>{const items=inCategory.filter(m=>m.family===f);if(!items.length)return '';return `<h4>${escape(f)} <span>${items.length}</span></h4><div class="method-card-grid">${items.map(m=>{const why=reason(m,context),current=m.identifier===replaceId,stages=m.tableau?.c?.length??m.stage_count;return `<button class="method-card ${current?'current':''}" data-method="${escape(m.identifier)}" aria-disabled="${!!why}" ${current?'aria-current="true"':''}><span class="method-card-top"><b>${escape(m.name)}</b><span class="order-badge">p${m.order}</span></span><span class="method-card-description">${escape(m.friendly_name??m.name)}</span><span class="method-card-tags"><span>${m.implicit?'Implicit':'Explicit'}</span>${stages?`<span>${stages} ${stages===1?'stage':'stages'}</span>`:''}${m.estimate_order?'<span>Embedded</span>':''}${m.stability?`<span>${escape(m.stability)} stable</span>`:''}</span><span class="method-card-action">${escape(why||(current?'Current method':replaceId?'Replace method ↗':'Add to comparison +'))}</span></button>`;}).join('')}</div>`;}).join('')}</section>`;}).join('')||'<div class="picker-empty">No methods match these filters. Try another category, family or order, or clear the search.</div>';
 }
 function preview(id){const m=byId(id);if(!m)return;onPreview(id);$('picker-preview').textContent=`${m.name} · ${m.family} · order ${m.order}${m.tableau?.c?.length?' · '+m.tableau.c.length+' stages':''}`;}
 $('method-categories').onclick=event=>{const button=event.target.closest('[data-category]');if(!button)return;category=button.dataset.category;render();$('method-categories').querySelector(`[data-category="${category}"]`).focus();dialog.querySelector('.picker-results').scrollTop=0;};
 for(const id of ['search','family','order'])$(id).addEventListener('input',()=>{render();dialog.querySelector('.picker-results').scrollTop=0;});
 $('method-cards').onclick=event=>{const button=event.target.closest('[data-method]');if(!button)return;const method=byId(button.dataset.method);if(reason(method,getContext()))return;chosenId=method.identifier;onChoose(method.identifier,replaceId);dialog.close();};
 for(const eventName of ['pointerover','focusin'])$('method-cards').addEventListener(eventName,event=>{const button=event.target.closest('[data-method]');if(button)preview(button.dataset.method);});
 $('reset-method-filters').onclick=()=>{category='all';$('search').value='';$('family').value='all';$('order').value='all';render();dialog.querySelector('.picker-results').scrollTop=0;$('search').focus();};
 $('close-method-browser').onclick=()=>dialog.close();
 dialog.addEventListener('click',event=>{if(event.target===dialog){const r=dialog.getBoundingClientRect();if(event.clientX<r.left||event.clientX>r.right||event.clientY<r.top||event.clientY>r.bottom)dialog.close();}});
 dialog.addEventListener('close',()=>{onClose();if(trigger?.isConnected)trigger.focus();else (document.querySelector(`[data-replace="${chosenId}"]`)??document.getElementById('browse-methods')).focus();});
 return {open(id=null,element=null){replaceId=id;trigger=element;chosenId=null;$('picker-preview').textContent='Hover or focus a card to inspect its method.';category=id?categoryFor(byId(id))?.id??'all':'all';$('search').value='';$('family').value='all';$('order').value='all';$('method-browser-title').textContent=id?'Replace '+byId(id).name:'Add a method';$('picker-instruction').textContent=id?'Pick a card to replace this comparison slot.':'Pick a card to add it to your comparison. Up to six methods.';render();dialog.showModal();dialog.querySelector('.picker-results').scrollTop=0;$('search').focus();if(id)preview(id);}};
}
