import {makeProblem,compatible,norm} from './problems.js';
import {solveCPU,convergenceCPU,geometryCPU,sensitivityCPU} from './engine.js';
import {GPUEngine,gpuSupport} from './gpu.js';
import {stabilityMap} from './stability.js';
const catalogPromise=fetch(new URL('./catalog.json',import.meta.url)).then(r=>{if(!r.ok)throw Error('Unable to load coefficient catalog');return r.json();});
let gpuPromise;
self.onmessage=async({data:task})=>{
 try{
 const catalog=await catalogPromise,problem=makeProblem(task.problem,task.params),results=[];
 for(let index=0;index<task.methods.length;index++){
  const method=catalog.schemes.find(s=>s.identifier===task.methods[index]);if(!method)throw Error('Unknown method');
  self.postMessage({type:'progress',id:task.id,index,total:task.methods.length,name:method.friendly_name??method.name});
  let result;try{
   if(task.mode==='ssp'&&!['advection','advectionStep','burgers'].includes(problem.id))throw Error('Use periodic upwind advection or inviscid Burgers for the SSP diagnostic (or select the SSP preset).');
   const issue=compatible(method,problem);if(issue)throw Error(issue);
   const config=task.config;let fallback=gpuSupport(method,problem,config),backend=config.backend??'auto';
   if(task.mode==='stability')result=stabilityMap(method,catalog,config);
   else if(task.mode==='sensitivity')result=sensitivityCPU(method,problem,catalog,config);
   else{
    if(task.mode==='convergence'&&backend==='auto')fallback='Auto uses float64 for convergence to resolve high orders';
    if(task.mode==='convergence'&&!problem.exact)fallback='A checked numerical convergence reference uses the CPU engine';
    let engine=null;
    if(backend!=='cpu'&&!fallback){try{gpuPromise??=GPUEngine.create();engine=await gpuPromise;}catch(e){fallback=e.message;}}
    if(backend==='gpu'&&fallback)throw Error(fallback);
    if(task.mode==='convergence'){
     if(engine){const configs=Array.from({length:config.levels},(_,i)=>({...config,dt:config.dt/2**i})),runs=await engine.solve(method,problem,catalog,configs);const rows=runs.map((r,i)=>{const prev=runs[i-1],floor=3e-6;return {dt:r.actualDt,error:r.failure?null:r.error,order:prev?.error>floor&&r.error>floor?Math.log(prev.error/r.error)/Math.log(prev.actualDt/r.actualDt):null,work:r.stats.rhs+r.stats.linear,time:r.elapsed,failure:r.failure};});result={method:method.identifier,rows,reference:'Analytic semi-discrete solution · WebGPU float32',referenceUncertainty:0,numericFloor:3e-6,backend:'WebGPU · float32'};}
     else result={...convergenceCPU(method,problem,catalog,config),backend:'CPU · float64',fallback:backend==='cpu'?null:fallback};
    }else{
     if(engine){
      try{result=(await engine.solve(method,problem,catalog,[config]))[0];if(result.failure&&backend==='auto')result={...solveCPU(method,problem,catalog,config),fallback:`WebGPU: ${result.failure}; retried in float64`};}
      catch(error){if(backend==='gpu')throw error;result={...solveCPU(method,problem,catalog,config),fallback:`WebGPU: ${error.message}; retried in float64`};}
     }else result={...solveCPU(method,problem,catalog,config),fallback:backend==='cpu'?null:fallback};
     if(task.mode==='geometry'){try{result.geometry=geometryCPU(method,problem,catalog,config);}catch(e){result.geometry={failure:e.message};}}
    }
   }
  }catch(e){result={method:method.identifier,failure:e.message};}
  results.push(result);self.postMessage({type:'partial',id:task.id,result,index});
 }
 self.postMessage({type:'complete',id:task.id,results});
 }catch(e){self.postMessage({type:'error',id:task.id,message:e.message});}
};
