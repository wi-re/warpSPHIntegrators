import test from 'node:test';import assert from 'node:assert/strict';import fs from 'node:fs';
import {baseConfig,presets,problemSetups} from '../src/experiments.js';
import {makeProblem,problemCatalog} from '../src/problems.js';
import {solveCPU} from '../src/engine.js';
import {compileStability,stabilityValue,stabilityMap} from '../src/stability.js';
import {revealSeries,pathAppearance} from '../src/presentation.js';
const catalog=JSON.parse(fs.readFileSync(new URL('../src/catalog.json',import.meta.url))),method=id=>catalog.schemes.find(s=>s.identifier===id);
test('every default demo and problem constellation runs successfully',()=>{
 for(const [name,setup] of [...Object.entries(presets),...Object.entries(problemSetups)]){
  const id=setup.problem??name,p=makeProblem(id,setup.params),config={...baseConfig,...setup.config,linearSolver:'dense'};
  for(const id of setup.methods){const r=solveCPU(method(id),p,catalog,config);assert.equal(r.failure,null,`${name} / ${id}`);assert.equal(r.t,config.T);assert.ok(r.y.every(Number.isFinite));}
 }
 assert.deepEqual(Object.keys(problemSetups).sort(),problemCatalog.map(p=>p.id).sort());
});
test('compiled stability agrees with direct stage solves and recurrences',()=>{
 for(const id of ['forwardEuler','rungeKutta4','dormandPrince','backwardEuler','trbdf2','radauIia3','gaussLegendre2','rkc2','stabilized_rkg2']){
  const m=method(id),evaluate=compileStability(m,catalog,9);
  for(const z of [[-.3,.2],[-1,1],[-2.1,.6],[.01,-.2]]){const exact=stabilityValue(m,z,catalog,9);assert.ok(Math.abs(evaluate(...z)-exact)<2e-8*Math.max(1,exact),`${id}: ${z}`);}
 }
});
test('stability regions use a high-resolution adjustable grid',()=>{
 const m=method('rungeKutta4'),a=stabilityMap(m,catalog),b=stabilityMap(m,catalog,{stabilityResolution:768});
 assert.equal(a.width,512);assert.equal(a.values.length,512*512);assert.equal(b.values.length,768*768);
 assert.throws(()=>stabilityMap(m,catalog,{stabilityResolution:4096}),/resolution/);
});
test('playback reveals both time-series curves and their current endpoints',()=>{
 const data=[[0,0],[1,2],[2,4]];assert.deepEqual(revealSeries(data,0),[[0,0]]);assert.deepEqual(revealSeries(data,.5),[[0,0],[.5,1]]);assert.deepEqual(revealSeries(data,1),[[0,0],[1,2]]);assert.deepEqual(revealSeries(data,2),data);assert.deepEqual(data,[[0,0],[1,2],[2,4]]);
});
test('linked highlighting emphasizes only the matching scheme',()=>{
 assert.equal(pathAppearance('rk4','rk4').selected,true);assert.equal(pathAppearance('rk4','rk4').alpha,1);assert.ok(pathAppearance('verlet','rk4').alpha<.3);assert.equal(pathAppearance('verlet',null).alpha,1);
});
