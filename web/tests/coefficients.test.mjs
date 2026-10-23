import test from 'node:test';import assert from 'node:assert/strict';import fs from 'node:fs';
import {coefficientRecord} from '../src/coefficients.js';
import {methodCFL,timestepScale,timestepIncrement,nudgeTimestep} from '../src/cfl.js';
import {makeProblem} from '../src/problems.js';
const catalog=JSON.parse(fs.readFileSync(new URL('../src/catalog.json',import.meta.url))),method=id=>catalog.schemes.find(s=>s.identifier===id);
test('all browser schemes have self-contained coefficient exports with finite, unrounded numeric data',()=>{
 const before=JSON.stringify(catalog);
 for(const m of catalog.schemes){const r=coefficientRecord(m,catalog);assert.ok(r.representation,m.identifier);assert.ok(Object.keys(r.coefficients).length,m.identifier);assert.equal(r.method.identifier,m.identifier);assert.doesNotThrow(()=>JSON.stringify(r,(_,v)=>{if(typeof v==='number'&&!Number.isFinite(v))throw Error('Nonfinite coefficient');return v;}));assert.ok(r.notes.length);if(m.tableau)assert.deepEqual(r.coefficients,m.tableau);}
 assert.equal(JSON.stringify(catalog),before);
});
test('exports follow the active structural and recurrence parameters and retain their snapshot',()=>{
 const r=coefficientRecord(method('generalizedAlpha'),catalog,{rhoInfinity:1});assert.equal(r.coefficients.alpha_m,.5);assert.equal(r.coefficients.alpha_f,.5);assert.equal(r.coefficients.beta,.25);assert.equal(r.coefficients.gamma,.5);
 const n=coefficientRecord(method('newmark'),catalog,{beta:1/6,gamma:.6});assert.equal(n.coefficients.beta,1/6);assert.equal(n.coefficients.gamma,.6);
 for(const stages of [3,9,65]){const c=coefficientRecord(method('rkc2'),catalog,{stages});assert.deepEqual(c.coefficients,catalog.recurrences.rkc2[stages]);assert.equal(c.parameters.stages,stages);c.coefficients.c[0]=99;assert.notEqual(catalog.recurrences.rkc2[stages].c[0],99);}
});
test('history and splitting weights use their actual driver convention',()=>{
 assert.deepEqual(coefficientRecord(method('ab3'),catalog).coefficients.predictor,[23/12,-16/12,5/12]);assert.deepEqual(coefficientRecord(method('bdf2'),catalog).coefficients,{state_weights:[4/3,-1/3],beta:2/3});
 for(const id of ['pefrl','vefrl']){const c=coefficientRecord(method(id),catalog).coefficients;assert.ok(Math.abs(c.drift.reduce((a,b)=>a+b)-1)<1e-14);assert.ok(Math.abs(c.kick.reduce((a,b)=>a+b)-1)<1e-14);}
 const c=coefficientRecord(method('irkn4'),catalog).coefficients;assert.deepEqual(c.warm_velocity_weights,[19/18,-1/18,-1/6,1/6,11/18,-11/18]);assert.equal(c.startup.b[1],1/3);
});
test('CFL values distinguish SSP guarantees, model measurements and unknown coefficients',()=>{
 assert.equal(methodCFL(method('ssprk104')).ssp,6);assert.equal(methodCFL(method('ssprkmsvs32')).ssp,.5);assert.match(methodCFL(method('ssprkmsvs32')).note,/warm/);
 assert.equal(methodCFL(method('ssprk54')).ssp,null);assert.equal(methodCFL(method('ssprk54')).upwind,1.7693);assert.equal(methodCFL(method('kykssprk42')).upwind,2.2838);assert.equal(methodCFL(method('rungeKutta4')).ssp,null);
});
test('problem CFL conversions scale with grid, speed and diffusivity',()=>{
 const a=timestepScale(makeProblem('advectionStep',{n:32,speed:2}));assert.equal(a.rate*.0125,.8);assert.equal(a.feLimit,1/64);
 const b=timestepScale(makeProblem('burgers',{n:32}));assert.equal(b.rate,32);
 const d=timestepScale(makeProblem('diffusion',{n:31,D:.1}));assert.equal(d.rate,.1*1024);assert.equal(d.feLimit*d.rate,.5);
 assert.equal(timestepScale(makeProblem('oscillator')).rate,null);assert.equal(timestepScale(makeProblem('acoustic')).feLimit,null);assert.equal(timestepScale(makeProblem('viscousBurgers')).feLimit,null);
});
test('timestep arrows make small relative changes even at microsecond scales',()=>{
 for(const h of [.04,.0125,.0000125,1e-8,1,99]){const delta=timestepIncrement(h);assert.ok(delta/h<=.101);assert.ok(delta/h>=.0099);const up=nudgeTimestep(h,1);assert.ok(up>h&&up<=100);if(h>1e-8)assert.ok(nudgeTimestep(h,-1)<h);}
 assert.equal(nudgeTimestep(.0000125,1),.0000135);assert.equal(nudgeTimestep(.0000125,-1),.0000115);assert.equal(nudgeTimestep(1e-8,-1),1e-8);assert.equal(nudgeTimestep(100,1),100);
});
