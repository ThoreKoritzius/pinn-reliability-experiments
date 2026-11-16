// Validate the actual config builder and field calculations used by the explorer.
const fs = require('node:fs');
const vm = require('node:vm');
const assert = require('node:assert/strict');
const path = process.argv[2];
if (!path) throw Error('Pass the generated explorer HTML path');
const html = fs.readFileSync(path,'utf8');
const data = JSON.parse(html.match(/id="tg-data">([\s\S]*?)<\/script>/)[1]);
for(const r of data.runs) for(const s of Object.values(r.stages)) s.decoded=s.fields.map(f=>{
  const b=Buffer.from(f.data,'base64');return Array.from({length:b.length/2},(_,i)=>b.readInt16LE(i*2)*f.scale);
});
const source=html.match(/<script type="module">([\s\S]*?)<\/script>/)[1];
const functions=source.slice(source.indexOf('function chartConfig()'),source.indexOf('function svgNode('));
const context={data,n:data.n,cells:data.n**2,lastFrame:data.runs[0].times.length-1,state:{stage:'after_lbfgs',field:'speed',chart:'velocity'},
  stageLabel:{after_adam:'Adam only',after_lbfgs:'Adam + L-BFGS'}};
vm.createContext(context);vm.runInContext(functions,context);
fs.mkdirSync('results/explorer-checks',{recursive:true});
let states=0,snapshots=0;
for(const stage of ['after_adam','after_lbfgs']) {
  context.state.stage=stage;
  for(const mode of ['velocity','pressure','energy','training','residual']) {
    context.state.chart=mode;
    const config=context.chartConfig();
    assert(config.rows.length>0&&config.rows.every(r=>Number.isFinite(r.x)&&(r.value===null||Number.isFinite(r.value))));
    assert(config.rows.length<=2000);
    fs.writeFileSync(`results/explorer-checks/${stage}-${mode}.json`,JSON.stringify(config));states++;
  }
  for(const run of data.runs) for(const field of ['omega','speed','u','v','p']) {
    context.state.field=field;
    const bound=context.bounds(run);
    assert(bound.flow>0&&bound.error>0);
    for(let t=0;t<run.times.length;t++) {
      const s=context.fieldSlice(run,stage,t,field);
      for(const key of ['truth','prediction','error']) {
        assert.equal(s[key].length,data.n**2);assert(s[key].every(Number.isFinite));
      }
      if(field==='speed') {
        const relative=Math.sqrt(s.error.reduce((a,b)=>a+b*b,0)/s.truth.reduce((a,b)=>a+b*b,0));
        assert(Math.abs(relative-run.stages[stage].stats.velocity[t])<0.00005,'Quantized fields changed error materially');
      } else {
        for(let k=0;k<s.error.length;k++) assert(Math.abs(s.error[k]-(s.prediction[k]-s.truth[k]))<1e-12);
      }
      assert(Math.max(...s.error.map(Math.abs))<=bound.error+1e-12);snapshots++;
    }
  }
}
const ids=[...html.matchAll(/\bid="([^"]+)"/g)].map(m=>m[1]);
assert.equal(ids.length,new Set(ids).size,'Duplicate DOM IDs');
for(const id of ['run','stage','field','slider','play','common-scale','chart-mode','readout','exact','pred','error','comparison']) assert(ids.includes('tg-'+id));
// Exercise the actual state restoration, including malformed snapshots.
vm.runInContext(source.slice(source.indexOf('function applySaved('),source.indexOf('function save(')),context);
const before={...context.state};context.applySaved({modelContent:{version:1,run:'absent',time:-1,stage:'invalid',field:'invalid',chart:'invalid'}});
assert.equal(context.state.time,before.time);assert.equal(context.state.stage,before.stage);
context.applySaved({modelContent:{version:1,run:data.runs[1].id,time:0,stage:'after_adam',field:'p',common:true,chart:'energy'}});
assert.equal(context.state.time,0);assert.equal(context.state.field,'p');assert.equal(context.state.run,data.runs[1].id);
assert.equal(context.state.common,true);
console.log(`PASS: ${states} complete chart configs, ${snapshots} field snapshots, quantization, color bounds, DOM IDs and state restoration.`);
