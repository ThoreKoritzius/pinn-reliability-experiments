// Execute the exported control/rendering code against a small DOM harness.
// This is a logic smoke test, not a real-browser layout test.
const fs=require('node:fs'),vm=require('node:vm'),assert=require('node:assert/strict');
const html=fs.readFileSync('results/explorer-data.html','utf8');
const data=html.match(/id="tg-data">([\s\S]*?)<\/script>/)[1];
const source=html.match(/<script type="module">([\s\S]*?)<\/script>/)[1];
class Element {
  constructor(tag='div'){this.tag=tag;this.children=[];this.attrs={};this.listeners={};this.style={};this.clientWidth=900;this.isConnected=true;this.parentElement={clientWidth:280};this.classList={toggle(){}};}
  remove(){}
  getContext(){return {fillRect(){},getImageData(){return {data:[240,240,240,255]};},createImageData(w,h){return {data:new Uint8ClampedArray(w*h*4)};},putImageData(){}};}
  toDataURL(){return 'data:image/png;base64,';}
  append(...nodes){this.children.push(...nodes);}
  replaceChildren(...nodes){this.children=nodes;}
  setAttribute(key,value){this.attrs[key]=value;}
  addEventListener(name,fn){this.listeners[name]=fn;}
  querySelector(selector){if(selector==='[data-viz-container]')return chartHost;return nodes.get(selector.slice(1));}
}
const nodes=new Map(),chartHost=new Element();
for(const [,id] of html.matchAll(/\bid="([^"]+)"/g))nodes.set(id,new Element());
nodes.get('tg-data').textContent=data;
const document={getElementById:id=>nodes.get(id),createElement:tag=>new Element(tag),createElementNS:(ns,tag)=>new Element(tag),createTextNode:text=>text};
let interval=null;
const context={document,window:{addEventListener(){}},console,atob:s=>Buffer.from(s,'base64').toString('binary'),
  getComputedStyle:()=>({color:'rgb(240,240,240)'}),
  ResizeObserver:class{observe(){}disconnect(){}},setInterval:fn=>{interval=fn;return 1;},clearInterval:()=>{interval=null;}};
vm.createContext(context);vm.runInContext(source,context);
assert(nodes.get('tg-exact').children.some(e=>e.tag==='image'));
assert(chartHost.children.length===4,'Chart did not mount');
const initialReadout=JSON.stringify(nodes.get('tg-readout').children);
const slider=nodes.get('tg-slider');slider.value='0';slider.listeners.input();
assert.equal(nodes.get('tg-t').textContent,'0.000');
assert.notEqual(JSON.stringify(nodes.get('tg-readout').children),initialReadout);
for(const [id,value] of [['run','pinn_seed1'],['stage','after_adam'],['field','p']]){
  const node=nodes.get('tg-'+id);node.value=value;node.listeners.change();
}
assert.equal(nodes.get('tg-error-title').textContent,'Signed error');
for(const value of ['velocity','pressure','energy','training','residual']){
  const node=nodes.get('tg-chart-mode');node.value=value;node.listeners.change();
  assert(chartHost.children[2].children.length>10,'Chart marks missing after selection');
}
nodes.get('tg-play').listeners.click();assert(interval);
for(let i=0;i<JSON.parse(data).runs[0].times.length-1;i++)interval();
assert.equal(nodes.get('tg-t').textContent,'2.000');assert.equal(interval,null);
assert.equal(nodes.get('tg-play').textContent,'Play');
assert(!source.includes('import('),'Standalone rendering still depends on a module import');
console.log('PASS: standalone maps, time slider, run/stage/field changes, five chart modes and bounded playback.');
