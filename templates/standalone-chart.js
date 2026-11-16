// Standalone chart adapter: no network imports, and usable from file://.
function drawOfflineChart(config) {
  const host=el('comparison').querySelector('[data-viz-container]');
  const w=Math.max(host.clientWidth,280),h=320,left=65,right=20,top=18,bottom=55;
  const positive=config.rows.filter(r=>r.value!==null);
  const xs=positive.map(r=>r.x),ys=positive.map(r=>r.value);
  const log=config.spec.yScale==='log';
  const xmin=Math.min(...xs),xmax=Math.max(...xs), ymin=log?Math.min(...ys)*.8:0,ymax=Math.max(...ys)*1.08;
  const xx=x=>left+(x-xmin)/(xmax-xmin||1)*(w-left-right);
  const yy=y=>top+(1-(log?Math.log(y/ymin)/Math.log(ymax/ymin):(y-ymin)/(ymax-ymin||1)))*(h-top-bottom);
  const title=document.createElement('h3');title.textContent=config.title;
  const svg=svgNode('svg',{viewBox:`0 0 ${w} ${h}`,width:'100%',height:h,role:'img','aria-label':config.title});
  const f=y=>config.spec.valueFormat.style==='percent'?percent(y):fmt(y);
  for(let i=0;i<=4;i++) {
    const x=xmin+(xmax-xmin)*i/4,y=log?ymin*(ymax/ymin)**(i/4):ymin+(ymax-ymin)*i/4;
    svg.append(svgNode('line',{x1:left,x2:w-right,y1:yy(y),y2:yy(y),stroke:'var(--border)','stroke-width':1}));
    svg.append(svgNode('text',{x:left-8,y:yy(y)+4,'text-anchor':'end',fill:'var(--foreground)',class:'text-small'},f(y)));
    svg.append(svgNode('text',{x:xx(x),y:h-bottom+22,'text-anchor':'middle',fill:'var(--foreground)',class:'text-small'},fmt(x)));
  }
  svg.append(svgNode('text',{x:(w+left-right)/2,y:h-5,'text-anchor':'middle',fill:'var(--foreground)',class:'text-small'},config.spec.xLabel));
  const legend=document.createElement('div');legend.className='viz-row text-small';
  const series=[...new Set(config.rows.map(r=>r.run))];
  series.forEach((name,i)=>{
    const color=`var(--viz-series-${i+1})`;
    const rows=config.rows.filter(r=>r.run===name&&r.value!==null);
    // Missing evaluation coordinates belong to other runs, so connect each run's own observations.
    svg.append(svgNode('polyline',{points:rows.map(r=>`${xx(r.x)},${yy(r.value)}`).join(' '),fill:'none',stroke:color,'stroke-width':2}));
    rows.forEach(r=>{
      const circle=svgNode('circle',{cx:xx(r.x),cy:yy(r.value),r:3,fill:color,'data-tooltip':`${name} · ${config.spec.xLabel}: ${fmt(r.x)} · ${f(r.value)}`});
      circle.append(svgNode('title',{},`${name}: x=${r.x}, value=${f(r.value)}`));svg.append(circle);
    });
    const item=document.createElement('span'),swatch=document.createElement('span');swatch.textContent='● ';swatch.style.color=color;item.append(swatch,document.createTextNode(name));legend.append(item);
  });
  const axis=document.createElement('div');axis.className='text-small text-muted';axis.textContent=config.spec.yLabel;
  host.replaceChildren(title,axis,svg,legend);
}
chart={set config(config){drawOfflineChart(config);}};
chart.config=chartConfig();
new ResizeObserver(()=>{if(root.isConnected)drawOfflineChart(chartConfig());}).observe(el('comparison'));
