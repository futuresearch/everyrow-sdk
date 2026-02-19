"""MCP App UI HTML templates for progress, results, and session widgets."""

_APP_SCRIPT_SRC = "https://unpkg.com/@modelcontextprotocol/ext-apps@1.0.1/app-with-deps"
UI_CSP_META = {"ui": {"csp": {"resourceDomains": ["https://unpkg.com"]}}}

PROGRESS_HTML = """<!DOCTYPE html>
<html><head><meta name="color-scheme" content="light dark">
<style>
body{font-family:system-ui;margin:16px;color:#333}
@media(prefers-color-scheme:dark){body{color:#ddd}}
.bar-bg{width:100%;background:#e0e0e0;border-radius:6px;overflow:hidden;height:28px;margin:12px 0}
.bar{height:100%;background:#4caf50;transition:width .3s;display:flex;align-items:center;padding:0 8px;color:#fff;font-size:13px;white-space:nowrap}
.stats{font-size:14px;margin:8px 0}
.status{font-weight:600;font-size:16px;margin-bottom:8px}
</style></head><body>
<div id="c">Waiting for data...</div>
<script type="module">
import{App}from"SCRIPT_SRC";
const app=new App({name:"EveryRow Progress",version:"1.0.0"});
const el=document.getElementById("c");
app.ontoolresult=({content})=>{
  const t=content?.find(c=>c.type==="text");
  if(!t)return;
  try{const d=JSON.parse(t.text);render(d)}catch{el.textContent=t.text}
};
function render(d){
  const pct=d.total>0?Math.round(d.completed/d.total*100):0;
  el.innerHTML=`
    <div class="status">${d.status}</div>
    <div class="bar-bg"><div class="bar" style="width:${pct}%">${pct}%</div></div>
    <div class="stats">
      ${d.completed}/${d.total} complete${d.failed?`, ${d.failed} failed`:""}${d.running?`, ${d.running} running`:""}
      &mdash; ${d.elapsed_s}s elapsed
    </div>
    ${d.session_url?`<a href="${d.session_url}" target="_blank">Open session</a>`:""}`;
}
await app.connect();
</script></body></html>""".replace("SCRIPT_SRC", _APP_SCRIPT_SRC)

RESULTS_HTML = """<!DOCTYPE html>
<html><head><meta name="color-scheme" content="light dark">
<style>
body{font-family:system-ui;margin:12px;color:#333;font-size:13px}
@media(prefers-color-scheme:dark){body{color:#ddd;background:#1a1a1a}
  th{background:#2d2d2d!important;border-color:#444!important}
  td{border-color:#444!important}
  tr:nth-child(even) td{background:#222!important}}
#sum{font-weight:600;font-size:14px;margin-bottom:8px}
.wrap{max-height:400px;overflow:auto;border:1px solid #ddd;border-radius:6px}
table{border-collapse:collapse;width:100%;font-size:13px}
th{background:#f8f8f8;position:sticky;top:0;padding:8px 10px;text-align:left;border-bottom:2px solid #ddd;font-size:12px;white-space:nowrap}
td{padding:6px 10px;border-bottom:1px solid #eee;max-width:400px;vertical-align:top;word-wrap:break-word;white-space:pre-wrap}
tr:nth-child(even) td{background:#fafafa}
a{color:#1976d2;text-decoration:none;word-break:break-all}
a:hover{text-decoration:underline}
td:first-child{position:sticky;left:0;background:inherit;z-index:1;font-weight:500}
th:first-child{position:sticky;left:0;z-index:2}
</style></head><body>
<div id="sum"></div>
<div class="wrap"><table id="tbl"></table></div>
<script type="module">
import{App}from"SCRIPT_SRC";
const app=new App({name:"EveryRow Results",version:"1.0.0"});
const el=document.getElementById("tbl");
const sum=document.getElementById("sum");

app.ontoolresult=({content})=>{
  const t=content?.find(c=>c.type==="text");
  if(!t)return;
  let meta;
  try{meta=JSON.parse(t.text);}catch(e){sum.textContent=t.text;return;}
  if(meta.results_url){
    if(meta.preview)render(meta.preview);
    sum.textContent="Loading full results...";
    const opts=meta.download_token?{headers:{"Authorization":"Bearer "+meta.download_token}}:{};
    fetch(meta.results_url,opts).then(r=>r.json()).then(data=>render(data)).catch(()=>{
      if(!meta.preview)sum.textContent="Failed to load results";
    });
  }else if(Array.isArray(meta)){
    render(meta);
  }else{
    sum.textContent=JSON.stringify(meta);
  }
};

function flat(obj,pre=""){
  const o={};
  for(const[k,v]of Object.entries(obj)){
    const key=pre?pre+"."+k:k;
    if(v&&typeof v==="object"&&!Array.isArray(v))Object.assign(o,flat(v,key));
    else o[key]=v;
  }
  return o;
}

function render(data){
  if(!Array.isArray(data))data=[data];
  if(!data.length){sum.textContent="No results";return;}
  data=data.map(r=>flat(r));
  const allCols=Object.keys(data[0]);
  const cols=[...allCols.filter(k=>!k.includes(".")),...allCols.filter(k=>k.includes("."))];
  sum.textContent=data.length+" rows, "+cols.length+" columns";
  let h="<thead><tr>"+cols.map(k=>"<th>"+esc(k)+"</th>").join("")+"</tr></thead><tbody>";
  for(let i=0;i<data.length;i++){
    h+="<tr>";
    for(let j=0;j<cols.length;j++){h+=td(data[i][cols[j]]);}
    h+="</tr>";
  }
  el.innerHTML=h+"</tbody>";
}

function td(v){
  if(v==null)return"<td></td>";
  const s=String(v);
  if(s.match(/^https?:\\/\\//))return'<td><a href="'+esc(s)+'" target="_blank">'+esc(s)+"</a></td>";
  return"<td>"+esc(s)+"</td>";
}
function esc(s){const d=document.createElement("div");d.textContent=String(s);return d.innerHTML;}

await app.connect();
</script></body></html>""".replace("SCRIPT_SRC", _APP_SCRIPT_SRC)

SESSION_HTML = """<!DOCTYPE html>
<html><head><meta name="color-scheme" content="light dark">
<style>
*{box-sizing:border-box}
body{font-family:system-ui,-apple-system,sans-serif;margin:0;padding:12px;color:#333;font-size:13px}
@media(prefers-color-scheme:dark){body{color:#ddd;background:#1a1a1a}
  .bar-bg{background:#333}.info{color:#aaa}.seg-legend{color:#aaa}}
a{color:#1976d2;text-decoration:none;font-weight:500}
a:hover{text-decoration:underline}
.bar-bg{width:100%;background:#e8e8e8;border-radius:6px;overflow:hidden;height:22px;margin:8px 0;
  display:flex}
.seg{height:100%;transition:width .5s;display:flex;align-items:center;justify-content:center;
  font-size:11px;color:#fff;overflow:hidden;white-space:nowrap}
.seg-done{background:#4caf50}
.seg-run{background:#2196f3}
.seg-fail{background:#e53935}
.seg-pend{background:transparent}
.info{font-size:12px;color:#666;margin:4px 0;display:flex;align-items:center;gap:12px;flex-wrap:wrap}
.seg-legend{display:flex;gap:10px;font-size:11px;color:#888}
.seg-legend span::before{content:"";display:inline-block;width:8px;height:8px;border-radius:2px;margin-right:3px}
.l-done::before{background:#4caf50!important}.l-run::before{background:#2196f3!important}
.l-fail::before{background:#e53935!important}
.status-done{color:#4caf50;font-weight:600}
.status-fail{color:#e53935;font-weight:600}
.eta{color:#888;font-size:11px}
@keyframes flash{0%,100%{background:transparent}50%{background:rgba(76,175,80,.15)}}
.flash{animation:flash 1s ease 3}
</style></head><body>
<div id="c">Waiting...</div>
<script type="module">
import{App}from"SCRIPT_SRC";
const app=new App({name:"EveryRow Session",version:"1.0.0"});
const el=document.getElementById("c");
let pollUrl=null,pollTimer=null,sessionUrl="",wasDone=false;

app.ontoolresult=({content})=>{
  const t=content?.find(c=>c.type==="text");if(!t)return;
  try{
    const d=JSON.parse(t.text);sessionUrl=d.session_url||"";render(d);
    if(d.progress_url&&!pollTimer){pollUrl=d.progress_url;startPoll()}
  }catch{el.textContent=t.text}
};

function render(d){
  const comp=d.completed||0,tot=d.total||0,fail=d.failed||0,run=d.running||0;
  const pend=Math.max(0,tot-comp-fail-run);
  const done=["completed","failed","revoked"].includes(d.status);
  const url=d.session_url||sessionUrl;
  const elapsed=d.elapsed_s||0;

  let h=url?`<a href="${url}" target="_blank">Open everyrow session &#x2197;</a>`:"";

  if(tot>0){
    const pDone=comp/tot*100,pRun=run/tot*100,pFail=fail/tot*100;
    h+=`<div class="bar-bg">`;
    if(pDone>0)h+=`<div class="seg seg-done" style="width:${pDone}%">${pDone>=10?Math.round(pDone)+"%":""}</div>`;
    if(pRun>0)h+=`<div class="seg seg-run" style="width:${pRun}%"></div>`;
    if(pFail>0)h+=`<div class="seg seg-fail" style="width:${pFail}%"></div>`;
    h+=`</div>`;

    h+=`<div class="info">`;
    if(done){
      const cls=d.status==="completed"?"status-done":"status-fail";
      h+=`<span class="${cls}">${d.status}</span>`;
      h+=`<span>${comp}/${tot}${fail?` (${fail} failed)`:""}</span>`;
      if(elapsed)h+=`<span>${fmtTime(elapsed)}</span>`;
    }else{
      h+=`<span>${comp}/${tot}</span>`;
      const eta=comp>0&&elapsed>0?Math.round((tot-comp)/(comp/elapsed)):0;
      if(eta>0)h+=`<span class="eta">~${fmtTime(eta)} remaining</span>`;
      if(elapsed)h+=`<span class="eta">${fmtTime(elapsed)} elapsed</span>`;
    }
    h+=`</div>`;

    if(!done){
      h+=`<div class="seg-legend">`;
      if(comp)h+=`<span class="l-done">${comp} done</span>`;
      if(run)h+=`<span class="l-run">${run} running</span>`;
      if(fail)h+=`<span class="l-fail">${fail} failed</span>`;
      if(pend)h+=`<span>${pend} pending</span>`;
      h+=`</div>`;
    }
  }else if(d.status){
    h+=`<div class="info">${d.status}${elapsed?` &mdash; ${fmtTime(elapsed)}`:""}</div>`;
  }

  el.innerHTML=h;

  if(done&&!wasDone){wasDone=true;el.classList.add("flash")}
  if(done&&pollTimer){clearInterval(pollTimer);pollTimer=null}
}

function fmtTime(s){
  if(s<60)return s+"s";
  const m=Math.floor(s/60),sec=s%60;
  return m+"m"+((sec>0)?(" "+sec+"s"):"");
}

function startPoll(){
  pollTimer=setInterval(async()=>{
    try{const r=await fetch(pollUrl);if(r.ok)render(await r.json())}catch{}
  },5000);
}

await app.connect();
</script></body></html>""".replace("SCRIPT_SRC", _APP_SCRIPT_SRC)
