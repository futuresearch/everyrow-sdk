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
:root{
  --bg:#fff;--bg-alt:#f8f9fa;--bg-hover:rgba(25,118,210,0.06);
  --bg-selected:rgba(25,118,210,0.10);--bg-toolbar:#fafafa;
  --text:#333;--text-sec:#666;--text-dim:#999;
  --border:#e0e0e0;--border-light:#eee;
  --accent:#1976d2;
  --research-dot:#42a5f5;
  --pop-bg:#fff;--pop-shadow:0 4px 20px rgba(0,0,0,0.15);
  --toast-bg:#333;--toast-text:#fff;
  --btn-bg:#f0f0f0;--btn-hover:#e0e0e0;--btn-text:#333;
  --btn-accent-bg:#1976d2;--btn-accent-text:#fff;--btn-accent-hover:#1565c0;
  --input-bg:#fff;--input-border:#ddd;--input-focus:#1976d2;
}
@media(prefers-color-scheme:dark){:root{
  --bg:#1a1a1a;--bg-alt:#222;--bg-hover:rgba(100,181,246,0.08);
  --bg-selected:rgba(100,181,246,0.12);--bg-toolbar:#242424;
  --text:#e0e0e0;--text-sec:#aaa;--text-dim:#777;
  --border:#444;--border-light:#333;
  --accent:#64b5f6;
  --research-dot:#64b5f6;
  --pop-bg:#2d2d2d;--pop-shadow:0 4px 20px rgba(0,0,0,0.4);
  --toast-bg:#e0e0e0;--toast-text:#1a1a1a;
  --btn-bg:#333;--btn-hover:#444;--btn-text:#e0e0e0;
  --btn-accent-bg:#1565c0;--btn-accent-text:#fff;--btn-accent-hover:#1976d2;
  --input-bg:#2d2d2d;--input-border:#555;--input-focus:#64b5f6;
}}
*{box-sizing:border-box}
body{font-family:system-ui,-apple-system,sans-serif;margin:0;padding:12px;color:var(--text);background:var(--bg);font-size:13px}
#toolbar{display:flex;align-items:center;gap:8px;padding:8px 4px;margin-bottom:8px;flex-wrap:wrap}
#toolbar #sum{font-weight:600;font-size:13px;flex:1;min-width:150px;color:var(--text-sec)}
#toolbar button{padding:5px 12px;border:1px solid var(--border);border-radius:5px;font-size:12px;cursor:pointer;background:var(--btn-bg);color:var(--btn-text);transition:background .15s}
#toolbar button:hover:not(:disabled){background:var(--btn-hover)}
#toolbar button:disabled{opacity:.4;cursor:default}
#toolbar #copyBtn:not(:disabled){background:var(--btn-accent-bg);color:var(--btn-accent-text);border-color:transparent}
#toolbar #copyBtn:not(:disabled):hover{background:var(--btn-accent-hover)}
.wrap{max-height:420px;overflow:auto;border:1px solid var(--border);border-radius:6px}
table{border-collapse:separate;border-spacing:0;width:100%;font-size:13px}
th,td{padding:6px 10px;text-align:left}
.hdr-row th{background:var(--bg-toolbar);position:sticky;top:0;z-index:3;border-bottom:2px solid var(--border);font-size:12px;font-weight:600;white-space:nowrap;cursor:pointer;user-select:none;transition:background .15s}
.hdr-row th:hover{background:var(--bg-hover)}
.sort-arrow{font-size:10px;margin-left:3px;opacity:.5}
.sort-arrow.active{opacity:1;color:var(--accent)}
.flt-row th{position:sticky;top:30px;z-index:3;background:var(--bg-toolbar);padding:4px;border-bottom:1px solid var(--border);cursor:default}
.flt-row input{width:100%;padding:3px 6px;border:1px solid var(--input-border);border-radius:3px;font-size:11px;background:var(--input-bg);color:var(--text);outline:none;transition:border-color .15s}
.flt-row input:focus{border-color:var(--input-focus)}
.flt-row input::placeholder{color:var(--text-dim)}
td{border-bottom:1px solid var(--border-light);max-width:400px;vertical-align:top;word-wrap:break-word;white-space:pre-wrap;position:relative;transition:background .1s}
td:hover{background:var(--bg-hover)}
td.has-research::after{content:"";position:absolute;top:6px;right:4px;width:6px;height:6px;border-radius:50%;background:var(--research-dot);opacity:.7}
tr.selected td{background:var(--bg-selected)!important}
tr:nth-child(even) td{background:var(--bg-alt)}
tr:nth-child(even).selected td{background:var(--bg-selected)!important}
a{color:var(--accent);text-decoration:none;word-break:break-all}
a:hover{text-decoration:underline}
td:first-child{position:sticky;left:0;background:inherit;z-index:1;font-weight:500}
.hdr-row th:first-child{position:sticky;left:0;z-index:4}
.flt-row th:first-child{position:sticky;left:0;z-index:4}
.popover{position:fixed;background:var(--pop-bg);border:1px solid var(--border);border-radius:8px;box-shadow:var(--pop-shadow);max-width:420px;min-width:200px;z-index:100;overflow:hidden;opacity:0;transform:translateY(-4px);transition:opacity .15s,transform .15s;pointer-events:none}
.popover.visible{opacity:1;transform:translateY(0);pointer-events:auto}
.pop-hdr{padding:8px 12px;font-size:11px;font-weight:600;color:var(--text-sec);border-bottom:1px solid var(--border-light);background:var(--bg-alt)}
.pop-body{padding:10px 12px;font-size:12px;line-height:1.5;white-space:pre-wrap;max-height:300px;overflow-y:auto;color:var(--text)}
.toast{position:fixed;bottom:20px;left:50%;transform:translateX(-50%) translateY(60px);background:var(--toast-bg);color:var(--toast-text);padding:8px 20px;border-radius:20px;font-size:13px;font-weight:500;opacity:0;transition:opacity .2s,transform .2s;pointer-events:none;z-index:200}
.toast.show{opacity:1;transform:translateX(-50%) translateY(0)}
</style></head><body>
<div id="toolbar">
  <span id="sum">Loading...</span>
  <button id="selAllBtn">Select all</button>
  <button id="copyBtn" disabled>Copy (0)</button>
</div>
<div class="wrap"><table id="tbl"></table></div>
<div id="pop" class="popover"><div class="pop-hdr"></div><div class="pop-body"></div></div>
<div id="toast" class="toast">Copied!</div>
<script type="module">
import*as _SDK from"SCRIPT_SRC";
const App=_SDK.App;
function applyTheme(){try{_SDK.applyDocumentTheme?.()}catch{}try{_SDK.applyHostStyleVariables?.()}catch{}try{_SDK.applyHostFonts?.()}catch{}}

const app=new App({name:"EveryRow Results",version:"2.0.0"});
const tbl=document.getElementById("tbl");
const sum=document.getElementById("sum");
const selAllBtn=document.getElementById("selAllBtn");
const copyBtn=document.getElementById("copyBtn");
const pop=document.getElementById("pop");
const popHdr=pop.querySelector(".pop-hdr");
const popBody=pop.querySelector(".pop-body");
const toast=document.getElementById("toast");

const S={rows:[],allCols:[],filteredIdx:[],sortCol:null,sortDir:0,filters:{},selected:new Set(),lastClick:null};

/* --- theming --- */
app.onhostcontextchanged=()=>applyTheme();

/* --- helpers --- */
function esc(s){const d=document.createElement("div");d.textContent=String(s);return d.innerHTML;}
function escAttr(s){return esc(s).replace(/"/g,"&quot;");}

/* --- data processing --- */
function flat(obj,pre){
  const o={};
  for(const[k,v]of Object.entries(obj)){
    const key=pre?pre+"."+k:k;
    if(v&&typeof v==="object"&&!Array.isArray(v))Object.assign(o,flat(v,key));
    else o[key]=v;
  }
  return o;
}

function flatWithResearch(obj){
  const research={};
  if(obj.research&&typeof obj.research==="object"&&!Array.isArray(obj.research)){
    for(const[k,v]of Object.entries(obj.research)){
      if(v!=null)research[k]=typeof v==="string"?v:String(v);
    }
  }
  return{display:flat(obj),research};
}

function processData(data){
  if(!Array.isArray(data))data=[data];
  if(!data.length){sum.textContent="No results";tbl.innerHTML="";return;}
  S.rows=data.map(r=>flatWithResearch(r));
  const colSet=new Set();
  S.rows.forEach(r=>{for(const k of Object.keys(r.display))colSet.add(k)});
  const all=[...colSet];
  S.allCols=[...all.filter(k=>!k.includes(".")),...all.filter(k=>k.includes("."))];
  S.sortCol=null;S.sortDir=0;S.filters={};S.selected.clear();S.lastClick=null;
  S.filteredIdx=S.rows.map((_,i)=>i);
  renderTable();
}

/* --- filter & sort --- */
function applyFilterAndSort(){
  let idx=S.rows.map((_,i)=>i);
  for(const[col,q]of Object.entries(S.filters)){
    if(!q)continue;
    const lq=q.toLowerCase();
    idx=idx.filter(i=>{const v=S.rows[i].display[col];return v!=null&&String(v).toLowerCase().includes(lq);});
  }
  if(S.sortCol&&S.sortDir!==0){
    const col=S.sortCol,dir=S.sortDir;
    idx.sort((a,b)=>{
      const va=S.rows[a].display[col],vb=S.rows[b].display[col];
      if(va==null&&vb==null)return 0;if(va==null)return 1;if(vb==null)return-1;
      return String(va).localeCompare(String(vb),undefined,{numeric:true,sensitivity:"base"})*dir;
    });
  }
  S.filteredIdx=idx;
  const filtSet=new Set(idx);
  for(const s of S.selected){if(!filtSet.has(s))S.selected.delete(s);}
  renderTable();
}

let filterTimer=null;
function onFilterInput(col,val){S.filters[col]=val;clearTimeout(filterTimer);filterTimer=setTimeout(()=>applyFilterAndSort(),150);}

/* --- research lookup --- */
function getResearch(row,col){
  const r=row.research;
  if(!r||!Object.keys(r).length)return null;
  if(r[col]!=null)return r[col];
  if(col.startsWith("research.")){const base=col.slice(9);if(r[base]!=null)return r[base];}
  return null;
}

/* --- render --- */
function renderTable(){
  const cols=S.allCols;
  if(!cols.length){tbl.innerHTML="";return;}
  const activeEl=document.activeElement;
  const activeFilterCol=activeEl&&activeEl.matches&&activeEl.matches('.flt-row input')?activeEl.dataset.col:null;
  const cursorPos=activeFilterCol?activeEl.selectionStart:0;

  let h='<thead><tr class="hdr-row">';
  for(const c of cols){
    let arrow='<span class="sort-arrow">&#9650;</span>';
    if(S.sortCol===c)arrow=S.sortDir===1?'<span class="sort-arrow active">&#9650;</span>':'<span class="sort-arrow active">&#9660;</span>';
    h+='<th data-col="'+escAttr(c)+'">'+esc(c)+arrow+'</th>';
  }
  h+='</tr><tr class="flt-row">';
  for(const c of cols){
    h+='<th><input data-col="'+escAttr(c)+'" placeholder="Filter..." value="'+escAttr(S.filters[c]||"")+'"></th>';
  }
  h+='</tr></thead><tbody>';
  for(const i of S.filteredIdx){
    const row=S.rows[i],sel=S.selected.has(i)?' class="selected"':"";
    h+='<tr data-idx="'+i+'"'+sel+'>';
    for(const c of cols){
      const hasR=getResearch(row,c)!=null;
      const v=row.display[c],cls=hasR?' class="has-research"':"",dc=' data-col="'+escAttr(c)+'"';
      if(v==null){h+="<td"+cls+dc+"></td>";}
      else{const s=String(v);h+=s.match(/^https?:\\/\\//)
        ?'<td'+cls+dc+'><a href="'+escAttr(s)+'" target="_blank">'+esc(s)+'</a></td>'
        :'<td'+cls+dc+'>'+esc(s)+'</td>';}
    }
    h+='</tr>';
  }
  tbl.innerHTML=h+'</tbody>';

  const total=S.rows.length,shown=S.filteredIdx.length;
  sum.textContent=(shown<total?shown+" of "+total:String(total))+" rows, "+cols.length+" columns";
  updateCopyBtn();

  tbl.querySelectorAll('.flt-row input').forEach(inp=>{
    inp.addEventListener('input',()=>onFilterInput(inp.dataset.col,inp.value));
  });
  if(activeFilterCol){
    tbl.querySelectorAll('.flt-row input').forEach(inp=>{
      if(inp.dataset.col===activeFilterCol){inp.focus();try{inp.setSelectionRange(cursorPos,cursorPos)}catch{}}
    });
  }

  requestAnimationFrame(()=>{
    const hdrRow=tbl.querySelector('.hdr-row');
    if(hdrRow){const h=hdrRow.getBoundingClientRect().height;tbl.querySelectorAll('.flt-row th').forEach(th=>th.style.top=h+'px');}
  });
}

/* --- sort --- */
tbl.addEventListener("click",e=>{
  const th=e.target.closest(".hdr-row th");
  if(!th)return;
  const col=th.dataset.col;
  if(S.sortCol===col){S.sortDir=S.sortDir===1?-1:S.sortDir===-1?0:1;if(S.sortDir===0)S.sortCol=null;}
  else{S.sortCol=col;S.sortDir=1;}
  applyFilterAndSort();
});

/* --- selection --- */
tbl.addEventListener("click",e=>{
  if(e.target.closest(".hdr-row")||e.target.closest(".flt-row")||e.target.closest("a"))return;
  const tr=e.target.closest("tbody tr");if(!tr)return;
  const idx=parseInt(tr.dataset.idx,10);if(isNaN(idx))return;
  if(e.shiftKey&&S.lastClick!=null){
    const posA=S.filteredIdx.indexOf(S.lastClick),posB=S.filteredIdx.indexOf(idx);
    if(posA>=0&&posB>=0){const lo=Math.min(posA,posB),hi=Math.max(posA,posB);for(let p=lo;p<=hi;p++)S.selected.add(S.filteredIdx[p]);}
  }else if(e.ctrlKey||e.metaKey){
    if(S.selected.has(idx))S.selected.delete(idx);else S.selected.add(idx);
  }else{
    if(S.selected.size===1&&S.selected.has(idx))S.selected.clear();
    else{S.selected.clear();S.selected.add(idx);}
  }
  S.lastClick=idx;updateSelection();updateCopyBtn();
});

function updateSelection(){
  tbl.querySelectorAll("tbody tr").forEach(tr=>{
    const idx=parseInt(tr.dataset.idx,10);tr.classList.toggle("selected",S.selected.has(idx));
  });
}
function updateCopyBtn(){const n=S.selected.size;copyBtn.textContent="Copy"+(n>0?" ("+n+")":"");copyBtn.disabled=n===0;}

/* --- select all --- */
selAllBtn.addEventListener("click",()=>{
  if(S.selected.size===S.filteredIdx.length)S.selected.clear();
  else{S.selected.clear();S.filteredIdx.forEach(i=>S.selected.add(i));}
  updateSelection();updateCopyBtn();
});

/* --- copy --- */
copyBtn.addEventListener("click",async()=>{
  if(!S.selected.size)return;
  const cols=S.allCols,lines=[cols.join("\\t")];
  for(const i of S.filteredIdx){
    if(!S.selected.has(i))continue;
    lines.push(cols.map(c=>{const v=S.rows[i].display[c];return v==null?"":String(v).replace(/\\t/g," ");}).join("\\t"));
  }
  try{await navigator.clipboard.writeText(lines.join("\\n"));showToast("Copied "+S.selected.size+" row"+(S.selected.size>1?"s":""));}
  catch{showToast("Copy failed");}
});
function showToast(msg){toast.textContent=msg;toast.classList.add("show");setTimeout(()=>toast.classList.remove("show"),2000);}

/* --- popover --- */
let popTimer=null,popTarget=null,popVisible=false;

function showPopover(td){
  const tr=td.closest("tr");const idx=parseInt(tr.dataset.idx,10);const col=td.dataset.col;
  const row=S.rows[idx];if(!row)return;
  const text=getResearch(row,col);if(text==null)return;
  popHdr.textContent="research."+col.replace(/^research\\./,"");
  popBody.textContent=text;
  const rect=td.getBoundingClientRect();
  let left=rect.left,top=rect.bottom+4;
  pop.classList.add("visible");popVisible=true;
  const pw=pop.offsetWidth,ph=pop.offsetHeight;
  if(left+pw>window.innerWidth-8)left=window.innerWidth-pw-8;
  if(left<8)left=8;
  if(top+ph>window.innerHeight-8)top=rect.top-ph-4;
  pop.style.left=left+"px";pop.style.top=top+"px";
}
function hidePopover(){pop.classList.remove("visible");popVisible=false;popTarget=null;}

document.addEventListener("mouseover",e=>{
  if(pop.contains(e.target)){clearTimeout(popTimer);return;}
  const td=e.target.closest?e.target.closest("td"):null;
  if(td&&tbl.contains(td)&&td.classList.contains("has-research")){
    if(td===popTarget&&popVisible){clearTimeout(popTimer);return;}
    clearTimeout(popTimer);if(popVisible)hidePopover();
    popTarget=td;popTimer=setTimeout(()=>showPopover(td),300);
  }else{
    clearTimeout(popTimer);popTarget=null;
    if(popVisible)popTimer=setTimeout(()=>{if(!pop.matches(":hover"))hidePopover();},150);
  }
});
pop.addEventListener("mouseleave",()=>{clearTimeout(popTimer);hidePopover();});
document.addEventListener("keydown",e=>{if(e.key==="Escape"&&popVisible)hidePopover();});

/* --- data loading --- */
app.ontoolresult=({content})=>{
  const t=content?.find(c=>c.type==="text");if(!t)return;
  let meta;try{meta=JSON.parse(t.text);}catch{sum.textContent=t.text;return;}
  if(meta.results_url){
    if(meta.preview)processData(meta.preview);
    sum.textContent="Loading full results...";
    const opts=meta.download_token?{headers:{"Authorization":"Bearer "+meta.download_token}}:{};
    fetch(meta.results_url,opts).then(r=>r.json()).then(data=>processData(data)).catch(()=>{
      if(!meta.preview)sum.textContent="Failed to load results";
    });
  }else if(Array.isArray(meta)){processData(meta);}
  else{sum.textContent=JSON.stringify(meta);}
};

await app.connect();
applyTheme();
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
