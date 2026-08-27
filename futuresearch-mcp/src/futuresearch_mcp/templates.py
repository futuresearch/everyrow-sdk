"""MCP App UI HTML template for the unified session widget."""

import html

_APP_SCRIPT_SRC = "https://unpkg.com/@modelcontextprotocol/ext-apps@1.7.1/app-with-deps"


UNIFIED_HTML = """<!DOCTYPE html>
<html><head><meta name="referrer" content="no-referrer"><meta name="color-scheme" content="light dark">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500;600&display=swap" rel="stylesheet">
<style>
:root{
  --bg:#fff;--bg-alt:#fafafa;--bg-hover:rgba(77,79,189,0.05);
  --bg-selected:rgba(77,79,189,0.10);--bg-toolbar:#f4f4f5;
  --text:#333;--text-sec:#525252;--text-dim:#a3a3a3;
  --border:rgba(0,0,0,0.08);--border-light:rgba(0,0,0,0.04);
  --accent:#4D4FBD;
  /* Distinct from --accent (links) so the eye reads them as different. */
  --section-purple:#1D1F8A;
  --median-fill:#4D4FBD;
  --median-text:#1D1F8A;
  --iqr-fill:rgba(77,79,189,0.15);
  --iqr-stroke:rgba(77,79,189,0.4);
  --research-dot:#9294F0;
  --pop-bg:#fff;--pop-shadow:0 4px 20px rgba(0,0,0,0.12);
  --toast-bg:#333;--toast-text:#fff;
  --btn-bg:#f5f5f5;--btn-hover:#e5e5e5;--btn-text:#333;
  --btn-accent-bg:#4D4FBD;--btn-accent-text:#fff;--btn-accent-hover:#1D1F8A;
  --input-bg:#fff;--input-border:#e5e5e5;--input-focus:#4D4FBD;
  --seg-done:#2d7a3e;--seg-run:#4D4FBD;--seg-fail:#e53935;
}
@media(prefers-color-scheme:dark){:root{
  --bg:#111111;--bg-alt:#1a1a1a;--bg-hover:rgba(146,148,240,0.08);
  --bg-selected:rgba(146,148,240,0.12);--bg-toolbar:#1a1a1a;
  --text:#e4e4e7;--text-sec:#a1a1aa;--text-dim:#71717a;
  --border:rgba(255,255,255,0.08);--border-light:rgba(255,255,255,0.04);
  --accent:#9294F0;
  --section-purple:#9294F0;
  --median-fill:#9294F0;
  --median-text:#CFCFFF;
  --iqr-fill:rgba(146,148,240,0.18);
  --iqr-stroke:rgba(146,148,240,0.45);
  --research-dot:#9294F0;
  --pop-bg:#1e1e1e;--pop-shadow:0 4px 20px rgba(0,0,0,0.5);
  --toast-bg:#e4e4e7;--toast-text:#111111;
  --btn-bg:#262626;--btn-hover:#404040;--btn-text:#e4e4e7;
  --btn-accent-bg:#4D4FBD;--btn-accent-text:#fff;--btn-accent-hover:#9294F0;
  --input-bg:#1e1e1e;--input-border:#3f3f46;--input-focus:#9294F0;
  --seg-done:#B8E6A0;--seg-run:#9294F0;--seg-fail:#e53935;
}}
*{box-sizing:border-box}
body{font-family:'JetBrains Mono',ui-monospace,monospace;margin:0;padding:0;color:var(--text);background:var(--bg);font-size:13px;height:0;overflow:hidden}

/* ── Progress section ── */
.progress-section{padding:12px 12px 0}
.prog-info{font-size:12px;color:var(--text-sec);margin:6px 0;display:flex;align-items:center;gap:12px;flex-wrap:wrap;letter-spacing:0.01em}
/* --seg-done is a text colour: dark enough to read on the page in light
   mode, light enough in dark. It is not usable as a banner background. */
.status-done{color:var(--seg-done);font-weight:500}.status-fail{color:var(--seg-fail);font-weight:500}
.eta{color:var(--text-dim);font-size:10px}
.poll-note{font-size:11px;color:var(--text-dim);margin:2px 0 6px;font-style:italic}
@keyframes flash{0%,100%{background:transparent}50%{background:rgba(45,122,62,.1)}}
.flash{animation:flash 1s ease 3}

/* ── Tab bar ── */
.tab-bar{display:flex;gap:0;border-bottom:1px solid var(--border);margin:0 12px 8px}
.tab-btn{padding:6px 16px;border:none;background:none;font-size:10px;font-weight:600;color:var(--text-dim);cursor:pointer;border-bottom:2px solid transparent;margin-bottom:-1px;transition:color .15s,border-color .15s;text-transform:uppercase;letter-spacing:0.05em}
.tab-btn:hover{color:var(--text)}
.tab-btn.active{color:var(--accent);border-bottom-color:var(--accent)}

/* ── Activity tab: per-researcher boxes ─────────────────────────────
   One box per unique trace_id seen across polls. Each box shows the
   researcher's latest micro-summary, plus a small icon, with a faint
   pulsing background while the overall task is running. The team-level
   aggregate text (when present) is shown as a small italic banner
   above the boxes. */
.activity-tab{padding:0 12px;max-height:360px;overflow-y:auto}
.activity-banner{margin:0 0 8px;padding:8px 12px;background:var(--bg-alt);border:1px solid var(--border);border-radius:4px;font-size:11px;line-height:1.5;color:var(--text-sec)}
/* Title row: "researcher team activity" label + horizontal strip of
   icons sitting inline on the same baseline. Icons live INSIDE the
   header rather than below the aggregate text so the visual identity
   of "the team" reads alongside the section title. */
.activity-banner-header{display:flex;align-items:center;gap:8px;flex-wrap:wrap;margin-bottom:4px}
.activity-banner-label{font-style:normal;font-size:9px;color:var(--text-dim);text-transform:uppercase;letter-spacing:0.05em}
.activity-banner-text{font-style:italic;display:block}
.banner-icon-row{display:inline-flex;align-items:center;gap:4px;flex-wrap:wrap}
.banner-icon-cell{width:18px;height:18px;border-radius:50%;background:var(--bg);border:1px solid var(--border);display:inline-flex;align-items:center;justify-content:center;color:var(--accent);transition:background-color .3s ease,border-color .3s ease}
.banner-icon-cell.running{animation:researcher-pulse 2.2s ease-in-out infinite}
.banner-icon-more{font-size:11px;color:var(--text-dim);font-family:'JetBrains Mono',monospace;margin-left:2px}
@keyframes researcher-pulse{
  0%,100%{background:var(--bg);border-color:var(--border);color:var(--accent)}
  50%{background:rgba(146,148,240,0.15);border-color:rgba(146,148,240,0.55);color:var(--median-fill)}
}
.researchers-list{display:flex;flex-direction:column;gap:6px;margin:0;padding:0;list-style:none}
/* Box itself is calm — pulse lives on the icon only. */
.researcher-box{border:1px solid var(--border);border-radius:4px;padding:8px 12px;background:var(--bg)}
.researcher-head{display:flex;align-items:center;gap:6px;margin-bottom:3px;font-size:11px}
/* Wrap the icon SVG in a same-sized circle so the pulse animation
   has a fixed background area to bloom into. */
.researcher-icon-wrap{width:18px;height:18px;border-radius:50%;background:var(--bg);border:1px solid var(--border);display:inline-flex;align-items:center;justify-content:center;color:var(--accent);flex-shrink:0;transition:background-color .3s ease,border-color .3s ease}
.researcher-icon-wrap.running{animation:researcher-pulse 2.2s ease-in-out infinite}
.researcher-icon{flex-shrink:0;display:block}
.researcher-label{font-weight:600;color:var(--text)}
.researcher-rows{color:var(--text-dim);font-size:10px;font-variant-numeric:tabular-nums}
.researcher-iter-bar{display:block;flex:1;height:5px;background:var(--border-light);border-radius:3px;overflow:hidden;margin-left:auto;min-width:50px;max-width:110px}
/* Must be block: inline spans ignore explicit height and width %. */
.researcher-iter-bar-fill{display:block;height:100%;background:var(--median-fill);border-radius:3px;transition:width .4s ease}
.researcher-summary{font-size:11px;line-height:1.5;color:var(--text-sec)}
.researchers-empty{padding:14px 4px;color:var(--text-dim);font-size:11px;text-align:center}

/* ── Results table ── */
#toolbar{display:flex;align-items:center;gap:8px;padding:8px 4px;margin-bottom:8px;flex-wrap:wrap}
#toolbar #sum{font-weight:500;font-size:11px;flex:1;min-width:150px;color:var(--text-sec)}
#toolbar button{padding:5px 12px;border:1px solid var(--border);border-radius:4px;font-size:11px;font-weight:500;cursor:pointer;background:var(--btn-bg);color:var(--btn-text);transition:background-color .15s ease}
#toolbar button:hover:not(:disabled){background:var(--btn-hover)}
#toolbar button:disabled{opacity:.4;cursor:default}
#toolbar #copyBtn:not(:disabled){background:var(--btn-accent-bg);color:var(--btn-accent-text);border-color:transparent}
#toolbar #copyBtn:not(:disabled):hover{background:var(--btn-accent-hover)}
.wrap{max-height:520px;overflow:auto;border:1px solid var(--border);border-radius:4px 4px 0 0}
table{border-collapse:separate;border-spacing:0;width:100%;font-size:12px}
th,td{padding:6px 10px;text-align:left}
.hdr-row th{background:var(--bg-toolbar);position:sticky;top:0;z-index:3;border-bottom:1px solid var(--border);font-size:10px;font-weight:600;white-space:nowrap;cursor:pointer;user-select:none;transition:background-color .15s ease;text-transform:uppercase;letter-spacing:0.05em;color:var(--accent)}
.hdr-row th:hover{background:var(--bg-hover)}
.sort-arrow{font-size:9px;margin-left:3px;opacity:.4}
.sort-arrow.active{opacity:1;color:var(--accent)}
.flt-row th{position:sticky;top:30px;z-index:3;background:var(--bg-toolbar);padding:4px;border-bottom:1px solid var(--border);cursor:default}
.flt-row input{width:100%;padding:3px 6px;border:1px solid var(--input-border);border-radius:4px;font-size:10px;background:var(--input-bg);color:var(--text);outline:none;transition:border-color .15s ease;font-family:inherit}
.flt-row input:focus{border-color:var(--input-focus)}
.flt-row input::placeholder{color:var(--text-dim)}
td{border-bottom:1px solid var(--border-light);max-width:400px;vertical-align:top;word-wrap:break-word;white-space:pre-wrap;position:relative;transition:background-color .15s ease}
td:hover{background:var(--bg-hover)}
td.has-research::after{content:"";position:absolute;top:6px;right:4px;width:6px;height:6px;border-radius:50%;background:var(--research-dot);opacity:.6}
tr.selected td{background:var(--bg-selected)!important}
td.cell-focused{outline:1px solid var(--accent);outline-offset:-1px;z-index:2}
tr:nth-child(even) td{background:var(--bg-alt)}
tr:nth-child(even).selected td{background:var(--bg-selected)!important}
a{color:var(--accent);text-decoration:none;word-break:break-all}
a:hover{text-decoration:underline;text-underline-offset:2px}
.row-num{position:sticky;left:0;z-index:1;background:var(--bg);width:40px;min-width:40px;max-width:40px;text-align:center;color:var(--text-dim);font-size:10px;font-variant-numeric:tabular-nums;cursor:pointer;user-select:none;padding:6px 4px;box-shadow:2px 0 4px rgba(0,0,0,.04)}
tr:nth-child(even) .row-num{background:var(--bg-alt)}
.hdr-row .row-num{z-index:4;font-weight:600;color:var(--text-sec);cursor:default;background:var(--bg-toolbar)}
.flt-row .row-num{z-index:4;cursor:default;background:var(--bg-toolbar)}
tr.selected .row-num{background:var(--bg-selected)!important}
.popover{position:fixed;background:var(--pop-bg);border:1px solid var(--border);border-radius:4px;box-shadow:var(--pop-shadow);max-width:min(720px,90vw);min-width:280px;max-height:min(500px,70vh);z-index:100;overflow:hidden;opacity:0;transform:translateY(-4px);transition:opacity .15s,transform .15s;pointer-events:none;display:flex;flex-direction:column}
.popover.visible{opacity:1;transform:translateY(0);pointer-events:auto}
.pop-hdr{padding:8px 12px;font-size:10px;font-weight:600;color:var(--text-sec);border-bottom:1px solid var(--border-light);background:var(--bg-alt);text-transform:uppercase;letter-spacing:0.03em}
.pop-body{padding:10px 12px;font-size:11px;line-height:1.5;white-space:pre-wrap;overflow-y:auto;color:var(--text);flex:1}
.toast{position:fixed;bottom:20px;left:50%;transform:translateX(-50%) translateY(60px);background:var(--toast-bg);color:var(--toast-text);padding:6px 16px;border-radius:4px;font-size:11px;font-weight:500;opacity:0;transition:opacity .2s,transform .2s;pointer-events:none;z-index:200}
.toast.show{opacity:1;transform:translateX(-50%) translateY(0)}
.resize-handle{height:4px;background:var(--border-light);cursor:ns-resize;border-radius:0 0 4px 4px;transition:background .15s;margin-top:-1px;border:1px solid var(--border);border-top:none}
.resize-handle:hover,.resize-handle.active{background:var(--accent);opacity:.4}
#expandBtn{font-size:14px;padding:5px 8px}
body.fullscreen .wrap{max-height:calc(100vh - 80px)!important}
body.fullscreen .resize-handle{display:none}
.copy-modal{position:fixed;inset:0;background:rgba(0,0,0,.5);z-index:300;display:flex;align-items:center;justify-content:center;opacity:0;pointer-events:none;transition:opacity .2s}
.copy-modal.show{opacity:1;pointer-events:auto}
.copy-modal-box{background:var(--bg);border:1px solid var(--border);border-radius:4px;padding:16px;max-width:600px;width:90%;max-height:80vh;display:flex;flex-direction:column;gap:8px}
.copy-modal-box textarea{width:100%;height:300px;font-family:inherit;font-size:11px;border:1px solid var(--border);border-radius:4px;padding:8px;background:var(--input-bg);color:var(--text);resize:vertical}
.copy-modal-box .modal-btns{display:flex;gap:8px;justify-content:flex-end}
.copy-modal-box button{padding:6px 16px;border:1px solid var(--border);border-radius:4px;background:var(--btn-bg);color:var(--btn-text);cursor:pointer;font-size:11px;font-family:inherit}
.done-banner{position:fixed;top:0;left:0;right:0;background:var(--btn-accent-bg);color:var(--btn-accent-text);padding:10px 16px;z-index:250;display:flex;align-items:center;gap:10px;font-size:12px;font-weight:500;box-shadow:0 2px 8px rgba(0,0,0,.15);transform:translateY(-100%);transition:transform .3s ease}
.done-banner.show{transform:translateY(0)}
.done-banner .banner-text{flex:1}
.done-banner .banner-close{background:none;border:none;color:var(--btn-accent-text);font-size:18px;cursor:pointer;padding:0 4px;line-height:1;opacity:.8}
.done-banner .banner-close:hover{opacity:1}
.col-resize-handle{position:absolute;top:0;right:-2px;width:4px;height:100%;cursor:col-resize;z-index:5;user-select:none}
.col-resize-handle:hover{background:var(--accent);opacity:.3}
body.col-resizing,body.col-resizing *{cursor:col-resize!important;user-select:none!important}
body.row-resizing,body.row-resizing *{cursor:row-resize!important;user-select:none!important}
.cell-text{display:inline}
.cell-more,.cell-less{cursor:pointer;color:var(--accent);font-size:10px;margin-left:4px;white-space:nowrap;font-weight:500;padding:1px 4px;border-radius:3px;background:rgba(77,79,189,0.08)}
.cell-more:hover,.cell-less:hover{text-decoration:underline;text-underline-offset:2px;background:rgba(77,79,189,0.15)}
.export-btns{display:inline-flex;gap:2px}
.export-btns a{font-family:inherit}
#globalSearch{padding:4px 8px;border:1px solid var(--input-border);border-radius:4px;font-size:11px;background:var(--input-bg);color:var(--text);outline:none;width:160px;transition:border-color .15s ease,width .2s ease;font-family:inherit}
#globalSearch:focus{border-color:var(--input-focus);width:220px}
#globalSearch::placeholder{color:var(--text-dim)}
.col-ghost{position:fixed;background:var(--bg-toolbar);border:1px solid var(--accent);border-radius:4px;padding:4px 8px;font-size:11px;font-weight:600;opacity:.85;pointer-events:none;z-index:200;white-space:nowrap}
body.col-dragging,body.col-dragging *{cursor:grabbing!important;user-select:none!important}
.hdr-row th.drag-over-left{box-shadow:inset 3px 0 0 var(--accent)}
.hdr-row th.drag-over-right{box-shadow:inset -3px 0 0 var(--accent)}
/* ── Forecast cards ── */
.fc-toolbar{display:flex;align-items:center;gap:8px;padding:8px 12px 0;flex-wrap:wrap}
.fc-toolbar #fcSum{flex:1;font-size:11px;color:var(--text-sec);min-width:120px}
.fc-toolbar button{padding:5px 12px;border:1px solid var(--border);border-radius:4px;font-size:11px;cursor:pointer;background:var(--btn-bg);color:var(--btn-text);transition:background-color .15s ease;font-family:inherit}
.fc-toolbar button:hover:not(:disabled){background:var(--btn-hover)}
.fc-toolbar button:disabled{opacity:.4;cursor:default}
.fc-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(290px,1fr));gap:10px;padding:10px 12px 12px;align-items:start}
.fc-grid.solo{grid-template-columns:1fr}
.fc-card{border:1px solid var(--border);border-radius:4px;padding:12px 14px;background:var(--bg);display:flex;flex-direction:column;gap:8px;transition:border-color .15s ease,background-color .15s ease;position:relative}
.fc-card[data-clickable="1"]{cursor:pointer}
.fc-card[data-clickable="1"]:hover{border-color:var(--accent)}
.fc-card.expanded{grid-column:1/-1;cursor:default}
.fc-card.solo{cursor:default}
.fc-card.failed{border-color:var(--seg-fail);opacity:.9}
.fc-card-header{display:flex;align-items:baseline;gap:8px;flex-wrap:wrap}
.fc-card-title{flex:1;min-width:0;font-size:13px;font-weight:600;color:var(--text);line-height:1.4;word-wrap:break-word}
.fc-card.solo .fc-card-title{font-size:16px;line-height:1.35}
.fc-row-badge{font-size:10px;color:var(--text-dim);font-weight:normal;font-variant-numeric:tabular-nums;flex-shrink:0}
.fc-fail-tag{font-size:10px;color:var(--seg-fail);font-weight:500;letter-spacing:0.05em;text-transform:uppercase}
.fc-prob{display:flex;align-items:baseline;gap:8px;margin-top:2px}
.fc-prob-value{font-size:28px;font-weight:600;color:var(--median-text);font-variant-numeric:tabular-nums;line-height:1}
.fc-card.solo .fc-prob-value{font-size:40px}
.fc-prob-label{font-size:10px;color:var(--text-dim);text-transform:uppercase;letter-spacing:0.05em}
/* One track+fill for every probability the widget draws, so a binary card and
   a categorical card carry the same mark at different counts. */
.fc-prob-bar,.fc-opt-bar{height:6px;border-radius:3px;background:var(--iqr-fill);overflow:hidden}
.fc-prob-bar-fill,.fc-opt-bar-fill{height:100%;border-radius:3px;background:var(--median-fill)}
.fc-prob-bar{margin-top:4px}
/* Grouped option bars (categorical / thresholded / decision). */
.fc-opts{display:flex;flex-direction:column;gap:7px;margin-top:2px}
.fc-opt-head{display:flex;align-items:baseline;gap:8px}
.fc-opt-label{flex:1;min-width:0;font-size:11px;color:var(--text-sec);line-height:1.4;word-wrap:break-word;display:-webkit-box;-webkit-line-clamp:2;-webkit-box-orient:vertical;overflow:hidden}
.fc-card.solo .fc-opt-label,.fc-card.expanded .fc-opt-label{display:block;overflow:visible;font-size:12px}
.fc-opt-value{flex-shrink:0;font-size:12px;font-weight:600;color:var(--text-sec);font-variant-numeric:tabular-nums}
.fc-card.solo .fc-opt-value{font-size:15px}
.fc-opt.lead .fc-opt-label{color:var(--text)}
.fc-opt.lead .fc-opt-value{color:var(--median-text)}
.fc-opt-bar{margin-top:3px}
.fc-opt-bar-fill{opacity:.55}
.fc-opt.lead .fc-opt-bar-fill{opacity:1}
.fc-opt-more{font-size:9px;color:var(--text-dim);font-style:italic}
.fc-alt{margin-top:6px}
.fc-alt .fc-pctl-header{margin-bottom:2px}
/* Conditional forecasts: the condition, then the outcome under each branch. */
.fc-condition{border-left:2px solid var(--iqr-stroke);padding-left:8px;margin-top:2px}
.fc-condition .fc-section-label{margin-top:0}
.fc-condition-text{font-size:11px;color:var(--text-sec);line-height:1.4;word-wrap:break-word}
.fc-card.solo .fc-condition-text{font-size:12px}
.fc-branch{margin-top:8px}
.fc-branch-label{font-size:10px;font-weight:500;color:var(--text-sec);margin-bottom:2px}
.fc-pctl-bar{margin-top:2px}
.fc-pctl-header{display:flex;align-items:baseline;gap:6px;margin-bottom:4px;font-size:11px;flex-wrap:wrap}
.fc-pctl-field{color:var(--text-sec);font-weight:500}
.fc-pctl-median{color:var(--median-text);font-weight:600;font-size:14px}
.fc-card.solo .fc-pctl-median{font-size:20px}
.fc-pctl-units{color:var(--text-dim);font-size:10px}
.fc-pctl-svg-label{font-size:9px;font-family:'JetBrains Mono',monospace}
.fc-pctl-grid{display:grid;grid-template-columns:repeat(5,1fr);gap:4px;margin-top:6px}
.fc-pctl-cell{text-align:center}
.fc-pctl-cell-label{font-size:9px;color:var(--text-dim);margin-bottom:2px}
.fc-pctl-cell-val{font-size:11px;font-weight:600;color:var(--text-sec)}
.fc-pctl-cell-val.median{color:var(--median-text);background:rgba(77,79,189,0.12);border-radius:3px;padding:1px 4px;display:inline-block}
.fc-section-label{font-size:9px;color:var(--text-dim);text-transform:uppercase;letter-spacing:0.05em;margin-top:4px}
.fc-field-row a{color:var(--accent);font-weight:500}
.fc-field-row.clamped{display:-webkit-box;-webkit-line-clamp:4;-webkit-box-orient:vertical;overflow:hidden}
.fc-card.solo .fc-field-row{font-size:13px}
.fc-field-row{font-size:12px;color:var(--text-sec);line-height:1.5;margin-top:4px;word-wrap:break-word}
.fc-field-key{color:var(--section-purple);font-weight:500;margin-right:4px;text-transform:lowercase}
.fc-sources{display:flex;flex-wrap:wrap;gap:4px;padding-top:8px;border-top:1px solid var(--border-light);margin-top:4px}
.fc-source-pill{font-size:9px;padding:2px 6px;border-radius:3px;background:var(--bg-toolbar);color:var(--text-sec);text-decoration:none;font-family:inherit;white-space:nowrap}
.fc-source-pill:hover{color:var(--accent);background:var(--bg-hover);text-decoration:none}
.fc-source-more{font-size:9px;color:var(--text-dim);padding:2px 4px}
.fc-empty{padding:24px 12px;text-align:center;color:var(--text-dim);font-size:12px}
.fc-expand-hint{font-size:9px;color:var(--text-dim);text-align:right;font-style:italic;margin-top:2px}
.fc-card.expanded .fc-expand-hint{display:none}
/* ── Widget frame ── */
.widget-frame{border:1px solid var(--border);border-radius:4px;margin:4px;overflow:hidden}
</style></head><body>
<div class="widget-frame" id="widgetFrame" style="display:none">

<!-- ── Progress section (hidden until progress mode) ── -->
<div id="progressSection" class="progress-section" style="display:none">
  <div id="progressContent"></div>
  <div id="pollNote" class="poll-note" style="display:none"></div>
</div>

<!-- ── Tab bar (hidden until progress mode) ── -->
<div id="tabBar" class="tab-bar" style="display:none">
  <button class="tab-btn active" data-tab="activity">Activity</button>
  <button class="tab-btn" data-tab="results">Results</button>
  <span style="flex:1"></span>
  <button id="expandBtn" title="Toggle fullscreen" style="font-size:13px;padding:4px 8px;border:none;background:none;color:var(--text-dim);cursor:pointer;margin-bottom:-1px">&#x2922;</button>
</div>

<!-- ── Activity tab ── -->
<div id="activityTab" class="activity-tab" style="display:none">
  <div class="activity-list" id="activityList"></div>
</div>

<!-- ── Results tab (full table UI, used for all non-forecast operations) ── -->
<div id="resultsTab" style="display:none">
<div style="display:flex;align-items:center;gap:8px;padding:0 0 6px"><input id="globalSearch" type="text" placeholder="Search all columns..." style="flex:1"></div>
<div id="toolbar">
  <span id="sum">Loading...</span>
  <button id="selAllBtn">Select all</button>
  <button id="copyBtn" disabled>Copy CSV (0)</button>
  <span class="export-btns"><button id="exportLink" title="Copy CSV download link to clipboard">Download CSV</button></span>
</div>
<div class="wrap" id="wrap" style="max-height:520px"><table id="tbl"></table></div>
<div class="resize-handle" id="resizeHandle"></div>
</div>

<!-- ── Forecast tab (cards instead of table; used when task_type=forecast) ── -->
<div id="forecastTab" style="display:none">
  <div class="fc-toolbar">
    <span id="fcSum">Loading...</span>
    <button id="fcExportLink" title="Download all rows as CSV">Download CSV</button>
  </div>
  <div class="fc-grid" id="fcGrid"></div>
</div>
</div><!-- close widget-frame -->

<div id="pop" class="popover"><div class="pop-hdr"></div><div class="pop-body"></div></div>
<div id="doneBanner" class="done-banner"><span class="banner-text">Task complete &mdash; ask Claude to analyze the results.</span><button class="banner-close" id="closeBanner">&times;</button></div>
<div id="toast" class="toast">Copied!</div>
<div id="copyModal" class="copy-modal"><div class="copy-modal-box"><div style="font-weight:600;font-size:13px">Select all and copy (Cmd+C / Ctrl+C)</div><textarea id="copyArea" readonly></textarea><div class="modal-btns"><button id="closeCopyModal">Close</button></div></div></div>
<script type="module">
import*as _SDK from"SCRIPT_SRC";
const App=_SDK.App;
function applyTheme(){try{_SDK.applyDocumentTheme?.()}catch{}try{_SDK.applyHostStyleVariables?.()}catch{}try{_SDK.applyHostFonts?.()}catch{}}

const app=new App({name:"FutureSearch",version:"3.0.0"});
const tbl=document.getElementById("tbl");
const sum=document.getElementById("sum");
const selAllBtn=document.getElementById("selAllBtn");
const copyBtn=document.getElementById("copyBtn");
const pop=document.getElementById("pop");
const popHdr=pop.querySelector(".pop-hdr");
const popBody=pop.querySelector(".pop-body");
const toast=document.getElementById("toast");
const wrap=document.getElementById("wrap");
const resizeHandle=document.getElementById("resizeHandle");
const expandBtn=document.getElementById("expandBtn");
const copyModal=document.getElementById("copyModal");
const copyArea=document.getElementById("copyArea");
const closeCopyModal=document.getElementById("closeCopyModal");
const widgetFrame=document.getElementById("widgetFrame");

/* ── progress & tab elements ── */
const progressSection=document.getElementById("progressSection");
const progressContent=document.getElementById("progressContent");
const pollNote=document.getElementById("pollNote");
const tabBar=document.getElementById("tabBar");
const activityTab=document.getElementById("activityTab");
const activityList=document.getElementById("activityList");
const resultsTab=document.getElementById("resultsTab");
const forecastTab=document.getElementById("forecastTab");
const fcGrid=document.getElementById("fcGrid");
const fcSum=document.getElementById("fcSum");
const fcExportLink=document.getElementById("fcExportLink");
/* Non-null only for forecast tasks, which render cards, not the table. */
let forecastMeta=null;
let forecastRows=null;
/* One scale and one row order across every card on screen. Held here because
   expanding a card re-renders it alone, and a card that changed scale on
   expansion would be worse than one that never shared. */
let forecastDomain=null;

let csvUrl="",pollToken="",downloadUrl="";
const TRUNC=200;
let didDrag=false;
const copyFmt="csv";
let widgetActive=false;
const S={rows:[],allCols:[],filteredIdx:[],sortCol:null,sortDir:0,filters:{},globalQuery:"",selected:new Set(),lastClick:null,isFullscreen:false,focusedCell:null};

/* ── progress state ── */
let pollUrl=null,pollTimer=null,wasDone=false,pollCursor=null;
let progressMode=false,resultsFetched=false;
let currentTaskId=null;
/* Per-researcher state: one entry per unique trace_id ever seen.
   Keys → {summary, updated_at, row_indices, iteration, num} where `num` is
   a stable 1-based display index assigned on first sighting. */
const researcherMap=new Map();
let latestAggregate="";
let activeTab="activity";
/* Iteration-budget heuristic for the per-researcher progress bar.
   Engine effort levels: low=0, medium=5, high=10 iterations. We don't
   know the effort level from the polling payload, so we start at 10
   (covers high-effort default) and ratchet up to the nearest multiple
   of 5 if an observed iteration exceeds it. The bar shows iter/max. */
let iterVisualMax=10;

/* --- theming & display mode --- */
app.onhostcontextchanged=(ctx)=>{
  applyTheme();
  const mode=ctx?.displayMode||"contained";
  const isFull=mode==="fullscreen";
  S.isFullscreen=isFull;
  document.body.classList.toggle("fullscreen",isFull);
  expandBtn.textContent=isFull?"\\u2921":"\\u2922";
  expandBtn.title=isFull?"Exit fullscreen":"Toggle fullscreen";
};

/* --- helpers --- */
function esc(s){const d=document.createElement("div");d.textContent=String(s);return d.innerHTML;}
function escAttr(s){return esc(s).replace(/"/g,"&quot;");}
function truncSafe(s,len){if(s.length<=len)return s;let t=s.slice(0,len);const urlRe=/(https?:\\/\\/[^\\s<>"'\\]]+)$/;const m=t.match(urlRe);if(m){const full=s.slice(m.index).match(/^https?:\\/\\/[^\\s<>"'\\]]+/);if(full&&full[0].length>m[1].length)t=s.slice(0,m.index+full[0].length);}return t;}
function linkify(s){const parts=[];const mdRe=/\\[(.+?)\\]\\((https?:\\/\\/[^)]+)\\)/g;let last=0,m;while((m=mdRe.exec(s))!==null){if(m.index>last)parts.push({type:"text",v:s.slice(last,m.index)});parts.push({type:"link",title:m[1],url:m[2]});last=mdRe.lastIndex;}if(last<s.length)parts.push({type:"text",v:s.slice(last)});let out="";for(const p of parts){if(p.type==="link"){out+='<a href="'+escAttr(p.url)+'" target="_blank" rel="noopener noreferrer">'+esc(p.title)+"</a>";continue;}const t=p.v;const urlRe=/(https?:\\/\\/[^\\s<>"'\\)]+)/g;let ul=0,um;while((um=urlRe.exec(t))!==null){if(um.index>ul)out+=esc(t.slice(ul,um.index));out+='<a href="'+escAttr(um[1])+'" target="_blank" rel="noopener noreferrer">'+esc(um[1])+"</a>";ul=urlRe.lastIndex;}if(ul<t.length)out+=esc(t.slice(ul));}return out;}

/* ── tab switching ── */
function switchTab(tab){
  activeTab=tab;
  tabBar.querySelectorAll(".tab-btn").forEach(b=>b.classList.toggle("active",b.dataset.tab===tab));
  activityTab.style.display=tab==="activity"?"block":"none";
  /* Results tab content is either the table (default) or the forecast
     card grid — both share the same tab; forecastMeta picks the view. */
  const showResults=tab==="results";
  if(forecastMeta){
    forecastTab.style.display=showResults?"block":"none";
    resultsTab.style.display="none";
  }else{
    resultsTab.style.display=showResults?"block":"none";
    forecastTab.style.display="none";
  }
}
tabBar.addEventListener("click",e=>{
  const btn=e.target.closest(".tab-btn");
  if(btn)switchTab(btn.dataset.tab);
});

/* ── progress rendering ── */
function fmtTime(s){
  if(s<60)return s+"s";
  const m=Math.floor(s/60),sec=s%60;
  return m+"m"+((sec>0)?(" "+sec+"s"):"");
}

/* Inline researcher icon — simple person glyph, scales with currentColor. */
const RESEARCHER_ICON_SVG=`<svg class="researcher-icon" viewBox="0 0 24 24" width="14" height="14" fill="currentColor" aria-hidden="true"><circle cx="12" cy="7.5" r="3.5"/><path d="M5 21 c0-3.6 3.1-6.5 7-6.5 s7 2.9 7 6.5 v.5 H5 z"/></svg>`;

/* Upsert per-trace state from a poll's summaries[] (each entry = one
   researcher's most recent micro-summary). Assigns a stable 1-based
   display number on first sighting. Ratchets iterVisualMax upward
   if any researcher exceeds the current value. */
function _updateResearchers(summaries){
  if(!Array.isArray(summaries))return;
  for(const s of summaries){
    if(!s||!s.trace_id)continue;
    const prev=researcherMap.get(s.trace_id);
    const num=prev?prev.num:(researcherMap.size+1);
    const rowIndices=s.row_indices||(s.row_index!=null?[s.row_index]:null);
    const iter=s.iteration_number??(prev?prev.iteration:0);
    researcherMap.set(s.trace_id,{
      num,
      summary:s.summary||"",
      updated_at:s.updated_at||(prev?prev.updated_at:""),
      iteration:iter,
      row_indices:rowIndices,
    });
    /* Ratchet to next multiple of 5 above any observed iteration. */
    if(iter>iterVisualMax)iterVisualMax=Math.ceil(iter/5)*5;
  }
}

function _renderActivity(isRunning){
  const researchers=Array.from(researcherMap.values()).sort((a,b)=>a.num-b.num);
  let h="";
  /* Banner: header row with the "researcher team activity" label and an
     inline strip of researcher icons; aggregate text sits below as the
     body. Icon cap at 10 with a "[…]" suffix if more. Each cell pulses
     while the task is running. */
  const showBanner=latestAggregate||researchers.length>0;
  if(showBanner){
    h+=`<div class="activity-banner">`;
    h+=`<div class="activity-banner-header">`;
    h+=`<span class="activity-banner-label">agent activity</span>`;
    if(researchers.length>0){
      h+=`<div class="banner-icon-row" aria-label="${researchers.length} agents">`;
      const ICON_CAP=10;
      const shown=researchers.slice(0,ICON_CAP);
      for(let i=0;i<shown.length;i++){
        h+=`<span class="banner-icon-cell${isRunning?" running":""}" style="animation-delay:${(i*0.18).toFixed(2)}s">${RESEARCHER_ICON_SVG}</span>`;
      }
      if(researchers.length>ICON_CAP){
        h+=`<span class="banner-icon-more">[&hellip;]</span>`;
      }
      h+=`</div>`;
    }
    h+=`</div>`;
    if(latestAggregate)h+=`<span class="activity-banner-text">${esc(latestAggregate)}</span>`;
    h+=`</div>`;
  }
  if(researchers.length===0){
    h+=`<div class="researchers-empty">Waiting for the first agent to report&hellip;</div>`;
    activityList.innerHTML=h;
    return;
  }
  h+=`<ul class="researchers-list">`;
  for(const r of researchers){
    const rowsLabel=(r.row_indices&&r.row_indices.length)
      ? (r.row_indices.length>1
          ? "rows "+r.row_indices.map(i=>i+1).join(", ")
          : "row "+(r.row_indices[0]+1))
      : "";
    h+=`<li class="researcher-box">`;
    h+=`<div class="researcher-head">`;
    /* Icon-only pulse: a fixed-size wrapper holds the SVG and animates.
       Stagger the pulse start times by ~180ms each so the row reads as
       a coordinated team rather than a strobing block. */
    h+=`<span class="researcher-icon-wrap${isRunning?" running":""}" style="animation-delay:${((r.num-1)*0.18).toFixed(2)}s">${RESEARCHER_ICON_SVG}</span>`;
    h+=`<span class="researcher-label">Researcher #${r.num}</span>`;
    if(rowsLabel)h+=`<span class="researcher-rows">${esc(rowsLabel)}</span>`;
    if(r.iteration){
      const pct=Math.min(100,Math.max(0,(r.iteration/iterVisualMax)*100));
      h+=`<span class="researcher-iter-bar" title="iteration ${esc(String(r.iteration))} of ~${iterVisualMax}">`;
      h+=`<span class="researcher-iter-bar-fill" style="width:${pct}%"></span>`;
      h+=`</span>`;
    }
    h+=`</div>`;
    h+=`<div class="researcher-summary">${esc(r.summary||"Starting research…")}</div>`;
    h+=`</li>`;
  }
  h+=`</ul>`;
  activityList.innerHTML=h;
}

function renderProgress(d){
  const comp=d.completed||0,tot=d.total||0,fail=d.failed||0,run=d.running||0;
  const done=["completed","failed","revoked"].includes(d.status);
  const elapsed=d.elapsed_s||0;

  /* Update per-researcher state from this poll. */
  _updateResearchers(d.summaries);
  if(d.aggregate_summary)latestAggregate=d.aggregate_summary;
  if(d.cursor)pollCursor=d.cursor;

  /* Streaming-view header: keep it minimal. Researcher count (when
     known) + elapsed time + terminal status word. The old segment bar
     and the "X done / X running / X pending / ETA" labels described
     rows, not researchers, and were misleading once we switched to the
     researcher-centric activity view. */
  const numR=researcherMap.size;
  let h="";
  h+=`<div class="prog-info">`;
  if(done){
    const cls=d.status==="completed"?"status-done":"status-fail";
    h+=`<span class="${esc(cls)}">${esc(d.status)}</span>`;
  }
  if(numR>0)h+=`<span>${numR} researcher${numR!==1?"s":""}</span>`;
  if(elapsed)h+=`<span class="eta">${fmtTime(elapsed)}${done?"":" elapsed"}</span>`;
  if(fail&&done)h+=`<span class="status-fail">${fail} failed</span>`;
  h+=`</div>`;
  progressContent.innerHTML=h;

  _renderActivity(!done);

  if(done&&!wasDone){
    wasDone=true;
    progressSection.classList.add("flash");
    /* auto-fetch results on completion */
    if(!resultsFetched)loadResults(d.results);
    showDoneBanner();
  }
  if(done){setPollNote("");stopPoll();}
}


/* ── show completion banner ── */
const doneBanner=document.getElementById("doneBanner");
document.getElementById("closeBanner").addEventListener("click",()=>doneBanner.classList.remove("show"));
function showDoneBanner(){doneBanner.classList.add("show");}

/* ── show results on completion ── */
function renderResults(rows){
  if(forecastMeta){
    /* Forecast tasks get the card grid; the table is hidden. */
    showForecastUI();
    renderForecastCards(Array.isArray(rows)?rows:[rows]);
  }else{
    /* All other operation types keep the original table UI. */
    showResultsUI();
    processData(rows);
  }
  switchTab("results");
  /* NOTE: we deliberately do NOT call app.sendMessage() or
     app.updateModelContext() here. Both are advertised by claude.ai for
     custom connectors but their delivery is reconnect-gated — the message
     only materializes in the chat on page refresh, producing a mystery
     user message that hurts UX more than it helps. The widget shows
     results; the user asks Claude to analyze. */
}

/* A finished task's rows come down with its data, so there is usually nothing
   left to fetch. The download URL is the fallback, and it needs a poll token,
   which stops working 24h after the task was submitted. */
async function loadResults(rows){
  if(resultsFetched)return;
  resultsFetched=true;
  if(rows){renderResults(rows);return;}
  if(!downloadUrl||!pollToken){return;}
  try{
    const jsonUrl=downloadUrl+(downloadUrl.includes("?")?"&":"?")+"token="+encodeURIComponent(pollToken)+"&format=json";
    let dataResp=await fetch(jsonUrl);
    if(dataResp.status===404){
      await new Promise(r=>setTimeout(r,2000));
      dataResp=await fetch(jsonUrl);
    }
    if(!dataResp.ok){resultsFetched=false;return;}
    renderResults(await dataResp.json());
  }catch(e){
    resultsFetched=false;
  }
}

/* ── show widget UI ── */
function showResultsUI(){
  widgetActive=true;
  document.body.style.height="auto";document.body.style.overflow="visible";document.body.style.padding="0";
  widgetFrame.style.display="block";
  resultsTab.style.display="block";
  resultsTab.style.padding="0 12px 12px";
}

function showForecastUI(){
  widgetActive=true;
  document.body.style.height="auto";document.body.style.overflow="visible";document.body.style.padding="0";
  widgetFrame.style.display="block";
  forecastTab.style.display="block";
  /* Hide the standard results tab — they share the same Results button. */
  resultsTab.style.display="none";
}

/* ──────────────────────────────────────────────────────────────────
   Forecast card rendering — used only when forecastMeta is set.
   ────────────────────────────────────────────────────────────────── */

const FC_PERCENTILES=[10,25,50,75,90];

function fcNum(n){
  if(n===0||n===0.0)return "0";
  if(typeof n!=="number"||!isFinite(n))return String(n);
  const r=Number(n.toPrecision(3));
  const a=Math.abs(r);
  if(a>=1e6)return (Math.round(r/1e5)/10)+"M";
  if(a>=1e3)return (Math.round(r/100)/10)+"K";
  if(a<1&&a>0)return String(r);
  return String(r);
}

function fcIsNever(s){
  if(s==="never")return true;
  const m=/^(\\d{4})-/.exec(s||"");
  return !!m&&parseInt(m[1],10)>=2099;
}

function fcExtractPctl(row,field){
  if(!field)return null;
  const out={};
  for(const p of FC_PERCENTILES){
    const raw=row[field+"_p"+p];
    const n=typeof raw==="number"?raw:Number(raw);
    if(isNaN(n))return null;
    out["p"+p]=n;
  }
  return out;
}

function fcExtractDatePctl(row,field){
  if(!field)return null;
  const out={};
  for(const p of FC_PERCENTILES){
    const raw=row[field+"_p"+p];
    if(typeof raw!=="string")return null;
    if(raw!=="never"&&!/^\\d{4}-\\d{2}-\\d{2}$/.test(raw))return null;
    out["p"+p]=raw;
  }
  return out;
}

/* Categorical, thresholded and binary-decision forecasts all put one
   probability per option in a single `probabilities` column, JSON-encoded by
   the engine. Only the meaning of an option and its ordering differ. */
const FC_GROUPED_TYPES=new Set(["categorical","thresholded"]);
const FC_GROUPED_LABEL={categorical:"outcomes",thresholded:"thresholds"};
const FC_GROUPED_VISIBLE=4;

function fcIsDecision(){return forecastMeta.framing.kind==="decision";}
function fcIsConditional(){return forecastMeta.framing.kind==="conditional";}

/* Grouped: one row carries several labelled outcomes at once. Categorical and
   thresholded are grouped by their outcome type. A decision is grouped by its
   framing instead — one outcome per alternative, whatever that outcome is. */
function fcIsGrouped(){
  return FC_GROUPED_TYPES.has(forecastMeta.forecast_type)||fcIsDecision();
}

/* Ranged: the outcome is a distribution, reported as percentiles. */
function fcIsRanged(){
  return forecastMeta.forecast_type==="numeric"||forecastMeta.forecast_type==="date";
}

function fcGroupedLabel(){
  if(fcIsDecision())return "alternatives";
  return FC_GROUPED_LABEL[forecastMeta.forecast_type]||"outcomes";
}

/* Engine columns holding a JSON map arrive as strings; some paths hand back
   the decoded object, so accept both. */
function fcParseJsonMap(raw){
  if(raw&&typeof raw==="object"&&!Array.isArray(raw))return raw;
  if(typeof raw==="string"){
    try{
      const p=JSON.parse(raw);
      if(p&&typeof p==="object"&&!Array.isArray(p))return p;
    }catch{}
  }
  return null;
}

/* Categorical is ranked, so sort it most-likely-first. Thresholded keys run
   least to most strict and decision keys are the user's own ordering of the
   alternatives, so both keep engine order. */
function fcExtractProbabilities(row,kind){
  const obj=fcParseJsonMap(row.probabilities);
  if(!obj)return null;
  const entries=[];
  for(const[label,v]of Object.entries(obj)){
    const n=typeof v==="number"?v:Number(v);
    if(isFinite(n))entries.push({label,value:n});
  }
  if(!entries.length)return null;
  if(kind==="categorical")entries.sort((a,b)=>b.value-a.value);
  return entries;
}

/* Put entries in a caller-supplied label order, keeping any label the order
   doesn't mention at the end. */
function fcApplyOrder(entries,order){
  const rank=new Map(order.map((label,i)=>[label,i]));
  return entries.slice().sort((a,b)=>
    (rank.get(a.label)??order.length)-(rank.get(b.label)??order.length));
}

function fcRenderProbBars(entries,kind,full){
  const shown=full?entries:entries.slice(0,FC_GROUPED_VISIBLE);
  /* Only categorical has a single winning outcome to highlight; thresholded
     bars are monotonic by construction and decision bars need not be. Taken
     as the max rather than the first row, since a shared order can put a
     lower-probability outcome first. */
  const lead=kind==="categorical"?Math.max(...entries.map(e=>e.value)):null;
  let out=`<div class="fc-opts"><div class="fc-section-label">${esc(fcGroupedLabel())}</div>`;
  for(const e of shown){
    const pct=Math.max(0,Math.min(100,e.value));
    const isLead=lead!==null&&e.value===lead;
    out+=`<div class="fc-opt${isLead?" lead":""}">`+
      `<div class="fc-opt-head">`+
      `<span class="fc-opt-label" title="${escAttr(e.label)}">${esc(e.label)}</span>`+
      `<span class="fc-opt-value">${esc(Math.round(pct))}%</span>`+
      `</div>`+
      /* A 1% outcome would otherwise draw a sub-pixel sliver. */
      `<div class="fc-opt-bar"><div class="fc-opt-bar-fill" style="width:${pct}%${pct>0?";min-width:2px":""}"></div></div>`+
      `</div>`;
  }
  if(shown.length<entries.length)out+=`<div class="fc-opt-more">+${entries.length-shown.length} more</div>`;
  return out+`</div>`;
}

function fcPickDateGran(dates){
  const ms=dates.filter(d=>!fcIsNever(d)).map(d=>new Date(d+"T00:00:00").getTime());
  if(ms.length<2)return "year-month";
  const range=Math.max(...ms)-Math.min(...ms);
  return range>=1.5*365.25*86400000?"year-month":"month-day";
}

function fcFmtDate(iso,compact,gran){
  if(fcIsNever(iso))return "never";
  const d=new Date(iso+"T00:00:00");
  const mon=d.toLocaleString("en",{month:"short"});
  const day=d.getDate();
  const year=d.getFullYear();
  if(gran==="month-day")return compact?(mon+" "+day):(mon+" "+day+", "+year);
  return compact?(mon+" '"+String(year).slice(2)):(mon+" "+day+", "+year);
}

/* Render an SVG numeric percentile bar (p10-p90 with IQR box + median dot).
   Non-compact mode staggers labels above/below so p25/p75 don't collide
   with their neighbors when percentiles are bunched together.
   `domain` ({min,max}) puts several bars on one scale; without it the bar
   spans its own p10-p90. */
function fcRenderPctlSVG(p,units,compact,domain){
  const W=200,svgH=compact?24:36,pad=6,trackW=W-2*pad;
  const topPad=compact?0:14;
  const bottomPad=compact?14:18;
  const midY=topPad+svgH/2;
  const totalH=topPad+svgH+bottomPad;
  const aboveLabelY=topPad-4;
  const belowLabelY=topPad+svgH+(compact?10:14);
  const lo=domain?domain.min:p.p10;
  const hi=domain?domain.max:p.p90;
  const range=hi-lo;
  const scale=v=>range===0?trackW/2:((v-lo)/range)*trackW;
  const x={};
  for(const k of FC_PERCENTILES)x["p"+k]=pad+scale(p["p"+k]);
  const tickH=compact?6:10;
  const dotR=compact?3.5:5;
  /* Compact: p10/p50/p90 below. Non-compact: p10/p50/p90 below + p25/p75 above. */
  const ticks=compact?[[10,false],[50,false],[90,false]]:[[10,false],[25,true],[50,false],[75,true],[90,false]];
  let labels="";
  for(const[k,above]of ticks){
    const y=above?aboveLabelY:belowLabelY;
    labels+=`<text x="${x["p"+k]}" y="${y}" text-anchor="middle" fill="currentColor" class="fc-pctl-svg-label">${esc(fcNum(p["p"+k]))}</text>`;
  }
  /* Extra mini ticks at p25/p75 in non-compact so the eye can still find them. */
  const extraTicks=compact?"":`
    <line x1="${x.p25}" y1="${midY-tickH/2}" x2="${x.p25}" y2="${midY+tickH/2}" style="stroke:var(--iqr-stroke)" stroke-width="0.5"/>
    <line x1="${x.p75}" y1="${midY-tickH/2}" x2="${x.p75}" y2="${midY+tickH/2}" style="stroke:var(--iqr-stroke)" stroke-width="0.5"/>`;
  return `<svg viewBox="0 0 ${W} ${totalH}" preserveAspectRatio="xMidYMid meet" width="100%" height="${totalH}" style="color:var(--text-sec);display:block;overflow:visible">
    <line x1="${x.p10}" y1="${midY}" x2="${x.p90}" y2="${midY}" stroke="#71717a" stroke-width="${compact?1:1.5}"/>
    <rect x="${x.p25}" y="${midY-(compact?5:7)}" width="${Math.max(x.p75-x.p25,1)}" height="${compact?10:14}" rx="2" style="fill:var(--iqr-fill);stroke:var(--iqr-stroke)" stroke-width="0.5"/>
    <line x1="${x.p10}" y1="${midY-tickH/2}" x2="${x.p10}" y2="${midY+tickH/2}" stroke="#71717a" stroke-width="1"/>
    <line x1="${x.p90}" y1="${midY-tickH/2}" x2="${x.p90}" y2="${midY+tickH/2}" stroke="#71717a" stroke-width="1"/>
    ${extraTicks}
    <circle cx="${x.p50}" cy="${midY}" r="${dotR}" style="fill:var(--median-fill)"/>
    ${labels}
  </svg>`;
}

/* Render an SVG date percentile bar. Same staggered-label trick as
   fcRenderPctlSVG. Plus a dashed →never tail when any percentile is the
   sentinel.
   `domain` ({minMs,maxMs,hasTail,gran}) puts several bars on one scale, and
   reserves the tail on all of them when any one of them runs to "never". */
function fcRenderDatePctlSVG(dp,compact,domain){
  const W=200,svgH=compact?24:36,pad=6,trackW=W-2*pad;
  const topPad=compact?0:14;
  const bottomPad=compact?14:18;
  const midY=topPad+svgH/2;
  const totalH=topPad+svgH+bottomPad;
  const aboveLabelY=topPad-4;
  const belowLabelY=topPad+svgH+(compact?10:14);
  const tickH=compact?6:10;
  const dotR=compact?3.5:5;
  const dates=FC_PERCENTILES.map(p=>dp["p"+p]);
  const neverFlags=dates.map(fcIsNever);
  /* This bar's own sentinels decide whether the →never tail is DRAWN; the
     shared domain decides whether its space is RESERVED, so a branch that
     ends in a real date still maps onto the same pixels as one that doesn't. */
  const hasTail=neverFlags.some(Boolean);
  const reserveTail=domain?domain.hasTail:hasTail;
  const medianIsNever=neverFlags[2];
  const gran=domain?domain.gran:fcPickDateGran(dates);

  const tailFrac=reserveTail?0.22:0;
  const realTrackW=trackW*(1-tailFrac);
  const tailStartX=pad+realTrackW;
  const tailEndX=pad+trackW;

  const realMs=dates.map((d,i)=>neverFlags[i]?null:new Date(d+"T00:00:00").getTime()).filter(v=>v!==null);
  const msLeft=domain?domain.minMs:(realMs[0]??0);
  const msRight=domain?domain.maxMs:(realMs[realMs.length-1]??msLeft);
  const realRange=msRight-msLeft;
  const scale=v=>realRange===0?realTrackW/2:((v-msLeft)/realRange)*realTrackW;
  const positions=dates.map((d,i)=>neverFlags[i]?tailStartX:(pad+scale(new Date(d+"T00:00:00").getTime())));
  const x10=positions[0],x25=positions[1],x50=positions[2],x75=positions[3],x90=positions[4];
  const trackRightX=hasTail?tailStartX:x90;
  const iqrRightX=neverFlags[3]?trackRightX:x75;

  /* Compact: p10/p50/p90 below. Non-compact: same + p25/p75 above.
     Skip sentinel positions (the tail already says "never"). */
  const tickRows=compact?[[0,false],[2,false],[4,false]]:[[0,false],[1,true],[2,false],[3,true],[4,false]];
  let labels="";
  for(const[i,above]of tickRows){
    if(neverFlags[i])continue;
    const y=above?aboveLabelY:belowLabelY;
    labels+=`<text x="${positions[i]}" y="${y}" text-anchor="middle" fill="currentColor" class="fc-pctl-svg-label">${esc(fcFmtDate(dates[i],true,gran))}</text>`;
  }

  /* Extra mini ticks at p25/p75 in non-compact (only when real). */
  let extraTicks="";
  if(!compact){
    if(!neverFlags[1])extraTicks+=`<line x1="${x25}" y1="${midY-tickH/2}" x2="${x25}" y2="${midY+tickH/2}" style="stroke:var(--iqr-stroke)" stroke-width="0.5"/>`;
    if(!neverFlags[3])extraTicks+=`<line x1="${x75}" y1="${midY-tickH/2}" x2="${x75}" y2="${midY+tickH/2}" style="stroke:var(--iqr-stroke)" stroke-width="0.5"/>`;
  }

  let tailMarkup="";
  if(hasTail){
    tailMarkup=`
      <line x1="${tailStartX}" y1="${midY}" x2="${tailEndX-4}" y2="${midY}" stroke="#71717a" stroke-width="1" stroke-dasharray="2,2"/>
      <polygon points="${tailEndX},${midY} ${tailEndX-5},${midY-3} ${tailEndX-5},${midY+3}" fill="#71717a"/>
      <text x="${tailEndX}" y="${belowLabelY}" text-anchor="end" fill="#a1a1aa" class="fc-pctl-svg-label" style="font-style:italic">never</text>`;
  }

  const medianDot=medianIsNever?"":`<circle cx="${x50}" cy="${midY}" r="${dotR}" style="fill:var(--median-fill)"/>`;
  const rightWhisker=neverFlags[4]?"":`<line x1="${x90}" y1="${midY-tickH/2}" x2="${x90}" y2="${midY+tickH/2}" stroke="#71717a" stroke-width="1"/>`;

  return `<svg viewBox="0 0 ${W} ${totalH}" preserveAspectRatio="xMidYMid meet" width="100%" height="${totalH}" style="color:var(--text-sec);display:block;overflow:visible">
    <line x1="${x10}" y1="${midY}" x2="${trackRightX}" y2="${midY}" stroke="#71717a" stroke-width="${compact?1:1.5}"/>
    <rect x="${x25}" y="${midY-(compact?5:7)}" width="${Math.max(iqrRightX-x25,1)}" height="${compact?10:14}" rx="2" style="fill:var(--iqr-fill);stroke:var(--iqr-stroke)" stroke-width="0.5"/>
    <line x1="${x10}" y1="${midY-tickH/2}" x2="${x10}" y2="${midY+tickH/2}" stroke="#71717a" stroke-width="1"/>
    ${rightWhisker}
    ${extraTicks}
    ${tailMarkup}
    ${medianDot}
    ${labels}
  </svg>`;
}

/* The window a set of percentile records has to share for their bars to be
   comparable — spanning the lowest p10 to the highest p90 across all of them.
   Used both for the two branches of a conditional forecast and for the
   alternatives of a decision. */
function fcNumericDomain(records){
  if(!records.length)return null;
  return {min:Math.min(...records.map(r=>r.p10)),max:Math.max(...records.map(r=>r.p90))};
}

function fcDateDomain(records){
  if(!records.length)return null;
  const dates=records.flatMap(r=>FC_PERCENTILES.map(k=>r["p"+k]));
  const realMs=dates.filter(d=>!fcIsNever(d)).map(d=>new Date(d+"T00:00:00").getTime());
  if(!realMs.length)return null;
  return {
    minMs:Math.min(...realMs),
    maxMs:Math.max(...realMs),
    hasTail:dates.some(fcIsNever),
    gran:fcPickDateGran(dates),
  };
}

/* A numeric/date decision puts one percentile record per alternative in a
   single `percentiles` column: {alternative: {p10..p90}}, numbers for a
   numeric outcome and YYYY-MM-DD / "never" strings for a date one. Rejects a
   map whose records don't all carry the five keys of one single kind. */
function fcExtractDecisionPctl(row,kind){
  const obj=fcParseJsonMap(row.percentiles);
  if(!obj)return null;
  const ok=kind==="numeric"
    ?v=>typeof v==="number"&&isFinite(v)
    :v=>v==="never"||(typeof v==="string"&&/^\\d{4}-\\d{2}-\\d{2}$/.test(v));
  const entries=[];
  for(const[label,record]of Object.entries(obj)){
    if(!record||typeof record!=="object"||Array.isArray(record))return null;
    if(!FC_PERCENTILES.every(p=>ok(record["p"+p])))return null;
    entries.push({label,record});
  }
  return entries.length?entries:null;
}

/* One range bar per decision alternative, all on a shared axis: the spread
   across the alternatives is the decision's effect, so bars that each filled
   their own track would hide the very thing being asked. Same section-label /
   capped-rows / "+N more" shape as the probability bars. */
function fcRenderDecisionPctlBars(kind,entries,full){
  const records=entries.map(e=>e.record);
  const domain=kind==="numeric"?fcNumericDomain(records):fcDateDomain(records);
  const shown=full?entries:entries.slice(0,FC_GROUPED_VISIBLE);
  let out=`<div class="fc-opts"><div class="fc-section-label">${esc(fcGroupedLabel())}</div>`;
  for(const{label,record}of shown){
    out+=`<div class="fc-alt"><div class="fc-pctl-header"><span class="fc-pctl-field">${esc(label)}</span>`;
    if(kind==="numeric"){
      out+=`<span class="fc-pctl-median">${esc(fcNum(record.p50))}</span>`;
      if(forecastMeta.units)out+=`<span class="fc-pctl-units">${esc(forecastMeta.units)}</span>`;
      out+=`</div>`+fcRenderPctlSVG(record,forecastMeta.units,!full,domain);
    }else{
      const gran=domain?domain.gran:fcPickDateGran(FC_PERCENTILES.map(p=>record["p"+p]));
      const isNever=fcIsNever(record.p50);
      out+=`<span class="fc-pctl-median"${isNever?' style="color:var(--text-dim)"':''}>${esc(fcFmtDate(record.p50,false,gran))}</span>`;
      out+=`</div>`+fcRenderDatePctlSVG(record,!full,domain);
    }
    out+=`</div>`;
  }
  if(shown.length<entries.length)out+=`<div class="fc-opt-more">+${entries.length-shown.length} more</div>`;
  return out+`</div>`;
}

/* Extract source domains from inline markdown links AND _source_bank
   (whichever is present). Citations get resolved server-side to
   `[title](url)` markdown so the link pattern catches them. */
function fcExtractDomain(url){
  try{const u=new URL(url.startsWith("http")?url:"https://"+url);return u.hostname.replace(/^www\\./,"");}
  catch{return url.slice(0,30);}
}

function fcExtractSources(row){
  const seen=new Set();
  const out=[];
  /* 1. Use _source_bank if present — richer and already deduped. */
  const sb=row._source_bank;
  let parsed=null;
  if(sb&&typeof sb==="object"&&!Array.isArray(sb))parsed=sb;
  else if(typeof sb==="string"){try{parsed=JSON.parse(sb);}catch{}}
  if(parsed){
    for(const v of Object.values(parsed)){
      if(!v||typeof v!=="object")continue;
      const u=v.url;
      if(typeof u!=="string"||!u)continue;
      const dom=fcExtractDomain(u);
      if(seen.has(dom))continue;
      seen.add(dom);
      out.push({url:u,domain:dom});
    }
    if(out.length)return out;
  }
  /* 2. Fallback: scan string fields for markdown links + bare URLs. */
  const mdRe=/\\[[^\\]]+\\]\\((https?:\\/\\/[^)]+)\\)/g;
  const urlRe=/(https?:\\/\\/[^\\s<>"')]+)/g;
  for(const v of Object.values(row)){
    if(typeof v!=="string")continue;
    let m;
    while((m=mdRe.exec(v))!==null){
      const dom=fcExtractDomain(m[1]);
      if(!seen.has(dom)){seen.add(dom);out.push({url:m[1],domain:dom});}
    }
    while((m=urlRe.exec(v))!==null){
      const dom=fcExtractDomain(m[1]);
      if(!seen.has(dom)){seen.add(dom);out.push({url:m[1],domain:dom});}
    }
  }
  return out;
}

/* Pick the row's title — typically `question` for forecast inputs. */
function fcRowTitle(row){
  const cands=["question","entity","name","item","subject","title"];
  for(const k of cands){
    const v=row[k];
    if(typeof v==="string"&&v.trim())return v.trim();
  }
  /* Fallback: first non-empty string that isn't rationale/probability. */
  const skip=new Set(["rationale","probability","probabilities","percentiles","units","_status","_error","_row_index","_completed_at","_source_bank","research"]);
  for(const[k,v]of Object.entries(row)){
    if(skip.has(k)||k.startsWith("_"))continue;
    if(typeof v==="string"&&v.trim()&&v.length<200)return v.trim();
  }
  return null;
}

/* First "section" of the rationale — first paragraph or first ~600 chars. */
function fcFirstSection(text){
  if(typeof text!=="string")return "";
  const trimmed=text.trim();
  if(!trimmed)return "";
  /* Prefer paragraph break. */
  const pIdx=trimmed.indexOf("\\n\\n");
  if(pIdx>0&&pIdx<800)return trimmed.slice(0,pIdx).trim();
  return trimmed.length>600?trimmed.slice(0,600).trim()+"…":trimmed;
}

function fcLinkifyRationale(text){
  return linkify(text);
}

/* The outcome type's viz for one data object. `full` is the expanded/solo
   card, which adds the percentile grid and drops the option-bar cap. Columns
   drawn are recorded in `used` so the card doesn't repeat them as raw text.
   Returns "" when the expected columns aren't there, which leaves them to the
   text fallback. */
function fcRenderOutcome(data,full,used,domain){
  let out="";
  /* A decision's alternatives carry a range each when its outcome is numeric
     or date, and a probability each when it is binary. Both are grouped, and
     both are tested before binary: a decision's own outcome type is usually
     binary, and it is the framing that decides how the row is drawn. */
  if(fcIsDecision()&&fcIsRanged()){
    const entries=fcExtractDecisionPctl(data,forecastMeta.forecast_type);
    if(entries){
      out+=fcRenderDecisionPctlBars(forecastMeta.forecast_type,entries,full);
      used.add("percentiles");
    }
  }else if(fcIsGrouped()){
    const entries=fcExtractProbabilities(data,forecastMeta.forecast_type);
    if(entries){
      const ordered=domain&&domain.order?fcApplyOrder(entries,domain.order):entries;
      out+=fcRenderProbBars(ordered,forecastMeta.forecast_type,full);
      used.add("probabilities");
    }
  }else if(forecastMeta.forecast_type==="binary"){
    const probRaw=data.probability;
    const prob=typeof probRaw==="number"?probRaw:Number(probRaw);
    if(!isNaN(prob)){
      const pct=Math.max(0,Math.min(100,prob));
      out+=`<div class="fc-prob"><span class="fc-prob-value">${esc(Math.round(pct))}%</span><span class="fc-prob-label">probability</span></div>`;
      out+=`<div class="fc-prob-bar"><div class="fc-prob-bar-fill" style="width:${pct}%${pct>0?";min-width:2px":""}"></div></div>`;
      used.add("probability");
    }
  }else if(forecastMeta.forecast_type==="numeric"&&forecastMeta.output_field){
    const p=fcExtractPctl(data,forecastMeta.output_field);
    if(p){
      out+=`<div class="fc-pctl-bar">`;
      out+=`<div class="fc-pctl-header"><span class="fc-pctl-field">${esc(forecastMeta.output_field)}</span><span class="fc-pctl-median">${esc(fcNum(p.p50))}</span>`;
      if(forecastMeta.units)out+=`<span class="fc-pctl-units">${esc(forecastMeta.units)}</span>`;
      out+=`</div>`;
      out+=fcRenderPctlSVG(p,forecastMeta.units,!full,domain);
      if(full){
        out+=`<div class="fc-pctl-grid">`;
        for(const k of FC_PERCENTILES){
          const isMed=k===50;
          out+=`<div class="fc-pctl-cell"><div class="fc-pctl-cell-label">p${k}</div><div class="fc-pctl-cell-val${isMed?" median":""}">${esc(fcNum(p["p"+k]))}</div></div>`;
        }
        out+=`</div>`;
      }
      out+=`</div>`;
      for(const k of FC_PERCENTILES)used.add(forecastMeta.output_field+"_p"+k);
    }
  }else if(forecastMeta.forecast_type==="date"&&forecastMeta.output_field){
    const dp=fcExtractDatePctl(data,forecastMeta.output_field);
    if(dp){
      const gran=domain?domain.gran:fcPickDateGran([dp.p10,dp.p25,dp.p50,dp.p75,dp.p90]);
      const isNever=fcIsNever(dp.p50);
      out+=`<div class="fc-pctl-bar">`;
      out+=`<div class="fc-pctl-header"><span class="fc-pctl-field">${esc(forecastMeta.output_field)}</span><span class="fc-pctl-median"${isNever?' style="color:var(--text-dim)"':''}>${esc(fcFmtDate(dp.p50,false,gran))}</span></div>`;
      out+=fcRenderDatePctlSVG(dp,!full,domain);
      if(full){
        out+=`<div class="fc-pctl-grid">`;
        for(const k of FC_PERCENTILES){
          const isMed=k===50;
          const label=isMed?"median":(k+"% by");
          out+=`<div class="fc-pctl-cell"><div class="fc-pctl-cell-label">${esc(label)}</div><div class="fc-pctl-cell-val${isMed?" median":""}">${esc(fcFmtDate(dp["p"+k],true,gran))}</div></div>`;
        }
        out+=`</div>`;
      }
      out+=`</div>`;
      for(const k of FC_PERCENTILES)used.add(forecastMeta.output_field+"_p"+k);
    }
  }
  return out;
}

/* A conditional forecast forecasts the outcome twice — once in the world where
   the condition holds, once where it doesn't — in columns suffixed with the
   branch, sharing one rationale. */
const FC_BRANCHES=[["given_condition","Condition holds"],["given_not_condition","Condition does NOT hold"]];

function fcBranchData(data,branch){
  const suffix="_"+branch;
  const out={};
  for(const[k,v]of Object.entries(data)){
    if(k.endsWith(suffix))out[k.slice(0,-suffix.length)]=v;
  }
  return out;
}

/* The condition the two branches split on: shared across the batch, or named
   per row by an input column. */
function fcConditionText(data){
  if(forecastMeta.framing.condition)return forecastMeta.framing.condition;
  const field=forecastMeta.framing.condition_field;
  if(field){
    const v=data[field];
    if(typeof v==="string"&&v.trim())return v.trim();
    if(typeof v==="number")return String(v);
  }
  return null;
}

/* The frame sibling cards must share to be comparable: one scale for the
   continuous outcomes, one row order for the grouped ones. Bars drawn on
   their own scales look alike however far apart they are, and the same
   outcome on a different row in each card hides the comparison just as well. */
function fcSharedDomain(datas){
  if(fcIsGrouped()){
    /* Rank by the total across siblings, so neither branch's ordering wins.
       Labels keep first-seen order otherwise, which is the engine's
       meaningful order for thresholded and decision. */
    const totals=new Map();
    for(const d of datas){
      const entries=fcExtractProbabilities(d,forecastMeta.forecast_type);
      if(!entries)continue;
      for(const e of entries)totals.set(e.label,(totals.get(e.label)||0)+e.value);
    }
    if(!totals.size)return null;
    const order=[...totals.keys()];
    if(forecastMeta.forecast_type==="categorical"){
      order.sort((a,b)=>totals.get(b)-totals.get(a));
    }
    return {order};
  }
  const field=forecastMeta.output_field;
  if(!field)return null;
  if(forecastMeta.forecast_type==="numeric"){
    return fcNumericDomain(datas.map(d=>fcExtractPctl(d,field)).filter(Boolean));
  }
  if(forecastMeta.forecast_type==="date"){
    return fcDateDomain(datas.map(d=>fcExtractDatePctl(d,field)).filter(Boolean));
  }
  return null;
}

/* Draw the outcome card once per branch, under a banner naming the condition
   so a conditional forecast can't be mistaken for a plain one. */
function fcRenderBranches(data,full,used){
  const branches=FC_BRANCHES.map(([branch,label])=>({branch,label,bdata:fcBranchData(data,branch)}));
  const domain=fcSharedDomain(branches.map(b=>b.bdata));
  let out="";
  for(const{branch,label,bdata}of branches){
    if(!Object.keys(bdata).length)continue;
    const drawn=new Set();
    const inner=fcRenderOutcome(bdata,full,drawn,domain);
    if(!inner)continue;
    for(const k of drawn)used.add(k+"_"+branch);
    out+=`<div class="fc-branch"><div class="fc-branch-label">${esc(label)}</div>${inner}</div>`;
  }
  if(!out)return "";
  const condition=fcConditionText(data);
  const banner=condition
    ?`<div class="fc-condition"><div class="fc-section-label">condition</div><div class="fc-condition-text">${esc(condition)}</div></div>`
    :"";
  /* Only suppress the input column when the banner is what's showing it. */
  if(condition&&!forecastMeta.framing.condition&&forecastMeta.framing.condition_field){
    used.add(forecastMeta.framing.condition_field);
  }
  return banner+out;
}

function fcBuildCard(row,idx,isSolo,isExpanded){
  const data=row.display||row;
  const status=data._status||"";
  const isFailed=status==="failed";
  const title=fcRowTitle(data);
  const rowBadge=data._row_index!=null?("#"+(data._row_index+1)):("#"+(idx+1));
  const sources=fcExtractSources(data);

  let body="";
  let displayedFieldKeys=new Set(["rationale","units"]);
  const full=isExpanded||isSolo;

  if(isFailed){
    const err=data._error||"This row failed.";
    body+=`<p class="fc-error" style="color:var(--seg-fail);font-size:11px;margin:0">${esc(err)}</p>`;
  }else if(fcIsConditional()){
    body+=fcRenderBranches(data,full,displayedFieldKeys);
  }else{
    body+=fcRenderOutcome(data,full,displayedFieldKeys,forecastDomain);
  }

  const rationale=typeof data.rationale==="string"?data.rationale:"";
  if(rationale){
    const text=full?rationale:fcFirstSection(rationale);
    body+=`<div class="fc-field-row${full?"":" clamped"}"><span class="fc-field-key">rationale:</span>${fcLinkifyRationale(text)}</div>`;
  }

  /* Any other scalar output field the row carries. */
  if(full){
    const extras=[];
    const skip=new Set(["_status","_error","_row_index","_completed_at","_source_bank","_expand_index","research","provenance_and_notes"]);
    for(const[k,v]of Object.entries(data)){
      if(skip.has(k)||k.startsWith("_"))continue;
      if(displayedFieldKeys.has(k))continue;
      if(k==="question"||k==="entity"||k==="name"||k==="item"||k==="subject"||k==="title")continue;
      if(v===null||v===undefined||v==="")continue;
      if(typeof v==="object")continue;
      extras.push([k,Array.isArray(v)?v.join(", "):String(v)]);
    }
    if(extras.length){
      for(const[k,v]of extras){
        body+=`<div class="fc-field-row"><span class="fc-field-key">${esc(k)}:</span>${esc(v)}</div>`;
      }
    }
  }

  /* Sources */
  const maxSources=full?20:5;
  if(sources.length){
    body+=`<div class="fc-sources">`;
    for(const s of sources.slice(0,maxSources)){
      body+=`<a class="fc-source-pill" href="${escAttr(s.url)}" target="_blank" rel="noopener noreferrer">${esc(s.domain)}</a>`;
    }
    if(sources.length>maxSources)body+=`<span class="fc-source-more">+${sources.length-maxSources} more</span>`;
    body+=`</div>`;
  }

  const headerHTML=`<div class="fc-card-header">`+
    (title?`<div class="fc-card-title">${esc(title)}</div>`:`<div class="fc-card-title">Forecast ${esc(rowBadge)}</div>`)+
    (isFailed?`<span class="fc-fail-tag">failed</span>`:"")+
    `<span class="fc-row-badge">${esc(rowBadge)}</span>`+
    `</div>`;

  const expandHint=(isSolo||isExpanded||isFailed)?"":`<div class="fc-expand-hint">click to expand</div>`;
  const cls="fc-card"+(isSolo?" solo":"")+(isExpanded?" expanded":"")+(isFailed?" failed":"");
  const clickable=(!isSolo&&!isFailed)?'data-clickable="1"':"";
  return `<div class="${cls}" data-idx="${idx}" ${clickable}>${headerHTML}${body}${expandHint}</div>`;
}

function renderForecastCards(rows){
  forecastRows=rows||[];
  const isSolo=forecastRows.length===1;
  fcGrid.classList.toggle("solo",isSolo);
  let html="";
  if(!forecastRows.length){
    fcGrid.innerHTML=`<div class="fc-empty">No forecast results.</div>`;
    fcSum.textContent="No results";
    return;
  }
  forecastDomain=fcSharedDomain(forecastRows.map(r=>r.display||r));
  for(let i=0;i<forecastRows.length;i++){
    html+=fcBuildCard(forecastRows[i],i,isSolo,false);
  }
  fcGrid.innerHTML=html;
  fcSum.textContent=forecastRows.length+" forecast"+(forecastRows.length>1?"s":"");
  fcExportLink.onclick=exportResults;
}

/* Click handler: expand/collapse a card. Solo cards are not clickable. */
fcGrid.addEventListener("click",e=>{
  const card=e.target.closest(".fc-card");
  if(!card||card.classList.contains("solo"))return;
  /* Don't toggle on link clicks (source pills, rationale citations). */
  if(e.target.closest("a"))return;
  const idx=parseInt(card.dataset.idx,10);
  if(isNaN(idx)||!forecastRows[idx])return;
  const wasExpanded=card.classList.contains("expanded");
  /* Collapse all others first so the grid doesn't fragment. */
  fcGrid.querySelectorAll(".fc-card.expanded").forEach(c=>{
    const i=parseInt(c.dataset.idx,10);
    if(!isNaN(i)){
      c.outerHTML=fcBuildCard(forecastRows[i],i,false,false);
    }
  });
  if(!wasExpanded){
    const fresh=fcGrid.querySelector(`.fc-card[data-idx="${idx}"]`);
    if(fresh)fresh.outerHTML=fcBuildCard(forecastRows[idx],idx,false,true);
  }
});

function enterProgressMode(d){
  progressMode=true;
  document.body.style.height="auto";document.body.style.overflow="visible";document.body.style.padding="0";
  widgetFrame.style.display="block";
  progressSection.style.display="block";
  tabBar.style.display="flex";
  activityTab.style.display="block";
  resultsTab.style.display="none";
  forecastTab.style.display="none";
  if(d.task_id)currentTaskId=d.task_id;
  /* Extract task_id from progress_url as fallback */
  if(!currentTaskId&&d.progress_url){const m=d.progress_url.match(/progress\\/([0-9a-f-]+)/);if(m)currentTaskId=m[1];}
  if(d.poll_token)pollToken=d.poll_token;
  if(d.download_url)downloadUrl=d.download_url;
  /* Forecast operations get a card grid instead of the table. */
  if(d.task_type==="forecast"&&d.forecast_type)forecastMeta=specFromWidgetMeta(d);
  renderProgress(d);
}

/* Submission metadata reaches the widget in the shape the API used before a
   forecast's framing was separated from its outcome type: a decision occupied
   the outcome type, with the real one carried alongside it. A host that won't
   proxy tool calls has no other source for any of this, so translating it here
   is a supported path, not a migration step. */
function specFromWidgetMeta(d){
  if(d.forecast_type==="decision"){
    return {
      /* Decisions were binary-only before the outcome type rode along. */
      forecast_type:d.decision_outcome_type||"binary",
      output_field:d.output_field||null,
      units:d.units||null,
      framing:{
        kind:"decision",
        alternatives_field:d.alternatives_field||null,
        intervention:null,
      },
    };
  }
  return {
    forecast_type:d.forecast_type,
    output_field:d.output_field||null,
    units:d.units||null,
    categories_field:d.categories_field||null,
    thresholds_field:d.thresholds_field||null,
    framing:d.is_conditional
      ?{kind:"conditional",condition:d.condition||null,condition_field:d.condition_field||null}
      :{kind:"unconditional"},
  };
}

/* Everything downstream reads framing.kind, so a spec without one cannot be
   drawn. Refusing it here keeps that assumption true in a single place, and
   leaves the submission metadata in place, which is at least coherent. */
function useForecastSpec(spec){
  /* Absent is legitimate: the task's outcome type isn't one the API publishes. */
  if(!spec)return;
  if(!spec.framing||!spec.framing.kind){
    console.warn("[fs] forecast spec has no framing; keeping submission metadata",spec);
    return;
  }
  forecastMeta=spec;
}

/* ── task data ──
   Two transports for the same payload. A tool call over the host bridge rides
   the connector's own authentication, so it still answers for a task whose
   REST poll token expired hours or months ago. REST is what a host that won't
   proxy tool calls is left with, and it stops answering 24h after submission.

   Both settle to {data} when they got it, {permanent,note} when nothing is
   going to bring the task back, or {} when a retry might still work. */

function hostProxiesTools(){
  const caps=app.getHostCapabilities?.();
  return !!(caps&&caps.serverTools);
}

async function fetchViaTool(cursor){
  try{
    /* Each tool takes a single pydantic model, so its schema nests every
       argument under "params". */
    const res=await app.callServerTool({
      name:"futuresearch_task_data",
      arguments:{params:{task_id:currentTaskId,cursor:cursor||null}},
    });
    if(res&&res.structuredContent)return {data:res.structuredContent};
  }catch(e){}
  return {};
}

async function fetchViaRest(cursor){
  if(!pollUrl)return {};
  let url=pollUrl;
  if(cursor)url+=(url.includes("?")?"&":"?")+"cursor="+encodeURIComponent(cursor);
  const opts=pollToken?{headers:{"Authorization":"Bearer "+pollToken}}:{};
  try{
    const r=await fetch(url,opts);
    if(r.ok)return {data:await r.json()};
    let code="";
    try{code=((await r.clone().json())||{}).code||"";}catch{}
    if(r.status===401||code==="session_expired")
      return {permanent:true,note:"Session expired — ask Claude for this task's status again to reconnect."};
    if(r.status===404)
      return {permanent:true,note:"This task is no longer available."};
  }catch(e){}
  return {};
}

/* A host that advertises the bridge can still fail an individual call, and
   REST may be in range, so a tool miss falls through rather than giving up. */
async function fetchTaskData(cursor){
  if(currentTaskId&&hostProxiesTools()){
    const viaTool=await fetchViaTool(cursor);
    if(viaTool.data)return viaTool;
  }
  return await fetchViaRest(cursor);
}

/* ── polling ── */
function stopPoll(){if(pollTimer){clearInterval(pollTimer);pollTimer=null;}}

/* Show a note without destroying the last rendered progress. */
function setPollNote(msg){
  if(!msg){pollNote.style.display="none";pollNote.textContent="";return;}
  pollNote.textContent=msg;
  pollNote.style.display="block";
}

function startPoll(){
  pollTimer=setInterval(async()=>{
    const r=await fetchTaskData(pollCursor);
    if(r.data){setPollNote("");renderProgress(r.data);return;}
    if(r.permanent){stopPoll();setPollNote(r.note);return;}
    setPollNote("Reconnecting…");
  },10000);
}

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
  if(obj.research!=null&&typeof obj.research==="object"&&!Array.isArray(obj.research)){
    for(const[k,v]of Object.entries(obj.research)){
      if(v!=null)research[k]=typeof v==="string"?v:String(v);
    }
  }
  const display=flat(obj);
  delete display.research;
  return{display,research};
}

function processData(data){
  if(!Array.isArray(data))data=[data];
  if(!data.length){sum.textContent="No results";tbl.innerHTML="";return;}
  S.rows=data.map(r=>flatWithResearch(r));
  const colSet=new Set();
  S.rows.forEach(r=>{for(const k of Object.keys(r.display))colSet.add(k)});
  const all=[...colSet];
  const visible=all.filter(k=>k!=="research"&&!k.startsWith("research."));
  S.allCols=[...visible.filter(k=>!k.includes(".")),...visible.filter(k=>k.includes("."))];
  S.sortCol=null;S.sortDir=0;S.filters={};S.globalQuery="";globalSearchEl.value="";S.selected.clear();S.lastClick=null;
  S.filteredIdx=S.rows.map((_,i)=>i);
  renderTable();
}

/* --- filter & sort --- */
function applyFilterAndSort(){
  let idx=S.rows.map((_,i)=>i);
  if(S.globalQuery){
    const gq=S.globalQuery.toLowerCase();
    idx=idx.filter(i=>{const row=S.rows[i].display;return Object.values(row).some(v=>v!=null&&String(v).toLowerCase().includes(gq));});
  }
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
const globalSearchEl=document.getElementById("globalSearch");
globalSearchEl.addEventListener("input",()=>{S.globalQuery=globalSearchEl.value;clearTimeout(filterTimer);filterTimer=setTimeout(()=>applyFilterAndSort(),150);});

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

  /* clear focusedCell if its row is no longer visible */
  if(S.focusedCell){const fs=new Set(S.filteredIdx);if(!fs.has(S.focusedCell.idx))S.focusedCell=null;}

  let h='<thead><tr class="hdr-row"><th class="row-num">#</th>';
  for(const c of cols){
    let arrow='<span class="sort-arrow">&#9650;</span>';
    if(S.sortCol===c)arrow=S.sortDir===1?'<span class="sort-arrow active">&#9650;</span>':'<span class="sort-arrow active">&#9660;</span>';
    h+='<th data-col="'+escAttr(c)+'" style="position:relative">'+esc(c)+arrow+'<div class="col-resize-handle"></div></th>';
  }
  h+='</tr><tr class="flt-row"><th class="row-num"></th>';
  for(const c of cols){
    h+='<th><input data-col="'+escAttr(c)+'" placeholder="Filter..." value="'+escAttr(S.filters[c]||"")+'"></th>';
  }
  h+='</tr></thead><tbody>';
  let rowNum=0;
  for(const i of S.filteredIdx){
    rowNum++;
    const row=S.rows[i],sel=S.selected.has(i)?' class="selected"':"";
    h+='<tr data-idx="'+i+'"'+sel+'><td class="row-num">'+rowNum+'</td>';
    for(const c of cols){
      const hasR=getResearch(row,c)!=null;
      const focused=S.focusedCell&&S.focusedCell.idx===i&&S.focusedCell.col===c;
      const v=row.display[c];
      let cls=hasR?(focused?' class="has-research cell-focused"':' class="has-research"'):(focused?' class="cell-focused"':"");
      const dc=' data-col="'+escAttr(c)+'"';
      if(v==null){h+="<td"+cls+dc+"></td>";}
      else{const s=String(v);
        if(s.length>TRUNC)h+='<td'+cls+dc+'><span class="cell-text">'+linkify(truncSafe(s,TRUNC))+'</span><span class="cell-more">&hellip; more</span></td>';
        else h+='<td'+cls+dc+'>'+linkify(s)+'</td>';
      }
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
  if(didDrag){didDrag=false;return;}
  if(e.target.closest(".col-resize-handle"))return;
  const th=e.target.closest(".hdr-row th");
  if(!th)return;
  const col=th.dataset.col;if(!col)return;
  if(S.sortCol===col){S.sortDir=S.sortDir===1?-1:S.sortDir===-1?0:1;if(S.sortDir===0)S.sortCol=null;}
  else{S.sortCol=col;S.sortDir=1;}
  applyFilterAndSort();
});

/* --- cell expand/collapse --- */
tbl.addEventListener("click",e=>{
  const more=e.target.closest(".cell-more");
  if(more){
    e.stopPropagation();
    const td=more.closest("td"),tr=td.closest("tr");
    const idx=parseInt(tr.dataset.idx,10),col=td.dataset.col;
    const full=String(S.rows[idx].display[col]);
    td.querySelector(".cell-text").innerHTML=linkify(full);
    more.textContent="less";more.className="cell-less";
    return;
  }
  const less=e.target.closest(".cell-less");
  if(less){
    e.stopPropagation();
    const td=less.closest("td"),tr=td.closest("tr");
    const idx=parseInt(tr.dataset.idx,10),col=td.dataset.col;
    const full=String(S.rows[idx].display[col]);
    td.querySelector(".cell-text").innerHTML=linkify(truncSafe(full,TRUNC));
    less.textContent="\\u2026 more";less.className="cell-more";
    return;
  }
});

/* --- selection (# column click toggles, shift extends range) --- */
tbl.addEventListener("click",e=>{
  if(e.target.closest(".hdr-row")||e.target.closest(".flt-row"))return;
  const td=e.target.closest("td");if(!td)return;
  const tr=td.closest("tbody tr");if(!tr)return;
  const idx=parseInt(tr.dataset.idx,10);if(isNaN(idx))return;

  if(td.classList.contains("row-num")){
    if(S.focusedCell){S.focusedCell=null;tbl.querySelectorAll("td.cell-focused").forEach(c=>c.classList.remove("cell-focused"));}
    if(e.shiftKey&&S.lastClick!=null){
      const posA=S.filteredIdx.indexOf(S.lastClick),posB=S.filteredIdx.indexOf(idx);
      if(posA>=0&&posB>=0){const lo=Math.min(posA,posB),hi=Math.max(posA,posB);for(let p=lo;p<=hi;p++)S.selected.add(S.filteredIdx[p]);}
    }else{
      if(S.selected.has(idx))S.selected.delete(idx);else S.selected.add(idx);
    }
    S.lastClick=idx;updateSelection();updateCopyBtn();
    return;
  }

  if(e.target.closest("a")||e.target.closest(".cell-more")||e.target.closest(".cell-less"))return;
  const col=td.dataset.col;if(!col)return;
  const prev=S.focusedCell;
  if(prev){const oldTd=tbl.querySelector('tbody tr[data-idx="'+prev.idx+'"] td[data-col="'+CSS.escape(prev.col)+'"]');if(oldTd)oldTd.classList.remove("cell-focused");}
  if(prev&&prev.idx===idx&&prev.col===col){S.focusedCell=null;}
  else{S.focusedCell={idx,col};td.classList.add("cell-focused");}
});

/* --- double-click data cell to copy value --- */
tbl.addEventListener("dblclick",e=>{
  if(e.target.closest(".col-resize-handle"))return;
  if(e.target.closest(".hdr-row")||e.target.closest(".flt-row"))return;
  const td=e.target.closest("tbody td");if(!td||td.classList.contains("row-num"))return;
  const tr=td.closest("tr");if(!tr)return;
  const idx=parseInt(tr.dataset.idx,10),col=td.dataset.col;
  if(isNaN(idx)||!col)return;
  const v=S.rows[idx]?.display[col];if(v==null)return;
  copyToClipboard(String(v)).then(ok=>{if(ok)showToast("Cell copied");});
});

function updateSelection(){
  tbl.querySelectorAll("tbody tr").forEach(tr=>{
    const idx=parseInt(tr.dataset.idx,10);tr.classList.toggle("selected",S.selected.has(idx));
  });
}
function updateCopyBtn(){const n=S.selected.size;const fl=copyFmt.toUpperCase();copyBtn.textContent=n>0?"Copy "+fl+" ("+n+")":"Copy "+fl;copyBtn.disabled=n===0;}

/* --- select all --- */
selAllBtn.addEventListener("click",()=>{
  if(S.selected.size===S.filteredIdx.length){S.selected.clear();showToast("Selection cleared");}
  else{S.selected.clear();S.filteredIdx.forEach(i=>S.selected.add(i));showToast("Selected all "+S.filteredIdx.length+" rows");}
  updateSelection();updateCopyBtn();
});

/* --- copy --- */
function buildCopyText(){
  const cols=S.allCols;
  const sel=S.filteredIdx.filter(i=>S.selected.has(i));
  if(copyFmt==="json"){
    const data=sel.map(i=>{const o={};for(const c of cols)o[c]=S.rows[i].display[c]??null;return o;});
    return JSON.stringify(data,null,2);
  }
  const isCSV=copyFmt==="csv";
  const sep=isCSV?",":"\\t";
  const q=v=>isCSV?'"'+v.replace(/"/g,'""')+'"':v.replace(/\\t/g," ");
  const lines=[cols.map(c=>q(c)).join(sep)];
  for(const i of sel){
    lines.push(cols.map(c=>{const v=S.rows[i].display[c];return v==null?(isCSV?'""':""):q(String(v));}).join(sep));
  }
  return lines.join("\\n");
}
function execCopy(text){
  const ta=document.createElement("textarea");
  ta.value=text;ta.style.cssText="position:fixed;left:-9999px";
  document.body.appendChild(ta);ta.select();
  let ok=false;try{ok=document.execCommand("copy")}catch{}
  document.body.removeChild(ta);return ok;
}
function showCopyModal(text){
  copyArea.value=text;copyModal.classList.add("show");
  copyArea.focus();copyArea.select();
}
copyBtn.addEventListener("click",async()=>{
  if(!S.selected.size)return;
  const text=buildCopyText();
  const msg="Copied "+S.selected.size+" row"+(S.selected.size>1?"s":"")+" as "+copyFmt.toUpperCase();
  /* Clipboard API often fails in sandboxed iframes — try it first,
     fall back to execCommand, then show modal for manual copy. */
  try{await navigator.clipboard.writeText(text);showToast(msg);return;}catch{}
  try{if(execCopy(text)){showToast(msg);return;}}catch{}
  showCopyModal(text);
});
closeCopyModal.addEventListener("click",()=>copyModal.classList.remove("show"));
copyModal.addEventListener("click",e=>{if(e.target===copyModal)copyModal.classList.remove("show");});
function showToast(msg){toast.textContent=msg;toast.classList.add("show");setTimeout(()=>toast.classList.remove("show"),2000);}


/* --- popover --- */
let popTimer=null,popTarget=null,popVisible=false;

function showPopover(td){
  const tr=td.closest("tr");const idx=parseInt(tr.dataset.idx,10);const col=td.dataset.col;
  const row=S.rows[idx];if(!row)return;
  const text=getResearch(row,col);if(text==null)return;
  popHdr.textContent="research."+col.replace(/^research\\./,"");
  popBody.innerHTML=linkify(text);
  const rect=td.getBoundingClientRect();
  let left=rect.left,top=rect.bottom-8;
  pop.classList.add("visible");popVisible=true;
  const pw=pop.offsetWidth,ph=pop.offsetHeight;
  if(left+pw>window.innerWidth-8)left=window.innerWidth-pw-8;
  if(left<8)left=8;
  if(top+ph>window.innerHeight-8)top=rect.top-ph+8;
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
    if(popVisible)popTimer=setTimeout(()=>{if(!pop.matches(":hover"))hidePopover();},400);
  }
});
pop.addEventListener("mouseleave",()=>{clearTimeout(popTimer);hidePopover();});
document.addEventListener("keydown",e=>{
  if(e.key==="Escape"){
    if(copyModal.classList.contains("show")){copyModal.classList.remove("show");return;}
    if(S.focusedCell){S.focusedCell=null;tbl.querySelectorAll("td.cell-focused").forEach(c=>c.classList.remove("cell-focused"));return;}
    if(popVisible)hidePopover();
    return;
  }
  if((e.metaKey||e.ctrlKey)&&e.key==="c"){
    const ae=document.activeElement;
    if(ae&&(ae.tagName==="INPUT"||ae.tagName==="TEXTAREA"))return;
    if(copyModal.classList.contains("show"))return;
    if(S.selected.size>0){
      e.preventDefault();
      const text=buildCopyText();
      const msg="Copied "+S.selected.size+" row"+(S.selected.size>1?"s":"")+" as "+copyFmt.toUpperCase();
      copyToClipboard(text).then(ok=>{if(ok)showToast(msg);else showCopyModal(text);});
      return;
    }
    if(S.focusedCell){
      e.preventDefault();
      const v=S.rows[S.focusedCell.idx]?.display[S.focusedCell.col];
      if(v!=null)copyToClipboard(String(v)).then(ok=>{if(ok)showToast("Cell copied");});
    }
  }
});

/* --- resize handle --- */
let resizing=false,startY=0,startH=0;
resizeHandle.addEventListener("mousedown",e=>{
  e.preventDefault();resizing=true;startY=e.clientY;startH=wrap.offsetHeight;
  resizeHandle.classList.add("active");
  document.addEventListener("mousemove",onResizeMove);
  document.addEventListener("mouseup",onResizeUp);
});
function onResizeMove(e){
  if(!resizing)return;
  const newH=Math.max(100,startH+(e.clientY-startY));
  wrap.style.maxHeight=newH+"px";
}
function onResizeUp(){
  resizing=false;resizeHandle.classList.remove("active");
  document.removeEventListener("mousemove",onResizeMove);
  document.removeEventListener("mouseup",onResizeUp);
}

/* --- fullscreen toggle --- */
expandBtn.addEventListener("click",async()=>{
  try{
    const next=S.isFullscreen?"contained":"fullscreen";
    await app.requestDisplayMode({mode:next});
  }catch(e){showToast("Fullscreen not supported");}
});

/* --- column resize --- */
let colResizing=false,colResizeTh=null,colStartX=0,colStartW=0;
tbl.addEventListener("mousedown",e=>{
  const handle=e.target.closest(".col-resize-handle");
  if(!handle)return;
  e.preventDefault();e.stopPropagation();
  colResizeTh=handle.parentElement;
  colStartX=e.clientX;colStartW=colResizeTh.offsetWidth;
  colResizing=true;
  tbl.style.tableLayout="fixed";
  document.body.classList.add("col-resizing");
  tbl.querySelectorAll(".hdr-row th").forEach(th=>{if(!th.style.width)th.style.width=th.offsetWidth+"px";});
  document.addEventListener("mousemove",onColResizeMove);
  document.addEventListener("mouseup",onColResizeUp);
});
function onColResizeMove(e){
  if(!colResizing)return;
  const delta=e.clientX-colStartX;
  colResizeTh.style.width=Math.max(30,colStartW+delta)+"px";
}
function onColResizeUp(){
  colResizing=false;colResizeTh=null;
  document.body.classList.remove("col-resizing");
  document.removeEventListener("mousemove",onColResizeMove);
  document.removeEventListener("mouseup",onColResizeUp);
}

/* --- column auto-fit (double-click resize handle) --- */
function measureColWidth(colIdx){
  const sp=document.createElement("span");
  sp.style.cssText="position:absolute;visibility:hidden;white-space:nowrap;padding:0 10px;font:13px system-ui";
  document.body.appendChild(sp);
  let maxW=0;
  const th=tbl.querySelectorAll(".hdr-row th")[colIdx];
  sp.style.fontWeight="600";sp.style.fontSize="12px";
  sp.textContent=th.dataset.col;
  maxW=Math.max(maxW,sp.offsetWidth+30);
  sp.style.fontWeight="normal";sp.style.fontSize="13px";
  tbl.querySelectorAll("tbody tr").forEach(tr=>{
    const td=tr.children[colIdx];
    if(td){sp.textContent=(td.textContent||"").slice(0,300);maxW=Math.max(maxW,sp.offsetWidth);}
  });
  document.body.removeChild(sp);
  return Math.min(Math.max(maxW+4,50),600);
}
tbl.addEventListener("dblclick",e=>{
  const handle=e.target.closest(".col-resize-handle");
  if(!handle)return;
  e.preventDefault();e.stopPropagation();
  const th=handle.parentElement;
  const colIdx=[...th.parentElement.children].indexOf(th);
  tbl.style.tableLayout="fixed";
  tbl.querySelectorAll(".hdr-row th").forEach(t=>{if(!t.style.width)t.style.width=t.offsetWidth+"px";});
  th.style.width=measureColWidth(colIdx)+"px";
});

/* --- column drag reorder --- */
let colDragging=false,dragCol=null,dragGhost=null,dragStartX=0,dragStartY=0;
const DRAG_THRESHOLD=5;
tbl.addEventListener("mousedown",e=>{
  if(e.target.closest(".col-resize-handle"))return;
  const th=e.target.closest(".hdr-row th");
  if(!th)return;
  dragCol=th.dataset.col;if(!dragCol)return;dragStartX=e.clientX;dragStartY=e.clientY;colDragging=false;
  document.addEventListener("mousemove",onColDragMove);
  document.addEventListener("mouseup",onColDragUp);
});
function onColDragMove(e){
  if(!dragCol)return;
  const dx=Math.abs(e.clientX-dragStartX),dy=Math.abs(e.clientY-dragStartY);
  if(!colDragging&&(dx>DRAG_THRESHOLD||dy>DRAG_THRESHOLD)){
    colDragging=true;didDrag=true;
    document.body.classList.add("col-dragging");
    dragGhost=document.createElement("div");
    dragGhost.className="col-ghost";dragGhost.textContent=dragCol;
    document.body.appendChild(dragGhost);
  }
  if(colDragging){
    dragGhost.style.left=(e.clientX+12)+"px";dragGhost.style.top=(e.clientY-12)+"px";
    const hdrs=[...tbl.querySelectorAll(".hdr-row th")].filter(h=>h.dataset.col);
    hdrs.forEach(h=>h.classList.remove("drag-over-left","drag-over-right"));
    const target=hdrs.find(h=>{const r=h.getBoundingClientRect();return e.clientX>=r.left&&e.clientX<=r.right;});
    if(target&&target.dataset.col!==dragCol){
      const r=target.getBoundingClientRect();
      target.classList.add(e.clientX<r.left+r.width/2?"drag-over-left":"drag-over-right");
    }
  }
}
function onColDragUp(e){
  document.removeEventListener("mousemove",onColDragMove);
  document.removeEventListener("mouseup",onColDragUp);
  if(colDragging){
    const hdrs=[...tbl.querySelectorAll(".hdr-row th")].filter(h=>h.dataset.col);
    hdrs.forEach(h=>h.classList.remove("drag-over-left","drag-over-right"));
    const target=hdrs.find(h=>{const r=h.getBoundingClientRect();return e.clientX>=r.left&&e.clientX<=r.right;});
    if(target&&target.dataset.col!==dragCol){
      const fromIdx=S.allCols.indexOf(dragCol);
      const toCol=target.dataset.col;
      let toIdx=S.allCols.indexOf(toCol);
      const r=target.getBoundingClientRect();
      if(e.clientX>=r.left+r.width/2)toIdx++;
      S.allCols.splice(fromIdx,1);
      if(fromIdx<toIdx)toIdx--;
      S.allCols.splice(toIdx,0,dragCol);
      renderTable();
    }
    if(dragGhost){dragGhost.remove();dragGhost=null;}
    document.body.classList.remove("col-dragging");
  }
  colDragging=false;dragCol=null;
}

/* --- export CSV / JSON --- */
async function copyToClipboard(text){
  try{await navigator.clipboard.writeText(text);return true;}catch{}
  if(execCopy(text))return true;
  return false;
}

/* The link the widget was handed stops working a day after the task was
   submitted, so ask for a live one at the moment it is wanted rather than
   holding one that quietly goes stale. */
async function refreshDownloadUrl(){
  if(currentTaskId&&hostProxiesTools()){
    try{
      const res=await app.callServerTool({
        name:"futuresearch_task_download",
        arguments:{params:{task_id:currentTaskId}},
      });
      const sc=res&&res.structuredContent;
      if(sc&&sc.download_url&&sc.poll_token){
        downloadUrl=sc.download_url;
        pollToken=sc.poll_token;
      }
    }catch(e){}
  }
  return getDownloadUrl();
}

async function exportResults(ev){
  const btn=ev&&ev.currentTarget;
  const label=btn&&btn.textContent;
  if(btn){btn.disabled=true;btn.textContent="Preparing...";}
  try{
    const url=await refreshDownloadUrl();
    if(!url){showToast("No download link yet");return;}
    app.openLink({url}).catch(()=>showCopyModal(url));
  }finally{
    if(btn){btn.disabled=false;btn.textContent=label;}
  }
}

function getDownloadUrl(){
  if(downloadUrl&&pollToken){
    return downloadUrl+(downloadUrl.includes("?")?"&":"?")+"token="+encodeURIComponent(pollToken);
  }
  return csvUrl;
}
document.getElementById("exportLink")?.addEventListener("click",exportResults);

/* --- row resize (drag bottom border) --- */
let rowResizing=false,rowResizeTr=null,rowStartY=0,rowStartH=0;
const ROW_EDGE=4;
function nearRowBottom(e,td){
  const r=td.getBoundingClientRect();
  return e.clientY>=r.bottom-ROW_EDGE&&e.clientY<=r.bottom+1;
}
tbl.addEventListener("mousemove",e=>{
  if(rowResizing||colResizing)return;
  const td=e.target.closest("tbody td");
  if(td&&nearRowBottom(e,td)){td.style.cursor="row-resize";}
  else if(td){td.style.cursor="";}
});
tbl.addEventListener("mousedown",e=>{
  const td=e.target.closest("tbody td");
  if(!td||!nearRowBottom(e,td))return;
  e.preventDefault();
  rowResizeTr=td.closest("tr");
  rowStartY=e.clientY;rowStartH=rowResizeTr.offsetHeight;
  rowResizing=true;
  document.body.classList.add("row-resizing");
  document.addEventListener("mousemove",onRowResizeMove);
  document.addEventListener("mouseup",onRowResizeUp);
});
function onRowResizeMove(e){
  if(!rowResizing)return;
  const delta=e.clientY-rowStartY;
  const newH=Math.max(16,rowStartH+delta)+"px";
  rowResizeTr.querySelectorAll("td").forEach(td=>td.style.height=newH);
}
function onRowResizeUp(){
  rowResizing=false;rowResizeTr=null;
  document.body.classList.remove("row-resizing");
  document.removeEventListener("mousemove",onRowResizeMove);
  document.removeEventListener("mouseup",onRowResizeUp);
}


/* --- data loading (for standalone results entry) --- */
async function fetchFullResultsWithFreshToken(hasPreview,total){
  const base=getDownloadUrl();
  if(!base){if(!hasPreview)sum.textContent="Download link expired";return;}
  const url=base+(base.includes("?")?"&":"?")+"format=json&include_research=1";
  fetchFullResults(url,{},hasPreview,total);
}
function fetchFullResults(url,opts,hasPreview,total){
  if(!hasPreview)sum.textContent="Loading"+(total?" "+total+" rows":"")+"...";
  fetch(url,opts).then(r=>{
    if(!r.ok)throw new Error(r.status+" "+r.statusText);
    return r.json();
  }).then(data=>processData(data)).catch(err=>{
    if(hasPreview){showToast("Full load failed, showing preview");}
    else{
      sum.innerHTML=esc("Failed to load: "+err.message)+' <button id="retryBtn" style="margin-left:8px;padding:2px 10px;border:1px solid var(--border);border-radius:4px;background:var(--btn-bg);color:var(--btn-text);cursor:pointer;font-size:12px">Retry</button>';
      document.getElementById("retryBtn")?.addEventListener("click",()=>fetchFullResultsWithFreshToken(hasPreview,total));
    }
  });
}

/* ── ontoolresult: unified entry point ── */

/* Take up a task the widget has just been handed. The host caches the tool
   result that mounted us, so the status it carries may be stale — "running"
   for a task that finished hours ago. Fetch current state before deciding
   whether there is anything left to poll for. */
async function followTask(d){
  enterProgressMode(d);
  pollUrl=d.progress_url;
  const r=await fetchTaskData(null);
  if(!r.data){
    if(r.permanent){setPollNote(r.note);return;}
    if(!pollTimer)startPoll();
    return;
  }
  /* The task's own spec outlives the metadata the widget was mounted with,
     which is replayed from Redis and expires a day after submission. */
  if(r.data.task_type==="forecast")useForecastSpec(r.data.spec);
  /* The server replays a re-mounted task's history into this payload, so
     there is nothing to catch up on here. */
  renderProgress(r.data);
  if(["completed","failed","revoked"].includes(r.data.status||d.status)){
    wasDone=true;
    if(!resultsFetched)loadResults(r.data.results);
  }else if(!pollTimer){
    startPoll();
  }
}

app.ontoolresult=({content,structuredContent})=>{

  /* Entry 1: structuredContent from futuresearch_status (widget data) */
  if(structuredContent&&structuredContent.progress_url){
    followTask(structuredContent);
    return;
  }

  /* Entry 2: content JSON from submission tools (progress_url embedded in text) */
  if(content){
    const t=content.find(c=>c.type==="text");
    if(t){
      try{
        const d=JSON.parse(t.text);
        if(d.progress_url){
          followTask(d);
          return;
        }
      }catch{}
    }
  }

  /* Legacy: standalone results data (kept for compatibility) */
  if(structuredContent){
    const meta=structuredContent;
    const isWidget=meta.fetch_full_results||meta.preview||Array.isArray(meta);
    if(!isWidget)return;
    showResultsUI();
    if(meta.poll_token){pollToken=meta.poll_token;}
    if(meta.download_url){downloadUrl=meta.download_url;}
    if(meta.csv_url){csvUrl=meta.csv_url;}
    if(meta.fetch_full_results){
      if(meta.preview)processData(meta.preview);
      fetchFullResultsWithFreshToken(!!meta.preview,meta.total);
    }else if(meta.preview){processData(meta.preview);}
    else if(Array.isArray(meta)){processData(meta);}
  }
};

await app.connect();
applyTheme();
</script></body></html>""".replace("SCRIPT_SRC", _APP_SCRIPT_SRC)


_AUTH_PAGES_CSS = """
*{box-sizing:border-box}
body{margin:0;min-height:100vh;display:flex;align-items:center;justify-content:center;
  padding:24px;font-family:system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;
  color:#1a1a1a;background:#fafafa}
.card{width:100%;max-width:400px;background:#fff;border:1px solid rgba(0,0,0,0.08);
  border-radius:14px;padding:28px;box-shadow:0 4px 24px rgba(0,0,0,0.06)}
h1{margin:0 0 6px;font-size:19px;font-weight:600}
p{margin:0 0 20px;font-size:14px;line-height:1.5;color:#525252}
p.who{margin-bottom:8px}
p.hint{margin:14px 0 0;font-size:13px}
p.error{color:#c62828;font-weight:500}
.opts{display:flex;flex-direction:column;gap:8px;margin-bottom:22px}
.opt{display:flex;align-items:center;gap:11px;padding:13px 15px;border-radius:10px;
  border:1px solid rgba(0,0,0,0.1);cursor:pointer;font-size:15px;transition:border-color .12s,background .12s}
.opt:hover{background:#f6f6fb}
.opt:has(input:checked){border-color:#4D4FBD;background:rgba(77,79,189,0.06)}
.opt input{accent-color:#4D4FBD;width:17px;height:17px;margin:0;flex:none}
.opt span{overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.providers{display:flex;flex-direction:column;gap:8px}
a.pbtn{display:flex;align-items:center;justify-content:center;gap:10px;padding:12px 15px;
  border-radius:10px;border:1px solid rgba(0,0,0,0.15);font-size:15px;font-weight:500;
  color:inherit;text-decoration:none;transition:border-color .12s,background .12s}
a.pbtn:hover{background:#f6f6fb}
a.pbtn svg{flex:none}
.divider{display:flex;align-items:center;gap:12px;margin:18px 0;color:#a3a3a3;font-size:13px}
.divider::before,.divider::after{content:"";flex:1;height:1px;background:rgba(0,0,0,0.1)}
.field{display:block;margin-bottom:12px;font-size:13px;font-weight:500;color:#525252}
.field input{display:block;width:100%;margin-top:5px;padding:10px 12px;font-size:15px;
  color:inherit;background:transparent;border:1px solid rgba(0,0,0,0.15);border-radius:10px}
.field input:focus{outline:none;border-color:#4D4FBD}
button{width:100%;padding:12px;border:0;border-radius:10px;background:#4D4FBD;color:#fff;
  font-size:15px;font-weight:600;cursor:pointer}
button:hover{background:#3f41a3}
@media(prefers-color-scheme:dark){
  body{color:#ededed;background:#161616}
  .card{background:#1e1e1e;border-color:rgba(255,255,255,0.1);box-shadow:none}
  p{color:#a3a3a3}
  p.error{color:#ef9a9a}
  .opt{border-color:rgba(255,255,255,0.14)}
  .opt:hover{background:#26263a}
  .opt:has(input:checked){background:rgba(146,148,240,0.12);border-color:#9294F0}
  a.pbtn{border-color:rgba(255,255,255,0.18)}
  a.pbtn:hover{background:#26263a}
  .divider::before,.divider::after{background:rgba(255,255,255,0.12)}
  .field input{border-color:rgba(255,255,255,0.18)}
  .field input:focus{border-color:#9294F0}
}
"""

_GOOGLE_SVG = """<svg width="18" height="18" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg" aria-hidden="true"><g transform="matrix(1, 0, 0, 1, 27.009001, -39.238998)"><path fill="#4285F4" d="M -3.264 51.509 C -3.264 50.719 -3.334 49.969 -3.454 49.239 L -14.754 49.239 L -14.754 53.749 L -8.284 53.749 C -8.574 55.229 -9.424 56.479 -10.684 57.329 L -10.684 60.329 L -6.824 60.329 C -4.564 58.239 -3.264 55.159 -3.264 51.509 Z"/><path fill="#34A853" d="M -14.754 63.239 C -11.514 63.239 -8.804 62.159 -6.824 60.329 L -10.684 57.329 C -11.764 58.049 -13.134 58.489 -14.754 58.489 C -17.884 58.489 -20.534 56.379 -21.484 53.529 L -25.464 53.529 L -25.464 56.619 C -23.494 60.539 -19.444 63.239 -14.754 63.239 Z"/><path fill="#FBBC05" d="M -21.484 53.529 C -21.734 52.809 -21.864 52.039 -21.864 51.239 C -21.864 50.439 -21.724 49.669 -21.484 48.949 L -21.484 45.859 L -25.464 45.859 C -26.284 47.479 -26.754 49.299 -26.754 51.239 C -26.754 53.179 -26.284 54.999 -25.464 56.619 L -21.484 53.529 Z"/><path fill="#EA4335" d="M -14.754 43.989 C -12.984 43.989 -11.404 44.599 -10.154 45.789 L -6.734 42.369 C -8.804 40.429 -11.514 39.239 -14.754 39.239 C -19.444 39.239 -23.494 41.939 -25.464 45.859 L -21.484 48.949 C -20.534 46.099 -17.884 43.989 -14.754 43.989 Z"/></g></svg>"""

_GITHUB_SVG = """<svg width="18" height="18" viewBox="0 0 98 96" xmlns="http://www.w3.org/2000/svg" aria-hidden="true"><path fill-rule="evenodd" clip-rule="evenodd" fill="currentColor" d="M48.854 0C21.839 0 0 22 0 49.217c0 21.756 13.993 40.172 33.405 46.69 2.427.49 3.316-1.059 3.316-2.362 0-1.141-.08-5.052-.08-9.127-13.59 2.934-16.42-5.867-16.42-5.867-2.184-5.704-5.42-7.17-5.42-7.17-4.448-3.015.324-3.015.324-3.015 4.934.326 7.523 5.052 7.523 5.052 4.367 7.496 11.404 5.378 14.235 4.074.404-3.178 1.699-5.378 3.074-6.6-10.839-1.141-22.243-5.378-22.243-24.283 0-5.378 1.94-9.778 5.014-13.2-.485-1.222-2.184-6.275.486-13.038 0 0 4.125-1.304 13.426 5.052a46.97 46.97 0 0 1 12.214-1.63c4.125 0 8.33.571 12.213 1.63 9.302-6.356 13.427-5.052 13.427-5.052 2.67 6.763.97 11.816.485 13.038 3.155 3.422 5.015 7.822 5.015 13.2 0 18.905-11.404 23.06-22.324 24.283 1.78 1.548 3.316 4.481 3.316 9.126 0 6.6-.08 11.897-.08 13.526 0 1.304.89 2.853 3.316 2.364 19.412-6.52 33.405-24.935 33.405-46.691C97.707 22 75.788 0 48.854 0z"/></svg>"""

_MICROSOFT_SVG = """<svg width="18" height="18" viewBox="0 0 23 23" xmlns="http://www.w3.org/2000/svg" aria-hidden="true"><rect x="0" y="0" width="10" height="10" fill="#f25022"/><rect x="11" y="0" width="10" height="10" fill="#7fba00"/><rect x="0" y="11" width="10" height="10" fill="#00a4ef"/><rect x="11" y="11" width="10" height="10" fill="#ffb900"/></svg>"""

# (url path segment, label, svg) per login-page button.
_LOGIN_PROVIDERS = [
    ("google", "Google", _GOOGLE_SVG),
    ("github", "GitHub", _GITHUB_SVG),
    ("azure", "Microsoft", _MICROSOFT_SVG),
]


def render_login_page(
    *,
    provider_url_base: str,
    password_action: str,
    state: str,
    email: str = "",
    error: str | None = None,
) -> str:
    """Render the MCP login page: OAuth provider buttons plus an email form.

    ``provider_url_base`` is the /auth/start/{state} URL; each provider button
    links to ``{provider_url_base}/{provider}``. ``email`` and ``error`` let a
    failed password attempt re-render with the address kept and a message
    shown.
    """
    base_esc = html.escape(provider_url_base, quote=True)
    buttons = "".join(
        f'<a class="pbtn" href="{base_esc}/{provider}">{svg}'
        f"<span>Continue with {label}</span></a>"
        for provider, label, svg in _LOGIN_PROVIDERS
    )
    error_html = f'<p class="error">{html.escape(error)}</p>' if error else ""
    action_esc = html.escape(password_action, quote=True)
    state_esc = html.escape(state, quote=True)
    email_esc = html.escape(email, quote=True)
    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="referrer" content="no-referrer">
<meta name="color-scheme" content="light dark">
<title>Sign in · FutureSearch</title>
<style>{_AUTH_PAGES_CSS}</style>
</head><body>
<main class="card">
<h1>Connect to FutureSearch</h1>
<p>Sign in the same way you do in the FutureSearch app.</p>
{error_html}
<div class="providers">{buttons}</div>
<div class="divider"><span>or</span></div>
<form method="post" action="{action_esc}">
<input type="hidden" name="state" value="{state_esc}">
<label class="field">Email
<input type="email" name="email" autocomplete="email" required value="{email_esc}"></label>
<label class="field">Password
<input type="password" name="password" autocomplete="current-password" required></label>
<button type="submit">Continue with email</button>
</form>
</main>
</body></html>"""


def render_account_selector(
    *,
    action: str,
    select_state: str,
    accounts: list[tuple[str, str]],
    signed_in_email: str | None,
) -> str:
    """Render the login-page account picker.

    ``accounts`` is a list of ``(account_id, display_name)`` pairs, personal
    first. With a single account it is preselected; with several, none is, so
    the choice has to be made deliberately. All values are user-influenced
    (team names, ids, the email), so every interpolation is HTML-escaped.
    """
    options = []
    for i, (account_id, name) in enumerate(accounts):
        checked = " checked" if len(accounts) == 1 else ""
        required = " required" if i == 0 else ""
        aid = html.escape(account_id, quote=True)
        label = html.escape(name)
        options.append(
            f'<label class="opt">'
            f'<input type="radio" name="account_id" value="{aid}"{checked}{required}>'
            f"<span>{label}</span></label>"
        )
    options_html = "".join(options)
    action_esc = html.escape(action, quote=True)
    state_esc = html.escape(select_state, quote=True)
    who_html = (
        f'<p class="who">Signed in as <strong>{html.escape(signed_in_email)}</strong></p>'
        if signed_in_email
        else ""
    )
    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="referrer" content="no-referrer">
<meta name="color-scheme" content="light dark">
<title>Choose an account · FutureSearch</title>
<style>{_AUTH_PAGES_CSS}</style>
</head><body>
<main class="card">
<h1>Choose an account</h1>
{who_html}
<p>Pick the account this connection will use for FutureSearch tasks and billing. To change it later, reconnect.</p>
<form method="post" action="{action_esc}">
<input type="hidden" name="select_state" value="{state_esc}">
<div class="opts">{options_html}</div>
<button type="submit">Continue</button>
</form>
<p class="hint">Missing a team you expected? You may be signed in with a different login than you use in the FutureSearch app. Try reconnecting with a different login method.</p>
</main>
</body></html>"""
