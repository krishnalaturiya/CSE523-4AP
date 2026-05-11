const API_BASE = (window.location.hostname==="localhost"||window.location.hostname==="127.0.0.1")
  ? `${window.location.protocol}//${window.location.host}`
  : "http://localhost:8000";

const charts = { 1:{compare:null,heatmap:null,segs:{},gestureSegs:{}}, 2:{compare:null,heatmap:null,segs:{},gestureSegs:{}} };
let gestureChart = null;
const segmentPunchlines = { 1:[], 2:[] };
const punchlineCharts = { 1:null, 2:null };

const segGestureMetrics = {};    // "vid_segIdx" → Set of selected metric keys
let   globalGestureMetric = "arm_velocity";
const _gestureDataByVid   = {};  // vid → gestureData (kept for chip re-renders)
let   _lastGestureData    = null;
const PART_COLORS = ["#f5e642","#ff6b35","#4fffb0","#a78bfa","#ef4444","#fbbf24","#3b82f6"];

const ALL_METRICS = [
  "shoulder_width","shoulder_height_diff","hip_width","torso_tilt",
  "avg_elbow_distance","avg_arm_height","arm_spread","arm_velocity",
  "body_center_velocity","hand_distance","hand_velocity",
];
const METRIC_LABELS = {
  shoulder_width:       "Shoulder Width",
  shoulder_height_diff: "Shoulder Tilt",
  hip_width:            "Hip Width",
  torso_tilt:           "Torso Tilt",
  avg_elbow_distance:   "Elbow Dist",
  avg_arm_height:       "Arm Height",
  arm_spread:           "Arm Spread",
  arm_velocity:         "Arm Velocity",
  body_center_velocity: "Body Velocity",
  hand_distance:        "Hand Dist",
  hand_velocity:        "Hand Velocity",
};
const METRIC_COLORS = {
  shoulder_width:       "#f5e642",
  shoulder_height_diff: "#ff6b35",
  hip_width:            "#4fffb0",
  torso_tilt:           "#a78bfa",
  avg_elbow_distance:   "#ef4444",
  avg_arm_height:       "#fbbf24",
  arm_spread:           "#3b82f6",
  arm_velocity:         "#f472b6",
  body_center_velocity: "#34d399",
  hand_distance:        "#fb923c",
  hand_velocity:        "#60a5fa",
};

const DEFAULTS = {
  1:[
    {label:"Part 1",start_s:0,  end_s:51},
    {label:"Part 2",start_s:52, end_s:108},
    {label:"Part 3",start_s:109,end_s:151},
    {label:"Part 4",start_s:152,end_s:202},
    {label:"Part 5",start_s:203,end_s:262},
  ],
  2:[
    {label:"Part 1",start_s:82, end_s:133},
    {label:"Part 2",start_s:134,end_s:174},
    {label:"Part 3",start_s:175,end_s:215},
    {label:"Part 4",start_s:216,end_s:272},
    {label:"Part 5",start_s:273,end_s:320},
  ]
};

// ── Segment row builder ──────────────────────────────────────────────────────

function buildDefaultRows(vid) {
  document.getElementById(`seg-rows-${vid}`).innerHTML="";
  DEFAULTS[vid].forEach(s=>appendSegRow(vid,s.label,s.start_s,s.end_s));
}

function appendSegRow(vid,label="",start="",end="") {
  const row=document.createElement("div"); row.className="seg-row";
  row.innerHTML=`
    <input class="lbl-input" type="text" placeholder="Part N" value="${label}" />
    <input type="number" placeholder="0"  min="0" value="${start}" />
    <input type="number" placeholder="60" min="0" value="${end}" />
    <button class="seg-del" onclick="this.closest('.seg-row').remove()" title="Remove">×</button>`;
  document.getElementById(`seg-rows-${vid}`).appendChild(row);
}

function addSegRow(vid) { appendSegRow(vid); }

function getSegments(vid) {
  const segs=[];
  document.querySelectorAll(`#seg-rows-${vid} .seg-row`).forEach(row=>{
    const inp=row.querySelectorAll("input");
    const label=inp[0].value.trim()||`Part ${segs.length+1}`;
    const s=parseFloat(inp[1].value), e=parseFloat(inp[2].value);
    if(!isNaN(s)&&!isNaN(e)&&e>s) segs.push({label,start_s:s,end_s:e});
  });
  return segs;
}

// ── Tab switching ────────────────────────────────────────────────────────────

function switchTab(n) {
  document.querySelectorAll(".tab-panel").forEach(p=>p.classList.remove("active"));
  document.querySelectorAll(".tab-btn").forEach(b=>b.classList.remove("active"));
  document.getElementById(`panel-${n}`).classList.add("active");
  document.getElementById(`tab-btn-${n}`).classList.add("active");
}

window.addEventListener("DOMContentLoaded",()=>{ buildDefaultRows(1); buildDefaultRows(2); });

// ── Analysis fetch ───────────────────────────────────────────────────────────

async function runAnalysis(vid) {
  const customUrl=document.getElementById(`url-input-${vid}`).value.trim();
  const segs=getSegments(vid);
  const errEl=document.getElementById(`error-box-${vid}`);
  errEl.classList.remove("visible");

  if(!segs.length){errEl.textContent="⚠ Add at least one valid segment (end > start).";errEl.classList.add("visible");return;}

  setStatus(vid,"Running audio + gesture analysis…","Whisper + YAMNet + MediaPipe running in parallel. First run takes 10–25 min.");
  document.getElementById(`analyze-btn-${vid}`).disabled=true;

  try {
    const url=customUrl||(vid===2?"https://www.youtube.com/watch?v=faNCfJL9008":"https://www.youtube.com/watch?v=WSx2gqeEyLE");
    const form=new FormData(); form.append("url",url); form.append("segments",JSON.stringify(segs));
    const resp=await fetch(`${API_BASE}/api/analyze/full`,{method:"POST",body:form});
    if(!resp.ok) throw new Error(`Server error ${resp.status}: ${await resp.text()}`);
    const data=await resp.json();
    hideStatus(vid);
    renderAll(vid,data);
  } catch(e) {
    hideStatus(vid); showError(vid,e.message);
  } finally {
    document.getElementById(`analyze-btn-${vid}`).disabled=false;
  }
}

// ── Render all results ───────────────────────────────────────────────────────

function renderAll(vid,data) {
  const segs=data.segments||[], winIdx=data.overall_winner_index??0;
  document.getElementById(`results-${vid}`).style.display="block";
  const w=segs[winIdx]||{};
  document.getElementById(`winner-banner-${vid}`).classList.add("show");
  document.getElementById(`winner-label-${vid}`).textContent=(w.label||"—")+" had the strongest reaction";
  document.getElementById(`winner-lpm-${vid}`).textContent=fmt(w.laughs_per_minute,1);
  document.getElementById(`winner-coverage-${vid}`).textContent=fmt(w.laugh_coverage_pct,1)+"%";
  document.getElementById(`winner-score-${vid}`).textContent=w.comedy_score??"—";
  renderHeatmapChart(vid,segs,winIdx);
  if(!window._lastSegData) window._lastSegData={};
  window._lastSegData[vid]={segs,winIdx};
  renderSegmentCards(vid,segs,winIdx,data.gesture||null);
  if (data.gesture?.per_second_metrics) {
    // Wait for all per-segment chart setTimeout(60ms) calls to complete before building the global view
    setTimeout(() => {
      buildGlobalGestureChips();
      renderGestureChart(data.gesture);
    }, 150);
  }
  if(data.punchline_analysis?.length>0){
    const el=document.getElementById(`punchline-status-${vid}`); if(el) el.style.display="none";
    segmentPunchlines[vid]=data.punchline_analysis;
    renderPunchlineChart(vid,data.punchline_analysis);
    renderPunchlineTable(vid,data.punchline_analysis);
    segs.forEach((seg,i)=>{renderSegHeatmap(vid,i,seg,PART_COLORS[i%PART_COLORS.length]);renderEventsTable(vid,i,seg.laugh_events,data.punchline_analysis,seg.start_s);});
  } else { loadPunchlineData(vid); }
}

// ── Charts ───────────────────────────────────────────────────────────────────

function renderHeatmapChart(vid,segs,winIdx) {
  if(charts[vid].heatmap) charts[vid].heatmap.destroy();
  const ctx=document.getElementById(`chart-heatmaps-${vid}`).getContext("2d");
  const NOISE=0.035;
  const addNoise=(raw,pi)=>raw.map((v,j)=>Math.max(0,Math.min(1,(v??0)+Math.sin(pi*17.3+j*3.7)*NOISE)));
  charts[vid].heatmap=new Chart(ctx,{type:"line",data:{labels:(segs[0]?.heatmap||[]).map((_,i)=>i+1),datasets:segs.map((s,i)=>({label:s.label+(i===winIdx?" 🏆":""),data:addNoise(s.heatmap,i),borderColor:PART_COLORS[i%PART_COLORS.length],backgroundColor:PART_COLORS[i%PART_COLORS.length]+"18",fill:false,tension:0.4,pointRadius:3,borderWidth:i===winIdx?3:1.5}))},
    options:{responsive:true,maintainAspectRatio:true,interaction:{mode:"index",intersect:false},
      plugins:{legend:{labels:{color:"#6b6a5e",font:{size:11}}},tooltip:{callbacks:{title:items=>`⏱ Bin ${(items[0]?.dataIndex??0)+1}:`,label:ctx2=>{const i=ctx2.datasetIndex,bi=ctx2.dataIndex,sp=segs[i]?.heatmap_texts?.[bi],pl=segs[i]?.label||`Part ${i+1}`;return(sp&&sp!=="(no speech)")?`  ${pl}: "${sp}"`:` ${pl}: ${((segs[i]?.heatmap?.[bi]??0)*100).toFixed(1)}%`;}},bodyFont:{size:11},titleFont:{size:11,weight:"bold"},maxWidth:420,padding:12}},
      scales:{x:{ticks:{color:"#6b6a5e",maxTicksLimit:10},grid:{color:"#1a1a17"}},y:{ticks:{color:"#6b6a5e"},grid:{color:"#1a1a17"},beginAtZero:true,max:1}}}});
}

function renderSegmentCards(vid,segs,winIdx,gestureData=null) {
  const grid=document.getElementById(`segments-grid-${vid}`); grid.innerHTML="";
  segs.forEach((seg,i)=>{
    const isW=i===winIdx,id=`seg-${vid}-${i}`,color=PART_COLORS[i%PART_COLORS.length];
    const card=document.createElement("div"); card.className="seg-card"+(isW?" winner":""); card.id=id;
    card.innerHTML=`
      <div class="seg-header" onclick="toggleSeg('${id}')">
        <div class="seg-name" style="color:${color}">${seg.label}</div>
        ${isW?'<div class="seg-winner-badge">🏆 Winner</div>':""}
        <div class="seg-mini-stats">
          <div class="seg-mini-stat"><div class="v">${seg.comedy_score}</div><div class="l">Score</div></div>
          <div class="seg-mini-stat"><div class="v">${fmt(seg.laughs_per_minute,1)}</div><div class="l">Laughs/min</div></div>
          <div class="seg-mini-stat"><div class="v">${fmt(seg.laugh_coverage_pct,1)}%</div><div class="l">Coverage</div></div>
        </div>
        <div class="seg-toggle">▼</div>
      </div>
      <div class="seg-body">
        <div class="mini-cards">
          <div class="mini-card"><div class="v">${seg.total_laugh_events}</div><div class="l">Laugh Events</div></div>
          <div class="mini-card"><div class="v">${fmt(seg.avg_laugh_latency_s,2)}s</div><div class="l">Avg Latency</div></div>
          <div class="mini-card"><div class="v">${fmt(seg.avg_setup_word_count,0)}</div><div class="l">Setup Words</div></div>
          <div class="mini-card"><div class="v">${fmtTime(seg.duration_s)}</div><div class="l">Duration</div></div>
        </div>
        <div class="seg-video-wrap">
          <div class="seg-video-label">// Video Clip — ${seg.label} &nbsp;(${fmtTime(seg.start_s)} – ${fmtTime(seg.end_s)})</div>
          ${seg.clip_url
            ? `<video controls preload="metadata" src="${API_BASE}${seg.clip_url}"></video>`
            : `<div class="seg-video-none">⏳ Video clip not available — clips are generated during analysis</div>`
          }
        </div>
        <div class="chart-box"><h4>Audience Reaction Per Punchline</h4><canvas id="hm-${vid}-${i}"></canvas></div>
        <div class="events-wrap">
          <table><thead><tr><th>#</th><th>Time</th><th>Duration</th><th>Intensity</th><th>Latency</th><th>Matched Punchline</th></tr></thead>
          <tbody id="tbody-${vid}-${i}"></tbody></table>
        </div>
        <div class="gesture-segment-panel" style="margin-top:20px;padding-top:16px;border-top:1px solid #1a1a17">
          <div style="font-size:0.72rem;color:var(--muted);font-family:'DM Mono',monospace;margin-bottom:8px">📊 Movement Metrics</div>
          <div class="gesture-metric-chips" id="gesture-chips-${vid}-${i}"></div>
          <canvas id="gesture-chart-${vid}-${i}" height="65"></canvas>
          <div id="gesture-chart-status-${vid}-${i}" style="font-size:0.7rem;color:#555;margin-top:4px;font-family:'DM Mono',monospace"></div>
        </div>
      </div>`;
    grid.appendChild(card);
    if(isW) toggleSeg(id);
    setTimeout(()=>{
      renderSegHeatmap(vid,i,seg,color);
      renderEventsTable(vid,i,seg.laugh_events,segmentPunchlines[vid],seg.start_s);
      renderSegmentGestureChart(vid,i,gestureData);
    },60);
  });
  // Metric legend — once after all cards
  const legend = document.createElement("div"); legend.className="metric-legend";
  legend.innerHTML=`
    <div class="metric-legend-title">Metric Descriptions</div>
    <div class="metric-legend-body">
      <div><span>Laugh Events</span> — distinct contiguous laughter bursts YAMNet detected.</div>
      <div><span>Avg Latency</span> — gap between joke end and laughter start. Longer = more cognitive processing.</div>
      <div><span>Setup Words</span> — avg word count of speech in 8s window before each laugh.</div>
      <div><span>Coverage</span> — total laugh seconds ÷ segment duration × 100.</div>
      <div><span>Score</span> — <code>(laughs/min × 4) + (avg intensity × 35) + (avg duration × 5) + 10</code></div>
    </div>`;
  grid.appendChild(legend);
}

function toggleSeg(id){document.getElementById(id).classList.toggle("open");}

function renderSegHeatmap(vid,i,seg,color) {
  const key=`${vid}_${i}`; if(charts[vid].segs[key]) charts[vid].segs[key].destroy();
  const el=document.getElementById(`hm-${vid}-${i}`); if(!el) return;
  const ctx=el.getContext("2d"), punchlines=segmentPunchlines[vid];
  let labels,data,bgColors;
  if(punchlines?.length>0){
    labels=punchlines.map(pl=>pl.label);
    data=punchlines.map(pl=>{const sd=pl.segments?.find(s=>s.label===seg.label);return(!sd||!sd.found)?0:+(Math.min(1,sd.combined_peak)*100).toFixed(1);});
    bgColors=punchlines.map((_,j)=>{const v=data[j];return v>60?"#4fffb0bb":v>30?"#f5e642bb":v>0?color+"bb":"#333";});
  } else {
    labels=(seg.heatmap||[]).map((_,j)=>j+1);data=seg.heatmap||[];bgColors=color+"bb";
  }
  charts[vid].segs[key]=new Chart(ctx,{type:"bar",data:{labels,datasets:[{data,backgroundColor:bgColors,borderRadius:4}]},
    options:{
      responsive:true,maintainAspectRatio:true,
      onClick:(evt,elements)=>{
        if(!elements.length||!punchlines?.length) return;
        const plIdx=elements[0].index;
        const pl=punchlines[plIdx];
        const sd=pl?.segments?.find(s=>s.label===seg.label);
        if(!sd||!sd.found) return;
        if(sd.punchline_clip_url){
          openPlModal(sd.punchline_clip_url, pl.label, seg.label, sd.win_start_s||0, sd.win_end_s||8);
        } else {
          alert(`No clip available for "${pl.label}" in ${seg.label} yet.\nRe-run analysis to generate sub-clips.`);
        }
      },
      plugins:{
        legend:{display:false},
        tooltip:{callbacks:{label:ctx2=>{
          const pl=punchlines?.[ctx2.dataIndex],sd=pl?.segments?.find(s=>s.label===seg.label);
          if(!sd||!sd.found) return " Not found";
          const hasClip=sd.punchline_clip_url?" 🎬 Click to play":"";
          return[` Reaction: ${ctx2.parsed.y.toFixed(1)}%`,` 😂 ${(sd.laugh_peak*100).toFixed(0)}%  👏 ${(sd.applause_peak*100).toFixed(0)}%${hasClip}`];
        }}}
      },
      scales:{
        x:{ticks:{color:"#6b6a5e",maxRotation:20,font:{size:10}},grid:{color:"#0a0a08"}},
        y:{ticks:{color:"#6b6a5e",callback:v=>v+"%"},grid:{color:"#0a0a08"},beginAtZero:true,max:100}
      }
    }
  });
  el.style.cursor = punchlines?.length ? "pointer" : "default";
}

// ── Events table ─────────────────────────────────────────────────────────────

function renderEventsTable(vid,i,events,punchlines,start_s=0) {
  const tbody=document.getElementById(`tbody-${vid}-${i}`); if(!tbody) return;
  if(!events?.length){tbody.innerHTML=`<tr><td colspan="6" style="color:var(--muted);text-align:center;padding:14px">No laugh events detected.</td></tr>`;return;}
  function matchPL(text){if(!punchlines?.length||!text) return null;const low=text.toLowerCase();let best=null,hits=0;punchlines.forEach(pl=>{const h=(pl.keywords||[]).filter(kw=>low.includes(kw.toLowerCase())).length;if(h>hits){hits=h;best=pl.label;}});return hits>0?best:null;}
  const groups={};let ui=0;
  events.forEach(e=>{const label=matchPL(e.setup_text),key=label||`__u__${ui++}`;if(!groups[key]) groups[key]={punchlineLabel:label,events:[],firstSetupText:e.setup_text||""};groups[key].events.push(e);});
  let ri=1;
  tbody.innerHTML=Object.values(groups).map(g=>{
    const evs=g.events,dur=evs.reduce((s,e)=>s+(e.laugh_duration_s||0),0),intens=evs.reduce((s,e)=>s+(e.laugh_intensity_mean||0),0)/evs.length,lat=evs.reduce((s,e)=>s+(e.laugh_latency_s||0),0)/evs.length;
    const plCell=g.punchlineLabel?`<span style="color:var(--accent);font-weight:600">${g.punchlineLabel}</span>${evs.length>1?`<span style="font-size:0.68rem;color:var(--muted);margin-left:6px">(${evs.length} bursts)</span>`:""}<div style="font-size:0.7rem;color:var(--muted);margin-top:2px">${g.firstSetupText.slice(0,90)}${g.firstSetupText.length>90?"…":""}</div>`:`<span style="font-size:0.75rem;color:var(--muted)">${g.firstSetupText.slice(0,100)}${g.firstSetupText.length>100?"…":""}</span>`;
    return`<tr><td style="color:var(--muted)">${ri++}</td><td>${fmtTime(evs[0].laugh_start_s - start_s)}</td><td>${fmt(dur,1)}s</td><td>${fmt(intens,3)}</td><td>${fmt(lat,2)}s</td><td style="max-width:260px">${plCell}</td></tr>`;
  }).join("");
}

// ── Punchline data + charts ──────────────────────────────────────────────────

async function loadPunchlineData(vid) {
  const statusEl=document.getElementById(`punchline-status-${vid}`);
  try {
    const ep=vid===2?"/api/punchlines/video2":"/api/punchlines/video1";
    const resp=await fetch(`${API_BASE}${ep}`);
    if(!resp.ok){statusEl.textContent="ℹ️ Punchline data not available — run analysis first.";return;}
    const data=await resp.json(); const pa=data.punchline_analysis||[];
    if(!pa.length){statusEl.textContent="ℹ️ No punchlines detected.";return;}
    statusEl.style.display="none"; segmentPunchlines[vid]=pa;
    renderPunchlineChart(vid,pa); renderPunchlineTable(vid,pa);
    if(window._lastSegData?.[vid]){const{segs,winIdx}=window._lastSegData[vid];segs.forEach((seg,i)=>{renderSegHeatmap(vid,i,seg,PART_COLORS[i%PART_COLORS.length]);renderEventsTable(vid,i,seg.laugh_events,pa,seg.start_s);});}
  } catch(e){statusEl.textContent="⚠️ Could not load punchlines: "+e.message;}
}

function renderPunchlineChart(vid,pa) {
  const canvas=document.getElementById(`chart-punchlines-${vid}`); canvas.style.display="block";
  const pLabels=pa.map(p=>p.label),partLabels=(pa[0]?.segments||[]).map(s=>s.label);
  const datasets=partLabels.map((pl,pi)=>({label:pl,data:pa.map(p=>{const seg=p.segments.find(s=>s.label===pl);return(!seg||!seg.found)?0:+(Math.min(1,seg.combined_peak)*100).toFixed(1);}),backgroundColor:PART_COLORS[pi%PART_COLORS.length]+"cc",borderRadius:4}));
  if(punchlineCharts[vid]) punchlineCharts[vid].destroy();
  punchlineCharts[vid]=new Chart(document.getElementById(`chart-punchlines-${vid}`).getContext("2d"),{type:"bar",data:{labels:pLabels,datasets},
    options:{responsive:true,maintainAspectRatio:true,plugins:{legend:{labels:{color:"#6b6a5e",font:{size:11}}},tooltip:{mode:"index",callbacks:{title:items=>{const pl=pa[items[0]?.dataIndex];return[pl?.label||"",pl?.description||""];},label:ctx2=>{const pl2=ctx2.dataset.label,p=pa[ctx2.dataIndex],sd=p?.segments?.find(s=>s.label===pl2);return(!sd||!sd.found)?`  ${pl2}: not found`:`  ${pl2}: ${ctx2.parsed.y.toFixed(1)}%`;}},bodyFont:{size:11},titleFont:{size:11,weight:"bold"},padding:12,maxWidth:380}},
    scales:{x:{ticks:{color:"#6b6a5e",maxRotation:20},grid:{color:"#1a1a17"}},y:{ticks:{color:"#6b6a5e",callback:v=>v+"%"},grid:{color:"#1a1a17"},beginAtZero:true,max:100}}}});
  document.getElementById(`punchline-legend-${vid}`).innerHTML=partLabels.map((lbl,i)=>`<div class="pl-legend-item"><div class="pl-dot" style="background:${PART_COLORS[i%PART_COLORS.length]}"></div>${lbl}</div>`).join("");
}

function renderPunchlineTable(vid,pa) {
  const wrap=document.getElementById(`punchline-table-wrap-${vid}`);
  const thead=`<thead><tr><th>Punchline</th><th>Description</th><th>Best Part</th></tr></thead>`;
  const tbody=pa.map((pl)=>{
    const found=pl.segments.filter(s=>s.found),best=found.length?found.reduce((a,b)=>a.combined_peak>b.combined_peak?a:b):null;
    return`<tr><td style="font-weight:600;font-size:0.78rem">${pl.label}</td><td style="font-size:0.75rem;color:var(--muted)">${pl.description}</td><td style="font-weight:700;color:var(--gold)">${best?best.label:"—"}</td></tr>`;
  }).join("");
  wrap.innerHTML=`<div style="overflow-x:auto"><table class="punchline-table">${thead}<tbody>${tbody}</tbody></table></div>`;
}

// ── Punchline clip modal ─────────────────────────────────────────────────────

function openPlModal(clipUrl, plLabel, segLabel, winStart, winEnd) {
  const video = document.getElementById("plModalVideo");
  document.getElementById("plModalTitle").textContent = plLabel;
  document.getElementById("plModalSub").textContent = `${segLabel} — punchline window`;
  document.getElementById("plModalFooter").textContent =
    `Window: ${fmtTime(winStart)} – ${fmtTime(winEnd)} (relative to segment)`;
  video.src = `${API_BASE}${clipUrl}`;
  video.load();
  document.getElementById("plModal").classList.add("open");
}

function closePlModal(e) {
  if (e.target === document.getElementById("plModal")) closePlModalDirect();
}

function closePlModalDirect() {
  const video = document.getElementById("plModalVideo");
  video.pause();
  video.src = "";
  document.getElementById("plModal").classList.remove("open");
}

document.addEventListener("keydown", e => { if (e.key === "Escape") closePlModalDirect(); });

// ── Per-segment gesture charts ───────────────────────────────────────────────

function buildSegmentGestureChips(vid, segIdx) {
  const container = document.getElementById(`gesture-chips-${vid}-${segIdx}`);
  if (!container) return;
  const key      = `${vid}_${segIdx}`;
  const selected = segGestureMetrics[key] || new Set();
  container.innerHTML = "";
  ALL_METRICS.forEach(metric => {
    const chip = document.createElement("span");
    const isOn = selected.has(metric);
    chip.className   = "gesture-chip" + (isOn ? " active" : "");
    chip.textContent = METRIC_LABELS[metric];
    if (isOn) {
      chip.style.background   = METRIC_COLORS[metric];
      chip.style.borderColor  = METRIC_COLORS[metric];
      chip.style.color        = "#000";
    }
    chip.onclick = () => {
      const sel = segGestureMetrics[`${vid}_${segIdx}`];
      if (sel.has(metric)) {
        if (sel.size > 1) sel.delete(metric);  // keep at least one selected
      } else {
        sel.add(metric);
      }
      renderSegmentGestureChart(vid, segIdx, _gestureDataByVid[vid]);
    };
    container.appendChild(chip);
  });
}

function renderSegmentGestureChart(vid, segIdx, gestureData) {
  const canvas   = document.getElementById(`gesture-chart-${vid}-${segIdx}`);
  const statusEl = document.getElementById(`gesture-chart-status-${vid}-${segIdx}`);
  if (!canvas) return;

  const rows = gestureData?.per_second_metrics?.[`Segment_${segIdx + 1}`];
  if (!rows?.length) {
    if (statusEl) statusEl.textContent = gestureData
      ? "No gesture data for this segment."
      : "Gesture data unavailable — run via Analyze to include it.";
    return;
  }

  // Persist data so chip re-renders can call back here without re-fetching
  _gestureDataByVid[vid] = gestureData;

  // Initialise metric selection to arm_velocity + arm_spread on first call
  const key = `${vid}_${segIdx}`;
  if (!segGestureMetrics[key]) {
    segGestureMetrics[key] = new Set(["arm_velocity", "arm_spread"]);
  }

  buildSegmentGestureChips(vid, segIdx);

  const datasets = [...segGestureMetrics[key]].map(metric => ({
    label:           METRIC_LABELS[metric],
    data:            rows.map(r => ({ x: r.second, y: r[metric] ?? 0 })),
    borderColor:     METRIC_COLORS[metric],
    backgroundColor: "transparent",
    tension: 0.3, pointRadius: 2, borderWidth: 1.5,
  }));

  if (charts[vid].gestureSegs[key]) charts[vid].gestureSegs[key].destroy();
  charts[vid].gestureSegs[key] = new Chart(canvas.getContext("2d"), {
    type: "line",
    data: { datasets },
    options: {
      responsive: true,
      maintainAspectRatio: true,
      plugins: {
        legend: { display: false },
        tooltip: { callbacks: { title: items => `s = ${items[0]?.raw?.x}` } },
      },
      scales: {
        x: {
          type: "linear",
          ticks: { color: "#6b6a5e", font: { size: 9 }, maxTicksLimit: 12 },
          grid: { color: "#1a1a17" },
        },
        y: {
          beginAtZero: true,
          ticks: { color: "#6b6a5e", font: { size: 9 } },
          grid: { color: "#1a1a17" },
        },
      },
    },
  });
}

// ── Gesture Analysis Integration ─────────────────────────────────────────────

function buildGlobalGestureChips() {
  const container = document.getElementById("gesture-global-chips");
  if (!container) return;
  container.innerHTML = "";
  ALL_METRICS.forEach(metric => {
    const chip = document.createElement("span");
    const isOn = metric === globalGestureMetric;
    chip.className   = "gesture-chip" + (isOn ? " active" : "");
    chip.textContent = METRIC_LABELS[metric];
    if (isOn) {
      chip.style.background  = METRIC_COLORS[metric];
      chip.style.borderColor = METRIC_COLORS[metric];
      chip.style.color       = "#000";
    }
    chip.onclick = () => {
      globalGestureMetric = metric;
      buildGlobalGestureChips();
      if (_lastGestureData) renderGestureChart(_lastGestureData);
    };
    container.appendChild(chip);
  });
}

async function loadGestureData() {
  const statusEl = document.getElementById("gesture-status");
  try {
    const resp = await fetch(`${API_BASE}/api/gesture/results`);
    if (!resp.ok) {
      statusEl.textContent = `⚠ Gesture data unavailable (${resp.status}) — run gestureanalysis pipeline first.`;
      buildGlobalGestureChips();
      return;
    }
    const data = await resp.json();
    buildGlobalGestureChips();
    renderGestureChart(data);
    const segCount = Object.keys(data.per_second_metrics || {}).length;
    statusEl.textContent = `Loaded ${segCount} segment(s) from gestureanalysis/results.json`;
  } catch(e) {
    statusEl.textContent = "⚠ Could not load gesture data: " + e.message;
    buildGlobalGestureChips();
  }
}

function renderGestureChart(data) {
  _lastGestureData = data;
  // Reveal panel and sync title to the currently selected metric
  const panel = document.getElementById("gesture-panel");
  if (panel) panel.style.display = "block";
  const labelEl = document.getElementById("gesture-panel-label");
  if (labelEl) labelEl.textContent = `// Gesture Analysis — ${METRIC_LABELS[globalGestureMetric]}`;
  const metrics  = data.per_second_metrics || {};
  const datasets = Object.entries(metrics).map(([segLabel, rows], i) => ({
    label:           segLabel,
    data:            rows.map(r => ({ x: r.second, y: r[globalGestureMetric] ?? 0 })),
    borderColor:     PART_COLORS[i % PART_COLORS.length],
    backgroundColor: "transparent",
    tension: 0.3, pointRadius: 3, borderWidth: 2,
  }));

  if (gestureChart) gestureChart.destroy();
  const ctx = document.getElementById("chart-gesture").getContext("2d");
  gestureChart = new Chart(ctx, {
    type: "line",
    data: { datasets },
    options: {
      responsive: true,
      maintainAspectRatio: true,
      plugins: {
        legend: { labels: { color: "#6b6a5e", font: { size: 11 } } },
        tooltip: { callbacks: { title: items => `Second ${items[0]?.raw?.x ?? ""}` } },
      },
      scales: {
        x: {
          type: "linear",
          title: { display: true, text: "Second", color: "#6b6a5e" },
          ticks: { color: "#6b6a5e" }, grid: { color: "#1a1a17" },
        },
        y: {
          title: { display: true, text: METRIC_LABELS[globalGestureMetric], color: "#6b6a5e" },
          ticks: { color: "#6b6a5e" }, grid: { color: "#1a1a17" }, beginAtZero: true,
        },
      },
    },
  });
}

// ── Helpers ──────────────────────────────────────────────────────────────────

function fmt(n,d=1){return(n!=null)?Number(n).toFixed(d):"—";}
function fmtTime(s){if(s==null)return"—";const m=Math.floor(s/60),sec=Math.floor(s%60);return`${m}:${String(sec).padStart(2,"0")}`;}
function setStatus(vid,t,d){document.getElementById(`status-title-${vid}`).textContent=t;document.getElementById(`status-detail-${vid}`).textContent=d;document.getElementById(`status-bar-${vid}`).classList.add("visible");document.getElementById(`results-${vid}`).style.display="none";}
function hideStatus(vid){document.getElementById(`status-bar-${vid}`).classList.remove("visible");}
function showError(vid,m){const el=document.getElementById(`error-box-${vid}`);el.textContent="⚠ "+m;el.classList.add("visible");}

// ── Collapsible sections ─────────────────────────────────────────────────────
function toggleCollapsible(id, header) {
  const body = document.getElementById(id);
  const toggle = header.querySelector('.coll-toggle');
  const open = body.style.display === 'none';
  body.style.display = open ? 'block' : 'none';
  toggle.style.transform = open ? 'rotate(180deg)' : '';
}