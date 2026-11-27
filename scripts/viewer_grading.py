"""Grading panel for the interactive viewer pages (used by build_viewer_page.py and add_grading_to_viewer.py).

The reviewer grades every checklist condition next to the scan: their own answer (present / absent /
cannot tell), severity, confidence and a comment. Agreement with the agent is worked out automatically.
'Save grades' downloads a CSV (one row per condition) to send back; work is also autosaved in the
browser (localStorage, per computer and browser). Nothing is sent over the network.
"""
from __future__ import annotations

import json

CSS = """
.page{grid-template-columns:1.15fr 1fr !important}
.gbar{display:flex;flex-wrap:wrap;gap:8px;align-items:center;background:#eef4ff;border:1px solid #b9cdf5;border-radius:10px;padding:8px 10px;margin-top:8px}
.gbar input{font:inherit;padding:3px 6px;border:1px solid #b9c3d3;border-radius:6px;width:150px}
.gbar .save{background:#1a7f37;border-color:#1a7f37;color:#fff;font-weight:600}
.gbar .clear{background:#fff;color:#57606a;border-color:#c9d1d9}
#gcount{font-size:12px;color:#3d4b66}
td.grade{min-width:205px;cursor:default;background:#fbfcff}
.gb{display:inline-flex;border:1px solid #c9d1d9;border-radius:7px;overflow:hidden}
.gb button{background:#fff;color:#1f2328;border:0;border-right:1px solid #c9d1d9;border-radius:0;padding:3px 7px;font-size:12px}
.gb button:last-child{border-right:0}
.gb button.sel[data-a=present]{background:#cf222e;color:#fff}.gb button.sel[data-a=absent]{background:#1a7f37;color:#fff}
.gb button.sel[data-a=cannot_tell]{background:#6e7781;color:#fff}
.grow{display:flex;gap:4px;margin-top:4px}
.grow select{font:inherit;font-size:11.5px;padding:1px 2px;border:1px solid #c9d1d9;border-radius:5px;max-width:100px}
.grade input.cm{width:100%;margin-top:4px;font:inherit;font-size:11.5px;padding:2px 5px;border:1px solid #c9d1d9;border-radius:5px}
.agr{display:inline-block;margin-top:4px;font-size:11px;font-weight:600;border-radius:8px;padding:0 6px}
.agr.agree{background:#d6f0da}.agr.missed{background:#fbd5d0}.agr.over{background:#fde7c4}.agr.open{background:#eceef1}
textarea#gcase{width:100%;min-height:46px;font:inherit;font-size:12.5px;margin-top:6px;border:1px solid #c9d1d9;border-radius:6px;padding:4px 6px}
"""

JS = r"""
(function(){
 const G = __GRADING__;
 const KEY = 'lungct_grades_' + G.case_id + '_' + G.run;
 const ANS = [['present','Present'],['absent','Absent'],['cannot_tell',"Can't tell"]];
 const SEV = ['','minimal','mild','moderate','severe'];
 const CONF = ['','sure','fairly sure','unsure'];
 let st = {reviewer:'', overall:'', g:{}};
 try { const s = localStorage.getItem(KEY); if (s) st = Object.assign(st, JSON.parse(s)); } catch(e) {}
 // grades already received as a CSV are shown unless this browser holds answers edited after that CSV was saved (or they were cleared here)
 const P = G.prefill, PT = P ? (Date.parse(P.saved_utc) || 0) : 0;
 if (P && !st.cleared && !(st.edited && st.edited > PT)) { const t = PT || Date.now(); st = {reviewer: P.reviewer, overall: P.overall, g: JSON.parse(JSON.stringify(P.g)), edited: t, saved: t, prefilled: true}; try { localStorage.setItem(KEY, JSON.stringify(st)); } catch(e) {} }
 const store = () => { try { localStorage.setItem(KEY, JSON.stringify(st)); } catch(e) {} };
 const save = () => { st.edited = Date.now(); store(); count(); };  // edited/saved times drive the status on index.html
 const agentAbsent = s => s === 'absent';
 function agreement(item, a) {
   if (!a) return ['', ''];
   if (a === 'cannot_tell' || item.status === 'cannot tell' || item.status === 'cannot assess') return ['open', 'open'];
   if (a === 'present') return item.status === 'present' ? ['agree', 'agrees with agent'] : ['missed', 'agent missed it'];
   return agentAbsent(item.status) ? ['agree', 'agrees with agent'] : ['over', 'agent over-called'];
 }
 // header bar: reviewer, save, progress, overall comment
 const tip = document.querySelector('.tip');
 const bar = document.createElement('div'); bar.className = 'gbar';
 bar.innerHTML = '<b>Your grades</b> Reviewer: <input id="grev" placeholder="your name">' +
   '<button class="save" id="gsave">&#128190; Save grades (CSV)</button><button class="clear" id="gclear">Clear</button>' +
   '<span id="gcount"></span><textarea id="gcase" placeholder="Overall comment on this case (optional)"></textarea>';
 tip.after(bar);
 if (st.prefilled && P) { const n = document.createElement('span'); n.style.cssText = 'font-size:12px;color:#14532d;font-weight:600';
   n.textContent = 'Filled in from the grades ' + P.reviewer + ' sent on ' + String(P.saved_utc).slice(0, 10); bar.querySelector('#gcount').after(n); }
 const rev = bar.querySelector('#grev'); rev.value = st.reviewer || '';
 rev.oninput = () => { st.reviewer = rev.value; save(); };
 const oc = bar.querySelector('#gcase'); oc.value = st.overall || '';
 oc.oninput = () => { st.overall = oc.value; save(); };
 // add a 'Your grade' column to the checklist table
 const table = document.querySelector('.side table');
 const head = table.querySelector('tr');
 const th = document.createElement('th'); th.textContent = 'Your grade'; head.appendChild(th);
 const rows = Array.from(table.querySelectorAll('tr')).slice(1);
 rows.forEach((tr, i) => {
   const item = G.items[i]; if (!item) return;
   const g = st.g[item.key] || (st.g[item.key] = {answer:'', severity:'', confidence:'', comment:''});
   const td = document.createElement('td'); td.className = 'grade';
   td.addEventListener('click', e => e.stopPropagation());
   td.innerHTML = '<div class="gb">' + ANS.map(a => '<button data-a="' + a[0] + '">' + a[1] + '</button>').join('') + '</div>' +
     '<div class="grow"><select class="sv" title="severity">' + SEV.map(s => '<option value="' + s + '">' + (s || 'severity') + '</option>').join('') + '</select>' +
     '<select class="cf" title="confidence">' + CONF.map(s => '<option value="' + s + '">' + (s || 'confidence') + '</option>').join('') + '</select></div>' +
     '<input class="cm" placeholder="comment"><div class="agrw"></div>';
   const btns = td.querySelectorAll('.gb button'), sv = td.querySelector('.sv'), cf = td.querySelector('.cf'),
         cm = td.querySelector('.cm'), agw = td.querySelector('.agrw');
   function paint() {
     btns.forEach(b => b.classList.toggle('sel', b.dataset.a === g.answer));
     sv.value = g.severity; cf.value = g.confidence; cm.value = g.comment;
     sv.disabled = g.answer !== 'present';
     const [c, t] = agreement(item, g.answer);
     agw.innerHTML = c ? '<span class="agr ' + c + '">' + t + '</span>' : '';
   }
   btns.forEach(b => b.onclick = () => { g.answer = (g.answer === b.dataset.a ? '' : b.dataset.a); if (g.answer !== 'present') g.severity = ''; paint(); save(); });
   sv.onchange = () => { g.severity = sv.value; save(); };
   cf.onchange = () => { g.confidence = cf.value; save(); };
   cm.oninput = () => { g.comment = cm.value; save(); };
   paint(); tr.appendChild(td);
 });
 function count() {
   const n = G.items.filter(it => (st.g[it.key] || {}).answer).length;
   document.getElementById('gcount').textContent = n + ' of ' + G.items.length + ' graded (autosaved in this browser)';
 }
 count();
 const q = v => '"' + String(v == null ? '' : v).replace(/"/g, '""') + '"';
 bar.querySelector('#gsave').onclick = () => {
   if (!st.reviewer.trim()) { rev.focus(); rev.style.borderColor = '#cf222e'; alert('Please type your name first.'); return; }
   const now = new Date().toISOString();
   const cols = ['case_id','reviewer','saved_utc','condition_key','condition','agent_answer','agent_status','doctor_answer','severity','confidence','agreement','comment','checklist_config','run'];
   const lines = [cols.join(',')];
   G.items.forEach(it => {
     const g = st.g[it.key] || {};
     lines.push([G.case_id, st.reviewer, now, it.key, it.label, it.display, it.status, g.answer || '', g.severity || '', g.confidence || '',
                 agreement(it, g.answer)[1], g.comment || '', G.config, G.run].map(q).join(','));
   });
   lines.push([G.case_id, st.reviewer, now, '_overall', 'Overall comment', '', '', '', '', '', '', st.overall || '', G.config, G.run].map(q).join(','));
   const blob = new Blob(['﻿' + lines.join('\r\n')], {type: 'text/csv'});
   const a = document.createElement('a');
   a.href = URL.createObjectURL(blob);
   a.download = 'grades_' + G.case_id + '_' + st.reviewer.trim().replace(/[^A-Za-z0-9]+/g, '_') + '_' + now.slice(0, 10) + '.csv';
   document.body.appendChild(a); a.click(); a.remove();
   st.saved = Date.now(); store();
 };
 bar.querySelector('#gclear').onclick = () => {
   if (!confirm('Clear all your grades for this case?')) return;
   st = {reviewer: st.reviewer, overall: '', g: {}, cleared: true}; save(); location.reload();
 };
})();
"""


def grading_items(chk: dict) -> list[dict]:
    return [{"key": it["key"], "label": it["label"], "status": it["status"],
             "display": it.get("display", it["status"])} for it in chk["items"]]


def load_grades_csv(path, case_id: str, chk: dict) -> dict:
    """Grades a reviewer already sent back (CSV saved by this panel), to show them again in a rebuilt page."""
    import csv
    with open(path, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    if not rows or any(r["case_id"] != case_id for r in rows):
        raise ValueError(f"{path}: not a grades file for {case_id}")
    keys = {it["key"] for it in chk["items"]}
    unknown = [r["condition_key"] for r in rows if r["condition_key"] != "_overall" and r["condition_key"] not in keys]
    if unknown:
        raise ValueError(f"{path}: conditions not in this checklist: {unknown}")
    g = {r["condition_key"]: {"answer": r["doctor_answer"], "severity": r["severity"], "confidence": r["confidence"],
                              "comment": r["comment"]} for r in rows if r["condition_key"] != "_overall"}
    overall = next((r["comment"] for r in rows if r["condition_key"] == "_overall"), "")
    return {"reviewer": rows[0]["reviewer"], "saved_utc": rows[0]["saved_utc"], "overall": overall, "g": g}


def grading_script(case_id: str, run_name: str, config: str, chk: dict, prefill: dict | None = None) -> str:
    data = {"case_id": case_id, "run": run_name, "config": config, "items": grading_items(chk), "prefill": prefill}
    return "<script>\n" + JS.replace("__GRADING__", json.dumps(data)) + "\n</script>"
