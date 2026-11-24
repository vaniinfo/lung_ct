"""Grading form v2 ("grades_v2") for the location-first viewer pages (build_viewer_page.py --viewer v2).

Differences from form v1 (viewer_grading.py, kept unchanged so earlier pages can be rebuilt):
  * Location is graded separately from the finding. For lung conditions the reviewer ticks the lobes
    (RUL RML RLL LUL LLL), or chooses "Indeterminate" or "Not assessed" (the explicit default). Pleural
    effusion is graded by side. The finding answer (present / absent / can't tell) is its own field and is
    never derived from, or merged with, the location.
  * Optional staged ("blinded") mode: step 1 shows the scan and the form only. The tool's answers, proposed
    locations, measurements and outlines appear after "Save and lock step 1". The locked step-1 answers are
    kept; later changes are stored next to them with a timestamp.
  * No new severity or extent categories: the severity and confidence choices are those of form v1.

Grades saved with form v1 (browser storage or CSV) still load: their location is recorded as
"not_recorded_form_v1", never as assessed. Nothing is sent over the network.
"""
from __future__ import annotations

import csv
import json

FORM_VERSION = "grades_v2"
LOBE_CONDITIONS = ("ground_glass", "consolidation", "lung_density", "atelectasis", "lung_nodule", "emphysema",
                   "bronchiectasis", "peribronchial_thickening", "interlobular_septal_thickening", "mosaic_attenuation",
                   "fibrotic_sequela")
SIDE_CONDITIONS = ("pleural_effusion",)

CSS = """
.gl{display:flex;flex-wrap:wrap;gap:3px;align-items:center;margin-top:5px}
.glt{font-size:11px;color:#57606a;margin-right:2px}
.gl button{background:#fff;color:#1f2328;border:1px solid #c9d1d9;border-radius:6px;padding:2px 6px;font-size:11.5px}
.gl button.sel{background:#0969da;border-color:#0969da;color:#fff}
.gl button.lx{color:#57606a;font-size:11px}.gl button.lx.sel{background:#6e7781;border-color:#6e7781;color:#fff}
.glnote{font-size:11px;color:#8a5a00;margin-top:2px}
.gl.idle{opacity:.45}.gl.idle button{pointer-events:none}
.glhint{font-size:11px;color:#57606a;margin-top:2px}
.stagebar{background:#fff4cc;border:1px solid #f0d36b;border-radius:10px;padding:8px 10px;margin-top:8px;font-size:13px;color:#3d3000}
.stagebar .lock{background:#9a6700;border-color:#9a6700;color:#fff;font-weight:600;margin-left:6px}
body.stage1 .agentonly{display:none !important}
.initial{font-size:11px;color:#57606a;margin-top:3px}
"""

JS = r"""
(function(){
 const G = __GRADING__;
 const KEY = 'lungct_grades_' + G.case_id + '_' + G.run + (G.blinded ? '_staged_v2' : '');
 const ANS = [['present','Present'],['absent','Absent'],['cannot_tell',"Can't tell"]];
 const SEV = ['','minimal','mild','moderate','severe'];
 const CONF = ['','sure','fairly sure','unsure'];
 const WHERE = {lobes: ['RUL','RML','RLL','LUL','LLL'], sides: ['Right','Left']};
 let st = {reviewer:'', overall:'', g:{}};
 try { const s = localStorage.getItem(KEY); if (s) st = Object.assign(st, JSON.parse(s)); } catch(e) {}
 // grades already received as a CSV are shown unless this browser holds answers edited after that CSV was saved
 const P = G.blinded ? null : G.prefill, PT = P ? (Date.parse(P.saved_utc) || 0) : 0;
 if (P && !st.cleared && !(st.edited && st.edited > PT)) { const t = PT || Date.now();
   st = {reviewer: P.reviewer, overall: P.overall, g: JSON.parse(JSON.stringify(P.g)), edited: t, saved: t, prefilled: true};
   try { localStorage.setItem(KEY, JSON.stringify(st)); } catch(e) {} }
 st.form = G.form;
 const store = () => { try { localStorage.setItem(KEY, JSON.stringify(st)); } catch(e) {} };
 const save = () => { st.edited = Date.now(); store(); count(); };
 const staged = () => G.blinded && !st.initial;          // step 1: the tool's output is still hidden
 const noLoc = () => ({status: 'not_assessed', where: []});
 function entry(key) {
   const g = st.g[key] || (st.g[key] = {answer:'', severity:'', confidence:'', comment:''});
   if (!g.loc) g.loc = (st.prefilled || !st.form_seen_v2) && g.answer ? {status: 'not_recorded_form_v1', where: []} : noLoc();
   return g;
 }
 // any change made after step 1 was locked is kept with a timestamp, next to the locked answer
 function setf(item, g, field, val) {
   const a = JSON.stringify(g[field]), b = JSON.stringify(val);
   if (a === b) return;
   if (G.blinded && st.initial) (st.amend = st.amend || []).push({utc: new Date().toISOString(), key: item.key, field: field, from: g[field], to: val});
   g[field] = val;
 }
 const agentAbsent = s => s === 'absent';
 function agreement(item, a) {
   if (!a) return ['', ''];
   if (a === 'cannot_tell' || item.status === 'cannot tell' || item.status === 'cannot assess') return ['open', 'open'];
   if (a === 'present') return item.status === 'present' ? ['agree', 'agrees with agent'] : ['missed', 'agent missed it'];
   return agentAbsent(item.status) ? ['agree', 'agrees with agent'] : ['over', 'agent over-called'];
 }
 const locStatus = (item, g) => !item.loc_kind ? 'not_applicable' : !g.answer ? '' : g.answer === 'absent' ? 'not_applicable' : g.loc.status;
 // header bar
 const tip = document.querySelector('.tip');
 const bar = document.createElement('div'); bar.className = 'gbar';
 bar.innerHTML = '<b>Your grades</b> Reviewer: <input id="grev" placeholder="your name">' +
   '<button class="save" id="gsave">&#128190; Save grades (CSV)</button><button class="clear" id="gclear">Clear</button>' +
   '<span id="gcount"></span><textarea id="gcase" placeholder="Overall comment on this case (optional)"></textarea>';
 tip.after(bar);
 const stage = document.createElement('div'); stage.className = 'stagebar'; stage.style.display = G.blinded ? '' : 'none';
 bar.before(stage);
 if (st.prefilled && P) { const n = document.createElement('span'); n.style.cssText = 'font-size:12px;color:#14532d;font-weight:600';
   n.textContent = 'Filled in from the grades ' + P.reviewer + ' sent on ' + String(P.saved_utc).slice(0, 10) +
     (P.form === 'grades_v1' ? ' (earlier form: location was not asked)' : ''); bar.querySelector('#gcount').after(n); }
 const rev = bar.querySelector('#grev'); rev.value = st.reviewer || '';
 rev.oninput = () => { st.reviewer = rev.value; save(); };
 const oc = bar.querySelector('#gcase'); oc.value = st.overall || '';
 oc.oninput = () => { st.overall = oc.value; save(); };
 const table = document.querySelector('.side table#chk');
 const head = table.querySelector('tr');
 const th = document.createElement('th'); th.textContent = 'Your grade and location'; head.appendChild(th);
 const rows = Array.from(table.querySelectorAll('tr')).slice(1);
 const painters = [];
 rows.forEach((tr, i) => {
   const item = G.items[i]; if (!item) return;
   const g = entry(item.key);
   const td = document.createElement('td'); td.className = 'grade';
   td.addEventListener('click', e => e.stopPropagation());
   const opts = WHERE[item.loc_kind] || [];
   td.innerHTML = '<div class="gb">' + ANS.map(a => '<button data-a="' + a[0] + '">' + a[1] + '</button>').join('') + '</div>' +
     '<div class="grow"><select class="sv" title="severity">' + SEV.map(s => '<option value="' + s + '">' + (s || 'severity') + '</option>').join('') + '</select>' +
     '<select class="cf" title="confidence">' + CONF.map(s => '<option value="' + s + '">' + (s || 'confidence') + '</option>').join('') + '</select></div>' +
     (item.loc_kind ? '<div class="gl"><span class="glt">Where:</span>' + opts.map(o => '<button class="lb" data-l="' + o + '">' + o + '</button>').join('') +
        '<button class="lx" data-s="indeterminate">Indeterminate</button><button class="lx" data-s="not_assessed">Not assessed</button></div><div class="glnote"></div>' : '') +
     '<input class="cm" placeholder="comment"><div class="agrw"></div><div class="initial"></div>';
   const btns = td.querySelectorAll('.gb button'), sv = td.querySelector('.sv'), cf = td.querySelector('.cf'),
         cm = td.querySelector('.cm'), agw = td.querySelector('.agrw'), gl = td.querySelector('.gl'), gn = td.querySelector('.glnote'),
         ini = td.querySelector('.initial');
   function paint() {
     btns.forEach(b => b.classList.toggle('sel', b.dataset.a === g.answer));
     sv.value = g.severity; cf.value = g.confidence; cm.value = g.comment;
     sv.disabled = g.answer !== 'present';
     if (gl) {
       // the location choices are always shown, so the reader sees them before answering; they are active once the answer is Present or Can't tell
       const on = g.answer === 'present' || g.answer === 'cannot_tell';
       gl.classList.toggle('idle', !on);
       gl.querySelectorAll('.lb').forEach(b => b.classList.toggle('sel', on && g.loc.status === 'stated' && g.loc.where.indexOf(b.dataset.l) >= 0));
       gl.querySelectorAll('.lx').forEach(b => b.classList.toggle('sel', on && g.loc.status === b.dataset.s));
       gn.textContent = on && g.loc.status === 'not_recorded_form_v1' ? 'Location was not asked in the earlier form (not recorded).' : '';
       gn.className = 'glnote';
       if (!on) { gn.className = 'glhint'; gn.textContent = g.answer === 'absent' ? 'Location does not apply to an absent finding.' : "Choose Present or Can't tell to mark where."; }
     }
     const [c, t] = staged() ? ['', ''] : agreement(item, g.answer);       // never shown before the tool's answer is revealed
     agw.innerHTML = c ? '<span class="agr ' + c + '">' + t + '</span>' : '';
     const g0 = st.initial && st.initial.g[item.key];
     ini.textContent = g0 && (g0.answer !== g.answer || JSON.stringify(g0.loc) !== JSON.stringify(g.loc))
       ? 'Step 1 (locked): ' + (g0.answer || '-') + (g0.loc && g0.loc.status === 'stated' ? ', ' + g0.loc.where.join(' ') : g0.loc && g0.answer !== 'absent' && item.loc_kind ? ', ' + g0.loc.status.replace('_', ' ') : '') : '';
   }
   painters.push(paint);
   btns.forEach(b => b.onclick = () => {
     const a = (g.answer === b.dataset.a ? '' : b.dataset.a);
     setf(item, g, 'answer', a);
     if (a !== 'present') setf(item, g, 'severity', '');
     if (a !== 'present' && a !== 'cannot_tell') setf(item, g, 'loc', noLoc());
     else if (g.loc.status === 'not_recorded_form_v1') setf(item, g, 'loc', noLoc());
     paint(); save(); });
   sv.onchange = () => { setf(item, g, 'severity', sv.value); save(); };
   cf.onchange = () => { setf(item, g, 'confidence', cf.value); save(); };
   cm.onchange = () => { setf(item, g, 'comment', cm.value); save(); };
   if (gl) {
     gl.querySelectorAll('.lb').forEach(b => b.onclick = () => {
       const w = g.loc.status === 'stated' ? g.loc.where.slice() : [];
       const k = w.indexOf(b.dataset.l); if (k >= 0) w.splice(k, 1); else w.push(b.dataset.l);
       w.sort((x, y) => opts.indexOf(x) - opts.indexOf(y));
       setf(item, g, 'loc', w.length ? {status: 'stated', where: w} : noLoc()); paint(); save(); });
     gl.querySelectorAll('.lx').forEach(b => b.onclick = () => { setf(item, g, 'loc', {status: b.dataset.s, where: []}); paint(); save(); });
   }
   paint(); tr.appendChild(td);
 });
 st.form_seen_v2 = true; store();
 function count() {
   const n = G.items.filter(it => (st.g[it.key] || {}).answer).length;
   document.getElementById('gcount').textContent = n + ' of ' + G.items.length + ' graded (autosaved in this browser)';
 }
 function paintStage() {
   document.body.classList.toggle('stage1', staged());
   bar.querySelector('#gsave').style.display = staged() ? 'none' : '';
   bar.querySelector('#gclear').style.display = G.blinded && st.initial ? 'none' : '';
   if (!G.blinded) return;
   if (staged()) {
     stage.innerHTML = '<b>Step 1 of 2: your own reading.</b> Grade every condition, and where it is, from the scan alone. ' +
       "The tool's answers, locations, measurements and coloured outlines stay hidden until you lock this step. " +
       'Locked answers are kept as they are.<button class="lock" id="glock">&#128274; Save and lock step 1</button>';
     stage.querySelector('#glock').onclick = () => {
       if (!st.reviewer.trim()) { rev.focus(); rev.style.borderColor = '#cf222e'; alert('Please type your name first.'); return; }
       const open = G.items.filter(it => !(st.g[it.key] || {}).answer);
       if (open.length) { alert(open.length + ' condition(s) have no answer yet (first: ' + open[0].label + '). Choose Present, Absent or Can\'t tell for each.'); return; }
       if (!confirm('Lock step 1? Your answers and locations will be kept as they are, and the tool\'s output will be shown.')) return;
       st.initial = {locked_utc: new Date().toISOString(), reviewer: st.reviewer, g: JSON.parse(JSON.stringify(st.g))};
       st.amend = []; save(); paintStage(); painters.forEach(p => p());
     };
   } else {
     stage.innerHTML = '<b>Step 2 of 2.</b> Step 1 was locked on ' + st.initial.locked_utc.slice(0, 16).replace('T', ' ') +
       ' UTC and is kept. The tool\'s answers, locations, measurements and outlines are now shown. You may change a grade; ' +
       'each change is saved with its time next to your locked answer. Then <b>Save grades (CSV)</b>.';
   }
 }
 count(); paintStage();
 const q = v => '"' + String(v == null ? '' : v).replace(/"/g, '""') + '"';
 bar.querySelector('#gsave').onclick = () => {
   if (!st.reviewer.trim()) { rev.focus(); rev.style.borderColor = '#cf222e'; alert('Please type your name first.'); return; }
   const now = new Date().toISOString();
   const cols = ['case_id','reviewer','saved_utc','condition_key','condition','agent_answer','agent_status','doctor_answer','severity','confidence','agreement','comment','checklist_config','run',
     'form_version','viewer_build','location_kind','location_status','location_where','agent_location_text','agent_lobes_named','agent_lobe_assignment_uncertain',
     'staged','initial_locked_utc','initial_answer','initial_severity','initial_confidence','initial_location_status','initial_location_where','initial_comment','changed_after_reveal','amendments_json'];
   const lines = [cols.join(',')];
   const ini = st.initial || null;
   G.items.forEach(it => {
     const g = entry(it.key), g0 = ini ? (ini.g[it.key] || {}) : null;
     const am = (st.amend || []).filter(a => a.key === it.key);
     lines.push([G.case_id, st.reviewer, now, it.key, it.label, it.display, it.status, g.answer || '', g.severity || '', g.confidence || '',
       agreement(it, g.answer)[1], g.comment || '', G.config, G.run,
       G.form, G.viewer, it.loc_kind || '', locStatus(it, g), g.loc.status === 'stated' ? g.loc.where.join(' ') : '',
       it.agent_location_text || '', (it.agent_lobes_named || []).join(' '), it.agent_uncertain == null ? '' : (it.agent_uncertain ? 'yes' : 'no'),
       G.blinded ? 'yes' : 'no', ini ? ini.locked_utc : '', g0 ? (g0.answer || '') : '', g0 ? (g0.severity || '') : '', g0 ? (g0.confidence || '') : '',
       g0 ? locStatus(it, Object.assign({loc: noLoc()}, g0)) : '', g0 && g0.loc && g0.loc.status === 'stated' ? g0.loc.where.join(' ') : '',
       g0 ? (g0.comment || '') : '', ini ? (am.length ? 'yes' : 'no') : '', am.length ? JSON.stringify(am) : ''].map(q).join(','));
   });
   const blank = cols.slice(14).map(() => '');
   lines.push([G.case_id, st.reviewer, now, '_overall', 'Overall comment', '', '', '', '', '', '', st.overall || '', G.config, G.run].concat(
     [G.form, G.viewer], blank.slice(2, 8), [G.blinded ? 'yes' : 'no', ini ? ini.locked_utc : ''], blank.slice(10)).map(q).join(','));
   const blob = new Blob(['﻿' + lines.join('\r\n')], {type: 'text/csv'});
   const a = document.createElement('a');
   a.href = URL.createObjectURL(blob);
   a.download = 'grades_' + G.case_id + '_' + st.reviewer.trim().replace(/[^A-Za-z0-9]+/g, '_') + '_' + now.slice(0, 10) + '.csv';
   document.body.appendChild(a); a.click(); a.remove();
   st.saved = Date.now(); store();
 };
 bar.querySelector('#gclear').onclick = () => {
   if (!confirm('Clear all your grades for this case?')) return;
   st = {reviewer: st.reviewer, overall: '', g: {}, cleared: true, form_seen_v2: true}; save(); location.reload();
 };
})();
"""


def loc_kind(key: str) -> str | None:
    return "lobes" if key in LOBE_CONDITIONS else "sides" if key in SIDE_CONDITIONS else None


def grading_items(chk: dict) -> list[dict]:
    out = []
    for it in chk["items"]:
        lr = it.get("location_report") or {}
        named = lr.get("lobes_named")
        if named is None and lr.get("parts"):
            named = sorted({n for p in lr["parts"] for n in p["lobes_named"]}, key="RUL RML RLL LUL LLL".split().index)
        out.append({"key": it["key"], "label": it["label"], "status": it["status"], "display": it.get("display", it["status"]),
                    "loc_kind": loc_kind(it["key"]), "agent_location_text": it.get("location_text") or it.get("location") or "",
                    "agent_lobes_named": named or [], "agent_uncertain": lr.get("lobe_assignment_uncertain")})
    return out


def load_grades_csv(path, case_id: str, chk: dict) -> dict:
    """Grades a reviewer already sent back, from form v1 or v2. A v1 file carries no location: every graded
    condition gets location status 'not_recorded_form_v1' so it can never be read as assessed."""
    with open(path, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    if not rows or any(r["case_id"] != case_id for r in rows):
        raise ValueError(f"{path}: not a grades file for {case_id}")
    keys = {it["key"] for it in chk["items"]}
    unknown = [r["condition_key"] for r in rows if r["condition_key"] != "_overall" and r["condition_key"] not in keys]
    if unknown:
        raise ValueError(f"{path}: conditions not in this checklist: {unknown}")
    form = rows[0].get("form_version") or "grades_v1"
    g = {}
    for r in rows:
        if r["condition_key"] == "_overall":
            continue
        if form == "grades_v1":
            loc = {"status": "not_recorded_form_v1" if r["doctor_answer"] else "not_assessed", "where": []}
        else:
            s = r.get("location_status") or "not_assessed"
            loc = {"status": "not_assessed" if s == "not_applicable" else s, "where": (r.get("location_where") or "").split()}
        g[r["condition_key"]] = {"answer": r["doctor_answer"], "severity": r["severity"], "confidence": r["confidence"],
                                 "comment": r["comment"], "loc": loc}
    overall = next((r["comment"] for r in rows if r["condition_key"] == "_overall"), "")
    return {"reviewer": rows[0]["reviewer"], "saved_utc": rows[0]["saved_utc"], "overall": overall, "g": g, "form": form}


POLICY_EXTRA_COLS = ["checklist_version", "pack", "agent_current_assessment", "agent_headline_shown", "agent_stored_status", "agent_decision_basis",
                     "agent_mask_state", "provisional_threshold_result_NOT_A_CURRENT_ASSESSMENT", "provisional_limits"]
POLICY_AGREEMENT_TEXT = {"suggested_reviewer_present": "segmentation suggested it; reviewer: present",
                         "suggested_reviewer_absent": "segmentation suggested it; reviewer: absent"}


PREFILL_EXTRA_COLS = ["answer_origin", "carried_from_policy", "carried_from_saved_utc"]


def policy_js(js: str, carried: bool = False) -> str:
    """Grading script for a policy-labelled pack (v3_8 and later). The base script is left untouched, so earlier pages rebuild
    byte-identically. Differences:
      - own storage key per policy and pack: grades typed on a page of another policy are never carried over;
      - a positive answer that rests on a predicted mask only ('suggested') is never scored as the agent's confirmed answer:
        agreement is 'suggested_reviewer_present' / 'suggested_reviewer_absent', not 'agrees with agent' / 'agent over-called';
      - the CSV column agent_status carries the CURRENT assessment category (suggested / cannot tell / ...), and extra columns
        carry the checklist version, pack, headline, stored status, decision basis, mask state, and the provisional threshold
        result in a column named as not being a current assessment."""
    def rep(a: str, b: str) -> None:
        nonlocal js
        assert js.count(a) == 1, a[:60]
        js = js.replace(a, b)
    rep("const KEY = 'lungct_grades_' + G.case_id + '_' + G.run + (G.blinded ? '_staged_v2' : '');",
        "const KEY = 'lungct_grades_' + G.case_id + '_' + G.run + (G.blinded ? '_staged_v2' : '') + '_' + G.policy.config + '_' + G.policy.pack;")
    rep(" function agreement(item, a) {\n   if (!a) return ['', ''];\n",
        " const AGT = " + json.dumps(POLICY_AGREEMENT_TEXT) + ";\n"
        " function agreement(item, a) {\n   if (!a) return ['', ''];\n"
        "   if (item.category === 'suggested') return a === 'cannot_tell' ? ['open', 'open'] : a === 'present' ? ['agree', 'suggested_reviewer_present']\n"
        "     : ['over', 'suggested_reviewer_absent'];\n")
    rep("agw.innerHTML = c ? '<span class=\"agr ' + c + '\">' + t + '</span>' : '';",
        "agw.innerHTML = c ? '<span class=\"agr ' + c + '\">' + (AGT[t] || t) + '</span>' : '';")
    rep(" tip.after(bar);\n",
        " tip.after(bar);\n"
        " { const pn = document.createElement('div'); pn.style.cssText = 'font-size:12px;color:#3d3000;margin-top:4px';\n"
        "   pn.textContent = 'Grades saved from this page are recorded as made on ' + G.policy.config + ' (pack ' + G.policy.pack + '). ' +\n"
        "     'Earlier grades were made on other pages and are not copied into this form.'; bar.appendChild(pn); }\n")
    rep("it.label, it.display, it.status, g.answer || ''", "it.label, it.display, it.category, g.answer || ''")
    rep("   const blob = new Blob(",
        "   { const x = " + json.dumps(POLICY_EXTRA_COLS) + ";\n"
        "     lines[0] += ',' + x.join(',');\n"
        "     G.items.forEach((it, i) => { lines[i + 1] += ',' + [G.policy.config, G.policy.pack, it.category, it.headline_shown, it.status, it.decision_basis,\n"
        "       it.mask_state, it.threshold_answer, it.threshold_limits].map(q).join(','); });\n"
        "     lines[lines.length - 1] += ',' + [G.policy.config, G.policy.pack, '', '', '', '', '', '', ''].map(q).join(','); }\n"
        "   const blob = new Blob(")
    if carried:
        # Pack with the reviewer's earlier grades filled in (asked for by the project owner, pack M6). Every exported row says whether the
        # answer was carried over unchanged from the earlier grade, changed on this page, or entered on this page, and which policy
        # and date the earlier grade was made on. A carried-over answer is NOT a new assessment of this page's wording.
        rep("     'Earlier grades were made on other pages and are not copied into this form.'; bar.appendChild(pn); }\n",
            "     'Your earlier grades (made on a page showing ' + (G.prefill ? G.prefill.made_on : 'an earlier policy') + ') are filled in below. ' +\n"
            "     'The answers on this page may differ from that page: please check each grade and change it where needed.'; bar.appendChild(pn); }\n")
        rep("   const blob = new Blob(",
            "   { const f = x => x ? [x.answer || '', x.severity || '', x.confidence || '', x.comment || '', (x.loc && x.loc.status) || '',\n"
            "         ((x.loc && x.loc.where) || []).join(' ')].join('|') : '';\n"
            "     const org = k => { const a = st.g[k], b = (st.prefilled && P && P.g[k]) || null;\n"
            "       if (!a || !a.answer) return ''; if (!b || !b.answer) return 'entered_on_this_page';\n"
            "       return f(a) === f(b) ? 'carried_over_from_earlier_grade_unchanged' : 'changed_on_this_page'; };\n"
            "     lines[0] += ',' + " + json.dumps(",".join(PREFILL_EXTRA_COLS)) + ";\n"
            "     G.items.forEach((it, i) => { const o = org(it.key), c = o && o !== 'entered_on_this_page';\n"
            "       lines[i + 1] += ',' + [o, c ? P.made_on : '', c ? P.saved_utc : ''].map(q).join(','); });\n"
            "     lines[lines.length - 1] += ',,,'; }\n"
            "   const blob = new Blob(")
    return js


def grading_script(case_id: str, run_name: str, config: str, chk: dict, viewer_build: str, prefill: dict | None = None,
                   blinded: bool = False, policy: dict | None = None, allow_policy_prefill: bool = False) -> str:
    if blinded and prefill:
        raise ValueError("a staged (blinded) page cannot be pre-filled with earlier grades")
    data = {"case_id": case_id, "run": run_name, "config": config, "form": FORM_VERSION, "viewer": viewer_build,
            "blinded": blinded, "items": grading_items(chk), "prefill": prefill}
    if policy is None:
        return "<script>\n" + JS.replace("__GRADING__", json.dumps(data)) + "\n</script>"
    from lung_ct.agent.checklist import export_row      # same definitions as the report / CSV export of the agent
    if prefill and not allow_policy_prefill:
        raise ValueError("a policy-labelled page is not pre-filled unless asked for explicitly: earlier grades were made on a page showing another policy")
    if prefill and not prefill.get("made_on"):
        raise ValueError("carried-over grades must name the policy they were made on")
    if policy["config"] != config:
        raise ValueError("policy label and checklist version differ")
    by = {i["key"]: i for i in chk["items"]}
    for it in data["items"]:
        r = export_row(case_id, config, by[it["key"]])
        if r["current_assessment"] == "absent" and by[it["key"]].get("mask_state"):
            raise ValueError(f"{it['key']}: a predicted mask below a reporting limit must never be exported as absent")
        it.update(category=r["current_assessment"], headline_shown=r["headline_shown"], decision_basis=r["decision_basis"], mask_state=r["mask_state"],
                  threshold_answer=r["provisional_threshold_result_NOT_A_CURRENT_ASSESSMENT"], threshold_limits=r["provisional_limits"])
    data["policy"] = policy
    return "<script>\n" + policy_js(JS, carried=bool(prefill)).replace("__GRADING__", json.dumps(data)) + "\n</script>"
