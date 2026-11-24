"""Check a policy-labelled viewer pack (core env): what each page shows, and what its grading form scores and exports.

Part 1 (always): parses every page of the pack (scan payload removed) and checks the visible text and the embedded grading
data against the checklist file the page was built from.
Part 2 (--browser edge|chrome): opens each UNMODIFIED page from file:// in a headless browser with a throw-away profile,
grades with the automated-test reviewer marker, and reads the agreement text and the exported CSV. The export is captured
in memory and written under runs/test_logs/ with a name that says it is not from a reviewer. Nothing is sent anywhere.

Usage: python scripts/check_policy_pack.py --folder runs/M5_policy_v3_8_20261123 --browser edge
"""
from __future__ import annotations

import argparse
import csv
import html
import io
import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from lung_ct.agent.checklist import assessment_category, export_row  # noqa: E402
from viewer_grading_v2 import POLICY_EXTRA_COLS  # noqa: E402

PAYLOAD = re.compile(r'<script type="application/octet-stream"[^>]*>[^<]*</script>\n?')
TAG = re.compile(r"<[^>]+>")


def text(s: str) -> str:
    return html.unescape(TAG.sub(" ", s))


def static_checks(page: Path, p: dict, policy: str, pack: str, ok) -> dict:
    raw = page.read_text(encoding="utf-8")
    code = PAYLOAD.sub("", raw)
    del raw
    head = code[:code.index("const META = ")]
    chk = json.loads((ROOT / p["checklist_file"]).read_text(encoding="utf-8"))
    items = {i["key"]: i for i in chk["items"]}
    pe = items["pleural_effusion"]
    c = p["case_id"]
    ok(c, "page names the policy, the pack, and that it is an open development page",
       f"Answer policy shown on this page: {policy}" in head and f"pack {pack}" in head and "development page, open" in head and "not blinded" in head)
    table = head[head.index('<table id="chk"'):head.index("</table>", head.index('<table id="chk"'))]
    rows = table.split("<tr")[2:]
    ok(c, "one checklist row per condition", len(rows) == len(chk["items"]), f"{len(rows)} rows")
    row = next(r for r in rows if "<b>Pleural effusion</b>" in r)
    badge = re.search(r"<span class='st ([^']+)'>([^<]*)</span>", row)
    cat = assessment_category(pe)
    ok(c, "pleural effusion badge = the checklist's display, styled by the current assessment", bool(badge) and html.unescape(badge.group(2)) == pe["display"]
       and badge.group(1) == cat.replace(" ", "-"), f"{badge.group(2) if badge else None} / class {badge.group(1) if badge else None}")
    rt = text(row)
    ok(c, "pleural effusion row never says 'none measured', 'No measurable', 'absent' or a bare 'present' badge",
       "none measured" not in rt and "No measurable" not in rt and badge.group(1) not in ("present", "absent") and badge.group(2) not in ("present", "absent", "none measured"))
    ok(c, "pleural effusion statement is the checklist's statement", pe["statement"] in rt)
    ok(c, "provisional threshold result is shown apart and labelled as not the current answer",
       "Provisional threshold result (kept separately; not the current answer)" in rt and "not validated diagnostic boundaries" in rt
       and "not  a finding of no effusion" in rt.replace("not a finding", "not  a finding") and "Tool's provisional cut-offs" not in rt)
    if cat == "suggested":
        ok(c, "positive mask-based answer reads 'suggested by segmentation'", badge.group(2) == "suggested by segmentation" and "Not confirmed" in rt)
    cons = next(r for r in rows if "<b>Consolidation</b>" in r)
    if items["consolidation"].get("dense_lung_candidates"):
        ok(c, "dense-lung candidates are shown as tagged candidates, not findings",
           "Dense-lung candidates (density-selected, tagged; not a finding)" in text(cons) and "Dense-lung candidates, by spatial tag" in text(head)
           and "candidates, not findings" in text(head))
    ok(c, "classifier described as supporting evidence only", "a classifier score is supporting evidence only" in text(head))
    shown = text(head).lower().replace("was named compressed lung", "").replace("not named compressed lung", "")   # the two sentences that say it is NOT named so
    ok(c, "dense lung next to fluid is nowhere called 'compressed lung' (only the two sentences saying it is no longer named so)", "compressed lung" not in shown)
    rep = head[head.index("<b>Report</b>"):]
    rep = text(rep[:rep.index("</ul>")])
    ok(c, "report card: pleural effusion not listed as present or as not detected",
       "Pleural effusion: present" not in rep and not re.search(r"Not detected by measurement:[^.]*Pleural effusion", rep) and "Findings: pleural effusion" not in rep)
    ok(c, "earlier review shown read-only with the policy it was made on",
       "Earlier review of this case" in head and f"made on a page showing <b>{p['earlier_review']['made_on_policy']}</b>" in head and "nothing is copied into the form" in head)
    m = re.search(r"const G = (\{.*?\});\n", code, re.S)
    G = json.loads(m.group(1))
    gi = {i["key"]: i for i in G["items"]}
    exp = {k: export_row(c, policy, items[k]) for k in items}
    ok(c, "grading data: policy and pack recorded, no earlier grades filled in", G["policy"] == {"config": policy, "pack": pack} and G["config"] == policy and G["prefill"] is None)
    ok(c, "grading data: every condition carries the current assessment category of the checklist",
       all(gi[k]["category"] == exp[k]["current_assessment"] for k in items))
    ok(c, "grading data: pleural effusion is never 'absent'; threshold result kept in its own field",
       gi["pleural_effusion"]["category"] in ("suggested", "cannot tell") and gi["pleural_effusion"]["threshold_answer"] == pe["provisional_threshold_result"]["answer"]
       and gi["pleural_effusion"]["display"] == pe["display"], f"{gi['pleural_effusion']['category']} / threshold {gi['pleural_effusion']['threshold_answer']}")
    ok(c, "grading script: own storage key per policy and pack; extra CSV columns present",
       "+ '_' + G.policy.config + '_' + G.policy.pack;" in code and all(col in code for col in POLICY_EXTRA_COLS)
       and "suggested_reviewer_present" in code and "suggested_reviewer_absent" in code)
    ok(c, "no viewer library or WebGL code in the page", "niivue" not in code.lower() and "webgl" not in code.lower())
    return {"category": cat, "display": pe["display"], "threshold": pe["provisional_threshold_result"]["answer"], "n_items": len(items), "expected": exp}


def browser_checks(page: Path, p: dict, policy: str, pack: str, st: dict, browser: str, port: int, out: Path, ok) -> None:
    from test_canvas_viewer import Browser
    from test_viewer_page import PREP
    c = p["case_id"]
    b = Browser(browser, "sw", page.resolve().as_uri(), port)
    try:
        for _ in range(240):      # the form is usable while the scan still loads
            try:
                if b.c.js("!!document.getElementById('gsave')", 10):
                    break
            except Exception:
                pass
            time.sleep(0.5)
        b.c.js(PREP)
        r = b.c.js("""(async () => {
 const rows = Array.from(document.querySelectorAll('#chk tr')).slice(1);
 const tr = rows.find(t => t.querySelector('td b').textContent === 'Pleural effusion'), td = tr.querySelector('td.grade');
 const pre = !!document.querySelector('#chk td.grade .gb button.sel');
 const out = {prefilled: pre, agree: {}, keys: Object.keys(localStorage).filter(k => k.indexOf('lungct_grades_') === 0)};
 for (const a of ['present', 'absent', 'cannot_tell']) { const btn = td.querySelector('.gb button[data-a="' + a + '"]'); btn.click();
   out.agree[a] = td.querySelector('.agrw').textContent; btn.click(); }
 const rev = document.getElementById('grev'); rev.value = window.__TEST_REVIEWER; rev.oninput();
 rows.forEach(t => { const isPe = t === tr; const w = t.querySelector('td.grade .gb button[data-a="' + (isPe ? 'absent' : 'cannot_tell') + '"]'); if (!w.classList.contains('sel')) w.click(); });
 document.getElementById('gsave').click(); await new Promise(z => setTimeout(z, 1500));
 out.csv = window.__csv; out.keys2 = Object.keys(localStorage).filter(k => k.indexOf('lungct_grades_') === 0);
 out.bar = document.querySelector('.gbar').textContent; return out; })()""", 120)
        txt = (r["csv"] or "").lstrip("﻿")
        (out / f"AUTOMATED_TEST_export_NOT_from_a_reviewer_{c}.csv").write_text(txt, encoding="utf-8", newline="")
        rows = list(csv.DictReader(io.StringIO(txt)))
        by = {x["condition_key"]: x for x in rows}
        pe, cat = by["pleural_effusion"], st["category"]
        ok(c, "browser: form starts empty (earlier grades are not filled in)", r["prefilled"] is False)
        ok(c, "browser: grades are stored under a key that names the policy and the pack", all(k.endswith(f"_{policy}_{pack}") for k in r["keys2"]) and len(r["keys2"]) == 1, str(r["keys2"]))
        ok(c, "browser: the form says which policy a saved grade belongs to", f"recorded as made on {policy} (pack {pack})" in r["bar"])
        if cat == "suggested":
            ok(c, "browser: a reviewer answer on a 'suggested' item is not scored as agreeing with / over-calling a confirmed answer",
               r["agree"]["present"] == "segmentation suggested it; reviewer: present" and r["agree"]["absent"] == "segmentation suggested it; reviewer: absent"
               and r["agree"]["cannot_tell"] == "open" and pe["agreement"] == "suggested_reviewer_absent", str(r["agree"]))
        else:
            ok(c, "browser: with the tool at 'cannot tell', every reviewer answer is scored 'open' (never 'agent missed it' / 'agrees with agent')",
               set(r["agree"].values()) == {"open"} and pe["agreement"] == "open", str(r["agree"]))
        ok(c, "browser: CSV has one row per condition plus the overall row, with the extra policy columns",
           len(rows) == st["n_items"] + 1 and all(col in rows[0] for col in POLICY_EXTRA_COLS), f"{len(rows)} rows")
        ok(c, "browser: every CSV row names the checklist version and the pack", all(x["checklist_version"] == policy and x["checklist_config"] == policy and x["pack"] == pack for x in rows))
        exp = st["expected"]
        ok(c, "browser: CSV agent_status / agent_current_assessment = the current assessment category for every condition",
           all(by[k]["agent_status"] == exp[k]["current_assessment"] == by[k]["agent_current_assessment"] and by[k]["agent_stored_status"] == exp[k]["stored_status"] for k in exp))
        ok(c, "browser: CSV pleural effusion is never exported as absent / none measured; threshold result only in its own named column",
           pe["agent_status"] in ("suggested", "cannot tell") and pe["agent_answer"] == st["display"] and "none measured" not in ",".join(pe.values())
           and pe["provisional_threshold_result_NOT_A_CURRENT_ASSESSMENT"] == st["threshold"] and pe["doctor_answer"] == "absent"
           and "provisional, not validated" in pe["provisional_limits"], f"{pe['agent_answer']} / {pe['agent_status']} / threshold column {pe['provisional_threshold_result_NOT_A_CURRENT_ASSESSMENT']}")
        ok(c, "browser: CSV reviewer is the automated-test marker", all(x["reviewer"] == "AUTOMATED TEST - not a reviewer" for x in rows))
    finally:
        b.close()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--folder", required=True)
    ap.add_argument("--browser", choices=["edge", "chrome"], default=None)
    ap.add_argument("--port", type=int, default=9397)
    args = ap.parse_args()
    folder = ROOT / args.folder
    info = json.loads((folder / "PACK_INFO.json").read_text(encoding="utf-8"))
    out = ROOT / "runs" / "test_logs" / f"{info['pack']}_policy_pack_check"
    out.mkdir(parents=True, exist_ok=True)
    res = []

    def ok(case, what, cond, detail=""):
        res.append({"case": case, "check": what, "pass": bool(cond), "detail": detail})
        print(("PASS " if cond else "FAIL ") + f"{case}: {what}" + (f"  [{detail}]" if detail else ""), flush=True)
    idx = (folder / "index.html").read_text(encoding="utf-8")
    ok("index", "index names the policy and the pack and links every page", f"answer policy {info['policy']}" in idx and all(p["file"] in idx for p in info["pages"])
       and "Not blinded" in idx)
    for k, p in enumerate(info["pages"]):
        page = folder / p["file"]
        st = static_checks(page, p, info["policy"], info["pack"], ok)
        if args.browser:
            browser_checks(page, p, info["policy"], info["pack"], st, args.browser, args.port + k, out, ok)
    n_fail = sum(not r["pass"] for r in res)
    (out / "result.json").write_text(json.dumps({"pack": info["pack"], "policy": info["policy"], "browser": args.browser, "checks": len(res), "failed": n_fail,
                                                 "results": res}, indent=1), encoding="utf-8")
    print(f"{len(res) - n_fail} of {len(res)} checks passed -> {out / 'result.json'}")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
