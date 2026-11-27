"""Pre-send checks for a policy-labelled viewer pack (core env): grades stay with their policy.

For every page of the pack, in ONE throw-away browser profile (file:// pages share browser storage, so this is the
situation on a reviewer's machine that already holds grades typed on an earlier pack):
  1. an earlier pack's page of the same case is opened (it may show the reviewer's earlier grades, as M4 does) and one
     answer is changed there, so that the browser holds grades under the earlier page's key;
  2. the pack page is opened: its form must be empty, its storage key must be new, and the earlier key must be untouched;
  3. the read-only "Earlier review" card must hold no form control, and an export made before any grade is typed must
     carry no answer, comment or name from it;
  4. every condition is graded, the CSV is captured in memory and written to <out>/import_test/ under a name that says
     it is not from a reviewer;
  5. the page is reloaded: the typed grades must come back (autosave), still under the pack's own key;
  6. the earlier pack's page is opened again: it must still show only its own grade.
Also, without a browser: the storage key of every page of every earlier pack is computed and compared with the pack's keys.
Nothing is downloaded or sent; no file outside <out> is written.

Usage: python scripts/check_policy_pack_presend.py --folder runs/M5_policy_v3_8_20261123 --earlier runs/M4_canvas_viewer_corrected_20261122 \
           --out runs/V3_M5_viewer_check_<time> --browser edge
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from test_canvas_viewer import Browser  # noqa: E402
from test_viewer_page import PREP  # noqa: E402

MARK = "AUTOMATED TEST - not a reviewer"
G_RE = re.compile(r'const G = (\{"case_id": "[^"]*", "run": "[^"]*", "config": "[^"]*".*?"blinded": (?:true|false))')
KEYS_JS = "Object.keys(localStorage).filter(k => k.indexOf('lungct_grades_') === 0).sort()"


def page_key(path: Path) -> dict | None:
    """Storage key a page's grading form uses, computed from the identifiers embedded in the page (no browser)."""
    head = ""
    with open(path, encoding="utf-8") as f:
        while True:
            chunk = f.read(1 << 22)
            if not chunk:
                break
            head += chunk
            if "const G = {" in head and "lungct_grades_" in head:
                tail = f.read(1 << 16)
                head += tail
                break
            head = head[-200000:]
    if "lungct_devadj_" in head and "const G = {" not in head:
        return {"key": "lungct_devadj_<build>", "form": "adjudication package"}
    m = re.search(r'"case_id": "([^"]*)", "run": "([^"]*)", "config": "([^"]*)"', head[head.index("const G = {"):]) if "const G = {" in head else None
    if not m:
        return None
    seg = head[head.index("const G = {"):]
    blinded = bool(re.search(r'"blinded": true', seg[:seg.index("\n")] if "\n" in seg else seg))
    pol = re.search(r'"policy": \{"config": "([^"]*)", "pack": "([^"]*)"\}', seg)
    key = f"lungct_grades_{m.group(1)}_{m.group(2)}" + ("_staged_v2" if blinded and "_staged_v2" in head else "")
    if pol and "G.policy.config" in head:
        key += f"_{pol.group(1)}_{pol.group(2)}"
    return {"key": key, "case_id": m.group(1), "config": m.group(3), "policy_pack": pol.group(2) if pol else None}


def wait_form(c, seconds=180) -> None:
    for _ in range(seconds * 2):
        try:
            if c.js("document.readyState !== 'loading' && !!document.getElementById('gsave')", 10):
                c.js(PREP)
                return
        except Exception:
            pass
        time.sleep(0.5)
    raise RuntimeError("the grading form did not appear")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--folder", required=True)
    ap.add_argument("--earlier", required=True, help="an earlier pack with a page per case (grades typed there must not appear in the new pack)")
    ap.add_argument("--other-packs", nargs="*", default=[], help="further earlier packs / packages for the static key comparison")
    ap.add_argument("--out", required=True)
    ap.add_argument("--browser", choices=["edge", "chrome"], default="edge")
    ap.add_argument("--port", type=int, default=9431)
    args = ap.parse_args()
    folder, earlier, out = ROOT / args.folder, ROOT / args.earlier, ROOT / args.out
    (out / "import_test").mkdir(parents=True, exist_ok=True)
    info = json.loads((folder / "PACK_INFO.json").read_text(encoding="utf-8"))
    policy, pack = info["policy"], info["pack"]
    res = []

    def ok(case, what, cond, detail=""):
        res.append({"case": case, "check": what, "pass": bool(cond), "detail": str(detail)[:400]})
        print(("PASS " if cond else "FAIL ") + f"{case}: {what}" + (f"  [{str(detail)[:200]}]" if detail != "" else ""), flush=True)

    # ---- static: storage keys of every page of every pack ----
    new_keys = {p["file"]: page_key(folder / p["file"]) for p in info["pages"]}
    old = {}
    for d in [earlier] + [ROOT / x for x in args.other_packs]:
        for f in sorted(d.rglob("*.html")):
            k = page_key(f)
            if k:
                old[f.relative_to(ROOT).as_posix()] = k
    (out / "storage_keys.json").write_text(json.dumps({"pack": new_keys, "earlier_pages": old}, indent=1), encoding="utf-8")
    nk = {v["key"] for v in new_keys.values() if v}
    ok("all", f"storage keys: each {pack} page has its own key naming the policy and the pack", len(nk) == len(info["pages"]) and all(k.endswith(f"_{policy}_{pack}") for k in nk), sorted(nk))
    ok("all", f"storage keys: none of the {len(old)} earlier pages (other packs, adjudication package) uses a key of the {pack} pack",
       bool(old) and not (nk & {v["key"] for v in old.values()}), f"{len({v['key'] for v in old.values()})} distinct earlier keys")

    for k, p in enumerate(info["pages"]):
        c_id = p["case_id"]
        short = c_id.split("_a_")[0]
        ep = next((f for f in sorted(earlier.glob("*.html")) if short in f.name and f.name != "index.html"), None)
        if ep is None:
            ok(c_id, "an earlier pack page exists for this case", False)
            continue
        b = Browser(args.browser, "sw", ep.resolve().as_uri(), args.port + k)
        try:
            c = b.c
            wait_form(c)
            # 1. a grade typed on the earlier page
            e1 = c.js("(() => { const td = document.querySelector('#chk td.grade'); const btn = td.querySelector('.gb button:not(.sel)'); btn.click();"
                      " return {keys: " + KEYS_JS + ", sel: document.querySelectorAll('#chk td.grade .gb button.sel').length, rev: document.getElementById('grev').value}; })()")
            ekey = e1["keys"][0] if e1["keys"] else None
            eval0 = c.js(f"localStorage.getItem({json.dumps(ekey)})") if ekey else None
            ok(c_id, "earlier page: holds grades (an answer was changed there) under the earlier key only", e1["sel"] >= 1 and len(e1["keys"]) == 1 and not ekey.endswith(f"_{policy}_{pack}")
               and bool(eval0), f"{e1['sel']} answers selected; key {ekey}")
            # 2. the pack page in the same profile
            c.call("Page.navigate", url=(folder / p["file"]).resolve().as_uri())
            time.sleep(1.5)
            wait_form(c)
            n = c.js("(() => ({keys: " + KEYS_JS + ", sel: document.querySelectorAll('#chk td.grade .gb button.sel').length, rev: document.getElementById('grev').value,"
                     " rows: document.querySelectorAll('#chk tr').length - 1, count: document.getElementById('gcount').textContent,"
                     " own: (() => { const k = " + KEYS_JS + ".filter(x => x.endsWith('_" + f"{policy}_{pack}" + "'))[0]; if (!k) return null; const st = JSON.parse(localStorage.getItem(k) || '{}');"
                     " return {reviewer: st.reviewer || '', answers: Object.values(st.g || {}).filter(x => x && x.answer).length}; })(),"
                     " ecard: (() => { const e = document.querySelector('.earlier'); if (!e) return null; return {controls: e.querySelectorAll('button, input, select, textarea, [contenteditable]').length,"
                     " inChk: !!e.closest('#chk'), gradeCells: e.querySelectorAll('td.grade, .gb').length, open: !!e.querySelector('details[open]'), text: e.textContent.slice(0, 4000),"
                     " rows: e.querySelectorAll('table tr').length - 1}; })()}))()")
            ok(c_id, f"{pack} page opened in a browser that holds the earlier grade: storage is shared (the earlier key is visible), yet the form is empty",
               ekey in n["keys"] and n["sel"] == 0 and n["rev"] == "" and n["count"].startswith("0 of") and (n["own"] is None or n["own"] == {"reviewer": "", "answers": 0}),
               f"{n['count']}; earlier key visible: {ekey in n['keys']}; the page's own key holds {n['own']}")
            ec = n["ecard"]
            ok(c_id, "'Earlier review' card: present, collapsed, read-only (no button, input or grade cell), outside the checklist table",
               bool(ec) and ec["controls"] == 0 and ec["gradeCells"] == 0 and not ec["inChk"] and not ec["open"] and ec["rows"] > 0, f"{ec['rows'] if ec else 0} earlier rows shown")
            er = list(csv.DictReader(open(ROOT / p["earlier_review"]["file"], newline="", encoding="utf-8-sig")))
            ok(c_id, "'Earlier review' card names the policy those grades were made on", bool(ec) and p["earlier_review"]["made_on_policy"] in ec["text"] and policy not in ec["text"].split("Shown for reference")[0])
            # 3. export before any grade is typed: nothing from the earlier review may be in it
            x0 = c.js("(async () => { const rev = document.getElementById('grev'); rev.value = '" + MARK + "'; rev.oninput(); document.getElementById('gsave').click(); await new Promise(z => setTimeout(z, 1500));"
                      " return {csv: window.__csv, alerts: window.__alerts}; })()", 60)
            t0 = (x0["csv"] or "").lstrip("﻿")
            r0 = list(csv.DictReader(io.StringIO(t0))) if t0 else []
            ecomments = [r["comment"].strip() for r in er if (r.get("comment") or "").strip()]
            ename = er[0]["reviewer"].strip()
            if r0:
                ok(c_id, "export made before grading carries no answer, severity, comment, name or date of the earlier review",
                   all(not r["doctor_answer"] and not r["severity"] and not r["comment"] and not r["agreement"] for r in r0) and ename not in t0
                   and not any(cm in t0 for cm in ecomments if len(cm) > 8) and er[0]["saved_utc"] not in t0, f"{len(r0)} rows, all without a reviewer answer")
            else:
                ok(c_id, "export before grading: the form refuses to save an empty form (nothing from the earlier review can be exported)", bool(x0["alerts"]), str(x0["alerts"])[:150])
            # 4. grade everything, export
            g = c.js("(async () => { const rows = Array.from(document.querySelectorAll('#chk tr')).slice(1); rows.forEach((tr, i) => { const td = tr.querySelector('td.grade');"
                     " const pe = tr.querySelector('td b') && tr.querySelector('td b').textContent === 'Pleural effusion';"
                     " const w = td.querySelector('.gb button[data-a=\"' + (pe ? 'present' : ['present', 'absent', 'cannot_tell'][i % 3]) + '\"]'); if (!w.classList.contains('sel')) w.click(); });"
                     " const oc = document.getElementById('gcase'); oc.value = 'automated test comment'; oc.oninput && oc.oninput(); oc.dispatchEvent(new Event('input', {bubbles: true}));"
                     " window.__csv = null; document.getElementById('gsave').click(); await new Promise(z => setTimeout(z, 1500));"
                     " return {csv: window.__csv, keys: " + KEYS_JS + ", count: document.getElementById('gcount').textContent, rows: rows.length,"
                     " answers: rows.map(tr => (tr.querySelector('td.grade .gb button.sel') || {dataset: {}}).dataset.a || '')}; })()", 120)
            txt = (g["csv"] or "").lstrip("﻿")
            f_out = out / "import_test" / f"grades_{c_id}_AUTOMATED_TEST_NOT_from_a_reviewer.csv"
            f_out.write_text(txt, encoding="utf-8", newline="")
            rows = list(csv.DictReader(io.StringIO(txt)))
            nkey = [x for x in g["keys"] if x != ekey]
            ok(c_id, "all conditions graded; grades stored under the pack's own key next to the untouched earlier key",
               g["count"].startswith(f"{g['rows']} of {g['rows']}") and len(nkey) == 1 and nkey[0] == new_keys[p["file"]]["key"] and ekey in g["keys"], str(g["keys"]))
            ok(c_id, "CSV: one row per condition plus the overall row; every row names the policy and the pack; reviewer is the test marker; earlier reviewer's name absent",
               len(rows) == g["rows"] + 1 and all(r["checklist_version"] == policy and r["pack"] == pack and r["reviewer"] == MARK for r in rows) and ename not in txt)
            ok(c_id, "CSV: the answers are the ones clicked in this form, in order", [r["doctor_answer"] for r in rows[:-1]] == g["answers"])
            # 5. reload: autosaved grades come back
            c.call("Page.reload")
            time.sleep(1.5)
            wait_form(c)
            rl = c.js("(() => { const rows = Array.from(document.querySelectorAll('#chk tr')).slice(1); return {answers: rows.map(tr => (tr.querySelector('td.grade .gb button.sel') || {dataset: {}}).dataset.a || ''),"
                      " rev: document.getElementById('grev').value, overall: document.getElementById('gcase').value, count: document.getElementById('gcount').textContent, keys: " + KEYS_JS + "}; })()")
            ok(c_id, "reload: answers, reviewer name and overall comment come back from the browser's own storage",
               rl["answers"] == g["answers"] and rl["rev"] == MARK and rl["count"].startswith(f"{g['rows']} of {g['rows']}"), f"{rl['count']}; overall comment restored: {rl['overall'] == 'automated test comment'}")
            # 6. back to the earlier page: still only its own grade
            c.call("Page.navigate", url=ep.resolve().as_uri())
            time.sleep(1.5)
            wait_form(c)
            e2 = c.js("(() => ({sel: document.querySelectorAll('#chk td.grade .gb button.sel').length, val: localStorage.getItem(" + json.dumps(ekey) + "), rev: document.getElementById('grev').value}))()")
            ok(c_id, "earlier page afterwards: the same answers and reviewer name as before, stored value unchanged (the new pack wrote nothing into it)",
               e2["sel"] == e1["sel"] and e2["val"] == eval0 and e2["rev"] == e1["rev"], f"{e2['sel']} answers selected")
        except Exception as e:  # noqa: BLE001  reported, never dropped
            ok(c_id, "browser run completed", False, f"{type(e).__name__}: {e}")
        finally:
            b.close()
    n_fail = sum(not r["pass"] for r in res)
    (out / "presend_result.json").write_text(json.dumps({"pack": pack, "policy": policy, "earlier_pack": args.earlier, "browser": args.browser, "checks": len(res),
                                                         "failed": n_fail, "results": res}, indent=1), encoding="utf-8")
    print(f"{len(res) - n_fail} of {len(res)} checks passed -> {out / 'presend_result.json'}")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
