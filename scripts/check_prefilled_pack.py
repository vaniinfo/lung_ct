"""Browser check of a policy pack whose form is filled with the reviewer's earlier grades (pack M6) (core env).

Per page, in a throw-away browser profile (nothing is downloaded or sent; only <out> is written):
  1. the form shows the earlier grades: same answer per condition as the authenticated earlier CSV, and the page says so;
  2. the CT-RATE report label column is visible on every checklist row (no earlier-grade column);
  3. an export made without touching anything marks every answered row 'carried_over_from_earlier_grade_unchanged'
     and names the policy and date of the earlier grade; policy and pack columns name this pack;
  4. after one answer is changed, exactly that row is exported as 'changed_on_this_page';
  5. a screenshot of the checklist is saved.
Usage: python scripts/check_prefilled_pack.py --folder runs/M6_policy_v3_8_prefilled_20261123 --out runs/V4_M6_viewer_check_<time> --browser edge
"""
from __future__ import annotations

import argparse
import base64
import csv
import io
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from check_policy_pack_presend import wait_form  # noqa: E402
from test_canvas_viewer import Browser  # noqa: E402

MARK = "AUTOMATED TEST - not a reviewer"


def parse(text: str) -> list[dict]:
    return list(csv.DictReader(io.StringIO((text or "").lstrip("﻿"))))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--folder", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--browser", choices=["edge", "chrome"], default="edge")
    ap.add_argument("--port", type=int, default=9471)
    args = ap.parse_args()
    folder, out = ROOT / args.folder, ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    info = json.loads((folder / "PACK_INFO.json").read_text(encoding="utf-8"))
    res = []

    def ok(case, what, cond, detail=""):
        res.append({"case": case, "check": what, "pass": bool(cond), "detail": str(detail)[:500]})
        print(("PASS " if cond else "FAIL ") + f"{case}: {what}" + (f"  [{str(detail)[:220]}]" if detail != "" else ""), flush=True)

    for k, p in enumerate(info["pages"]):
        cid = p["case_id"]
        with open(ROOT / p["earlier_review"]["file"], newline="", encoding="utf-8-sig") as f:
            er = [r for r in csv.DictReader(f) if r["condition_key"] != "_overall"]
        want = {r["condition_key"]: r["doctor_answer"] for r in er}
        b = Browser(args.browser, "sw", (folder / p["file"]).resolve().as_uri(), args.port + k)
        try:
            c = b.c
            wait_form(c)
            s = c.js("(() => { const rows = [...document.querySelectorAll('#chk tr')].filter(r => r.querySelector('td.grade'));"
                     " const vis = e => !!e && e.offsetParent !== null && e.getBoundingClientRect().width > 20;"
                     " return {n: rows.length, sel: rows.map(r => { const x = r.querySelector('td.grade .gb button.sel'); return x ? x.textContent.trim() : ''; }),"
                     "  ref: rows.map(r => [...r.querySelectorAll('td.refcol')].map(t => vis(t) ? t.innerText.replace(/\\n/g, ' / ') : 'HIDDEN')),"
                     "  heads: [...document.querySelectorAll('#chk th.refcol')].map(t => vis(t) ? t.innerText.replace(/\\n/g, ' / ') : 'HIDDEN'),"
                     "  note: document.querySelector('.gbar').innerText, rev: document.getElementById('grev').value,"
                     "  key: Object.keys(localStorage).filter(k => k.indexOf('lungct_grades_') === 0)}; })()")
            lab = {"present": "Present", "absent": "Absent", "cannot_tell": "Can't tell", "": ""}
            exp = [lab[want[r["condition_key"]]] for r in er]
            ok(cid, "form shows the earlier grade on every condition", s["n"] == len(er) and s["sel"] == exp, f"{sum(bool(x) for x in s['sel'])} of {s['n']} filled")
            ok(cid, "page says the grades were filled in from the earlier review and names its policy",
               "filled in" in s["note"].lower() and p["earlier_review"]["made_on_policy"] in s["note"] and "check each grade" in s["note"], s["note"][-260:])
            ok(cid, "reviewer name comes from the earlier file", s["rev"] == er[0]["reviewer"], s["rev"])
            ok(cid, "storage key names the policy and this pack", len(s["key"]) == 1 and s["key"][0].endswith(f"_{info['policy']}_{info['pack']}"), s["key"])
            ok(cid, "the report-label column is visible on every row, with its header; no earlier-grade column", len(s["heads"]) == 1 and "HIDDEN" not in s["heads"]
               and all(len(r) == 1 and "HIDDEN" not in r for r in s["ref"]) and "earlier grade" not in " ".join(s["heads"]), s["heads"])
            ok(cid, "header carries the caution", "not mentioned" in s["heads"][0], s["heads"])
            c.call("Emulation.setDeviceMetricsOverride", width=1900, height=1400, deviceScaleFactor=1, mobile=False)
            time.sleep(1.0)
            (out / f"{cid}_page.png").write_bytes(base64.b64decode(c.call("Page.captureScreenshot", format="png")["data"]))
            # export untouched
            x = c.js("(async () => { window.__csv = null; document.getElementById('gsave').click(); await new Promise(z => setTimeout(z, 1500)); return window.__csv; })()", 60)
            rows = [r for r in parse(x) if r["condition_key"] != "_overall"]
            ans = [r for r in rows if r["doctor_answer"]]
            ok(cid, "untouched export: every answered row is marked carried over unchanged, with policy and date of the earlier grade",
               len(rows) == len(er) and ans and all(r["answer_origin"] == "carried_over_from_earlier_grade_unchanged" and r["carried_from_policy"] == p["earlier_review"]["made_on_policy"]
                                                      and r["carried_from_saved_utc"] == er[0]["saved_utc"] for r in ans)
               and all(r["answer_origin"] == "" for r in rows if not r["doctor_answer"]), f"{len(ans)} answered rows")
            ok(cid, "untouched export: answers equal the earlier file; policy and pack columns name this pack",
               {r["condition_key"]: r["doctor_answer"] for r in rows} == want and all(r["checklist_version"] == info["policy"] and r["pack"] == info["pack"] for r in rows))
            pe = next(r for r in rows if r["condition_key"] == "pleural_effusion")
            ok(cid, "pleural effusion row is never exported as a confirmed tool answer from a mask", pe["agent_status"] in ("suggested", "cannot tell", "cannot assess"),
               f"{pe['agent_status']}; reviewer {pe['doctor_answer']}; agreement {pe['agreement']}")
            (out / f"{cid}_export_untouched_AUTOMATED_TEST.csv").write_text(x.lstrip("﻿"), encoding="utf-8", newline="")
            # change one answer
            ch = c.js("(async () => { const td = document.querySelector('#chk td.grade'); const key = td.closest('tr').querySelector('td b').textContent;"
                      " td.querySelector('.gb button:not(.sel)').click(); window.__csv = null; document.getElementById('gsave').click();"
                      " await new Promise(z => setTimeout(z, 1500)); return {csv: window.__csv, label: key}; })()", 60)
            rows2 = [r for r in parse(ch["csv"]) if r["condition_key"] != "_overall"]
            changed = [r["condition"] for r in rows2 if r["answer_origin"] == "changed_on_this_page"]
            ok(cid, "after one answer is changed, exactly that row is exported as changed on this page", changed == [ch["label"]]
               and sum(r["answer_origin"] == "carried_over_from_earlier_grade_unchanged" for r in rows2) == len(ans) - 1, changed)
        except Exception as e:  # noqa: BLE001
            ok(cid, "check ran", False, f"{type(e).__name__}: {e}")
        finally:
            b.close()
    summ = {"pack": info["pack"], "folder": args.folder, "browser": args.browser, "checks": len(res), "passed": sum(r["pass"] for r in res), "results": res}
    (out / f"prefilled_pack_check_{args.browser}.json").write_text(json.dumps(summ, indent=1), encoding="utf-8")
    print(f"{summ['passed']} of {summ['checks']} checks passed")
    return 0 if summ["passed"] == summ["checks"] else 1


if __name__ == "__main__":
    sys.exit(main())
