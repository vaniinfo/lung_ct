"""Technical checks of one RS01 reading page from file:// in a real browser (core env). Not clinician usability feedback.

Checks: the CT viewer opens (Axial 2.5 mm, ready, no error); the form lists every condition; Save stays disabled until a reader is chosen and
every question is answered; the CSV holds one row per condition with the chosen answers, location and comment; answers are kept per reader
after a reload; nothing on the page names a model, a score or a sample cell.

Usage: .venv/Scripts/python.exe scripts/test_reading_page.py --page runs/RS01_review_sample_<date>/pages/page_01_<scan>.html
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from test_canvas_viewer import Browser, wait_ready  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--page", required=True, type=Path)
    ap.add_argument("--browser", default="edge")
    ap.add_argument("--port", type=int, default=9631)
    args = ap.parse_args()
    page = args.page.resolve()
    res, ok_all = [], True

    def ok(name, cond, detail=None):
        nonlocal ok_all
        ok_all &= bool(cond)
        res.append({"name": name, "pass": bool(cond), "detail": detail})
        print(("PASS " if cond else "FAIL ") + name + (f"  [{str(detail)[:200]}]" if detail is not None else ""), flush=True)

    b = Browser(args.browser, "igpu", page.as_uri(), args.port)
    try:
        _, st = wait_ready(b)
        ok("viewer ready without error", st.get("ready") and not st.get("err"), st)
        js = b.c.js
        keys = js("window.__reading.keys", 10)
        ok("form lists 18 conditions", len(keys) == 18, len(keys))
        ok("save disabled before a reader is chosen", js("document.getElementById('save').disabled", 10))
        js("(() => { const s = document.getElementById('reader'); s.selectedIndex = 1; s.dispatchEvent(new Event('change')); })()", 10)
        reader = js("document.getElementById('reader').value", 10)
        ok("reader names offered", reader not in ("", None), reader)
        js("(() => { for (const k of window.__reading.keys.slice(0, 17)) { const el = document.querySelector('input[name=\"' + k + '\"][value=\"absent\"]');"
           " el.checked = true; el.dispatchEvent(new Event('change', {bubbles: true})); } })()", 10)
        ok("save disabled with one question open", js("document.getElementById('save').disabled", 10))
        last = keys[-1]
        js(f"(() => {{ const el = document.querySelector('input[name=\"{last}\"][value=\"present\"]'); el.checked = true; el.dispatchEvent(new Event('change', {{bubbles: true}}));"
           f" const l = document.querySelector('input[name=\"{last}__loc\"]'); l.value = 'right lower lobe, \"near\" fissure'; l.dispatchEvent(new Event('input', {{bubbles: true}}));"
           " const c = document.getElementById('comment'); c.value = 'test, comment'; c.dispatchEvent(new Event('input', {bubbles: true})); })()", 10)
        ok("save enabled when all answered", not js("document.getElementById('save').disabled", 10))
        rows = list(csv.DictReader(io.StringIO(js("window.__reading.csv()", 10))))
        ok("CSV: one row per condition", len(rows) == 18, len(rows))
        ok("CSV: answers, location, comment, reader", rows[-1]["answer"] == "present" and rows[-1]["location"] == 'right lower lobe, "near" fissure'
           and rows[0]["answer"] == "absent" and rows[0]["comment"] == "test, comment" and rows[0]["reader"] == reader, rows[-1])
        b.c.call("Page.reload")
        time.sleep(1)
        _, st = wait_ready(b)
        js("(() => { const s = document.getElementById('reader'); s.selectedIndex = 1; s.dispatchEvent(new Event('change')); })()", 10)
        ok("answers kept after reload for the same reader", js("window.__reading.state().answers", 10)[last] == "present")
        js("(() => { const s = document.getElementById('reader'); s.selectedIndex = 2; s.dispatchEvent(new Event('change')); })()", 10)
        ok("other reader starts empty", all(v == "" for v in js("window.__reading.state().answers", 10).values()))
        text = page.read_text(encoding="utf-8")
        ok("no model, score or cell named on the page", not any(w in text for w in ("NV-Reason", "COLIPRI", "score_yes", "model_only", "report_only")))
    finally:
        js_clear = "localStorage.clear()"
        try:
            b.c.js(js_clear, 10)
        except Exception:
            pass
        b.close()
    out = ROOT / "runs" / "test_logs" / f"reading_page_test_{page.stem}.json"
    out.write_text(json.dumps({"page": str(page), "passed": ok_all, "checks": res, "kind": "technical checks only"}, indent=1), encoding="utf-8")
    print(("PASSED" if ok_all else "FAILED") + f" ({sum(r['pass'] for r in res)} of {len(res)}) -> {out}")
    return 0 if ok_all else 1


if __name__ == "__main__":
    raise SystemExit(main())
