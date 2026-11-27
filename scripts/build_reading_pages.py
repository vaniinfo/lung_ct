"""WITHDRAWN 2026-11-27: do not use for pages sent to readers; it does not follow docs/REVIEW_PAGE_FORMAT.md (use build_viewer_page.py).

Blinded reading pages for the two-reader review RS01 (core env): CT viewer + one plain question per condition.

Each page is one self-contained HTML (file://, nothing sent anywhere) with
  left   the viewer-v6 CT display (opens on Axial 2.5 mm, lung window; MIP 10 mm; native slices under Details), no outlines
  right  the reader's name (chosen once per page), and for every condition: Present / Absent / Cannot tell, an optional location,
         and one comment box for the scan. Answers autosave in the browser; "Save answers" downloads
         reading_<page>_<scan>_<reader>_<time>.csv. A page can be read by each reader separately (answers are kept per reader name).
The page contains NO model answer, score, report text, report label or sample cell. The reader names come from --readers (command line),
so they are not written into code or configs.

Usage: .venv/Scripts/python.exe scripts/build_reading_pages.py --sample runs/RS01_review_sample_<date> --readers "Reader A" "Reader B"
Writes <sample>/pages/{page_NN_<scan>.html, index.html, PAGES.json}
"""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import nibabel as nib
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import viewer_v6 as v6  # noqa: E402

READING_BUILD = "reading_page_v1_on_" + v6.VIEWER_BUILD
CHOICES = [("present", "Present"), ("absent", "Absent"), ("cannot_tell", "Cannot tell")]

CSS = """
.page{display:flex;height:100vh}.page>.viewer{flex:1;min-width:0}
#form{width:430px;flex:none;overflow-y:auto;background:#f7f7f5;color:#1d1d1f;padding:12px 14px;font:13px/1.4 'Segoe UI',Arial,sans-serif}
#form h2{font-size:15px;margin:0 0 4px}#form .note{color:#555;font-size:12px;margin:0 0 10px}
#form select,#form input[type=text],#form textarea{font:13px 'Segoe UI',Arial,sans-serif;width:100%;padding:4px 6px;border:1px solid #bbb;border-radius:6px}
.q{border-top:1px solid #ddd;padding:7px 0}.q .lab{font-weight:600}.q .opts{display:flex;gap:12px;margin:3px 0}
.q.done .lab::after{content:' \\2713';color:#1a7f37}.q input[type=text]{margin-top:3px}
#save{background:#2f6feb;color:#fff;border:0;border-radius:8px;padding:8px 14px;font-size:14px;margin-top:10px;cursor:pointer}
#save:disabled{background:#999}#count{margin-left:8px;color:#555}#saved{color:#1a7f37;font-size:12px;margin-top:4px}
"""


def form_html(page_no: int, scan: str, conditions: list[dict], readers: list[str]) -> str:
    rows = []
    for c in conditions:
        k = html.escape(c["key"])
        opts = "".join(f"<label><input type='radio' name='{k}' value='{v}'> {t}</label>" for v, t in CHOICES)
        rows.append(f"<div class='q' data-key='{k}'><div class='lab'>{html.escape(c['label'])}</div>"
                    f"<div class='opts'>{opts}</div><input type='text' name='{k}__loc' placeholder='Location (optional), e.g. right lower lobe'></div>")
    readers_opts = "<option value=''>-- choose your name --</option>" + "".join(f"<option>{html.escape(r)}</option>" for r in readers)
    return (f"<div id='form'><h2>Page {page_no}: {html.escape(scan)}</h2>"
            "<p class='note'>For each finding, is it present on this CT? Answer from the images only. There is no report or AI answer on this page. "
            "Save when done; the file is downloaded to your computer.</p>"
            f"<label>Reader: <select id='reader'>{readers_opts}</select></label>"
            + "".join(rows)
            + "<div class='q'><div class='lab'>Comment (optional)</div><textarea id='comment' rows='3'></textarea></div>"
            "<button id='save' disabled>Save answers</button><span id='count'></span><div id='saved'></div></div>")


def form_js(page_no: int, scan: str, conditions: list[dict], build: str) -> str:
    keys = [c["key"] for c in conditions]
    labels = {c["key"]: c["label"] for c in conditions}
    return """<script>
(() => {
  const PAGE = %d, SCAN = %s, KEYS = %s, LABELS = %s, BUILD = %s;
  const $ = id => document.getElementById(id), form = $('form');
  const store = r => 'rs01_' + SCAN + '_' + r;
  function state() {
    const s = {answers: {}, locations: {}, comment: $('comment').value};
    for (const k of KEYS) {
      const c = form.querySelector('input[name="' + k + '"]:checked');
      s.answers[k] = c ? c.value : '';
      s.locations[k] = form.querySelector('input[name="' + k + '__loc"]').value;
    }
    return s;
  }
  function show() {
    const s = state(), n = KEYS.filter(k => s.answers[k]).length, r = $('reader').value;
    for (const q of form.querySelectorAll('.q[data-key]')) q.classList.toggle('done', !!s.answers[q.dataset.key]);
    $('count').textContent = n + ' of ' + KEYS.length + ' answered';
    $('save').disabled = !r || n < KEYS.length;
    if (r) localStorage.setItem(store(r), JSON.stringify(s));
  }
  function load(r) {
    const s = r && JSON.parse(localStorage.getItem(store(r)) || 'null');
    for (const k of KEYS) {
      for (const el of form.querySelectorAll('input[name="' + k + '"]')) el.checked = !!s && s.answers[k] === el.value;
      form.querySelector('input[name="' + k + '__loc"]').value = s ? (s.locations[k] || '') : '';
    }
    $('comment').value = s ? s.comment || '' : '';
    $('saved').textContent = '';
  }
  const q = v => '"' + String(v).replace(/"/g, '""').replace(/[\\r\\n]+/g, ' ') + '"';
  function csv() {
    const s = state(), r = $('reader').value, t = new Date().toISOString();
    const lines = ['page,scan,reader,condition_key,condition,answer,location,comment,saved_utc,page_build'];
    for (const k of KEYS) lines.push([PAGE, SCAN, r, k, LABELS[k], s.answers[k], s.locations[k], s.comment, t, BUILD].map(q).join(','));
    return lines.join('\\r\\n') + '\\r\\n';
  }
  function save() {
    const r = $('reader').value, stamp = new Date().toISOString().replace(/[-:]/g, '').slice(0, 15);
    const a = document.createElement('a');
    a.href = URL.createObjectURL(new Blob([csv()], {type: 'text/csv'}));
    a.download = 'reading_' + String(PAGE).padStart(2, '0') + '_' + SCAN + '_' + r.replace(/[^A-Za-z0-9]+/g, '_') + '_' + stamp + '.csv';
    document.body.appendChild(a); a.click(); a.remove();
    $('saved').textContent = 'Saved as ' + a.download + ' (in your Downloads folder).';
  }
  $('reader').addEventListener('change', () => { load($('reader').value); show(); });
  form.addEventListener('input', show);
  form.addEventListener('change', show);
  $('save').addEventListener('click', save);
  window.__reading = {state, csv, keys: KEYS};
  show();
})();
</script>""" % (page_no, json.dumps(scan), json.dumps(keys), json.dumps(labels), json.dumps(build))


def label_of(phrase: str) -> str:
    """'a lung nodule' -> 'Lung nodule'."""
    for art in ("a ", "an "):
        if phrase.startswith(art):
            phrase = phrase[len(art):]
    return phrase[0].upper() + phrase[1:]


def reading_page(page_no: int, scan: str, arr: np.ndarray, affine: np.ndarray, conditions: list[dict], readers: list[str]) -> str:
    page = v6.standalone_page(f"Reading page {page_no}", arr, affine)
    a = page.index("<div class='page'>") + len("<div class='page'>")
    b = page.index("</div>\n<script>", a)
    page = page[:b] + form_html(page_no, scan, conditions, readers) + page[b:]
    page = page.replace("</style>", CSS + "</style>", 1)
    for banned in ("score_yes", "ct_rate_label", "model_only", "report_only", "agree_present", "NV-Reason", "COLIPRI"):
        assert banned not in page, banned
    return page.replace("</body></html>", form_js(page_no, scan, conditions, READING_BUILD) + "\n</body></html>")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--sample", required=True)
    ap.add_argument("--readers", nargs="+", required=True)
    ap.add_argument("--volumes", default="D:/datasets/ct_rate/calib400/volumes")
    ap.add_argument("--prompt-sets", default="configs/nvreason_prompt_sets_v1.json")
    ap.add_argument("--cases", default="D:/datasets/ct_rate/calib400/cases.json", help="read only to check that no report text is on a page")
    ap.add_argument("--only", type=int, nargs="*", default=None, help="page numbers to build (default all)")
    args = ap.parse_args()
    sample = json.loads((ROOT / args.sample / "sample.json").read_text(encoding="utf-8"))
    cfg = json.loads((ROOT / args.prompt_sets).read_text(encoding="utf-8"))
    conditions = [{"key": c["key"], "label": label_of(c["phrase"])} for c in cfg["conditions"]]
    gt = json.loads(Path(args.cases).read_text(encoding="utf-8"))["cases"]
    out = ROOT / args.sample / "pages"
    out.mkdir(exist_ok=True)
    built = []
    for p in sample["pages"]:
        if args.only and p["page"] not in args.only:
            continue
        src = Path(args.volumes) / f"{p['scan']}.nii.gz"
        img = nib.load(str(src))
        arr = np.asanyarray(img.dataobj).astype(np.int16)
        name = f"page_{p['page']:02d}_{p['scan']}.html"
        page = reading_page(p["page"], p["scan"], arr, img.affine, conditions, args.readers)
        for field in ("findings", "impression"):            # the scan's own report text must not be on its page
            text = (gt.get(p["scan"], {}).get(field) or "").strip()
            assert not text or text[:60] not in page, f"report {field} found on page {p['page']}"
        (out / name).write_text(page, encoding="utf-8")
        built.append({"page": p["page"], "scan": p["scan"], "file": name, "bytes": (out / name).stat().st_size,
                      "ct_sha256": hashlib.sha256(src.read_bytes()).hexdigest()})
        print(f"[{p['page']:02d}] {name} {built[-1]['bytes'] / 2**20:.0f} MB", flush=True)
    links = "".join(f"<li><a href='{b['file']}'>Page {b['page']}</a></li>" for b in sorted(built, key=lambda x: x["page"]))
    (out / "index.html").write_text(
        "<!DOCTYPE html><html><head><meta charset='utf-8'><title>Reading pages</title></head><body style='font:15px Segoe UI,Arial;max-width:760px;margin:30px auto'>"
        f"<h1>Reading pages ({len(built)} scans)</h1><p>Research use only. Each page shows one CT and asks {len(conditions)} questions: is the finding present?"
        " Answer from the images; there is no report and no AI answer on the pages. Choose your name at the top of each page, answer every question, then press"
        " <b>Save answers</b>; a small CSV file is saved to your Downloads folder. Please send all CSV files back when done. Your answers are also kept in this browser"
        " if you close a page and come back.</p><p>Pages open from this folder (no internet needed). Each page is large and may take a few seconds to open.</p>"
        f"<ol>{links}</ol></body></html>", encoding="utf-8")
    (out / "PAGES.json").write_text(json.dumps({"build": READING_BUILD, "created_utc": datetime.now(timezone.utc).isoformat(), "sample": args.sample,
                                               "conditions": conditions, "readers_given_on_command_line": len(args.readers), "pages": built}, indent=1), encoding="utf-8")
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
