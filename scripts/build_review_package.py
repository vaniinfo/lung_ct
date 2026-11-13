"""Clinician review package (core env): blinded view, comparison view, airway landmark sheet.

Reads finished runs only - no model inference, no prompt or measurement changes.

Output runs/P6_clinician_review_v1_<ts>/:
  README_reviewer.md                 workflow, definitions, caveats
  data_dictionary.csv                every scoring column with allowed values
  view1_independent/                 BLINDED: full CT only (no model output, annotation, flags)
    index.html, R##/ct_viewer.html, R##/lung/*.jpg, R##/mediastinal/*.jpg, R##/R##_ct.nii.gz
    scoring_view1_independent.csv    blank
  view2_comparison/                  open only after View 1 is complete
    index.html (flagged cases highlighted), R##/comparison.html,
    R##/annotation_viewer.html, R##/annotation/*.jpg
    scoring_view2_comparison.csv     blank
    unblinding_key.csv               review code <-> case id, CT SHA-256
  model_input_contact_sheets/R##_model_input.png   exact 85 model-input slices
  central_airway_review/             landmark review for selected cases
    README.md, central_airway_review.csv (blank judgements), <case>_landmarks.png/.html
  package_manifest.json

Usage:
    python scripts/build_review_package.py --report-run runs/P4_report_20cases_<ts> \\
        --qc-run runs/P1_qc_20cases_<ts> --seg-run runs/P2_seg_10cases_<ts> \\
        --airway-run runs/P5_airway_10cases_<ts> --sensitivity-run runs/P5_sensitivity_10cases_<ts>
"""
import argparse
import csv
import hashlib
import html
import io
import json
import random
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import nibabel as nib  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from lung_ct.airway.sensitivity import CENTRAL_METRICS, central_exceptions  # noqa: E402
from lung_ct.airway.tree import METHOD, analyze_airway_tree  # noqa: E402
from lung_ct.reporting.review_flags import load_rules  # noqa: E402

SEED = 20261108
WINDOWS = {"lung": (-1350.0, 150.0), "mediastinal": (-160.0, 240.0)}   # W1500/L-600, W400/L40
INFECTION_DIR = ROOT / "data/raw/covid_ct/Infection_Mask"
AIRWAY_MASK = "lung_vessels/lung_airways.nii.gz"

V1_COLS = ["review_code", "assessability", "opacity_presence", "laterality",
           "craniocaudal_distribution", "axial_distribution", "other_findings",
           "image_quality_comments", "comments", "reviewer_id", "review_date", "minutes_spent"]
V2_ITEMS = ["assessability", "opacity_presence", "laterality", "craniocaudal_distribution",
            "axial_distribution", "description", "limitations", "unsupported_claims",
            "omissions", "overall"]
V2_COLS = ["review_code", "case_id", "automated_flags", "item", "model_output",
           "reference_annotation", "agreement", "error_type", "comments", "reviewer_id",
           "review_date"]
AIR_COLS = ["case_id", "selection_rationale", "segment", "reported_length_mm",
            "parameter_stability", "length_range_across_settings_mm", "proximal_point_ras_mm",
            "distal_point_ras_mm", "algorithm_endpoint_definition", "intended_proximal_landmark",
            "intended_distal_landmark", "proximal_landmark_correct", "distal_landmark_correct",
            "centreline_within_lumen", "length_plausible", "comments", "reviewer_id",
            "review_date"]
DICTIONARY = [
    # sheet, column, allowed values / meaning
    ("view1", "review_code", "neutral code; do not attempt to identify the case"),
    ("view1", "assessability", "assessable | limited | not_assessable"),
    ("view1", "opacity_presence", "present | absent | indeterminate (pulmonary opacities)"),
    ("view1", "laterality", "bilateral | right | left | not_applicable | indeterminate"),
    ("view1", "craniocaudal_distribution", "upper_predominant | middle_predominant | "
     "lower_predominant | no_predominance | not_applicable | indeterminate"),
    ("view1", "axial_distribution", "peripheral | central | mixed | not_applicable | indeterminate"),
    ("view1", "other_findings", "free text"),
    ("view1", "image_quality_comments", "free text (coverage, motion, noise, reconstruction)"),
    ("view1", "comments / reviewer_id / review_date / minutes_spent", "free text / initials / "
     "YYYY-MM-DD / integer"),
    ("view2", "item", "row per item: " + ", ".join(V2_ITEMS)),
    ("view2", "model_output", "pre-filled: the model's value or text for that item"),
    ("view2", "reference_annotation", "pre-filled: what the supplied infection annotation "
     "shows, where applicable (not an exhaustive opacity annotation)"),
    ("view2", "agreement", "agree | partially_agree | disagree | cannot_judge (your judgement "
     "of the model output against the images)"),
    ("view2", "error_type", "none | incorrect_finding | incorrect_laterality | "
     "incorrect_distribution | unsupported_claim | omission | incorrect_input_description | "
     "other"),
    ("view2", "unsupported_claims row", "comments: list statements not supported by the images"),
    ("view2", "omissions row", "comments: list visible findings the model did not report"),
    ("airway", "proximal_landmark_correct / distal_landmark_correct / centreline_within_lumen "
     "/ length_plausible", "yes | no | unsure"),
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_state() -> str:
    c = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
                       text=True).stdout.strip()
    d = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, capture_output=True,
                       text=True).stdout.strip()
    return c + ("+dirty" if d else "")


def e(x) -> str:
    return html.escape("" if x is None else str(x))


def write_csv(path: Path, rows: list[dict], cols: list[str]):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)


# ---------------------------------------------------------------- image rendering ----
def axial_display(slc: np.ndarray) -> np.ndarray:
    """Canonical-RAS axial slice [x, y] -> display [row, col]: anterior top, patient right
    on screen left (radiological)."""
    return slc.T[::-1, ::-1]


def window_u8(a: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return (np.clip((a - lo) / (hi - lo), 0, 1) * 255).round().astype(np.uint8)


def save_jpg(arr_u8: np.ndarray, path: Path, aspect: float):
    im = Image.fromarray(arr_u8)
    if abs(aspect - 1.0) > 1e-3:          # rows/cols spacing ratio -> square display pixels
        im = im.resize((im.width, max(1, round(im.height * aspect))), Image.BILINEAR)
    im.save(path, quality=90)


def render_stack(ct_canon, out_dir: Path, kinds: dict) -> list[dict]:
    """Write one JPEG per axial slice (superior -> inferior) for each kind.
    kinds: name -> callable(hu_slice, k) -> uint8 (H, W[, 3]) in display orientation."""
    data = np.asanyarray(ct_canon.dataobj)
    sp = np.sqrt((ct_canon.affine[:3, :3] ** 2).sum(axis=0))
    aspect = sp[1] / sp[0]                 # display rows ~ y, cols ~ x
    slices = []
    for n, k in enumerate(range(data.shape[2] - 1, -1, -1), 1):
        z = float((ct_canon.affine @ np.array([0, 0, k, 1.0]))[2])
        for name, fn in kinds.items():
            d = out_dir / name
            d.mkdir(parents=True, exist_ok=True)
            save_jpg(fn(data[:, :, k].astype(np.float32), k), d / f"{n:04d}.jpg", aspect)
        slices.append({"n": n, "z_mm": round(z, 1)})
    return slices


VIEWER_JS = """
<script>
const S=__SLICES__, SETS=__SETS__; let i=Math.floor(S.length/2), set=SETS[0][0];
function show(){document.getElementById('im').src=set+'/'+String(S[i].n).padStart(4,'0')+'.jpg';
 document.getElementById('pos').textContent=(i+1)+' / '+S.length+'   z = '+S[i].z_mm+' mm';
 document.getElementById('sl').value=i;}
function go(d){i=Math.max(0,Math.min(S.length-1,i+d));show();}
function setw(s){set=s;document.querySelectorAll('.wb').forEach(b=>b.classList.toggle('on',b.dataset.s==s));show();}
document.addEventListener('keydown',ev=>{if(ev.key=='ArrowUp'||ev.key=='ArrowLeft')go(-1);
 if(ev.key=='ArrowDown'||ev.key=='ArrowRight')go(1);if(ev.key=='PageUp')go(-10);if(ev.key=='PageDown')go(10);});
window.onload=()=>{document.getElementById('box').addEventListener('wheel',ev=>{ev.preventDefault();go(ev.deltaY>0?1:-1);},{passive:false});
 document.getElementById('sl').max=S.length-1;document.getElementById('sl').oninput=ev=>{i=+ev.target.value;show();};setw(set);};
</script>"""

VIEWER_CSS = """<style>body{background:#111;color:#ddd;font-family:Segoe UI,Arial;margin:12px}
#box{position:relative;display:inline-block}#im{width:640px;image-rendering:auto;display:block}
.lab{position:absolute;color:#ff0;font-weight:bold;font-size:18px}
.wb{background:#333;color:#ddd;border:1px solid #666;padding:4px 10px;margin-right:4px;cursor:pointer}
.wb.on{background:#2e7d32}#sl{width:640px}a{color:#8cf}small{color:#aaa}</style>"""


def viewer_html(title: str, slices: list[dict], sets: list[tuple[str, str]], note: str,
                back: str) -> str:
    js = VIEWER_JS.replace("__SLICES__", json.dumps(slices)).replace(
        "__SETS__", json.dumps(sets))
    buttons = "".join(f'<button class="wb" data-s="{e(s)}" onclick="setw(\'{e(s)}\')">'
                      f'{e(label)}</button>' for s, label in sets)
    return (f"<!DOCTYPE html><html><head><meta charset='utf-8'><title>{e(title)}</title>"
            f"{VIEWER_CSS}{js}</head><body><h3>{e(title)}</h3>"
            f"<p><a href='{e(back)}'>&larr; index</a> &nbsp; {buttons} &nbsp; <span id='pos'></span></p>"
            "<div id='box'><img id='im'><span class='lab' style='left:6px;top:48%'>R</span>"
            "<span class='lab' style='right:6px;top:48%'>L</span>"
            "<span class='lab' style='left:48%;top:4px'>A</span>"
            "<span class='lab' style='left:48%;bottom:4px'>P</span></div><br>"
            "<input type='range' id='sl' min='0' value='0'><br>"
            "<small>Scroll: mouse wheel, arrow keys, PageUp/PageDown. Axial slices of the full "
            "examination, superior to inferior, radiological display (patient right on screen "
            f"left).</small><p><small>{e(note)}</small></p></body></html>")


# ---------------------------------------------------------- central airway figures ----
SEG_COLORS = {"trachea": "#00bcd4", "right_main": "#ff9800", "left_main": "#e040fb",
              "left_main_alt": "#76ff03"}


def central_branches(res) -> dict:
    root = [r for r in res.branches if r["kind"] == "root"][0]
    g1 = {r["side"]: r for r in res.branches if r["generation"] == 1}
    out = {"trachea": root}
    if "right" in g1:
        out["right_main"] = g1["right"]
    if "left" in g1:
        out["left_main"] = g1["left"]
    return out


def airway_figure(ct_img, segs: dict, paths: dict, out: Path, title: str):
    """Coronal CT minIP slab with the centrelines + axial crops at each endpoint."""
    ct = nib.as_closest_canonical(ct_img)
    data = np.asanyarray(ct.dataobj).astype(np.float32)
    sp = np.sqrt((ct.affine[:3, :3] ** 2).sum(axis=0))
    inv = np.linalg.inv(ct.affine)

    def vox(p):
        return (inv @ np.append(p, 1.0))[:3]

    allpts = np.vstack([vox(p) for pts in paths.values() for p in pts])
    j0 = max(int(allpts[:, 1].min() - 8 / sp[1]), 0)
    j1 = min(int(allpts[:, 1].max() + 8 / sp[1]) + 1, data.shape[1])
    minip = data[:, j0:j1, :].min(axis=1)
    I, K = minip.shape
    # endpoints to inspect: label -> (world point, description)
    pts = {}
    if "trachea" in segs:
        pts["T0 trachea proximal"] = segs["trachea"]["proximal_ras_mm"]
        pts["C trachea distal (carina?)"] = segs["trachea"]["distal_ras_mm"]
    if "right_main" in segs:
        pts["R right main distal"] = segs["right_main"]["distal_ras_mm"]
    if "left_main" in segs:
        pts["L left main distal"] = segs["left_main"]["distal_ras_mm"]
    if "left_main_alt" in segs:
        pts["L' left main distal (alt. setting)"] = segs["left_main_alt"]["distal_ras_mm"]

    n_ax = len(pts)
    fig = plt.figure(figsize=(4 * (n_ax + 2), 6.5))
    gs = fig.add_gridspec(1, n_ax + 2)
    ax = fig.add_subplot(gs[0, :2])
    ax.imshow(window_u8(minip.T[::-1, ::-1], -1000, -500), cmap="gray",
              aspect=sp[2] / sp[0])
    for name in sorted(paths, key=lambda n: 0 if n.endswith("_alt") else 1):  # alt underneath
        v = np.array([vox(q) for q in paths[name]])
        alt = name.endswith("_alt")
        ax.plot(I - 1 - v[:, 0], K - 1 - v[:, 2], color=SEG_COLORS[name], lw=3.0 if alt else 1.4,
                ls="--" if alt else "-", alpha=0.8 if alt else 1.0, label=name)
    for lab, p in pts.items():
        v = vox(np.array(p))
        ax.plot(I - 1 - v[0], K - 1 - v[2], "o", ms=7, mfc="none", mec="yellow", mew=1.5)
        ax.text(I - 1 - v[0] + 4, K - 1 - v[2], lab.split()[0], color="yellow", fontsize=9,
                weight="bold")
    ax.legend(fontsize=7, loc="lower right")
    ax.set_title(f"coronal minIP slab (A-P voxels {j0}-{j1}), W -1000..-500 HU\n"
                 "R on screen left", fontsize=8)
    ax.axis("off")
    half = int(round(40 / sp[0]))
    for n, (lab, p) in enumerate(pts.items()):
        v = vox(np.array(p))
        i, j, k = (int(round(c)) for c in v)
        k = int(np.clip(k, 0, data.shape[2] - 1))
        disp = axial_display(data[:, :, k])
        r, c = data.shape[1] - 1 - j, data.shape[0] - 1 - i
        r0, c0 = max(r - half, 0), max(c - half, 0)
        crop = disp[r0:r + half, c0:c + half]
        a2 = fig.add_subplot(gs[0, n + 2])
        a2.imshow(window_u8(crop, -1350, 150), cmap="gray")
        a2.axhline(r - r0, color="yellow", lw=0.6, alpha=0.7)
        a2.axvline(c - c0, color="yellow", lw=0.6, alpha=0.7)
        for name, pp in paths.items():     # centreline points within +-1 slice
            vv = np.array([vox(q) for q in pp])
            near = np.abs(vv[:, 2] - k) <= 1
            if near.any():
                a2.plot(data.shape[0] - 1 - vv[near, 0] - c0, data.shape[1] - 1 - vv[near, 1] - r0,
                        ".", ms=3, color=SEG_COLORS[name])
        z = float((ct.affine @ np.array([0, 0, k, 1.0]))[2])
        a2.set_title(f"{lab}\naxial z={z:.1f} mm, 80x80 mm crop", fontsize=8)
        a2.axis("off")
    fig.suptitle(f"{title}\n{METHOD}", fontsize=9)
    fig.savefig(out, dpi=95, bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------------------------ main ----
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    for a in ("report", "qc", "seg", "airway", "sensitivity"):
        ap.add_argument(f"--{a}-run", required=True, type=Path)
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()

    rep_dir = ROOT / args.report_run
    qc = json.loads((ROOT / args.qc_run / "run_manifest.json").read_text())
    seg = {c["case_id"]: c for c in json.loads(
        (ROOT / args.seg_run / "run_manifest.json").read_text())["cases"]}
    with open(rep_dir / "review_targets.csv", newline="") as f:
        targets = list(csv.DictReader(f))
    with open(ROOT / args.sensitivity_run / "stability_by_case.csv", newline="") as f:
        stab = list(csv.DictReader(f))
    rules = load_rules()

    cases = [c for c in qc["cases"] if c["ct"]["hu_valid"]]
    ids = [c["case_id"] for c in cases]
    order = ids[:]
    random.Random(args.seed).shuffle(order)
    code = {cid: f"R{n:02d}" for n, cid in enumerate(order, 1)}
    qc_by = {c["case_id"]: c for c in cases}

    out = ROOT / "runs" / f"P6_clinician_review_v1_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    v1, v2 = out / "view1_independent", out / "view2_comparison"
    mi_dir, air_dir = out / "model_input_contact_sheets", out / "central_airway_review"
    for d in (v1, v2, mi_dir, air_dir):
        d.mkdir(parents=True)

    key_rows, v1_rows, v2_rows = [], [], []
    v1_index, v2_index = [], []
    for cid in sorted(ids, key=lambda c: code[c]):
        rc = code[cid]
        print(f"[{rc}] ...", end=" ", flush=True)
        ct_path = ROOT / qc_by[cid]["ct"]["file"]
        ct_img = nib.load(str(ct_path))
        ct_canon = nib.as_closest_canonical(ct_img)

        # ---- View 1: full examination, blinded
        d1 = v1 / rc
        d1.mkdir()
        shutil.copy2(ct_path, d1 / f"{rc}_ct.nii.gz")          # byte-identical copy
        kinds = {w: (lambda s, k, lo=lo, hi=hi: window_u8(axial_display(s), lo, hi))
                 for w, (lo, hi) in WINDOWS.items()}
        slices = render_stack(ct_canon, d1, kinds)
        (d1 / "ct_viewer.html").write_text(viewer_html(
            f"Case {rc} - CT examination (independent assessment)", slices,
            [("lung", "Lung W1500/L-600"), ("mediastinal", "Mediastinal W400/L40")],
            f"Full examination: {len(slices)} axial slices. For multiplanar review open "
            f"{rc}_ct.nii.gz in 3D Slicer or ITK-SNAP.", "../index.html"), encoding="utf-8")
        v1_index.append(f"<li><a href='{rc}/ct_viewer.html'>{rc}</a></li>")
        v1_rows.append({"review_code": rc})

        # ---- View 2: comparison
        d2 = v2 / rc
        d2.mkdir()
        cdir = rep_dir / "cases" / cid
        metrics = json.loads((cdir / "metrics.json").read_text())
        ref = metrics["reference_from_expert_masks"]
        side = ref.get("infection_by_side") or {}
        burden = (ref.get("infection_burden") or {}).get("infection_burden_pct")
        ref_txt = (f"infection annotation {burden:.2f}% of lung; L "
                   f"{side['left']['infection_ml']:.0f} mL, R {side['right']['infection_ml']:.0f} mL"
                   if side and burden is not None else "n/a")
        inf = np.asanyarray(nib.as_closest_canonical(
            nib.load(str(INFECTION_DIR / f"{cid}.nii.gz"))).dataobj) > 0
        lo, hi = WINDOWS["lung"]

        def annot(s, k, inf=inf, lo=lo, hi=hi):
            g = window_u8(axial_display(s), lo, hi).astype(np.float32)
            m = axial_display(inf[:, :, k])
            rgb = np.stack([g, g, g], -1)
            rgb[m] = rgb[m] * 0.55 + np.array([255, 40, 40]) * 0.45
            return rgb.round().astype(np.uint8)
        render_stack(ct_canon, d2, {"annotation": annot})
        (d2 / "annotation_viewer.html").write_text(viewer_html(
            f"Case {rc} ({cid}) - reference infection annotation", slices,
            [("annotation", "Lung window + reference infection annotation (red)"),
             (f"../../view1_independent/{rc}/lung", "Lung window, no overlay")],
            "Reference = supplied COVID-19 infection annotation. It is not an exhaustive "
            "annotation of every opacity.", "../index.html"), encoding="utf-8")

        parsed = json.loads((cdir / "vlm_parsed.json").read_text()) \
            if (cdir / "vlm_parsed.json").exists() else None
        raw = (cdir / "vlm_raw.txt").read_text(encoding="utf-8") \
            if (cdir / "vlm_raw.txt").exists() else ""
        mi = json.loads((cdir / "model_input.json").read_text()) \
            if (cdir / "model_input.json").exists() else {}
        rep = mi.get("representation", {})
        shutil.copy2(cdir / "model_input_sheet.png", mi_dir / f"{rc}_model_input.png")
        fl = [t for t in targets if t["case_id"] == cid]
        flag_txt = " | ".join(f"{t['flag']} [{t['kind']}]" for t in fl)

        model_vals = {}
        if parsed:
            d = parsed["distribution"]
            model_vals = {"assessability": parsed["assessability"],
                          "opacity_presence": parsed["opacity_presence"],
                          "laterality": d["laterality"], "craniocaudal_distribution":
                          d["craniocaudal"], "axial_distribution": d["axial"],
                          "description": parsed["description"],
                          "limitations": "; ".join(parsed["limitations"]) or "(none stated)"}
        ref_for = {"opacity_presence": ref_txt, "laterality": ref_txt}
        item_flag = {}
        finding_map = {"limitations": "limitations", "distribution_laterality": "laterality",
                       "description": "description", "opacity_presence": "opacity_presence"}
        for t in fl:
            item = finding_map.get(rules["rules"][t["flag"]]["review_finding"], "overall")
            item_flag.setdefault(item, []).append(f"{t['flag']} [{t['kind']}]")
        for item in V2_ITEMS:
            v2_rows.append({"review_code": rc, "case_id": cid,
                            "automated_flags": " | ".join(item_flag.get(item, [])),
                            "item": item, "model_output": model_vals.get(item, ""),
                            "reference_annotation": ref_for.get(item, "")})

        flag_html = "".join(f"<div class='flag'><b>{e(t['flag'])}</b> [{e(t['kind'])}]: "
                            f"{e(t['detail'])}</div>" for t in fl)
        (d2 / "comparison.html").write_text(f"""<!DOCTYPE html><html><head><meta charset='utf-8'>
<title>{rc} comparison</title><style>body{{font-family:Segoe UI,Arial;margin:20px;max-width:1100px}}
table{{border-collapse:collapse}}td,th{{border:1px solid #ccc;padding:4px 8px;font-size:13px;vertical-align:top}}
th{{background:#f2f2f2;text-align:left}}.flag{{background:#fdecea;border:1px solid #e57373;padding:5px;margin:4px 0}}
pre{{background:#f6f6f6;border:1px solid #ddd;padding:8px;white-space:pre-wrap}}small{{color:#555}}</style></head><body>
<h2>Case {rc} &mdash; comparison view <small>({e(cid)})</small></h2>
<p><a href='../index.html'>&larr; index</a> | <a href='annotation_viewer.html'>CT with reference annotation</a> |
<a href='../../view1_independent/{rc}/ct_viewer.html'>CT without overlay</a> |
<a href='../../model_input_contact_sheets/{rc}_model_input.png'>exact model input (contact sheet)</a></p>
<p><small>Research use only. Model output is unvalidated. Complete View 1 for all cases before using this page.</small></p>
<h3>Automated flags ({e(rules['rules_version'])})</h3>
<p><small>Exploratory rules written after inspecting this pilot; not prospectively defined criteria.
input_description_error = the model described its supplied input incorrectly. review_flag = prioritise review,
not a confirmed error.</small></p>{flag_html or '<p>none</p>'}
<h3>Model output (MedGemma 1.5 4B)</h3>
<table>{''.join(f'<tr><th>{e(k)}</th><td>{e(v)}</td></tr>' for k, v in model_vals.items()) or '<tr><td>no valid output</td></tr>'}</table>
<p><b>Raw response (verbatim)</b></p><pre>{e(raw)}</pre>
<p><small>Model input: {e(rep.get('n_images'))} of {e(rep.get('n_source_slices'))} axial slices sampled
superior&rarr;inferior (z {e(rep.get('superior_most_z_mm'))} to {e(rep.get('inferior_most_z_mm'))} mm),
three HU windows as RGB. The model did not see the full examination.</small></p>
<h3>Reference annotation</h3><p>{e(ref_txt)}</p>
<p><small>Supplied COVID-19 infection annotation; not an exhaustive opacity annotation.</small></p>
</body></html>""", encoding="utf-8")
        v2_index.append((rc, cid, fl))
        key_rows.append({"review_code": rc, "case_id": cid, "ct_file": qc_by[cid]["ct"]["file"],
                         "ct_sha256": qc_by[cid]["ct"]["sha256"],
                         "copy_sha256": sha256(d1 / f"{rc}_ct.nii.gz")})
        print("ok")

    (v1 / "index.html").write_text(
        "<!DOCTYPE html><html><head><meta charset='utf-8'><title>View 1</title></head><body "
        "style='font-family:Segoe UI,Arial;margin:20px'><h2>View 1 - independent assessment"
        "</h2><p>Assess each examination before opening View 2. Record results in "
        "scoring_view1_independent.csv.</p><ul>" + "".join(v1_index) + "</ul></body></html>",
        encoding="utf-8")
    rows_html = "".join(
        f"<tr style='background:{'#fdecea' if fl else '#fff'}'><td><a href='{rc}/comparison.html'>"
        f"{rc}</a></td><td>{e(cid)}</td><td>{e('; '.join(sorted({t['kind'] for t in fl})) or '-')}"
        f"</td><td>{e('; '.join(t['flag'] for t in fl) or '-')}</td></tr>"
        for rc, cid, fl in v2_index)
    (v2 / "index.html").write_text(
        "<!DOCTYPE html><html><head><meta charset='utf-8'><title>View 2</title><style>"
        "td,th{border:1px solid #ccc;padding:4px 8px}table{border-collapse:collapse}</style>"
        "</head><body style='font-family:Segoe UI,Arial;margin:20px'><h2>View 2 - comparison"
        "</h2><p>Flagged cases are highlighted (exploratory rules "
        f"{e(rules['rules_version'])}; flags are not confirmed errors).</p><table><tr><th>Code"
        "</th><th>Case</th><th>Flag kinds</th><th>Flags</th></tr>" + rows_html +
        "</table></body></html>", encoding="utf-8")
    write_csv(v1 / "scoring_view1_independent.csv", v1_rows, V1_COLS)
    write_csv(v2 / "scoring_view2_comparison.csv", v2_rows, V2_COLS)
    write_csv(v2 / "unblinding_key.csv", key_rows,
              ["review_code", "case_id", "ct_file", "ct_sha256", "copy_sha256"])

    # ---- central airway review
    exc_cases = sorted({r["case_id"] for r in central_exceptions(stab)})
    lengths = {}
    for r in stab:
        if r["metric"] in ("gen1_right_length_mm", "gen1_left_length_mm"):
            lengths.setdefault(r["metric"], {})[r["case_id"]] = float(r["default"])
    others_r = {c: v for c, v in lengths["gen1_right_length_mm"].items() if c not in exc_cases}
    others_l = {c: v for c, v in lengths["gen1_left_length_mm"].items() if c not in exc_cases}
    selection = {}
    for c in exc_cases:
        bad = [r["metric"] for r in central_exceptions(stab) if r["case_id"] == c]
        selection[c] = f"exception: central metric(s) unstable across settings ({', '.join(bad)})"
    shortest_r = min(others_r, key=others_r.get)
    longest_l = max(others_l, key=others_l.get)
    selection.setdefault(shortest_r, f"shortest right main bronchus among non-exception cases "
                         f"({others_r[shortest_r]:.1f} mm): extreme value, most likely to reveal a "
                         "misplaced distal endpoint")
    selection.setdefault(longest_l, f"longest left main bronchus among non-exception cases "
                         f"({others_l[longest_l]:.1f} mm): extreme value, most likely to reveal a "
                         "missed bifurcation")
    selection.setdefault("coronacases_001", "E0 reference case (first case through the pipeline); "
                         "typical values")
    air_rows, air_index = [], []
    stab_by = {(r["case_id"], r["metric"]): r for r in stab}
    seg_metric = {"trachea": "trachea_length_mm", "right_main": "gen1_right_length_mm",
                  "left_main": "gen1_left_length_mm"}
    landmarks = {
        "trachea": ("most superior point of the predicted airway mask (limited by scan "
                    "coverage/segmentation - NOT the anatomical tracheal origin)",
                    "carina (tracheal bifurcation)"),
        "right_main": ("carina", "origin of the right upper lobe bronchus (first branching of "
                       "the right main bronchus)"),
        "left_main": ("carina", "bifurcation into left upper and left lower lobe bronchi "
                      "(secondary carina)"),
    }
    for cid, why in selection.items():
        img = nib.load(str(ROOT / seg[cid]["output_dir"] / AIRWAY_MASK))
        res = analyze_airway_tree(img)
        segs = central_branches(res)
        paths = {k: res.branch_paths[v["branch_id"]] for k, v in segs.items()}
        alt_note = ""
        exc_l = stab_by.get((cid, "gen1_left_length_mm"))
        if exc_l and exc_l["rating"] == "unstable" and exc_l["most_influential"]:
            k_, v_ = exc_l["most_influential"].split("=")
            res_alt = analyze_airway_tree(img, {k_: float(v_)})
            alt = central_branches(res_alt).get("left_main")
            if alt:
                segs["left_main_alt"] = alt
                paths["left_main_alt"] = res_alt.branch_paths[alt["branch_id"]]
                alt_note = (f"Alternative setting {exc_l['most_influential']} gives left main "
                            f"length {alt['length_mm']:.1f} mm (endpoint L').")
        fig_name = f"{cid}_landmarks.png"
        airway_figure(nib.load(str(ROOT / qc_by[cid]["ct"]["file"])), segs, paths,
                      air_dir / fig_name, f"{cid} - central airway endpoints")
        for sname in ("trachea", "right_main", "left_main", "left_main_alt"):
            if sname not in segs:
                continue
            b = segs[sname]
            st = stab_by.get((cid, seg_metric.get(sname, "gen1_left_length_mm")), {})
            base = sname.replace("_alt", "")
            air_rows.append({
                "case_id": cid, "selection_rationale": why,
                "segment": sname + (" (alternative setting " + exc_l["most_influential"] + ")"
                                    if sname.endswith("_alt") else ""),
                "reported_length_mm": b["length_mm"],
                "parameter_stability": st.get("rating", "") if not sname.endswith("_alt") else "",
                "length_range_across_settings_mm": f"{st.get('min', '')}-{st.get('max', '')}",
                "proximal_point_ras_mm": b["proximal_ras_mm"],
                "distal_point_ras_mm": b["distal_ras_mm"],
                "algorithm_endpoint_definition": (
                    "proximal = root (most superior skeleton endpoint); distal = first "
                    "bifurcation node after pruning/contraction" if base == "trachea" else
                    "proximal = trachea distal node; distal = next bifurcation node after "
                    "pruning/contraction"),
                "intended_proximal_landmark": landmarks[base][0],
                "intended_distal_landmark": landmarks[base][1]})
        air_index.append(f"<h3>{e(cid)}</h3><p><b>Selection:</b> {e(why)}</p>"
                         f"{'<p>' + e(alt_note) + '</p>' if alt_note else ''}"
                         f"<img src='{fig_name}' style='max-width:100%'>")
    write_csv(air_dir / "central_airway_review.csv", air_rows, AIR_COLS)
    n_hu = len(ids)
    (air_dir / "central_airway_review.html").write_text(
        "<!DOCTYPE html><html><head><meta charset='utf-8'><title>Central airway review</title>"
        "</head><body style='font-family:Segoe UI,Arial;margin:20px;max-width:1500px'>"
        "<h2>Central-airway landmark review</h2><p><i>Complete Views 1 and 2 first.</i> For each selected case: do the marked "
        "endpoints sit at the intended anatomical landmarks, and does the centreline stay in "
        "the lumen? Record answers in central_airway_review.csv. Yellow circles = endpoints "
        "(T0 trachea proximal, C trachea distal, R/L main bronchus distal, L' alternative "
        "setting). Axial crops show the endpoint (crosshair) and nearby centreline points.</p>"
        f"<p><b>This is a selected subset ({len(selection)} of {n_hu} cases). It does not "
        "validate the measurements for the whole cohort.</b></p>" + "".join(air_index) +
        "</body></html>", encoding="utf-8")
    (air_dir / "README.md").write_text(f"""# Central-airway landmark review

Purpose: verify that the endpoints used for each reported tracheal and main-bronchial length
correspond to the intended anatomical landmarks. Open `central_airway_review.html`; record
judgements in `central_airway_review.csv` (columns proximal_landmark_correct,
distal_landmark_correct, centreline_within_lumen, length_plausible: yes | no | unsure).

Method: {METHOD}. Lengths are along the smoothed centreline between the two endpoints.
Central-airway measurements were generally stable across tested settings; anatomical
accuracy remains unvalidated.

Selected cases ({len(selection)} of {n_hu}). Selection rules were defined by the analyst
after the parameter-sensitivity sweep and before clinician review: every case with an
unstable central-airway metric, the shortest right and longest left main bronchus among the
remaining cases, and the E0 reference case.
""" + "".join(f"- {c}: {w}\n" for c, w in selection.items()) + """
Cases are identified by dataset case ID only (no View 1 review codes), so this sheet does
not unblind View 1. Complete Views 1 and 2 first.

Known limitation: the tracheal proximal endpoint is the top of the predicted airway mask,
which depends on scan coverage and segmentation; tracheal length is therefore not the
anatomical tracheal length.

This subset does not validate the measurements for the whole cohort.
""", encoding="utf-8")

    write_csv(out / "data_dictionary.csv",
              [{"sheet": s, "column": c, "allowed_values_or_meaning": v} for s, c, v in DICTIONARY],
              ["sheet", "column", "allowed_values_or_meaning"])
    (out / "README_reviewer.md").write_text(f"""# Clinician review package v1 - lung CT pilot

Research use only. No result in this package is a validated clinical finding.
{n_hu} cases (all HU-valid scans of the pilot). The 10 display-scaled scans were excluded
from HU-dependent analysis by protocol and are not part of this review.

## Step 1 - independent assessment (View 1, blinded)
Open `view1_independent/index.html`. Cases use neutral codes in randomised order
(seed {args.seed}). Each case shows the full CT examination (all axial slices, lung and
mediastinal windows); for multiplanar review open `R##/R##_ct.nii.gz` in 3D Slicer or
ITK-SNAP. Model outputs, reference annotations and automated flags are not shown.
Record your assessment for every case in `view1_independent/scoring_view1_independent.csv`
(assessability, opacity presence, laterality, cranio-caudal and axial distribution, other
findings, comments). **Complete all cases before opening View 2.**

## Step 2 - comparison (View 2)
Open `view2_comparison/index.html`. For each case: the model response (verbatim and
parsed), the reference infection annotation (scrollable overlay on the full CT), automated
flags, and links to the exact model input (`model_input_contact_sheets/`, the 85 sampled
slices the model saw) and to the plain CT. Record agreement, error type, unsupported
claims, omissions and comments in `view2_comparison/scoring_view2_comparison.csv`.

- The reference is a COVID-19 infection annotation, not an exhaustive annotation of every
  opacity. Disagreement with it is not automatically a model error.
- Automated flags ({rules['rules_version']}) are EXPLORATORY rules written after inspecting
  this pilot, not prospectively defined criteria. `input_description_error` = the model
  described its supplied input incorrectly (e.g. "single slice" when 85 images were
  supplied). `review_flag` = prioritise review, not a confirmed error.
  `processing_sensitivity` = output changed with an image-processor implementation detail.
- All clinical judgement columns are blank by design.

## Step 3 - central-airway landmark review
See `central_airway_review/README.md`.

## Files
- `data_dictionary.csv`: allowed values for every scoring column.
- `view2_comparison/unblinding_key.csv`: code to case mapping (do not open before Step 2).
""", encoding="utf-8")

    manifest = {
        "package": out.name, "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_state(), "seed": args.seed,
        "inputs": {k: str(getattr(args, f"{k}_run")) for k in
                   ("report", "qc", "seg", "airway", "sensitivity")},
        "review_flag_rules": {"version": rules["rules_version"], "sha256": rules["_sha256"]},
        "cases": n_hu, "central_airway_selection": selection,
        "windows_hu": WINDOWS, "baseline_tag": "baseline-feasibility-v1",
    }
    (out / "package_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"done -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
