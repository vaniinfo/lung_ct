"""Failure analysis of the PleThora pleural-effusion check (core env). No threshold is changed or proposed here.

Step 1 (--attributes): per-scan attributes for every analysed scan -> <run>/scan_attributes.csv
Step 2 (--overlays):   overlay PNGs for the scans selected for inspection -> <run>/overlays/, <run>/inspection_selection.json

Reference for this analysis = the FIRST reviewer file of each scan (the outline PleThora distributes as its primary
one). Revision 1 of the comparison used the reviewer file whose volume equals the spreadsheet value; both are reported.

Groups (fixed before any overlay was looked at; mutually exclusive, first match wins). "tool mL" is an estimate from
the predicted mask. Limits 5 and 20 mL are the existing checklist limits, read from the config, not changed.
  scans coded effusion:
    empty_mask            tool mL = 0
    threshold_abstention  5 <= tool mL < 20                                   (answer "undecided")
    undersegmentation     0 < tool mL < 0.5 x reference mL                    (any answer)
    over_outlining        tool mL > 1.5 x reference mL and tool - reference > 20 mL
    control_good_overlap  Dice >= 0.70
    other                 everything else (volume within 0.5-1.5 x reference but Dice < 0.70)
  scans coded no effusion:
    neg_empty_mask        tool mL = 0
    neg_low_nonzero       0 < tool mL < 5                                     (answer "none")
    threshold_abstention  5 <= tool mL < 20
    false_positive_candidate   tool mL >= 20                                  (answer "present")

Usage: python -I scripts/plethora_failure_analysis.py --run runs/E14_plethora_pleural_effusion_<ts> --attributes --overlays
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage as ndi

ROOT = Path(__file__).resolve().parent.parent
P = ROOT / "data" / "raw" / "plethora"
D = ROOT / "data" / "derived" / "plethora"
UNDER, OVER, OVER_ML, GOOD_DICE = 0.5, 1.5, 20.0, 0.70


def canon(path: Path):
    return nib.as_closest_canonical(nib.load(str(path)))     # RAS+: index 0 increases towards patient RIGHT, 1 -> anterior, 2 -> head


def load(pid: str) -> dict:
    ct = canon(P / "volumes" / f"{pid}.nii.gz")
    hu = np.asanyarray(ct.dataobj)

    def grid(path: Path) -> np.ndarray:
        m = canon(path)
        assert m.shape == ct.shape and np.allclose(m.affine, ct.affine, atol=0.05), path
        return np.asanyarray(m.dataobj)
    o = {"hu": hu, "zooms": tuple(float(z) for z in ct.header.get_zooms()[:3]), "affine": ct.affine}
    o["vox_ml"] = float(np.prod(o["zooms"])) / 1000.0
    t = D / pid / "pleural_pericard_effusion"
    o["pred"] = grid(t / "pleural_effusion.nii.gz") > 0
    o["tool_lung"] = grid(t / "lung_pleural.nii.gz") > 0
    o["tool_peri"] = grid(t / "pericardial_effusion.nii.gz") > 0
    th = sorted((P / "thorax").rglob(f"{pid}_thor_cav_primary_reviewer.nii.gz"))
    o["thorax"] = grid(th[0]).astype(np.uint8) if th else None
    o["refs"] = {f.name.split("_effusion_")[1].split("_reviewer")[0]: grid(f) > 0 for f in sorted((P / "effusions").rglob(f"{pid}_effusion_*_reviewer.nii.gz"))}
    seg = P / "gtv" / f"{pid}_gtv.nii.gz"
    o["gtv"] = grid(seg) > 0 if seg.exists() else None
    return o


def bbox(m: np.ndarray, pad_mm: float, zooms) -> tuple[slice, ...]:
    idx = np.nonzero(m)
    return tuple(slice(max(0, int(i.min()) - int(round(pad_mm / z))), int(i.max()) + int(round(pad_mm / z)) + 1) for i, z in zip(idx, zooms))


def mean_hu(hu, m) -> float | None:
    return round(float(hu[m].mean()), 1) if m.any() else None


def n_comp(m: np.ndarray, vox_ml: float, min_ml: float = 0.5) -> tuple[int, float]:
    lab, n = ndi.label(m)
    if n == 0:
        return 0, 0.0
    sizes = np.bincount(lab.ravel())[1:] * vox_ml
    return int((sizes >= min_ml).sum()), round(float(sizes.max()), 1)


def group_of(r: dict, lo: float, hi: float) -> str:
    t, ref = r["tool_ml"], r["reference_first_ml"]
    if r["coded"] == "effusion":
        if t == 0:
            return "empty_mask"
        if lo <= t < hi:
            return "threshold_abstention"
        if t < UNDER * ref:
            return "undersegmentation"
        if t > OVER * ref and t - ref > OVER_ML:
            return "over_outlining"
        return "control_good_overlap" if (r["dice_first"] or 0) >= GOOD_DICE else "other"
    if t == 0:
        return "neg_empty_mask"
    if t < lo:
        return "neg_low_nonzero"
    return "threshold_abstention" if t < hi else "false_positive_candidate"


def attributes(pid: str, coded: str, csv_row: dict) -> dict:
    o = load(pid)
    hu, pred, z, v = o["hu"], o["pred"], o["zooms"], o["vox_ml"]
    r = {"patient_id": pid, "coded": coded, "tool_ml": round(float(pred.sum()) * v, 1), "tool_pericardial_ml": round(float(o["tool_peri"].sum()) * v, 1),
         "tool_lung_label_ml": round(float(o["tool_lung"].sum()) * v, 0)}
    r["tool_components_ge_0_5ml"], r["tool_largest_component_ml"] = n_comp(pred, v)
    r["tool_mean_hu"] = mean_hu(hu, pred)
    th = o["thorax"]
    mid = None
    if th is not None and th.any():
        r["tool_fraction_inside_reference_thoracic_cavity"] = round(float((pred & (th > 0)).sum()) / max(1, int(pred.sum())), 3) if pred.any() else None
        cx = {int(k): float(np.nonzero(th == k)[0].mean()) for k in np.unique(th) if k}
        if len(cx) == 2:
            (kl, xl), (kr, xr) = sorted(cx.items(), key=lambda kv: kv[1])          # RAS+: lower index 0 = patient LEFT, higher = patient RIGHT
            mid = int(round((xl + xr) / 2))
            r["thorax_label_on_patient_right"], r["thorax_right_ml_by_position"], r["thorax_left_ml_by_position"] = kr, round(float((th == kr).sum()) * v), round(float((th == kl).sum()) * v)
            r["thorax_right_ml_spreadsheet"], r["thorax_left_ml_spreadsheet"] = csv_row["Right.Thorax.Volume"], csv_row["Left.Thorax.Volume"]
    if mid is not None:
        r["tool_right_ml"], r["tool_left_ml"] = round(float(pred[mid:].sum()) * v, 1), round(float(pred[:mid].sum()) * v, 1)
    if o["gtv"] is not None:
        g = o["gtv"]
        r["gtv_ml"] = round(float(g.sum()) * v, 1)
        r["tool_ml_inside_gtv"] = round(float((pred & g).sum()) * v, 1)
        if pred.any() and g.any():
            bb = bbox(pred | g, 5, z)
            dg = ndi.distance_transform_edt(~g[bb], sampling=z)[pred[bb]]
            r["tool_fraction_within_10mm_of_gtv"] = round(float((dg <= 10).mean()), 3)
    refs = o["refs"]
    r["reviewer_volumes_ml"] = json.dumps({k: round(float(m.sum()) * v, 1) for k, m in refs.items()})
    if coded == "effusion":
        ref = refs["first"]
        r["reference_first_ml"] = round(float(ref.sum()) * v, 1)
        s = int(pred.sum()) + int(ref.sum())
        r["dice_first"] = round(2.0 * int((pred & ref).sum()) / s, 4) if s else None
        r["reference_mean_hu"], r["reference_missed_mean_hu"] = mean_hu(hu, ref), mean_hu(hu, ref & ~pred)
        r["tool_outside_reference_mean_hu"] = mean_hu(hu, pred & ~ref)
        r["tool_ml_overlapping_reference"] = round(float((pred & ref).sum()) * v, 1)
        r["reference_components_ge_0_5ml"], r["reference_largest_component_ml"] = n_comp(ref, v)
        if mid is not None:
            r["reference_right_ml"], r["reference_left_ml"] = round(float(ref[mid:].sum()) * v, 1), round(float(ref[:mid].sum()) * v, 1)
        if ref.any():
            bb = bbox(ref, 3, z)
            r["reference_largest_inscribed_sphere_diameter_mm"] = round(2 * float(ndi.distance_transform_edt(ref[bb], sampling=z).max()), 1)
            zz = np.nonzero(ref.any(axis=(0, 1)))[0]
            r["reference_slices"], r["reference_max_axial_area_cm2"] = int(len(zz)), round(float(ref.sum(axis=(0, 1)).max()) * z[0] * z[1] / 100, 1)
            # where did the reference fluid go in the tool's own labels?
            n = int(ref.sum())
            r["reference_fraction_tool_calls_lung"] = round(float((ref & o["tool_lung"]).sum()) / n, 3)
            r["reference_fraction_tool_calls_pericardial"] = round(float((ref & o["tool_peri"]).sum()) / n, 3)
            if o["tool_lung"].any():                                   # proxy for the crop the effusion model saw (lung box + 50 mm)
                box = np.zeros(ref.shape, bool)
                box[bbox(o["tool_lung"], 50, z)] = True
                r["reference_fraction_inside_tool_lung_box_plus_50mm"] = round(float((ref & box).sum()) / n, 3)
            if pred.any():
                bb = bbox(pred | ref, 5, z)
                d = ndi.distance_transform_edt(~ref[bb], sampling=z)[pred[bb]]
                r["tool_min_distance_to_reference_mm"], r["tool_median_distance_to_reference_mm"] = round(float(d.min()), 1), round(float(np.median(d)), 1)
                r["tool_fraction_within_10mm_of_reference"] = round(float((d <= 10).mean()), 3)
            if o["gtv"] is not None and o["gtv"].any():
                bb = bbox(ref | o["gtv"], 5, z)
                r["reference_min_distance_to_gtv_mm"] = round(float(ndi.distance_transform_edt(~o["gtv"][bb], sampling=z)[ref[bb]].min()), 1)
    else:
        r["reference_first_ml"], r["dice_first"] = 0.0, None
    return r


def gtv_attributes(pid: str, coded: str) -> dict:
    """Tumour (GTV-1) against the predicted mask and the reference. Masks only; grids checked against each other."""
    f = P / "gtv" / f"{pid}_gtv.nii.gz"
    r = {"patient_id": pid}
    if not f.exists():
        return {**r, "gtv_status": "no tumour outline available"}
    g_img, p_img = canon(f), canon(D / pid / "pleural_pericard_effusion" / "pleural_effusion.nii.gz")
    assert g_img.shape == p_img.shape and np.allclose(g_img.affine, p_img.affine, atol=0.05)
    g, pred = np.asanyarray(g_img.dataobj) > 0, np.asanyarray(p_img.dataobj) > 0
    z = tuple(float(q) for q in g_img.header.get_zooms()[:3])
    v = float(np.prod(z)) / 1000.0
    r.update(gtv_status="ok", gtv_ml=round(float(g.sum()) * v, 1), tool_ml_inside_gtv=round(float((pred & g).sum()) * v, 1))
    if pred.any() and g.any():
        bb = bbox(pred | g, 5, z)
        d = ndi.distance_transform_edt(~g[bb], sampling=z)[pred[bb]]
        r.update(tool_fraction_inside_gtv=round(float((pred & g).sum()) / int(pred.sum()), 3), tool_fraction_within_10mm_of_gtv=round(float((d <= 10).mean()), 3),
                 tool_min_distance_to_gtv_mm=round(float(d.min()), 1))
    if coded == "effusion":
        rf = sorted((P / "effusions").rglob(f"{pid}_effusion_first_reviewer.nii.gz"))
        m = canon(rf[0])
        assert m.shape == g_img.shape and np.allclose(m.affine, g_img.affine, atol=0.05)
        ref = np.asanyarray(m.dataobj) > 0
        r["reference_ml_inside_gtv"] = round(float((ref & g).sum()) * v, 1)
        if ref.any() and g.any():
            bb = bbox(ref | g, 5, z)
            d = ndi.distance_transform_edt(~g[bb], sampling=z)[ref[bb]]
            r.update(reference_min_distance_to_gtv_mm=round(float(d.min()), 1), reference_fraction_within_10mm_of_gtv=round(float((d <= 10).mean()), 3))
    return r


def overlay(pid: str, row: dict, out: Path) -> dict:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    o = load(pid)
    hu, pred = o["hu"], o["pred"]
    ref = o["refs"].get("first")
    ref = ref if ref is not None else np.zeros(pred.shape, bool)
    a_ref, a_pred = ref.sum(axis=(0, 1)), pred.sum(axis=(0, 1))
    zs: list[tuple[int, str]] = []

    def add(k, why):
        if k is not None and all(abs(int(k) - q) >= 2 for q, _ in zs):
            zs.append((int(k), why))

    def pct(a, q):
        nz = np.nonzero(a)[0]
        return int(nz[int(round(q * (len(nz) - 1)))]) if len(nz) else None
    if ref.any():
        add(int(a_ref.argmax()), "largest reference area")
    if pred.any():
        add(int(a_pred.argmax()), "largest tool area")
    for a in (a_ref, a_pred):
        for q in (0.5, 0.2, 0.8, 0.35, 0.65):
            if len(zs) < 3:
                add(pct(a, q), "other level")
    if not zs:
        zs = [(hu.shape[2] // 2, "mid scan (both outlines empty)")]
    zs = sorted(zs[:3])
    body = hu > -400
    xs, ys = np.nonzero(body.any(axis=(1, 2)))[0], np.nonzero(body.any(axis=(0, 2)))[0]
    sx, sy = slice(max(0, xs.min() - 5), xs.max() + 6), slice(max(0, ys.min() - 5), ys.max() + 6)
    asp = o["zooms"][1] / o["zooms"][0]
    fig, ax = plt.subplots(2, len(zs), figsize=(6.2 * len(zs), 10.2), squeeze=False)
    for j, (k, why) in enumerate(zs):
        img = hu[sx, sy, k].T[::-1, ::-1]              # RAS+ array: rows flipped -> anterior at top; columns flipped -> patient right at image left
        for i in range(2):
            a = ax[i][j]
            a.imshow(np.clip(img, -160, 240), cmap="gray", vmin=-160, vmax=240, aspect=asp, interpolation="nearest")
            a.set_xticks([])
            a.set_yticks([])
            a.text(0.02, 0.5, "R", color="yellow", transform=a.transAxes, fontsize=16, fontweight="bold", va="center")
            a.text(0.96, 0.5, "L", color="yellow", transform=a.transAxes, fontsize=16, fontweight="bold", va="center")
            a.text(0.5, 0.97, "A", color="yellow", transform=a.transAxes, fontsize=13, ha="center", va="top")
        ax[0][j].set_title(f"slice {k} ({why}); CT only, window -160..240 HU", fontsize=11)
        for m, col, lw in ((o["gtv"], "#ffd400", 1.2), (ref, "#00e5ff", 1.5), (pred, "#ff3df2", 1.5)):
            if m is not None and m[sx, sy, k].any():
                ax[1][j].contour(m[sx, sy, k].T[::-1, ::-1].astype(float), levels=[0.5], colors=[col], linewidths=lw)
        ax[1][j].set_title(f"reference {a_ref[k] * o['zooms'][0] * o['zooms'][1] / 100:.1f} cm2, tool {a_pred[k] * o['zooms'][0] * o['zooms'][1] / 100:.1f} cm2 on this slice", fontsize=11)
    leg = [Line2D([0], [0], color="#00e5ff", lw=2, label="expert reference (first reviewer)"), Line2D([0], [0], color="#ff3df2", lw=2, label="tool: predicted pleural-effusion mask")]
    if o["gtv"] is not None:
        leg.append(Line2D([0], [0], color="#ffd400", lw=2, label="tumour (GTV-1, NSCLC-Radiomics)"))
    fig.legend(handles=leg, loc="lower center", ncol=3, fontsize=12, frameon=False)
    fig.suptitle(f"{pid} | coded: {row['coded']} | group: {row['group']} | reference {row['reference_first_ml']} mL | tool estimate {row['tool_ml']} mL | Dice {row['dice_first']}\n"
                 "R / L = patient right / left from the image affine (radiological display). Development material, not for clinical use.", fontsize=12)
    fig.tight_layout(rect=(0, 0.035, 1, 0.94))
    f = out / f"{row['group']}__{pid}.png"
    fig.savefig(f, dpi=80)
    plt.close(fig)
    return {"file": f.name, "slices": [{"k": k, "why": w} for k, w in zs]}


def pick(rows: list[dict], n: int, key: str) -> list[dict]:
    rows = sorted(rows, key=lambda r: (r[key], r["patient_id"]))
    if len(rows) <= n:
        return rows
    if n == 1:
        return [rows[len(rows) // 2]]
    return [rows[int(round(i * (len(rows) - 1) / (n - 1)))] for i in range(n)]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run", required=True)
    ap.add_argument("--attributes", action="store_true")
    ap.add_argument("--overlays", action="store_true")
    ap.add_argument("--gtv", action="store_true", help="tumour attributes -> <run>/gtv_attributes.csv (needs scripts/plethora_fetch_gtv.py)")
    args = ap.parse_args()
    run = ROOT / args.run
    cond = next(c for c in json.loads((ROOT / "configs" / "checklist_v3_5.json").read_text(encoding="utf-8"))["conditions"] if c["key"] == "pleural_effusion")
    hi, lo = float(cond["present_at_least"]), float(cond["absent_below"])
    res = [r for r in csv.DictReader(open(run / "results.csv", encoding="utf-8")) if r["status"] == "ok"]
    sheet = {r["PatientID"]: r for r in csv.DictReader(open(P / "download" / "Thoracic-and-Pleural-Effusion-Segmentations-April-2020.csv", encoding="utf-8-sig"))}
    att = run / "scan_attributes.csv"
    if args.attributes:
        rows = []
        for i, r in enumerate(res, 1):
            a = attributes(r["patient_id"], r["group"], sheet[r["patient_id"]])
            a["tool_answer"] = "present" if a["tool_ml"] >= hi else ("none" if a["tool_ml"] < lo else "undecided")
            a["rev1_reference_reviewer"], a["rev1_reference_ml"], a["rev1_dice"] = r["reference_reviewer"], r["reference_ml"], r["dice"]
            a["group"] = group_of(a, lo, hi)
            rows.append(a)
            print(i, a["patient_id"], a["coded"], a["group"], a["reference_first_ml"], a["tool_ml"], a["dice_first"], flush=True)
        cols = ["patient_id", "coded", "group", "tool_answer", "reference_first_ml", "tool_ml", "dice_first"]
        cols += sorted({k for a in rows for k in a} - set(cols))
        with open(att, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            w.writerows(rows)
    if args.gtv:
        rows = []
        for i, r in enumerate(res, 1):
            rows.append(gtv_attributes(r["patient_id"], r["group"]))
            print(i, rows[-1], flush=True)
        cols = ["patient_id", "gtv_status"] + sorted({k for a in rows for k in a} - {"patient_id", "gtv_status"})
        with open(run / "gtv_attributes.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            w.writerows(rows)
    if args.overlays:
        rows = list(csv.DictReader(open(att, encoding="utf-8")))
        for r in rows:
            for k in ("tool_ml", "reference_first_ml"):
                r[k] = float(r[k])
        by = {}
        for r in rows:
            by.setdefault((r["coded"], r["group"]), []).append(r)
        plan = [("effusion", "empty_mask", 99, "reference_first_ml"), ("effusion", "undersegmentation", 5, "reference_first_ml"), ("effusion", "threshold_abstention", 4, "reference_first_ml"),
                ("no_effusion", "threshold_abstention", 3, "tool_ml"), ("no_effusion", "false_positive_candidate", 99, "tool_ml"), ("effusion", "over_outlining", 5, "tool_ml"),
                ("effusion", "control_good_overlap", 4, "reference_first_ml"), ("effusion", "other", 3, "reference_first_ml"), ("no_effusion", "neg_low_nonzero", 2, "tool_ml"),
                ("no_effusion", "neg_empty_mask", 1, "patient_id")]
        out = run / "overlays"
        out.mkdir(exist_ok=True)
        sel = []
        for coded, g, n, key in plan:
            for r in pick(by.get((coded, g), []), n, key):
                info = overlay(r["patient_id"], r, out)
                sel.append({"patient_id": r["patient_id"], "coded": coded, "group": g, **info})
                print(coded, g, r["patient_id"], info["file"], flush=True)
        (run / "inspection_selection.json").write_text(json.dumps({
            "rule": "all scans of 'empty_mask' and 'false_positive_candidate'; otherwise scans evenly spaced along the sort key within the group (deterministic, no manual choice)",
            "plan": [{"coded": c, "group": g, "max_scans": n, "sorted_by": k, "group_size": len(by.get((c, g), []))} for c, g, n, k in plan], "selected": sel}, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
