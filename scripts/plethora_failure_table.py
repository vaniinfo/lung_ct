"""Merge the PleThora failure-analysis attributes, tumour attributes and inspection notes (core env).

Writes <run>/failure_analysis.csv (one row per inspected scan), <run>/failure_groups.csv (one row per analysed scan)
and <run>/failure_summary.json. Reads only CSV / JSON written by plethora_compare.py and plethora_failure_analysis.py.
Changes no threshold; the 5 and 20 mL limits are read from the checklist config.

Usage: python -I scripts/plethora_failure_table.py --run runs/E14_plethora_pleural_effusion_<ts>
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import statistics as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
P = ROOT / "data" / "raw" / "plethora"
BANDS = [(0, 5), (5, 20), (20, 50), (50, 150), (150, 1e9)]


def f(x):
    return float(str(x).replace(",", "")) if x not in ("", None, "None") else None


def med(vals, nd=1):
    v = [x for x in vals if x is not None]
    return round(st.median(v), nd) if v else None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run", required=True)
    args = ap.parse_args()
    run = ROOT / args.run
    cond = next(c for c in json.loads((ROOT / "configs" / "checklist_v3_5.json").read_text(encoding="utf-8"))["conditions"] if c["key"] == "pleural_effusion")
    hi, lo = float(cond["present_at_least"]), float(cond["absent_below"])
    A = list(csv.DictReader(open(run / "scan_attributes.csv", encoding="utf-8")))
    G = {r["patient_id"]: r for r in csv.DictReader(open(run / "gtv_attributes.csv", encoding="utf-8"))}
    notes = json.loads((run / "inspection_notes.json").read_text(encoding="utf-8"))
    sel = json.loads((run / "inspection_selection.json").read_text(encoding="utf-8"))
    sheet = {r["PatientID"]: r for r in csv.DictReader(open(P / "download" / "Thoracic-and-Pleural-Effusion-Segmentations-April-2020.csv", encoding="utf-8-sig"))}
    for r in A:
        g = G.get(r["patient_id"], {})
        for k in ("gtv_status", "gtv_ml", "tool_ml_inside_gtv", "tool_fraction_inside_gtv", "tool_fraction_within_10mm_of_gtv", "reference_ml_inside_gtv",
                  "reference_min_distance_to_gtv_mm", "reference_fraction_within_10mm_of_gtv"):
            r[k] = g.get(k, "")
        t, ig = f(r["tool_ml"]), f(r["tool_ml_inside_gtv"])
        r["tool_ml_outside_tumour_outline"] = round(t - ig, 1) if ig is not None else ""
    pos, neg = [r for r in A if r["coded"] == "effusion"], [r for r in A if r["coded"] == "no_effusion"]

    def ans(ml):
        return "present" if ml >= hi else ("none" if ml < lo else "undecided")

    def count(rs, key="tool_answer"):
        c = collections.Counter(r[key] for r in rs)
        return {a: c.get(a, 0) for a in ("present", "undecided", "none")}

    def bands(key):
        out = []
        for a, b in BANDS:
            rs = [r for r in pos if a <= f(r[key]) < b]
            out.append({"reference_ml": f"{a:g}-{b:g}" if b < 1e8 else f"{a:g}+", "scans": len(rs), **count(rs)})
        return out
    # ---- left / right cross-checks (image geometry against the PleThora spreadsheet) ----
    side = collections.Counter()
    for r in pos:
        s = sheet[r["patient_id"]]
        sr, sl = f(s["Right.Pleural.Effusion.Volume"]) or 0, f(s["Left.Pleural.Effusion.Volume"]) or 0
        pr, pl = f(r.get("reference_right_ml")), f(r.get("reference_left_ml"))
        if pr is None or (sr > 0) == (sl > 0) or not (pr > 5 * pl or pl > 5 * pr):
            continue
        side["agree" if (sr > 0) == (pr > pl) else "disagree: " + r["patient_id"]] += 1
    thor = collections.Counter()
    for r in A:
        a, b = f(r.get("thorax_right_ml_by_position")), f(r.get("thorax_right_ml_spreadsheet"))
        c, d = f(r.get("thorax_left_ml_by_position")), f(r.get("thorax_left_ml_spreadsheet"))
        if None in (a, b, c, d):
            thor["not comparable"] += 1
        elif abs(a - b) <= 0.03 * b and abs(c - d) <= 0.03 * d:
            thor["right and left volumes match the spreadsheet (within 3%)"] += 1
        elif abs(a - d) <= 0.03 * d and abs(c - b) <= 0.03 * b:
            thor["volumes match with right and left exchanged"] += 1
        else:
            thor["neither"] += 1
    # ---- which reviewer file equals the spreadsheet volume ----
    which = collections.Counter(r["rev1_reference_reviewer"] for r in pos)
    spread = [(max(v.values()) / max(0.1, min(v.values())), r["patient_id"], v) for r in pos for v in [json.loads(r["reviewer_volumes_ml"])]]
    # ---- groups ----
    grp = collections.Counter((r["coded"], r["group"], r["tool_answer"]) for r in A)
    keys = ["reference_first_ml", "tool_ml", "dice_first", "reference_mean_hu", "tool_mean_hu", "tool_outside_reference_mean_hu", "reference_largest_inscribed_sphere_diameter_mm",
            "reference_slices", "reference_max_axial_area_cm2", "reference_fraction_tool_calls_lung", "reference_fraction_inside_tool_lung_box_plus_50mm",
            "tool_fraction_within_10mm_of_reference", "tool_fraction_inside_reference_thoracic_cavity", "tool_fraction_inside_gtv", "tool_fraction_within_10mm_of_gtv", "gtv_ml"]
    by = collections.defaultdict(list)
    for r in A:
        by[(r["coded"], r["group"])].append(r)
    gstats = [{"coded": c, "group": g, "scans": len(rs), "answers": count(rs), **{"median_" + k: med([f(r.get(k)) for r in rs], 2) for k in keys}} for (c, g), rs in sorted(by.items())]
    mostly_tumour = {f"{c}/{g}": f"{sum((f(r['tool_fraction_inside_gtv']) or 0) >= 0.5 for r in rs)} of {sum(f(r['tool_fraction_inside_gtv']) is not None for r in rs)}"
                     for (c, g), rs in sorted(by.items())}
    # ---- diagnostic what-if (NOT a method: the tumour outline is expert data that the tool does not have) ----
    wi_neg = count([{"a": ans(f(r["tool_ml_outside_tumour_outline"]))} for r in neg if r["tool_ml_outside_tumour_outline"] != ""], "a")
    wi_pos = count([{"a": ans(f(r["tool_ml_outside_tumour_outline"]))} for r in pos if r["tool_ml_outside_tumour_outline"] != ""], "a")
    d_first = [f(r["tool_ml"]) - f(r["reference_first_ml"]) for r in pos]
    d_rev1 = [f(r["tool_ml"]) - f(r["rev1_reference_ml"]) for r in pos]
    summ = {
        "limits_tested": {"present_at_least_ml": hi, "none_below_ml": lo, "changed": False},
        "detection": {"scans_coded_effusion": {"n": len(pos), **count(pos), "tool_mask_empty": sum(f(r["tool_ml"]) == 0 for r in pos),
                                               "by_reference_volume_first_reviewer": bands("reference_first_ml"), "by_reference_volume_spreadsheet_matched_file": bands("rev1_reference_ml")},
                      "scans_coded_no_effusion": {"n": len(neg), **count(neg), "tool_mask_empty": sum(f(r["tool_ml"]) == 0 for r in neg),
                                                  "tool_mask_nonzero_under_5_ml": sum(0 < f(r["tool_ml"]) < lo for r in neg)},
                      "tool_estimate_5_to_20_ml": {"coded_effusion": sum(lo <= f(r["tool_ml"]) < hi for r in pos), "coded_no_effusion": sum(lo <= f(r["tool_ml"]) < hi for r in neg)},
                      "tool_estimate_nonzero_under_5_ml": {"coded_effusion": sum(0 < f(r["tool_ml"]) < lo for r in pos), "coded_no_effusion": sum(0 < f(r["tool_ml"]) < lo for r in neg)},
                      "tool_mask_empty": {"coded_effusion": sum(f(r["tool_ml"]) == 0 for r in pos), "coded_no_effusion": sum(f(r["tool_ml"]) == 0 for r in neg)}},
        "mask_and_volume_accuracy_scans_coded_effusion": {
            "reference_first_reviewer": {"median_dice": med([f(r["dice_first"]) for r in pos], 3), "median_absolute_volume_difference_ml": med([abs(x) for x in d_first]),
                                         "median_difference_tool_minus_reference_ml": med(d_first)},
            "reference_spreadsheet_matched_file_rev1": {"median_dice": med([f(r["rev1_dice"]) for r in pos], 3), "median_absolute_volume_difference_ml": med([abs(x) for x in d_rev1]),
                                                        "median_difference_tool_minus_reference_ml": med(d_rev1)},
            "dice_by_group_first_reviewer": {g: med([f(r["dice_first"]) for r in rs], 3) for (c, g), rs in sorted(by.items()) if c == "effusion"}},
        "reference_checks": {"reviewer_file_equal_to_spreadsheet_volume": dict(which), "scans_with_reviewer_volume_ratio_over_2": [{"patient_id": p, "volumes_ml": v} for q, p, v in sorted(spread, reverse=True) if q > 2],
                             "effusion_side_image_vs_spreadsheet_unilateral_scans": dict(side), "thoracic_cavity_right_left_volume_vs_spreadsheet": dict(thor)},
        "groups": [{"coded": c, "group": g, "answer": a, "scans": n} for (c, g, a), n in sorted(grp.items())],
        "group_medians": gstats,
        "tool_mask_at_least_half_inside_tumour_outline": mostly_tumour,
        "diagnostic_what_if_tool_volume_outside_tumour_outline": {"note": "NOT a method and not a proposal: the tumour outline (GTV-1) is expert data. It only shows how much of the result is explained by tumour.",
                                                                    "scans_coded_no_effusion": wi_neg, "scans_coded_effusion": wi_pos},
        "inspection": {"overlays_rendered": len(sel["selected"]), "inspected": len(notes["notes"]), "rendered_but_not_inspected": notes["rendered_but_not_inspected"], "by": notes["who"]}}
    (run / "failure_summary.json").write_text(json.dumps(summ, indent=2), encoding="utf-8")
    cols = ["patient_id", "coded", "group", "tool_answer", "reference_first_ml", "tool_ml", "dice_first", "reviewer_volumes_ml", "reference_mean_hu", "tool_mean_hu", "tool_outside_reference_mean_hu",
            "reference_largest_inscribed_sphere_diameter_mm", "reference_slices", "reference_max_axial_area_cm2", "reference_right_ml", "reference_left_ml", "tool_right_ml", "tool_left_ml",
            "tool_components_ge_0_5ml", "reference_fraction_tool_calls_lung", "reference_fraction_inside_tool_lung_box_plus_50mm", "tool_min_distance_to_reference_mm",
            "tool_fraction_within_10mm_of_reference", "tool_fraction_inside_reference_thoracic_cavity", "gtv_status", "gtv_ml", "tool_ml_inside_gtv", "tool_fraction_inside_gtv",
            "tool_ml_outside_tumour_outline", "reference_min_distance_to_gtv_mm"]
    with open(run / "failure_groups.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(A)
    byid = {r["patient_id"]: r for r in A}
    with open(run / "failure_analysis.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols + ["overlay", "inspected", "what_the_overlay_shows", "hypothesis_not_a_finding"], extrasaction="ignore")
        w.writeheader()
        for s in sel["selected"]:
            n = notes["notes"].get(s["patient_id"])
            w.writerow({**byid[s["patient_id"]], "overlay": "overlays/" + s["file"], "inspected": "yes" if n else "no (rendered only)",
                        "what_the_overlay_shows": n["seen"] if n else "", "hypothesis_not_a_finding": n["hypothesis"] if n else ""})
    print(json.dumps({k: summ[k] for k in ("detection", "mask_and_volume_accuracy_scans_coded_effusion", "reference_checks", "groups", "tool_mask_at_least_half_inside_tumour_outline",
                                           "diagnostic_what_if_tool_volume_outside_tumour_outline")}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
