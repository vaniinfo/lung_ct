"""Development-only comparison (core env): keep posterior and fluid-adjacent dense lung as TAGGED evidence.

Baseline (unchanged, checklist v3_x): dense lung in the consolidation HU range that lies within 15 mm of outlined
pleural fluid is moved to the "compressed lung" measure (and through it towards atelectasis), and dense lung in the
posterior band is dropped. Both disappear from the consolidation evidence.

This script changes nothing in the baseline. It writes a separate evidence inventory in which every dense-lung voxel
that survives the vessel / airway removal and the opening step carries exactly one tag:

  baseline_measured              not next to fluid, not posterior      (= the baseline consolidation measure)
  fluid_adjacent                 within 15 mm of outlined pleural fluid, not posterior
  posterior                      in the posterior band, not next to fluid
  fluid_adjacent_and_posterior   both

Nothing is assigned to a diagnosis: a tagged region is NOT called consolidation and NOT called atelectasis, no
threshold is lowered and no checklist answer is recomputed. The baseline value is recomputed and must equal the stored
evidence. A standalone review page asks a clinician what each retained region represents (consolidation, atelectasis,
both, another structure or process, cannot be assessed), with outline adequacy as a separate question and scan-level
presence of each condition as further separate questions. The tag is not shown on the page.

Usage: python scripts/dev_audit_retained_regions.py --cases valid_301_a_2 valid_541_a_2 valid_832_a_2 valid_988_a_2
           --pages-file ... --out runs/D1_dev_audit_20261122
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from scipy import ndimage

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from dev_audit_candidates import LOBE_ORDER, choice, page_numbers, review_page, three_plane  # noqa: E402
from dev_audit_density import Case  # noqa: E402
from lung_ct.agent.checklist import load_checklist_config  # noqa: E402

TAGS = ["baseline_measured", "fluid_adjacent", "posterior", "fluid_adjacent_and_posterior"]
MAX_CARDS_PER_TAG = 3          # review practicality only: how many pieces of one tag get a card. Not a diagnostic limit.
COVER = 0.8                    # cards are added until this share of the tag's volume is shown (or the maximum is reached)
INVENTORY_VERSION = "dense_region_inventory_dev_v1"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cases", nargs="+", required=True)
    ap.add_argument("--pages-file", nargs="+", required=True)
    ap.add_argument("--config", default="configs/checklist_v3_4.json")
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    out = (args.out if args.out.is_absolute() else ROOT / args.out) / "retained_regions"
    out.mkdir(parents=True, exist_ok=True)
    ck = load_checklist_config(ROOT / args.config)
    runs = {}
    for pf in args.pages_file:
        for line in (ROOT / pf).read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = ROOT / Path(line.strip().replace("\\", "/")).parent
                runs[json.loads((r / "evidence.json").read_text(encoding="utf-8"))["case_id"]] = r
    inv, pieces_rows, cards, items = [], [], "", []
    for cid in args.cases:
        print(f"[{cid}]", flush=True)
        c = Case(cid, runs[cid], ck)
        st = c.stages("consolidation")
        s4 = st[3]                                           # dense lung after erosion, vessel / airway removal and opening
        near = c.near_eff if c.near_eff is not None else np.zeros(s4.shape, bool)
        masks = {"baseline_measured": s4 & ~near & ~c.dep, "fluid_adjacent": s4 & near & ~c.dep,
                 "posterior": s4 & c.dep & ~near, "fluid_adjacent_and_posterior": s4 & near & c.dep}
        assert int(sum(m.sum() for m in masks.values())) == int(s4.sum()), "tags must partition the region"
        assert np.array_equal(masks["baseline_measured"], st[5]), "baseline tag must equal the baseline measured region"
        stored = c.rows["E85"]["value"]
        base_ml = round(float(masks["baseline_measured"].sum()) * c.vml)
        if base_ml != stored:
            raise SystemExit(f"{cid}: baseline not reproduced ({base_ml} vs stored {stored} mL)")
        rec = {"case_id": cid, "inventory_version": INVENTORY_VERSION, "baseline_consolidation_ml_stored_E85": stored, "baseline_reproduced": True,
               "baseline_next_to_fluid_ml_stored_E88": c.rows.get("E88", {}).get("value"), "pleural_fluid_outlined_ml": round(float(near.any()) and float(c.pe.sum()) * c.vml, 1) if c.pe is not None else 0.0,
               "dense_lung_before_the_two_exclusions_ml": c.ml(s4), "tags": {}}
        n_card = 0
        for tag in TAGS:
            m = masks[tag]
            lab, n = ndimage.label(m)
            sizes = np.bincount(lab.ravel())[1:] * c.vml if n else np.array([])
            order = np.argsort(-sizes) if n else []
            rec["tags"][tag] = {"ml": c.ml(m), "per_lobe_ml": c.per_lobe(m), "pieces": int(n), "largest_piece_ml": round(float(sizes.max()), 2) if n else 0.0,
                                "median_hu": round(float(np.median(c.hu_r[m]))) if m.any() else None, "cards": 0}
            shown, k = 0.0, 0
            tot = float(sizes.sum())
            for idx in order:
                if k >= MAX_CARDS_PER_TAG or (tot > 0 and shown / tot >= COVER) or tot == 0:
                    break
                piece = lab == (int(idx) + 1)
                cen = np.array(ndimage.center_of_mass(piece))
                lv = np.bincount(c.lobe_idx[piece], minlength=6)
                sl = ndimage.find_objects(piece.astype(np.uint8))[0]
                ext = max((s.stop - s.start) * z for s, z in zip(sl, c.zr))
                n_card += 1
                pid = f"{cid}|dense_region|{n_card:02d}"
                pn = page_numbers(c, cen)
                pieces_rows.append({"region_id": pid, "case_id": cid, "tag": tag, "ml": round(float(sizes[idx]), 2), "share_of_tag_pct": round(100 * float(sizes[idx]) / tot, 1),
                                    "tool_lobe": c.short[int(lv[1:].argmax())] if lv[1:].sum() else "outside lobes", "median_hu": round(float(np.median(c.hu_r[piece]))),
                                    "longest_extent_mm": round(float(ext), 1), "viewer_axial_image": pn["axial"], "viewer_coronal_image": pn["coronal"], "viewer_sagittal_image": pn["sagittal"]})
                half = max(40.0, float(ext) * 0.8)
                a = three_plane(c, cen, piece, half, (-1350, 150), False)
                b = three_plane(c, cen, piece, half, (-1350, 150), True, "#ff3b30")
                d = three_plane(c, cen, piece, half, (-160, 240), False)
                items.append({"id": pid, "case_id": cid})
                cards += (f"<div class='card' data-item='{pid}'><span class='tag'>{cid.replace('_a_2', '')}</span><b>Dense-lung region {n_card}</b> "
                          f"<span class='muted'>&middot; in the viewer: axial image {pn['axial']}, coronal {pn['coronal']}, sagittal {pn['sagittal']} &middot; outline volume {sizes[idx]:.1f} mL</span>"
                          f"<div class='imgs'><div><div class='muted'>plain, lung window</div><img src='data:image/png;base64,{a}'></div>"
                          f"<div><div class='muted'>with the outline in red, lung window</div><img src='data:image/png;base64,{b}'></div>"
                          f"<div><div class='muted'>plain, soft-tissue window</div><img src='data:image/png;base64,{d}'></div></div>"
                          f"<div class='q'><b>1. What does the outlined region represent?</b>"
                          f"{choice('represents', [('consolidation', 'Consolidation'), ('atelectasis', 'Atelectasis'), ('both', 'Both'), ('other', 'Another structure or process'), ('cannot_assess', 'Cannot be assessed')])}"
                          f" <input data-t='other_what' placeholder='if another structure or process: what?' style='width:340px'></div>"
                          f"<div class='q'><b>2. Separately: is the red outline adequate for that region?</b>"
                          f"{choice('mask', [('adequate', 'Adequate'), ('too_large', 'Too large'), ('too_small', 'Too small'), ('wrong_place', 'Wrong place'), ('uncertain', 'Uncertain')])}</div>"
                          f"<div class='q'><b>Lobe</b>{choice('lobe', [(x, x) for x in LOBE_ORDER] + [('indeterminate', 'Indeterminate'), ('not_assessed', 'Not assessed')], multi=True, exclusive=('indeterminate', 'not_assessed'))}</div>"
                          f"<div class='q'><textarea data-t='comment' placeholder='comment (optional)'></textarea></div></div>")
                shown += float(sizes[idx])
                k += 1
            rec["tags"][tag]["cards"] = k
            rec["tags"][tag]["ml_shown_on_cards"] = round(shown, 1)
        sid = f"{cid}|scan_level"
        items.append({"id": sid, "case_id": cid})
        cards += (f"<div class='card' data-item='{sid}'><span class='tag'>{cid.replace('_a_2', '')}</span><b>This scan as a whole</b> <span class='muted'>(separate from the outlines above)</span>"
                  f"<div class='q'><b>Consolidation in this scan</b>{choice('consolidation', [('present', 'Present'), ('absent', 'Absent'), ('cannot_tell', 'Cannot tell')])}</div>"
                  f"<div class='q'><b>Atelectasis in this scan</b>{choice('atelectasis', [('present', 'Present'), ('absent', 'Absent'), ('cannot_tell', 'Cannot tell')])}</div>"
                  f"<div class='q'><b>Dense lung that you see and that none of the outlines above covers</b>{choice('uncovered', [('none', 'None'), ('some', 'Yes'), ('not_assessed', 'Not assessed')])}</div>"
                  f"<div class='q'><textarea data-t='comment' placeholder='where / comment (optional)'></textarea></div></div>")
        inv.append(rec)
        print("  ", {t: (v["ml"], v["pieces"], v["cards"]) for t, v in rec["tags"].items()}, flush=True)
        del c
    (out / "tagged_dense_regions.json").write_text(json.dumps({
        "inventory_version": INVENTORY_VERSION, "status": "DEVELOPMENT ONLY. No diagnosis is assigned to a tag, no threshold changed, baseline evidence and answers untouched.",
        "tag_definitions": {"baseline_measured": "dense lung not within 15 mm of outlined pleural fluid and not in the posterior band (= baseline consolidation measure E85)",
                            "fluid_adjacent": "within 15 mm of outlined pleural fluid, not in the posterior band (baseline: counted as compressed lung, E88)",
                            "posterior": "in the posterior band (most posterior 20% of each lung per image), not next to fluid (baseline: dropped, reported only in E90)",
                            "fluid_adjacent_and_posterior": "both (baseline: counted as compressed lung, E88)"},
        "review_card_selection": f"largest pieces of each tag until {int(COVER * 100)}% of its volume is shown, at most {MAX_CARDS_PER_TAG} per tag; a practical limit for the review page, not a diagnostic rule",
        "cases": inv}, indent=2), encoding="utf-8")
    with open(out / "tagged_dense_regions.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["case_id", "baseline_consolidation_ml_E85", "tag", "ml", "pieces", "largest_piece_ml", "median_hu"] + [f"ml_{n}" for n in LOBE_ORDER] + ["cards_on_review_page", "ml_shown_on_cards"])
        for r in inv:
            for t in TAGS:
                v = r["tags"][t]
                w.writerow([r["case_id"], r["baseline_consolidation_ml_stored_E85"], t, v["ml"], v["pieces"], v["largest_piece_ml"], v["median_hu"]] + [v["per_lobe_ml"][n] for n in LOBE_ORDER] + [v["cards"], v["ml_shown_on_cards"]])
    if pieces_rows:
        with open(out / "review_regions_key_ANALYST_ONLY.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(pieces_rows[0].keys()))
            w.writeheader()
            w.writerows(pieces_rows)
    (out / "dense_region_review.html").write_text(review_page(
        "Dense lung regions: what are they?",
        "Each card shows ONE region of dense lung outlined automatically from CT density. The tool does not know what it is. Please say what the region represents, and, "
        "as a separate question, whether the red outline is adequate. At the end of each scan there are questions about the scan as a whole. "
        "Images are centred on the region; patient's right is on the image left.",
        "dense_regions", "dense_region_review_v1", cards, items, ["represents", "other_what", "mask", "lobe", "consolidation", "atelectasis", "uncovered", "comment"], "represents"), encoding="utf-8")
    print(f"{len(pieces_rows)} region cards, {len(args.cases)} scan-level cards -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
