"""Evidence rows E97 / E98 from the TotalSegmentator lung_nodules mask of finished runs (core env).

Reads the stored mask data/derived/<case>/segmentation/lung_nodules/lung_nodules.nii.gz (already on the CT grid; checked)
and the lobe masks; runs no model. evidence.json is not rewritten: the rows go to <out>/evidence_rows/<case>.json together
with the rows of an earlier replay folder (E88, E92-E94), in the format scripts/rescore_with_inventory_rows.py and
build_viewer_page.py --extra-evidence-rows already read.

Definitions (same as scripts/nodule_candidates.py, E11):
  candidate            one connected outline of the predicted mask (26-connectivity)
  volume               voxel count x voxel volume (mm3)
  equivalent diameter  diameter of a sphere of that volume (NOT a measured nodule diameter)
  lobe                 lobe with the largest overlap, else the nearest lobe
A candidate is an unvalidated model output, not a nodule. No size limit and no filter is applied.

Usage: python scripts/nodule_evidence_rows.py --runs runs/A1_agent_valid_301_a_2_... [...] --base-rows runs/E15_dense_lung_inventory_replay_20261123 \
           --out runs/E19_nodule_rows_20261123
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage as ndi

ROOT = Path(__file__).resolve().parent.parent
LOBES = {"lung_upper_lobe_right": "RUL", "lung_middle_lobe_right": "RML", "lung_lower_lobe_right": "RLL",
         "lung_upper_lobe_left": "LUL", "lung_lower_lobe_left": "LLL"}
ORDER = "RUL RML RLL LUL LLL".split()


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def candidates(case_id: str) -> tuple[list[dict], str]:
    seg = ROOT / "data" / "derived" / case_id / "segmentation"
    ct = nib.load(str(ROOT / "data/raw/ct_rate/volumes" / f"{case_id}.nii.gz"))
    mp = seg / "lung_nodules" / "lung_nodules.nii.gz"

    def on_grid(p: Path) -> np.ndarray:
        img = nib.load(str(p))
        if img.shape != ct.shape or not np.allclose(img.affine, ct.affine, atol=1e-3):
            raise ValueError(f"{p.name} is not on the CT grid")
        return np.asanyarray(img.dataobj)
    nod = on_grid(mp) > 0
    hu = np.asanyarray(ct.dataobj)
    zooms = np.array(ct.header.get_zooms()[:3], float)
    vox = float(abs(np.linalg.det(ct.affine[:3, :3])))
    lobes = np.zeros(ct.shape, np.uint8)
    names = list(LOBES)
    for i, n in enumerate(names, 1):
        lobes[on_grid(seg / "total" / f"{n}.nii.gz") > 0] = i
    idx = None
    zax = [a in "SI" for a in nib.aff2axcodes(ct.affine)].index(True)
    lab, _ = ndi.label(nod, structure=np.ones((3, 3, 3)))
    out = []
    for k, sl in enumerate(ndi.find_objects(lab), 1):
        m = lab[sl] == k
        nv = int(m.sum())
        inside = np.bincount(lobes[sl][m], minlength=6)
        c = tuple(int(round(x)) for x in np.array(ndi.center_of_mass(m)) + [s.start for s in sl])
        if inside[1:].sum() > 0:
            lobe, how = LOBES[names[int(inside[1:].argmax())]], "largest overlap"
        else:
            if idx is None:
                _, idx = ndi.distance_transform_edt(lobes == 0, sampling=zooms, return_indices=True)
            j = tuple(int(idx[d][c]) for d in range(3))
            lobe, how = LOBES[names[lobes[j] - 1]], "nearest lobe"
        out.append({"voxels": nv, "volume_mm3": round(nv * vox, 1), "equivalent_diameter_mm": round((6 * nv * vox / math.pi) ** (1 / 3), 1),
                    "extent_mm": [round((s.stop - s.start) * z, 1) for s, z in zip(sl, zooms)], "lobe": lobe, "lobe_how": how,
                    "source_ct_slice_index": int(c[zax]), "source_ct_slices": int(ct.shape[zax]), "median_hu": round(float(np.median(hu[sl][m])))})
    out.sort(key=lambda r: -r["volume_mm3"])
    for i, r in enumerate(out, 1):
        r["candidate"] = i
    return out, sha(mp)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--base-rows", required=True, help="replay folder whose evidence_rows/<case>.json (E88, E92-E94) are carried over")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = ROOT / args.out
    (out / "evidence_rows").mkdir(parents=True, exist_ok=True)
    summ = []
    for rn in args.runs:
        run = ROOT / rn
        ev = json.loads((run / "evidence.json").read_text(encoding="utf-8"))
        cid = ev["case_id"]
        base = json.loads((ROOT / args.base_rows / "evidence_rows" / f"{cid}.json").read_text(encoding="utf-8"))
        if base["run"] != run.name:
            raise SystemExit(f"{cid}: base rows belong to run {base['run']}")
        cands, mask_sha = candidates(cid)
        lobes = sorted({c["lobe"] for c in cands}, key=ORDER.index)
        rel = "low - output of an unvalidated segmentation model; a candidate can be consolidation, a mass, a vessel or scar"
        src = "TotalSegmentator task lung_nodules, stored mask (no new inference); connected outlines, 26-connectivity"
        text = "; ".join(f"{c['lobe']} {c['equivalent_diameter_mm']:g} mm" for c in cands) or "none outlined"   # no image number: the page crops the scan, so its numbering differs
        rows = [r for r in base["rows"] if r["id"] not in ("E97", "E98")] + [
            {"id": "E97", "group": "nodule candidates", "name": "Nodule candidates outlined by the segmentation model (count)", "value": len(cands), "unit": "candidates",
             "reliability": rel, "source": src, "note": "candidates, not nodules; no size limit and no filter applied; zero candidates does not exclude a nodule"},
            {"id": "E98", "group": "nodule candidates", "name": "Nodule candidates: lobe and equivalent-sphere diameter, largest first", "value": text, "unit": None,
             "reliability": rel, "source": src, "where": ", ".join(lobes) if lobes else None, "candidates": cands,
             "note": "equivalent-sphere diameter from the outlined volume: not a measured nodule diameter; lobe = largest overlap or nearest lobe; "
                     "listed largest first"}]
        rec = {"case_id": cid, "run": run.name, "evidence_sha256": sha(run / "evidence.json"), "lung_nodules_mask_sha256": mask_sha,
               "base_rows": f"{args.base_rows}/evidence_rows/{cid}.json", "rows": rows}
        (out / "evidence_rows" / f"{cid}.json").write_text(json.dumps(rec, indent=1), encoding="utf-8")
        summ.append({"case_id": cid, "candidates": len(cands), "lobes": lobes, "equivalent_diameters_mm": [c["equivalent_diameter_mm"] for c in cands], "mask_sha256": mask_sha})
        print(cid, len(cands), text, flush=True)
    (out / "nodule_rows_summary.json").write_text(json.dumps(summ, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
