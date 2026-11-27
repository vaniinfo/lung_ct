"""Image-content check of PleThora outline alignment, per reviewer file (core env).

Header agreement (same grid after reorientation) was shown before. This checks the CONTENT: a pleural-fluid outline
lies between aerated lung and ribs, so a misplaced outline picks up bone or air. For every reviewer file the share
of outline voxels that are bone (> 200 HU) or air-like (< -500 HU) is computed as stored, and again after shifting the
outline by up to 3 voxels in the slice plane and 1 slice, and after mirroring it left-right and front-back.
A correctly placed outline should be at (or next to) the minimum, and far better than the mirrored copies.
Reviewer files are kept separate; nothing is merged.

Usage: python scripts/plethora_alignment_check.py --out runs/E16_.../alignment_by_reviewer_file.csv
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import re
import sys
from pathlib import Path

import nibabel as nib
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
P = ROOT / "data" / "raw" / "plethora"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    sel = json.loads((P / "selection.json").read_text(encoding="utf-8"))["cases"]
    rows = []
    for c in sel:
        pid = c["patient_id"]
        d = P / "effusions" / "Effusions" / pid
        ctp = P / "volumes" / f"{pid}.nii.gz"
        if not d.exists() or not ctp.exists():
            continue
        cti = nib.as_closest_canonical(nib.load(ctp))
        ct = np.asarray(cti.dataobj)
        bad = (ct > 200) | (ct < -500)
        for f in sorted(d.glob("*.nii.gz")):
            name = re.search(r"effusion_(\w+)_reviewer", f.name).group(1)
            mi = nib.as_closest_canonical(nib.load(f))
            same_grid = mi.shape == cti.shape and bool(np.allclose(mi.affine, cti.affine, atol=1e-2))
            r = {"patient_id": pid, "reviewer_file": name, "same_grid_as_ct": same_grid}
            if same_grid:
                m = np.asarray(mi.dataobj) > 0
                n = int(m.sum())
                idx = np.argwhere(m)
                lo, hi = np.maximum(idx.min(0) - 4, 0), np.minimum(idx.max(0) + 5, m.shape)
                sub, bsub = m[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]], bad
                score = {}
                for dx, dy, dz in itertools.product(range(-3, 4), range(-3, 4), (-1, 0, 1)):
                    a0, a1 = lo + (dx, dy, dz), hi + (dx, dy, dz)
                    if (a0 < 0).any() or (a1 > np.array(m.shape)).any():
                        continue
                    score[(dx, dy, dz)] = float((sub & bsub[a0[0]:a1[0], a0[1]:a1[1], a0[2]:a1[2]]).sum()) / n
                s0 = score[(0, 0, 0)]
                best = min(score, key=score.get)
                r.update(voxels=n, bone_or_air_fraction_as_stored=round(s0, 4), best_shift_voxels="%d,%d,%d" % best, bone_or_air_fraction_at_best_shift=round(score[best], 4),
                         rank_of_stored_position=1 + sum(v < s0 for v in score.values()), positions_tested=len(score),
                         best_shift_within_1_voxel=bool(max(abs(v) for v in best) <= 1),
                         mirrored_left_right=round(float((m[::-1] & bad).sum()) / n, 4), mirrored_front_back=round(float((m[:, ::-1] & bad).sum()) / n, 4))
            rows.append(r)
        print(pid, [(r["reviewer_file"], r.get("bone_or_air_fraction_as_stored"), r.get("best_shift_voxels"), r.get("mirrored_left_right")) for r in rows if r["patient_id"] == pid], flush=True)
    cols = list(dict.fromkeys(k for r in rows for k in r))
    out = ROOT / args.out
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())
