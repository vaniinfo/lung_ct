"""Run the primary audit checks on a finding inventory (core env). Writes verdict records, kept apart from the
inventory so the blinded viewer never loads them.

For each finding: parse the statement (fixed keyword rules), assign every outline voxel to a lung lobe
(inside a TotalSegmentator lobe, or the nearest lobe within a set distance for pleural findings), then apply
the checks in lung_ct.audit.checks. Attenuation inside the outline is stored as descriptive statistics only.
Geometry is verified first (outline, CT and lobe masks must share shape and affine); a failed case is reported.

Output: runs/A3_audit_verdicts_<label>_<time>/{verdicts.json, verdicts.md}
Usage:  python scripts/audit_run_checks.py --inventory runs/A2_audit_inventory_<..>/inventory.json [--label usability]
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from lung_ct.agent.evidence import LOBES  # noqa: E402
from lung_ct.audit.checks import CHECK_VERSION, PRIMARY, THRESHOLDS, audit_finding, case_level_answer_gaps, describe_attenuation  # noqa: E402
from lung_ct.audit.parse import PARSER_VERSION  # noqa: E402
from lung_ct.audit.schema import VERDICT_SCHEMA  # noqa: E402

SHORT = [v[2] for v in LOBES.values()]      # RUL, RML, RLL, LUL, LLL (label 1..5)


def lobe_assignment(mask: np.ndarray, lobe_lab: np.ndarray, spacing, max_mm: float):
    """Share of the outline per lobe. Voxels inside a lobe count for it; voxels outside the lobes go to the nearest
    lobe if it is within max_mm (physical distance), else they stay unassigned."""
    n = int(mask.sum())
    if n == 0:
        return None, None, {}
    idx = np.argwhere(mask)
    pad = int(np.ceil(max_mm / min(spacing))) + 2
    lo = np.maximum(idx.min(0) - pad, 0)
    hi = np.minimum(idx.max(0) + pad + 1, mask.shape)
    sl = tuple(slice(a, b) for a, b in zip(lo, hi))
    m, lab = mask[sl], lobe_lab[sl]
    inside = lab[m]
    counts = np.bincount(inside, minlength=6).astype(np.int64)
    out = m & (lab == 0)
    n_near = 0
    if out.any() and (lab > 0).any():
        dist, ind = ndimage.distance_transform_edt(lab == 0, sampling=spacing, return_indices=True)
        near = out & (dist <= max_mm)
        n_near = int(near.sum())
        if n_near:
            src = tuple(ind[d][near] for d in range(3))
            counts += np.bincount(lab[src], minlength=6)
    assigned = int(counts[1:].sum())
    share = {s: (float(counts[i + 1]) / assigned if assigned else 0.0) for i, s in enumerate(SHORT)}
    info = {"voxels": n, "inside_lobes": int((inside > 0).sum()), "assigned_to_nearest_lobe": n_near,
            "unassigned": n - assigned}
    return (share if assigned else None), assigned / n, info


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--inventory", required=True, type=Path)
    ap.add_argument("--label", default="")
    args = ap.parse_args()
    ip = args.inventory if args.inventory.is_absolute() else ROOT / args.inventory
    inv = json.loads(ip.read_text(encoding="utf-8"))
    answers = {a["case_id"]: {x["condition"]: x["answer"] for x in a["answers"]} for a in inv["fixed_condition_answers"]}
    by_case: dict[str, list] = {}
    for f in inv["findings"]:
        by_case.setdefault(f["case_id"], []).append(f)
    verdicts, cases, failed = [], [], []
    for cid, fs in sorted(by_case.items()):
        try:
            ct = nib.load(str(ROOT / fs[0]["ct"]["file"]))
            spacing = tuple(float(x) for x in ct.header.get_zooms()[:3])
            vox_ml = abs(float(np.linalg.det(ct.affine[:3, :3]))) / 1000.0

            def load(path: Path):
                img = nib.load(str(path))
                if img.shape != ct.shape or not np.allclose(img.affine, ct.affine, atol=1e-3):
                    raise ValueError(f"geometry differs from the CT: {path.name}")
                return np.asanyarray(img.dataobj)

            seg = ROOT / "data" / "derived" / cid / "segmentation" / "total"
            lab = np.zeros(ct.shape, np.uint8)
            for i, name in enumerate(LOBES, 1):
                lab[load(seg / f"{name}.nii.gz") > 0] = i
            hu = np.asanyarray(ct.dataobj)
            parsed_all = []
            for f in sorted(fs, key=lambda x: x["index"]):
                mask = load(ROOT / f["mask"]["file"]) > 0
                n = int(mask.sum())
                share, assigned, info = lobe_assignment(mask, lab, spacing, THRESHOLDS["nearest_lobe_max_mm"])
                res = audit_finding(f["statement"], n, share, assigned, answers.get(cid))
                parsed_all.append(res["parsed"])
                verdicts.append({
                    "schema": VERDICT_SCHEMA, "finding_id": f["finding_id"], "case_id": cid, "patient_id": f["patient_id"],
                    "statement": f["statement"], "parsed": res["parsed"], "checks": res["checks"],
                    "any_primary_flag": res["any_primary_flag"],
                    "flags": [k for k in PRIMARY if res["checks"][k]["status"] == "flag"],
                    "measurements": {"volume_ml": round(n * vox_ml, 3), "assigned_share": None if assigned is None else round(assigned, 4),
                                     "lobe_share": None if share is None else {k: round(v, 4) for k, v in share.items()},
                                     "assignment": info},
                    "attenuation_descriptive": describe_attenuation(hu[mask]) if n else None,
                })
            cases.append({"case_id": cid, "n_findings": len(fs),
                          "answered_yes_without_finding": case_level_answer_gaps(answers.get(cid, {}), parsed_all)})
        except Exception as e:  # reported, never dropped
            failed.append({"case_id": cid, "error": f"{type(e).__name__}: {e}"})
    tag = f"{args.label}_" if args.label else ""
    out = ROOT / "runs" / f"A3_audit_verdicts_{tag}{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out.mkdir(parents=True)
    n_flag = sum(v["any_primary_flag"] for v in verdicts)
    body = {"schema": "audit_verdicts_v1", "inventory": ip.relative_to(ROOT).as_posix(), "inventory_sha256": inv["inventory_sha256"],
            "check_version": CHECK_VERSION, "parser_version": PARSER_VERSION, "thresholds": THRESHOLDS,
            "created_utc": datetime.now(timezone.utc).isoformat(), "n_findings": len(verdicts), "n_flagged": n_flag,
            "meaning_of_pass": "No tested inconsistency was found. It does not mean the finding is correct.",
            "verdicts": verdicts, "cases": cases, "failed_cases": failed}
    (out / "verdicts.json").write_text(json.dumps(body, indent=2, ensure_ascii=False), encoding="utf-8")
    md = [f"# Audit verdicts ({len(verdicts)} findings, {n_flag} with a primary flag)", "",
          f"{CHECK_VERSION}, {PARSER_VERSION}; thresholds provisional. A pass means no tested inconsistency was found.", "",
          "| case | statement | mL | empty | side | lobe | contradiction |", "|---|---|---|---|---|---|---|"]
    mark = {"flag": "FLAG", "pass": "pass", "not_applicable": "n/a"}
    for v in verdicts:
        c = v["checks"]
        md.append(f"| {v['case_id']} | {v['statement']} | {v['measurements']['volume_ml']:g} | "
                  + " | ".join(f"{mark[c[k]['status']]}: {c[k]['detail']}" if c[k]["status"] != "pass" or k in ("side", "lobe")
                               else "pass" for k in PRIMARY) + " |")
    md += ["", "Scan level (not finding-level flags): conditions answered Yes with no generated finding stating them"]
    md += [f"- {c['case_id']}: {', '.join(c['answered_yes_without_finding']) or 'none'}" for c in cases]
    md += [f"- FAILED {f['case_id']}: {f['error']}" for f in failed]
    (out / "verdicts.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
