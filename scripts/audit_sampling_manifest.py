"""Sampling manifest for clinician review (core env): all flagged findings + a random sample of unflagged ones.

Writes two files on purpose:
  manifest.json    complete finding inventory with stratum, inclusion probability, seed and selection.
                   For analysis only. NEVER given to the blinded viewer or the reader.
  selection.json   the selected finding ids only (no strata, no flags), sorted. This is what
                   build_blinded_viewer.py --selection reads.
Both record the inventory hash and the verdict file hash, and must be written before grading starts.

Usage: python scripts/audit_sampling_manifest.py --inventory runs/A2_.../inventory.json
           --verdicts runs/A3_.../verdicts.json --n-unflagged 40 --seed 20261120
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from lung_ct.audit.sampling import build_manifest  # noqa: E402


def sha256_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--inventory", required=True, type=Path)
    ap.add_argument("--verdicts", required=True, type=Path)
    ap.add_argument("--n-unflagged", required=True, type=int)
    ap.add_argument("--seed", required=True, type=int)
    ap.add_argument("--label", default="")
    args = ap.parse_args()
    ip = args.inventory if args.inventory.is_absolute() else ROOT / args.inventory
    vp = args.verdicts if args.verdicts.is_absolute() else ROOT / args.verdicts
    inv = json.loads(ip.read_text(encoding="utf-8"))
    ver = json.loads(vp.read_text(encoding="utf-8"))
    if ver["inventory_sha256"] != inv["inventory_sha256"]:
        raise SystemExit("verdicts were computed for a different inventory")
    if ver["failed_cases"]:
        raise SystemExit(f"verdicts have failed cases {[c['case_id'] for c in ver['failed_cases']]}: fix or exclude them explicitly first")
    flag = {v["finding_id"]: v["any_primary_flag"] for v in ver["verdicts"]}
    missing = [f["finding_id"] for f in inv["findings"] if f["finding_id"] not in flag]
    if missing:
        raise SystemExit(f"{len(missing)} findings have no verdict, e.g. {missing[0]}")
    m = build_manifest([{"finding_id": f["finding_id"], "patient_id": f["patient_id"], "flagged": flag[f["finding_id"]]}
                        for f in inv["findings"]], args.n_unflagged, args.seed)
    m.update(inventory_sha256=inv["inventory_sha256"], verdicts_file=vp.relative_to(ROOT).as_posix(), verdicts_sha256=sha256_file(vp),
             check_version=ver["check_version"], created_utc=datetime.now(timezone.utc).isoformat(),
             scope="Weighted estimates concern errors among generated findings only, not abnormalities the model never mentioned.")
    tag = f"{args.label}_" if args.label else ""
    out = ROOT / "runs" / f"A4_audit_sampling_{tag}{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out.mkdir(parents=True)
    (out / "manifest.json").write_text(json.dumps(m, indent=2), encoding="utf-8")
    sel = {"inventory_sha256": inv["inventory_sha256"], "finding_ids": sorted(r["finding_id"] for r in m["rows"] if r["selected"])}
    (out / "selection.json").write_text(json.dumps(sel, indent=2), encoding="utf-8")
    print(f"{m['n_findings']} findings: {m['n_flagged']} flagged (all selected), {m['n_unflagged_selected']} of {m['n_unflagged']} "
          f"unflagged sampled (seed {args.seed}); {m['n_selected']} selected in {m['n_patients_selected']} patients")
    print(f"manifest sha256 {sha256_file(out / 'manifest.json')[:12]} -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
