"""Record the same-day revision of the checklist_v3_8 files (core env). Reads the current files; rewrites none of them.

On 2026-11-23 the 58 files runs/A1_agent_*/checklist_checklist_v3_8.json were written, then REWRITTEN the same day
(revision 2) after two stored strings were reworded. The first revision was not under version control. This script
  - records the SHA-256 of every current (revision 2) file,
  - reconstructs revision 1 from revision 2 by putting the two old strings back, saves the reconstructions, and
  - checks the reconstruction against the four revision-1 hashes that were recorded before the rewrite
    (runs/test_logs/M5_hashes_before.txt). Only those four are verified; the other 54 are reconstructions.
Usage: python scripts/record_v3_8_revision.py --out runs/E17_.../v3_8_revision_record
"""
import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NEW_REASON = "pleural fluid outlined by the segmentation model (lobe outlines are less reliable next to it)"
OLD_REASON = "pleural effusion present (compressed lung)"


def swap(o) -> int:
    """Put the old reason string back wherever the new one stands as a list element (also in nested location parts)."""
    n = 0
    if isinstance(o, dict):
        for v in o.values():
            n += swap(v)
    elif isinstance(o, list):
        for k, v in enumerate(o):
            if v == NEW_REASON:
                o[k] = OLD_REASON
                n += 1
            else:
                n += swap(v)
    return n


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--pages-file", default="runs/test_logs/pages_58runs_with_v3_4.txt")
    args = ap.parse_args()
    out = ROOT / args.out
    (out / "rev1_reconstructed").mkdir(parents=True, exist_ok=True)
    before = {}
    for line in (ROOT / "runs/test_logs/M5_hashes_before.txt").read_text(encoding="utf-8").splitlines():
        m = re.match(r"([0-9a-f]{64}) \*?(.*checklist_checklist_v3_8\.json)$", line.strip())
        if m:
            before[m.group(2).split("/")[0]] = m.group(1)
    runs = [ROOT / Path(x.strip().replace("\\", "/")).parent for x in (ROOT / args.pages_file).read_text(encoding="utf-8").splitlines() if x.strip()]
    rec = []
    for run in runs:
        f = run / "checklist_checklist_v3_8.json"
        raw = f.read_bytes()
        d = json.loads(raw)
        changed = []
        for it in d["items"]:
            n = swap(it)
            if n:
                changed.append(f"{it['key']}: lobe-uncertainty reason ({n} place(s))")
            if it["key"] == "pleural_effusion" and it.get("basis", "").startswith("predicted mask only ("):
                it["basis"] = f"measured ({it['measured_value']:g} mL; present >= 20, absent < 5)"
                changed.append("pleural_effusion: basis")
        r1 = json.dumps(d, indent=2).encode("utf-8").replace(bytes([10]), bytes([13, 10]))     # the files were written in Windows text mode
        (out / "rev1_reconstructed" / f"{run.name}__checklist_checklist_v3_8.json").write_bytes(r1)
        h1 = hashlib.sha256(r1).hexdigest()
        rec.append({"run": run.name, "revision_2_sha256_current_file": hashlib.sha256(raw).hexdigest(), "revision_1_sha256_reconstructed": h1,
                    "revision_1_sha256_recorded_before_rewrite": before.get(run.name), "reconstruction_matches_recorded_hash": (h1 == before[run.name]) if run.name in before else None,
                    "strings_changed_between_revisions": changed, "content_differs": bool(changed)})
    ver = [r for r in rec if r["reconstruction_matches_recorded_hash"] is not None]
    summ = {"what": "checklist_checklist_v3_8.json: revision 1 written and overwritten on 2026-11-23; revision 2 is current",
            "statuses_changed_between_revisions": 0, "files": len(rec), "files_whose_content_differs": sum(r["content_differs"] for r in rec),
            "revision_1_hashes_recorded_before_rewrite": len(ver), "reconstructions_matching_those_hashes": sum(r["reconstruction_matches_recorded_hash"] for r in ver),
            "config_sha256_current": hashlib.sha256((ROOT / "configs/checklist_v3_8.json").read_bytes()).hexdigest(),
            "note": "revision 1 of the other files is a reconstruction, not a preserved copy; the config's first revision was not preserved either", "records": rec}
    (out / "v3_8_revision_record.json").write_text(json.dumps(summ, indent=1), encoding="utf-8")
    print({k: v for k, v in summ.items() if k != "records"})
    return 0


if __name__ == "__main__":
    sys.exit(main())
