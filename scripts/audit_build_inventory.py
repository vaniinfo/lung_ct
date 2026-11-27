"""Finding inventory for the audit (core env): every finding-mask pair a HoloCT run emitted, as versioned records.

Reads ONLY the model run (runs/P3_holoct_*): statements, masks, yes/no answers and provenance. It never reads
CT-RATE labels, report text, the lung_ct checklist or any audit verdict, so the inventory is safe to hand to
the blinded viewer (scripts/build_blinded_viewer.py).

Output: runs/A2_audit_inventory_<label>_<time>/inventory.json
  findings[]                  audit_finding_v1 records (one per generated finding with its mask reference)
  fixed_condition_answers[]   the model's yes/no answers per case (compared separately from mask geometry)
  provenance                  model, code commit, weights and config hashes, source run, inventory hash inputs

Usage: python scripts/audit_build_inventory.py --run runs/P3_holoct_meeting4_<time> [--label usability]
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
from lung_ct.audit.schema import FINDING_SCHEMA, INVENTORY_SCHEMA, finding_id, patient_id  # noqa: E402

SOURCE = "findings_grounded"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run", required=True, type=Path)
    ap.add_argument("--label", default="")
    args = ap.parse_args()
    run = args.run if args.run.is_absolute() else ROOT / args.run
    man = json.loads((run / "run_manifest.json").read_text(encoding="utf-8"))
    cfg = json.loads((ROOT / man["config"]).read_text(encoding="utf-8"))
    model = {"repo": man["model_repo"], "code_commit": man["holoct_code_commit"], "weights_sha256": man["weights_sha256"],
             "prompt_config": cfg["config_version"], "prompt_config_sha256": man["config_sha256"],
             "run_id": run.name, "torch": man.get("torch"), "transformers": man.get("transformers")}
    findings, answers, problems = [], [], []
    for cid in man["cases"]:
        cdir = run / cid
        ct = ROOT / "data/raw/ct_rate/volumes" / f"{cid}.nii.gz"
        if not (cdir / "answers.json").is_file():
            problems.append({"case_id": cid, "problem": "no answers.json (case failed or not run)"})
            continue
        a = json.loads((cdir / "answers.json").read_text(encoding="utf-8"))
        answers.append({"case_id": cid, "patient_id": patient_id(cid), "run_id": run.name, "ct_sha256": a["ct_sha256"],
                        "answers": [{"condition": x["ct_rate_label"], "prompt": x["prompt"], "raw": x["raw"],
                                     "answer": x["answer"]} for x in a["answers"]]})
        g = cdir / SOURCE / "result.json"
        if not g.is_file():
            problems.append({"case_id": cid, "problem": f"no {SOURCE}/result.json (grounded findings failed or not run)"})
            continue
        res = json.loads(g.read_text(encoding="utf-8"))
        for m in res["masks"]:
            mf = cdir / SOURCE / m["file"]
            findings.append({
                "schema": FINDING_SCHEMA,
                "finding_id": finding_id(cid, run.name, SOURCE, m["index"]),
                "case_id": cid, "patient_id": patient_id(cid), "source": SOURCE, "index": m["index"],
                "statement": m["label"],                        # exact text between <s> and </s>; None if the model gave none
                "raw_response": res["raw_text"], "prompt": res["prompt"],
                "mask": {"file": mf.relative_to(ROOT).as_posix(), "sha256": sha256_file(mf),
                         "foreground_voxels": m["foreground_voxels"], "shape": res["mask_shape"],
                         "affine_matches_ct": res["mask_affine_matches_input"]},
                "ct": {"file": ct.relative_to(ROOT).as_posix(), "sha256": a["ct_sha256"]},
                "generation": {"seed": 42, "do_sample": True, "temperature": 0.15, "top_p": 0.85,
                               "repetition_penalty": 1.15, "max_new_tokens": 1024,
                               "token_limit_reached": res["token_limit_reached"]},
                "model": model,
            })
    body = {"schema": INVENTORY_SCHEMA, "source_run": run.name, "model": model, "findings": findings,
            "fixed_condition_answers": answers, "problems": problems}
    digest = hashlib.sha256(json.dumps(body, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()
    tag = f"{args.label}_" if args.label else ""
    out = ROOT / "runs" / f"A2_audit_inventory_{tag}{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out.mkdir(parents=True)
    body.update(inventory_sha256=digest, created_utc=datetime.now(timezone.utc).isoformat(),
                note="Contains model output only. No labels, report text, checklist answers or audit verdicts.")
    (out / "inventory.json").write_text(json.dumps(body, indent=2, ensure_ascii=False), encoding="utf-8")
    per = {}
    for f in findings:
        per[f["case_id"]] = per.get(f["case_id"], 0) + 1
    print(f"{len(findings)} findings in {len(per)} cases {per}; {len(problems)} problems; sha256 {digest[:12]}")
    for p in problems:
        print("  PROBLEM", p)
    print(f"-> {out / 'inventory.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
