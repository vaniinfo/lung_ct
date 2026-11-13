"""Phase 5 validation: parameter sensitivity of airway metrics on real cases (core env).

One-at-a-time variation of pruning, contraction, loop contraction and centreline smoothing
around the defaults (grid in lung_ct/airway/sensitivity.py). Writes to
runs/P5_sensitivity_<n>cases_<ts>/:
  sensitivity_long.csv     every case x setting x metric value
  stability_by_case.csv    per case x metric: rating, relative range, most influential param
  stability_summary.md     per metric: how many cases stable / moderate / unstable
  run_manifest.json
Ratings are processing-stability only; they say nothing about anatomical correctness.

Usage:
    python scripts/airway_sensitivity.py --seg-run runs/P2_seg_10cases_<ts> [--cases ...]
"""
import argparse
import csv
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import nibabel as nib  # noqa: E402
import numpy as np  # noqa: E402

from lung_ct.airway.sensitivity import (  # noqa: E402
    GRID, MODERATE_REL, STABLE_REL, metrics_from, rate, settings)
from lung_ct.airway.tree import DEFAULTS, METHOD, analyze_airway_tree  # noqa: E402

MASK = "lung_vessels/lung_airways.nii.gz"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seg-run", required=True, type=Path)
    ap.add_argument("--cases", nargs="+")
    args = ap.parse_args()
    seg = json.loads((ROOT / args.seg_run / "run_manifest.json").read_text())
    cases = [c for c in seg["cases"] if c["status"] == "ok"
             and (not args.cases or c["case_id"] in args.cases)]
    run_id = f"P5_sensitivity_{len(cases)}cases_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = ROOT / "runs" / run_id
    run_dir.mkdir(parents=True)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
                            text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain"], cwd=ROOT,
                                capture_output=True, text=True).stdout.strip())

    long_rows, by_case = [], []
    for c in cases:
        cid = c["case_id"]
        img = nib.load(str(ROOT / c["output_dir"] / MASK))
        coarsest = float(max(np.sqrt((img.affine[:3, :3] ** 2).sum(axis=0))))
        results = []
        t0 = time.time()
        for param, value, overrides in settings(coarsest):
            m = metrics_from(analyze_airway_tree(img, overrides))
            results.append((param, value, m))
            long_rows.extend({"case_id": cid, "param": param, "value": value,
                              "metric": k, "result": v} for k, v in m.items())
        default = results[0][2]
        for metric in default:
            vals = [r[2][metric] for r in results]
            rating, rel = rate(metric, vals, default[metric])
            # most influential parameter: largest |change| from default
            best, best_d = "", -1.0
            for param, value, m in results[1:]:
                if m[metric] is None or default[metric] is None:
                    d = float("inf")
                else:
                    d = abs(m[metric] - default[metric])
                if d > best_d:
                    best, best_d = f"{param}={value}", d
            by_case.append({"case_id": cid, "metric": metric, "default": default[metric],
                            "min": min((v for v in vals if v is not None), default=None),
                            "max": max((v for v in vals if v is not None), default=None),
                            "relative_range": None if rel is None else round(rel, 4),
                            "rating": rating, "most_influential": best if best_d > 0 else ""})
        print(f"[{cid}] {len(results)} settings, {time.time() - t0:.0f}s; unstable: "
              f"{[r['metric'] for r in by_case if r['case_id'] == cid and r['rating'] == 'unstable']}")

    def write(name, rows):
        with open(run_dir / name, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)

    write("sensitivity_long.csv", long_rows)
    write("stability_by_case.csv", by_case)

    metrics = list(dict.fromkeys(r["metric"] for r in by_case))
    lines = [f"# Airway metric sensitivity - {run_id}", "",
             f"Method: {METHOD}. Code {commit}{'+dirty' if dirty else ''}. "
             f"One-at-a-time grid around defaults {json.dumps(GRID)}.",
             f"Rating: generation 0-3 counts must be identical across all settings; other "
             f"metrics stable if relative range <= {STABLE_REL:.0%}, moderate <= "
             f"{MODERATE_REL:.0%}, else unstable. Processing stability only - not anatomical "
             "validity.", "",
             "| Metric | stable | moderate | unstable | median rel. range | max rel. range | "
             "unstable cases | most influential (mode) |",
             "|---|---|---|---|---|---|---|---|"]
    for mt in metrics:
        rs = [r for r in by_case if r["metric"] == mt]
        rels = [r["relative_range"] for r in rs if r["relative_range"] is not None]
        infl = [r["most_influential"].split("=")[0] for r in rs if r["most_influential"]]
        mode = max(set(infl), key=infl.count) if infl else ""
        unstable = [r["case_id"].replace("coronacases_", "") for r in rs if r["rating"] == "unstable"]
        lines.append(f"| {mt} | {sum(r['rating'] == 'stable' for r in rs)} | "
                     f"{sum(r['rating'] == 'moderate' for r in rs)} | {len(unstable)} | "
                     f"{f'{np.median(rels):.1%}' if rels else '-'} | "
                     f"{f'{max(rels):.1%}' if rels else '-'} | {', '.join(unstable) or '-'} | {mode} |")
    (run_dir / "stability_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (run_dir / "run_manifest.json").write_text(json.dumps({
        "run_id": run_id, "git_commit": commit, "git_dirty": dirty,
        "finished_utc": datetime.now(timezone.utc).isoformat(), "seg_run": str(args.seg_run),
        "method": METHOD, "defaults": DEFAULTS, "grid": GRID,
        "thresholds": {"stable_rel": STABLE_REL, "moderate_rel": MODERATE_REL},
        "cases": [c["case_id"] for c in cases]}, indent=2))
    print(f"done -> {run_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
