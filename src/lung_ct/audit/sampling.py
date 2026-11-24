"""Sampling manifest for clinician review: all flagged findings plus a simple random sample of unflagged ones.

Every generated finding gets a row with its stratum, inclusion probability and whether it was selected, so
design-weighted estimates (weight = 1 / inclusion probability) can be computed later. The manifest is written
before grading and is never loaded by the blinded viewer; the viewer only receives the list of selected ids.

Scope: the weighted estimate concerns errors among GENERATED findings. It says nothing about abnormalities the
model never mentioned.
"""
from __future__ import annotations

import random

SAMPLING_VERSION = "audit_sampling_v1"


def build_manifest(findings: list[dict], n_unflagged: int, seed: int) -> dict:
    """findings: [{'finding_id', 'patient_id', 'flagged': bool}] - the COMPLETE inventory, one row per finding.
    Deterministic for a given inventory, n_unflagged and seed (ids are sorted before drawing)."""
    ids = [f["finding_id"] for f in findings]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate finding ids in the inventory")
    if n_unflagged < 0:
        raise ValueError("n_unflagged must be >= 0")
    unflagged = sorted(f["finding_id"] for f in findings if not f["flagged"])
    n = min(n_unflagged, len(unflagged))
    chosen = set(random.Random(seed).sample(unflagged, n))
    p_un = (n / len(unflagged)) if unflagged else None
    rows = []
    for f in sorted(findings, key=lambda x: x["finding_id"]):
        flagged = bool(f["flagged"])
        sel = flagged or f["finding_id"] in chosen
        rows.append({"finding_id": f["finding_id"], "patient_id": f["patient_id"],
                     "stratum": "flagged" if flagged else "unflagged",
                     "inclusion_probability": 1.0 if flagged else p_un,
                     "selected": sel,
                     "weight": (1.0 if flagged else (1.0 / p_un if p_un else None)) if sel else None})
    return {"sampling_version": SAMPLING_VERSION, "seed": seed, "design": "all flagged + simple random sample of unflagged",
            "n_findings": len(rows), "n_flagged": sum(r["stratum"] == "flagged" for r in rows),
            "n_unflagged": len(unflagged), "n_unflagged_requested": n_unflagged, "n_unflagged_selected": n,
            "n_selected": sum(r["selected"] for r in rows),
            "n_patients_selected": len({r["patient_id"] for r in rows if r["selected"]}), "rows": rows}
