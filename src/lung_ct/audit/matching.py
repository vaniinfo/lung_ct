"""Comparing two runs of the model on the same scan (repeat, other seed, other reconstruction).

Two separate comparisons, never mixed:
  compare_answers   fixed-condition Yes/No answers, condition by condition (no geometry involved)
  match_findings    generated findings, matched by the geometry of their outlines

match_findings works on pair metrics (Dice, centroid distance in mm) so it can be tested without images.
Outcomes per finding:  matched | split | merge | ambiguous | unmatched (appeared / disappeared)
  link        two outlines overlap (Dice >= dice_min), or neither is empty and their centroids are close
              (<= centroid_max_mm)
  matched     exactly one link on both sides
  split       one finding of run A links to several findings of run B (each of which links only to it)
  merge       several findings of run A link to one finding of run B
  ambiguous   any other many-to-many pattern, or a link that rests on centroid distance alone while the
              finding types differ
  unmatched   no link: 'disappeared' (only in A) or 'appeared' (only in B). Empty outlines never link.
Thresholds are PROVISIONAL until the protocol is frozen.
"""
from __future__ import annotations

MATCH_VERSION = "audit_matching_v0_draft"
MATCH_THRESHOLDS = {"dice_min": 0.1, "centroid_max_mm": 30.0}


def compare_answers(a: dict, b: dict) -> dict:
    """a, b: {condition: 'yes' | 'no' | 'unparsed'} from two runs. Returns per-condition result and counts."""
    rows = {}
    for c in sorted(set(a) | set(b)):
        x, y = a.get(c), b.get(c)
        rows[c] = {"a": x, "b": y,
                   "result": "not_comparable" if x not in ("yes", "no") or y not in ("yes", "no") else ("same" if x == y else "flip")}
    counts = {k: sum(r["result"] == k for r in rows.values()) for k in ("same", "flip", "not_comparable")}
    return {"conditions": rows, "counts": counts}


def match_findings(a: list[dict], b: list[dict], pairs: dict, thresholds: dict | None = None) -> dict:
    """a, b: findings of the two runs, each {'id', 'voxels', 'types': [...]}.
    pairs: {(id_a, id_b): {'dice': float, 'centroid_mm': float | None}} for every pair (missing pair = no relation)."""
    th = {**MATCH_THRESHOLDS, **(thresholds or {})}
    vox = {f["id"]: f["voxels"] for f in a + b}
    types = {f["id"]: set(f.get("types") or []) for f in a + b}
    links, weak = [], set()
    for (ia, ib), m in pairs.items():
        if vox.get(ia, 0) == 0 or vox.get(ib, 0) == 0:
            continue
        overlap = m.get("dice", 0.0) >= th["dice_min"]
        near = m.get("centroid_mm") is not None and m["centroid_mm"] <= th["centroid_max_mm"]
        if overlap or near:
            links.append((ia, ib))
            if not overlap and types[ia] != types[ib]:
                weak.add((ia, ib))
    la = {f["id"]: [ib for ia, ib in links if ia == f["id"]] for f in a}
    lb = {f["id"]: [ia for ia, ib in links if ib == f["id"]] for f in b}
    out_a, out_b = {}, {}
    for f in a:
        i, ls = f["id"], la[f["id"]]
        if not ls:
            out_a[i] = {"outcome": "unmatched", "kind": "disappeared", "links": []}
        elif any((i, j) in weak for j in ls):
            out_a[i] = {"outcome": "ambiguous", "links": ls}
        elif len(ls) == 1:
            back = lb[ls[0]]
            out_a[i] = {"outcome": "matched" if len(back) == 1 else "merge", "links": ls}
        else:
            out_a[i] = {"outcome": "split" if all(len(lb[j]) == 1 for j in ls) else "ambiguous", "links": ls}
    for f in b:
        j, ls = f["id"], lb[f["id"]]
        if not ls:
            out_b[j] = {"outcome": "unmatched", "kind": "appeared", "links": []}
        elif any((i, j) in weak for i in ls):
            out_b[j] = {"outcome": "ambiguous", "links": ls}
        elif len(ls) == 1:
            fwd = la[ls[0]]
            out_b[j] = {"outcome": "matched" if len(fwd) == 1 else "split", "links": ls}
        else:
            out_b[j] = {"outcome": "merge" if all(len(la[i]) == 1 for i in ls) else "ambiguous", "links": ls}
    counts = {}
    for r in list(out_a.values()) + list(out_b.values()):
        k = r["outcome"] if r["outcome"] != "unmatched" else r["kind"]
        counts[k] = counts.get(k, 0) + 1
    return {"a": out_a, "b": out_b, "counts": counts, "thresholds": th, "match_version": MATCH_VERSION}
