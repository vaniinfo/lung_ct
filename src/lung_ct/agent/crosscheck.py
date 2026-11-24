"""Pipeline step 4: cross-check AI readers against the evidence table (and each other).

Rule of thumb used here: for WHERE (laterality, zone, peripheral/central) the measurements
decide and a disagreeing reader is flagged; for WHAT (presence, type of finding) the reader's
words are kept but marked as unconfirmed when no evidence supports them. Nothing is silently
resolved: every disagreement becomes a row for the clinician.
"""
from __future__ import annotations

import re

FIELDS = ("laterality", "craniocaudal", "axial")
FIELD_EVIDENCE = {"laterality": ["E11", "E12", "E13"], "craniocaudal": ["E14"], "axial": ["E15"]}
_ZONE_WORDS = {"upper": "upper_predominant", "middle": "middle_predominant", "lower": "lower_predominant"}


def _row(ev: dict, rid: str) -> dict | None:
    return next((r for r in ev["rows"] if r["id"] == rid), None)


def description_field_conflicts(parsed: dict) -> list[str]:
    """Reader's own text vs its own structured fields (flagged, never corrected)."""
    out = []
    desc = (parsed.get("description") or "").lower()
    d = parsed.get("distribution", {})
    zones = {z for w, z in _ZONE_WORDS.items() if re.search(rf"\b{w}\b[^.]*\b(lobe|lobes|zone|zones|lung fields?)\b", desc)}
    cc = d.get("craniocaudal")
    if zones and cc in ("no_predominance", "not_applicable") and "predominant" in desc:
        out.append(f"text says predominantly {', '.join(sorted(z.split('_')[0] for z in zones))} "
                   f"but structured field says {cc}")
    elif len(zones) == 1 and cc in _ZONE_WORDS.values() and cc not in zones:
        out.append(f"text mentions {next(iter(zones)).split('_')[0]} but field says {cc}")
    lat = d.get("laterality")
    if lat == "bilateral" and re.search(r"\b(only|solely) (in )?the (right|left)\b", desc):
        out.append("text says one side only but field says bilateral")
    if parsed.get("opacity_presence") == "absent" and re.search(r"\bopacit", desc) and "no " not in desc:
        out.append("text describes opacities but field says absent")
    return out


def crosscheck(ev: dict, readers: dict[str, dict | None], cfg: dict) -> dict:
    ans = ev["derived_answers"]["distribution"]
    items = []
    r_share_row = _row(ev, "E13")
    r_share = (r_share_row or {}).get("value")
    strong = cfg["crosscheck"]["strong_side_share"] * 100
    dominant = None
    if r_share is not None:
        if r_share >= strong:
            dominant = "right"
        elif r_share <= 100 - strong:
            dominant = "left"

    for name, parsed in readers.items():
        if parsed is None:
            items.append({"reader": name, "field": "all", "status": "no_valid_output",
                          "detail": "reader returned no schema-valid answer", "evidence": []})
            continue
        if parsed.get("opacity_presence") == "absent":
            tot = (_row(ev, "E10") or {}).get("value")
            items.append({"reader": name, "field": "opacity_presence", "status": "conflict",
                          "reader_value": "absent", "evidence_value": f"{tot} mL dense lung tissue",
                          "detail": "reader saw no opacities; measurements find dense lung tissue "
                                    "(proxy, low reliability for presence)", "evidence": ["E10"]})
        for f in FIELDS:
            if parsed.get("opacity_presence") == "absent":
                break  # location fields are not_applicable; the presence conflict above covers it
            rv, evv = parsed.get("distribution", {}).get(f), ans.get(f)
            if rv in (None, "indeterminate", "not_applicable"):
                status = "reader_abstained"
            else:
                status = "agree" if rv == evv else "conflict"
            item = {"reader": name, "field": f, "status": status, "reader_value": rv,
                    "evidence_value": evv, "evidence": FIELD_EVIDENCE[f]}
            if f == "laterality" and dominant and rv == "bilateral":
                item["status"] = "conflict" if status == "agree" else status
                item["detail"] = (f"evidence: {r_share if dominant == 'right' else 100 - r_share}% of dense "
                                  f"tissue is in the {dominant} lung; reader did not report this predominance")
            items.append(item)
        for msg in description_field_conflicts(parsed):
            items.append({"reader": name, "field": "self_consistency", "status": "conflict",
                          "detail": msg, "evidence": []})

    items.extend(what_checks(ev, readers))

    names = [n for n, p in readers.items() if p is not None]
    for f in ("opacity_presence",) + FIELDS:
        vals = {}
        for n in names:
            p = readers[n]
            v = p.get(f) if f == "opacity_presence" else p.get("distribution", {}).get(f)
            if f == "opacity_presence" or v not in (None, "indeterminate", "not_applicable"):
                vals[n] = v
        if len(set(vals.values())) > 1:
            items.append({"reader": "readers", "field": f, "status": "readers_disagree",
                          "detail": "; ".join(f"{k}: {v}" for k, v in vals.items()), "evidence": []})

    decisions = {f: {"value": ans.get(f), "basis": "measurements (step 1)",
                     "evidence": FIELD_EVIDENCE[f]} for f in FIELDS}
    if dominant and ans.get("laterality") == "bilateral":
        decisions["laterality"]["note"] = f"bilateral, strongly {dominant}-predominant"
    pres = [readers[n].get("opacity_presence") for n in names]
    decisions["opacity_presence"] = {
        "value": "present" if "present" in pres else ("absent" if pres else "unknown"),
        "basis": "readers (step 3); measurement proxy cannot confirm presence without normal-scan calibration",
        "evidence": ["E10"]}
    n_conf = sum(i["status"] in ("conflict", "readers_disagree", "no_valid_output") for i in items)
    return {"items": items, "decisions": decisions, "n_conflicts": n_conf,
            "rule": "WHERE: measurements decide, disagreeing readers flagged. WHAT: readers kept, "
                    "unconfirmed where no evidence supports them. Nothing silently resolved."}


# reader words -> classifier finding key (lung findings only)
_FINDING_WORDS = {
    "consolidation": r"consolidat",
    "ground_glass": r"ground[- ]glass|\bggo",
    "lung_opacity": r"opacit",
    "atelectasis": r"atelecta",
    "lung_nodule": r"nodul",
    "emphysema": r"emphysem",
    "bronchiectasis": r"bronchiecta",
    "septal_thickening": r"septal thicken|crazy[- ]paving",
    "pleural_effusion": r"pleural effusion",
}


def what_checks(ev: dict, readers: dict[str, dict | None]) -> list[dict]:
    """Reader claims vs classifier scores (WHAT). Uncalibrated thresholds: flags for review only."""
    cl = ev.get("classifier")
    if not cl:
        return []
    rows = {r["id"]: r for r in ev["rows"] if r["group"] == "classifier"}
    by_key = {}
    for rid, r in rows.items():
        label = r["name"].replace(" (classifier probability)", "")
        by_key[label] = (rid, r["value"])
    key_to_label = {
        "consolidation": "Consolidation", "ground_glass": "Ground-glass opacity", "lung_opacity": "Lung opacity",
        "atelectasis": "Atelectasis", "lung_nodule": "Lung nodule", "emphysema": "Emphysema",
        "bronchiectasis": "Bronchiectasis", "septal_thickening": "Interlobular septal thickening",
        "pleural_effusion": "Pleural effusion"}
    hi, lo = cl["high_threshold"], cl["low_threshold"]
    out, mentioned_any = [], set()
    for name, p in readers.items():
        if p is None:
            continue
        desc = (p.get("description") or "").lower()
        for key, pat in _FINDING_WORDS.items():
            label = key_to_label[key]
            if label not in by_key or not re.search(pat, desc):
                continue
            negated = re.search(rf"\bno\b[^.]*({pat})", desc)
            if negated:
                continue
            mentioned_any.add(key)
            rid, prob = by_key[label]
            if prob is not None and prob < lo:
                out.append({"reader": name, "field": f"finding: {label}", "status": "conflict",
                            "reader_value": "mentioned", "evidence_value": f"p={prob:.2f}",
                            "detail": f"reader describes {label.lower()}, classifier score is low "
                                      "(uncalibrated)", "evidence": [rid]})
            elif prob is not None and prob >= hi:
                out.append({"reader": name, "field": f"finding: {label}", "status": "agree",
                            "reader_value": "mentioned", "evidence_value": f"p={prob:.2f}", "evidence": [rid]})
    for key, label in key_to_label.items():
        if key in mentioned_any or label not in by_key:
            continue
        rid, prob = by_key[label]
        if prob is not None and prob >= hi:
            out.append({"reader": "classifier", "field": f"finding: {label}", "status": "readers_disagree",
                        "evidence_value": f"p={prob:.2f}",
                        "detail": f"classifier suggests {label.lower()}; no reader mentioned it (uncalibrated)",
                        "evidence": [rid]})
    return out
