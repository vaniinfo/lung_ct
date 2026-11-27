"""Condition checklist (pipeline steps 4-5 for the 2026-11-15 format): one line per condition,
status present / absent / cannot tell / cannot assess, with the basis, location and evidence ids.

Pure Python (json/re only), importable from the GPU environment. Rules are in
configs/checklist_v1.json ("decision_rule"); this module applies them and never invents findings.
"""
from __future__ import annotations

import json
import re

STATUSES = ("present", "absent", "cannot tell", "cannot assess")
READER_STATUSES = {"present", "absent", "cannot_tell"}

# condition key -> evidence rows holding its measured value (summed) and its location row
MEASURED = {
    "ground_glass": (["E81"], "E82"),
    "consolidation": (["E85"], "E87"),
    "emphysema": (["E30"], None),          # v3 uses E31 (airways excluded) when present
    "atelectasis": (["E88"], None),         # v3: dense lung next to pleural fluid
    "pleural_effusion": (["E71", "E72"], None),
    "pericardial_effusion": (["E73"], None),
    "cardiomegaly": (["E76"], None),
    "arterial_calcification": (["E78"], None),
    "lung_nodule": (["E97"], None),         # v3_10: count of candidates outlined by the lung_nodules segmentation task
}
CONTEXT_ROWS = {"lung_nodule": ["E98"], "atelectasis": ["E90"], "cardiomegaly": ["E74", "E75", "E77"], "lung_density": ["E90"],
                "pericardial_effusion": ["E79"]}


def load_checklist_config(path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _measured_status(c: dict, rows: dict) -> tuple[str | None, float | None, list[str]]:
    if "present_at_least" not in c:
        return None, None, []
    ids, _ = MEASURED.get(c["key"], ([], None))
    if c["key"] == "emphysema" and "E31" in rows:
        ids = ["E31"]
    vals = [rows[i]["value"] for i in ids if i in rows]
    if not ids or not vals or any(v is None for v in vals):
        return None, None, [i for i in ids if i in rows]
    v = float(sum(vals))
    if v >= c["present_at_least"]:
        return "present", v, ids
    if c.get("absent_below") is not None and v < c["absent_below"]:
        return "absent", v, ids
    return "grey", v, ids


def _classifier_vote(key: str | None, rows: dict, hi: float, lo: float) -> tuple[str | None, float | None, str | None]:
    if not key:
        return None, None, None
    for r in rows.values():
        if r.get("group") == "classifier" and r.get("ckey") == key and r["value"] is not None:
            p = float(r["value"])
            return ("present" if p >= hi else "absent" if p <= lo else "unsure"), p, r["id"]
    return None, None, None


def _side_text(rows: dict, c: dict) -> str | None:
    lr = c.get("location_row")          # v3_10: a row that already holds the location in words (nodule candidates: lobes)
    if lr and isinstance(rows.get(lr, {}).get("where"), str):
        return rows[lr]["where"]
    if c["key"] == "pleural_effusion" and "E71" in rows and "E72" in rows:
        l, r = rows["E71"]["value"] or 0, rows["E72"]["value"] or 0
        if l + r == 0:
            return None
        return "both sides" if min(l, r) >= c["absent_below"] else ("right" if r > l else "left")
    _, loc_id = MEASURED.get(c["key"], ([], None))
    loc = rows.get(loc_id, {}).get("value") if loc_id else None
    if isinstance(loc, dict):
        side = {"both": "both lungs", "right": "right lung", "left": "left lung"}[loc["side"]]
        zone = "" if loc["zone"] == "no single zone" else f", mainly {loc['zone']} zone"
        return f"{side}{zone}; most in {loc['leading_lobe']} ({loc['leading_lobe_share_pct']}%)"
    return None


def location_report(key: str, status: str, rows: dict, lr: dict, pleural_present: bool) -> dict | None:
    """Location of the HU-selected region of one condition (v3_4, lr = config 'location_reporting').

    Returns None when the condition has no region row, the row is missing / not measured, or the region is empty.
    Naming rule and uncertainty flags are the provisional ones in the config; the full distribution is copied through
    untouched so nothing is lost by the rule. Never changes a status."""
    reg = (lr.get("regions") or {}).get(key)
    if not reg or status not in lr.get("show_for_status", ["present", "cannot tell"]):
        return None
    val = (rows.get(reg["row"]) or {}).get("value")
    if not isinstance(val, dict) or not val.get("region_voxels"):
        return None
    cut = float(lr["name_lobe_if_pct_of_region_at_least"])
    shares = sorted(((d["pct_of_region"] or 0.0, n) for n, d in val["lobes"].items()), key=lambda t: (-t[0], t[1]))
    named = [n for sh, n in shares if sh >= cut]
    u = lr.get("uncertain_lobe_assignment") or {}
    why = []
    nf = val.get("near_fissure_pct_of_region")
    if nf is not None and u.get("near_fissure_pct_of_region_at_least") is not None and nf >= u["near_fissure_pct_of_region_at_least"]:
        why.append(f"{nf:g}% of the region is within {val['near_fissure_mm']:g} mm of a fissure")
    out = val.get("outside_lobes_pct_of_region")
    if out is not None and u.get("outside_lobes_pct_of_region_at_least") is not None and out >= u["outside_lobes_pct_of_region_at_least"]:
        why.append(f"{out:g}% of the region is in no lobe outline")
    if u.get("poor_inflation") and poor_inflation(rows):
        why.append("lungs poorly inflated")
    if u.get("pleural_effusion_present") and pleural_present:
        why.append(u.get("pleural_effusion_reason_text") or "pleural effusion present (compressed lung)")
    sides = {n[0].upper() for n in named}
    confirmed = status == "present"
    lobes_txt = ", ".join(named) if named else "no single lobe (see the measurement panel)"
    text = f"{reg['name']} in {lobes_txt}" + ("; lobe assignment uncertain" if why else "")
    return {"rule": lr["version"], "row": reg["row"], "region": reg["name"],
            "region_is": "HU-selected region" + ("" if confirmed else ", not a confirmed finding"),
            "lobes_named": named, "naming_cutoff_pct_of_region": cut,
            "side": "both lungs" if sides == {"R", "L"} else "right lung" if sides == {"R"} else "left lung" if sides == {"L"} else None,
            "pct_of_region": {n: sh for sh, n in shares}, "pct_of_lobe": {n: d["pct_of_lobe"] for n, d in val["lobes"].items()},
            "lobe_assignment_uncertain": bool(why), "uncertain_because": why, "text": text}


def add_locations(items: list[dict], rows: dict, ck: dict) -> None:
    """v3_4: attach location_report / location_text. The older 'location' string is left exactly as it was."""
    lr = ck.get("location_reporting")
    if not lr:
        return
    by = {i["key"]: i for i in items}
    pleural = by.get("pleural_effusion", {}).get("status") == "present"
    for it in items:
        rep_ = location_report(it["key"], it["status"], rows, lr, pleural)
        if rep_:
            it["location_report"], it["location_text"] = rep_, rep_["text"]
            if rep_["row"] not in it["evidence"]:
                it["evidence"].append(rep_["row"])
    ld = by.get("lung_density")
    if ld is not None and ld["status"] in lr.get("show_for_status", []):
        subs = [by[k]["location_report"] for k in ("ground_glass", "consolidation") if by.get(k, {}).get("location_report")]
        if subs:
            ld["location_report"] = {"rule": lr["version"], "parts": subs,
                                     "lobe_assignment_uncertain": any(x["lobe_assignment_uncertain"] for x in subs)}
            ld["location_text"] = "; ".join(x["text"] for x in subs)


def poor_inflation(rows: dict) -> bool:
    return "POOR" in str(rows.get("E96", {}).get("note") or "")


def build_checklist(ev: dict, readers: dict, ck: dict, hi: float = 0.7, lo: float = 0.3) -> dict:
    """readers: name -> parsed v2 checklist answer ({"conditions": {key: {"status":..}}}) or anything else."""
    rows = {r["id"]: r for r in ev["rows"]}
    contrast = (rows.get("E70", {}).get("note") or "scan type: unknown").replace("scan type: ", "")
    rv = {n: (p or {}).get("conditions") for n, p in readers.items()
          if isinstance(p, dict) and isinstance(p.get("conditions"), dict)}
    items = []
    for c in ck["conditions"]:
        key = c["key"]
        it = {"key": key, "label": c["label"], "ct_rate_label": c.get("ct_rate_label"), "evidence": [],
              "flags": [], "location": None}
        cv, p, cid = _classifier_vote(c.get("classifier_key"), rows, hi, lo)
        if cid:
            it["evidence"].append(cid)
            it["classifier_p"] = round(p, 3)
        reader_votes = {}
        for n, conds in rv.items():
            a = conds.get(key)
            s = (a.get("status") if isinstance(a, dict) else a) if a is not None else None
            if s in READER_STATUSES:
                reader_votes[n] = s
        it["readers"] = reader_votes
        n_present = sum(s == "present" for s in reader_votes.values())
        vote = ck.get("readers_vote", True)  # v2: readers shown, not counted
        n_vote = n_present if vote else 0

        if c.get("needs_contrast") and contrast != "contrast":
            it.update(status="cannot assess", basis=f"needs a contrast scan; this scan is {contrast}")
            items.append(it)
            continue
        if key == "lung_density":
            sub = {x["key"]: x for x in items if x["key"] in ("ground_glass", "consolidation")}
            st = [sub[k]["status"] for k in sub]
            it["evidence"] += [e for k in sub for e in sub[k]["evidence"] if e not in it["evidence"]]
            if "present" in st:
                kinds = [sub[k]["label"].lower() for k in sub if sub[k]["status"] == "present"]
                it.update(status="present", basis="measured: " + " and ".join(kinds),
                          location="; ".join(sub[k]["location"] for k in sub
                                             if sub[k]["status"] == "present" and sub[k]["location"]) or None)
            elif st and all(s == "absent" for s in st):
                it.update(status="absent", basis="measured: neither ground-glass nor consolidation range above threshold")
            else:
                it.update(status="cannot tell", basis="measurements in the grey zone or missing")
            if cv == "present" and it["status"] == "absent":
                it["flags"].append(f"classifier suggests present (p={p:.2f}); measurements do not")
            it["evidence"] += [e for e in CONTEXT_ROWS.get(key, []) if e in rows]
            items.append(it)
            continue

        ms, mv, mids = _measured_status(c, rows)
        it["evidence"] = mids + it["evidence"]
        if mv is not None:
            it["measured_value"] = mv
        no_absence = bool(c.get("predicted_mask_cannot_establish_absence")) and ms is not None   # v3_7
        if no_absence:  # answer of the unchanged historical rule, kept for comparison with earlier versions
            it["historical_rule_answer"] = {"present": "present", "absent": "none", "grey": "undecided"}[ms]
        if no_absence and ms == "absent":
            it.update(status="cannot tell", basis=f"predicted mask below the provisional reporting limit ({mv:g} {c.get('unit', '')}; limits "
                                                  f"{c['present_at_least']:g} / {c['absent_below']:g} {c.get('unit', '')}, unchanged, not validated "
                                                  f"diagnostic boundaries); a predicted mask cannot establish absence")
            if (c.get("mask_wording") or {}).get("basis_empty"):          # v3_10: wording for a count, not a volume
                it["basis"] = c["mask_wording"]["basis_empty"].format(value=mv)
            if cv == "present":
                it["flags"].append(f"classifier says present (p={p:.2f})")
        elif ms in ("present", "absent"):
            it.update(status=ms, basis=f"measured ({mv:g} {c.get('unit', '')}; present >= {c['present_at_least']:g}"
                                       + (f", absent < {c['absent_below']:g})" if c.get("absent_below") is not None
                                          else "; a low value is not proof of absence)"))
            if ms == "present" and (c.get("mask_wording") or {}).get("headline_present"):     # v3_8: no 'present >= / absent <' wording
                it["basis"] = (f"predicted mask only ({mv:g} {c.get('unit', '')}, at or above the provisional {c['present_at_least']:g} {c.get('unit', '')} "
                               f"reporting limit; limits unchanged, not validated diagnostic boundaries)")
                if c["mask_wording"].get("basis_present"):                # v3_10
                    it["basis"] = c["mask_wording"]["basis_present"].format(value=mv)
            if ms == "present":
                it["location"] = _side_text(rows, c)
            opp = "absent" if ms == "present" else "present"
            if cv == opp:
                it["flags"].append(f"classifier says {opp} (p={p:.2f})")
            dis = [n for n, s in reader_votes.items() if s == opp]
            if dis:
                it["flags"].append(f"reader(s) {', '.join(dis)} say {opp}")
        else:
            grey = f"measurement in grey zone ({mv:g} {c.get('unit', '')}); " if ms == "grey" else ""
            tr = c.get("reader_decides_present")     # v3_11: one named AI reader decides 'present' for this condition (never 'absent')
            ra = (rv.get(tr) or {}).get(key) if tr else None
            sc = ra.get("score_yes") if isinstance(ra, dict) else None
            sc_txt = f", score {sc:.2f} (0 to 1, uncalibrated)" if isinstance(sc, (int, float)) else ""
            if tr and reader_votes.get(tr) == "present":
                it.update(status="present", basis=grey + f"AI reader {tr} answers present{sc_txt}; by project-owner decision this reader's "
                                                         f"'present' is taken as the answer for this condition (not validated; nothing measured)"
                                                         + (f"; classifier p={p:.2f}" if p is not None else ""))
                it["reader_decided"] = {"reader": tr, "answer": "present", "score_yes": sc,
                                        "side": ra.get("side") if isinstance(ra, dict) else None}
            elif cv == "present" and n_vote >= 1:
                it.update(status="present", basis=grey + f"classifier (p={p:.2f}) and {n_present} AI reader(s) agree; not measured")
            elif (cv == "absent" and n_vote == 0 and c.get("allow_classifier_absent", True)
                  and ck.get("classifier_can_establish_absence", True)):   # v3_5: False = a low score never means absent
                it.update(status="absent", basis=grey + f"classifier low (p={p:.2f})" +
                          (" and no reader says present" if vote and reader_votes else "; classifier only (uncalibrated)"))
                if not vote and n_present:
                    it["flags"].append(f"{n_present} AI reader(s) say present (readers do not vote in {ck['config_version']})")
            else:
                why = []
                if cv is None:
                    why.append("no classifier score")
                elif cv == "absent" and not ck.get("classifier_can_establish_absence", True):
                    why.append(f"classifier p={p:.2f} (low, but an unvalidated score cannot establish absence)")
                else:
                    why.append(f"classifier p={p:.2f}")
                if tr and tr in reader_votes:    # v3_11: the deciding reader did not say present
                    why.append(f"AI reader {tr} answers {reader_votes[tr].replace('_', ' ')}{sc_txt}; a reader's 'absent' cannot establish absence")
                elif tr:
                    why.append(f"no answer from AI reader {tr}")
                elif reader_votes:
                    why.append(f"{n_present} of {len(reader_votes)} readers say present" + ("" if vote else " (shown, not counted)"))
                else:
                    why.append("no reader answer")
                it.update(status="cannot tell", basis=grey + "; ".join(why))
        if ck.get("two_source_rule"):        # v3_12
            _two_source(it, c, ms, mv, p, reader_votes, rv)
        gate = c.get("poor_inflation")  # v3 breath-in check
        if gate and poor_inflation(rows):
            if gate == "no_absent" and it["status"] == "absent":
                it.update(status="cannot tell", basis="lungs poorly inflated (E95, E96): " + it["basis"])
            elif gate == "no_present" and it["status"] == "present":
                # v3_3: a value far above what under-inflation alone gives keeps 'present' (optional, per condition)
                keep = c.get("poor_inflation_present_at_least")
                val = it.get("measured_value")
                if keep is not None and val is not None and val >= keep:
                    it["basis"] += (f"; lungs poorly inflated (E95, E96), but the value is at or above the "
                                    f"poor-inflation limit of {keep:g} {c.get('unit', '')}")
                    it["flags"].append("lungs poorly inflated: amount may be overestimated")
                else:
                    it.update(status="cannot tell", basis="lungs poorly inflated (E95, E96), density may be normal lung: "
                                                          + it["basis"])
            it["evidence"] += [e for e in ("E95", "E96") if e in rows and e not in it["evidence"]]
        if c.get("contrast_helps") and contrast != "contrast":
            it["flags"].append("contrast would help")
        it["evidence"] += [e for e in CONTEXT_ROWS.get(key, []) if e in rows and e not in it["evidence"]]
        items.append(it)
    add_locations(items, rows, ck)
    add_statements(items, ck)
    for it in items:                   # v3_11: say where a reader-decided answer comes from
        rd_ = it.get("reader_decided")
        if it.get("two_source"):       # v3_12
            ts_ = it["two_source"]
            it["decision_basis"] = "two sources agree"
            it["statement"] = (f"{it['label']}: present" + (f", {it['location']}" if it.get("location") else "") + ". Two sources agree: AI reader "
                               f"{ts_['reader']} answers present, and " + " and ".join(ts_["agreeing"]) + ".")
        elif it.get("mask_only_positive"):
            mo = it["mask_only_positive"]
            it["display"] = "cannot tell"
            it["statement"] = mo["text"]
        elif rd_:
            it["decision_basis"] = f"AI reader {rd_['reader']}"
            it["statement"] = f"{it['label']}: present, according to AI reader {rd_['reader']} (not measured by the tool)."
    add_threshold_results(items, ck)
    add_dense_candidates(items, rows, ck)
    if ck.get("neutral_wording"):      # v3_5: text only (inspiration not asserted); statuses and numbers are already final here
        from lung_ct.agent.wording import neutral_item
        for it in items:
            neutral_item(it)
    counts = {s: sum(i["status"] == s for i in items) for s in STATUSES}
    return {"config_version": ck["config_version"], "scan_type": contrast, "items": items, "counts": counts,
            "rule": ck["decision_rule"]}


def _two_source(it: dict, c: dict, ms, mv, p, reader_votes: dict, rv: dict) -> None:
    """v3_12. (1) A positive answer from a predicted mask alone becomes 'cannot tell'; what the mask showed is kept as evidence.
    (2) For conditions with 'two_source_present': 'present' when the named AI reader answers present AND a second, independent source agrees
    (classifier score above the condition's cut-off, or a predicted-mask value at or above a limit). Only where the answer is otherwise
    'cannot tell': a decisive measurement is never overruled. A reader's 'absent' never gives 'absent'."""
    mw = c.get("mask_wording") or {}
    unit = c.get("unit", "")
    if it["status"] == "present" and ms == "present" and mw.get("headline_present"):
        it.update(status="cannot tell", basis=f"predicted mask only ({mv:g} {unit}); one source alone does not give 'present' in v3_12")
        it["mask_only_positive"] = {"value": mv, "unit": unit, "text": mw.get("text_mask_only", "{label}: cannot tell.").format(
            label=it["label"], value=mv, where=it.get("location") or "side not determined")}
    rule = c.get("two_source_present")
    if not rule or it["status"] != "cannot tell":
        return
    rd = rule["reader"]
    if reader_votes.get(rd) != "present":
        return
    agreeing = []
    cut = rule.get("classifier_above")
    if cut is not None and p is not None and p > cut:
        agreeing.append(f"the classifier COLIPRI scores {p:.2f}, above this condition's cut-off {cut:g}")
    lim = rule.get("measured_at_least")
    if lim is not None and mv is not None and mv >= lim:
        agreeing.append(f"the segmentation model outlined {mv:g} {unit} (at least {lim:g} {unit})")
    if not agreeing:
        it["basis"] += f"; AI reader {rd} answers present, but no second source agrees (v3_12 needs two)"
        return
    ra = (rv.get(rd) or {}).get(it["key"])
    sc = ra.get("score_yes") if isinstance(ra, dict) else None
    it.update(status="present", basis=f"two sources agree: AI reader {rd} answers present; " + "; ".join(agreeing)
              + f" (rule {rule['version']}, fitted on development scans, not validated)")
    it.pop("mask_only_positive", None)
    it["two_source"] = {"reader": rd, "agreeing": agreeing, "rule": rule["version"]}
    it["reader_decided"] = {"reader": rd, "answer": "present", "score_yes": sc, "side": ra.get("side") if isinstance(ra, dict) else None,
                            "with": agreeing}
    if ms == "present" or (lim is not None and mv is not None and mv >= lim):
        it["location"] = it.get("location") or None


# ---------- wording (v3_2) ----------
# "absent" is the scoring status; what the report SAYS depends on how it was reached:
#   measured below the detection limit -> "none measured" ("No measurable pleural effusion (under 5 mL)")
#   classifier only                    -> "not suggested" (uncalibrated classifier, nothing measured)
DISPLAY = {"present": "present", "cannot tell": "cannot tell", "cannot assess": "cannot assess"}


def assessment_category(it: dict) -> str:
    """The CURRENT assessment of an item, for scoring and export: present / suggested / absent / cannot tell / cannot assess.

    'suggested' = a positive answer that rests on a predicted mask only (v3_8 headline). It is never counted as a
    confirmed 'present'. The provisional threshold result is NOT read here: a historical 'none' is not an 'absent'.
    """
    return "suggested" if it["status"] == "present" and it.get("headline") else it["status"]


EXPORT_COLUMNS = ["case_id", "checklist_version", "condition_key", "condition", "current_assessment", "headline_shown", "stored_status", "mask_state",
                  "measured_value", "unit", "decision_basis", "provisional_threshold_result_NOT_A_CURRENT_ASSESSMENT", "provisional_limits",
                  "statement", "basis"]


def export_row(case_id: str, config_version: str, it: dict) -> dict:
    """One CSV row per condition. The current assessment and the provisional threshold result are separate, named columns."""
    t = it.get("provisional_threshold_result") or {}
    return {"case_id": case_id, "checklist_version": config_version, "condition_key": it["key"], "condition": it["label"],
            "current_assessment": assessment_category(it), "headline_shown": it.get("headline") or it.get("display", it["status"]),
            "stored_status": it["status"], "mask_state": it.get("mask_state", ""), "measured_value": it.get("measured_value", ""),
            "unit": t.get("unit") or "", "decision_basis": it.get("decision_basis", ""),
            "provisional_threshold_result_NOT_A_CURRENT_ASSESSMENT": t.get("answer", ""),
            "provisional_limits": (f"present >= {t['present_at_least']:g}, none < {t['absent_below']:g} {t.get('unit') or ''} (provisional, not validated)"
                                   if t else ""),
            "statement": it.get("statement", ""), "basis": it.get("basis", "")}


def add_threshold_results(items: list[dict], ck: dict) -> None:
    """v3_8: keep the answer of the unchanged provisional limits apart from the headline. Changes no status."""
    conds = {c["key"]: c for c in ck["conditions"]}
    for it in items:
        c = conds.get(it["key"], {})
        note = (c.get("mask_wording") or {}).get("threshold_result_note")
        if note and it.get("historical_rule_answer"):
            it["provisional_threshold_result"] = {"answer": it["historical_rule_answer"], "value": it.get("measured_value"), "unit": c.get("unit"),
                                                  "present_at_least": c.get("present_at_least"), "absent_below": c.get("absent_below"),
                                                  "note": note.format(present=c.get("present_at_least"), limit=c.get("absent_below"))}


def add_dense_candidates(items: list[dict], rows: dict, ck: dict) -> None:
    """v3_6 (dense_lung_inventory_v2): attach the retained dense-lung candidates to consolidation, atelectasis and
    increased lung density as evidence. Text and evidence ids only: no status is set or changed here, and no limit is
    applied to the inventory."""
    cfg = (ck.get("measurements") or {}).get("dense_lung_inventory")
    val = (rows.get("E94") or {}).get("value")
    if not cfg or not isinstance(val, dict):
        return
    t = val["tags"]
    outside = round(sum(t[k]["ml"] for k in t if k != "no_spatial_tag"), 1)
    lobes = {}
    for k in t:
        for n, v in t[k]["per_lobe_ml"].items():
            lobes[n] = round(lobes.get(n, 0.0) + v, 1)
    named = [n for n, v in sorted(lobes.items(), key=lambda x: (-x[1], x[0])) if val["total_ml"] and v >= 0.1 * val["total_ml"]]
    if val["total_ml"] <= 0:
        txt = "Dense-lung candidates: none selected by CT density (estimate; small or sub-solid changes are not covered)."
    else:
        parts = [f"{t['no_spatial_tag']['ml']:g} mL without a spatial tag"]
        if t["fluid_adjacent"]["ml"] or t["fluid_adjacent_and_posterior"]["ml"]:
            parts.append(f"{round(t['fluid_adjacent']['ml'] + t['fluid_adjacent_and_posterior']['ml'], 1):g} mL next to predicted pleural fluid")
        if t["posterior"]["ml"] or t["fluid_adjacent_and_posterior"]["ml"]:
            parts.append(f"{round(t['posterior']['ml'] + t['fluid_adjacent_and_posterior']['ml'], 1):g} mL in the posterior band")
        both = t["fluid_adjacent_and_posterior"]["ml"]
        txt = (f"Dense-lung candidates selected by CT density: {val['total_ml']:g} mL in total"
               + (f", in {', '.join(named)}" if named else "") + " (" + "; ".join(parts)
               + (f"; {both:g} mL carry both tags" if both else "") + "). What they represent (consolidation, atelectasis, dependent density or "
               "another process) is not determined by the tool.")
    for it in items:
        if it["key"] not in cfg.get("attach_to", ["consolidation", "atelectasis", "lung_density"]):
            continue
        it["dense_lung_candidates"] = {"version": val["version"], "total_ml": val["total_ml"], "outside_historical_baseline_ml": outside,
                                       "tags_ml": {k: t[k]["ml"] for k in t}, "per_lobe_ml": lobes, "lobes_named": named, "text": txt,
                                       "interpretation": "none: candidates only"}
        it["evidence"] += [e for e in ("E92", "E93", "E94") if e in rows and e not in it["evidence"]]
        if it["status"] == "absent" and outside > 0:
            it["flags"].append(f"{outside:g} mL of dense-lung candidates lie outside the historical baseline measure (spatially tagged, not interpreted)")
        if it.get("statement"):
            it["statement"] = it["statement"].rstrip() + " " + txt


def add_statements(items: list[dict], ck: dict) -> None:
    conds = {c["key"]: c for c in ck["conditions"]}
    for it in items:
        c = conds.get(it["key"], {})
        lab = it["label"]
        if it["status"] == "cannot tell" and it.get("historical_rule_answer") == "none" and c.get("mask_wording"):   # v3_7
            mw, v = c["mask_wording"], it.get("measured_value") or 0
            k = "empty" if v == 0 else "low"
            it["display"] = mw[f"display_{k}"]
            it["mask_state"] = "empty_mask" if v == 0 else "nonzero_low_volume"
            it["statement"] = mw[f"text_{k}"].format(value=v, limit=c.get("absent_below"), present=c.get("present_at_least"))
        elif it["status"] == "absent":
            measured = it["basis"].startswith("measured")
            it["display"] = "none measured" if measured else "not suggested"
            if measured and c.get("mask_wording"):      # v3_6: empty mask and low-volume mask are different observations
                mw, v = c["mask_wording"], it.get("measured_value") or 0
                k = "empty" if v == 0 else "low"
                it["display"] = mw[f"display_{k}"]
                it["mask_state"] = "empty_mask" if v == 0 else "nonzero_low_volume"
                it["statement"] = mw[f"text_{k}"].format(value=v, limit=c.get("absent_below"), present=c.get("present_at_least"))
            elif measured:
                tmpl = c.get("absent_text")
                if tmpl:
                    it["statement"] = tmpl.format(value=it.get("measured_value"), limit=c.get("absent_below"),
                                                  unit=c.get("unit", ""))
                else:
                    it["statement"] = f"No measurable {lab.lower()}."
            else:
                it["statement"] = f"{lab}: not suggested by the classifier (uncalibrated; nothing measured)."
        else:
            it["display"] = DISPLAY[it["status"]]
            lt = it.get("location_text")  # v3_4: location first, region named as HU-selected
            if lt and it["status"] == "present":
                it["statement"] = f"{lab}: present. HU-selected {lt}."
            elif lt and it["status"] == "cannot tell":
                it["statement"] = f"{lab}: cannot tell. Candidate region, not a confirmed finding: HU-selected {lt}."
            elif it["status"] == "present" and c.get("mask_wording") and it.get("measured_value") is not None and not it.get("two_source"):
                it["mask_state"] = "volume_at_or_above_present_limit"
                if c["mask_wording"].get("headline_present"):     # v3_8: a positive answer from a predicted mask alone is a suggestion
                    it["headline"], it["display"] = c["mask_wording"]["headline_present"], c["mask_wording"]["display_present"]
                    it["decision_basis"] = "predicted mask only"
                it["statement"] = c["mask_wording"]["text_present"].format(value=it["measured_value"], where=it.get("location") or "side not determined",
                                                                             present=c.get("present_at_least"))
            elif it["status"] == "present":
                it["statement"] = f"{lab}: present" + (f", {it['location']}" if it.get("location") else "") + "."
            elif it["status"] == "cannot tell":
                it["statement"] = f"{lab}: cannot tell from this scan."
                # wording only: a measured value under the 'present' limit, where a low value is not proof of absence
                if c.get("mask_wording") and it.get("measured_value") is not None:
                    it["mask_state"] = "volume_between_limits"
                if c.get("grey_text") and it.get("measured_value") is not None and it["basis"].startswith("measurement in grey zone"):
                    it["statement"] = c["grey_text"].format(value=it["measured_value"], present=c.get("present_at_least"),
                                                            unit=c.get("unit", ""))
            else:
                it["statement"] = f"{lab}: cannot assess ({it['basis']})."


# ---------- report text ----------

def banned_hits(text: str, ck: dict) -> list[str]:
    return sorted({w for w in ck["banned_terms"] if re.search(rf"\b{re.escape(w)}\b", text, re.I)})


def rule_based_checklist_report(cl: dict) -> dict:
    sugg = [i for i in cl["items"] if i["status"] == "present" and i.get("headline")]        # v3_8: suggested by a predicted mask, not confirmed
    pres = [i for i in cl["items"] if i["status"] == "present" and i["key"] != "lung_density" and not i.get("headline")]

    def where(i):  # v3_4 location text when the checklist has one, else the older string
        return ("HU-selected " + i["location_text"]) if i.get("location_text") else i.get("location")
    findings = [{"text": f"{i['label']}: present" + (f", {where(i)}" if where(i) else "")
                 + f" ({i['basis']}).", "evidence": i["evidence"]} for i in pres]
    unsure = [i["label"] + (f" (candidate region, not a confirmed finding: HU-selected {i['location_text']})"
                            if i.get("location_text") and i["key"] != "lung_density" else "")
              for i in cl["items"] if i["status"] == "cannot tell"]
    na = [f"{i['label']} ({i['basis']})" for i in cl["items"] if i["status"] == "cannot assess"]
    nm = [i["statement"] for i in cl["items"] if i.get("display") == "none measured" and i["key"] != "lung_density"]
    if nm:
        findings.append({"text": "Not detected by measurement: " + " ".join(nm),
                         "evidence": sorted({e for i in cl["items"] if i.get("display") == "none measured"
                                             for e in i["evidence"]})})
    if sugg:
        findings.append({"text": "Suggested by segmentation (not confirmed): " + " ".join(i["statement"] for i in sugg),
                         "evidence": sorted({e for i in sugg for e in i["evidence"]})})
    low = [i for i in cl["items"] if i.get("mask_state") in ("empty_mask", "nonzero_low_volume")]   # v3_6: never worded as excluded
    if low:
        findings.append({"text": "Below the reporting limit of a predicted outline (this does not exclude the finding): "
                                 + " ".join(i["statement"] for i in low), "evidence": sorted({e for i in low for e in i["evidence"]})})
    imp = ("Findings: " + "; ".join(f"{i['label'].lower()}" + (f" ({where(i)})" if where(i) else "")
                                    for i in pres) + "." if pres else "No checklist condition was confirmed present (see 'not detected by measurement' "
                                       "and 'cannot tell' below).")
    if sugg:
        imp += " Suggested by segmentation, not confirmed: " + "; ".join(i["label"].lower() + (f" ({where(i)})" if where(i) else "") for i in sugg) + "."
    return {"impression": imp + f" Scan type: {cl['scan_type']}. Research use only.",
            "findings": findings,
            "disagreements": [f"{i['label']}: {f}" for i in cl["items"] for f in i["flags"] if f != "contrast would help"],
            "limitations": (["Cannot tell: " + ", ".join(unsure) + "."] if unsure else [])
                           + (["Cannot assess: " + "; ".join(na) + "."] if na else [])
                           + ["'No measurable' means below the tool's detection limit; minimal findings "
                              "(for example a thin rim of fluid) can be missed.",
                              f"Thresholds are provisional ({cl['config_version']}) and not clinically validated.",
                              "Classifier scores are uncalibrated."]}


def _mask_worded(i: dict) -> bool:
    """v3_8: items whose wording must be copied because the status word alone would mislead."""
    return bool(i.get("headline")) or bool(i.get("provisional_threshold_result"))


def build_checklist_prompt(cl: dict, ev: dict, ck: dict) -> str:
    rows = {r["id"]: r for r in ev["rows"]}
    lines = []
    for i in cl["items"]:
        lines.append(f"- {i['label']}: {i.get('display', i['status']).upper()}"
                     + (f" | say: \"{i['statement']}\"" if i.get("statement") and (i["status"] == "absent" or _mask_worded(i)) else "")
                     + (f" | location: {i['location']}" if i.get("location") else "")
                     + f" | basis: {i['basis']} | evidence: {', '.join(i['evidence']) or 'none'}"
                     + (f" | check: {'; '.join(i['flags'])}" if i["flags"] else ""))
    used = sorted({e for i in cl["items"] for e in i["evidence"] if e in rows})
    evl = []
    for e in used:
        r = rows[e]
        v = r["value"]
        if isinstance(v, dict):
            v = json.dumps(v)
        evl.append(f"[{e}] {r['name']}: {v} {r['unit'] or ''}".strip())
    return (
        "You are writing a short research (not clinical) chest CT summary for a radiologist to review.\n"
        "The CONDITION CHECKLIST below is final: do not change any status and do not add conditions. "
        "Write an impression of one to three sentences, then one finding per condition marked PRESENT "
        "(with its location if given), then list the conditions marked CANNOT ASSESS with the reason. "
        "For a condition marked NONE MEASURED write 'no measurable ...' (use the 'say' text); never write "
        "'no <condition>' or 'absent', because the tool cannot see minimal findings. For NOT SUGGESTED say it "
        "was not suggested by the classifier. "
        + ("A condition marked SUGGESTED BY SEGMENTATION is NOT marked present: write one finding for it that begins with its "
           "'say' text, and never call it present, confirmed or a finding of the scan. For a condition marked CANNOT TELL (NO FLUID "
           "OUTLINED) or CANNOT TELL (LOW VOLUME OUTLINED) use the 'say' text; never write that it is absent, excluded or not present. "
           if any(_mask_worded(i) for i in cl["items"]) else "") +
        "Every finding must cite the evidence ids listed for that condition; any number must be copied "
        "from a cited row. Never use the words 'opacity', 'opacities' or 'opacification': say "
        "'ground-glass', 'consolidation' or 'increased lung density'. Keep thinking short.\n\n"
        f"SCAN TYPE: {cl['scan_type']}\n\nCONDITION CHECKLIST:\n" + "\n".join(lines)
        + "\n\nEVIDENCE ROWS:\n" + "\n".join(evl)
        + "\n\nReturn only a JSON object:\n"
        "{\n  \"impression\": \"<one to three sentences>\",\n"
        "  \"findings\": [{\"text\": \"<one present condition>\", \"evidence\": [\"E..\", ...]}],\n"
        "  \"disagreements\": [\"<what the reviewer should check>\"],\n"
        "  \"limitations\": [\"<limitation>\"]\n}"
    )


def validate_checklist_report(obj: object, ev: dict, cl: dict, ck: dict, base_validate) -> dict:
    """base checks (ids, numbers) + banned words + every finding must be a condition marked present."""
    v = base_validate(obj, ev)
    if not isinstance(obj, dict):
        return v
    text = json.dumps(obj)
    hits = banned_hits(text, ck)
    if hits:
        v["errors"].append(f"banned terms used: {hits}")
    pres = {i["label"].lower() for i in cl["items"] if i["status"] == "present"}
    absent = [i["label"].lower() for i in cl["items"] if i["status"] in ("absent", "cannot tell")]
    sugg = [i["label"].lower().split(" (")[0] for i in cl["items"] if assessment_category(i) == "suggested"]
    for k, f in enumerate(obj.get("findings") or []):      # v3_8: a suggestion must be written as a suggestion
        t = (f.get("text") or "").lower() if isinstance(f, dict) else ""
        for lab in sugg:
            if lab in t and "suggested by segmentation" not in t:
                v["flags"].append({"finding": k, "flag": f"states {lab} without 'suggested by segmentation': it is not a confirmed finding"})
    for k, f in enumerate(obj.get("findings") or []):
        t = (f.get("text") or "").lower() if isinstance(f, dict) else ""
        if not any(p.split(" (")[0] in t for p in pres):
            negated = any(w in t for w in ("not ", "no ", "absent", "cannot", "without"))
            claims = [a for a in absent if a.split(" (")[0] in t and not negated]
            if claims:
                v["flags"].append({"finding": k, "flag": f"states a condition the checklist does not mark present: {claims}"})
    nm = [i["label"].lower().split(" (")[0] for i in cl["items"] if i.get("display") == "none measured"
          or (i.get("provisional_threshold_result") or {}).get("answer") == "none"]   # v3_8: a historical 'none' is never an absence
    def strings(o):
        if isinstance(o, str):
            yield o
        elif isinstance(o, dict):
            for x in o.values():
                yield from strings(x)
        elif isinstance(o, list):
            for x in o:
                yield from strings(x)
    frags = [f for st in strings(obj) for f in re.split(r"(?<=[.;])\s+", st.lower())]
    for frag in frags:
        for lab in nm:
            if lab in frag and re.search(r"\b(no|absent|without|not present)\b", frag) and \
                    not re.search(r"measur|detect|limit|exclude|outlined", frag):
                v["flags"].append({"finding": "text", "flag": f"overclaims absence of {lab}: say 'no measurable'"})
                break
    v["valid"] = not v["errors"]
    return v
