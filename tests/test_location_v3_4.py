"""v3_4: per-lobe distribution rows, the provisional lobe-naming rule, and the guarantee that the update
changes no decision. Also the grading form v2 loader (earlier grades must never read as 'location assessed')."""
import copy
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from lung_ct.agent.checklist import build_checklist, load_checklist_config, location_report, rule_based_checklist_report  # noqa: E402
from lung_ct.agent.conditions import fissure_distance_mm, lobe_distribution  # noqa: E402

V33 = load_checklist_config(ROOT / "configs" / "checklist_v3_3.json")
V34 = load_checklist_config(ROOT / "configs" / "checklist_v3_4.json")
NAMES = ["RUL", "RML", "RLL", "LUL", "LLL"]


def _lobes():
    """Right lung = x 0..9 split into three lobes along z; left lung = x 12..21 split into two. 1 x 1 x 2 mm voxels."""
    li = np.zeros((22, 10, 30), np.uint8)
    li[0:10, :, 20:30], li[0:10, :, 10:20], li[0:10, :, 0:10] = 1, 2, 3
    li[12:22, :, 15:30], li[12:22, :, 0:15] = 4, 5
    return li


def test_v3_4_changes_no_rule():
    wording = ("absent_text", "grey_text", "grey_label", "note")   # sentences and notes only; no number or rule
    strip = lambda c: {k: v for k, v in c.items() if k not in wording}  # noqa: E731
    assert [strip(c) for c in V34["conditions"]] == [strip(c) for c in V33["conditions"]]
    assert V34["decision_rule"] == V33["decision_rule"] and V34["noise"] == V33["noise"]
    assert {k: v for k, v in V34["measurements"].items() if k != "lobe_distribution"} == V33["measurements"]
    assert "PROVISIONAL" in V34["location_reporting"]["status"]


def test_lobe_distribution_denominators_and_fissure():
    li = _lobes()
    zooms = (1.0, 1.0, 2.0)
    fis = fissure_distance_mm(li, NAMES, zooms)
    assert fis[5, 5, 19] == 0 and fis[5, 5, 20] == 0           # the two voxels either side of the RUL/RML boundary
    assert fis[5, 5, 25] == 10.0                               # 5 voxels of 2 mm away from it
    assert fis[0, 5, 29] > 5 and fis[12, 5, 29] > 5            # the gap between right and left lung is NOT a fissure
    mask = np.zeros(li.shape, bool)
    mask[0:10, :, 25:30] = True      # 500 voxels, all RUL, far from the fissure
    mask[0:10, 0:2, 18:20] = True    # 40 voxels of RML touching the RUL fissure
    mask[10:12, 0, 0] = True         # 2 voxels in no lobe
    d = lobe_distribution(mask, li, NAMES, vml=0.002, fissure_mm=fis, near_mm=5.0)
    assert d["region_voxels"] == 542 and d["region_ml"] == round(542 * 0.002, 2)
    assert d["lobes"]["RUL"]["voxels"] == 500 and d["lobes"]["RML"]["voxels"] == 40 and d["lobes"]["LLL"]["voxels"] == 0
    assert d["lobes"]["RUL"]["pct_of_lobe"] == 50.0             # 500 of the lobe's 1000 voxels
    assert d["lobes"]["RUL"]["pct_of_region"] == round(100 * 500 / 542, 1)
    assert d["lobes"]["RML"]["pct_of_lobe"] == 4.0
    assert d["lobes"]["RML"]["near_fissure_pct_of_lobe_part"] == 100.0 and d["lobes"]["RUL"]["near_fissure_pct_of_lobe_part"] == 0.0
    assert d["outside_lobes_pct_of_region"] == round(100 * 2 / 542, 1)
    assert d["near_fissure_pct_of_region"] == round(100 * 40 / 542, 1)
    assert set(d["lobes"]) == set(NAMES)                        # every lobe is kept, also the empty ones
    empty = lobe_distribution(np.zeros(li.shape, bool), li, NAMES, 0.002, fis, 5.0)
    assert empty["region_voxels"] == 0 and empty["lobes"]["RUL"]["pct_of_region"] is None


def _dist(shares, near=0.0, outside=0.0):
    return {"region_ml": 100.0, "region_voxels": 1000, "near_fissure_mm": 5.0, "near_fissure_pct_of_region": near,
            "outside_lobes_pct_of_region": outside,
            "lobes": {n: {"ml": s, "voxels": int(s * 10), "pct_of_lobe": s / 2, "pct_of_region": s} for n, s in shares.items()}}


def test_naming_rule_does_not_name_a_lobe_for_a_few_voxels():
    lr = V34["location_reporting"]
    rows = {"E83": {"id": "E83", "value": _dist({"RUL": 55.0, "RML": 9.9, "RLL": 25.1, "LUL": 10.0, "LLL": 0.0})}}
    r = location_report("ground_glass", "present", rows, lr, pleural_present=False)
    assert r["lobes_named"] == ["RUL", "RLL", "LUL"]            # 9.9% is not named, 10.0% is; largest share first
    assert r["pct_of_region"]["RML"] == 9.9                     # ...but the full distribution is preserved
    assert r["side"] == "both lungs" and not r["lobe_assignment_uncertain"]
    assert r["region_is"] == "HU-selected region"
    cand = location_report("ground_glass", "cannot tell", rows, lr, False)
    assert cand["region_is"] == "HU-selected region, not a confirmed finding"
    assert location_report("ground_glass", "absent", rows, lr, False) is None
    assert location_report("lung_nodule", "present", rows, lr, False) is None       # no region row for this condition
    assert location_report("ground_glass", "present", {"E83": {"id": "E83", "value": None}}, lr, False) is None


def test_uncertain_lobe_assignment_is_flagged():
    lr = V34["location_reporting"]
    sh = {"RUL": 60.0, "RML": 40.0, "RLL": 0.0, "LUL": 0.0, "LLL": 0.0}
    assert not location_report("consolidation", "present", {"E89": {"value": _dist(sh, near=49.9)}}, lr, False)["lobe_assignment_uncertain"]
    r = location_report("consolidation", "present", {"E89": {"value": _dist(sh, near=50.0)}}, lr, False)
    assert r["lobe_assignment_uncertain"] and "fissure" in r["uncertain_because"][0] and r["text"].endswith("lobe assignment uncertain")
    assert location_report("consolidation", "present", {"E89": {"value": _dist(sh, outside=10.0)}}, lr, False)["lobe_assignment_uncertain"]
    assert location_report("consolidation", "present", {"E89": {"value": _dist(sh)}}, lr, True)["uncertain_because"] == \
        ["pleural effusion present (compressed lung)"]
    poor = {"E89": {"value": _dist(sh)}, "E96": {"note": "poor inflation if above -780 HU; inflation: POOR"}}
    assert location_report("consolidation", "present", poor, lr, False)["uncertain_because"] == ["lungs poorly inflated"]


def _ev(gg_pct, cons_ml, with_dist=True):
    def row(i, v, **k):
        return {"id": i, "group": "x", "name": i, "value": v, "unit": None, **k}
    rows = [row("E70", 40, note="scan type: non-contrast"), row("E81", gg_pct), row("E85", cons_ml), row("E88", 0),
            row("E82", {"side": "right", "right_share_pct": 90, "zone": "upper", "zone_shares_pct": {"lower": 5, "middle": 15, "upper": 80},
                        "leading_lobe": "RUL", "leading_lobe_share_pct": 70}),
            row("E96", -850, note="inflation: adequate"), row("E71", 0.0), row("E72", 0.0)]
    if with_dist:
        rows += [row("E83", _dist({"RUL": 70.0, "RML": 20.0, "RLL": 6.0, "LUL": 4.0, "LLL": 0.0})),
                 row("E89", _dist({"RUL": 0.0, "RML": 0.0, "RLL": 100.0, "LUL": 0.0, "LLL": 0.0}))]
    return {"case_id": "t", "rows": rows}


def test_statuses_identical_and_statement_is_location_first():
    for gg, cons in ((0.2, 0.0), (3.0, 15.0), (12.0, 80.0), (40.0, 300.0)):
        a = build_checklist(_ev(gg, cons), {}, V33)["items"]
        b = build_checklist(_ev(gg, cons), {}, V34)["items"]
        assert [(i["key"], i["status"], i.get("measured_value"), i.get("location")) for i in a] == \
               [(i["key"], i["status"], i.get("measured_value"), i.get("location")) for i in b]
    items = {i["key"]: i for i in build_checklist(_ev(12.0, 15.0), {}, V34)["items"]}
    gg, co = items["ground_glass"], items["consolidation"]
    assert gg["status"] == "present" and gg["statement"] == "Ground-glass: present. HU-selected ground-glass-range region in RUL, RML."
    assert gg["location"] == "right lung, mainly upper zone; most in RUL (70%)"      # the earlier string is kept, unchanged
    assert co["status"] == "cannot tell" and "Candidate region, not a confirmed finding" in co["statement"] and "RLL" in co["statement"]
    assert "E83" in gg["evidence"] and "%" not in gg["statement"]
    rep = rule_based_checklist_report({"items": list(items.values()), "scan_type": "non-contrast", "config_version": "checklist_v3_4"})
    assert "HU-selected ground-glass-range region in RUL, RML" in rep["impression"]
    assert any("candidate region, not a confirmed finding" in x for x in rep["limitations"])
    # a v3_4 config on evidence that has no distribution rows falls back to the earlier wording, never invents lobes
    old = {i["key"]: i for i in build_checklist(_ev(12.0, 15.0, with_dist=False), {}, V34)["items"]}
    assert old["ground_glass"]["statement"] == "Ground-glass: present, right lung, mainly upper zone; most in RUL (70%)."
    assert "location_report" not in old["ground_glass"]


def test_generated_percentages_name_their_denominator():
    for c in V34["conditions"]:
        for f in ("absent_text", "grey_text", "grey_label", "note"):
            t = c.get(f, "")
            for k in range(len(t)):
                if t[k] == "%":
                    assert t[k:k + 15].startswith("% of total lung"), (c["key"], f, t)


def test_form_v2_loader_keeps_v1_grades_and_never_invents_location(tmp_path):
    import viewer_grading_v2 as gv2
    chk = {"items": [{"key": "ground_glass", "label": "Ground-glass", "status": "present",
                      "location_report": {"lobes_named": ["RUL"], "lobe_assignment_uncertain": True}, "location_text": "x in RUL"},
                     {"key": "pleural_effusion", "label": "Pleural effusion", "status": "absent"},
                     {"key": "cardiomegaly", "label": "Cardiomegaly", "status": "absent"}]}
    v1 = tmp_path / "v1.csv"
    v1.write_text("case_id,reviewer,saved_utc,condition_key,condition,agent_answer,agent_status,doctor_answer,severity,confidence,agreement,comment,checklist_config,run\n"
                  '"c","Dr X","2026-11-20T00:00:00Z","ground_glass","Ground-glass","present","present","present","mild","","agrees with agent","In RLL","checklist_v3_2","r"\n'
                  '"c","Dr X","2026-11-20T00:00:00Z","cardiomegaly","Cardiomegaly","none","absent","","","","","","checklist_v3_2","r"\n'
                  '"c","Dr X","2026-11-20T00:00:00Z","_overall","Overall comment","","","","","","","ok","checklist_v3_2","r"\n', encoding="utf-8")
    p = gv2.load_grades_csv(v1, "c", chk)
    assert p["form"] == "grades_v1" and p["overall"] == "ok"
    assert p["g"]["ground_glass"] == {"answer": "present", "severity": "mild", "confidence": "", "comment": "In RLL",
                                      "loc": {"status": "not_recorded_form_v1", "where": []}}
    assert p["g"]["cardiomegaly"]["loc"]["status"] == "not_assessed"            # ungraded stays ungraded
    v2 = tmp_path / "v2.csv"
    v2.write_text("case_id,reviewer,saved_utc,condition_key,doctor_answer,severity,confidence,comment,form_version,location_status,location_where\n"
                  '"c","Dr X","2026-11-23T00:00:00Z","ground_glass","present","","","","grades_v2","stated","RUL RLL"\n'
                  '"c","Dr X","2026-11-23T00:00:00Z","pleural_effusion","cannot_tell","","","","grades_v2","indeterminate",""\n', encoding="utf-8")
    q = gv2.load_grades_csv(v2, "c", chk)
    assert q["g"]["ground_glass"]["loc"] == {"status": "stated", "where": ["RUL", "RLL"]}
    assert q["g"]["pleural_effusion"]["loc"]["status"] == "indeterminate"
    items = {i["key"]: i for i in gv2.grading_items(chk)}
    assert items["ground_glass"]["loc_kind"] == "lobes" and items["pleural_effusion"]["loc_kind"] == "sides"
    assert items["cardiomegaly"]["loc_kind"] is None and items["ground_glass"]["agent_uncertain"] is True
    js = gv2.grading_script("c", "r", "checklist_v3_4", chk, "viewer", blinded=True)
    assert '"blinded": true' in js and "Not assessed" in js and "Indeterminate" in js
    try:
        gv2.grading_script("c", "r", "checklist_v3_4", chk, "viewer", prefill=p, blinded=True)
        raise AssertionError("a staged page must not accept earlier grades")
    except ValueError:
        pass
    assert json.loads(json.dumps(copy.deepcopy(p)))  # serialisable


def test_viewer_v3_ct_coding_is_lossless_and_header_is_unchanged():
    import base64
    import zlib
    import nibabel as nib
    import viewer_v3 as v3
    rng = np.random.default_rng(7)
    a = rng.integers(-1024, 3072, size=(7, 5, 6)).astype(np.int16)      # full clipped HU range: many 3-byte escapes
    a[2:5, 1:4, 2:5] = -850                                             # flat region: 1-byte codes
    a[0, 0, 0], a[6, 4, 5] = 3071, -1024
    stream, meta = v3.encode_ct_stream(a)
    back = v3.decode_ct_stream(stream, meta)
    assert back.dtype == np.int16 and np.array_equal(back, a)
    assert meta["voxels"] == a.size and meta["stream_bytes"] == len(stream) > a.size
    import hashlib
    assert meta["sha256"] == hashlib.sha256(a.astype("<i2").tobytes(order="F")).hexdigest()
    assert meta["checksum"] == sum((int(v) + meta["offset"] + 1) * ((i % 65521) + 1) for i, v in enumerate(a.ravel(order="F"))) % 2**32
    smooth = (np.indices((9, 8, 7)).sum(0) * 3 - 900).astype(np.int16)
    s2, m2 = v3.encode_ct_stream(smooth)
    assert np.array_equal(v3.decode_ct_stream(s2, m2), smooth)
    # embedded blocks: gzip container, complete base64 per block, header identical to nibabel's for the same grid
    aff = np.array([[-0.8, 0, 0, 100.0], [0, -0.8, 0, 90.0], [0, 0, 1.5, -300.0], [0, 0, 0, 1]])
    html, m = v3.ct_payload(a, aff)
    b64 = "".join(x.split(">")[1].split("<")[0] for x in html.strip().split("\n"))
    assert zlib.decompress(base64.b64decode(b64), wbits=31) == stream and m["parts"] == 1
    ref = nib.Nifti1Image(a, aff)
    ref.header.set_xyzt_units("mm")
    ref.set_qform(aff, code=1)
    ref.set_sform(aff, code=1)
    hdr = nib.Nifti1Header.from_fileobj(__import__("io").BytesIO(base64.b64decode(m["header_b64"])))
    for f in ("dim", "pixdim", "datatype", "bitpix", "qform_code", "sform_code", "srow_x", "srow_y", "srow_z",
              "quatern_b", "quatern_c", "quatern_d", "qoffset_x", "qoffset_y", "qoffset_z", "xyzt_units", "scl_slope", "scl_inter"):
        assert np.array_equal(np.nan_to_num(hdr[f]), np.nan_to_num(ref.header[f])), f
    assert float(hdr["vox_offset"]) == 352
    mask = (a > 0).astype(np.uint8)
    oh, om = v3.overlay_payload("pleural", mask, aff)
    raw = zlib.decompress(base64.b64decode(oh.split(">")[1].split("<")[0]), wbits=31)
    assert raw == mask.tobytes(order="F") and om["raw_bytes"] == mask.size and om["sha256"] == hashlib.sha256(raw).hexdigest()


def test_v3_5_classifier_cannot_establish_absence_or_change_any_answer():
    v35 = load_checklist_config(ROOT / "configs" / "checklist_v3_5.json")
    assert v35["classifier_can_establish_absence"] is False and v35["conditions"] == V34["conditions"]
    only_wording = {"pericardial_thickness_show_grade", "pericardial_thickness_note"}     # E79 text for new runs; no measurement setting
    assert {k: v for k, v in v35["measurements"].items() if k not in only_wording} == {k: v for k, v in V34["measurements"].items() if k not in only_wording}
    from lung_ct.agent.conditions import pericardial_thickness_row
    m = np.zeros((40, 40, 20), bool)
    m[10:30, 10:30, 5:9] = True
    a, b = pericardial_thickness_row(m, np.diag([1.0, 1.0, 2.0, 1.0]), V34), pericardial_thickness_row(m, np.diag([1.0, 1.0, 2.0, 1.0]), v35)
    assert a["value"] == b["value"] and "graded" in a["note"] and "graded" not in b["note"] and "NOT a caliper" in b["reliability"]

    def ev(p):
        e = _ev(3.0, 15.0)
        keys = sorted({c["classifier_key"] for c in v35["conditions"] if c.get("classifier_key")})
        e["rows"] += [{"id": f"E{50 + i}", "group": "classifier", "ckey": k, "name": k, "value": p, "unit": "p"} for i, k in enumerate(keys)]
        return e
    low34 = {i["key"]: i for i in build_checklist(ev(0.01), {}, V34)["items"]}
    low35 = {i["key"]: i for i in build_checklist(ev(0.01), {}, v35)["items"]}
    high35 = {i["key"]: i for i in build_checklist(ev(0.99), {}, v35)["items"]}
    assert low34["hiatal_hernia"]["status"] == "absent" and low34["hiatal_hernia"]["display"] == "not suggested"      # the old behaviour
    assert low35["hiatal_hernia"]["status"] == "cannot tell" and "cannot establish absence" in low35["hiatal_hernia"]["basis"]
    assert not any("classifier low" in i["basis"] for i in low35.values())
    assert not any(i.get("display") == "not suggested" for i in low35.values())
    # under v3_5 the classifier score cannot change ANY answer: lowest and highest score give the same statuses
    assert {k: v["status"] for k, v in low35.items()} == {k: v["status"] for k, v in high35.items()}
    # and it never turns a measured answer around: only former classifier negatives differ from v3_4
    for k in low34:
        if low34[k]["status"] != low35[k]["status"]:
            assert low34[k]["status"] == "absent" and "classifier low" in low34[k]["basis"] and low35[k]["status"] == "cannot tell", k


def test_neutral_wording_changes_text_only_and_removes_unsupported_claims():
    from lung_ct.agent.wording import banned_hits, neutral_text
    v35 = load_checklist_config(ROOT / "configs" / "checklist_v3_5.json")
    v35_plain = {k: v for k, v in v35.items() if k != "neutral_wording"}
    e = _ev(12.0, 80.0)
    for r in e["rows"]:
        if r["id"] == "E96":
            r["note"] = "poor inflation if above -780 HU; inflation: POOR"
    a = build_checklist(e, {}, v35_plain)["items"]
    b = build_checklist(e, {}, v35)["items"]
    assert [(i["key"], i["status"], i.get("measured_value")) for i in a] == [(i["key"], i["status"], i.get("measured_value")) for i in b]   # answers identical
    ta, tb = json.dumps(a), json.dumps(b)
    assert "poorly inflated" in ta and not banned_hits(tb) and "inspiration" in tb and "segmented lung volume" in tb
    old = ("Total lung volume (inflation check) | poor inflation if below 3000 mL | Median lung density (inflation check) | poor inflation if above -780 HU; inflation: POOR | "
           "Pericardial fluid, greatest thickness | thickest spot of the outlined fluid. If this is an effusion, 15 mm would be graded moderate (small under 10 mm, moderate up to 20 mm, "
           "large above). Nearly every scan has a few mm outlined | shallow breath | lungs poorly inflated (E95, E96): x")
    new = neutral_text(old)
    assert not banned_hits(new) and "3000 mL of segmented lung" in new and "-780 HU" in new and "cause not determined" in new
    assert "largest inscribed sphere diameter" in new and "not a thickness measurement" in new and "moderate" not in new and "small under" not in new
    assert neutral_text("Ground-glass: present.") == "Ground-glass: present."                    # unrelated text untouched


def test_v3_5_classifier_is_supporting_evidence_only():
    """Adopted 2026-11-23: no classifier score, low or high, can set or change an answer; the score stays visible."""
    v35 = load_checklist_config(ROOT / "configs" / "checklist_v3_5.json")
    keys = sorted({c["classifier_key"] for c in v35["conditions"] if c.get("classifier_key")})

    def ev(gg, cons, p):
        e = _ev(gg, cons)
        if p is not None:
            e["rows"] += [{"id": f"E9{i:02d}", "group": "classifier", "ckey": k, "name": k, "value": p, "unit": None} for i, k in enumerate(keys)]
        return e
    for gg, cons in ((0.2, 0.0), (3.0, 15.0), (12.0, 80.0)):
        runs = {p: build_checklist(ev(gg, cons, p), {}, v35)["items"] for p in (None, 0.01, 0.5, 0.99)}
        ref = [(i["key"], i["status"], i.get("measured_value")) for i in runs[None]]
        for p, items in runs.items():
            assert [(i["key"], i["status"], i.get("measured_value")) for i in items] == ref, p
        low = [i for i in runs[0.01] if i.get("classifier_p") is not None]
        assert low and all(i["classifier_p"] == 0.01 for i in low)                                  # score still reported
        assert any("cannot establish absence" in i["basis"] for i in low)
        assert not any(i["status"] == "absent" and "classifier low" in i["basis"] for i in low)
    old = build_checklist(ev(0.2, 0.0, 0.01), {}, V34)["items"]                                     # comparison config unchanged
    assert any(i["status"] == "absent" and "classifier low" in i["basis"] for i in old)
