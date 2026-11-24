"""checklist_v3_6: dense_lung_inventory_v2 (evidence preservation) and prospective pleural-effusion wording."""
import json
from pathlib import Path

import numpy as np

from lung_ct.agent.checklist import build_checklist, load_checklist_config
from lung_ct.agent.conditions import DENSE_TAGS, dense_lung_inventory_rows
from test_location_v3_4 import _ev

ROOT = Path(__file__).resolve().parent.parent
V35 = load_checklist_config(ROOT / "configs" / "checklist_v3_5.json")
V36 = load_checklist_config(ROOT / "configs" / "checklist_v3_6.json")
SHORT = ["RUL", "RML", "RLL", "LUL", "LLL"]


def _inventory(scale=1):
    """30 x 30 x 10 grid, 1 mL per voxel. Dense block of 4 x 6 x 2; fluid tag covers x >= 12, posterior tag y < 5."""
    d = np.zeros((30, 30, 10), bool)
    d[10:14, 2:8, 3:5] = True
    if scale == 0:
        d[:] = False
    near = np.zeros(d.shape, bool)
    near[12:, :, :] = True
    dep = np.zeros(d.shape, bool)
    dep[:, :5, :] = True
    lobe = np.zeros(d.shape, np.uint8)
    lobe[:15] = 3
    lobe[15:] = 5
    cfg = V36["measurements"]["dense_lung_inventory"]
    return d, near, dep, dense_lung_inventory_rows(d, near, dep, lobe, SHORT, 1.0, None, cfg, [-300, 100], 15, 0.2, None)


def test_inventory_keeps_everything_and_tags_partition_the_region():
    d, near, dep, rows = _inventory()
    assert [r["id"] for r in rows] == ["E92", "E93", "E94"]
    val = rows[2]["value"]
    assert rows[0]["value"] == float(d.sum()) == 48.0 and val["version"] == "dense_lung_inventory_v2"
    t = val["tags"]
    assert tuple(t) == DENSE_TAGS and round(sum(x["ml"] for x in t.values()), 1) == 48.0
    assert t["no_spatial_tag"]["ml"] == float((d & ~near & ~dep).sum()) == 12.0            # = historical baseline region
    assert t["fluid_adjacent"]["ml"] == 12.0 and t["posterior"]["ml"] == 12.0 and t["fluid_adjacent_and_posterior"]["ml"] == 12.0
    assert all(sum(x["per_lobe_ml"].values()) + x["outside_lobes_ml"] == x["ml"] for x in t.values())
    assert rows[1]["value"]["lobes"]["RLL"]["ml"] == 48.0
    txt = json.dumps(rows).lower()
    assert "compressed" not in txt and "not a finding" in txt and "no presence / absence limit" in txt
    # without a fluid outline nothing is tagged fluid-adjacent, and nothing is lost
    cfg = V36["measurements"]["dense_lung_inventory"]
    r2 = dense_lung_inventory_rows(d, None, dep, np.full(d.shape, 3, np.uint8), SHORT, 1.0, None, cfg, [-300, 100], 15, 0.2, None)
    assert r2[0]["value"] == 48.0 and r2[2]["value"]["tags"]["fluid_adjacent"]["ml"] == 0 and r2[2]["value"]["pleural_fluid_outline_present"] is False


def test_v3_6_changes_no_limit_and_no_measurement_setting():
    assert {k: v for k, v in V36["measurements"].items() if k != "dense_lung_inventory"} == V35["measurements"]
    a, b = {c["key"]: c for c in V35["conditions"]}, {c["key"]: c for c in V36["conditions"]}
    assert a.keys() == b.keys()
    for k in a:
        if k == "atelectasis":
            assert "present_at_least" not in b[k] and b[k]["historical_present_at_least"] == a[k]["present_at_least"]
            continue
        for f in ("present_at_least", "absent_below", "poor_inflation", "poor_inflation_present_at_least", "allow_classifier_absent", "classifier_key"):
            assert a[k].get(f) == b[k].get(f), (k, f)
    assert V36["classifier_can_establish_absence"] is False


def _ev36(cons_ml, e88, pleural=(0.0, 0.0), inv=True):
    e = _ev(0.2, cons_ml)
    for r in e["rows"]:
        if r["id"] == "E88":
            r["value"] = e88
        if r["id"] == "E71":
            r["value"] = pleural[0]
        if r["id"] == "E72":
            r["value"] = pleural[1]
    if inv:
        e["rows"] += _inventory()[3]
    return e


def test_inventory_never_sets_a_status_and_atelectasis_is_not_named_from_proximity():
    for cons, e88 in ((0.0, 0.0), (15.0, 28.0), (80.0, 246.0)):
        a = {i["key"]: i for i in build_checklist(_ev36(cons, e88, inv=False), {}, V35)["items"]}
        b = {i["key"]: i for i in build_checklist(_ev36(cons, e88), {}, V36)["items"]}
        b0 = {i["key"]: i for i in build_checklist(_ev36(cons, e88, inv=False), {}, V36)["items"]}
        for k in a:
            if k != "atelectasis":
                assert a[k]["status"] == b[k]["status"] == b0[k]["status"], k                    # inventory rows change no answer
        assert b["atelectasis"]["status"] == "cannot tell"
        assert a["atelectasis"]["status"] == ("present" if e88 >= 30 else "cannot tell")          # v3_5 unchanged
        for k in ("consolidation", "atelectasis", "lung_density"):
            dc = b[k]["dense_lung_candidates"]
            assert dc["total_ml"] == 48.0 and dc["outside_historical_baseline_ml"] == 36.0 and "E94" in b[k]["evidence"]
            assert "is not determined by the tool" in b[k]["statement"] and "compressed" not in json.dumps(b[k]).lower()
        assert "dense_lung_candidates" not in b["pleural_effusion"] and "dense_lung_candidates" not in a["consolidation"]
    b = {i["key"]: i for i in build_checklist(_ev36(0.0, 0.0), {}, V36)["items"]}
    assert b["consolidation"]["status"] == "absent" and any("outside the historical baseline" in f for f in b["consolidation"]["flags"])


def test_pleural_wording_separates_empty_mask_from_low_volume_and_excludes_nothing():
    def pe(l, r, ck):
        return next(i for i in build_checklist(_ev36(0.0, 0.0, (l, r)), {}, ck)["items"] if i["key"] == "pleural_effusion")
    empty, low, mid, big = pe(0, 0, V36), pe(1.5, 1.5, V36), pe(10.0, 8.4, V36), pe(40.0, 10.0, V36)
    assert (empty["status"], low["status"], mid["status"], big["status"]) == ("absent", "absent", "cannot tell", "present")   # limits unchanged
    assert [pe(*v, V35)["status"] for v in ((0, 0), (1.5, 1.5), (10.0, 8.4), (40.0, 10.0))] == ["absent", "absent", "cannot tell", "present"]
    assert empty["display"] == "no fluid outlined" and empty["mask_state"] == "empty_mask" and "does not exclude" in empty["statement"]
    assert low["display"] == "low volume outlined" and low["mask_state"] == "nonzero_low_volume" and "neither confirms nor excludes" in low["statement"]
    assert empty["statement"] != low["statement"]
    for it in (low, mid, big):
        assert "estimate" in it["statement"] and "predicted mask" in it["statement"]
    for it in (empty, low, mid):
        assert "No measurable" not in it["statement"] and "none measured" not in it.get("display", "")
    assert pe(0, 0, V35)["display"] == "none measured"                                            # historical wording untouched
    rep = __import__("lung_ct.agent.checklist", fromlist=["x"]).rule_based_checklist_report(build_checklist(_ev36(0.0, 0.0), {}, V36))
    assert any("does not exclude the finding" in f["text"] and "Pleural effusion" in f["text"] for f in rep["findings"])
    assert not any(f["text"].startswith("Not detected by measurement") and "Pleural effusion" in f["text"] for f in rep["findings"])
