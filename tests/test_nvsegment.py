"""Second segmentation (NV-Segment-CT) and viewer v7: config, volume table, label-map payloads, comparison arithmetic (no model, no GPU).
What a browser draws is checked by scripts/test_seg_viewer.py."""
import base64
import json
import re
import sys
import zlib
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import compare_nvsegment_totalseg as cmp  # noqa: E402
import run_nvsegment as ns  # noqa: E402
import viewer_v6 as v6  # noqa: E402
import viewer_v7 as v7  # noqa: E402

CFG = json.loads((ROOT / "configs" / "nvsegment_ct_v1.json").read_text(encoding="utf-8"))
EV, EX = CFG["passes"]["everything"]["labels"], CFG["passes"]["extra"]["labels"]


def test_config_pins_revision_and_weights_and_runs_no_remote_code():
    m = CFG["model"]
    assert len(m["revision"]) == 40 and len(m["weights_sha256"]) == 64 and m["weights_file"].endswith(".safetensors")
    assert "NOT executed" in m["network"] and "NVIDIA Open Model License" in m["license"]
    ns.check_config(CFG)


def test_the_two_passes_cover_every_supported_class_once():
    not_run = {int(k) for k in CFG["passes"]["not_run"]}
    assert len(EV) == 117 and EX == [23, 24, 25, 26, 27, 128, 132]
    assert not set(EV) & set(EX) and (set(EV) | set(EX) | not_run) == set(range(1, 133))
    assert CFG["class_names"]["23"] == "lung tumor" and CFG["class_names"]["132"] == "airway" and CFG["class_names"]["28"] == "left lung upper lobe"


def test_check_config_refuses_overlapping_passes():
    bad = json.loads(json.dumps(CFG))
    bad["passes"]["extra"]["labels"] = [23, 28]
    with pytest.raises(ValueError):
        ns.check_config(bad)


def test_viewer_groups_show_every_class_exactly_once():
    groups = CFG["viewer_groups"]
    assert groups[0]["labels"] == "all" and groups[0]["source"] == "everything"
    ev = [x for g in groups[1:] if g["source"] == "everything" for x in g["labels"]]
    ex = [x for g in groups if g["source"] == "extra" for x in g["labels"]]
    assert sorted(ev) == sorted(EV) and sorted(ex) == sorted(EX)
    assert sorted(CFG["viewer_extra_priority"]["order"]) == sorted(EX)


def test_volume_rows_use_voxel_volume_and_keep_empty_classes():
    lab = np.zeros((10, 10, 10), np.uint8)
    lab[:2] = 28                       # 200 voxels
    extra = {x: np.zeros(lab.shape, np.uint8) for x in EX}
    extra[132][0, 0, :5] = 1           # 5 voxels
    rows = {r["label"]: r for r in ns.volume_rows(lab, extra, CFG, voxel_mm3=0.5 * 0.5 * 2.0)}
    assert len(rows) == 124
    assert rows[28]["voxels"] == 200 and rows[28]["ml"] == pytest.approx(200 * 0.5 / 1000)
    assert rows[132]["voxels"] == 5 and rows[132]["pass"] == "extra" and rows[132]["ml"] == pytest.approx(0.003, abs=1e-3)
    assert rows[23]["voxels"] == 0 and rows[23]["ml"] == 0 and rows[29]["voxels"] == 0          # empty classes are rows, not gaps


def test_slug_matches_the_file_names_the_page_builder_looks_for():
    assert ns.slug("lung tumor") == "lung_tumor" and ns.slug("colon cancer primaries") == "colon_cancer_primaries"
    assert [ns.slug(CFG["class_names"][str(x)]) for x in EX] == [re.sub(r"[^a-z0-9]+", "_", CFG["class_names"][str(x)].lower()).strip("_") for x in EX]


def test_dice_is_not_defined_for_two_empty_masks():
    a = np.zeros((4, 4, 4), bool)
    b = a.copy()
    assert cmp.dice(a, b) is None
    a[:2] = True
    b[1:3] = True
    assert cmp.dice(a, b) == pytest.approx(0.5) and cmp.dice(a, a) == 1.0 and cmp.dice(a, np.zeros_like(a)) == 0.0
    assert cmp.ml(a, 2.0) == pytest.approx(32 * 2.0 / 1000, abs=0.01)


def _decode(page_html: str, key: str, meta: dict) -> np.ndarray:
    parts = re.findall(r'<script type="application/octet-stream" class="vp" data-vol="' + key + r'" data-part="(\d+)">([^<]*)</script>', page_html)
    assert parts, "payload blocks not found"
    raw = zlib.decompress(base64.b64decode("".join(t for _, t in sorted(parts, key=lambda p: int(p[0])))), wbits=31)
    return np.frombuffer(raw, np.uint8).reshape(meta["dims"], order="F")


def test_label_payload_is_lossless_and_cropped_to_the_labels():
    lab = np.zeros((12, 9, 7), np.uint8)
    lab[3:6, 2:5, 1:4] = 28
    lab[8, 7, 5] = 132
    html, meta = v7.label_payload("nvseg", lab)
    assert meta["offset"] == [3, 2, 1] and meta["dims"] == [6, 6, 5] and meta["labels_present"] == [28, 132] and meta["voxels_on"] == 28
    sub = _decode(html, "nvseg", meta)
    back = np.zeros(lab.shape, np.uint8)
    o, d = meta["offset"], meta["dims"]
    back[o[0]:o[0] + d[0], o[1]:o[1] + d[1], o[2]:o[2] + d[2]] = sub
    assert np.array_equal(back, lab)


def test_extra_classes_merge_by_stated_priority_and_count_overlap():
    a, b = np.zeros((4, 4, 4), np.uint8), np.zeros((4, 4, 4), np.uint8)
    a[0], b[0, 0], b[1] = 1, 1, 1
    merged, overlap = v7.merge_extra({23: a, 132: b}, [23, 132])
    assert overlap == 4 and (merged[0] == 23).all() and (merged[1] == 132).all() and not merged[2:].any()
    merged2, _ = v7.merge_extra({23: a, 132: b}, [132, 23])
    assert (merged2[0, 0] == 132).all() and (merged2[0, 1:] == 23).all()
    with pytest.raises(AssertionError):
        v7.merge_extra({23: a}, [23, 132])


def test_seg_bundle_buttons_only_for_classes_with_voxels_and_names_the_rest():
    nv = np.zeros((6, 6, 6), np.uint8)
    nv[0], nv[1], nv[2] = 28, 57, 1
    ts = np.zeros_like(nv)
    ts[0] = 28
    _, vols, ov, seg = v7.seg_bundle({"tsseg": ts, "nvseg": nv, "nvsegx": np.zeros_like(nv)}, CFG["viewer_groups"], CFG["class_names"], {"x": 1})
    keys = [o["key"] for o in ov]
    assert keys == ["ts_lobes", "nv_all", "nv_lobes", "nv_trachea", "nv_other"] and set(vols) == {"tsseg", "nvseg"}
    assert next(o for o in ov if o["key"] == "nv_all")["labels"] == [1, 28, 57]
    assert all(o["src"] in vols and o["group"] in (v7.GROUP_TS, v7.GROUP_NV) for o in ov)
    assert "TotalSegmentator: Trachea" in seg["empty"] and "NV-Segment-CT: Class 'lung tumor'" in seg["empty"] and "NV-Segment-CT: Airway" in seg["empty"]
    assert "TotalSegmentator: Airways" not in seg["empty"]            # that map was not given at all: nothing is claimed about it
    assert seg["names"] == {"1": "liver", "28": "left lung upper lobe", "57": "trachea"} and seg["info"] == {"x": 1}


def test_same_structure_has_the_same_colour_in_both_models_and_lobes_differ():
    lobes = [v7.label_colour(x) for x in (28, 29, 30, 31, 32)]
    assert len(set(lobes)) == 5 and all(re.fullmatch(r"#[0-9a-f]{6}", v7.label_colour(x)) for x in range(1, 133))
    assert v7.label_colour(28) == v7.FIXED_COLOURS[28]                # one table for both models: the colour depends on the class number only


def test_v7_keeps_v6_and_adds_one_bar():
    assert v7.VIEWER_BUILD != v6.VIEWER_BUILD and v7.VIEWER_BUILD.startswith("checklist_viewer_v7_")
    assert v7.VIEWER_HTML.count('id="segbar"') == 1 and v7.VIEWER_HTML.replace(v7._SEG_BAR.rstrip("\n") + "\n", "") == v6.VIEWER_HTML
    assert "body.slabmode #segbar{display:none}" in v7.CSS and v7.CSS.startswith(v6.CSS)
    for word in ("setSlab", "slabData", "fillSlab", "SEGDEC", "labelAt", "comparison only"):
        assert word in v7.JS or word in v7.VIEWER_HTML or word in v7.GROUP_NV
    assert "fetch(" not in v7.JS and "XMLHttpRequest" not in v7.JS            # the page asks no server for anything


def test_ts_label_map_counts_overlap_and_keeps_the_lower_class():
    a, b = np.zeros((3, 3, 3), np.uint8), np.zeros((3, 3, 3), np.uint8)
    a[0], b[0, 0], b[1] = 1, 1, 1
    lab, overlap = v7.ts_label_map({28: a, 57: b})
    assert overlap == 3 and (lab[0] == 28).all() and (lab[1] == 57).all()


def test_scores_against_a_reference_lung_and_coverage_of_infected_lung():
    import score_segmenters_vs_reference as sc
    ref = np.zeros((10, 10, 10), bool)
    ref[:5] = True                                   # 500 voxels
    infected = np.zeros_like(ref)
    infected[3:7] = True                             # 200 of its 400 voxels lie inside the reference lung
    pred = np.zeros_like(ref)
    pred[:4] = True                                  # misses the last reference slab, covers half of the infected lung
    s = sc.side_scores(pred, ref, infected, vox_ml=0.002)
    assert s["dice"] == pytest.approx(2 * 400 / 900) and s["volume_error_pct"] == pytest.approx(-20.0)
    assert s["ref_infected_ml"] == pytest.approx(0.4) and s["infected_covered_pct"] == pytest.approx(50.0)
    assert s["missed_ref_ml"] == pytest.approx(0.2) and s["outside_ref_ml"] == 0
    none = sc.side_scores(pred, ref, np.zeros_like(ref), 0.002)
    assert none["infected_covered_pct"] is None      # no infected lung in the reference: not defined, not 100 %
