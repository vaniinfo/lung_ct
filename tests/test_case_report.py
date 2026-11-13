"""Case reports: standalone, escaped, failures visible; reviewer rows; infection by side."""
import re

import nibabel as nib
import numpy as np
import pytest
from PIL import Image

from lung_ct.metrics.volumes import infection_by_side
from lung_ct.reporting.case_report import (
    REVIEW_COLUMNS, contact_sheet, img_tag, render_case_html, reviewer_rows)

PARSED = {"assessability": "assessable", "opacity_presence": "present",
          "distribution": {"laterality": "bilateral", "craniocaudal": "lower_predominant",
                           "axial": "peripheral"},
          "description": "Patchy opacities.", "limitations": []}


def _case(vlm=True, parsed=PARSED, raw='{"a": 1}'):
    c = {"case_id": "case_x", "report_run_id": "P4_test",
         "stages": {"qc": {"status": "ok"}, "segmentation": {"status": "ok"},
                    "vlm": {"status": "ok" if vlm else "excluded",
                            "detail": "non-HU: display-scaled"}},
         "qc": {"ct": {"file": "x.nii.gz", "hu_valid": True}},
         "reference": {"lung_volumes": {"total_lung_volume_ml": 3000.0},
                       "infection_burden": {"infection_burden_pct": 12.5},
                       "infection_by_side": {"left": {"infection_ml": 10.0, "infection_pct_of_lung": 1.0},
                                             "right": {"infection_ml": 20.0, "infection_pct_of_lung": 2.0}}},
         "predicted": None, "comparison": None, "vlm": None}
    c["stages"]["segmentation"]["detail"] = ""
    if vlm:
        c["vlm"] = {"model_id": "m", "model_revision": "r", "raw_text": raw, "parsed": parsed,
                    "result": {"parse_status": "valid" if parsed else "invalid_json",
                               "errors": [], "warnings": []},
                    "model_input": {}}
    return c


def test_html_is_standalone_and_escapes_model_text():
    png = Image.new("RGB", (4, 4))
    import io
    buf = io.BytesIO()
    png.save(buf, format="PNG")
    html = render_case_html(_case(raw="<script>alert(1)</script>"),
                            {"ref_overlays": img_tag(buf.getvalue(), "x")})
    assert "<script>" not in html and "&lt;script&gt;" in html
    srcs = re.findall(r'(?:src|href)="([^"]+)"', html)
    assert srcs and all(s.startswith("data:image/png;base64,") for s in srcs)
    assert "Research use only" in html and "unvalidated" in html


def test_excluded_vlm_stage_is_shown_with_reason():
    html = render_case_html(_case(vlm=False), {})
    assert "excluded" in html and "non-HU: display-scaled" in html
    rows = reviewer_rows(_case(vlm=False))
    assert len(rows) == 1 and rows[0]["model_observation"].startswith("NO VALID OUTPUT")


def test_reviewer_rows_have_blank_clinician_columns():
    rows = reviewer_rows(_case())
    assert [r["finding"] for r in rows][:3] == ["assessability", "opacity_presence",
                                                 "distribution_laterality"]
    for r in rows:
        assert list(r) == REVIEW_COLUMNS or set(r) == set(REVIEW_COLUMNS)
        assert r["clinician_judgment"] == r["reviewer_id"] == ""
    assert "12.50%" in rows[1]["reference_availability"]
    failed = reviewer_rows(_case(parsed=None, raw="garbage"))
    assert failed[0]["model_observation"] == "NO VALID OUTPUT (invalid_json)"


def test_contact_sheet(tmp_path):
    paths = []
    for i in range(20):
        p = tmp_path / f"s{i:02d}.jpg"
        Image.new("RGB", (64, 64), (i * 10, 0, 0)).save(p)
        paths.append(p)
    import io
    with Image.open(io.BytesIO(contact_sheet(paths, cols=17, thumb=32))) as im:
        assert im.size == (17 * 32, 2 * 32)
    with pytest.raises(ValueError):
        contact_sheet([])


def test_infection_by_side_uses_given_labels():
    lung = np.zeros((10, 10, 10), np.uint8)
    lung[:5] = 1
    lung[5:] = 2
    inf = np.zeros_like(lung)
    inf[:5, :2] = 1          # 100 voxels in label-1 lung
    inf[5:, :1] = 1          # 50 voxels in label-2 lung
    aff = np.diag([2.0, 1.0, 1.0, 1.0])   # 2 mm^3 voxels
    r = infection_by_side(nib.Nifti1Image(inf, aff), nib.Nifti1Image(lung, aff),
                          left_label=1, right_label=2)
    assert r["left"]["infection_ml"] == pytest.approx(0.2)
    assert r["right"]["infection_ml"] == pytest.approx(0.1)
    assert r["left"]["infection_pct_of_lung"] == pytest.approx(20.0)
    assert r["right"]["infection_pct_of_lung"] == pytest.approx(10.0)


def test_review_flags_rules():
    from lung_ct.reporting.review_flags import vlm_review_flags
    side = {"left": {"infection_ml": 0.0}, "right": {"infection_ml": 130.0}}
    p = {**PARSED, "limitations": ["The images show a single slice through the chest."]}
    flags = {f["flag"] for f in vlm_review_flags(p, "x", 85, side)}
    assert flags == {"incorrect_input_description", "laterality_vs_reference_annotation"}
    both = {"left": {"infection_ml": 50.0}, "right": {"infection_ml": 60.0}}
    assert vlm_review_flags(PARSED, "x", 85, both) == []
    assert vlm_review_flags(PARSED, "x", 1, None) == []
    f = vlm_review_flags(PARSED, "a", 85, None, compare_raw="b",
                         compare_parsed={**PARSED, "opacity_presence": "absent"})
    assert {x["flag"] for x in f} == {"text_changes_with_image_processor",
                                      "fields_change_with_image_processor"}
    assert vlm_review_flags(None, None, 85, side) == []


def test_stage_categories():
    from lung_ct.reporting.review_flags import stage_category
    assert stage_category("segmentation", "excluded", False) == "planned_exclusion"
    assert stage_category("segmentation_eval", "not_run", True) == "planned_exclusion"
    assert stage_category("qc", "warn", False) == "qc_warning"
    assert stage_category("vlm", "parse_failure", False) == "execution_failure"
    assert stage_category("vlm", "failed", False) == "execution_failure"
    assert stage_category("vlm", "not_run", False) == "not_in_run"
    assert stage_category("vlm", "ok", False) is None
