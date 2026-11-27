"""NV-Reason-CT reader: config, answer reading and reader-vs-label scoring (no model, no GPU)."""
import json
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import run_nvreason as nv  # noqa: E402
import score_reader_checklists as sr  # noqa: E402
from lung_ct.vlm.checklist_schema import READER_CONDITIONS, validate_checklist  # noqa: E402

CFG = json.loads((ROOT / "configs" / "nvreason_ct_v1.json").read_text(encoding="utf-8"))


def test_config_conditions_match_reader_schema_in_order():
    assert [c["key"] for c in CFG["questions"]["conditions"]] == list(READER_CONDITIONS)


def test_config_revision_pinned_and_code_reviewed():
    m = CFG["model"]
    assert len(m["revision"]) == 40 and m["trust_remote_code"] is True
    assert m["remote_code_review"].startswith("2026-")
    assert "OpenMDW-1.1" in m["license"]
    assert m["quantization"] is None and m["attn_implementation"] == "sdpa"


def test_presence_question_is_the_documented_wording():
    q = CFG["questions"]
    c = next(c for c in q["conditions"] if c["key"] == "pleural_effusion")
    assert q["presence_template"].format(phrase=c["phrase"]) == "Is a pleural effusion present in this CT? Answer only Yes or No."


def test_only_ground_glass_lacks_a_model_label():
    assert [c["key"] for c in CFG["questions"]["conditions"] if c["model_label"] is None] == ["ground_glass"]


@pytest.mark.parametrize("text,want", [
    ("Yes", "present"), ("No.", "absent"), (" yes\n", "present"), ("<think>\n\n</think>\n\nNo", "absent"),
    ("<think>maybe yes</think>No", "absent"), ("Possibly", None), ("", None), ("Not sure", None), ("Yes, small", "present"),
])
def test_read_yes_no(text, want):
    assert nv.read_yes_no(text) == want


@pytest.mark.parametrize("text,want", [
    ("Right", "right"), ("left.", "left"), ("Both", "both"), ("Bilateral", "both"), ("Yes", None), ("", None),
    ("<think>x</think>\n\nLeft", "left"),
])
def test_read_side(text, want):
    assert nv.read_side(text) == want


def test_split_thinking():
    assert nv.split_thinking("<think>\nplan\n</think>\n\nFINDINGS: a") == ("plan", "FINDINGS: a")
    assert nv.split_thinking("FINDINGS: a") == ("", "FINDINGS: a")


def test_unpinned_revision_is_refused():
    with pytest.raises(RuntimeError):
        nv.snapshot_dir({"model_id": "nvidia/NV-Reason-CT", "revision": None})


def test_source_geometry_reports_what_cannot_fit(tmp_path):
    aff = np.diag([-0.8, -0.8, 1.5, 1.0])                                  # LPS, 0.8 x 0.8 x 1.5 mm
    p = tmp_path / "a.nii.gz"
    nib.save(nib.Nifti1Image(np.zeros((400, 400, 300), np.int16), aff), p)
    g = nv.source_geometry(p)
    assert g["orientation"] == "LPS"
    assert g["extent_mm"] == {"left_right": 320.0, "front_back": 320.0, "head_foot": 450.0}
    assert g["outside_cube_mm"] == {"left_right": 0.0, "front_back": 0.0, "head_foot": 66.0}


def test_save_cube_orientation_and_statistics(tmp_path):
    torch = pytest.importorskip("torch")
    cube = torch.full((1, 8, 8, 8), -1.0)
    cube[0, 7, :, :] = 0.15                                                # most superior slab bright
    info = nv.save_cube(cube, tmp_path)
    from PIL import Image
    cor = np.asarray(Image.open(tmp_path / "crop_coronal.png"))
    assert cor[0].min() == 255 and cor[-1].max() == 59                     # head at the top of the coronal picture
    assert info["hu_min"] == -1000.0 and info["hu_max"] == 150.0
    assert info["fraction_at_minus_1000"] == pytest.approx(7 / 8)


def _answer(present=()):
    return {"assessability": "assessable", "limitations": [],
            "conditions": {k: {"status": "present" if k in present else "absent", "side": "not_applicable"}
                           for k in READER_CONDITIONS}}


def test_assembled_answer_shape_is_schema_valid():
    assert validate_checklist(_answer({"emphysema"})) == []


def test_outcome_table():
    assert sr.outcome("present", True) == "hit" and sr.outcome("present", False) == "extra"
    assert sr.outcome("absent", True) == "miss" and sr.outcome("absent", False) == "agree_no"
    assert sr.outcome("cannot_tell", True) == "open" and sr.outcome(None, True) == "no_output"


def test_score_counts_and_common_cases_only():
    labels = {"a": {"Emphysema", "Lung nodule"}, "b": set(), "c": {"Cardiomegaly"}}
    runs = {"r1": {"a": _answer({"emphysema"}), "b": _answer({"cardiomegaly"}), "c": _answer()},
            "r2": {"a": None, "b": _answer()}}
    res = sr.score(runs, labels)
    assert res["cases"] == ["a", "b"]                                     # 'c' is not in every run
    r1, r2 = res["readers"]["r1"], res["readers"]["r2"]
    assert r1["found_of_labelled"] == "1 of 2" and r1["labelled_of_said_present"] == "1 of 2"
    assert r1["total"]["extra"] == 1 and r1["cases_all_absent"] == 0
    assert r2["total"]["no_output"] == len(sr.LABEL) and r2["cases_with_valid_answer"] == 1 and r2["cases_all_absent"] == 1
    assert res["labelled_scans"]["emphysema"] == 1 and res["labelled_scans"]["cardiomegaly"] == 0


def test_ground_glass_has_no_ct_rate_label():
    assert "ground_glass" not in sr.LABEL and len(sr.LABEL) == 17
