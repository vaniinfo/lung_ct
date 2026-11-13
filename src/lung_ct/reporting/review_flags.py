"""Versioned, rule-based review flags and stage-outcome categories for reports.

Rules and thresholds live in configs/review_flag_rules_v1.json (frozen). They are
EXPLORATORY: written after inspecting this pilot's outputs, not prospectively defined
evaluation criteria. Flag kinds:
  input_description_error  - the model describes the supplied input incorrectly
                             (checkable against the logged input)
  review_flag              - a discrepancy that prioritises clinician review; not a
                             confirmed error (e.g. laterality vs a non-exhaustive annotation)
  processing_sensitivity   - output changes with an implementation detail
"""
from __future__ import annotations

import hashlib
import json
import re
from functools import lru_cache
from pathlib import Path

RULES_PATH = Path(__file__).resolve().parents[3] / "configs" / "review_flag_rules_v1.json"


@lru_cache(maxsize=4)
def load_rules(path: str | None = None) -> dict:
    p = Path(path) if path else RULES_PATH
    raw = p.read_bytes()
    rules = json.loads(raw)
    rules["_sha256"] = hashlib.sha256(raw).hexdigest()
    rules["_path"] = str(p)
    return rules


def flag_finding_map(rules: dict | None = None) -> dict:
    rules = rules or load_rules()
    return {k: v["review_finding"] for k, v in rules["rules"].items()}


def _flag(rules, name, detail):
    r = rules["rules"][name]
    return {"flag": name, "kind": r["kind"], "detail": detail,
            "rules_version": rules["rules_version"]}


def vlm_review_flags(parsed: dict | None, raw_text: str | None, n_images: int | None,
                     infection_by_side: dict | None,
                     compare_raw: str | None = None,
                     compare_parsed: dict | None = None,
                     rules: dict | None = None) -> list[dict]:
    rules = rules or load_rules()
    R = rules["rules"]
    flags = []
    if parsed is None:
        return flags

    r = R["incorrect_input_description"]
    pat = re.compile(r["pattern_regex"],
                     re.IGNORECASE if r.get("pattern_flags") == "IGNORECASE" else 0)
    texts = []
    for fld in r["fields_searched"]:
        v = parsed.get(fld, "")
        texts.extend(v if isinstance(v, list) else [v])
    m = pat.search(" ".join(texts))
    if n_images and n_images > 1 and m:
        flags.append(_flag(rules, "incorrect_input_description",
                           f"model states '{m.group(0)}'; {n_images} images were supplied"))

    r = R["laterality_vs_reference_annotation"]
    if infection_by_side:
        l_ml = infection_by_side["left"]["infection_ml"]
        r_ml = infection_by_side["right"]["infection_ml"]
        tot = l_ml + r_ml
        lat = parsed["distribution"]["laterality"]
        if tot > 0:
            small = min(l_ml, r_ml) / tot
            ref_side = "bilateral" if small >= r["unilateral_fraction_threshold"] else \
                ("right" if r_ml > l_ml else "left")
            if lat in r["model_values_checked"] and lat != ref_side:
                flags.append(_flag(rules, "laterality_vs_reference_annotation",
                                   f"model laterality '{lat}'; reference infection annotation "
                                   f"L {l_ml:.0f} mL / R {r_ml:.0f} mL (smaller side "
                                   f"{small:.1%}) -> predominantly {ref_side}. Review flag only: "
                                   "the annotation is not exhaustive."))

    if compare_raw is not None and raw_text is not None and compare_raw.strip() != raw_text.strip():
        flags.append(_flag(rules, "text_changes_with_image_processor",
                           "raw output differs from the run with the alternative image "
                           "processor (same images, prompt and decoding)"))
    if compare_parsed is not None:
        changed = [k for k in R["fields_change_with_image_processor"]["fields_compared"]
                   if compare_parsed.get(k) != parsed.get(k)]
        if changed:
            flags.append(_flag(rules, "fields_change_with_image_processor",
                               f"structured fields differ: {changed}"))
    return flags


def stage_category(stage: str, status: str, upstream_excluded: bool) -> str | None:
    """Categorise a non-ok stage outcome.

    planned_exclusion  - excluded by protocol (e.g. non-HU input), or not run because an
                         upstream stage was excluded
    qc_warning         - QC completed with a warning (e.g. display-scaled intensities)
    execution_failure  - the stage ran and failed / produced unusable output
    not_in_run         - stage not included in the supplied run
    """
    if status == "ok":
        return None
    if status == "excluded" or (status == "not_run" and upstream_excluded):
        return "planned_exclusion"
    if stage == "qc" and status == "warn":
        return "qc_warning"
    if status == "not_run":
        return "not_in_run"
    return "execution_failure"
