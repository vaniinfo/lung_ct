"""Fixed keyword parser for generated finding statements. No model, no learning: every rule is listed here.

parse_statement(text) returns what the statement SAYS about side, lobe and finding type, plus three
properties that switch location checks off: negation, uncertainty and more than one finding in one statement.
A check is applied only when the stated location is unambiguous; otherwise the check reports 'not_applicable'
with the reason, never a guess.

PROVISIONAL (audit_review_protocol_v0_draft): rules may change until the protocol is frozen.
"""
from __future__ import annotations

import re

PARSER_VERSION = "audit_parser_v0_draft"

LOBE_ORDER = ("RUL", "RML", "RLL", "LUL", "LLL")
_LEVEL = {"upper": "U", "superior": "U", "middle": "M", "lower": "L", "inferior": "L"}

# finding types: first matching pattern per type; a statement can hold several types
TYPES = {
    "ground_glass": r"ground[\s-]*glass|crazy[\s-]*paving",
    "consolidation": r"consolidat|air bronchogram|pneumonic infiltrat",
    "pleural_effusion": r"pleural (?:effusion|fluid)|effusions? in (?:the |both )?pleural|hydrothorax|loculat",
    "nodule": r"nodul|\bmass(?:es)?\b|metasta",
    "emphysema": r"emphysem|bulla",
    "fibrotic_scarring": r"scar|fibro|sequel",
    "atelectasis": r"atelecta|collaps",
    "bronchiectasis": r"bronchiecta|\bectasi|bronch\w* dilat",      # \b: 'atelectasis' also contains 'ectasi'
    "septal_thickening": r"septal (?:thicken|prominen)|interlobular",
    "peribronchial_thickening": r"peribronch",
    "mosaic_attenuation": r"mosaic",
    "calcification": r"calcifi(?:cation|ed plaque)",
    "cyst": r"\bcyst",
    "pneumothorax": r"pneumothora",
    "pleural_thickening": r"pleural thicken",
}
# (main, descriptor): the second type only describes the first one in the same statement, so it is not a second finding.
# Ground-glass + consolidation, and effusion + atelectasis, deliberately stay two findings.
_DESCRIPTOR_ONLY = {
    ("nodule", "calcification"),          # "calcific nodule"
    ("consolidation", "atelectasis"),     # "consolidation-atelectasis" is one region
}
_NEGATION = (r"\bno\b|\bnot\b|\bwithout\b|\babsen(?:t|ce)\b|\bnegative for\b|\bfree of\b|\bresolved\b|\bresolution of\b|"
             r"\bno longer\b|\bnone\b|\bunremarkable\b|\bnormal\b|\bclear\b")
_UNCERTAIN = (r"\bpossibl|\bprobabl|\bsuspicio|\bsuspect|\bmay\b|\bmight\b|\bcould\b|cannot be (?:excluded|ruled out)|"
              r"can ?not be (?:excluded|ruled out)|\bquestionabl|\bequivocal|\blikely\b|\bversus\b|\bvs\.?\b|\?|"
              r"\bdifferential\b")
# Not treated as uncertainty: "nonspecific", "indeterminate", "compatible with", "suggestive of". They qualify what a
# finding means, not whether it is there.
_BILATERAL = (r"\bbilateral(?:ly)?\b|\bbibasilar\b|\bbiapical\b|\beach lung\b|"
              r"\bboth (?:lungs?|sides|pleural \w+|hemithora\w*|apices|bases|(?:lung )?(?:upper|lower) lobes)\b")
# "both" alone is not bilateral ("both the upper and lower lobes of the right lung"); "diffuse" can be one-sided.
_COMPARE = r"\bcompared (?:to|with)\b|\bthan\b|\bmore (?:pronounced|prominent|marked|evident)\b|\bpredominan"


def _lobes(t: str) -> list[str]:
    found = []

    def add(side: str | None, level: str):
        lv = _LEVEL[level]
        sides = [side] if side else ["right", "left"]
        for s in sides:
            if lv == "M":
                code = "RML" if s == "right" else None          # there is no left middle lobe
            else:
                code = ("R" if s == "right" else "L") + lv + "L"
            if code and code not in found:
                found.append(code)

    lvl = r"(upper|superior|middle|lower|inferior)"
    # "right upper lobe", "left lower lobes", "right lung upper lobe"
    for m in re.finditer(rf"\b(right|left)\s+(?:lung\s+)?{lvl}\s+lobes?\b", t):
        add(m.group(1), m.group(2))
    # "upper lobe of the right lung", "lower lobes of both lungs"
    for m in re.finditer(rf"\b{lvl}\s+lobes?\s+of\s+(?:the\s+)?(right|left|both)\s+lungs?\b", t):
        add(None if m.group(2) == "both" else m.group(2), m.group(1))
    # "both lower lobes", "bilateral upper lobes"
    for m in re.finditer(rf"\b(?:both|bilateral)\s+(?:lung\s+)?{lvl}\s+lobes?\b", t):
        add(None, m.group(1))
    if re.search(r"\blingula", t) and "LUL" not in found:
        found.append("LUL")
    # a bare "middle lobe" can only be the right middle lobe
    if re.search(r"\bmiddle lobe\b", t) and "RML" not in found:
        found.append("RML")
    return [c for c in LOBE_ORDER if c in found]


def _bare_lobe_without_side(t: str) -> bool:
    """'in the lower lobe' with no side anywhere in the phrase: lobe level known, side not."""
    for m in re.finditer(r"\b(upper|superior|lower|inferior)\s+lobes?\b", t):
        before = t[max(0, m.start() - 20):m.start()]
        after = t[m.end():m.end() + 30]
        if not re.search(r"\b(right|left|both|bilateral)\b", before) and not re.search(r"^\s+of\s+(?:the\s+)?(right|left|both)", after):
            return True
    return False


def parse_statement(text: str | None) -> dict:
    """What the statement says. All fields are derived by the fixed rules above."""
    out = {"parser_version": PARSER_VERSION, "text": text, "types": [], "side": None, "lobes": [],
           "negated": False, "uncertain": False, "multiple_findings": False, "side_comparison": False,
           "lobe_side_missing": False, "location_checkable": False, "why_not_checkable": []}
    if not text or not text.strip():
        out["why_not_checkable"].append("empty statement")
        return out
    t = re.sub(r"\s+", " ", text.lower().replace("-", " ").replace("，", ", ").replace("；", "; "))
    out["types"] = [k for k, pat in TYPES.items() if re.search(pat, t)]
    out["negated"] = bool(re.search(_NEGATION, t))
    out["uncertain"] = bool(re.search(_UNCERTAIN, t))
    out["side_comparison"] = bool(re.search(_COMPARE, t))
    out["lobes"] = _lobes(t)
    out["lobe_side_missing"] = _bare_lobe_without_side(t)

    has_r, has_l = bool(re.search(r"\bright\b", t)), bool(re.search(r"\bleft\b", t))
    if re.search(_BILATERAL, t) or (has_r and has_l and not out["side_comparison"]):
        out["side"] = "bilateral"
    elif has_r and has_l:
        out["side"] = None                      # "more on the right than the left": both mentioned, no single stated side
    elif has_r:
        out["side"] = "right"
    elif has_l:
        out["side"] = "left"
    elif out["lobes"]:
        sides = {"right" if c.startswith("R") else "left" for c in out["lobes"]}
        out["side"] = sides.pop() if len(sides) == 1 else "bilateral"

    # more than one finding in one statement: several types (except pure descriptors) or an explicit list
    types = set(out["types"])
    for main, desc in _DESCRIPTOR_ONLY:
        if main in types and desc in types:
            types.discard(desc)
    listed = bool(re.search(r";|, and\b|\bas well as\b|\balong with\b|\baccompan|\bassociated with\b|\bwith (?:adjacent|associated)\b", t))
    out["multiple_findings"] = len(types) > 1 or (listed and len(out["types"]) > 1)

    # Uncertain wording is recorded but does not switch the location checks off: where the outline lies can be
    # compared with the stated place whether or not the statement hedges.
    for flag, why in (("negated", "negated statement"),
                      ("multiple_findings", "several findings in one statement"),
                      ("side_comparison", "side comparison (more on one side)")):
        if out[flag]:
            out["why_not_checkable"].append(why)
    if out["side"] is None and not out["lobes"]:
        out["why_not_checkable"].append("no side or lobe stated")
    out["location_checkable"] = not out["why_not_checkable"]
    return out
