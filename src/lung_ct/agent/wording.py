"""Neutral wording for two statements the tool cannot support (text only; no number, rule or answer changes).

1. Inspiration. The checklist has a gate that fires when the SEGMENTED lung volume is below a limit or the median
   density inside the lung outline is above a limit. Earlier text called this "lungs poorly inflated" / "shallow
   breath" / "poor inflation". The tool has no independent assessment of inspiratory adequacy: low volume or high
   density can equally come from disease (consolidation, effusion, fibrosis), body size, or an incomplete lung outline.
   The neutral text says what was measured and that inspiration was not assessed.

2. Pericardial "greatest thickness" (row E79). It is the diameter of the largest sphere that fits inside the
   predicted outline: a property of the automatic mask, not a thickness measurement of fluid, and it must not carry a
   small / moderate / large label.

Used by checklist configs with "neutral_wording": true (from checklist_v3_5) and by corrected page copies
(build_viewer_page.py --corrected-wording), which leave existing evidence and checklist files untouched.
"""
from __future__ import annotations

import re

FLAG = "lung volume / density check"
PLAIN = [
    ("lungs poorly inflated (E95, E96), density may be normal lung",
     "low segmented lung volume or high median lung density (E95, E96; inspiration was not assessed independently), so density in this range may be normal lung"),
    ("lungs poorly inflated (E95, E96)", "low segmented lung volume or high median lung density (E95, E96; inspiration was not assessed independently)"),
    ("poor-inflation limit", f"limit applied when the {FLAG} is flagged"),
    ("lungs poorly inflated: amount may be overestimated", f"{FLAG} flagged: amount may be overestimated"),
    ("lungs poorly inflated", "low segmented lung volume or high median lung density (inspiration not assessed independently)"),
    ("With poorly inflated lungs", f"When the {FLAG} is flagged,"),
    ("with poorly inflated lungs", f"when the {FLAG} is flagged"),
    ("poorly inflated", f"flagged by the {FLAG}"),
    ("Total lung volume (inflation check)", f"Segmented lung volume ({FLAG})"),
    ("Median lung density (inflation check)", f"Median density inside the lung outline ({FLAG})"),
    ("; inflation: adequate", "; this scan: not flagged"),
    ("shallow-breath", "flagged-check"),
    ("shallow breath", f"{FLAG} flagged"),
    ("Breath-in check", "Lung volume / density check"),
    ("breath-in check", FLAG),
    ("Pericardial fluid, greatest thickness", "Pericardial fluid outline, largest inscribed sphere diameter"),
    ("Greatest thickness (E79) is shown for information.",
     "The largest inscribed sphere diameter of the outline (E79) is shown for information; it is not a thickness measurement."),
]
REGEX = [
    (re.compile(r"poor inflation if below (\d+) mL"), rf"the {FLAG} is flagged below \1 mL of segmented lung"),
    (re.compile(r"poor inflation if above (-?\d+) HU; inflation: POOR"),
     rf"the {FLAG} is flagged above \1 HU; this scan: FLAGGED (cause not determined: shallow inspiration, disease, body size or an incomplete lung outline can each do this)"),
    (re.compile(r"poor inflation if above (-?\d+) HU"), rf"the {FLAG} is flagged above \1 HU"),
    (re.compile(r"poor inflation if median above (-?\d+) HU or volume below (\d+) mL"), rf"flagged if the median is above \1 HU or the segmented volume is below \2 mL"),
    (re.compile(r"thickest spot of the outlined fluid\. If this is an effusion, [^.]*?\(small under [^)]*\)\. Nearly every scan"),
     "diameter of the largest sphere that fits inside the predicted outline; not a thickness measurement of fluid and not a size grade. Nearly every scan"),
]
# phrases that must not remain in a corrected page or in neutral checklist text
BANNED = [re.compile(p, re.I) for p in (r"poorly inflated", r"poor[- ]inflation", r"shallow[- ]breath", r"inflation: (POOR|adequate)", r"\(inflation check\)", r"breath-in check",
                                        r"greatest thickness", r"would be graded")]


def neutral_text(s: str) -> str:
    for a, b in PLAIN:
        s = s.replace(a, b)
    for rx, b in REGEX:
        s = rx.sub(b, s)
    return s


def banned_hits(s: str) -> list[str]:
    return sorted({m.group(0) for rx in BANNED for m in rx.finditer(s)})


def neutral_item(item: dict) -> None:
    """Apply the neutral wording to the text fields of one checklist item, in place. Status and numbers are not touched."""
    for k in ("basis", "statement", "location_text"):
        if isinstance(item.get(k), str):
            item[k] = neutral_text(item[k])
    if item.get("flags"):
        item["flags"] = [neutral_text(f) for f in item["flags"]]
    lr = item.get("location_report")
    for part in ([lr] if lr and not lr.get("parts") else (lr or {}).get("parts", [])):
        part["uncertain_because"] = [neutral_text(x) for x in part.get("uncertain_because", [])]
        if isinstance(part.get("text"), str):
            part["text"] = neutral_text(part["text"])
