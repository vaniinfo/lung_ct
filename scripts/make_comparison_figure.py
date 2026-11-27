"""Related-work comparison (core env): one data table -> Markdown table, source matrix, SVG and PNG.

Everything about a cell (status, short note, source, location, version) is written once in SYSTEMS below.
Statuses: Documented / Partial / Not reported / Unverified for published systems; Implemented for ThoraxCheck (validation is stated separately).
"Not reported" means the examined sources do not describe it; it does not mean absent.

Writes docs/related_work/comparison_table.md, docs/related_work/source_matrix.md,
docs/images/related_work_comparison.svg and .png.
Usage: .venv/Scripts/python.exe scripts/make_comparison_figure.py
"""
from __future__ import annotations

import html
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REVIEW_DATE = "2026-11-27"
TITLE = "Reported capabilities of CT interpretation and evidence-review systems"

CRITERIA = [
    ("C1", "Measurements from predicted masks",
     "At inference the system computes numeric quantities (volume, HU statistics, size or diameter, count, ratio) from segmentation masks it predicts, "
     "and uses them in its output. A mask, region crop or heatmap alone does not count."),
    ("C2", "Versioned per-condition rules",
     "The final answer for each condition is produced by documented, deterministic, condition-specific rules (thresholds, precedence) with a version "
     "identifier. A fixed label set, a classifier head, a report template, question answering or a language model deciding does not count."),
    ("C3", "Unresolved state with reason",
     "The output schema has a state distinct from present / absent (for example 'cannot tell'), emitted under defined conditions, with a reason recorded "
     "per item. A free-text verbal refusal or a low confidence score is Partial at most."),
    ("C4", "Image-quality checks in rules",
     "Measured image-quality properties (noise, inspiration, contrast phase, HU validity, orientation, artefacts) change or block answers. "
     "Input format, shape or file validation does not count."),
    ("C5", "Claims linked to evidence",
     "Each per-condition assessment or report claim references specific evidence items (measurement identifiers, specific tool outputs, masks or regions) "
     "that a reviewer can inspect. A reasoning or tool-call trace not tied to individual claims is Partial at most."),
    ("C6", "End-to-end on one 24 GB GPU",
     "The sources report the complete system (all components needed for its main output) running on one GPU with at most 24 GB, or a whole-pipeline peak "
     "memory of at most 24 GB. Parameter count alone does not establish this; a cloud language-model API means the pipeline is not on one local GPU."),
    ("C7", "Structured clinician grading UI",
     "A user interface in which clinicians view the scan and the outputs and record structured per-item grades that are stored or exported. "
     "Having such an interface says nothing about clinical effectiveness."),
]

D, P, N, U, I = "Documented", "Partial", "Not reported", "Unverified", "Implemented"


def cell(status, note, src="", loc="", ver=""):
    return {"status": status, "note": note, "src": src, "loc": loc, "ver": ver}


AX = lambda n, v: f"https://arxiv.org/html/{n}{v}"  # noqa: E731

SYSTEMS = [
    {"group": "This work", "name": "ThoraxCheck (ours)", "kind": "Complete pipeline, CT, 2026", "ref": "this repository (Python package lung_ct)",
     "version": "checklist_v3_12, viewer v7, code as of " + REVIEW_DATE,
     "cells": [
         cell(I, "Volumes, HU-range regions, aortic calcium, heart ratio, fluid volume, nodule candidates from TotalSegmentator masks", "repository",
              "src/lung_ct/agent/evidence.py, conditions.py; runs/A1_agent_*", "v3_12"),
         cell(I, "Per-condition limits, precedence and two-source rule in a versioned config; limits fitted on development scans, not validated", "repository",
              "configs/checklist_v3_12.json; src/lung_ct/agent/checklist.py (build_checklist, _two_source)", "v3_12"),
         cell(I, "'Cannot tell' and 'cannot assess' with a stored basis per item; shown on the page and exported", "repository",
              "checklist_checklist_v3_12.json in each run; export_row in checklist.py; e.g. valid_592_a_1: 14 of 20 unresolved, all with basis", "v3_12"),
         cell(I, "Noise, lung-inflation and contrast gates change answers; HU validity and orientation exclude scans. Gate limits not validated", "repository",
              "evidence.py (image_noise, blocked); checklist.py (poor_inflation, needs_contrast); anatomy_contrast_check.py. On 10 RS01 scans: contrast gate 10x, noise and inflation 0x",
              "v3_12"),
         cell(I, "Each item lists evidence row IDs and names model sources; the items share one segmentation, so they are not independent evidence", "repository",
              "item['evidence'] in checklist files; valid_592_a_1: 19 of 20 items list IDs, all resolve to evidence rows", "v3_12"),
         cell(I, "All ten steps completed for 10 scans on one RTX 3090 (24 GB), models run one after another; peak 12.6 GB (NV-Reason-CT)", "run records",
              "runs/test_logs/pipeline_rs01_first10/*.log; pack M15 (2026-11-27)", REVIEW_DATE),
         cell(I, "Per-item grading (answer, severity, confidence, location, comment), CSV export and collection; browser-tested; used by one clinician on 4 development scans",
              "repository", "build_viewer_page.py, viewer_grading_v2.py; scripts/test_slab_viewer.py (grading and CSV); collect_grades.py; tests/test_grade_import.py",
              "grades_v2"),
     ]},
    {"group": "CT systems and agents", "name": "RadAgent", "kind": "Complete system, CT, 2026",
     "ref": "Roschewitz et al., arXiv 2604.15231", "version": "arXiv v2 (1 Jun 2026); no public code found",
     "cells": [
         cell(N, "Segmentation tools return masks; a voxel count is used only to pick a slice", AX("2604.15231", "v2"), "Methods: RadAgent Toolbox; Fig. 4; Table A.1", "v2"),
         cell(N, "Qwen3-14B decides over tool outputs; the checklist is 'short and coarse'", AX("2604.15231", "v2"), "Methods: Diagnostic checklist; Windowing", "v2"),
         cell(N, "No designed indeterminate state; hedging only as free text from VQA tools", AX("2604.15231", "v2"), "Fig. A.14", "v2"),
         cell(N, "Windowing for display only", AX("2604.15231", "v2"), "Methods: Windowing", "v2"),
         cell(P, "Scratchpad and tool-call trace; not tied to individual claims", AX("2604.15231", "v2"), "Results; Methods: Robustness and faithfulness; Fig. 3", "v2"),
         cell(N, "Reported on eight GPUs on two nodes; per-GPU memory not given", AX("2604.15231", "v2"), "Methods: Toolbox; Table A.1", "v2"),
         cell(N, "Human-in-the-loop interface named as future work", AX("2604.15231", "v2"), "Discussion", "v2"),
     ]},
    {"group": "CT systems and agents", "name": "MedScribe", "kind": "Complete system, CT, 2026",
     "ref": "Orlando et al., arXiv 2605.01779", "version": "arXiv v1 (3 May 2026); code not public",
     "cells": [
         cell(D, "86-dimensional features from TotalSegmentator masks: volumes, extent, laterality fractions, HU statistics", AX("2605.01779", "v1"), "Sec. 2.2; Sec. 3", "v1"),
         cell(N, "Prompted language model (ReAct) with retrieval decides", AX("2605.01779", "v1"), "Sec. 2.1, 2.3, 3", "v1"),
         cell(N, "Borderline or contradictory findings 'not explicitly handled'", AX("2605.01779", "v1"), "Sec. 5", "v1"),
         cell(N, "Mask-volume plausibility used only in post-hoc failure analysis", AX("2605.01779", "v1"), "Sec. 4", "v1"),
         cell(P, "'Transparent chain of evidence' claimed; claims not linked to individual features", AX("2605.01779", "v1"), "Sec. 5; Suppl. S4", "v1"),
         cell(N, "Single H100 reported; memory use not stated", AX("2605.01779", "v1"), "Sec. 3; Suppl. S6, S7", "v1"),
         cell(N, "No review interface described", AX("2605.01779", "v1"), "whole text", "v1"),
     ]},
    {"group": "CT systems and agents", "name": "3DMedAgent", "kind": "Complete system, CT and other 3D, ICML 2026",
     "ref": "Wang et al., arXiv 2602.18064; github.com/jinlab-imvr/3DMedAgent", "version": "arXiv v2 (8 Mar 2026); repo commit 3ee6a49 (2026-05-25)",
     "cells": [
         cell(D, "Organ size, mean HU and z-range from predicted masks into memory (Eq. 1); repo adds lesion volume, HU and location",
              AX("2602.18064", "v2") + " ; https://github.com/jinlab-imvr/3DMedAgent", "Sec. 4.1, Eq. 1; Structural_Report/process_organs/liver.py; segmentation.py", "v2; 3ee6a49"),
         cell(N, "Multimodal LLM gives the answer; repo thresholds only shape memory text, unversioned", AX("2602.18064", "v2"), "Sec. 4.3; liver.py lines 19-25", "v2; 3ee6a49"),
         cell(N, "Records assumptions; no separate unresolved state", AX("2602.18064", "v2"), "Sec. 4.3", "v2"),
         cell(N, "One statement gated on a contrast-phase value that appears supplied, not measured (unconfirmed)", "https://github.com/jinlab-imvr/3DMedAgent",
              "liver.py line 40", "3ee6a49"),
         cell(P, "Structured output lists memory evidence used per step (Eq. 4); not shown per claim", AX("2602.18064", "v2"), "Sec. 4.3, Eq. 4", "v2"),
         cell(N, "No GPU memory reported; one variant uses an external API model", AX("2602.18064", "v2"), "Sec. 5.1; README", "v2; 3ee6a49"),
         cell(N, "Command-line pipeline; no grading interface", AX("2602.18064", "v2"), "Sec. 5.5", "v2"),
     ]},
    {"group": "CT models (standalone)", "name": "NV-Reason-CT", "kind": "Standalone model, chest and abdominal CT, 2026; also used inside ThoraxCheck",
     "ref": "Myronenko et al., arXiv 2609.27511; huggingface.co/nvidia/NV-Reason-CT", "version": "arXiv v2 (25 Sep 2026); model card rev 386b93e (v0.1)",
     "cells": [
         cell(N, "Body and lung masks only place the crop", AX("2609.27511", "v2"), "Sec. 3.3-3.4", "v2"),
         cell(N, "The model decides; ontology rules label training data only", AX("2609.27511", "v2"), "Sec. 4.2; Sec. 5.1.1-5.1.2", "v2"),
         cell(P, "Trained to state uncertainty or evidentiary limits in free text; no schema state", AX("2609.27511", "v2"), "Abstract; Sec. 4.2", "v2"),
         cell(N, "Valid HU input required by README; no check described", "https://github.com/NVIDIA-Medtech/NV-Reason-CT", "README; paper Sec. 3.3, 6", "b4abd35"),
         cell(P, "Reasoning trace not tied to claims; faithfulness not shown", AX("2609.27511", "v2"), "Sec. 3.2, 6", "v2"),
         cell(N, "Tested on H100 / L40S; no memory figure (our own runs: 12.6 GB peak on a 24 GB GPU, not from the source)",
              "https://huggingface.co/nvidia/NV-Reason-CT", "model card: Hardware", "386b93e"),
         cell(P, "2 radiologists rated 10 cases on a 5-point scale via a web interface; storage not reported", AX("2609.27511", "v2"), "Sec. 5.5", "v2"),
     ]},
    {"group": "CT models (standalone)", "name": "EXACT", "kind": "Standalone model, chest CT, 2026",
     "ref": "Bai et al., arXiv 2604.24146; github.com/JasonW375/EXACT", "version": "arXiv v2 PDF (4 Nov 2026); repo 62bdbcc (2026-10-05); no licence found",
     "cells": [
         cell(N, "Organ masks restrict anomaly maps; scores by pooling, no measurements", "https://arxiv.org/pdf/2604.24146v2", "Results: anomaly localization; Methods Eq. 11, 14", "v2"),
         cell(N, "18 fixed labels with ROC-fitted thresholds per cohort; no versioned rules", "https://arxiv.org/pdf/2604.24146v2", "Methods pp. 34-35; Suppl. Table 3", "v2"),
         cell(N, "Binary or probability outputs only", "https://arxiv.org/pdf/2604.24146v2", "Methods", "v2"),
         cell(N, "Clipping, CLAHE, rescaling only", "https://arxiv.org/pdf/2604.24146v2", "Methods: preprocessing", "v2"),
         cell(P, "One organ-masked anomaly map per disease; report sentences not linked", "https://arxiv.org/pdf/2604.24146v2", "Fig. 4; README outputs", "v2; 62bdbcc"),
         cell(N, "Training hardware only; inference memory not reported", "https://arxiv.org/pdf/2604.24146v2", "Methods", "v2"),
         cell(N, "Command-line tools; no reader study", "https://github.com/JasonW375/EXACT", "README", "62bdbcc"),
     ]},
    {"group": "CT models (standalone)", "name": "HoloCT", "kind": "Standalone model, CT, 2026",
     "ref": "Xie et al., arXiv 2611.06471; github.com/alibaba-damo-academy/HoloCT", "version": "arXiv v1 (4 Nov 2026); repo 0d3b0a1 (2026-11-07)",
     "cells": [
         cell(N, "Predicts masks; no volume, size, HU or count output", AX("2611.06471", ""), "Sec. 4.2, 4.5", "v1"),
         cell(N, "Language model output; string matching used for evaluation", AX("2611.06471", ""), "Sec. 4.5", "v1"),
         cell(N, "Binary findings; uncertainty named as future work", AX("2611.06471", ""), "Sec. 4.5; Discussion", "v1"),
         cell(N, "Reorient, resample, crop, clip; automated QC named as future need", AX("2611.06471", ""), "Sec. 4.2; Discussion", "v1"),
         cell(D, "One predicted 3D mask per reported finding", AX("2611.06471", "") + " ; repo README", "Sec. 2.6; Fig. 6f; README outputs", "v1; 0d3b0a1"),
         cell(N, "Demo on one GPU in bf16; no memory figure", "https://huggingface.co/Alibaba-DAMO-Academy/HoloCT-8B", "model card: Requirements; Sec. 4.8", "card 2026-11"),
         cell(N, "Radiologists as comparators; no grading interface", AX("2611.06471", ""), "Sec. 4.6; Fig. 6c", "v1"),
     ]},
    {"group": "CT report-generation models", "name": "MedRegion-CT", "kind": "Report-generation model, CT, ECCV 2026",
     "ref": "Kyung et al., arXiv 2506.23102; github.com/babbu3682/MedRegion-CT", "version": "arXiv v3 (6 Jul 2026); repo b7cf18b",
     "cells": [
         cell(D, "Lesion volume, maximum diameter and location from TotalSegmentator pseudo-masks, given to the language model", AX("2506.23102", "v3"),
              "Sec. 3.3; Alg. 1; Suppl. Sec. 2; repo make_attr.py", "v3; b7cf18b"),
         cell(N, "Language model writes the findings", AX("2506.23102", "v3"), "Sec. 3.3", "v3"),
         cell(N, "Zero volume gives a 'no lesion' prompt; no indeterminate state", AX("2506.23102", "v3"), "Alg. 1", "v3"),
         cell(N, "Preprocessing only", AX("2506.23102", "v3"), "Sec. 4.1", "v3"),
         cell(N, "Sentences not linked to masks or measurements", AX("2506.23102", "v3"), "Results figures", "v3"),
         cell(N, "Training on A100 80 GB; inference hardware not stated", AX("2506.23102", "v3"), "Sec. 4.1; Suppl. Sec. 1", "v3"),
         cell(N, "One resident scored 50 cases; no interface described", AX("2506.23102", "v3"), "Reader study", "v3"),
     ]},
    {"group": "CT report-generation models", "name": "Reg2RG", "kind": "Report-generation model, CT, 2024-25",
     "ref": "Chen et al., arXiv 2411.15539; github.com/zhi-xuan-chen/Reg2RG", "version": "arXiv v2 (5 May 2025); repo 7d94b34",
     "cells": [
         cell(N, "Region masks encoded as learned features; no numeric quantities", AX("2411.15539", "v2"), "Method: mask encoder", "v2"),
         cell(N, "Region template; the language model generates", AX("2411.15539", "v2"), "Method", "v2"),
         cell(N, "No unresolved state described", AX("2411.15539", "v2"), "whole text", "v2"),
         cell(N, "No quality gating", AX("2411.15539", "v2"), "Experiments", "v2"),
         cell(P, "Each region-level report tied to its anatomical region mask, not to lesions", AX("2411.15539", "v2"), "Fig. 5; Method", "v2"),
         cell(N, "Trained on two 24 GB GPUs; single-GPU inference not stated", AX("2411.15539", "v2"), "Implementation details", "v2"),
         cell(N, "Automated metrics only", AX("2411.15539", "v2"), "Experiments", "v2"),
     ]},
    {"group": "Post-generation verification", "name": "EviGround-CT", "kind": "Verification model, chest CT, BIBM 2026",
     "ref": "Li et al., arXiv 2610.33038", "version": "arXiv v1 (15 Oct 2026); no code release",
     "cells": [
         cell(N, "Predicted evidence mask only; no size or volume output", AX("2610.33038", ""), "Sec. III; IV-E", "v1"),
         cell(N, "Keyword and negation extractor for 9 categories; no answer rules or version", AX("2610.33038", ""), "Sec. IV-C, IV-E; Table IV", "v1"),
         cell(N, "Supported / unsupported decided against reference masks at evaluation; unmapped sentences excluded", AX("2610.33038", ""), "Sec. IV-C, IV-E", "v1"),
         cell(N, "Clip, normalise, resize only", AX("2610.33038", ""), "Sec. IV-A", "v1"),
         cell(D, "Each positive report sentence gets its own predicted 3D evidence mask; inference needs only CT and report",
              AX("2610.33038", ""), "Sec. III; Eq. 1", "v1"),
         cell(N, "Training on two H100; inference memory not reported", AX("2610.33038", ""), "Sec. V-B", "v1"),
         cell(N, "Radiologist review named as future work", AX("2610.33038", ""), "Sec. VII", "v1"),
     ]},
    {"group": "Related approaches: chest X-ray", "name": "CXRAgent", "kind": "Complete system, chest X-ray, 2025-26",
     "ref": "Lou et al., arXiv 2510.21324; github.com/laojiahuo2003/CXRAgent", "version": "arXiv v2 (16 Jul 2026); repo 0514b48",
     "cells": [
         cell(N, "Vision-language tools and grounding boxes; no mask measurements", AX("2510.21324", "v2"), "Sec. III-A; repo medAgent/tools/grounding.py", "v2; 0514b48"),
         cell(N, "Director language model (GPT-4o or Qwen-VL-Max) decides", AX("2510.21324", "v2"), "Sec. III-C, IV-A", "v2"),
         cell(P, "High / medium / low confidence with refuting evidence; no distinct unresolved state", AX("2510.21324", "v2"), "Sec. III-A", "v2"),
         cell(N, "No exposure, rotation or inspiration checks", AX("2510.21324", "v2"), "whole text", "v2"),
         cell(P, "Each tool statement linked to supporting / refuting evidence; final report not", AX("2510.21324", "v2"), "Sec. III-A", "v2"),
         cell(N, "Two RTX A6000 and a cloud language-model API", AX("2510.21324", "v2"), "Sec. IV-A, IV-G; README", "v2; 0514b48"),
         cell(P, "Blinded radiologist comparison of 100 samples; no grading interface", AX("2510.21324", "v2"), "Sec. IV-F; Table XII", "v2"),
     ]},
    {"group": "Related approaches: chest X-ray", "name": "MedRAX", "kind": "Complete system, chest X-ray, 2025",
     "ref": "Fallahpour et al., arXiv 2502.02673; github.com/bowang-lab/MedRAX", "version": "arXiv v2 (29 May 2025); repo d243cc5 (2026-11-05)",
     "cells": [
         cell(P, "Repo code computes area, width, height and mean intensity from predicted masks; use in answers not reported",
              "https://github.com/bowang-lab/MedRAX", "medrax/tools/segmentation.py (_compute_organ_metrics); paper Sec. 3.2", "d243cc5; v2"),
         cell(N, "GPT-4o writes the final answer", AX("2502.02673", "v2"), "Sec. 3.1; Alg. 1; Sec. 5.1", "v2"),
         cell(N, "Paper notes no robust uncertainty quantification", AX("2502.02673", "v2"), "Sec. 6", "v2"),
         cell(N, "DICOM handling is format only", AX("2502.02673", "v2"), "Sec. 3.2", "v2"),
         cell(P, "Interface shows intermediate tool outputs; answers not linked per claim", AX("2502.02673", "v2"), "Sec. 3.4; Fig. 2", "v2"),
         cell(N, "One RTX 6000 (memory unclear) plus a cloud language-model API", AX("2502.02673", "v2"), "Sec. 5.1; README", "v2; d243cc5"),
         cell(P, "Chat interface with image upload; no structured grading", AX("2502.02673", "v2"), "Sec. 3.4", "v2"),
     ]},
]

SHORT = {D: "Documented", P: "Partial", N: "Not reported", U: "Unverified", I: "Implemented"}


TAGS = {  # short label shown in Partial cells (figure and Markdown table)
    ("RadAgent", "C5"): "trace only", ("MedScribe", "C5"): "claimed, not per claim", ("3DMedAgent", "C5"): "evidence per step",
    ("NV-Reason-CT", "C3"): "free-text uncertainty", ("NV-Reason-CT", "C5"): "trace only", ("NV-Reason-CT", "C7"): "rating study, no grading UI",
    ("EXACT", "C5"): "one map per disease", ("Reg2RG", "C5"): "region-level masks",
    ("CXRAgent", "C3"): "confidence levels", ("CXRAgent", "C5"): "per tool statement", ("CXRAgent", "C7"): "comparison, no UI",
    ("MedRAX", "C1"): "in repository code only", ("MedRAX", "C5"): "tool outputs shown", ("MedRAX", "C7"): "chat UI only",
}


def short_partial(name: str, cid: str) -> str:
    return TAGS[(name, cid)]


def markdown_table() -> str:
    head = "| System | " + " | ".join(f"{c[1]} ({c[0]})" for c in CRITERIA) + " |\n|---|" + "---|" * len(CRITERIA) + "\n"
    out, grp = [], None
    for s in SYSTEMS:
        if s["group"] != grp:
            grp = s["group"]
            out.append(f"| **{grp}** |" + " |" * len(CRITERIA))
        cells = []
        for (cid, _, _), c in zip(CRITERIA, s["cells"]):
            t = c["status"] if c["status"] != P else f"Partial: {short_partial(s['name'], cid)}"
            cells.append(t)
        out.append(f"| {s['name']}<br><sub>{s['kind']}</sub> | " + " | ".join(cells) + " |")
    return head + "\n".join(out) + "\n"


def source_matrix() -> str:
    md = [f"# Source matrix for the related-work comparison", "",
          f"Source review date: {REVIEW_DATE}. Generated by `scripts/make_comparison_figure.py`; edit the data there, not here.",
          "Statuses: Documented, Partial, Not reported (the examined sources do not describe it; not the same as absent), Unverified; "
          "Implemented for ThoraxCheck (validation is described separately in docs/RELATED_WORK.md).", "",
          "Criteria:", ""]
    md += [f"- **{c[0]} {c[1]}**: {c[2]}" for c in CRITERIA]
    for s in SYSTEMS:
        md += ["", f"## {s['name']}", "", f"{s['kind']}. Reference: {s['ref']}. Version examined: {s['version']}.", "",
               "| Criterion | Status | Explanation | Source | Location | Version |", "|---|---|---|---|---|---|"]
        for (cid, label, _), c in zip(CRITERIA, s["cells"]):
            md.append(f"| {cid} {label} | {c['status']} | {c['note']} | {c['src']} | {c['loc']} | {c['ver']} |")
    return "\n".join(md) + "\n"


SHADE = {D: ("#e7ebf0", "#1f2328"), I: ("#e7ebf0", "#1f2328"), P: ("#f4f5f7", "#1f2328"), N: ("#ffffff", "#6b7480"), U: ("#ffffff", "#6b7480")}


def svg() -> tuple[str, int, int]:
    c0, cw, rh, gh, top = 250, 172, 54, 30, 150
    n_groups = len({s["group"] for s in SYSTEMS})
    W = c0 + cw * len(CRITERIA) + 30
    H = top + rh * len(SYSTEMS) + gh * n_groups + 110
    p = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" font-family="Segoe UI, Helvetica, Arial, sans-serif">',
         f'<rect width="{W}" height="{H}" fill="#ffffff"/>',
         f'<text x="24" y="40" font-size="22" font-weight="700" fill="#1f2328">{html.escape(TITLE)}</text>',
         f'<text x="24" y="66" font-size="13" fill="#4f5a66">Sources reviewed {REVIEW_DATE}. Statuses follow the definitions in docs/RELATED_WORK.md; '
         'details and locations in docs/related_work/source_matrix.md.</text>']
    for i, (cid, label, _) in enumerate(CRITERIA):
        x = c0 + i * cw + cw / 2
        words, lines, cur = label.split(), [], ""
        for w_ in words:
            if len(cur) + len(w_) + 1 > 17:
                lines.append(cur)
                cur = w_
            else:
                cur = (cur + " " + w_).strip()
        lines.append(cur)
        for j, ln in enumerate(lines):
            p.append(f'<text x="{x}" y="{top - 46 + j * 16 - (len(lines) - 2) * 8}" font-size="13" font-weight="700" text-anchor="middle" fill="#1f2328">{html.escape(ln)}</text>')
        p.append(f'<text x="{x}" y="{top - 8}" font-size="11" text-anchor="middle" fill="#6b7480">{cid}</text>')
    y, grp = top, None
    for s in SYSTEMS:
        if s["group"] != grp:
            grp = s["group"]
            p.append(f'<rect x="16" y="{y}" width="{W - 32}" height="{gh}" fill="#f7f8fa"/>')
            p.append(f'<text x="24" y="{y + 20}" font-size="12" font-weight="700" letter-spacing="1" fill="#4f5a66">{html.escape(grp.upper())}</text>')
            y += gh
        p.append(f'<line x1="16" y1="{y}" x2="{W - 16}" y2="{y}" stroke="#e3e6ea"/>')
        p.append(f'<text x="24" y="{y + 23}" font-size="14" font-weight="700" fill="#1f2328">{html.escape(s["name"])}</text>')
        kind = s["kind"].split(";")[0]
        kind = kind if len(kind) <= 44 else kind[:42] + "..."
        p.append(f'<text x="24" y="{y + 41}" font-size="11" fill="#6b7480">{html.escape(kind)}</text>')
        for i, c in enumerate(s["cells"]):
            x = c0 + i * cw
            bg, fg = SHADE[c["status"]]
            p.append(f'<rect x="{x + 4}" y="{y + 6}" width="{cw - 8}" height="{rh - 12}" rx="6" fill="{bg}" stroke="#d5d9de"/>')
            p.append(f'<text x="{x + cw / 2}" y="{y + 25}" font-size="12" font-weight="600" text-anchor="middle" fill="{fg}">{SHORT[c["status"]]}</text>')
            if c["status"] == P:
                p.append(f'<text x="{x + cw / 2}" y="{y + 40}" font-size="10" text-anchor="middle" fill="#4f5a66">{html.escape(short_partial(s["name"], CRITERIA[i][0]))}</text>')
        y += rh
    p.append(f'<line x1="16" y1="{y}" x2="{W - 16}" y2="{y}" stroke="#e3e6ea"/>')
    p.append(f'<text x="24" y="{y + 34}" font-size="12" fill="#1f2328"><tspan font-weight="700">Implemented</tspan>: in our code and run records (not a validation). '
             '<tspan font-weight="700">Documented</tspan>: the source shows it. <tspan font-weight="700">Partial</tspan>: only the named part is shown. '
             '<tspan font-weight="700">Not reported</tspan>: not described in the examined sources.</text>')
    p.append(f'<text x="24" y="{y + 56}" font-size="12" font-weight="700" fill="#1f2328">Not reported does not mean absent. More features do not mean better or clinically superior.</text>')
    p.append("</svg>")
    return "\n".join(p), W, H


def main() -> int:
    out = ROOT / "docs" / "related_work"
    out.mkdir(parents=True, exist_ok=True)
    (out / "comparison_table.md").write_text(f"# {TITLE}\n\nSource review date: {REVIEW_DATE}. Generated by `scripts/make_comparison_figure.py`.\n"
                                             "Not reported does not mean absent.\n\n" + markdown_table(), encoding="utf-8")
    (out / "source_matrix.md").write_text(source_matrix(), encoding="utf-8")
    s, w, h = svg()
    sp = ROOT / "docs" / "images" / "related_work_comparison.svg"
    sp.write_text(s, encoding="utf-8")
    sys.path.insert(0, str(ROOT / "scripts"))
    from make_docs_figures import to_png
    to_png(sp, sp.with_suffix(".png"), w, h, port=9652)
    print(out, sp, w, h)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
