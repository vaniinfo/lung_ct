"""Documentation figures (core env): the ThoraxCheck pipeline diagram as SVG, and PNG copies rendered in a local browser.

Writes docs/images/thoraxcheck_pipeline.svg / .png. The comparison figure is written by scripts/make_comparison_figure.py.
PNG rendering uses Edge or Chrome in headless mode on the local SVG file (nothing is sent anywhere).

Usage: .venv/Scripts/python.exe scripts/make_docs_figures.py
"""
from __future__ import annotations

import html
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs" / "images"
FONT = "Segoe UI, Helvetica, Arial, sans-serif"
BROWSERS = [r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe", r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
            r"C:\Program Files\Google\Chrome\Application\chrome.exe"]


def text_block(x, y, lines, size=15, lh=21, weight_first=True, colour="#1f2328"):
    out = []
    for i, ln in enumerate(lines):
        bold = ln.startswith("**")
        t = html.escape(ln.strip("*"))
        w = "700" if (bold or (weight_first and i == 0)) else "400"
        sz = size + 3 if (weight_first and i == 0) else size
        out.append(f'<text x="{x}" y="{y + i * lh}" font-size="{sz}" font-weight="{w}" fill="{colour}">{t}</text>')
    return "\n".join(out)


def box(x, y, w, h, fill, stroke, lines, colour="#1f2328", size=15, lh=22):
    return (f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="14" fill="{fill}" stroke="{stroke}" stroke-width="1.5"/>\n'
            + text_block(x + 20, y + 34, lines, size=size, lh=lh, colour=colour))


def arrow(x1, y1, x2, y2, dashed=False):
    d = ' stroke-dasharray="6 6"' if dashed else ""
    return f'<path d="M{x1},{y1} L{x2},{y2}" stroke="#5b6b82" stroke-width="2" fill="none" marker-end="url(#ah)"{d}/>'


def pipeline_svg() -> str:
    W, H = 1500, 660
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" font-family="{FONT}">',
             '<defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="8" markerHeight="8" orient="auto">'
             '<path d="M0,0 L10,5 L0,10 z" fill="#5b6b82"/></marker></defs>',
             f'<rect width="{W}" height="{H}" fill="#f7f9fc"/>',
             '<text x="60" y="64" font-size="32" font-weight="700" fill="#17375e">ThoraxCheck: from scan to clinician review</text>',
             '<text x="60" y="96" font-size="17" fill="#5b6b82">Evidence-based framework for chest CT assessment · simplified pipeline · checklist v3_12 · viewer v7</text>']
    for x, t in ((60, "01 INPUT"), (420, "02 EVIDENCE"), (830, "03 DECISION"), (1180, "04 REVIEW")):
        parts.append(f'<text x="{x}" y="150" font-size="14" font-weight="700" letter-spacing="2" fill="#5b6b82">{t}</text>')
    parts.append(box(60, 175, 280, 340, "#e8f0fb", "#b9cdf5", ["CT scan + quality checks", "**Image values and geometry", "HU values validated",
                                                                 "Slice spacing checked", "**Orientation from anatomy", "Heart left, liver right",
                                                                 "Heart in front, trachea above", "**Only verified scans go on", "Failures are listed"]))
    parts.append(box(420, 175, 340, 125, "#e8f6ee", "#a9d8b8", ["TotalSegmentator + measurements", "Organ, fluid and nodule masks",
                                                                  "Volumes, HU ranges, lobe locations"], colour="#14532d"))
    parts.append(box(420, 320, 340, 95, "#eef0fb", "#c3c9f0", ["COLIPRI classifier", "Supporting score per condition"], colour="#1e3a8a"))
    parts.append(box(420, 435, 340, 95, "#f3eefc", "#cdbdf0", ["NV-Reason-CT reader", "Report and yes / no answers"], colour="#4c1d95"))
    parts.append(box(830, 235, 290, 230, "#17375e", "#17375e", ["Checklist rules", "Combine the evidence", "Apply quality gates",
                                                                  "Answer + its basis", "\"Cannot tell\" when unresolved"], colour="#ffffff"))
    parts.append(box(1180, 235, 270, 230, "#fff3df", "#e6cf8a", ["Review page", "CT with optional outlines", "Answers and reports",
                                                                   "**Clinician grades", "Saved as CSV"], colour="#78350f"))
    parts.append(box(760, 565, 290, 58, "#ffffff", "#d5d9de", ["NV-Segment-CT: display only"], colour="#4f5a66", size=15))
    parts += [arrow(340, 345, 375, 345), f'<path d="M375,237 L375,482" stroke="#5b6b82" stroke-width="2" fill="none"/>',
              arrow(375, 237, 416, 237), arrow(375, 367, 416, 367), arrow(375, 482, 416, 482),
              f'<path d="M760,237 L790,237 L790,330" stroke="#5b6b82" stroke-width="2" fill="none"/>',
              arrow(760, 367, 826, 350), f'<path d="M760,482 L790,482 L790,370" stroke="#5b6b82" stroke-width="2" fill="none"/>',
              arrow(790, 330, 826, 330), arrow(790, 370, 826, 370),
              arrow(1120, 350, 1176, 350), '<path d="M1050,594 L1315,594 L1315,470" stroke="#5b6b82" stroke-width="2" fill="none" stroke-dasharray="6 6" marker-end="url(#ah)"/>']
    parts.append("</svg>")
    return "\n".join(parts)


def to_png(svg: Path, png: Path, w: int, h: int, port: int = 9651) -> None:
    """Render the SVG in a local headless browser (DevTools protocol) and save a PNG of exactly w x h pixels."""
    import base64
    sys.path.insert(0, str(ROOT / "scripts"))
    from test_canvas_viewer import Browser
    b = Browser("edge", "sw", svg.resolve().as_uri(), port)
    try:
        b.c.call("Emulation.setDeviceMetricsOverride", width=w, height=h, deviceScaleFactor=1, mobile=False)
        import time
        time.sleep(1.0)
        data = b.c.call("Page.captureScreenshot", format="png", clip={"x": 0, "y": 0, "width": w, "height": h, "scale": 1})["data"]
        png.write_bytes(base64.b64decode(data))
    finally:
        b.close()


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    svg = OUT / "thoraxcheck_pipeline.svg"
    svg.write_text(pipeline_svg(), encoding="utf-8")
    to_png(svg, svg.with_suffix(".png"), 1500, 660)
    print(svg, svg.with_suffix(".png"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
