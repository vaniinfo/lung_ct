"""Download the CT-CLIP checkpoint(s) from the gated CT-RATE repository on Hugging Face (vlm env). Nothing is uploaded.

Only model weights are downloaded. No CT image, mask or report leaves this machine.
The token is read from the HF_TOKEN environment variable only; it is never printed or written to a file.
Licence of the weights: CC BY-NC-SA 4.0 (research use, no redistribution).

Files (sizes from the repository listing, 2026-11-23):
  base       models/CT-CLIP-Related/CT-CLIP_v2.pt        1.77 GB   zero-shot model (default)
  vocabfine  models/CT-CLIP-Related/CT_VocabFine_v2.pt   1.77 GB   fine-tuned with the 18 label prompts
  classfine  models/CT-CLIP-Related/CT_LiPro_v2.pt       1.77 GB   fine-tuned with a linear classification head
Written to models/ct_clip/ with a provenance file (source, size, SHA-256, time).

Usage: environments/vlm/Scripts/python.exe scripts/ctclip_fetch.py [--which base vocabfine classfine]
"""
import argparse
import hashlib
import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REPO = "ibrahimhamamci/CT-RATE"
FILES = {"base": "models/CT-CLIP-Related/CT-CLIP_v2.pt", "vocabfine": "models/CT-CLIP-Related/CT_VocabFine_v2.pt",
         "classfine": "models/CT-CLIP-Related/CT_LiPro_v2.pt"}
OUT = ROOT / "models" / "ct_clip"
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--which", nargs="+", choices=list(FILES), default=["base"])
    args = ap.parse_args()
    if not os.environ.get("HF_TOKEN"):
        print("HF_TOKEN is not set. The CT-RATE repository is gated: accept its terms on Hugging Face and set HF_TOKEN in this shell.")
        return 2
    free_gb = shutil.disk_usage(ROOT).free / 2 ** 30
    need_gb = 1.8 * len(args.which) * 2          # file + cache copy while downloading
    print(f"free on this drive: {free_gb:.0f} GB; needed at peak: about {need_gb:.1f} GB")
    if free_gb < need_gb + 5:
        print("not enough free space")
        return 3
    from huggingface_hub import hf_hub_download
    OUT.mkdir(parents=True, exist_ok=True)
    prov_path = OUT / "PROVENANCE.json"
    prov = json.loads(prov_path.read_text(encoding="utf-8")) if prov_path.exists() else {"repo": f"https://huggingface.co/datasets/{REPO}", "licence": "CC BY-NC-SA 4.0", "files": {}}
    for k in args.which:
        dst = OUT / Path(FILES[k]).name
        if not dst.exists():
            src = Path(hf_hub_download(REPO, FILES[k], repo_type="dataset", cache_dir=str(OUT / "_hf_cache")))
            shutil.copyfile(src, dst)
        prov["files"][k] = {"repo_path": FILES[k], "local": str(dst.relative_to(ROOT)).replace("\\", "/"), "bytes": dst.stat().st_size, "sha256": sha256(dst),
                            "recorded_utc": datetime.now(timezone.utc).isoformat()}
        print(k, dst.name, dst.stat().st_size, "bytes")
    shutil.rmtree(OUT / "_hf_cache", ignore_errors=True)      # the copy above is the kept file
    prov_path.write_text(json.dumps(prov, indent=1), encoding="utf-8")
    print("done ->", OUT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
