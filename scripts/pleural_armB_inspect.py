"""Read-only inspection of the alternative pleural-effusion model archive on Zenodo (core env). Downloads no weights.

Zenodo record 21918112, Task307_pleuralfinal.zip (nnU-Net v1 format, CC BY 4.0). Read by HTTP byte range:
  - the archive's file list (names, sizes, CRC-32)
  - the small text / json files
  - plans.pkl, parsed with a RESTRICTED unpickler that only builds plain containers and numpy arrays; no other code
    in the pickle can run. Checkpoint files (*.model, *.model.pkl) are not read.
Writes <out>/armB_archive_listing.json and <out>/armB_plans_summary.json.

Usage: python -I scripts/pleural_armB_inspect.py --out runs/E17_pleural_model_comparison_prep_20261123
"""
from __future__ import annotations

import argparse
import io
import json
import pickle
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from ts_effusion_training_labels import RangeFile  # noqa: E402

URL = "https://zenodo.org/api/records/21918112/files/Task307_pleuralfinal.zip/content"
ALLOWED = {("collections", "OrderedDict"), ("numpy", "ndarray"), ("numpy", "dtype"), ("numpy.core.multiarray", "_reconstruct"), ("numpy.core.multiarray", "scalar"),
           ("numpy._core.multiarray", "_reconstruct"), ("numpy._core.multiarray", "scalar"), ("builtins", "set"), ("builtins", "frozenset"), ("builtins", "slice"),
           ("builtins", "range"), ("builtins", "tuple"), ("builtins", "list"), ("builtins", "dict"), ("copyreg", "_reconstructor"), ("builtins", "object")}


class Restricted(pickle.Unpickler):
    refused: list = []

    def find_class(self, module, name):
        if (module, name) in ALLOWED:
            return super().find_class(module, name)
        Restricted.refused.append(f"{module}.{name}")

        class Placeholder:      # inert stand-in: stores nothing, runs nothing
            def __init__(self, *a, **k): pass
            def __setstate__(self, s): pass
            def __repr__(self): return f"<refused {module}.{name}>"
        return Placeholder


def plain(o, depth=0):
    import numpy as np
    if depth > 6:
        return "..."
    if isinstance(o, dict):
        return {str(k): plain(v, depth + 1) for k, v in list(o.items())[:60]}
    if isinstance(o, (list, tuple, set, frozenset)):
        return [plain(v, depth + 1) for v in list(o)[:40]]
    if isinstance(o, np.ndarray):
        return o.tolist() if o.size <= 64 else f"array{o.shape}"
    if isinstance(o, (np.generic,)):
        return o.item()
    if isinstance(o, (str, int, float, bool)) or o is None:
        return o
    return repr(o)[:120]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    rf = RangeFile(URL)
    rf.seek(max(0, rf.size - (1 << 22)))
    rf.read(rf.size - rf.pos)
    zf = zipfile.ZipFile(rf)
    infos = zf.infolist()
    listing = [{"name": i.filename, "bytes": i.file_size, "bytes_compressed": i.compress_size, "crc32": f"{i.CRC:08x}", "date": "%04d-%02d-%02d" % i.date_time[:3]} for i in infos]
    summ = {"url": URL, "archive_bytes": rf.size, "entries": len(infos), "bytes_unpacked_total": sum(i.file_size for i in infos), "files": listing}
    (out / "armB_archive_listing.json").write_text(json.dumps(summ, indent=1), encoding="utf-8")
    for x in listing:
        print(x["bytes"], x["date"], x["name"])
    plans = {}
    for i in infos:
        n = i.filename.lower()
        if i.file_size and i.file_size < 3_000_000 and (n.endswith((".json", ".txt", ".md")) or n.endswith("plans.pkl")):
            rf.block = max(1 << 16, i.compress_size + 8192)
            data = zf.read(i)
            if n.endswith("plans.pkl"):
                Restricted.refused = []
                p = Restricted(io.BytesIO(data)).load()
                keep = {k: plain(v) for k, v in p.items() if k in ("num_stages", "num_modalities", "modalities", "normalization_schemes", "dataset_properties", "all_classes",
                                                                    "num_classes", "base_num_features", "transpose_forward", "transpose_backward", "data_identifier",
                                                                    "plans_per_stage", "preprocessor_name", "use_mask_for_norm", "conv_per_stage", "min_region_size_per_class",
                                                                    "keep_only_largest_region", "min_size_per_class", "original_spacings", "original_sizes")}
                if isinstance(keep.get("original_spacings"), list):
                    keep["n_training_cases_in_plans"] = len(p.get("original_spacings") or [])
                    keep.pop("original_spacings", None)
                    keep.pop("original_sizes", None)
                dp = keep.get("dataset_properties")
                if isinstance(dp, dict):
                    keep["dataset_properties"] = {k: v for k, v in dp.items() if k in ("intensityproperties", "all_classes", "modalities", "size_reductions") and k != "size_reductions"}
                plans[i.filename] = {"keys": sorted(map(str, p.keys())), "refused_classes": sorted(set(Restricted.refused)), "summary": keep}
            else:
                plans[i.filename] = data.decode("utf-8", "replace")[:6000]
    (out / "armB_plans_summary.json").write_text(json.dumps(plans, indent=1, default=str), encoding="utf-8")
    print(json.dumps(plans, indent=1, default=str)[:6000])
    print("bytes read:", rf.fetched)
    return 0


if __name__ == "__main__":
    sys.exit(main())
