"""Download one public file over several HTTP range requests, resumable, then check its SHA-256.

Used by setup_nvreason.bat for the NV-Reason-CT weights: without a token one connection to the
model hub is slow, several in parallel are not. Nothing is uploaded; no token is used.

Usage: python scripts/fetch_public_file.py <url> <output file> <size in bytes> <sha256>
"""
import hashlib
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

url, out, size, sha = sys.argv[1], Path(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
parts = out.parent / "parts"
parts.mkdir(parents=True, exist_ok=True)
CH = 64 * 2**20
ranges = [(i, a, min(a + CH, size) - 1) for i, a in enumerate(range(0, size, CH))]


def get(job):
    i, a, b = job
    p = parts / f"{i:04d}.bin"
    want = b - a + 1
    for attempt in range(30):
        have = p.stat().st_size if p.exists() else 0
        if have == want:
            return
        if have > want:
            p.unlink(); have = 0
        try:
            req = urllib.request.Request(url, headers={"Range": f"bytes={a + have}-{b}", "User-Agent": "python-urllib"})
            with urllib.request.urlopen(req, timeout=60) as r, open(p, "ab") as f:
                while True:
                    buf = r.read(1 << 20)
                    if not buf:
                        break
                    f.write(buf)
        except Exception as e:  # noqa: BLE001
            time.sleep(3)
    raise RuntimeError(f"part {i} failed")


t0 = time.time()
with ThreadPoolExecutor(32) as ex:
    list(ex.map(get, ranges))
h = hashlib.sha256()
with open(out, "wb") as f:
    for i, a, b in ranges:
        d = (parts / f"{i:04d}.bin").read_bytes()
        h.update(d); f.write(d)
ok = h.hexdigest() == sha and out.stat().st_size == size
print("SHA_OK" if ok else "SHA_MISMATCH", h.hexdigest(), out.stat().st_size, round(time.time() - t0), "s")
if ok:
    for p in parts.glob("*.bin"):
        p.unlink()
    parts.rmdir()
sys.exit(0 if ok else 1)
