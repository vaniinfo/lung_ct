"""Download COVID-19 CT Lung and Infection Segmentation dataset from Zenodo.

Source: https://zenodo.org/records/3757476
License: CC BY 4.0 (per Zenodo record 3757476, checked 2026-11-13)
Citation: https://github.com/JunMa11/COVID-19-CT-Seg-Benchmark
"""
import hashlib
import sys
import urllib.request
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw" / "covid_ct"

# Files from Zenodo record 3757476
FILES = {
    "COVID-19-CT-Seg_20cases.zip": {
        "url": "https://zenodo.org/records/3757476/files/COVID-19-CT-Seg_20cases.zip",
        "description": "20 CT scans (~1.1GB)",
        "md5": "873617e1fdcbe92f8aa0ce83a4798a1f",
    },
    "Lung_Mask.zip": {
        "url": "https://zenodo.org/records/3757476/files/Lung_Mask.zip",
        "description": "Left/right lung masks",
        "md5": "972b3d9b4c7b64d2c518b936bd98cb47",
    },
    "Infection_Mask.zip": {
        "url": "https://zenodo.org/records/3757476/files/Infection_Mask.zip",
        "description": "Infection segmentation masks",
        "md5": "98fa9f5401fa8409dccd1681a36febcf",
    },
}


def hash_file(path: Path, algo: str = "sha256") -> str:
    h = hashlib.new(algo)
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def safe_extract(archive: Path, extract_dir: Path) -> None:
    """Extract a zip, rejecting entries that would escape extract_dir."""
    import zipfile
    root = extract_dir.resolve()
    with zipfile.ZipFile(archive, "r") as zf:
        for member in zf.infolist():
            target = (root / member.filename).resolve()
            if root != target and root not in target.parents:
                raise RuntimeError(f"Unsafe path in {archive.name}: {member.filename}")
        zf.extractall(root)


def download_with_progress(url: str, dest: Path) -> None:
    """Download a file with progress reporting."""
    print(f"  Downloading: {url}")
    print(f"  Destination: {dest}")

    def reporthook(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            pct = min(100, downloaded * 100 // total_size)
            mb = downloaded / (1024 * 1024)
            total_mb = total_size / (1024 * 1024)
            print(f"\r  Progress: {pct}% ({mb:.1f}/{total_mb:.1f} MB)", end="", flush=True)

    part = dest.with_name(dest.name + ".part")
    urllib.request.urlretrieve(url, part, reporthook)
    print()  # newline after progress
    part.replace(dest)  # only a complete download gets the final name


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    checksums = {}
    for filename, info in FILES.items():
        dest = DATA_DIR / filename
        if dest.exists():
            print(f"[SKIP] {filename} already exists ({dest.stat().st_size / 1024 / 1024:.1f} MB)")
        else:
            print(f"[DOWNLOAD] {info['description']}")
            download_with_progress(info["url"], dest)
            print(f"  Size: {dest.stat().st_size / 1024 / 1024:.1f} MB")

        md5 = hash_file(dest, "md5")
        if md5 != info["md5"]:
            print(f"[FAIL] {filename}: MD5 {md5} != Zenodo {info['md5']}. "
                  f"Delete the file and re-run.", file=sys.stderr)
            return 1
        print(f"  MD5 matches Zenodo record: {md5}")
        checksums[filename] = hash_file(dest, "sha256")
        print(f"  SHA-256: {checksums[filename]}")

    # Save checksums
    checksum_file = DATA_DIR / "checksums_sha256.txt"
    with open(checksum_file, "w") as f:
        for name, sha in checksums.items():
            f.write(f"{sha}  {name}\n")
    print(f"\nChecksums saved to {checksum_file}")

    # Extract archives
    for filename in FILES:
        archive = DATA_DIR / filename
        extract_dir = DATA_DIR / filename.replace(".zip", "")
        if extract_dir.exists():
            print(f"[SKIP] Already extracted: {extract_dir}")
        else:
            print(f"[EXTRACT] {filename} -> {extract_dir}")
            safe_extract(archive, extract_dir)
            print(f"  Extracted {len(list(extract_dir.rglob('*')))} items")

    print("\nDone. COVID-19 CT dataset ready.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
