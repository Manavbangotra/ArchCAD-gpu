#!/usr/bin/env python3
"""
Download the original FloorPlanCAD release (vector SVG, not the rasterised
mirrors) from the project's Google Drive links.

The README references this script but it was never published; the file ids
below come from https://floorplancad.github.io/.

Usage:
    python dataset/download_data.py --split test
    python dataset/download_data.py --split all --out dataset/FloorplanCAD
"""

import argparse
import os
import os.path as osp
import tarfile
import zipfile

# File ids published on the FloorPlanCAD project page.
DRIVE_FILES = {
    "train_1": "1HcyKt6qWeXog-tRfvEjdO3O3TN91PXGL",
    "train_2": "1kSS7OB_EEu7VJzb0W8DK9_nu1EvshioV",
    "test": "1jxpYgxnLUbXEzMOsjaMPQFSuvmvHimiZ",
}


def archive_kind(path: str) -> str:
    """Identify the archive by magic bytes rather than by filename.

    Despite being shared as "zip" downloads, these files are actually
    xz-compressed tars (magic fd 37 7a 58 5a). Trusting the extension makes
    extraction fail on every split.
    """
    with open(path, "rb") as f:
        head = f.read(6)
    if head.startswith(b"\xfd7zXZ\x00"):
        return "tar.xz"
    if head.startswith(b"PK\x03\x04"):
        return "zip"
    if head[:2] == b"\x1f\x8b":
        return "tar.gz"
    return "unknown"


def verify(path: str) -> bool:
    """Check the archive can actually be read end to end.

    A download can finish at exactly the right size and still be corrupt — that
    happened here once with a 1.84GB file whose xz block was damaged. Verifying
    first turns a confusing extraction failure into a clean retry.
    """
    kind = archive_kind(path)
    try:
        if kind == "zip":
            with zipfile.ZipFile(path) as zf:
                return zf.testzip() is None
        if kind in ("tar.xz", "tar.gz"):
            mode = "r:xz" if kind == "tar.xz" else "r:gz"
            with tarfile.open(path, mode) as tf:
                for _ in tf:  # walking every member forces a full decode
                    pass
            return True
    except Exception as exc:
        print(f"[verify] {osp.basename(path)} is damaged: {exc}")
        return False
    print(f"[verify] unrecognised archive format for {path}")
    return False


def extract_archive(path: str, target: str) -> None:
    kind = archive_kind(path)
    os.makedirs(target, exist_ok=True)
    print(f"[extract] {osp.basename(path)} ({kind}) -> {target}")

    if kind == "zip":
        with zipfile.ZipFile(path) as zf:
            zf.extractall(target)
    elif kind in ("tar.xz", "tar.gz"):
        mode = "r:xz" if kind == "tar.xz" else "r:gz"
        with tarfile.open(path, mode) as tf:
            tf.extractall(target)
    else:
        raise SystemExit(f"Cannot extract {path}: unrecognised format")


def download(split: str, out_dir: str, extract: bool = True, retries: int = 1) -> str:
    try:
        import gdown
    except ImportError:
        raise SystemExit("gdown is required: pip install gdown")

    os.makedirs(out_dir, exist_ok=True)
    # Named .archive because the true format is only known after downloading.
    path = osp.join(out_dir, f"{split}.archive")

    for attempt in range(retries + 1):
        if not osp.isfile(path):
            print(f"[download] {split} -> {path}")
            gdown.download(id=DRIVE_FILES[split], output=path, quiet=False)
        else:
            print(f"[skip] {path} already exists")

        print(f"[verify] checking {osp.basename(path)} ...")
        if verify(path):
            print(f"[verify] {osp.basename(path)} OK ({archive_kind(path)})")
            break

        if attempt < retries:
            print("[verify] re-downloading once ...")
            os.remove(path)
        else:
            raise SystemExit(
                f"{path} is still corrupt after {retries + 1} attempts. "
                "Delete it and try again later, or download it manually from "
                "https://floorplancad.github.io/"
            )

    if extract:
        extract_archive(path, osp.join(out_dir, split))

    return path


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="test",
                    choices=list(DRIVE_FILES) + ["all"],
                    help="which archive to fetch (default: test)")
    ap.add_argument("--out", default="dataset/FloorplanCAD",
                    help="destination directory")
    ap.add_argument("--no-extract", action="store_true",
                    help="download the zip but do not unpack it")
    args = ap.parse_args()

    splits = list(DRIVE_FILES) if args.split == "all" else [args.split]
    for split in splits:
        download(split, args.out, extract=not args.no_extract)


if __name__ == "__main__":
    main()
