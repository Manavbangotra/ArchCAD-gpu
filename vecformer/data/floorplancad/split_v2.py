"""
Arrange preprocessed FloorPlanCAD-V2 line JSONs into train/val/test directories.

The V2 release ships train_1/, train_2/ and test/ (15,663 drawings); the
preprocessor mirrors that layout. Two protocols:

  textcad   TextCAD (arXiv 2607.12678, App. C.1): all drawings re-split about
            6:3:1 into train/val/test = 9533 : 4597 : 1533. The paper does not
            publish its file lists, so this is a seeded random split with those
            counts (scaled if the drawing count differs).
  official  the release's own train (train_1 + train_2) and test; val is a seeded
            8% of train, the share VecFormer's V1 split holds out.

Files are hard-linked (copied where links are unsupported), so both protocols
cost almost no disk.

    python data/floorplancad/split_v2.py --input_dir datasets/FloorPlanCAD-V2-lines \
        --output_dir datasets/FloorPlanCAD-V2-textcad --protocol textcad
"""
import argparse
import os
import os.path as osp
import random
import shutil

TEXTCAD_COUNTS = (9533, 4597, 1533)


def list_jsons(root, sub):
    d = osp.join(root, sub)
    if not osp.isdir(d):
        return []
    out = []
    for dirpath, _, files in os.walk(d):
        out += [osp.relpath(osp.join(dirpath, f), root) for f in files if f.endswith(".json")]
    return sorted(out)


def plan(input_dir, protocol, seed=0):
    """-> {split: [relative paths]}"""
    rng = random.Random(seed)
    if protocol == "textcad":
        files = list_jsons(input_dir, "train_1") + list_jsons(input_dir, "train_2") + list_jsons(input_dir, "test")
        rng.shuffle(files)
        total = sum(TEXTCAD_COUNTS)
        n_train = round(len(files) * TEXTCAD_COUNTS[0] / total)
        n_val = round(len(files) * TEXTCAD_COUNTS[1] / total)
        return dict(train=sorted(files[:n_train]), val=sorted(files[n_train:n_train + n_val]),
                    test=sorted(files[n_train + n_val:]))
    if protocol == "official":
        train = list_jsons(input_dir, "train_1") + list_jsons(input_dir, "train_2")
        rng.shuffle(train)
        n_val = round(len(train) * 0.08)
        return dict(train=sorted(train[n_val:]), val=sorted(train[:n_val]), test=list_jsons(input_dir, "test"))
    raise ValueError(protocol)


def materialise(input_dir, output_dir, splits):
    for split, files in splits.items():
        for rel in files:
            # flatten train_1/x.json -> train/train_1_x.json: names repeat across parts
            dst = osp.join(output_dir, split, rel.replace(os.sep, "_").replace("/", "_"))
            os.makedirs(osp.dirname(dst), exist_ok=True)
            if osp.exists(dst):
                continue
            try:
                os.link(osp.join(input_dir, rel), dst)
            except OSError:
                shutil.copy2(osp.join(input_dir, rel), dst)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input_dir", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--protocol", choices=["textcad", "official"], default="textcad")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    splits = plan(a.input_dir, a.protocol, a.seed)
    materialise(a.input_dir, a.output_dir, splits)
    print({k: len(v) for k, v in splits.items()})


if __name__ == "__main__":
    main()
