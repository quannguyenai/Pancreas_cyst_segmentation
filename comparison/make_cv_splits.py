"""make_cv_splits.py — Generate per-fold train/val split txts for the comparison
baselines, reusing the EXACT 5-fold partition the nnU-Net approaches use.

The canonical folds live in nnUNet_preprocessed/<dataset>/splits_final.json as
lists of case IDs (e.g. "AHN09"). We map each ID back to its (image, mask) pair
via data/all_train.txt and write comparison/splits/fold{k}_{train,val}.txt in the
same "image_path,mask_path" CSV format the Cyst dataset loader expects. This makes
the baselines directly comparable to Approach A (identical folds, no test leakage).

Idempotent: re-running overwrites the split files.

Usage:
  python comparison/make_cv_splits.py --config configs/paths.yaml
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from configs import load_config


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/paths.yaml")
    ap.add_argument(
        "--splits-json", default=None,
        help="Path to nnU-Net splits_final.json defining the canonical folds. "
             "Default: Dataset001_PancreasCyst_3DA preprocessed splits.",
    )
    ap.add_argument("--out-dir", default=None,
                    help="Where to write fold txts (default: comparison/splits).")
    args = ap.parse_args()

    cfg = load_config(args.config)
    root = Path(cfg["data"]["all_train_txt"]).parent.parent  # repo root

    all_train_txt = Path(cfg["data"]["all_train_txt"])
    splits_json = Path(args.splits_json) if args.splits_json else (
        Path(cfg["nnunet"]["preprocessed"])
        / "Dataset001_PancreasCyst_3DA" / "splits_final.json"
    )
    out_dir = Path(args.out_dir) if args.out_dir else (root / "comparison" / "splits")
    out_dir.mkdir(parents=True, exist_ok=True)

    # stem -> "image_path,mask_path" line from the 284-case train pool
    pair_line: dict[str, str] = {}
    for line in all_train_txt.read_text().splitlines()[1:]:
        if not line.strip():
            continue
        img = line.split(",")[0]
        stem = Path(img).name.replace(".nii.gz", "")
        pair_line[stem] = line.strip()
    print(f"[make_cv_splits] train pool: {len(pair_line)} pairs from {all_train_txt.name}")

    folds = json.loads(splits_json.read_text())
    print(f"[make_cv_splits] {len(folds)} folds from {splits_json}")

    header = "image_path,mask_path"
    for k, fold in enumerate(folds):
        for which in ("train", "val"):
            ids = fold[which]
            missing = [i for i in ids if i not in pair_line]
            if missing:
                raise SystemExit(
                    f"fold{k}/{which}: {len(missing)} ids not in {all_train_txt.name}: "
                    f"{missing[:5]}"
                )
            lines = [header] + [pair_line[i] for i in ids]
            out = out_dir / f"fold{k}_{which}.txt"
            out.write_text("\n".join(lines) + "\n")
        print(f"  fold{k}: train={len(fold['train'])} val={len(fold['val'])} -> {out_dir}")

    print(f"[make_cv_splits] done. Test split stays {cfg['data']['test_txt']} (74 cases).")


if __name__ == "__main__":
    main()
