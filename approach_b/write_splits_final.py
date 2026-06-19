"""write_splits_final.py — Pin nnUNet fold 0 to the project's fixed train/val split.

nnUNet v2 will otherwise generate its own random 5-fold CV over all ``numTraining``
cases (here 284 = 247 train + 37 val), which would leak the project's validation
cases into training and break comparability with the Approach-B baseline.

This writes ``nnUNet_preprocessed/Dataset0XX_.../splits_final.json`` as a single
fold (fold 0) whose ``train`` list is the 247 project-train stems and ``val`` list
is the 37 project-val stems. Train only fold 0.

Usage
-----
python approach_b/write_splits_final.py --config configs/paths.yaml --variant fixedbox
python approach_b/write_splits_final.py --variant priorchan
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from configs import load_config

# Maps variant → (dataset_id config key, dataset_name config key).
VARIANT_KEYS = {
    "default":   ("nnunet_dataset_id",   "nnunet_dataset_name"),
    "fixedbox":  ("fixedbox_dataset_id", "fixedbox_dataset_name"),
    "priorchan": ("priorchan_dataset_id", "priorchan_dataset_name"),
}


def read_stems(txt_path: Path) -> list[str]:
    """Case identifiers (filename stem, no _0000) from a split txt file."""
    stems = []
    with open(txt_path) as f:
        for line in f.readlines()[1:]:  # skip header
            line = line.strip()
            if not line:
                continue
            img = line.split(",")[0]
            stems.append(Path(img).name.replace(".nii.gz", ""))
    return stems


def main() -> None:
    p = argparse.ArgumentParser(
        description="Write a single-fold splits_final.json (fold 0 = project train/val).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--config", default="configs/paths.yaml")
    p.add_argument("--variant", default="fixedbox", choices=list(VARIANT_KEYS))
    args = p.parse_args()

    cfg = load_config(args.config)
    ab = cfg["approach_b"]

    train_stems = read_stems(Path(cfg["data"]["train_txt"]))
    val_stems = read_stems(Path(cfg["data"]["val_txt"]))

    overlap = set(train_stems) & set(val_stems)
    if overlap:
        raise SystemExit(f"[ERROR] train/val overlap ({len(overlap)} cases): {sorted(overlap)[:5]}…")

    id_key, name_key = VARIANT_KEYS[args.variant]
    dataset_id = int(ab[id_key])
    dataset_name = ab[name_key]
    dataset_dir = Path(cfg["nnunet"]["preprocessed"]) / f"Dataset{dataset_id:03d}_{dataset_name}"
    if not dataset_dir.exists():
        raise SystemExit(
            f"[ERROR] Preprocessed dir not found: {dataset_dir}\n"
            f"        Run nnUNetv2_plan_and_preprocess -d {dataset_id} first."
        )

    # nnUNet expects a list of folds; we provide exactly one (fold 0).
    splits = [{"train": train_stems, "val": val_stems}]
    out = dataset_dir / "splits_final.json"
    out.write_text(json.dumps(splits, indent=4) + "\n")

    print(f"[INFO] Wrote {out}")
    print(f"       fold 0: train={len(train_stems)}  val={len(val_stems)}  "
          f"(total {len(train_stems) + len(val_stems)})")
    print("       Train with: bash approach_b/train_variant.sh "
          f"{args.variant} 0")


if __name__ == "__main__":
    main()
