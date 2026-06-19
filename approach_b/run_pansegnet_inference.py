"""Run PaNSegNet inference using nnUNet v1-style pretrained weights.

The official PaNSegNet release is trained under nnUNet v1 with the
``nnTransUNetTrainerV2`` trainer. This script locates the downloaded model
folder, prepares an nnUNet-style input directory (``*_0000.nii.gz``), and
invokes ``nnUNet_predict`` to write pancreas masks aligned to the input CTs.
"""

from __future__ import annotations

import argparse
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from configs import load_config


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run PaNSegNet inference via nnUNet v1.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--config", default="configs/paths.yaml")
    p.add_argument("--input-dir", default=None)
    p.add_argument("--output-dir", default=None)
    p.add_argument("--model-root", default=None)
    p.add_argument("--task-id", type=int, default=None)
    return p.parse_args()


def discover_predict_cmd(repo_root: Path) -> str:
    env_override = os.environ.get("PANSEGNET_NNUNET_PREDICT")
    if env_override:
        return env_override

    local_cmd = repo_root / ".venv" / "bin" / "nnUNet_predict"
    if local_cmd.exists():
        return str(local_cmd)

    approach_d_cmd = repo_root / "approach_d" / ".venv" / "bin" / "nnUNet_predict"
    if approach_d_cmd.exists():
        return str(approach_d_cmd)

    system_cmd = shutil.which("nnUNet_predict")
    if system_cmd:
        return system_cmd

    raise RuntimeError(
        "nnUNet_predict not found. Install nnUNet v1 or set PANSEGNET_NNUNET_PREDICT."
    )


def find_trainer_dir(
    model_root: Path,
    model_config: str,
    task_id: int,
    trainer: str,
    plans: str,
) -> Path:
    task_prefix = f"Task{task_id:03d}_"
    candidate_name = f"{trainer}__{plans}"
    candidates: list[Path] = []

    for path in model_root.rglob(candidate_name):
        if not path.is_dir():
            continue
        parent = path.parent
        if not parent.name.startswith(task_prefix):
            continue

        # Some nnUNet v1 exports are stored as:
        #   .../<model_config>/TaskXXX_Name/<trainer__plans>
        # while some Drive bundles are flattened as:
        #   .../TaskXXX_Name/<trainer__plans>
        if parent.parent.name == model_config or parent.parent == model_root:
            candidates.append(path)

    if not candidates:
        raise FileNotFoundError(
            f"Could not find {candidate_name!r} for task {task_id} under {model_root}"
        )
    if len(candidates) > 1:
        candidates.sort()
    return candidates[0]


def ensure_results_folder(trainer_dir: Path) -> tuple[Path, str, tempfile.TemporaryDirectory[str] | None]:
    task_dir = trainer_dir.parent
    maybe_model_dir = task_dir.parent

    if maybe_model_dir.name in {"2d", "3d_fullres", "3d_lowres", "3d_cascade_fullres"}:
        model_dir = maybe_model_dir
        if model_dir.parent.name == "nnUNet":
            return model_dir.parent.parent, task_dir.name, None
        temp_root = tempfile.TemporaryDirectory()
        temp_path = Path(temp_root.name)
        (temp_path / "nnUNet").symlink_to(model_dir.parent)
        return temp_path, task_dir.name, temp_root
    else:
        # Flattened Drive bundle:
        #   <root>/TaskXXX_Name/<trainer__plans>
        # We fabricate the missing config directory expected by nnUNet v1:
        #   <tmp>/nnUNet/3d_fullres/TaskXXX_Name/<trainer__plans>
        temp_root = tempfile.TemporaryDirectory()
        temp_path = Path(temp_root.name)
        nnunet_root = temp_path / "nnUNet"
        nnunet_root.mkdir(parents=True, exist_ok=True)
        (nnunet_root / "3d_fullres").symlink_to(maybe_model_dir)
        return temp_path, task_dir.name, temp_root


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    repo_root = Path(cfg["root"])

    model_cfg = cfg["approach_b"]
    model_root = Path(args.model_root or model_cfg["pancreas_model_dir"])
    input_dir = Path(args.input_dir or cfg["data"]["images"])
    output_dir = Path(args.output_dir or model_cfg["pancreas_preds"])
    task_id = int(args.task_id or model_cfg["pancreas_model_task_id"])

    if not model_root.exists():
        raise FileNotFoundError(
            f"PaNSegNet model directory not found: {model_root}\n"
            "Unpack the pretrained PaNSegNet model there first."
        )
    if not input_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {input_dir}")

    predict_cmd = discover_predict_cmd(repo_root)
    trainer_dir = find_trainer_dir(
        model_root=model_root,
        model_config=model_cfg["pancreas_model_config"],
        task_id=task_id,
        trainer=model_cfg["pancreas_model_trainer"],
        plans=model_cfg["pancreas_model_plans"],
    )
    results_folder, task_name, temp_results_root = ensure_results_folder(trainer_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp_input:
        tmp_input_path = Path(tmp_input)
        nii_files = sorted(input_dir.glob("*.nii.gz"))
        if not nii_files:
            raise FileNotFoundError(f"No .nii.gz files found in {input_dir}")

        for image_path in nii_files:
            stem = image_path.name.replace(".nii.gz", "")
            (tmp_input_path / f"{stem}_0000.nii.gz").symlink_to(image_path.resolve())

        cmd = [
            predict_cmd,
            "-i", str(tmp_input_path),
            "-o", str(output_dir),
            "-t", task_name,
            "-m", model_cfg["pancreas_model_config"],
            "-tr", model_cfg["pancreas_model_trainer"],
            "--folds",
            *[str(x) for x in model_cfg["pancreas_model_folds"]],
        ]

        env = os.environ.copy()
        env["RESULTS_FOLDER"] = str(results_folder)

        print("========================================================")
        print("Approach B — Stage 1: PaNSegNet Pancreas Segmentation")
        print(f"  Task    : {task_name}")
        print(f"  Command : {' '.join(shlex.quote(x) for x in cmd)}")
        print(f"  Results : {results_folder}")
        print(f"  Input   : {input_dir}")
        print(f"  Output  : {output_dir}")
        print("========================================================")
        subprocess.run(cmd, check=True, env=env)

    if temp_results_root is not None:
        temp_results_root.cleanup()

    print("")
    print(f"Pancreas predictions written to: {output_dir}")
    print("Next: python approach_b/crop_to_pancreas.py --config configs/paths.yaml --split all")


if __name__ == "__main__":
    main()
