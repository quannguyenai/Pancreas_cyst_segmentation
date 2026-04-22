"""approach_d/visualize.py — Interactive slice viewer: MRI + GT + Prediction.

Usage (from repo root):
    python approach_d/visualize.py NU28 NU188 EMC077

Controls:
    Slider       — scroll through axial slices
    Left/Right   — previous/next case
    A/S/C keys   — toggle Axial / Sagittal / Coronal view
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.widgets as mwidgets
import nibabel as nib
import numpy as np


# ── helpers ───────────────────────────────────────────────────────────────────

def _build_gt_map(test_txt: str) -> dict[str, tuple[Path, Path]]:
    m: dict[str, tuple[Path, Path]] = {}
    for line in Path(test_txt).read_text().splitlines()[1:]:
        line = line.strip()
        if not line:
            continue
        p = line.split(",")
        stem = Path(p[0]).name.replace(".nii.gz", "")
        m[stem] = (Path(p[0]), Path(p[1]))
    return m


def _load(path: Path) -> np.ndarray:
    return np.asarray(nib.load(str(path)).dataobj).astype(np.float32)


def _pct_norm(vol: np.ndarray) -> np.ndarray:
    lo, hi = np.percentile(vol, 1), np.percentile(vol, 99)
    return np.clip((vol - lo) / max(hi - lo, 1e-6), 0, 1)


def _get_slice(vol: np.ndarray, axis: int, idx: int) -> np.ndarray:
    if axis == 0:
        return vol[idx]
    elif axis == 1:
        return vol[:, idx]
    else:
        return vol[:, :, idx]


AXIS_LABEL = {0: "Axial (Z)", 1: "Coronal (Y)", 2: "Sagittal (X)"}


# ── viewer ────────────────────────────────────────────────────────────────────

class CaseViewer:
    GT_COLOR   = np.array([0.0, 1.0, 0.0])   # green
    PRED_COLOR = np.array([1.0, 0.3, 0.0])   # red-orange

    def __init__(self, cases: list[str], gt_map, pred_dir: Path):
        self.cases    = cases
        self.gt_map   = gt_map
        self.pred_dir = pred_dir
        self.cidx     = 0
        self.axis     = 0      # 0=axial, 1=coronal, 2=sagittal
        self._cache: dict[str, tuple] = {}

        self.fig = plt.figure(figsize=(15, 6))
        self.fig.patch.set_facecolor("#1a1a2e")

        # axes: image | GT overlay | Pred overlay
        self.ax_img  = self.fig.add_axes([0.02, 0.18, 0.30, 0.76])
        self.ax_gt   = self.fig.add_axes([0.35, 0.18, 0.30, 0.76])
        self.ax_pred = self.fig.add_axes([0.68, 0.18, 0.30, 0.76])

        for ax in (self.ax_img, self.ax_gt, self.ax_pred):
            ax.set_facecolor("black")
            ax.axis("off")

        # slice slider
        ax_sl = self.fig.add_axes([0.12, 0.06, 0.76, 0.04])
        self.slider = mwidgets.Slider(ax_sl, "Slice", 0, 1, valinit=0,
                                      color="#e94560", track_color="#16213e")
        self.slider.on_changed(self._on_slice)

        # title + stats text
        self.title_txt = self.fig.text(0.5, 0.97, "", ha="center", va="top",
                                       fontsize=13, color="white", fontweight="bold")
        self.stats_txt = self.fig.text(0.5, 0.005, "", ha="center", va="bottom",
                                       fontsize=9, color="#aaaaaa")

        self.fig.canvas.mpl_connect("key_press_event", self._on_key)
        self._load_case()
        plt.show()

    # ── data loading ──────────────────────────────────────────────────────────

    def _load_case(self):
        name = self.cases[self.cidx]
        if name not in self._cache:
            info = self.gt_map.get(name)
            if info is None:
                print(f"[WARN] {name} not in test.txt")
                return
            img_path, gt_path = info
            pred_path = self.pred_dir / f"{name}.nii.gz"

            img  = _pct_norm(_load(img_path))
            gt   = (_load(gt_path) > 0).astype(np.uint8)
            pred = (_load(pred_path) > 0).astype(np.uint8) if pred_path.exists() else np.zeros_like(gt)

            dc     = self._dice(pred, gt)
            n_gt   = int(gt.sum())
            n_pred = int(pred.sum())
            self._cache[name] = (img, gt, pred, dc, n_gt, n_pred)

        img, gt, pred, dc, n_gt, n_pred = self._cache[self.cases[self.cidx]]
        nslices = img.shape[self.axis]

        # set slider to slice with most GT, or middle if no GT
        if gt.sum() > 0:
            sums = [_get_slice(gt, self.axis, i).sum() for i in range(nslices)]
            best = int(np.argmax(sums))
        else:
            best = nslices // 2

        self.slider.valmin = 0
        self.slider.valmax = nslices - 1
        self.slider.ax.set_xlim(0, nslices - 1)
        self.slider.set_val(best)   # triggers _on_slice

        case_label = f"[{self.cidx+1}/{len(self.cases)}]  {self.cases[self.cidx]}   ({AXIS_LABEL[self.axis]})"
        self.title_txt.set_text(case_label)
        nan_tag = " (model missed entirely)" if n_pred == 0 and n_gt > 0 else ""
        self.stats_txt.set_text(
            f"Dice: {dc:.4f}  |  GT voxels: {n_gt}  |  Pred voxels: {n_pred}{nan_tag}"
            f"     ←/→ change case   |   A/C/S change view"
        )
        self.fig.canvas.draw_idle()

    # ── rendering ─────────────────────────────────────────────────────────────

    def _render(self, sl: int):
        name = self.cases[self.cidx]
        if name not in self._cache:
            return
        img, gt, pred, *_ = self._cache[name]

        sl = int(np.clip(sl, 0, img.shape[self.axis] - 1))

        img_sl  = np.rot90(_get_slice(img,  self.axis, sl))
        gt_sl   = np.rot90(_get_slice(gt,   self.axis, sl))
        pred_sl = np.rot90(_get_slice(pred, self.axis, sl))

        def _overlay(gray_sl, mask_sl, color):
            rgb = np.stack([gray_sl] * 3, axis=-1)
            if mask_sl.any():
                alpha = 0.45
                rgb[mask_sl > 0] = (1 - alpha) * rgb[mask_sl > 0] + alpha * color
            return rgb

        for ax, data, title in [
            (self.ax_img,  np.stack([img_sl]*3, axis=-1), "MRI"),
            (self.ax_gt,   _overlay(img_sl, gt_sl,   self.GT_COLOR),   "Ground Truth  (green)"),
            (self.ax_pred, _overlay(img_sl, pred_sl, self.PRED_COLOR), "Prediction  (orange)"),
        ]:
            ax.clear()
            ax.imshow(data, cmap="gray", vmin=0, vmax=1, interpolation="bilinear")
            ax.set_title(title, color="white", fontsize=10, pad=4)
            ax.axis("off")

        self.fig.canvas.draw_idle()

    # ── callbacks ─────────────────────────────────────────────────────────────

    def _on_slice(self, val):
        self._render(int(val))

    def _on_key(self, event):
        if event.key == "right":
            self.cidx = (self.cidx + 1) % len(self.cases)
            self._load_case()
        elif event.key == "left":
            self.cidx = (self.cidx - 1) % len(self.cases)
            self._load_case()
        elif event.key in ("a", "A"):
            self.axis = 0; self._load_case()
        elif event.key in ("c", "C"):
            self.axis = 1; self._load_case()
        elif event.key in ("s", "S"):
            self.axis = 2; self._load_case()

    @staticmethod
    def _dice(pred, gt) -> float:
        p, g = pred > 0, gt > 0
        inter = (p & g).sum()
        denom = p.sum() + g.sum()
        return float(2 * inter / denom) if denom > 0 else 1.0


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    repo_root = Path(__file__).parent.parent
    gt_map    = _build_gt_map(repo_root / "data/test.txt")
    pred_dir  = repo_root / "approach_d/predictions"

    # default: all 3 requested cases + the 3 NaN cases
    default = ["NU28", "NU188", "EMC077", "NU71", "NYU0165"]
    cases = sys.argv[1:] if len(sys.argv) > 1 else default

    # keep only cases present in test set
    valid = [c for c in cases if c in gt_map]
    missing = [c for c in cases if c not in gt_map]
    if missing:
        print(f"[WARN] Not in test.txt: {missing}")
    if not valid:
        sys.exit("No valid cases to visualize.")

    print(f"Visualizing {len(valid)} cases: {valid}")
    print("Keys: ←/→ = prev/next case   |   A = axial   C = coronal   S = sagittal")

    CaseViewer(valid, gt_map, pred_dir)


if __name__ == "__main__":
    main()
