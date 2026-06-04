"""Simple ipywidgets MRI / GT / prediction slice viewer.

Run from a notebook in this directory:

    %run visualize_widgets.py
    viewer(["NU28", "NU188", "EMC077"])

It reads image and ground-truth paths from ../data/test.txt and predictions
from ./predictions/<case>.nii.gz.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np

try:
    from IPython.display import display
    from ipywidgets import Checkbox, Dropdown, FloatSlider, HBox, IntSlider, Output, VBox
except ModuleNotFoundError:
    display = None
    Checkbox = Dropdown = FloatSlider = HBox = IntSlider = Output = VBox = None


REPO = Path(__file__).resolve().parent.parent
PRED_DIR = Path(__file__).resolve().parent / "predictions"
TEST_TXT = REPO / "data/test.txt"

AXES = {
    "Axial": 2,
    "Coronal": 1,
    "Sagittal": 0,
}


def build_case_map() -> dict[str, tuple[Path, Path]]:
    case_map: dict[str, tuple[Path, Path]] = {}
    for line in TEST_TXT.read_text().splitlines()[1:]:
        line = line.strip()
        if not line:
            continue
        image_path, mask_path = line.split(",")[:2]
        case = Path(image_path).name.replace(".nii.gz", "")
        case_map[case] = (Path(image_path), Path(mask_path))
    return case_map


def load_case(case: str, case_map: dict[str, tuple[Path, Path]]):
    image_path, mask_path = case_map[case]
    pred_path = PRED_DIR / f"{case}.nii.gz"

    img_data = nib.load(str(image_path)).get_fdata()
    mask_data = nib.load(str(mask_path)).get_fdata()
    if pred_path.exists():
        pred_data = nib.load(str(pred_path)).get_fdata()
    else:
        pred_data = np.zeros_like(mask_data)

    assert img_data.shape == pred_data.shape == mask_data.shape, (
        f"Shapes do not match for {case}: "
        f"image={img_data.shape}, pred={pred_data.shape}, mask={mask_data.shape}"
    )
    return img_data, pred_data, mask_data, image_path, pred_path, mask_path


def get_slice(arr: np.ndarray, axis: int, slice_idx: int, rotate: bool = True) -> np.ndarray:
    if axis == 0:
        sl = arr[slice_idx, :, :]
    elif axis == 1:
        sl = arr[:, slice_idx, :]
    else:
        sl = arr[:, :, slice_idx]
    return np.rot90(sl) if rotate else sl


def mask_dice(pred_data: np.ndarray, mask_data: np.ndarray) -> float:
    pred_bin = pred_data > 0
    mask_bin = mask_data > 0
    denom = int(pred_bin.sum() + mask_bin.sum())
    return float(2 * int((pred_bin & mask_bin).sum()) / denom) if denom else 1.0


def best_slice(mask_data: np.ndarray, axis: int) -> int:
    if mask_data.max() <= 0:
        return mask_data.shape[axis] // 2
    reduce_axes = tuple(i for i in range(mask_data.ndim) if i != axis)
    return int(np.argmax(mask_data.sum(axis=reduce_axes)))


def viewer(cases=("NU28", "NU188", "EMC077")):
    if Output is None:
        raise ImportError("viewer() requires IPython and ipywidgets. Use the CLI HTML renderer instead.")

    case_map = build_case_map()
    valid_cases = [case for case in cases if case in case_map]
    if not valid_cases:
        raise ValueError(f"No requested cases found in {TEST_TXT}")

    state = {}
    output = Output()

    case_dd = Dropdown(options=valid_cases, value=valid_cases[0], description="Case")
    axis_dd = Dropdown(options=list(AXES), value="Axial", description="View")
    slice_sl = IntSlider(description="Slice", min=0, max=1, step=1, value=0, continuous_update=False)
    vmin_sl = FloatSlider(description="vmin", continuous_update=False)
    vmax_sl = FloatSlider(description="vmax", continuous_update=False)
    rotate_cb = Checkbox(value=True, description="Rotate 90 deg")
    contour_cb = Checkbox(value=True, description="Show contour")

    def load_current_case():
        case = case_dd.value
        img_data, pred_data, mask_data, image_path, pred_path, mask_path = load_case(case, case_map)
        p1, p99 = np.percentile(img_data, [1, 99])
        step = max(float((img_data.max() - img_data.min()) / 200), 1e-6)

        state.update(
            case=case,
            img=img_data,
            pred=pred_data,
            mask=mask_data,
            image_path=image_path,
            pred_path=pred_path,
            mask_path=mask_path,
            dice=mask_dice(pred_data, mask_data),
        )

        vmin_sl.min = float(img_data.min())
        vmin_sl.max = float(img_data.max())
        vmin_sl.step = step
        vmin_sl.value = float(p1)
        vmax_sl.min = float(img_data.min())
        vmax_sl.max = float(img_data.max())
        vmax_sl.step = step
        vmax_sl.value = float(p99)
        update_slice_limits()

    def update_slice_limits():
        axis = AXES[axis_dd.value]
        mask_data = state["mask"]
        slice_sl.max = mask_data.shape[axis] - 1
        slice_sl.value = best_slice(mask_data, axis)

    def redraw(*_):
        if not state:
            return

        img_data = state["img"]
        pred_data = state["pred"]
        mask_data = state["mask"]
        axis = AXES[axis_dd.value]
        slice_idx = int(slice_sl.value)

        img_sl = get_slice(img_data, axis, slice_idx, rotate_cb.value)
        pred_sl = get_slice(pred_data, axis, slice_idx, rotate_cb.value)
        mask_sl = get_slice(mask_data, axis, slice_idx, rotate_cb.value)
        pred_bin = (pred_sl > 0).astype(np.uint8)
        mask_bin = (mask_sl > 0).astype(np.uint8)

        with output:
            output.clear_output(wait=True)
            print(f"Case: {state['case']} | View: {axis_dd.value} | Slice: {slice_idx}")
            print(f"Image shape: {img_data.shape} | Pred shape: {pred_data.shape} | Mask shape: {mask_data.shape}")
            print(
                f"GT voxels: {int((mask_data > 0).sum())} | "
                f"Pred voxels: {int((pred_data > 0).sum())} | Dice: {state['dice']:.4f}"
            )

            fig, axes = plt.subplots(1, 3, figsize=(18, 6))

            axes[0].imshow(img_sl, cmap="gray", vmin=vmin_sl.value, vmax=vmax_sl.value)
            axes[0].set_title(f"Image only ({axis_dd.value} {slice_idx})")
            axes[0].axis("off")

            axes[1].imshow(img_sl, cmap="gray", vmin=vmin_sl.value, vmax=vmax_sl.value)
            axes[1].imshow(mask_bin, cmap="Reds", alpha=0.45)
            if contour_cb.value and mask_bin.max() > 0:
                axes[1].contour(mask_bin, colors="yellow", linewidths=1)
            axes[1].set_title("Ground truth")
            axes[1].axis("off")

            axes[2].imshow(img_sl, cmap="gray", vmin=vmin_sl.value, vmax=vmax_sl.value)
            axes[2].imshow(mask_bin, cmap="Reds", alpha=0.35)
            axes[2].imshow(pred_bin, cmap="Greens", alpha=0.35)
            if contour_cb.value and mask_bin.max() > 0:
                axes[2].contour(mask_bin, colors="yellow", linewidths=1)
            if contour_cb.value and pred_bin.max() > 0:
                axes[2].contour(pred_bin, colors="lime", linewidths=1)
            axes[2].set_title("GT (red) vs Pred (green)")
            axes[2].axis("off")

            plt.tight_layout()
            plt.show()

    def on_case_change(*_):
        load_current_case()
        redraw()

    def on_axis_change(*_):
        update_slice_limits()
        redraw()

    case_dd.observe(on_case_change, names="value")
    axis_dd.observe(on_axis_change, names="value")
    for widget in (slice_sl, vmin_sl, vmax_sl, rotate_cb, contour_cb):
        widget.observe(redraw, names="value")

    load_current_case()
    controls = VBox(
        [
            HBox([case_dd, axis_dd]),
            slice_sl,
            HBox([vmin_sl, vmax_sl]),
            HBox([rotate_cb, contour_cb]),
        ]
    )
    display(controls, output)
    redraw()


if __name__ == "__main__":
    import argparse, base64, io, json

    # Terminal / WSL2: no display — generate a self-contained HTML file instead
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser(description="Generate an HTML slice viewer for test-set predictions.")
    parser.add_argument("cases", nargs="*", default=["NU28", "NU188", "EMC077"])
    parser.add_argument("--pred-dir", type=Path, default=PRED_DIR)
    parser.add_argument("--out-html", type=Path, default=Path(__file__).parent / "visualization.html")
    parser.add_argument("--title", default="Test Predictions")
    parser.add_argument("--n-slices", type=int, default=12)
    args = parser.parse_args()

    PRED_DIR = args.pred_dir
    CASES = args.cases
    N_SLICES = args.n_slices
    OUT_HTML = args.out_html
    OUT_HTML.parent.mkdir(parents=True, exist_ok=True)

    case_map = build_case_map()
    cases_valid = [c for c in CASES if c in case_map]
    if not cases_valid:
        sys.exit(f"None of {CASES} found in test.txt")

    def fig_to_b64(fig):
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=90, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        buf.seek(0)
        return base64.b64encode(buf.read()).decode()

    cases_data = {}
    for name in cases_valid:
        print(f"  Rendering {name} ...", end=" ", flush=True)
        img_data, pred_data, mask_data, _, _, _ = load_case(name, case_map)
        p1, p99 = float(np.percentile(img_data, 1)), float(np.percentile(img_data, 99))
        dc = mask_dice(pred_data, mask_data)
        n_gt   = int((mask_data > 0).sum())
        n_pred = int((pred_data > 0).sum())
        bz     = int(best_slice(mask_data, axis=2))  # axial

        candidates = sorted(set([bz] + list(np.linspace(0, img_data.shape[2]-1,
                                                          N_SLICES, dtype=int))))
        frames = []
        for z in candidates:
            img_sl  = np.rot90(img_data[:, :, z])
            pred_sl = (np.rot90(pred_data[:, :, z]) > 0).astype(np.uint8)
            mask_sl = (np.rot90(mask_data[:, :, z]) > 0).astype(np.uint8)

            fig, axes = plt.subplots(1, 3, figsize=(12, 4))
            fig.patch.set_facecolor("#111")
            plt.subplots_adjust(left=0.01, right=0.99, top=0.88, bottom=0.02, wspace=0.04)
            for ax, data, title, cmap in [
                (axes[0], img_sl,  "MRI",            None),
                (axes[1], img_sl,  "Ground Truth",   None),
                (axes[2], img_sl,  "GT vs Pred",     None),
            ]:
                ax.imshow(data, cmap="gray", vmin=p1, vmax=p99, interpolation="bilinear")
                ax.set_title(title, color="white", fontsize=10)
                ax.axis("off"); ax.set_facecolor("black")

            axes[1].imshow(mask_sl, cmap="Reds",   alpha=0.45)
            if mask_sl.max() > 0:
                axes[1].contour(mask_sl, colors="yellow", linewidths=1)

            axes[2].imshow(mask_sl, cmap="Reds",   alpha=0.35)
            axes[2].imshow(pred_sl, cmap="Greens", alpha=0.35)
            if mask_sl.max() > 0:
                axes[2].contour(mask_sl, colors="yellow", linewidths=1)
            if pred_sl.max() > 0:
                axes[2].contour(pred_sl, colors="lime", linewidths=1)

            label = f"{'★ ' if z == bz else ''}Slice {z}"
            fig.suptitle(label, color="#aaa", fontsize=9)
            frames.append((int(z), fig_to_b64(fig)))
            plt.close(fig)

        cases_data[name] = dict(dc=float(dc), n_gt=n_gt, n_pred=n_pred,
                                best_z=bz, frames=frames)
        print(f"Dice={dc:.4f}")

    # build HTML
    tabs = "".join(
        f'<div class="tab" data-name="{n}" onclick="show(\'{n}\')">'
        f'{n} Dice:{d["dc"]:.3f}{"  ⚠" if d["n_pred"]==0 and d["n_gt"]>0 else ""}</div>'
        for n, d in cases_data.items()
    )
    panels = []
    for name, d in cases_data.items():
        n = len(d["frames"])
        bi = next((i for i, (z, _) in enumerate(d["frames"]) if z == d["best_z"]), 0)
        bad = (f'<span style="color:#f66;font-weight:bold"> ⚠ predicted 0 voxels '
               f'(GT={d["n_gt"]} vox)</span>' if d["n_pred"]==0 and d["n_gt"]>0 else "")
        panels.append(
            f'<div class="panel" id="p_{name}">'
            f'<p style="font-size:.88em;color:#aaa">GT:{d["n_gt"]} vox | '
            f'Pred:{d["n_pred"]} vox | Dice:{d["dc"]:.4f}{bad}</p>'
            f'<input type="range" id="sl_{name}" data-name="{name}" '
            f'min="0" max="{n-1}" value="{bi}" style="width:80%;accent-color:#e94560">'
            f'<label id="lbl_{name}" style="font-size:.85em;color:#bbb"></label><br>'
            f'<img id="img_{name}" style="width:100%;max-width:960px;border-radius:6px"/>'
            f'</div>'
        )
    json_data = {n: {"frames": d["frames"]} for n, d in cases_data.items()}

    html = f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<title>{args.title}</title>
<style>
  body{{background:#111;color:#eee;font-family:sans-serif;padding:12px}}
  .tabs{{display:flex;flex-wrap:wrap;gap:6px;margin-bottom:12px}}
  .tab{{padding:6px 14px;background:#222;border:1px solid #444;border-radius:4px;cursor:pointer;font-size:.85em}}
  .tab.active{{background:#e94560;border-color:#e94560}}
  .panel{{display:none}}.panel.active{{display:block}}
</style></head><body>
<h2 style="margin-bottom:8px">{args.title}</h2>
<div class="tabs">{tabs}</div>
{"".join(panels)}
<script>
const D={json.dumps(json_data)};
function show(n){{
  document.querySelectorAll('.tab').forEach(t=>t.classList.toggle('active',t.dataset.name===n));
  document.querySelectorAll('.panel').forEach(p=>p.classList.toggle('active',p.id==='p_'+n));
  upd(n);
}}
function upd(n){{
  const sl=document.getElementById('sl_'+n);
  document.getElementById('img_'+n).src='data:image/png;base64,'+D[n].frames[sl.value][1];
  document.getElementById('lbl_'+n).textContent=' Slice '+D[n].frames[sl.value][0];
}}
document.querySelectorAll('input[type=range]').forEach(s=>s.addEventListener('input',()=>upd(s.dataset.name)));
show('{cases_valid[0]}');
</script></body></html>"""

    OUT_HTML.write_text(html)
    print(f"\nSaved → {OUT_HTML}")
