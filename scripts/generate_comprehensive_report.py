#!/usr/bin/env python3
"""Generate a comprehensive PI-facing experiment report."""

from __future__ import annotations

import os
from pathlib import Path
from textwrap import dedent

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    Image,
    KeepTogether,
    ListFlowable,
    ListItem,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


ROOT = Path("/home/huy/quan_nguyen/aima/pancrea_cyst")
OUT = ROOT / "results" / "analysis"
REPORT_MD = OUT / "Pancreatic_Cyst_Segmentation_Comprehensive_Report.md"
REPORT_PDF = OUT / "Pancreatic_Cyst_Segmentation_Comprehensive_Report.pdf"

APPROACH_DIRS = {
    "A-3D": ROOT / "approach_a" / "prediction" / "3d_fullres",
    "A-2.5D": ROOT / "approach_a" / "prediction" / "2d_stack5",
    "B": ROOT / "approach_b" / "predictio" / "full_space",
    "D": ROOT / "approach_d" / "prediction",
}


def read_csv(name: str) -> pd.DataFrame:
    return pd.read_csv(OUT / name)


def fmt(x: float, ndigits: int = 4) -> str:
    if pd.isna(x):
        return ""
    return f"{float(x):.{ndigits}f}"


def case_map() -> dict[str, tuple[Path, Path]]:
    mapping: dict[str, tuple[Path, Path]] = {}
    lines = (ROOT / "data" / "test.txt").read_text().splitlines()[1:]
    for line in lines:
        image_path, mask_path = line.split(",")[:2]
        case = Path(image_path).name.replace(".nii.gz", "")
        mapping[case] = (Path(image_path), Path(mask_path))
    return mapping


def load(path: Path) -> np.ndarray:
    return np.asanyarray(nib.load(str(path)).dataobj)


def best_slice(mask: np.ndarray) -> int:
    if mask.max() <= 0:
        return mask.shape[2] // 2
    return int(np.argmax((mask > 0).sum(axis=(0, 1))))


def dice(pred: np.ndarray, gt: np.ndarray) -> float:
    p = pred > 0
    g = gt > 0
    denom = int(p.sum() + g.sum())
    return 2 * int((p & g).sum()) / denom if denom else 1.0


def make_case_comparison(case: str, out_path: Path) -> Path:
    mapping = case_map()
    img_path, mask_path = mapping[case]
    img = load(img_path).astype(float)
    gt = load(mask_path) > 0
    z = best_slice(gt)
    p1, p99 = np.percentile(img, [1, 99])
    img_sl = np.rot90(img[:, :, z])
    gt_sl = np.rot90(gt[:, :, z]).astype(np.uint8)

    fig, axes = plt.subplots(1, 1 + len(APPROACH_DIRS), figsize=(16, 3.7))
    fig.patch.set_facecolor("white")

    def draw(ax, pred_sl=None, title=""):
        ax.imshow(img_sl, cmap="gray", vmin=p1, vmax=p99)
        if gt_sl.max() > 0:
            ax.imshow(gt_sl, cmap="Reds", alpha=0.35)
            ax.contour(gt_sl, colors="yellow", linewidths=0.8)
        if pred_sl is not None and pred_sl.max() > 0:
            ax.imshow(pred_sl, cmap="Greens", alpha=0.35)
            ax.contour(pred_sl, colors="lime", linewidths=0.8)
        ax.set_title(title, fontsize=9)
        ax.axis("off")

    draw(axes[0], None, f"{case}: GT only\naxial z={z}")
    for ax, (approach, pred_dir) in zip(axes[1:], APPROACH_DIRS.items()):
        pred = load(pred_dir / f"{case}.nii.gz")
        pred_sl = np.rot90(pred[:, :, z] > 0).astype(np.uint8)
        draw(
            ax,
            pred_sl,
            f"{approach}\nDice {dice(pred, gt):.3f}, pred {int((pred > 0).sum())} vox",
        )
    fig.suptitle("Yellow/red = ground truth, lime/green = prediction", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out_path


def markdown_table(df: pd.DataFrame, cols: list[str], headers: list[str] | None = None, float_cols: set[str] | None = None) -> str:
    float_cols = float_cols or set()
    headers = headers or cols
    rows = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in df.iterrows():
        vals = []
        for c in cols:
            v = row[c]
            vals.append(fmt(v) if c in float_cols else str(v))
        rows.append("| " + " | ".join(vals) + " |")
    return "\n".join(rows)


def make_markdown() -> str:
    summary = read_csv("summary_segmentation_metrics.csv")
    volumes = read_csv("cyst_volume_distribution_by_split.csv")
    by_bin = read_csv("metrics_by_cyst_size_bin.csv")
    failures = read_csv("universal_failure_cases.csv")
    top5 = read_csv("top5_successful_cases.csv")
    wins = read_csv("approach_wins_by_case.csv")
    non_a3d = read_csv("cases_where_non_a3d_beats_a3d.csv")

    volume_summary = (
        volumes.groupby("split")["gt_pct"]
        .describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9])
        .reset_index()
        .rename(columns={"50%": "median", "10%": "p10", "90%": "p90"})
    )
    win_counts = wins["winner"].value_counts().rename_axis("approach").reset_index(name="n_case_wins")
    top_non_a3d = non_a3d.head(10).copy()

    summary_view = summary[
        [
            "approach",
            "dice_mean",
            "precision_mean",
            "recall_mean",
            "accuracy_mean",
            "balanced_accuracy_mean",
            "iou_mean",
            "micro_f1",
            "micro_precision",
            "micro_recall",
        ]
    ].copy()
    summary_view = summary_view.rename(columns={"dice_mean": "f1_dice_mean"})

    md = f"""# Pancreatic Cyst Segmentation on MRI: Comprehensive Experiment Report

Generated from local results in `{ROOT}`.

## Executive Summary

Plain 3D nnU-Net on full MRI volumes (**A-3D**) remains the strongest current baseline. It has the highest mean test Dice/F1, highest recall, highest IoU, and most per-case wins. The more complex pancreas-ROI cascade (**B**) and TransUNet-style model (**D**) do not improve the overall result in this single-fold experiment.

The main limitation is not average-case performance, but robustness. Test performance drops sharply for very small cysts, and a small set of universal failures remains difficult for every model. IU59 is the key exception: it is a large cyst, but all models fail or severely under-segment it, likely because the cyst is atypically positioned relative to the predicted pancreas.

## Dataset

| Item | Value |
|---|---:|
| Modality | MRI T2-weighted |
| Total cases | 358 |
| Train cases | 247 |
| Validation cases | 37 |
| Test cases | 74 |
| Task | Binary pancreatic cyst segmentation |
| Evaluation setup | Single fold, no cross-validation |

### Cyst Volume Distribution by Split

{markdown_table(volume_summary, ["split", "count", "mean", "std", "min", "p10", "25%", "median", "75%", "p90", "max"], float_cols={"count", "mean", "std", "min", "p10", "25%", "median", "75%", "p90", "max"})}

The cyst volume distribution is heavily skewed. Many cysts occupy far below 0.05% of all image voxels, so small localization errors can collapse Dice/F1.

![Cyst volume distribution](cyst_volume_distribution_with_universal_failures.png)

## Compared Approaches

| Approach | Description |
|---|---|
| A-3D | nnU-Net v2 `3d_fullres` on full volumes |
| A-2.5D | nnU-Net v2 2D model using 5-slice stack-as-channels input |
| B | PaNSegNet pancreas localization, pancreas ROI crop, nnU-Net 3D cyst segmentation, paste back to full space |
| D | PaNSegNet / TransUNet-style architecture through nnU-Net v1 |

## Main Test Metrics

Voxel-level F1 is mathematically identical to Dice for binary segmentation. Accuracy is reported, but it should not be used as the primary model-selection metric because background voxels dominate the volume.

{markdown_table(summary_view, ["approach", "f1_dice_mean", "precision_mean", "recall_mean", "accuracy_mean", "balanced_accuracy_mean", "iou_mean", "micro_f1", "micro_precision", "micro_recall"], ["Approach", "Mean Dice/F1", "Mean Precision", "Mean Recall", "Mean Accuracy", "Mean Balanced Acc.", "Mean IoU", "Micro F1", "Micro Precision", "Micro Recall"], float_cols={"f1_dice_mean", "precision_mean", "recall_mean", "accuracy_mean", "balanced_accuracy_mean", "iou_mean", "micro_f1", "micro_precision", "micro_recall"})}

### Interpretation

- **A-3D is best overall**: highest mean Dice/F1, recall, IoU, and per-case wins.
- **D is second by mean Dice/F1**, but does not outperform A-3D at this dataset size.
- **B has high precision but lower recall**, consistent with a conservative ROI/cascade behavior.
- **A-2.5D has the lowest mean performance**, but it still helps on several individual cases, including partial localization of IU59.

## Per-Case Wins

{markdown_table(win_counts, ["approach", "n_case_wins"])}

![Approach wins](approach_wins_by_case.png)

## Performance by Cyst Size

{markdown_table(by_bin, ["volume_bin", "approach", "n_cases", "dice_mean", "dice_median", "precision_mean", "recall_mean", "pred_gt_ratio_median"], ["Cyst volume bin", "Approach", "N", "Mean Dice/F1", "Median Dice/F1", "Mean Precision", "Mean Recall", "Median Pred/GT Volume"], float_cols={"dice_mean", "dice_median", "precision_mean", "recall_mean", "pred_gt_ratio_median"})}

![Dice by cyst size bin](dice_by_cyst_size_bin.png)

![Dice vs cyst volume](dice_vs_cyst_volume.png)

## Precision-Recall Behavior

![Precision recall by case](precision_recall_by_case.png)

Precision-recall plots show that failures are not all blank masks. Some cases are wrong-location false positives, some are severe under-segmentations, and some are complete misses.

## Universal Failure Cases

Universal failures are cases where every approach has Dice/F1 below 0.2.

{markdown_table(failures, ["case", "gt_voxels", "gt_pct", "max_dice", "mean_dice"], ["Case", "GT voxels", "GT % of image", "Max Dice/F1", "Mean Dice/F1"], float_cols={"gt_pct", "max_dice", "mean_dice"})}

### IU59 Deep Dive

IU59 is important because it is a large-cyst failure, not a small-object failure.

| IU59 fact | Value |
|---|---:|
| GT cyst voxels | 33,927 |
| GT cyst % of image | 0.4930% |
| A-3D prediction | 0 voxels |
| A-2.5D prediction | 2,093 voxels |
| B prediction | 43 voxels, wrong location |
| D prediction | 0 voxels |
| Fraction of GT cyst inside predicted pancreas mask | 1.4% |

The likely explanation is atypical anatomical context: the cyst is large, but mostly outside the predicted pancreas body. A-2.5D detects a small high-confidence core, while A-3D and D suppress the entire lesion. This suggests an out-of-distribution localization/context failure, not simply a resolution problem.

![IU59 comparison](case_compare_IU59.png)

## Top 5 Successful Cases

Top successful cases were selected by highest mean Dice/F1 across all four approaches.

{markdown_table(top5, ["case", "gt_voxels", "gt_pct", "A-3D", "A-2.5D", "B", "D", "mean_dice", "min_dice"], ["Case", "GT voxels", "GT %", "A-3D", "A-2.5D", "B", "D", "Mean Dice/F1", "Min Dice/F1"], float_cols={"gt_pct", "A-3D", "A-2.5D", "B", "D", "mean_dice", "min_dice"})}

![CAD262 comparison](case_compare_CAD262.png)

## Where Other Approaches Beat A-3D

A-3D is best on average, but another approach has higher Dice/F1 on 43 of 74 individual test cases. This does not overturn the main result, but it suggests that ensembling or case-specific strengths may be useful.

Top examples where a non-A-3D model beats A-3D:

{markdown_table(top_non_a3d, ["case", "gt_voxels", "gt_pct", "A-3D", "A-2.5D", "B", "D", "best_non_A3D", "best_non_A3D_minus_A3D"], ["Case", "GT voxels", "GT %", "A-3D", "A-2.5D", "B", "D", "Best non-A3D", "Delta vs A-3D"], float_cols={"gt_pct", "A-3D", "A-2.5D", "B", "D", "best_non_A3D_minus_A3D"})}

![Dice delta vs A-3D](dice_delta_vs_a3d.png)

## Limitations

1. Single-fold experiment only; no 5-fold cross-validation.
2. Distance metrics from earlier CSVs should be verified before being emphasized.
3. Accuracy is inflated by background dominance and should not drive conclusions.
4. Approach B needs dedicated ROI coverage analysis to separate pancreas localization errors from cyst segmentation errors.
5. Failure modes are heterogeneous: tiny cysts, wrong-location false positives, under-segmentation, and large outlier failures such as IU59.

## Recommendations and Future Plan

1. **Keep A-3D as the main baseline and deployment candidate.**
2. **Run 5-fold cross-validation** for a more stable estimate and confidence intervals.
3. **Report stratified metrics by cyst size** in every future summary.
4. **Verify HD95/ASD implementation** before including distance metrics in PI-facing conclusions.
5. **Perform case-level failure taxonomy**: true miss, wrong-location false positive, under-segmentation, over-segmentation.
6. **Investigate IU59-like anatomy** by checking cyst relationship to pancreas masks and scanner/protocol metadata.
7. **Improve small-cyst handling** using volume-aware sampling, hard-case mining, or loss weighting.
8. **Consider ensemble experiments** because alternative approaches outperform A-3D on some individual cases.
9. **For Approach B**, quantify whether GT cysts are inside pancreas ROI/crop and whether crop margins need adjustment.
10. **Add uncertainty/confidence outputs** to identify cases that should be flagged for manual review.

## Bottom Line

A-3D is the strongest and simplest current method. The next research step should not be another architecture first; it should be stronger evaluation, size-stratified reporting, verified metrics, and targeted analysis of hard cases such as very small cysts and IU59-like atypical large cysts.
"""
    return md


def para(text: str, style) -> Paragraph:
    return Paragraph(text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"), style)


def pdf_table(df: pd.DataFrame, cols: list[str], headers: list[str], float_cols: set[str] | None = None, max_rows: int | None = None):
    float_cols = float_cols or set()
    if max_rows is not None:
        df = df.head(max_rows)
    data = [headers]
    for _, row in df.iterrows():
        vals = []
        for c in cols:
            vals.append(fmt(row[c]) if c in float_cols else str(row[c]))
        data.append(vals)
    table = Table(data, repeatRows=1)
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#d9eaf7")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.black),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#aaaaaa")),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 7),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f7f7f7")]),
            ]
        )
    )
    return table


def pdf_image(path: Path, width: float = 6.7 * inch):
    img = Image(str(path))
    ratio = img.imageHeight / img.imageWidth
    img.drawWidth = width
    img.drawHeight = width * ratio
    return img


def make_pdf() -> None:
    summary = read_csv("summary_segmentation_metrics.csv")
    volumes = read_csv("cyst_volume_distribution_by_split.csv")
    by_bin = read_csv("metrics_by_cyst_size_bin.csv")
    failures = read_csv("universal_failure_cases.csv")
    top5 = read_csv("top5_successful_cases.csv")
    wins = read_csv("approach_wins_by_case.csv")
    non_a3d = read_csv("cases_where_non_a3d_beats_a3d.csv").head(10)

    volume_summary = (
        volumes.groupby("split")["gt_pct"]
        .describe(percentiles=[0.1, 0.5, 0.9])
        .reset_index()
        .rename(columns={"50%": "median", "10%": "p10", "90%": "p90"})
    )
    win_counts = wins["winner"].value_counts().rename_axis("approach").reset_index(name="n_case_wins")
    summary_view = summary[
        ["approach", "dice_mean", "precision_mean", "recall_mean", "accuracy_mean", "balanced_accuracy_mean", "iou_mean", "micro_f1"]
    ].copy()
    summary_view = summary_view.rename(columns={"dice_mean": "f1_dice_mean"})

    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="Small", parent=styles["BodyText"], fontSize=8, leading=10, alignment=TA_LEFT))
    styles.add(ParagraphStyle(name="Tight", parent=styles["BodyText"], fontSize=9, leading=11, spaceAfter=4))

    doc = SimpleDocTemplate(
        str(REPORT_PDF),
        pagesize=letter,
        rightMargin=0.45 * inch,
        leftMargin=0.45 * inch,
        topMargin=0.45 * inch,
        bottomMargin=0.45 * inch,
    )
    story = []
    H1, H2, body = styles["Title"], styles["Heading2"], styles["Tight"]

    story += [Paragraph("Pancreatic Cyst Segmentation on MRI", H1), Paragraph("Comprehensive Experiment Report", styles["Heading1"])]
    story += [
        para(
            "Plain 3D nnU-Net (A-3D) is the strongest current baseline. The main remaining problem is robustness on very small cysts and atypical large outliers such as IU59.",
            body,
        ),
        Spacer(1, 0.12 * inch),
    ]

    story += [Paragraph("Dataset", H2)]
    story.append(
        pdf_table(
            pd.DataFrame(
                [
                    ["Modality", "MRI T2-weighted"],
                    ["Total cases", "358"],
                    ["Train / Val / Test", "247 / 37 / 74"],
                    ["Task", "Binary pancreatic cyst segmentation"],
                    ["Evaluation", "Single fold, no cross-validation"],
                ],
                columns=["Item", "Value"],
            ),
            ["Item", "Value"],
            ["Item", "Value"],
        )
    )
    story += [Spacer(1, 0.1 * inch), pdf_table(volume_summary, ["split", "count", "mean", "std", "min", "p10", "median", "p90", "max"], ["Split", "N", "Mean %", "Std", "Min", "P10", "Median", "P90", "Max"], {"count", "mean", "std", "min", "p10", "median", "p90", "max"})]
    story += [Spacer(1, 0.1 * inch), pdf_image(OUT / "cyst_volume_distribution_with_universal_failures.png")]

    story += [PageBreak(), Paragraph("Main Test Results", H2)]
    story.append(
        pdf_table(
            summary_view,
            ["approach", "f1_dice_mean", "precision_mean", "recall_mean", "accuracy_mean", "balanced_accuracy_mean", "iou_mean", "micro_f1"],
            ["Approach", "Dice/F1", "Precision", "Recall", "Accuracy", "Balanced Acc.", "IoU", "Micro F1"],
            {"f1_dice_mean", "precision_mean", "recall_mean", "accuracy_mean", "balanced_accuracy_mean", "iou_mean", "micro_f1"},
        )
    )
    story += [
        Spacer(1, 0.08 * inch),
        para("Accuracy is high for every approach because background voxels dominate; Dice/F1, recall, precision, and IoU are more informative.", body),
        Spacer(1, 0.12 * inch),
        Paragraph("Per-Case Wins", H2),
        pdf_table(win_counts, ["approach", "n_case_wins"], ["Approach", "Case wins"]),
        Spacer(1, 0.1 * inch),
        pdf_image(OUT / "approach_wins_by_case.png", width=5.1 * inch),
    ]

    story += [PageBreak(), Paragraph("Performance by Cyst Size", H2)]
    story.append(
        pdf_table(
            by_bin,
            ["volume_bin", "approach", "n_cases", "dice_mean", "dice_median", "precision_mean", "recall_mean", "pred_gt_ratio_median"],
            ["Volume bin", "Approach", "N", "Mean Dice", "Median Dice", "Precision", "Recall", "Pred/GT"],
            {"dice_mean", "dice_median", "precision_mean", "recall_mean", "pred_gt_ratio_median"},
        )
    )
    story += [Spacer(1, 0.1 * inch), pdf_image(OUT / "dice_by_cyst_size_bin.png"), Spacer(1, 0.08 * inch), pdf_image(OUT / "dice_vs_cyst_volume.png")]

    story += [PageBreak(), Paragraph("Precision-Recall and Model Differences", H2)]
    story += [pdf_image(OUT / "precision_recall_by_case.png", width=5.2 * inch), Spacer(1, 0.1 * inch), pdf_image(OUT / "dice_delta_vs_a3d.png")]
    story += [
        Spacer(1, 0.08 * inch),
        para("A-3D is best on average, but another approach beats it on 43/74 individual cases. This suggests future ensembling or hard-case routing may be useful.", body),
        pdf_table(
            non_a3d,
            ["case", "gt_voxels", "gt_pct", "A-3D", "A-2.5D", "B", "D", "best_non_A3D", "best_non_A3D_minus_A3D"],
            ["Case", "GT vox", "GT %", "A-3D", "A-2.5D", "B", "D", "Best non-A3D", "Delta"],
            {"gt_pct", "A-3D", "A-2.5D", "B", "D", "best_non_A3D_minus_A3D"},
        ),
    ]

    story += [PageBreak(), Paragraph("Universal Failure Cases", H2)]
    story.append(
        pdf_table(
            failures,
            ["case", "gt_voxels", "gt_pct", "max_dice", "mean_dice"],
            ["Case", "GT vox", "GT %", "Max Dice", "Mean Dice"],
            {"gt_pct", "max_dice", "mean_dice"},
        )
    )
    story += [
        Spacer(1, 0.08 * inch),
        para("IU59 is a large-cyst outlier. It has 33,927 GT voxels, but A-3D and D predict zero voxels. A-2.5D localizes only a small core. Only 1.4% of the GT cyst overlaps the predicted pancreas mask, suggesting atypical anatomical context.", body),
        Spacer(1, 0.1 * inch),
        pdf_image(OUT / "case_compare_IU59.png"),
    ]

    story += [PageBreak(), Paragraph("Top Successful Cases", H2)]
    story.append(
        pdf_table(
            top5,
            ["case", "gt_voxels", "gt_pct", "A-3D", "A-2.5D", "B", "D", "mean_dice", "min_dice"],
            ["Case", "GT vox", "GT %", "A-3D", "A-2.5D", "B", "D", "Mean", "Min"],
            {"gt_pct", "A-3D", "A-2.5D", "B", "D", "mean_dice", "min_dice"},
        )
    )
    story += [Spacer(1, 0.1 * inch), pdf_image(OUT / "case_compare_CAD262.png")]

    story += [PageBreak(), Paragraph("Recommendations and Future Plan", H2)]
    items = [
        "Keep A-3D as the main baseline and deployment candidate.",
        "Run 5-fold cross-validation and report confidence intervals.",
        "Report stratified metrics by cyst-size bin in every summary.",
        "Verify HD95/ASD before emphasizing distance metrics.",
        "Perform failure taxonomy: true miss, wrong-location FP, under-segmentation, over-segmentation.",
        "Investigate IU59-like anatomy and relationship to pancreas masks.",
        "Improve small-cyst handling with volume-aware sampling, hard-case mining, or loss weighting.",
        "Evaluate ensembling because other approaches beat A-3D on some individual cases.",
        "For Approach B, quantify GT-cyst coverage inside pancreas ROI/crop.",
        "Add uncertainty/confidence outputs to flag manual-review cases.",
    ]
    story.append(ListFlowable([ListItem(para(i, body)) for i in items], bulletType="1"))
    story += [Spacer(1, 0.15 * inch), para("Bottom line: A-3D is strongest and simplest. Next work should prioritize robust evaluation and targeted failure-mode improvements rather than adding another architecture first.", body)]

    doc.build(story)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    make_case_comparison("IU59", OUT / "case_compare_IU59.png")
    make_case_comparison("CAD262", OUT / "case_compare_CAD262.png")
    REPORT_MD.write_text(make_markdown())
    make_pdf()
    print(f"Wrote Markdown: {REPORT_MD}")
    print(f"Wrote PDF: {REPORT_PDF}")


if __name__ == "__main__":
    main()
