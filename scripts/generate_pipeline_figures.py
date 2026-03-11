from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parents[1]
OUTDIR = ROOT / "figures"
OUTDIR.mkdir(exist_ok=True)


plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "axes.facecolor": "white",
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "svg.fonttype": "none",
    }
)


COLORS = {
    "ink": "#1D2433",
    "muted": "#5E6B7A",
    "line": "#9AA8B8",
    "panel": "#F6F7FA",
    "input": "#E7F1FB",
    "filter": "#EDF5EC",
    "select": "#F7F0E7",
    "model": "#F8E9EC",
    "eval": "#EEEAF8",
    "accent": "#D97A57",
}


def add_box(ax, x, y, w, h, text, fc, ec=None, fs=11, weight="regular", align="center"):
    box = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.010,rounding_size=0.022",
        linewidth=1.2,
        edgecolor=ec or COLORS["line"],
        facecolor=fc,
    )
    ax.add_patch(box)
    ax.text(
        x + (w / 2 if align == "center" else 0.04 * w),
        y + h / 2,
        text,
        ha=align,
        va="center",
        fontsize=fs,
        color=COLORS["ink"],
        fontweight=weight,
        linespacing=1.18,
        wrap=True,
    )
    return box


def add_arrow(ax, x1, y1, x2, y2, text=None, text_dx=0.0, text_dy=0.022, fs=9):
    arrow = FancyArrowPatch(
        (x1, y1),
        (x2, y2),
        arrowstyle="-|>",
        mutation_scale=13,
        linewidth=1.2,
        color=COLORS["muted"],
        shrinkA=2,
        shrinkB=2,
        connectionstyle="arc3,rad=0.0",
    )
    ax.add_patch(arrow)
    if text:
        ax.text(
            (x1 + x2) / 2 + text_dx,
            (y1 + y2) / 2 + text_dy,
            text,
            ha="center",
            va="bottom",
            fontsize=fs,
            color=COLORS["muted"],
        )
    return arrow


def add_badge(ax, x, y, label, color):
    circ = Circle((x, y), 0.018, facecolor=color, edgecolor="none", alpha=0.95)
    ax.add_patch(circ)
    ax.text(x, y - 0.001, label, ha="center", va="center", fontsize=9, color="white", fontweight="bold")


def setup_ax(fig, letter, title, subtitle=None):
    ax = fig.add_subplot(111)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(0.03, 0.96, letter, fontsize=15, fontweight="bold", color=COLORS["ink"], va="top")
    ax.text(0.09, 0.955, title, fontsize=18, color=COLORS["ink"], va="top")
    if subtitle:
        ax.text(0.09, 0.905, subtitle, fontsize=10.5, color=COLORS["muted"], va="top")
    return ax


def add_caption(ax, text):
    ax.text(0.03, 0.03, text, fontsize=9.5, color=COLORS["muted"], va="bottom")


def draw_overview():
    fig = plt.figure(figsize=(13.5, 5.2))
    ax = setup_ax(
        fig,
        letter="A",
        title="MESA workflow overview",
        subtitle="Each modality is processed independently, then combined by stacked learning.",
    )

    x_input, w_input, h_input = 0.05, 0.14, 0.18
    y_inputs = [0.58, 0.36, 0.14]
    input_labels = [
        "Modality 1\nsample × feature",
        "Modality 2\nsample × feature",
        "Modality n\nsample × feature",
    ]
    for y, label in zip(y_inputs, input_labels):
        add_box(ax, x_input, y, w_input, h_input, label, COLORS["input"], fs=11)

    add_box(
        ax,
        0.25,
        0.26,
        0.22,
        0.42,
        "Per-modality preprocessing\n\n• missing-value filter + imputation\n• variance filter\n• univariate selector",
        COLORS["filter"],
        fs=11,
    )
    add_box(
        ax,
        0.53,
        0.30,
        0.17,
        0.34,
        "Optional redundancy pruning\n\nkeep one representative\nper correlated block",
        COLORS["select"],
        fs=11,
    )
    add_box(
        ax,
        0.74,
        0.30,
        0.15,
        0.34,
        "Boruta selection\n\nfit modality-specific\npredictor",
        COLORS["model"],
        fs=11,
    )
    add_box(
        ax,
        0.70,
        0.12,
        0.17,
        0.10,
        "selected features\n+ fitted model",
        COLORS["panel"],
        fs=10,
    )
    add_box(
        ax,
        0.89,
        0.34,
        0.08,
        0.24,
        "MESA\nstacking",
        COLORS["eval"],
        fs=12,
        weight="bold",
    )
    add_box(ax, 0.90, 0.18, 0.09, 0.08, "classification\nprobabilities", COLORS["panel"], fs=9.4)
    add_box(ax, 0.90, 0.07, 0.09, 0.08, "regression\npredictions", COLORS["panel"], fs=9.4)

    add_badge(ax, 0.258, 0.655, "1", COLORS["accent"])
    add_badge(ax, 0.538, 0.615, "2", COLORS["accent"])
    add_badge(ax, 0.748, 0.615, "3", COLORS["accent"])
    add_badge(ax, 0.898, 0.605, "4", COLORS["accent"])

    add_arrow(ax, 0.19, 0.67, 0.25, 0.56)
    add_arrow(ax, 0.19, 0.45, 0.25, 0.47)
    add_arrow(ax, 0.19, 0.23, 0.25, 0.38)
    add_arrow(ax, 0.47, 0.47, 0.53, 0.47)
    add_arrow(ax, 0.70, 0.47, 0.74, 0.47)
    add_arrow(ax, 0.815, 0.30, 0.785, 0.22)
    add_arrow(ax, 0.89, 0.47, 0.89, 0.47)
    add_arrow(ax, 0.93, 0.34, 0.94, 0.26)
    add_arrow(ax, 0.93, 0.34, 0.94, 0.15)

    add_caption(
        ax,
        "Figure A. Compact overview of the modality-level MESA pipeline. Redundancy pruning is optional and is most useful when neighboring or highly correlated features carry overlapping signal.",
    )
    return fig


def draw_detailed():
    fig = plt.figure(figsize=(13.5, 8.6))
    ax = setup_ax(
        fig,
        letter="B",
        title="Detailed MESA method schematic",
        subtitle="Illustrative layout for documentation and slides; exact behavior follows the implementation in mesa/MESA.py.",
    )

    top_y = 0.73
    mid_y = 0.49
    low_y = 0.23
    bottom_y = 0.07

    add_box(ax, 0.05, top_y, 0.18, 0.12, "Input modality matrix\nsamples × features", COLORS["input"], fs=11)
    add_box(ax, 0.28, top_y, 0.18, 0.12, "Step 1\nmissing-value filtering\n+ imputation", COLORS["filter"], fs=11)
    add_box(ax, 0.51, top_y, 0.15, 0.12, "Step 2\nvariance filter", COLORS["filter"], fs=11)
    add_box(ax, 0.71, top_y, 0.22, 0.12, "Step 3\nunivariate selector\nclassification: Wilcoxon\nregression: f_regression", COLORS["select"], fs=10.3)

    add_box(ax, 0.15, mid_y, 0.24, 0.13, "Optional normalization\nL2 Normalizer()", COLORS["panel"], fs=11)
    add_box(ax, 0.45, mid_y, 0.22, 0.13, "Optional redundancy pruning\nbuild correlated blocks\nkeep one representative", COLORS["select"], fs=11)
    add_box(ax, 0.73, mid_y, 0.20, 0.13, "Step 4\nBoruta selection\nretain top_n features", COLORS["model"], fs=11)

    add_box(ax, 0.06, low_y, 0.22, 0.14, "Score mode\nretain the strongest feature\nby univariate signal", COLORS["panel"], fs=10.5)
    add_box(ax, 0.34, low_y, 0.22, 0.14, "Model mode\nretain the strongest feature\nby CV model performance", COLORS["panel"], fs=10.5)
    add_box(ax, 0.63, low_y, 0.17, 0.14, "Per-modality predictor\nclassification:\nRandomForestClassifier\nregression:\nRandomForestRegressor", COLORS["model"], fs=9.3)
    add_box(ax, 0.84, low_y, 0.12, 0.14, "Output\nselected features\n+ fitted model", COLORS["panel"], fs=10.2)

    add_box(ax, 0.54, bottom_y, 0.25, 0.09, "MESA multimodal stacking\ncombine modality-level outputs\nwith a task-aware meta-estimator", COLORS["eval"], fs=10.2, weight="bold")
    add_box(ax, 0.80, bottom_y, 0.17, 0.09, "MESA_CV\nclassification: ROC AUC\nregression: R² or other metrics", COLORS["eval"], fs=9.2)

    add_badge(ax, 0.285, top_y + 0.11, "1", COLORS["accent"])
    add_badge(ax, 0.515, top_y + 0.11, "2", COLORS["accent"])
    add_badge(ax, 0.715, top_y + 0.11, "3", COLORS["accent"])
    add_badge(ax, 0.735, mid_y + 0.12, "4", COLORS["accent"])

    add_arrow(ax, 0.23, top_y + 0.06, 0.28, top_y + 0.06)
    add_arrow(ax, 0.46, top_y + 0.06, 0.51, top_y + 0.06)
    add_arrow(ax, 0.66, top_y + 0.06, 0.71, top_y + 0.06)
    add_arrow(ax, 0.82, top_y, 0.82, mid_y + 0.13)
    add_arrow(ax, 0.60, top_y + 0.01, 0.27, mid_y + 0.13, text="optional", text_dy=0.015)
    add_arrow(ax, 0.39, mid_y + 0.065, 0.45, mid_y + 0.065)
    add_arrow(ax, 0.67, mid_y + 0.065, 0.73, mid_y + 0.065)
    add_arrow(ax, 0.56, mid_y, 0.17, low_y + 0.14)
    add_arrow(ax, 0.56, mid_y, 0.45, low_y + 0.14)
    add_arrow(ax, 0.83, mid_y, 0.72, low_y + 0.14)
    add_arrow(ax, 0.80, low_y + 0.07, 0.84, low_y + 0.07)
    add_arrow(ax, 0.72, low_y, 0.665, bottom_y + 0.09)
    add_arrow(ax, 0.90, low_y, 0.885, bottom_y + 0.09)

    ax.text(0.065, 0.64, "Clean and rank", fontsize=10, color=COLORS["muted"])
    ax.text(0.47, 0.40, "Prune redundancy", fontsize=10, color=COLORS["muted"])
    ax.text(0.745, 0.40, "Select + fit", fontsize=10, color=COLORS["muted"])
    ax.text(0.60, 0.005, "Figure B. Expanded schematic of preprocessing, redundancy pruning, Boruta selection, prediction, multimodal stacking, and cross-validation.", fontsize=9.5, color=COLORS["muted"], ha="center", va="bottom")
    return fig


def draw_illustration():
    fig = plt.figure(figsize=(13.5, 7.8))
    ax = fig.add_subplot(111)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # soft background panels
    add_box(ax, 0.04, 0.10, 0.24, 0.74, "", "#F4F8FC", ec="#DCE6F0")
    add_box(ax, 0.31, 0.18, 0.38, 0.56, "", "#FAF8F5", ec="#E3D8CC")
    add_box(ax, 0.72, 0.10, 0.24, 0.74, "", "#F8F5FB", ec="#DFD7EC")

    ax.text(0.05, 0.93, "MESA multimodal biomarker modeling", fontsize=23, fontweight="bold", color=COLORS["ink"], va="top")
    ax.text(
        0.05,
        0.885,
        "An illustration of how independent cfDNA feature layers are distilled into a single predictive model.",
        fontsize=12.5,
        color=COLORS["muted"],
        va="top",
    )

    # Left: modality cards
    add_box(ax, 0.08, 0.62, 0.16, 0.15, "Methylation\nCpG matrix", COLORS["input"], fs=13, weight="bold")
    add_box(ax, 0.08, 0.41, 0.16, 0.15, "Fragmentation\nfeature matrix", "#EAF6F0", fs=13, weight="bold")
    add_box(ax, 0.08, 0.20, 0.16, 0.15, "Other omics\nfeature matrix", "#F8F1E9", fs=13, weight="bold")
    ax.text(0.16, 0.80, "Independent modalities", ha="center", fontsize=12, color=COLORS["muted"])

    # tiny dot grids to look less like a flowchart
    for base_x, base_y, color in [(0.10, 0.655, "#6B93C0"), (0.10, 0.445, "#69A888"), (0.10, 0.235, "#C78F5A")]:
        for i in range(5):
            for j in range(4):
                circ = Circle((base_x + 0.018 * i, base_y + 0.02 * j), 0.0048, facecolor=color, edgecolor="none", alpha=0.85)
                ax.add_patch(circ)

    # Center: MESA engine
    mesa = FancyBboxPatch(
        (0.39, 0.30),
        0.22,
        0.30,
        boxstyle="round,pad=0.02,rounding_size=0.03",
        linewidth=2.0,
        edgecolor="#D39D77",
        facecolor="#FFF5EA",
    )
    ax.add_patch(mesa)
    ax.text(0.50, 0.55, "MESA", ha="center", va="center", fontsize=24, fontweight="bold", color="#8A5A32")
    ax.text(0.50, 0.49, "feature filtering", ha="center", fontsize=12.5, color=COLORS["ink"])
    ax.text(0.50, 0.44, "redundancy pruning", ha="center", fontsize=12.5, color=COLORS["ink"])
    ax.text(0.50, 0.39, "Boruta selection", ha="center", fontsize=12.5, color=COLORS["ink"])
    ax.text(0.50, 0.34, "task-aware prediction", ha="center", fontsize=12.5, color=COLORS["ink"])

    add_badge(ax, 0.41, 0.59, "1", COLORS["accent"])
    add_badge(ax, 0.41, 0.49, "2", COLORS["accent"])
    add_badge(ax, 0.41, 0.39, "3", COLORS["accent"])

    # Right: outputs
    add_box(ax, 0.76, 0.58, 0.16, 0.16, "Classification\nprobabilities", "#EFE9FA", fs=13, weight="bold")
    add_box(ax, 0.76, 0.36, 0.16, 0.16, "Regression\npredictions", "#F9ECF0", fs=13, weight="bold")
    add_box(ax, 0.76, 0.14, 0.16, 0.16, "Cross-validation\nAUC / R² / RMSE", "#EEF2FB", fs=13, weight="bold")
    ax.text(0.84, 0.80, "Task-aware outputs", ha="center", fontsize=12, color=COLORS["muted"])

    # Feature chips under MESA
    chip_y = 0.20
    chip_specs = [
        (0.35, "selected\nCpGs", "#EBD6BF"),
        (0.47, "modality\nmodels", "#E6DDF7"),
        (0.59, "stacked\nsignal", "#DCECF6"),
    ]
    for x, label, fc in chip_specs:
        add_box(ax, x, chip_y, 0.10, 0.08, label, fc, fs=10.5, weight="bold")

    # Curved connectors, softer than flowchart arrows
    for y1, y2, rad in [(0.695, 0.54, -0.12), (0.485, 0.46, 0.0), (0.275, 0.38, 0.12)]:
        arc = FancyArrowPatch(
            (0.24, y1),
            (0.39, y2),
            arrowstyle="-",
            linewidth=3,
            color="#B7C4D4",
            connectionstyle=f"arc3,rad={rad}",
            alpha=0.9,
        )
        ax.add_patch(arc)

    for y1, y2, rad, color in [(0.54, 0.66, 0.10, "#C8B7E8"), (0.45, 0.44, 0.0, "#E1B8C3"), (0.36, 0.22, -0.12, "#B9CAE8")]:
        arc = FancyArrowPatch(
            (0.61, y1),
            (0.76, y2),
            arrowstyle="-",
            linewidth=3,
            color=color,
            connectionstyle=f"arc3,rad={rad}",
            alpha=0.95,
        )
        ax.add_patch(arc)

    # subtle arrows on top of arcs
    for x1, y1, x2, y2 in [(0.35, 0.61, 0.39, 0.56), (0.35, 0.49, 0.39, 0.47), (0.35, 0.28, 0.39, 0.38), (0.61, 0.54, 0.76, 0.64), (0.61, 0.45, 0.76, 0.44), (0.61, 0.36, 0.76, 0.22)]:
        add_arrow(ax, x1, y1, x2, y2)

    # legend-like callouts
    add_box(ax, 0.32, 0.73, 0.12, 0.08, "sample-level\nanalysis", "#FFFFFF", fs=10.5)
    add_box(ax, 0.47, 0.73, 0.14, 0.08, "interpretable\nfeature selection", "#FFFFFF", fs=10.5)
    add_box(ax, 0.64, 0.73, 0.12, 0.08, "multimodal\nintegration", "#FFFFFF", fs=10.5)

    ax.text(
        0.50,
        0.06,
        "Figure C. Illustration-style summary of MESA. Multiple cfDNA-derived feature layers are refined into a compact, task-aware biomarker model.",
        ha="center",
        va="center",
        fontsize=10.5,
        color=COLORS["muted"],
    )
    return fig


def save(fig, stem):
    fig.savefig(OUTDIR / f"{stem}.svg", bbox_inches="tight")
    fig.savefig(OUTDIR / f"{stem}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    save(draw_overview(), "mesa_pipeline_overview")
    save(draw_detailed(), "mesa_pipeline_detailed")
    save(draw_illustration(), "mesa_pipeline_illustration")
    print(f"Wrote figures to {OUTDIR}")


if __name__ == "__main__":
    main()
